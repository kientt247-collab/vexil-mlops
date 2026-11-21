"""Player ratings (Elo), the one stateful input of the final model.

Every account starts at 1500. When a match ends, each of its ten players moves by
``K * (result - expected)``, where ``expected`` comes from the Elo formula on the two team
means. K is 64 for a player's first 20 rated matches and 32 afterwards. The model input of
a match is ``(mean Radiant rating - mean Dire rating) * ln(10) / 400``, read when the match
starts. A missing account id reads 1500, and a match without ten account ids is not rated.

Time rule: a result is applied only once its match has ended (start + duration), and a
match reads the ratings as they stand at its own start. A match therefore never sees its
own result, nor the result of a match that was still running when it started.

``RatingState`` carries everything needed to continue later: the ratings, the number of
rated matches per account, and the results that are known but not applied yet. Saving it
after a week and loading it before the next gives the same numbers as one pass over all
weeks, which is what lets the weekly job avoid replaying the whole year.
"""

from __future__ import annotations

import gzip
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

START_RATING = 1500.0
LOG_ODDS_PER_POINT = math.log(10) / 400
K_ESTABLISHED, K_NEW, NEW_MATCHES = 32, 64, 20
INPUT_NAME = "pelx"


@dataclass(frozen=True)
class MatchMeta:
    """The facts the rating needs about a match. None of them is the result."""

    match_id: int
    start: int
    end: int
    radiant: tuple[int | None, ...]
    dire: tuple[int | None, ...]

    @classmethod
    def from_rows(cls, match: dict, players: list[dict]) -> MatchMeta:
        ordered = sorted(players, key=lambda p: p["player_slot"])
        return cls(
            match_id=match["match_id"],
            start=match["start_time"],
            end=match["start_time"] + (match.get("duration") or 0),
            radiant=tuple(p.get("account_id") for p in ordered if p["player_slot"] < 128),
            dire=tuple(p.get("account_id") for p in ordered if p["player_slot"] >= 128),
        )

    @property
    def rateable(self) -> bool:
        sides = (self.radiant, self.dire)
        return all(len(side) == 5 and all(side) for side in sides)


@dataclass
class RatingState:
    ratings: dict[int, float] = field(default_factory=dict)
    games: dict[int, int] = field(default_factory=dict)
    # results known but not applied yet: (end time, match id, radiant ids, dire ids, result)
    pending: list[tuple] = field(default_factory=list)

    # ------------------------------------------------------------------ core
    def _side_mean(self, accounts: tuple[int | None, ...]) -> float:
        if not accounts:
            return START_RATING
        values = [self.ratings.get(a, START_RATING) if a else START_RATING for a in accounts]
        return float(np.mean(values))

    def _apply(self, radiant: tuple[int, ...], dire: tuple[int, ...], result: int) -> None:
        radiant_mean = sum(self.ratings.get(a, START_RATING) for a in radiant) / len(radiant)
        dire_mean = sum(self.ratings.get(a, START_RATING) for a in dire) / len(dire)
        surprise = result - 1.0 / (1.0 + 10 ** ((dire_mean - radiant_mean) / 400))
        for sign, side in ((1.0, radiant), (-1.0, dire)):
            for account in side:
                k = K_NEW if self.games.get(account, 0) < NEW_MATCHES else K_ESTABLISHED
                self.ratings[account] = (
                    self.ratings.get(account, START_RATING) + sign * k * surprise
                )
                self.games[account] = self.games.get(account, 0) + 1

    def advance_to(self, time: int) -> None:
        """Apply every known result of a match that ended before `time`."""
        while self.pending and self.pending[0][0] < time:
            _, _, radiant, dire, result = heapq.heappop(self.pending)
            self._apply(tuple(radiant), tuple(dire), result)

    def read(self, meta: MatchMeta) -> float:
        """Model input of a match, as known at its start."""
        self.advance_to(meta.start)
        return (self._side_mean(meta.radiant) - self._side_mean(meta.dire)) * LOG_ODDS_PER_POINT

    def add_result(self, meta: MatchMeta, result: int) -> None:
        """Register a result. It takes effect for matches that start after this one ended."""
        if meta.rateable:
            heapq.heappush(
                self.pending, (meta.end, meta.match_id, list(meta.radiant), list(meta.dire), result)
            )

    def process(self, metas: list[MatchMeta], results: dict[int, int]) -> dict[int, float]:
        """Read the input of each match in start order, registering results as they come."""
        inputs = {}
        for meta in sorted(metas, key=lambda m: (m.start, m.match_id)):
            inputs[meta.match_id] = self.read(meta)
            if meta.match_id in results:
                self.add_result(meta, results[meta.match_id])
        return inputs

    def new_player_share(self, meta: MatchMeta) -> float:
        """Share of the ten player slots held by accounts with fewer than 20 rated matches."""
        slots = [*meta.radiant, *meta.dire]
        if not slots:
            return 1.0
        return sum((self.games.get(a, 0) if a else 0) < NEW_MATCHES for a in slots) / len(slots)

    # ------------------------------------------------------------------ snapshots
    def save(self, path: Path) -> None:
        payload = {
            "ratings": {str(k): v for k, v in sorted(self.ratings.items())},
            "games": {str(k): v for k, v in sorted(self.games.items())},
            "pending": sorted(self.pending),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(path, "wb", mtime=0) as handle:
            handle.write(json.dumps(payload, separators=(",", ":")).encode())

    @classmethod
    def load(cls, path: Path) -> RatingState:
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        state = cls(
            ratings={int(k): v for k, v in payload["ratings"].items()},
            games={int(k): v for k, v in payload["games"].items()},
            pending=[tuple(item) for item in payload["pending"]],
        )
        heapq.heapify(state.pending)
        return state


def reference_input(metas: list[MatchMeta], results: dict[int, int], target: MatchMeta) -> float:
    """From-scratch reference for the leak test.

    Starts from nothing and uses only matches that ended before `target` started.
    """
    state = RatingState()
    earlier = [m for m in metas if m.match_id in results and m.end < target.start and m.rateable]
    for meta in sorted(earlier, key=lambda m: (m.end, m.match_id)):
        state._apply(meta.radiant, meta.dire, results[meta.match_id])
    return (state._side_mean(target.radiant) - state._side_mean(target.dire)) * LOG_ODDS_PER_POINT
