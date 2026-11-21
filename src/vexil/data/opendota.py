"""Client for the OpenDota explorer (a public SQL endpoint) and the weekly pull.

Free-tier manners: one call about every 1.1 seconds, back-off on HTTP 429 and 5xx.
A week costs one call for the match ids, one for patches, and two per 200 matches.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

from vexil.weeks import week_bounds

BASE_URL = "https://api.opendota.com/api"
CHUNK = 200
INTERVAL_SECONDS = 1.1

# Columns kept from `matches`. End-of-match totals are left out; `duration` is a
# post-match value that defines a usable match and is never a model input.
MATCH_COLUMNS = (
    "match_id", "start_time", "duration", "radiant_win", "leagueid", "series_id", "series_type",
    "radiant_team_id", "dire_team_id", "radiant_captain", "dire_captain", "game_mode",
    "lobby_type", "cluster", "version", "engine", "human_players", "first_blood_time",
    "radiant_gold_adv", "radiant_xp_adv", "objectives", "picks_bans", "draft_timings",
    "teamfights",
)  # fmt: skip
# Columns kept from `player_matches`: identity plus everything that carries a time stamp.
PLAYER_COLUMNS = (
    "match_id", "player_slot", "account_id", "hero_id", "hero_variant", "party_id",
    "times", "gold_t", "xp_t", "lh_t", "dn_t",
    "kills_log", "purchase_log", "buyback_log", "runes_log",
    "obs_log", "sen_log", "obs_left_log", "sen_left_log",
    "connection_log", "neutral_tokens_log", "neutral_item_history",
)  # fmt: skip

Explorer = Callable[[str], list[dict]]


class OpenDotaClient:
    """Calls the explorer with pacing and retries. One instance per run."""

    def __init__(self, interval: float = INTERVAL_SECONDS, tries: int = 6) -> None:
        self.interval, self.tries, self.calls, self._last = interval, tries, 0, 0.0

    def explorer(self, sql: str) -> list[dict]:
        url = f"{BASE_URL}/explorer?sql={urllib.parse.quote(sql)}"
        for attempt in range(self.tries):
            wait = self._last + self.interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            self.calls += 1
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "vexil-mlops (weekly pull)",
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                payload = json.loads(body)
                if payload.get("err"):
                    raise RuntimeError(f"explorer error: {payload['err']}")
                return payload["rows"]
            except urllib.error.HTTPError as error:
                detail = error.read()[:300]
                retryable = error.code in (429, 500, 502, 503, 504) or (
                    error.code == 400 and b"timeout" in detail
                )
                if not retryable:
                    raise RuntimeError(f"HTTP {error.code} from OpenDota: {detail!r}") from error
                problem = f"HTTP {error.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, EOFError) as error:
                problem = repr(error)[:120]
            pause = min(120, 5 * 2**attempt)
            print(f"  retry {attempt + 1} after {problem}, sleeping {pause} s", flush=True)
            time.sleep(pause)
        raise RuntimeError("giving up on the OpenDota explorer")


def pull_week(explorer: Explorer, week: str) -> tuple[list[dict], list[dict]]:
    """All matches that start in a week, with their player rows.

    Each match row also gets ``patch``, ``league_tier`` and ``league_name`` as they are at
    pull time, so a weekly file can be read without any other table.
    """
    start, end = week_bounds(week)
    window = f"m.start_time >= {start} AND m.start_time < {end}"
    ids = [
        row["match_id"]
        for row in explorer(f"SELECT m.match_id FROM matches m WHERE {window} ORDER BY m.match_id")
    ]
    if not ids:
        return [], []
    patches = {
        row["match_id"]: row["patch"]
        for row in explorer(
            "SELECT mp.match_id, mp.patch FROM match_patch mp JOIN matches m USING (match_id) "
            f"WHERE {window} ORDER BY mp.match_id"
        )
    }
    leagues = {
        row["leagueid"]: row
        for row in explorer(
            "SELECT DISTINCT l.leagueid, l.name, l.tier FROM leagues l "
            f"JOIN matches m USING (leagueid) WHERE {window}"
        )
    }
    matches: list[dict] = []
    players: list[dict] = []
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i : i + CHUNK]
        id_list = ",".join(str(match_id) for match_id in chunk)
        rows = explorer(
            f"SELECT {','.join(MATCH_COLUMNS)} FROM matches "
            f"WHERE match_id IN ({id_list}) ORDER BY match_id"
        )
        if [row["match_id"] for row in rows] != chunk:
            raise RuntimeError("match rows do not equal the id list")
        matches += rows
        players += explorer(
            f"SELECT {','.join(PLAYER_COLUMNS)} FROM player_matches "
            f"WHERE match_id IN ({id_list}) ORDER BY match_id, player_slot"
        )
    for match in matches:
        league = leagues.get(match["leagueid"], {})
        match["patch"] = patches.get(match["match_id"])
        match["league_tier"] = league.get("tier")
        match["league_name"] = league.get("name")
    return matches, players
