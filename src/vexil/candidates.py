"""Run the audited ``train.py`` files of the vexil-auto loop inside the package.

The four candidates (A1, A3, A4, A5) are the final ``train.py`` of agents of the auto-research
loop. They are not rewritten: the same file, checked against its SHA-256, is loaded and called
as ``fit(ctx, train_match_ids, seed)`` and ``predict(state, ctx, rows)``. ``CandidateCtx`` gives
such a file what the loop's evaluator gave it, built from the vexil store instead of the
evaluator bundle:

* ``rows``: one row per (match, minute) with the in-game columns;
* ``pregame(match_id)``: start, team ids, account ids and hero ids per side (sorted by player slot);
* ``results_before(ts)``: results of finished matches of any tier that ended strictly before
  ``ts`` (never later than ``ctx.now``, the latest start in the call);
* ``label(match_id)``: allowed for training matches in ``fit`` only;
* ``raw``: not offered (no candidate reads it).

Nothing here reads a result except through the ``results`` dictionary the caller passes in,
which in production comes from ``vexil.labels`` (logged, and withheld until the lock exists).
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vexil.features.a1 import A1_INPUTS, a1_table
from vexil.features.extra import EXTRA_INPUTS, extra_table
from vexil.matchdata import Match, Player, n_rows

COLUMNS = (*A1_INPUTS, *EXTRA_INPUTS)


def match_meta(match: Match, players: list[Player]) -> dict:
    """Facts known before a match starts, plus its end time (start + duration)."""
    ordered = sorted(players, key=lambda p: p["player_slot"])
    rad = [p for p in ordered if p["player_slot"] < 128]
    dire = [p for p in ordered if p["player_slot"] >= 128]
    return {
        "id": match["match_id"],
        "start": match["start_time"],
        "end": match["start_time"] + (match.get("duration") or 0),
        "rt": match.get("radiant_team_id"),
        "dt": match.get("dire_team_id"),
        "ra": [p.get("account_id") for p in rad],
        "da": [p.get("account_id") for p in dire],
        "rh": [p.get("hero_id") for p in rad],
        "dh": [p.get("hero_id") for p in dire],
        "leagueid": int(match.get("leagueid") or 0),
        "tier": str(match.get("league_tier")),
        "game_mode": int(match.get("game_mode") or 0),
    }


class CandidateTable:
    """Rows and match facts of a set of matches, in the layout the candidates read."""

    def __init__(self) -> None:
        self.rows: dict[str, np.ndarray] = {}
        self.meta: dict[int, dict] = {}
        self.results: dict[int, int] = {}
        self.start: dict[int, int] = {}
        self.end_sorted = np.array([], dtype=np.int64)
        self.end_ids: list[int] = []

    @classmethod
    def build(
        cls,
        matches: Iterable[tuple[Match, list[Player]]],
        results: dict[int, int],
        items: dict,
        order: list[int] | None = None,
    ) -> CandidateTable:
        """Build from (match, players) pairs. `results` maps match id to 1 if Radiant won.

        Every match with a known result enters `meta` (the candidates replay results of any
        tier). Only usable matches have rows. `order` fixes the order of the matches in `rows`.
        """
        table = cls()
        per_match: dict[int, dict[str, np.ndarray]] = {}
        for match, players in matches:
            mid = match["match_id"]
            table.meta[mid] = match_meta(match, players)
            table.start[mid] = match["start_time"]
            rows = n_rows(match)
            if rows:
                cols = {k: v for k, v in a1_table(match, players).items() if k in A1_INPUTS}
                cols.update(extra_table(match, players, items))
                cols["t"] = np.arange(rows)
                per_match[mid] = cols
        table.results = {m: r for m, r in results.items() if m in table.meta}
        ids = [m for m in (order or sorted(per_match)) if m in per_match]
        if ids:
            table.rows = {
                "match_id": np.concatenate([np.full(len(per_match[m]["t"]), m) for m in ids]),
                "t": np.concatenate([per_match[m]["t"] for m in ids]),
                **{k: np.concatenate([per_match[m][k] for m in ids]) for k in COLUMNS},
            }
        ends = sorted((m["end"], m["id"]) for m in table.meta.values())
        table.end_sorted = np.array([e for e, _ in ends], dtype=np.int64)
        table.end_ids = [i for _, i in ends]
        return table


class CandidateCtx:
    """What a candidate ``train.py`` may see in one call (same contract as the loop's evaluator)."""

    def __init__(
        self, table: CandidateTable, row_idx: np.ndarray, label_ids: Iterable[int] | None = None
    ):
        self._table = table
        self._row_idx = np.asarray(row_idx)
        self._labels_ok = None if label_ids is None else {int(i) for i in label_ids}
        mids = table.rows["match_id"][self._row_idx]
        self.now = int(max(table.start[int(m)] for m in set(mids.tolist()))) if len(mids) else 0

    @property
    def rows(self) -> pd.DataFrame:
        d = self._table.rows
        cols = {k: d[k][self._row_idx] for k in d if k not in ("match_id", "t")}
        cols = {k: (v.astype(np.float64) if v.dtype == np.float32 else v) for k, v in cols.items()}
        df = pd.DataFrame(
            {
                "match_id": d["match_id"][self._row_idx],
                "t": d["t"][self._row_idx].astype(int),
                **cols,
            }
        )
        df["start_time"] = [self._table.start[int(m)] for m in df.match_id]
        return df

    def raw(self, match_id: int, t: int):
        raise PermissionError("raw is not offered to candidates in the package")

    def pregame(self, match_id: int) -> dict:
        m = self._table.meta[int(match_id)]
        keys = ("start", "rt", "dt", "ra", "da", "rh", "dh", "id", "leagueid", "tier", "game_mode")
        return {k: m[k] for k in keys}

    def results_before(self, ts: int) -> list[dict]:
        if ts > self.now:
            raise PermissionError(f"results_before({ts}): later than the latest start ({self.now})")
        t = self._table
        k = int(np.searchsorted(t.end_sorted, ts, side="left"))
        out = []
        for mid in t.end_ids[:k]:
            if mid not in t.results:
                continue
            m = t.meta[mid]
            out.append(
                {
                    "id": mid,
                    "start": m["start"],
                    "end": m["end"],
                    "rt": m["rt"],
                    "dt": m["dt"],
                    "ra": m["ra"],
                    "da": m["da"],
                    "rh": m["rh"],
                    "dh": m["dh"],
                    "radiant_win": int(t.results[mid]),
                }
            )
        return out

    def label(self, match_id: int) -> int:
        if self._labels_ok is None or int(match_id) not in self._labels_ok:
            raise PermissionError(f"label({match_id}): not a training match of this call")
        return int(self._table.results[int(match_id)])


def load_train(path: Path, expected_sha256: str):
    """Load a candidate ``train.py`` after checking its hash. The file is never edited."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"{path}: sha256 {digest[:12]} differs from the registered {expected_sha256[:12]}"
        )
    spec = importlib.util.spec_from_file_location(f"candidate_{digest[:8]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------- state files
def encode_state(obj):
    """Fitted state of a candidate as JSON-safe data. Floats survive exactly (repr round trip)."""
    if isinstance(obj, np.ndarray):
        return {"__nd__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):
        return encode_state(obj.item())
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            return {"__dict__": {k: encode_state(v) for k, v in obj.items()}}
        if all(isinstance(k, int | np.integer) for k in obj):
            return {"__intdict__": [[int(k), encode_state(v)] for k, v in obj.items()]}
        raise TypeError("state dictionaries need keys that are all str or all int")
    if isinstance(obj, list | tuple):
        return {"__list__": [encode_state(v) for v in obj]}
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    raise TypeError(f"cannot store a state value of type {type(obj).__name__}")


def decode_state(obj):
    if isinstance(obj, dict):
        if "__nd__" in obj:
            return np.array(obj["__nd__"], dtype=obj["dtype"])
        if "__dict__" in obj:
            return {k: decode_state(v) for k, v in obj["__dict__"].items()}
        if "__intdict__" in obj:
            return {int(k): decode_state(v) for k, v in obj["__intdict__"]}
        if "__list__" in obj:
            return [decode_state(v) for v in obj["__list__"]]
        raise ValueError("unknown state encoding")
    return obj


# ---------------------------------------------------------------------------- history
@dataclass
class ResultHistory:
    """Facts and results of every finished match of any tier, as the candidates replay them.

    The weekly job keeps one snapshot before each week, like the rating snapshots. A candidate
    sees only results of matches that ended strictly before the match it predicts.
    """

    meta: dict[int, dict]
    results: dict[int, int]

    @classmethod
    def empty(cls) -> ResultHistory:
        return cls({}, {})

    def add_week(self, matches: Iterable[tuple[Match, list[Player]]], results: dict[int, int]):
        """Add the facts of a week's matches and the results the loader released for them."""
        for match, players in matches:
            self.meta[match["match_id"]] = match_meta(match, players)
        for mid, win in results.items():
            if mid in self.meta:
                self.results[mid] = int(win)

    def save(self, path: Path) -> None:
        payload = {
            "meta": [self.meta[k] for k in sorted(self.meta)],
            "results": {str(k): v for k, v in sorted(self.results.items())},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(path, "wb", mtime=0) as handle:
            handle.write(json.dumps(payload, separators=(",", ":")).encode())

    @classmethod
    def load(cls, path: Path) -> ResultHistory:
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        return cls(
            {m["id"]: m for m in payload["meta"]},
            {int(k): v for k, v in payload["results"].items()},
        )


# ---------------------------------------------------------------------------- the model kind
@dataclass
class FrozenTrainModel:
    """A candidate: an audited ``train.py`` plus the state it was fitted to."""

    module: object
    state: object
    train_file: str
    train_sha256: str

    def predict_rows(self, table: CandidateTable, row_idx: np.ndarray) -> np.ndarray:
        ctx = CandidateCtx(table, row_idx)
        p = np.asarray(self.module.predict(self.state, ctx, ctx.rows), dtype=float)
        if (
            p.shape != (len(row_idx),)
            or not np.isfinite(p).all()
            or (p <= 0).any()
            or (p >= 1).any()
        ):
            raise ValueError("a candidate returned a bad prediction array")
        return p


def load_candidate(path: Path) -> FrozenTrainModel:
    """Read a candidate file (kind ``frozen_train``): check the train.py hash, restore the state."""
    path = Path(path)
    spec = json.loads(path.read_text())
    if spec.get("kind") != "frozen_train":
        raise ValueError(f"{path}: not a frozen_train file")
    train = path.parent / spec["train_file"]
    module = load_train(train, spec["train_sha256"])
    return FrozenTrainModel(
        module, decode_state(spec["state"]), spec["train_file"], spec["train_sha256"]
    )


def fit_candidate(
    train_path: Path, train_sha256: str, table: CandidateTable, train_ids: list[int], seed: int = 0
) -> dict:
    """Fit a candidate on the given training matches and return the content of its model file."""
    module = load_train(train_path, train_sha256)
    rows = np.flatnonzero(np.isin(table.rows["match_id"], list(train_ids)))
    ids = sorted({int(m) for m in table.rows["match_id"][rows]})
    state = module.fit(CandidateCtx(table, rows, label_ids=ids), ids, seed)
    return {
        "kind": "frozen_train",
        "train_file": Path(train_path).name,
        "train_sha256": train_sha256,
        "seed": seed,
        "n_rows": int(len(rows)),
        "n_matches": len(ids),
        "state": encode_state(state),
    }


def week_table(
    history: ResultHistory,
    matches: list[tuple[Match, list[Player]]],
    predict_ids: set[int],
    results: dict[int, int],
    items: dict,
) -> CandidateTable:
    """Table for one week of scoring.

    ``matches`` are all matches of the week (any tier); only those in ``predict_ids`` get rows.
    The replay facts are the history before the week plus the week's matches, and the results
    are the history's plus the ones the loader released for the week. A candidate asking for
    ``results_before(ts)`` still gets only matches that ended strictly before ``ts``.
    """
    table = CandidateTable.build(
        [(m, p) for m, p in matches if m["match_id"] in predict_ids],
        {},
        items,
        order=sorted(predict_ids),
    )
    for match, players in matches:
        table.meta.setdefault(match["match_id"], match_meta(match, players))
    table.meta = dict(history.meta) | table.meta
    table.results = {m: r for m, r in (history.results | results).items() if m in table.meta}
    table.start = {m: v["start"] for m, v in table.meta.items()}
    ends = sorted((v["end"], m) for m, v in table.meta.items())
    table.end_sorted = np.array([e for e, _ in ends], dtype=np.int64)
    table.end_ids = [i for _, i in ends]
    return table
