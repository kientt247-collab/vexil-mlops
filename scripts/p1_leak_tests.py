"""P1 (leak tests on real matches) for the four fitted candidates.

For a fixed sample of real development matches M and the final model files:

* cut test: the predictions for minutes <= t from M cut at minute t equal those from the full M;
* poison test: the same with everything after minute t replaced by random values and events;
* later-result test: flip the result of every other match that ended at or after the start of M;
  the predictions for M must not move (nothing that ends after M starts reaches M);
* reach test: every result handed to a candidate for M ended strictly before M started;
* planted bugs (the later-result test must catch the one that breaks both layers): results are
  accepted by start time instead of end time in train.py only (A3 copy), in ctx only, or in both.
  Two layers guard the rule (ctx.results_before and the Elo reader of train.py), so a bug in
  one layer alone is stopped by the other and must not move any prediction.

Everything reads development matches only. Labels come through vexil.labels, purpose "P1-leak".

Usage: uv run python scripts/p1_leak_tests.py STORE_ROOT [N_SAMPLE]
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from vexil.candidates import (
    CandidateCtx,
    CandidateTable,
    ResultHistory,
    fit_candidate,
    load_candidate,
    match_meta,
    week_table,
)
from vexil.data.store import Store
from vexil.features.extra import load_items
from vexil.labels import LabelLoader
from vexil.leakcheck import poison
from vexil.matchdata import cut, is_usable
from vexil.scoring import JUDGED_TIERS

DEV_END = 1785456000
FOLDER = Path("models/candidates")
TAGS = ("A1", "A3", "A4", "A5")
CUTS = (5, 12, 20)
TOL = 1e-12


class LeakyCtx(CandidateCtx):
    """Planted bug: results are handed over by start time, so a running match's result leaks."""

    def results_before(self, ts):
        t = self._table
        out = []
        for mid in sorted(t.meta, key=lambda m: (t.meta[m]["end"], m)):
            if mid in t.results and t.meta[mid]["start"] < ts:
                m = t.meta[mid]
                out.append(
                    {
                        "id": mid, "start": m["start"], "end": m["end"], "rt": m["rt"],
                        "dt": m["dt"], "ra": m["ra"], "da": m["da"], "rh": m["rh"],
                        "dh": m["dh"], "radiant_win": int(t.results[mid]),
                    }
                )  # fmt: skip
        return out


def main(store_root: str, n_sample: int = 40) -> None:
    store = Store(Path(store_root), Path("data/manifest.json"))
    loader = LabelLoader(store, Path("/tmp/p1_label_reads.log"), Path("lock.json"))
    weeks = [w for w in store.weeks() if w <= "2026-07-27"]
    results: dict[int, int] = {}
    for w in weeks:
        results.update(loader.week_labels(w, "P1-leak"))
    # one pass over the store; only match facts are kept in memory, not whole matches
    history = ResultHistory.empty()
    usable, long_enough = [], set()
    for w in weeks:
        for match, players in store.read_week(w):
            if match["start_time"] >= DEV_END:
                continue
            mid = match["match_id"]
            history.meta[mid] = match_meta(match, players)
            if is_usable(match) and match.get("league_tier") in JUDGED_TIERS and mid in results:
                usable.append(mid)
                if len(match["radiant_gold_adv"]) > 25:
                    long_enough.add(mid)
    history.results = {m: r for m, r in results.items() if m in history.meta}
    items = load_items()
    usable = sorted(usable)

    def overlaps(m):
        s = history.meta[m]["start"]
        return any(v["start"] < s < v["end"] and k != m for k, v in history.meta.items())

    rng = random.Random(0)
    pool = sorted(long_enough)
    rng.shuffle(pool)
    sample = []
    n_overlap = 0
    for m in pool:
        if len(sample) >= n_sample:
            break
        ov = overlaps(m)
        if ov or len(sample) - n_overlap < n_sample // 2:
            sample.append(m)
            n_overlap += ov
    print(f"sample: {len(sample)} matches, {n_overlap} overlap another match", flush=True)
    wanted = set(sample)
    pairs = {
        match["match_id"]: (match, players)
        for w in weeks
        for match, players in store.read_week(w)
        if match["match_id"] in wanted
    }

    models = {t: load_candidate(FOLDER / t / "model.json") for t in TAGS}
    buggy_candidate = _planted_bug_model(store, weeks, history, results, items, usable)
    variants = {
        # the leak is in train.py only: ctx hands over nothing that ends after the start
        "A3_BUG_IN_TRAIN_ONLY": buggy_candidate,
        # the leak is in ctx only: train.py still reads by end time
        "A3_BUG_IN_CTX_ONLY": replace(models["A3"], ctx_class=LeakyCtx),
        # both layers leak
        "A3_BUG_IN_BOTH": replace(buggy_candidate, ctx_class=LeakyCtx),
    }

    def predict(model, m, match_players, hist, res):
        h = ResultHistory(
            {k: v for k, v in hist.meta.items() if k != m},
            {k: v for k, v in hist.results.items() if k != m},
        )
        table = week_table(h, [match_players], {m}, res, items)
        return model.predict_rows(table, np.arange(len(table.rows["match_id"]))), table

    report = {"sample": len(sample), "overlapping": n_overlap, "tol": TOL, "candidates": {}}
    for tag, model in {**models, **variants}.items():
        worst = {"cut": 0.0, "poison": 0.0, "later_flip": 0.0}
        reach_ok, n_flip_moved = True, 0
        for m in sample:
            match, players = pairs[m]
            start = match["start_time"]
            base, table = predict(model, m, (match, players), history, results)
            # reach test: what a candidate may ask for
            ctx = CandidateCtx(table, np.arange(len(table.rows["match_id"])))
            reach_ok &= all(r["end"] < start for r in ctx.results_before(start))
            for t in CUTS:
                cm, cp = cut(match, players, t)
                pc, _ = predict(model, m, (cm, cp), history, results)
                pp, _ = predict(
                    model, m, poison(match, players, t, random.Random(m + t)), history, results
                )
                keep = min(t, len(base), len(pc), len(pp))  # a cut match keeps minutes 0..t-1
                worst["cut"] = max(worst["cut"], float(np.abs(pc[:keep] - base[:keep]).max()))
                worst["poison"] = max(worst["poison"], float(np.abs(pp[:keep] - base[:keep]).max()))
            flipped = {
                k: (1 - r if history.meta.get(k, {}).get("end", 0) >= start and k != m else r)
                for k, r in results.items()
            }
            pf, _ = predict(model, m, (match, players), history, flipped)
            d = float(np.abs(pf - base).max())
            worst["later_flip"] = max(worst["later_flip"], d)
            n_flip_moved += d > TOL
        report["candidates"][tag] = {
            **{k: v for k, v in worst.items()},
            "matches_moved_by_later_results": n_flip_moved,
            "reach_ok": bool(reach_ok),
        }
        print(tag, json.dumps(report["candidates"][tag]), flush=True)
    ok = all(
        r["cut"] <= TOL and r["poison"] <= TOL and r["later_flip"] <= TOL and r["reach_ok"]
        for t, r in report["candidates"].items()
        if t in TAGS
    )
    caught = report["candidates"]["A3_BUG_IN_BOTH"]["matches_moved_by_later_results"] > 0
    report["real_candidates_pass"] = ok
    report["planted_bug_caught"] = caught
    print("real candidates pass:", ok, "| planted bug caught:", caught)
    Path("docs/evidence/p1_leak_tests.json").write_text(json.dumps(report, indent=1) + "\n")
    if not (ok and caught):
        raise SystemExit(1)


def _planted_bug_model(store, weeks, history, results, items, usable):
    """A3 with `end < start` replaced by `start < start` in its Elo reader, fitted like A3."""
    src = (FOLDER / "A3" / "train.py").read_text()
    needle = 'results[i]["end"] < start'
    assert src.count(needle) >= 1, "the A3 source no longer has the line the bug is planted in"
    bad = src.replace(needle, 'results[i]["start"] < start')
    folder = Path(tempfile.mkdtemp())
    (folder / "train.py").write_text(bad)
    sha = hashlib.sha256(bad.encode()).hexdigest()
    fit_ids = set(usable)

    def matches():
        for w in weeks:
            for match, players in store.read_week(w):
                if match["match_id"] in fit_ids:
                    yield match, players

    table = CandidateTable.build(matches(), results, items)
    # the buggy copy replays the same history facts as the real one
    table.meta = dict(history.meta)
    table.results = dict(history.results)
    ends = sorted((v["end"], k) for k, v in table.meta.items())
    table.end_sorted = np.array([e for e, _ in ends], dtype=np.int64)
    table.end_ids = [i for _, i in ends]
    table.start = {k: v["start"] for k, v in table.meta.items()}
    spec = fit_candidate(folder / "train.py", sha, table, usable)
    (folder / "model.json").write_text(json.dumps(spec))
    return load_candidate(folder / "model.json")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
