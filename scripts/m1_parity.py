"""Node M1 parity check against the phase 2 workspace. Runs in a session, not in CI.

    uv run python scripts/m1_parity.py --phase2 /path/to/v2

It needs the phase 2 folder (raw store, tables, saved predictions), which is not part of
this repository. It writes ``docs/evidence/m1_parity.json`` and ``models/anchor.json``.

Label use: development labels are read once, through the phase 2 loader (which logs the
read), to fit the anchor. The label column of the saved holdout predictions is never loaded.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from vexil.features.a1 import A1_INPUTS, a1_at_minute, a1_table
from vexil.hashing import sha256_file
from vexil.leakcheck import poison
from vexil.matchdata import cut, is_usable, n_rows
from vexil.models.fit import fit_gold_time
from vexil.models.linear import GoldTimeModel, load_model

REPO = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def iter_store(store: Path):
    """Yield (match, players) in match_id order. The match result is removed on read."""
    for match_path in sorted(store.glob("matches_*.jsonl.gz")):
        players_by_match: dict[int, list[dict]] = {}
        for row in read_jsonl(
            match_path.with_name(match_path.name.replace("matches_", "players_"))
        ):
            players_by_match.setdefault(row["match_id"], []).append(row)
        for match in read_jsonl(match_path):
            match.pop("radiant_win", None)
            yield match, players_by_match.get(match["match_id"], [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", type=Path, required=True)
    args = parser.parse_args()
    p2, out = args.phase2 / "p2", {}
    started = time.time()

    # 1. inputs rebuilt from the raw store by the package equal the phase 2 table
    saved = np.load(p2 / "out" / "table_a1.npz")
    rebuilt = {name: [] for name in A1_INPUTS}
    n_matches = 0
    # leak tests on real matches run in the same pass: 200 matches x 3 minutes
    import pandas as pd

    matches = pd.read_csv(p2 / "out" / "matches.csv")
    rng = random.Random(20261121)
    sample = set(rng.sample(sorted(matches.match_id[matches.usable]), 200))
    leak = {"checks": 0, "cut": 0, "poison": 0, "bug_cut": 0, "bug_poison": 0}

    def close(got, want):
        return all(abs(got[k] - want[k]) <= 1e-3 * max(1.0, abs(want[k])) for k in A1_INPUTS)

    for match, players in iter_store(args.phase2 / "data" / "opendota"):
        if not is_usable(match):
            continue
        n_matches += 1
        table = a1_table(match, players)
        for name, values in table.items():
            rebuilt[name].append(values)
        if match["match_id"] not in sample:
            continue
        for t in rng.sample(range(n_rows(match)), 3):
            leak["checks"] += 1
            want = {k: float(v[t]) for k, v in table.items()}
            cut_copy = cut(match, players, t)
            poisoned = poison(match, players, t, rng)
            leak["cut"] += close(a1_at_minute(*cut_copy, t), want)
            leak["poison"] += close(a1_at_minute(*poisoned, t), want)
            try:
                leak["bug_cut"] += not close(a1_at_minute(*cut_copy, t + 1), want)
            except IndexError:
                leak["bug_cut"] += 1
            leak["bug_poison"] += not close(a1_at_minute(*poisoned, t + 1), want)
    rebuilt = {name: np.concatenate(parts) for name, parts in rebuilt.items()}
    out["inputs"] = {
        "usable matches": n_matches,
        "minute rows": int(len(rebuilt["kill_diff"])),
        "largest absolute difference to the phase 2 table": max(
            float(np.abs(rebuilt[name].astype(float) - saved[name].astype(float)).max())
            for name in A1_INPUTS
        ),
    }

    out["leak tests on real matches"] = {
        "checks": leak["checks"],
        "cut test passed": leak["cut"],
        "poison test passed": leak["poison"],
        "planted bug (reads minute t+1) caught by the cut test": leak["bug_cut"],
        "planted bug caught by the poison test": leak["bug_poison"],
    }

    # 2. predictions of the two frozen models on the saved holdout rows (labels not loaded)
    base = np.load(p2 / "out" / "table_base.npz")
    rating = np.load(p2 / "out" / "table_p1.npz")["pelx"].astype(float)  # built in development
    preds = pd.read_csv(
        p2 / "out" / "n12" / "holdout_predictions.csv",
        usecols=["match_id", "t", "p_candidate", "p_rstar"],
    )
    row_of = {
        (int(m), int(t)): i
        for i, (m, t) in enumerate(
            zip(matches.match_id.to_numpy()[base["m"]], base["t"], strict=True)
        )
    }
    rows = np.array(
        [row_of[(int(m), int(t))] for m, t in zip(preds.match_id, preds.t, strict=True)]
    )
    inputs = {name: rebuilt[name][rows].astype(float) for name in A1_INPUTS}
    inputs["t"] = base["t"][rows].astype(float)
    out["holdout predictions"] = {
        "rows": int(len(rows)),
        "rows in minutes 0-29": int((inputs["t"] <= 29).sum()),
    }
    rstar_path = p2 / "out" / "final" / "rstar.json"
    diff = np.abs(load_model(rstar_path).predict(inputs) - preds["p_rstar"].to_numpy())
    out["holdout predictions"]["rstar"] = {
        "sha256": sha256_file(rstar_path),
        "largest absolute difference": float(diff.max()),
    }
    out["holdout predictions"]["candidate"] = (
        "deferred to M3: phase 2 computed the holdout ratings inside its Confirm run and did "
        "not save them, so this comparison needs the rating code of M3"
    )

    # 2b. model arithmetic: package against the frozen phase 2 code on every row of the table,
    #     with the development rating table as the rating input (no labels involved)
    sys.path.insert(0, str(p2))
    import final_model as phase2_model

    table = {name: rebuilt[name].astype(float) for name in A1_INPUTS}
    table["t"] = base["t"].astype(float)
    table["pelx"] = rating
    every_row = np.ones(len(table["t"]), bool)
    out["model arithmetic against phase 2 code, all rows"] = {"rows": int(every_row.sum())}
    for name in ("candidate", "rstar"):
        path = p2 / "out" / "final" / f"{name}.json"
        theirs = phase2_model.predict(phase2_model.load(path), table, every_row)
        ours = load_model(path).predict(table)
        out["model arithmetic against phase 2 code, all rows"][name] = {
            "sha256": sha256_file(path),
            "largest absolute difference": float(np.abs(ours - theirs).max()),
        }

    # 3. anchor: out-of-fold parity with phase 2 ("R2"), then the frozen fit on development
    import kr  # phase 2 code: folds, population and the logged label loader

    ctx = kr.Ctx("M1-anchor (vexil-mlops)")
    t_all, gold_all = ctx.D["t"], ctx.D["gold"]
    saved_oof = np.load(p2 / "out" / "n2_oof.npz")["R2"]
    worst = 0.0
    for train, test in ctx.folds:
        spec = fit_gold_time(t_all[train], gold_all[train], ctx.y[train])
        model = GoldTimeModel(
            spec["mu_u"],
            spec["sd_u"],
            np.array(spec["scale"]),
            np.array(spec["coef"]),
            spec["intercept"],
        )
        p = model.predict({"t": t_all[test], "gold": gold_all[test]})
        worst = max(worst, float(np.abs(p - saved_oof[test]).max()))
    fit_rows = ctx.pop[ctx.D["m"]]
    spec = fit_gold_time(t_all[fit_rows], gold_all[fit_rows], ctx.y[fit_rows])
    spec["fitted_on"] = "development matches of the judged tiers, start before 2026-07-31 00:00 UTC"
    anchor_path = REPO / "models" / "anchor.json"
    anchor_path.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    out["anchor"] = {
        "out-of-fold rows compared": int(sum(test.sum() for _, test in ctx.folds)),
        "largest absolute difference to the phase 2 out-of-fold predictions": worst,
        "fitting rows": spec["n_rows"],
        "sha256": sha256_file(anchor_path),
    }
    out["seconds"] = round(time.time() - started, 1)
    (REPO / "docs" / "evidence" / "m1_parity.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
