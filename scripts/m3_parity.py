"""Node M3 parity check of the rating code against phase 2. Runs in a session.

    uv run python scripts/m3_parity.py --phase2 /path/to/v2 --store _store

Label use: match results up to the phase 2 cutoff are read through the label loader (one
logged read per week) to run the ratings. No model is fitted or chosen. The label column of
the saved holdout predictions is not loaded.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vexil.data.store import Store
from vexil.features.a1 import A1_INPUTS, a1_table
from vexil.labels import LabelLoader
from vexil.matchdata import is_usable
from vexil.models.linear import load_model
from vexil.ratings import MatchMeta, RatingState, reference_input
from vexil.weeks import DEVELOPMENT_END, HOLDOUT_START

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=REPO / "_store")
    args = parser.parse_args()
    p2 = args.phase2 / "p2" / "out"
    started = time.time()
    store = Store(args.store, REPO / "data" / "manifest.json")
    loader = LabelLoader(store, REPO / "_work" / "label_reads.log", REPO / "lock.json")
    out: dict = {}

    # one pass, week by week, with a snapshot after every week
    preds = pd.read_csv(
        p2 / "n12" / "holdout_predictions.csv", usecols=["match_id", "t", "p_candidate"]
    )
    wanted = set(preds.match_id)
    state, inputs, metas, results = RatingState(), {}, [], {}
    holdout_rows: dict[int, dict[str, np.ndarray]] = {}
    weeks = store.weeks()
    resume_week = random.Random(20261121).choice(weeks[5:-5])
    with tempfile.TemporaryDirectory() as scratch:
        for week in weeks:
            if week == resume_week:
                state.save(Path(scratch) / "state.json.gz")
                snapshot_bytes = (Path(scratch) / "state.json.gz").stat().st_size
                resumed = RatingState.load(Path(scratch) / "state.json.gz")
                resumed_inputs: dict[int, float] = {}
            week_results = loader.week_labels(week, purpose="M3 rating parity")
            week_metas = []
            for match, players in store.read_week(week):
                week_metas.append(MatchMeta.from_rows(match, players))
                if match["match_id"] in wanted and is_usable(match):
                    holdout_rows[match["match_id"]] = a1_table(match, players)
            inputs.update(state.process(week_metas, week_results))
            if week >= resume_week:
                resumed_inputs.update(resumed.process(week_metas, week_results))
            metas += week_metas
            results.update(week_results)
    out["matches"] = len(metas)
    out["matches with a result"] = len(results)
    out["rated accounts"] = len(state.ratings)
    out["resume from the snapshot before week"] = {
        "week": resume_week,
        "snapshot bytes": snapshot_bytes,
        "matches after it": len(resumed_inputs),
        "largest absolute difference to the single pass": max(
            abs(resumed_inputs[k] - inputs[k]) for k in resumed_inputs
        ),
    }

    # a. development: equal to the rating column phase 2 built in development
    table = pd.read_csv(p2 / "matches.csv")
    base = np.load(p2 / "table_base.npz")
    saved = np.load(p2 / "table_p1.npz")["pelx"]
    first_row = np.flatnonzero(base["t"] == 0)
    saved_by_match = dict(
        zip(table.match_id.to_numpy()[base["m"][first_row]], saved[first_row], strict=True)
    )
    start_of = dict(zip(table.match_id, table.start_time, strict=True))
    dev = [m for m in saved_by_match if start_of[m] < DEVELOPMENT_END]
    out["development matches compared with the phase 2 rating column"] = {
        "matches": len(dev),
        "note": "the phase 2 column is stored as 32-bit floats; compared at that precision",
        "largest absolute difference": max(
            abs(float(np.float32(inputs[m])) - float(saved_by_match[m])) for m in dev
        ),
        "largest absolute difference before rounding to 32 bits": max(
            abs(inputs[m] - float(saved_by_match[m])) for m in dev
        ),
    }

    # b. holdout: predictions of the final model equal the saved ones
    model = load_model(REPO / "models" / "candidate.json")
    worst, n_rows = 0.0, 0
    for match_id, group in preds.groupby("match_id"):
        t = group.t.to_numpy()
        row_inputs = {name: holdout_rows[match_id][name][t].astype(float) for name in A1_INPUTS}
        row_inputs["t"] = t.astype(float)
        row_inputs["pelx"] = np.full(len(t), inputs[match_id])
        worst = max(
            worst, float(np.abs(model.predict(row_inputs) - group.p_candidate.to_numpy()).max())
        )
        n_rows += len(t)
    out["final model, saved holdout predictions"] = {
        "rows": n_rows,
        "matches": len(holdout_rows),
        "largest absolute difference": worst,
    }

    # c. leak tests on real matches
    rng = random.Random(7)
    holdout_metas = [m for m in metas if m.start >= HOLDOUT_START and m.match_id in wanted]
    sample = rng.sample(holdout_metas, 150)
    reference_ok = flipped_ok = 0
    for target in sample:
        reference_ok += (
            abs(reference_input(metas, results, target) - inputs[target.match_id]) < 1e-9
        )
    by_id = {m.match_id: m for m in metas}
    for target in sample[:25]:
        flipped = {k: (1 - v if by_id[k].end >= target.start else v) for k, v in results.items()}
        flipped_ok += (
            abs(RatingState().process(metas, flipped)[target.match_id] - inputs[target.match_id])
            < 1e-9
        )
    out["leak tests on holdout matches"] = {
        "equal to the from-scratch rating that uses only matches ended before the start": (
            f"{reference_ok} of {len(sample)}"
        ),
        "unchanged when every result not yet known at the start is flipped": f"{flipped_ok} of 25",
    }
    out["seconds"] = round(time.time() - started, 1)
    (REPO / "docs" / "evidence" / "m3_parity.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
