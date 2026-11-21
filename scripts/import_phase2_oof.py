"""Import the out-of-fold predictions that phase 2 saved for the development region.

    uv run python scripts/import_phase2_oof.py --phase2 /path/to/v2 --store _store

The frozen models were fitted on all development matches, so their own predictions there
are in-sample. Phase 2 also saved, for every development match from 2026-01-14 on, a
prediction made by the same recipe fitted on earlier matches only. Those are what the
dashboard shows for the development region. Results are read through the label loader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from vexil.data.store import Store
from vexil.evaluate import match_scores, summarise
from vexil.labels import LabelLoader
from vexil.weeks import week_label

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=REPO / "_store")
    args = parser.parse_args()
    out_dir = args.phase2 / "p2" / "out"
    table = pd.read_csv(out_dir / "matches.csv")
    base = np.load(out_dir / "table_base.npz")
    final = np.load(out_dir / "n11_oof.npz")
    ladder = np.load(out_dir / "n2_oof.npz")
    rows = final["block"] > 0
    assert np.array_equal(rows, ladder["block"] > 0)
    match_id = table.match_id.to_numpy()[base["m"][rows]]
    week = np.array([week_label(int(s)) for s in table.start_time.to_numpy()[base["m"][rows]]])
    columns = {"final-model": final["candidate"], "rstar": final["rstar"], "anchor": ladder["R2"]}
    predictions = pl.concat(
        pl.DataFrame(
            {
                "week": week,
                "match_id": match_id,
                "t": base["t"][rows].astype("int16"),
                "model_id": model_id,
                "p": values[rows].astype(float),
            }
        )  # fmt: skip
        for model_id, values in columns.items()
    )
    store = Store(args.store, REPO / "data" / "manifest.json")
    loader = LabelLoader(store, REPO / "_work" / "label_reads.log", REPO / "lock.json")
    results: dict[int, int] = {}
    for name in sorted(set(week)):
        results.update(
            loader.week_labels(name, purpose="import of phase 2 out-of-fold predictions")
        )
    scores = match_scores(predictions, results)
    folder = REPO / "monitoring"
    predictions.with_columns(pl.col("p").cast(pl.Float32)).write_parquet(
        folder / "oof_phase2.parquet", compression="zstd"
    )
    scores.write_parquet(folder / "match_scores_oof.parquet", compression="zstd")
    pooled = {
        r["model_id"]: r["log_loss"] for r in summarise(scores, ["model_id"]).iter_rows(named=True)
    }
    phase2 = json.loads((out_dir / "n11_results.json").read_text())
    summary = {
        "matches": scores["match_id"].n_unique(),
        "minute rows per model": int(rows.sum()),
        "weeks": [str(min(week)), str(max(week))],
        "pooled out-of-fold log loss 0-29": pooled,
        "phase 2 number for R* (out-of-fold, minutes 0-29)": phase2["R* oof_0_29"],
    }
    (REPO / "docs" / "evidence" / "oof_import.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
