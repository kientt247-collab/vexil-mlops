"""Node M5: score every stored week and compare the holdout numbers with phase 2.

    uv run python scripts/m5_parity.py --phase2 /path/to/v2 --store _store

Label use: results up to the phase 2 cutoff are read through the label loader (logged) for
the ratings and for the scores. The holdout labels were first read by phase 2 on
2026-11-21; the pooled numbers reproduced here are phase 2's own, and every other number
this project computes on the holdout region is POST-HOC.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from vexil.data.store import Store
from vexil.evaluate import PHASES, calibration, match_scores, paired_gain, summarise
from vexil.labels import LabelLoader
from vexil.ratings import RatingState
from vexil.registry import load_registry
from vexil.run import score_weeks, write_tables

REPO = Path(__file__).resolve().parents[1]
NAMES = {"final-model": "candidate", "rstar": "R*"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=REPO / "_store")
    args = parser.parse_args()
    started = time.time()
    store = Store(args.store, REPO / "data" / "manifest.json")
    loader = LabelLoader(store, REPO / "_work" / "label_reads.log", REPO / "lock.json")
    registry = load_registry(REPO / "models" / "registry.yaml")
    state = RatingState()
    outputs, results = score_weeks(store, registry, loader, state, store.weeks(), "M5 evaluation")
    write_tables(REPO / "monitoring", outputs, results)

    matches = pl.concat([o.matches for o in outputs.values()])
    predictions = pl.concat([o.predictions for o in outputs.values()])
    scores = match_scores(predictions, results)
    out: dict = {
        "weeks": len(outputs),
        "scored matches by region": dict(
            matches.group_by("region").len().sort("region").iter_rows()
        ),
        "prediction rows": predictions.height,
    }

    # phase 2 holdout set: the match ids of the saved predictions (their label column is not read)
    saved = pd.read_csv(
        args.phase2 / "p2" / "out" / "n12" / "holdout_predictions.csv", usecols=["match_id"]
    )
    phase2_ids = sorted(set(saved.match_id))
    ours = set(matches.filter(pl.col("region") == "holdout")["match_id"])
    out["holdout match set"] = {
        "phase 2": len(phase2_ids),
        "this pipeline": len(ours),
        "only in this pipeline": len(ours - set(phase2_ids)),
        "only in phase 2": len(set(phase2_ids) - ours),
    }
    expected = json.loads(
        (args.phase2 / "p2" / "out" / "n12" / "holdout_results.json").read_text()
    )["scores"]
    on_set = scores.filter(pl.col("match_id").is_in(phase2_ids))
    pooled = {row["model_id"]: row for row in summarise(on_set, ["model_id"]).iter_rows(named=True)}
    rows = predictions.filter(pl.col("match_id").is_in(phase2_ids) & (pl.col("t") < 30))

    def slope(model_id: str, low: int = 0, high: int = 30) -> float:
        part = rows.filter(
            (pl.col("model_id") == model_id) & (pl.col("t") >= low) & (pl.col("t") < high)
        )
        y = np.array([results[m] for m in part["match_id"]])
        return calibration(y, part["p"].to_numpy())[0]

    gain = paired_gain(on_set, "final-model", "rstar", [])["gain"][0]
    compare = {
        "rows 0-29": (int(pooled["final-model"]["rows"]), expected["rows 0-29"]),
        "candidate log loss 0-29": (
            pooled["final-model"]["log_loss"],
            expected["candidate log loss 0-29"],
        ),
        "R* log loss 0-29": (pooled["rstar"]["log_loss"], expected["R* log loss 0-29"]),
        "gain": (gain, expected["gain"]),
        "candidate slope 0-29": (slope("final-model"), expected["candidate slope 0-29"]),
        "R* slope 0-29": (slope("rstar"), expected["R* slope 0-29"]),
    }
    by_phase = expected["by phase (0-9, 10-19, 20-29, 30+)"]
    for model_id, name in NAMES.items():
        for i, phase in enumerate(PHASES):
            compare[f"{name} log loss {phase}"] = (
                pooled[model_id][f"log_loss_{phase}"],
                by_phase[name][i],
            )
    for i, (low, high) in enumerate(((0, 10), (10, 20), (20, 30))):
        compare[f"candidate slope {PHASES[i]}"] = (
            slope("final-model", low, high),
            by_phase["candidate slope"][i],
        )
    out["parity with phase 2 on its holdout set"] = {
        key: {"ours": ours_value, "phase 2": theirs, "difference": abs(ours_value - theirs)}
        for key, (ours_value, theirs) in compare.items()
    }
    out["largest difference"] = max(
        v["difference"] for v in out["parity with phase 2 on its holdout set"].values()
    )

    # weekly numbers add back to the pooled ones
    weekly = summarise(on_set, ["week", "model_id"])
    readded = weekly.group_by("model_id").agg(
        ((pl.col("log_loss") * pl.col("rows")).sum() / pl.col("rows").sum()).alias("log_loss")
    )
    out["weekly numbers added back, largest difference to pooled"] = max(
        abs(row["log_loss"] - pooled[row["model_id"]]["log_loss"])
        for row in readded.iter_rows(named=True)
    )
    out["anchor on the phase 2 holdout set (POST-HOC)"] = {
        "log loss 0-29": pooled["anchor"]["log_loss"],
        "gain of the final model over the anchor": paired_gain(on_set, "final-model", "anchor", [])[
            "gain"
        ][0],
        "gain of R* over the anchor": paired_gain(on_set, "rstar", "anchor", [])["gain"][0],
    }
    out["seconds"] = round(time.time() - started, 1)
    (REPO / "docs" / "evidence" / "m5_parity.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
