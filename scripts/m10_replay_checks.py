"""Node M10 checks on a finished replay. Runs in a session.

    uv run python scripts/m10_replay_checks.py --full /path/to/replay_root --store _store

1. Pipeline-level leak test: the replay is run again with the store cut after week k, for
   three values of k. Everything the job wrote for weeks up to k must be identical to what
   the full-year replay wrote for those weeks.
2. Totals: the scores of the full replay equal the tables built in one pass at node M5.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import polars as pl

from vexil.evaluate import paired_gain, summarise
from vexil.hashing import sha256_file
from vexil.issues import LogChannel
from vexil.job import Layout, replay

REPO = Path(__file__).resolve().parents[1]
FIRST = "2025-12-29"
CUTS = ("2026-03-02", "2026-06-29", "2026-09-28")
SEED_FILES = ("rules.yaml", "gate.yaml", "oof_phase2.parquet", "match_scores_oof.parquet")


def scratch_root(target: Path, last_week: str) -> Layout:
    """A fresh root with the frozen files and a manifest that ends at `last_week`."""
    (target / "monitoring").mkdir(parents=True)
    (target / "data").mkdir()
    shutil.copytree(
        REPO / "models", target / "models", ignore=shutil.ignore_patterns("challengers")
    )
    registry = (REPO / "models" / "registry.yaml").read_text().split("\n  - id: refit-")[0]
    registry = registry.split("\n  - id: recalibration-")[0].rstrip("\n") + "\n"
    (target / "models" / "registry.yaml").write_text(registry)
    for name in SEED_FILES:
        shutil.copy(REPO / "monitoring" / name, target / "monitoring" / name)
    manifest = json.loads((REPO / "data" / "manifest.json").read_text())
    manifest["weeks"] = {w: e for w, e in manifest["weeks"].items() if w <= last_week}
    (target / "data" / "manifest.json").write_text(json.dumps(manifest))
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=REPO / "_store")
    args = parser.parse_args()
    started = time.time()
    full = args.full / "monitoring"
    out: dict = {"pipeline-level leak test": {}}
    for cut in CUTS:
        with tempfile.TemporaryDirectory() as scratch:
            root = scratch_root(Path(scratch) / "root", cut)
            replay(
                Layout(root, args.store),
                FIRST,
                cut,
                LogChannel(root / "monitoring" / "issues.jsonl"),
            )
            part = root / "monitoring"
            weeks = sorted(p.stem for p in (part / "predictions").glob("*.parquet"))
            same_files = sum(
                sha256_file(part / "predictions" / f"{w}.parquet")
                == sha256_file(full / "predictions" / f"{w}.parquet")
                for w in weeks
            )
            tables = {}
            for name, keys in (
                ("match_scores", ["week", "match_id", "model_id"]),
                ("matches", ["week", "match_id"]),
                ("weekly_counts", ["week"]),
                ("alert_states", ["week", "family", "subject", "statistic"]),
            ):
                ours = pl.read_parquet(part / f"{name}.parquet").sort(keys)
                theirs = (
                    pl.read_parquet(full / f"{name}.parquet")
                    .filter(pl.col("week") <= cut)
                    .sort(keys)
                )
                if name == "alert_states":
                    theirs = theirs.filter(pl.col("week").is_in(ours["week"].unique().to_list()))
                tables[name] = {"rows": ours.height, "identical": ours.equals(theirs)}
            out["pipeline-level leak test"][cut] = {
                "weeks": len(weeks),
                "prediction files byte-identical": f"{same_files} of {len(weeks)}",
                "tables": tables,
            }

    # totals against the one-pass tables of M5 (the three starting models)
    one_pass = pl.read_parquet(REPO / "docs" / "evidence" / "m5_match_scores.parquet")
    starting = ["final-model", "rstar", "anchor"]
    replayed = pl.read_parquet(full / "match_scores.parquet").filter(
        pl.col("model_id").is_in(starting)
    )
    keys = ["week", "match_id", "model_id"]
    out["scores of the replay against the one-pass tables of M5"] = {
        "rows": replayed.height,
        "identical": replayed.sort(keys).select(one_pass.columns).equals(one_pass.sort(keys)),
    }
    holdout = replayed.join(
        pl.read_parquet(full / "matches.parquet").select("match_id", "region"), on="match_id"
    )
    holdout = holdout.filter(pl.col("region") == "holdout")
    pooled = {
        r["model_id"]: r["log_loss"] for r in summarise(holdout, ["model_id"]).iter_rows(named=True)
    }
    out["holdout numbers from the replay"] = {
        "final model": pooled["final-model"],
        "R*": pooled["rstar"],
        "gain": paired_gain(holdout, "final-model", "rstar", [])["gain"][0],
    }
    out["seconds"] = round(time.time() - started, 1)
    (REPO / "docs" / "evidence" / "m10_replay_checks.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
