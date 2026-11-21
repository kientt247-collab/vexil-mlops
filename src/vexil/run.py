"""Running the pipeline over one or more weeks, and keeping its tables on disk.

Tables live under ``monitoring/``:

* ``predictions/<week>.parquet``  one row per match, minute and model
* ``matches.parquet``             one row per scored match
* ``match_scores.parquet``        one row per match and model (needs the results)
* ``oof_phase2.parquet``          out-of-fold predictions saved by phase 2, development only
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from vexil.data.store import Store
from vexil.evaluate import match_scores
from vexil.labels import LabelLoader
from vexil.pipeline import MATCH_SCHEMA, WeekOutput, process_week
from vexil.ratings import RatingState
from vexil.registry import Registry


def score_weeks(
    store: Store,
    registry: Registry,
    loader: LabelLoader,
    state: RatingState,
    weeks: list[str],
    purpose: str,
) -> tuple[dict[str, WeekOutput], dict[int, int]]:
    """Process weeks in order. Returns the output per week and all results that were read."""
    outputs, results = {}, {}
    for week in weeks:
        week_results = loader.week_labels(week, purpose=purpose)
        outputs[week] = process_week(store, week, registry, state, week_results)
        results.update(week_results)
    return outputs, results


def write_tables(folder: Path, outputs: dict[str, WeekOutput], results: dict[int, int]) -> None:
    """Write (or replace) the tables of the given weeks under `folder`."""
    folder = Path(folder)
    (folder / "predictions").mkdir(parents=True, exist_ok=True)
    weeks = list(outputs)
    for week, output in outputs.items():
        output.predictions.with_columns(pl.col("p").cast(pl.Float32)).write_parquet(
            folder / "predictions" / f"{week}.parquet", compression="zstd"
        )
    new_matches = pl.concat([o.matches for o in outputs.values()])
    new_scores = pl.concat([match_scores(o.predictions, results) for o in outputs.values()])
    for name, new in (("matches", new_matches), ("match_scores", new_scores)):
        path = folder / f"{name}.parquet"
        if path.exists():
            kept = pl.read_parquet(path).filter(~pl.col("week").is_in(weeks))
            # "diagonal": a column added by a newer version is null in older rows
            new = pl.concat([kept, new], how="diagonal_relaxed").select(new.columns)
            new = new.sort(["week", "match_id"], maintain_order=True)
        new.write_parquet(path, compression="zstd")


def read_matches(folder: Path) -> pl.DataFrame:
    path = Path(folder) / "matches.parquet"
    return pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=MATCH_SCHEMA)
