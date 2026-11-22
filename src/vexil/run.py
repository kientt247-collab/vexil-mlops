"""Running the pipeline over one or more weeks, and keeping its tables on disk.

Tables live under ``monitoring/``:

* ``predictions/<week>.parquet``  one row per match, minute and model
* ``matches.parquet``             one row per scored match
* ``match_scores.parquet``        one row per match and model (needs the results)
* ``weekly_counts.parquet``       data-health counts per week
* ``oof_phase2.parquet``          out-of-fold predictions saved by phase 2, development only
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from vexil.candidates import ResultHistory
from vexil.data.store import Store
from vexil.evaluate import match_scores
from vexil.hashing import sha256_file
from vexil.labels import LabelLoader
from vexil.matchdata import is_usable
from vexil.pipeline import MATCH_SCHEMA, WeekOutput, process_week
from vexil.ratings import MatchMeta, RatingState
from vexil.registry import Registry
from vexil.scoring import JUDGED_TIERS, match_inputs


def score_weeks(
    store: Store,
    registry: Registry,
    loader: LabelLoader,
    state: RatingState,
    weeks: list[str],
    purpose: str,
    history: ResultHistory | None = None,
) -> tuple[dict[str, WeekOutput], dict[int, int]]:
    """Process weeks in order. Returns the output per week and all results that were read.

    One result history runs through all the weeks (a fresh one if none is given, which is
    right when `weeks` start at the first week of the store).
    """
    history = history if history is not None else ResultHistory.empty()
    outputs, results = {}, {}
    for week in weeks:
        week_results = loader.week_labels(week, purpose=purpose)
        outputs[week] = process_week(store, week, registry, state, week_results, history)
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
    new_counts = pl.DataFrame([{"week": week, **o.counts} for week, o in outputs.items()])
    tables = (("matches", new_matches), ("match_scores", new_scores), ("weekly_counts", new_counts))
    for name, new in tables:
        path = folder / f"{name}.parquet"
        if path.exists():
            kept = pl.read_parquet(path).filter(~pl.col("week").is_in(weeks))
            # "diagonal": a column added by a newer version is null in older rows
            new = pl.concat([kept, new], how="diagonal_relaxed").select(new.columns)
            new = new.sort("week", maintain_order=True)
        new.write_parquet(path, compression="zstd")


def read_matches(folder: Path) -> pl.DataFrame:
    path = Path(folder) / "matches.parquet"
    return pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=MATCH_SCHEMA)


def effective_scores(folder: Path) -> pl.DataFrame:
    """Scores the monitoring uses, with match facts attached.

    Development matches take the out-of-fold predictions saved by phase 2 (the frozen models
    were fitted on those matches, so their own predictions there are in-sample). Every other
    region takes the predictions of the registered models.
    """
    folder = Path(folder)
    matches = read_matches(folder)
    facts = matches.select(
        "match_id", "start_time", "region", "league_tier", "league_name", "patch", "pair"
    )
    scores = pl.read_parquet(folder / "match_scores.parquet").join(facts, on="match_id")
    scores = scores.filter(pl.col("region") != "development").with_columns(
        source=pl.lit("registered")
    )
    oof_path = folder / "match_scores_oof.parquet"
    if oof_path.exists():
        oof = pl.read_parquet(oof_path).join(facts, on="match_id")
        oof = oof.filter(pl.col("region") == "development").with_columns(
            source=pl.lit("out_of_fold")
        )
        scores = pl.concat([oof.select(scores.columns), scores])
    return scores.sort(["week", "match_id", "model_id"])


def training_rows(store: Store, loader: LabelLoader, weeks: list[str], until: int, purpose: str):
    """Minute rows and outcomes of every usable judged-tier match that started before `until`.

    Ratings are replayed from the first stored week, so each row carries the rating input
    exactly as the pipeline would have computed it at the time.
    """
    state = RatingState()
    columns: dict[str, list] = {}
    y_rows = []
    for week in weeks:
        results = loader.week_labels(week, purpose=purpose)
        rows = list(store.read_week(week))
        metas = [MatchMeta.from_rows(match, players) for match, players in rows]
        ratings = state.process(metas, results)
        for match, players in rows:
            match_id = match["match_id"]
            keep = is_usable(match) and match.get("league_tier") in JUDGED_TIERS
            if not keep or match["start_time"] >= until or match_id not in results:
                continue
            inputs = match_inputs(match, players, ratings[match_id])
            for name, values in inputs.items():
                columns.setdefault(name, []).append(values)
            y_rows.append(np.full(len(inputs["t"]), results[match_id]))
    table = {name: np.concatenate(parts) for name, parts in columns.items()}
    return table, np.concatenate(y_rows)


def write_challenger(folder: Path, model_id: str, spec: dict) -> tuple[Path, str]:
    """Save a challenger model file. Returns its path and SHA-256."""
    path = Path(folder) / f"{model_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    return path, sha256_file(path)
