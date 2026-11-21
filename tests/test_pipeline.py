import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from vexil.data.store import Store
from vexil.evaluate import add_groups, calibration, log_loss_rows, paired_gain, summarise
from vexil.labels import LabelLoader
from vexil.ratings import RatingState
from vexil.registry import load_registry
from vexil.run import score_weeks, write_tables
from vexil.weeks import CUTOFF, week_bounds

from .synthetic import make_match

REPO = Path(__file__).resolve().parents[1]
WEEKS = ("2026-11-02", "2026-11-09", "2026-11-16")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Three synthetic weeks through the real pipeline with the real registry."""
    tmp = tmp_path_factory.mktemp("run")
    store = Store(tmp / "store", tmp / "manifest.json")
    seed = 0
    for week in WEEKS:
        start, end = week_bounds(week)
        matches, players = [], []
        for when in range(start + 3600, end, 86400 // 2):
            seed += 1
            match, match_players = make_match(seed, start_time=when)
            match.update(league_tier="professional", league_name="Test League", patch="7.42")
            if seed % 5 == 0:
                match["league_tier"] = "excluded"  # rated, never scored
            matches.append(match)
            players += match_players
        store.write_week(week, matches, players, source="test")
    loader = LabelLoader(store, tmp / "label_reads.log", tmp / "lock.json")
    registry = load_registry(REPO / "models" / "registry.yaml")
    outputs, results = score_weeks(store, registry, loader, RatingState(), list(WEEKS), "test")
    write_tables(tmp / "monitoring", outputs, results)
    return tmp, store, outputs, results


def test_only_judged_matches_outside_the_gap_are_scored(run):
    _, _, outputs, _ = run
    for output in outputs.values():
        counts = output.counts
        assert counts["judged"] == output.matches.height
        assert counts["matches"] > counts["judged_tier"] >= counts["judged"] + counts["gap"]
        assert set(output.matches["region"]) <= {"holdout"}
    assert outputs["2026-11-16"].counts["gap"] > 0  # 20 to 22 November are never scored
    assert outputs["2026-11-02"].counts["gap"] == 0


def test_every_scored_match_has_one_row_per_minute_and_model(run):
    _, _, outputs, _ = run
    for output in outputs.values():
        per_match = output.predictions.group_by("match_id").len()
        expected = output.matches.select(
            "match_id", (pl.col("rows") * 3).alias("len").cast(pl.UInt32)
        )
        assert per_match.sort("match_id").equals(expected.sort("match_id"))
        assert output.predictions["p"].is_between(0, 1).all()


def test_results_after_the_cutoff_never_reach_the_scores_without_the_lock(run):
    tmp, _, _, results = run
    scores = pl.read_parquet(tmp / "monitoring" / "match_scores.parquet")
    matches = pl.read_parquet(tmp / "monitoring" / "matches.parquet")
    assert scores.join(matches, on="match_id").filter(pl.col("start_time") >= CUTOFF).height == 0
    log = [json.loads(line) for line in (tmp / "label_reads.log").read_text().splitlines()]
    assert sum(record["after_cutoff"] for record in log) == 0
    assert log[-1]["withheld"] > 0


def test_scores_equal_a_direct_computation_and_weeks_add_up(run):
    tmp, _, outputs, results = run
    scores = pl.read_parquet(tmp / "monitoring" / "match_scores.parquet")
    predictions = pl.concat([o.predictions for o in outputs.values()])
    early = predictions.filter((pl.col("t") < 30) & (pl.col("model_id") == "rstar"))
    y = np.array([results[m] for m in early["match_id"]])
    direct = log_loss_rows(y, early["p"].to_numpy()).mean()
    pooled = summarise(scores.filter(pl.col("model_id") == "rstar"), ["model_id"])
    assert pooled["log_loss"][0] == pytest.approx(direct, abs=1e-12)
    weekly = summarise(scores.filter(pl.col("model_id") == "rstar"), ["week"])
    readded = (weekly["log_loss"] * weekly["rows"]).sum() / weekly["rows"].sum()
    assert readded == pytest.approx(direct, abs=1e-12)
    gain = paired_gain(scores, "final-model", "rstar", [])["gain"][0]
    final = summarise(scores.filter(pl.col("model_id") == "final-model"), ["model_id"])["log_loss"][
        0
    ]
    assert gain == pytest.approx(direct - final, abs=1e-12)


def test_a_rerun_of_the_same_weeks_replaces_rows_instead_of_adding_them(run):
    tmp, store, outputs, results = run
    before = pl.read_parquet(tmp / "monitoring" / "match_scores.parquet")
    write_tables(tmp / "monitoring", {WEEKS[1]: outputs[WEEKS[1]]}, results)
    after = pl.read_parquet(tmp / "monitoring" / "match_scores.parquet")
    assert after.sort(["week", "match_id", "model_id"]).equals(
        before.sort(["week", "match_id", "model_id"])
    )


def test_calibration_recovers_a_planted_slope():
    rng = np.random.default_rng(1)
    true_log_odds = rng.normal(0, 1.5, 200_000)
    y = (rng.random(200_000) < 1 / (1 + np.exp(-true_log_odds))).astype(float)
    overconfident = 1 / (1 + np.exp(-true_log_odds / 0.85))
    slope, intercept = calibration(y, overconfident)
    assert slope == pytest.approx(0.85, abs=0.01) and intercept == pytest.approx(0, abs=0.01)


def test_groups_split_after_three_hours():
    matches = pl.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "pair": ["t:1-2", "t:1-2", "t:1-2", "t:3-4"],
            "start_time": [0, 3600, 3600 + 3 * 3600, 100],
        }
    )
    groups = dict(add_groups(matches).select("match_id", "group").iter_rows())
    assert groups[1] == groups[2] != groups[3]
    assert len(set(groups.values())) == 3
