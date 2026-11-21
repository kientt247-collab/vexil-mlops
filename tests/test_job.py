"""The weekly job end to end on a small synthetic store: replay, leak test, dashboard, lock."""

import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from vexil.dashboard import build, build_data
from vexil.data.store import Store
from vexil.evaluate import paired_gain
from vexil.issues import LogChannel
from vexil.job import ClockError, Layout, LockError, lock_contents, replay, run_job
from vexil.run import effective_scores
from vexil.weeks import week_bounds, weeks_between

from .synthetic import make_match

REPO = Path(__file__).resolve().parents[1]
WEEKS = weeks_between("2026-08-03", "2026-10-19")  # 12 weeks inside the phase 2 holdout region


def make_root(tmp: Path, weeks: list[str]) -> Layout:
    """A scratch repository root with the frozen files and a synthetic store."""
    root = tmp / "root"
    (root / "monitoring").mkdir(parents=True)
    shutil.copytree(REPO / "models", root / "models", ignore=shutil.ignore_patterns("challengers"))
    registry = (REPO / "models" / "registry.yaml").read_text()
    for marker in ("\n  - id: refit-", "\n  - id: recalibration-"):
        registry = registry.split(marker)[0]
    (root / "models" / "registry.yaml").write_text(registry.rstrip("\n") + "\n")
    for name in ("rules.yaml", "gate.yaml"):
        shutil.copy(REPO / "monitoring" / name, root / "monitoring" / name)
    store = Store(tmp / "store", root / "data" / "manifest.json")
    seed = 0
    for week in weeks:
        start, end = week_bounds(week)
        matches, players = [], []
        for when in range(start + 3600, end, 86400 // 4):
            seed += 1
            match, match_players = make_match(seed, start_time=when)
            match.update(league_tier="professional", league_name="Test League", patch="7.41")
            matches.append(match)
            players += match_players
        store.write_week(week, matches, players, source="test")
    return Layout(root, tmp / "store")


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("job")
    layout = make_root(tmp, WEEKS)
    channel = LogChannel(layout.monitoring / "replay_issues.jsonl")
    summaries = replay(layout, WEEKS[0], WEEKS[-1], channel)
    return layout, summaries


def test_replay_scores_every_week_once_and_keeps_few_snapshots(replayed):
    layout, summaries = replayed
    assert len(summaries) == len(WEEKS) + 2
    scores = pl.read_parquet(layout.monitoring / "match_scores.parquet")
    assert sorted(scores["week"].unique()) == WEEKS
    assert scores.group_by(["match_id", "model_id"]).len()["len"].max() == 1
    states = pl.read_parquet(layout.monitoring / "alert_states.parquet")
    assert sorted(states["week"].unique()) == WEEKS  # the closing runs made every week final
    assert len(list(layout.state.glob("ratings-before-*.json.gz"))) <= 4
    log = [json.loads(line) for line in layout.label_log.read_text().splitlines()]
    assert all(record["after_cutoff"] == 0 for record in log)


def test_pipeline_level_leak_test(replayed, tmp_path):
    """What the job wrote for weeks up to k does not depend on later weeks being in the store."""
    layout, _ = replayed
    cut = WEEKS[6]
    short = make_root(tmp_path, WEEKS[:7])
    replay(short, WEEKS[0], cut, LogChannel(short.monitoring / "replay_issues.jsonl"))
    for week in WEEKS[:7]:
        ours = (short.monitoring / "predictions" / f"{week}.parquet").read_bytes()
        assert ours == (layout.monitoring / "predictions" / f"{week}.parquet").read_bytes()
    keys = ["week", "match_id", "model_id"]
    ours = pl.read_parquet(short.monitoring / "match_scores.parquet").sort(keys)
    full = pl.read_parquet(layout.monitoring / "match_scores.parquet").filter(pl.col("week") <= cut)
    assert ours.equals(full.sort(keys))
    keys = ["week", "family", "subject", "statistic"]
    ours = pl.read_parquet(short.monitoring / "alert_states.parquet").sort(keys)
    full = pl.read_parquet(layout.monitoring / "alert_states.parquet").filter(pl.col("week") <= cut)
    assert ours.equals(full.sort(keys))


def test_running_the_same_monday_twice_changes_nothing(replayed):
    layout, summaries = replayed
    before = (layout.monitoring / "match_scores.parquet").read_bytes()
    states = (layout.monitoring / "alert_states.parquet").read_bytes()
    channel = LogChannel(layout.monitoring / "again.jsonl")
    run_job(layout, summaries[-1]["now"], channel, until_week=WEEKS[-1])
    assert (layout.monitoring / "match_scores.parquet").read_bytes() == before
    assert (layout.monitoring / "alert_states.parquet").read_bytes() == states


def test_dashboard_numbers_equal_the_tables(replayed, tmp_path):
    layout, _ = replayed
    data = build_data(layout)
    assert data["weeks"] == WEEKS and data["champion"] == "final-model"
    scores = effective_scores(layout.monitoring)
    # rolling 8-week gain of the champion over the anchor, last week, straight from the tables
    last8 = scores.filter(pl.col("week").is_in(WEEKS[-8:]))
    direct = paired_gain(last8, "final-model", "anchor", [])["gain"][0]
    assert data["models"]["final-model"]["gain_rolling"][-1] == pytest.approx(direct, abs=1e-6)
    week = scores.filter(pl.col("week") == WEEKS[3])
    assert data["models"]["rstar"]["gain_week"][3] == pytest.approx(
        paired_gain(week, "rstar", "anchor", [])["gain"][0], abs=1e-6
    )
    counts = pl.read_parquet(layout.monitoring / "weekly_counts.parquet").sort("week")
    assert data["judged_matches"] == counts["judged"].to_list()
    pooled = paired_gain(scores, "final-model", "rstar", [])["gain"][0]
    assert data["regions"][0]["gain_champion_over_baseline"] == pytest.approx(pooled, abs=1e-4)
    page = build(layout, tmp_path / "site").read_text()
    embedded = json.loads(page.split("const D = ")[1].split(";\nconst $")[0])
    assert embedded["models"] == json.loads((tmp_path / "site" / "data.json").read_text())["models"]
    assert "/*__DATA__*/" not in page


def test_a_locked_file_that_changes_stops_the_job(replayed, tmp_path):
    layout = make_root(tmp_path, WEEKS[:3])
    shutil.copytree(REPO / "src", layout.root / "src")
    layout.lock.write_text(json.dumps(lock_contents(layout, 1, "2026-11-22 00:00 UTC")))
    channel = LogChannel(layout.monitoring / "issues.jsonl")
    run_job(layout, week_bounds(WEEKS[3])[0] + 11820, channel)  # hashes match: the job runs
    rules = layout.rules.read_text()
    layout.rules.write_text(rules.replace("window_weeks: 8", "window_weeks: 6"))
    with pytest.raises(LockError, match="rules.yaml"):
        run_job(layout, week_bounds(WEEKS[3])[0] + 11820, channel)
    layout.rules.write_text(rules)
    (layout.root / "src" / "vexil" / "alerts.py").write_text("# changed\n")
    with pytest.raises(LockError, match="code"):
        run_job(layout, week_bounds(WEEKS[3])[0] + 11820, channel)


def test_a_live_run_earlier_than_the_last_run_is_refused(tmp_path):
    layout = make_root(tmp_path, WEEKS[:3])
    channel = LogChannel(layout.monitoring / "issues.jsonl")
    monday = week_bounds(WEEKS[3])[0] + 11820
    run_job(layout, monday, channel)

    def explorer(sql):
        raise AssertionError("the pull must not start")

    with pytest.raises(ClockError, match="earlier than the last recorded run"):
        run_job(layout, monday - 86400, channel, explorer)
