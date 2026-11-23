"""A frozen_train challenger through the whole weekly job: replay, scored_from, history, gate."""

import hashlib
import json

import polars as pl
import pytest

from vexil.candidates import CandidateTable, ResultHistory, fit_candidate
from vexil.data.store import Store
from vexil.features.extra import load_items
from vexil.issues import LogChannel
from vexil.job import SNAPSHOTS_KEPT, replay
from vexil.labels import LabelLoader
from vexil.registry import append_entry, load_registry

from .test_job import REPO, WEEKS, make_root

FIRST = 2  # the candidate is scored from this week on


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("candjob")
    layout = make_root(tmp, WEEKS)
    store = Store(layout.store_dir, layout.manifest)
    loader = LabelLoader(store, layout.label_log, layout.lock)
    first = list(store.read_week(WEEKS[0]))
    results = loader.week_labels(WEEKS[0], "test fit")
    table = CandidateTable.build(first, results, load_items())
    ids = sorted({int(m) for m in table.rows["match_id"]} & set(results))
    folder = layout.root / "models" / "candidates" / "A3"
    train = REPO / "models" / "candidates" / "A3" / "train.py"
    sha = hashlib.sha256(train.read_bytes()).hexdigest()
    (folder / "model.json").write_text(json.dumps(fit_candidate(train, sha, table, ids)))
    append_entry(
        layout.registry,
        {
            "id": "cand-A3", "role": "challenger", "kind": "frozen_train",
            "file": "models/candidates/A3/model.json",
            "sha256": hashlib.sha256((folder / "model.json").read_bytes()).hexdigest(),
            "description": "test", "scored_from": WEEKS[FIRST],
        },
    )  # fmt: skip
    layout.challengers.write_text(
        json.dumps(
            [
                {
                    "id": "cand-A3", "kind": "frozen_train", "trigger": "test",
                    "shadow_from": WEEKS[FIRST], "status": "shadow",
                }
            ]
        )
    )  # fmt: skip
    summaries = replay(layout, WEEKS[0], WEEKS[-1], LogChannel(layout.monitoring / "issues.jsonl"))
    return layout, summaries


def test_the_candidate_is_scored_only_from_its_first_week(replayed):
    layout, _ = replayed
    scores = pl.read_parquet(layout.monitoring / "match_scores.parquet")
    weeks = sorted(scores.filter(pl.col("model_id") == "cand-A3")["week"].unique())
    assert weeks and weeks[0] == WEEKS[FIRST]
    assert sorted(scores.filter(pl.col("model_id") == "rstar")["week"].unique())[0] == WEEKS[0]


def test_the_job_keeps_a_history_snapshot_next_to_the_rating_snapshots(replayed):
    layout, _ = replayed
    snapshots = sorted(layout.state.glob("history-before-*.json.gz"))
    assert 1 <= len(snapshots) <= SNAPSHOTS_KEPT
    history = ResultHistory.load(snapshots[-1])
    assert len(history.meta) > 0 and set(history.results) <= set(history.meta)


def test_the_gate_reaches_the_candidate_after_its_shadow_weeks(replayed):
    layout, _ = replayed
    record = next(r for r in json.loads(layout.challengers.read_text()) if r["id"] == "cand-A3")
    assert record["status"] in ("rejected", "proposed"), record
    registry = load_registry(layout.registry, layout.root)
    role = next(e.role for e in registry.entries if e.id == "cand-A3")
    assert role == ("retired" if record["status"] == "rejected" else "challenger")
