"""A candidate (kind frozen_train) scored by the real pipeline next to the champion."""

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from vexil.candidates import CandidateTable, ResultHistory, load_candidate, week_table
from vexil.data.store import Store
from vexil.features.extra import load_items
from vexil.labels import LabelLoader
from vexil.ratings import RatingState
from vexil.registry import append_entry, load_registry
from vexil.run import score_weeks
from vexil.weeks import week_bounds

from .synthetic import make_match

REPO = Path(__file__).resolve().parents[1]
WEEKS = ("2026-11-02", "2026-11-09")
TAGS = ("A1", "A3", "A4", "A5")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cand")
    shutil.copytree(REPO / "models", tmp / "models")
    store = Store(tmp / "store", tmp / "manifest.json")
    seed = 0
    for week in WEEKS:
        start, end = week_bounds(week)
        matches, players = [], []
        for when in range(start + 3600, end, 86400 // 2):
            seed += 1
            match, match_players = make_match(seed, start_time=when)
            match.update(league_tier="professional", league_name="Test League", patch="7.42")
            for p in match_players:
                p["hero_id"] = 1 + p["player_slot"] % 128
            if seed % 5 == 0:
                match["league_tier"] = "excluded"
            matches.append(match)
            players += match_players
        store.write_week(week, matches, players, source="test")
    # a model file fitted on the first week, written like the real ones
    from vexil.candidates import fit_candidate

    loader = LabelLoader(store, tmp / "label_reads.log", tmp / "lock.json")
    first = list(store.read_week(WEEKS[0]))
    results = loader.week_labels(WEEKS[0], "test")
    table = CandidateTable.build(first, results, load_items())
    ids = sorted({int(m) for m in table.rows["match_id"]} & set(results))
    registry_path = tmp / "models" / "registry.yaml"
    for tag in TAGS:
        folder = tmp / "models" / "candidates" / tag
        train = folder / "train.py"
        sha = hashlib.sha256(train.read_bytes()).hexdigest()
        import json

        spec = fit_candidate(train, sha, table, ids)
        (folder / "model.json").write_text(json.dumps(spec))
        append_entry(
            registry_path,
            {
                "id": f"cand-{tag}", "role": "challenger", "kind": "frozen_train",
                "file": f"models/candidates/{tag}/model.json",
                "sha256": hashlib.sha256((folder / "model.json").read_bytes()).hexdigest(),
                "description": "test candidate",
            },
        )  # fmt: skip
    registry = load_registry(registry_path, tmp)
    outputs, results = score_weeks(store, registry, loader, RatingState(), list(WEEKS), "test")
    return tmp, store, registry, outputs, results


def test_candidates_get_one_row_per_match_minute_and_model(run):
    _, _, _, outputs, _ = run
    for output in outputs.values():
        pred = output.predictions
        per_model = pred.group_by("model_id").len()
        counts = dict(zip(per_model["model_id"], per_model["len"], strict=True))
        assert {f"cand-{t}" for t in TAGS} <= set(counts)
        assert len({counts[f"cand-{t}"] for t in TAGS}) == 1
        assert counts[f"cand-{TAGS[0]}"] == counts["final-model"]
        assert pred["p"].is_finite().all() and pred["p"].min() >= 0 and pred["p"].max() <= 1


def test_pipeline_rows_equal_direct_prediction(run):
    """What the weekly job stores is what the candidate gives for the same table."""
    tmp, store, registry, outputs, results = run
    week = WEEKS[1]
    history = ResultHistory.empty()
    first = list(store.read_week(WEEKS[0]))
    history.add_week(
        first, {m: r for m, r in results.items() if m in {x["match_id"] for x, _ in first}}
    )
    rows = list(store.read_week(week))
    ids = set(outputs[week].matches["match_id"].to_list())
    table = week_table(history, rows, ids, {m: r for m, r in results.items()}, load_items())
    for tag in TAGS:
        direct = load_candidate(tmp / "models" / "candidates" / tag / "model.json").predict_rows(
            table, np.arange(len(table.rows["match_id"]))
        )
        stored = outputs[week].predictions.filter(
            outputs[week].predictions["model_id"] == f"cand-{tag}"
        )
        order = np.lexsort((stored["t"].to_numpy(), stored["match_id"].to_numpy()))
        assert np.allclose(stored["p"].to_numpy()[order], direct, atol=1e-12, rtol=0)


def test_history_advances_by_the_week(run):
    _, store, _, _, results = run
    history = ResultHistory.empty()
    for week in WEEKS:
        rows = list(store.read_week(week))
        history.add_week(rows, results)
    assert len(history.meta) == sum(len(list(store.read_week(w))) for w in WEEKS)
