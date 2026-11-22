"""The candidate layer: state files, result history, the ctx contract, and week batches."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from vexil.candidates import (
    CandidateCtx,
    CandidateTable,
    ResultHistory,
    decode_state,
    encode_state,
    fit_candidate,
    load_candidate,
    load_train,
    week_table,
)
from vexil.features.extra import load_items

from .synthetic import make_match

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = REPO / "models" / "candidates"
TAGS = ("A1", "A3", "A4", "A5")
T0 = 1_770_000_000
STEP = 5000  # seconds between starts; a 30 minute match lasts 1860 s, so none overlap


def _matches(first: int, last: int):
    out = []
    for i in range(first, last):
        match, players = make_match(i, start_time=T0 + i * STEP)
        for p in players:
            p["hero_id"] = 1 + (p["player_slot"] % 128) + i % 3
        out.append((match, players))
    return out


def _results(matches):
    return {m["match_id"]: int(m["radiant_win"]) for m, _ in matches}


@pytest.fixture(scope="module")
def world():
    matches = _matches(0, 60)
    return matches, _results(matches), load_items()


def _sha(tag):
    return hashlib.sha256((CANDIDATES / tag / "train.py").read_bytes()).hexdigest()


def test_state_round_trip_is_exact():
    rng = np.random.default_rng(0)
    state = {
        "w": rng.normal(size=(3, 4)),
        "f": rng.normal(size=5).astype("float32"),
        "elo": {7: 1500.25, 1_000_000_007: 0.1 + 0.2},
        "nested": {"k": [1, 2.5, None, "x", True]},
        "scalar": np.float64(1 / 3),
    }
    back = decode_state(json.loads(json.dumps(encode_state(state))))
    assert np.array_equal(back["w"], state["w"]) and back["w"].dtype == state["w"].dtype
    assert np.array_equal(back["f"], state["f"]) and back["f"].dtype == np.float32
    assert back["elo"] == state["elo"] and all(isinstance(k, int) for k in back["elo"])
    assert back["nested"]["k"] == [1, 2.5, None, "x", True]
    assert back["scalar"] == state["scalar"]


def test_state_codec_refuses_what_it_cannot_store():
    with pytest.raises(TypeError):
        encode_state({"a": object()})
    with pytest.raises(TypeError):
        encode_state({1: 1, "a": 2})


def test_train_file_with_a_different_hash_is_refused(tmp_path):
    copy = tmp_path / "train.py"
    copy.write_text((CANDIDATES / "A1" / "train.py").read_text() + "\n# edited\n")
    with pytest.raises(ValueError, match="differs from the registered"):
        load_train(copy, _sha("A1"))


@pytest.mark.parametrize("tag", TAGS)
def test_repo_copies_are_the_audited_files(tag):
    audited = {"A1": "7c737230", "A3": "7ada1523", "A4": "33d1c4b0", "A5": "1cf8e61d"}
    assert _sha(tag).startswith(audited[tag])


def test_history_round_trip(tmp_path, world):
    matches, results, _ = world
    history = ResultHistory.empty()
    history.add_week(matches[:20], results)
    history.save(tmp_path / "h.json.gz")
    back = ResultHistory.load(tmp_path / "h.json.gz")
    assert back.meta == history.meta and back.results == history.results
    assert len(back.meta) == 20 and len(back.results) == 20


def test_history_keeps_results_only_of_known_matches(world):
    matches, results, _ = world
    history = ResultHistory.empty()
    history.add_week(matches[:5], results)  # results of 60 matches offered, 5 are known
    assert len(history.results) == 5


def _ctx(table, ids):
    rows = np.flatnonzero(np.isin(table.rows["match_id"], list(ids)))
    return CandidateCtx(table, rows)


def check_results_boundaries(ctx_class):
    """Hand-built case: which results may reach a match that starts at 10_000."""

    def match(mid, start, duration):
        m, p = make_match(mid, minutes=10, start_time=start)
        m["duration"] = duration
        for q in p:
            q["hero_id"] = 1
        return m, p

    a = match(1, 1_000, 8_000)  # ends 9_000: before the start of the match
    b = match(2, 1_000, 9_000)  # ends exactly 10_000: not strictly before, must stay out
    c = match(3, 9_000, 4_000)  # starts before, ends after: must stay out
    d = match(4, 5_000, 1_000)  # ends 6_000 but has no result: skipped
    e = match(5, 10_000, 2_000)  # the match itself
    f = match(6, 20_000, 2_000)  # starts later; used to test the clock guard
    matches = [a, b, c, d, e, f]
    base = 9_000_000_000
    results = {base + 1: 1, base + 2: 0, base + 3: 1, base + 5: 1, base + 6: 0}
    table = CandidateTable.build(matches, results, load_items(), order=[base + 5, base + 6])
    ctx = ctx_class(table, np.flatnonzero(table.rows["match_id"] == base + 5))
    seen = [r["id"] - base for r in ctx.results_before(10_000)]
    assert seen == [1], seen
    with pytest.raises(PermissionError):
        ctx.results_before(10_001)  # later than the latest start of the call
    with pytest.raises(PermissionError):
        ctx.label(base + 5)  # labels only in fit
    with pytest.raises(PermissionError):
        ctx.raw(base + 5, 0)


def test_results_before_boundaries_hand_built():
    check_results_boundaries(CandidateCtx)


def test_planted_bug_in_results_before_is_caught():
    """A ctx that filters by start instead of end lets a running match's result through."""

    class StartInsteadOfEnd(CandidateCtx):
        def results_before(self, ts):
            t = self._table
            return [
                {"id": m, "start": t.meta[m]["start"], "end": t.meta[m]["end"],
                 **{k: t.meta[m][k] for k in ("rt", "dt", "ra", "da", "rh", "dh")},
                 "radiant_win": int(t.results[m])}
                for m in sorted(t.meta, key=lambda m: t.meta[m]["end"])
                if m in t.results and t.meta[m]["start"] < ts
            ]  # fmt: skip

    with pytest.raises(AssertionError):
        check_results_boundaries(StartInsteadOfEnd)


def test_history_plus_week_equals_one_full_table(world):
    matches, results, items = world
    full = CandidateTable.build(matches, results, items)
    history = ResultHistory.empty()
    history.add_week(matches[:40], results)
    week = matches[40:]
    ids = {m["match_id"] for m, _ in week}
    partial = week_table(history, week, ids, results, items)
    for i in (0, 5, 19):
        mid = week[i][0]["match_id"]
        ts = week[i][0]["start_time"]
        for table in (full, partial):
            table.start.setdefault(mid, ts)
        a = CandidateCtx(full, np.flatnonzero(full.rows["match_id"] == mid))
        b = CandidateCtx(partial, np.flatnonzero(partial.rows["match_id"] == mid))
        assert [r["id"] for r in a.results_before(ts)] == [r["id"] for r in b.results_before(ts)]
    for key in full.rows:
        mask = np.isin(full.rows["match_id"], list(ids))
        assert np.array_equal(full.rows[key][mask], partial.rows[key])


@pytest.fixture(scope="module")
def fitted(world, tmp_path_factory):
    matches, results, items = world
    table = CandidateTable.build(matches, results, items)
    train_ids = sorted(m["match_id"] for m, _ in matches[:40])
    folder = tmp_path_factory.mktemp("fitted")
    models = {}
    for tag in TAGS:
        spec = fit_candidate(CANDIDATES / tag / "train.py", _sha(tag), table, train_ids)
        (folder / tag).mkdir()
        (folder / tag / "train.py").write_bytes((CANDIDATES / tag / "train.py").read_bytes())
        (folder / tag / "model.json").write_text(json.dumps(spec))
        models[tag] = load_candidate(folder / tag / "model.json")
    return models


@pytest.mark.parametrize("tag", TAGS)
def test_week_batch_equals_one_match_at_a_time(world, fitted, tag):
    """Scoring a week in one batch gives each match what it gets when scored alone."""
    matches, results, items = world
    history = ResultHistory.empty()
    history.add_week(matches[:40], results)
    week = matches[40:]
    ids = {m["match_id"] for m, _ in week}
    table = week_table(history, week, ids, results, items)
    batch = fitted[tag].predict_rows(table, np.arange(len(table.rows["match_id"])))
    for match, _ in week[::4]:
        mid = match["match_id"]
        single = week_table(history, week, {mid}, results, items)
        own = fitted[tag].predict_rows(single, np.arange(len(single.rows["match_id"])))
        assert np.abs(batch[table.rows["match_id"] == mid] - own).max() < 1e-12


@pytest.mark.parametrize("tag", TAGS)
def test_a_result_that_ends_after_the_start_never_changes_the_prediction(world, fitted, tag):
    """Flip every result of matches that start at or after the target: its predictions stay."""
    matches, results, items = world
    history = ResultHistory.empty()
    history.add_week(matches[:40], results)
    week = matches[40:]
    target = week[5][0]
    ids = {target["match_id"]}
    base = week_table(history, week, ids, results, items)
    p0 = fitted[tag].predict_rows(base, np.arange(len(base.rows["match_id"])))
    later = {m["match_id"] for m, _ in week if m["start_time"] >= target["start_time"]}
    flipped = {m: (1 - r if m in later else r) for m, r in results.items()}
    again = week_table(history, week, ids, flipped, items)
    p1 = fitted[tag].predict_rows(again, np.arange(len(again.rows["match_id"])))
    assert np.array_equal(p0, p1)


@pytest.mark.parametrize("tag", TAGS)
def test_a_short_match_is_scored_row_by_row(world, fitted, tag):
    """A hand-built 12 minute match: one finite probability per minute row, no more."""
    matches, results, items = world
    history = ResultHistory.empty()
    history.add_week(matches[:40], results)
    short, players = make_match(500, minutes=12, start_time=T0 + 500 * STEP)
    for p in players:
        p["hero_id"] = 1 + (p["player_slot"] % 128)
    mid = short["match_id"]
    table = week_table(history, [(short, players)], {mid}, {mid: 1}, items)
    p = fitted[tag].predict_rows(table, np.arange(len(table.rows["match_id"])))
    assert len(p) == 12 and np.isfinite(p).all()
    assert list(table.rows["t"]) == list(range(12))
