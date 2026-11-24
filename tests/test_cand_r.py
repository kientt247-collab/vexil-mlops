"""cand-R: the champion's design with a warm-started rating (frozen table of initial ratings).

Hand-built cases (a match that straddles the cut, a 12-minute match with hand-set ratings), planted
defects that a test must catch, cut and poison tests, and resume equals one full pass.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from vexil.candidates import CandidateCtx, CandidateTable, load_candidate, load_train
from vexil.features.extra import load_items

from .synthetic import make_match

REPO = Path(__file__).resolve().parents[1]
FOLDER = REPO / "models" / "candidates" / "R"
TRAIN = FOLDER / "train.py"
RAD, DIRE = tuple(range(1, 6)), tuple(range(6, 11))
SCALE = math.log(10) / 400


@pytest.fixture()
def r_module():
    return load_train(TRAIN, hashlib.sha256(TRAIN.read_bytes()).hexdigest())


def _result(mid, start, end, win, ra=RAD, da=DIRE):
    return {"id": mid, "start": start, "end": end, "ra": list(ra), "da": list(da),
            "radiant_win": win}  # fmt: skip


def _table(radiant=1532.0, dire=1468.0, games=1):
    rating = {a: radiant for a in RAD} | {a: dire for a in DIRE}
    return rating, {a: games for a in rating}


def _hand_f_after_straddle():
    """Table 1532 / 1468; then a straddling match ends (K 64, Radiant wins)."""
    expected = 1 / (1 + 10 ** (-64 / 400))
    delta = 64 * (1 - expected)
    return (64 + 2 * delta) * SCALE


def test_repo_files_are_the_registered_ones(r_module):
    table = FOLDER / r_module.TABLE_FILE
    assert hashlib.sha256(table.read_bytes()).hexdigest() == r_module.TABLE_SHA256
    spec = json.loads((FOLDER / "model.json").read_text())
    assert spec["train_sha256"] == hashlib.sha256(TRAIN.read_bytes()).hexdigest()
    assert load_candidate(FOLDER / "model.json").state["coef"].shape == (13,)
    with open(table, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {int(r["state_time_utc"]) for r in rows} == {r_module.CUT}
    assert all(int(r["n_rated"]) >= 1 for r in rows)


def test_straddling_match_is_applied_once_at_its_end(r_module):
    cut = r_module.CUT
    results = [
        _result(1, cut - 3000, cut - 1000, 1),  # already inside the table: must be ignored
        _result(2, cut - 1000, cut + 1000, 1),  # started before the cut, ended after it
    ]
    f = r_module.rating_input(results, [(3, cut + 2000, RAD, DIRE)], table=_table())
    assert f[3] == pytest.approx(_hand_f_after_straddle(), abs=1e-12)
    assert f[3] == pytest.approx(0.6697202504267878, abs=1e-12)


def test_planted_missing_cut_filter_is_caught(r_module, monkeypatch):
    cut = r_module.CUT
    results = [_result(1, cut - 3000, cut - 1000, 1), _result(2, cut - 1000, cut + 1000, 1)]
    queries = [(3, cut + 2000, RAD, DIRE)]
    right = r_module.rating_input(results, queries, table=_table())[3]
    monkeypatch.setattr(r_module, "CUT", 0)  # the planted defect: replay everything again
    wrong = r_module.rating_input(results, queries, table=_table())[3]
    assert abs(wrong - right) > 1e-3


def test_a_result_ending_at_the_start_is_not_yet_known(r_module):
    cut = r_module.CUT
    start = cut + 5000
    table_only = (1532.0 - 1468.0) * SCALE
    same_time = [_result(1, start - 1800, start, 1)]
    one_second_before = [_result(1, start - 1800, start - 1, 1)]
    assert r_module.rating_input(same_time, [(9, start, RAD, DIRE)], table=_table())[9] == (
        pytest.approx(table_only, abs=1e-15)
    )
    later = r_module.rating_input(one_second_before, [(9, start, RAD, DIRE)], table=_table())[9]
    assert later > table_only + 0.1


def _world(seed=3, n=60):
    """Random results over 30 accounts; match i starts 5000 s after match i-1."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        accounts = rng.choice(np.arange(1, 31), size=10, replace=False).tolist()
        start = 1_770_000_000 + i * 5000
        end = start + 1800 + int(rng.integers(0, 3000))
        win = int(rng.integers(0, 2))
        out.append(_result(i, start, end, win, accounts[:5], accounts[5:]))
    return out


def _reference_state(results, split):
    """Independent loop: Elo state after every result that ended before `split`."""
    rating, games = {}, {}
    for r in sorted((r for r in results if r["end"] < split), key=lambda r: (r["end"], r["id"])):
        a = sum(rating.get(x, 1500.0) for x in r["ra"]) / 5
        b = sum(rating.get(x, 1500.0) for x in r["da"]) / 5
        surprise = r["radiant_win"] - 1 / (1 + 10 ** ((b - a) / 400))
        for sign, side in ((1, r["ra"]), (-1, r["da"])):
            for x in side:
                k = 64 if games.get(x, 0) < 20 else 32
                rating[x] = rating.get(x, 1500.0) + sign * k * surprise
                games[x] = games.get(x, 0) + 1
    return rating, games


def test_resume_equals_one_full_pass(r_module, monkeypatch):
    results = sorted(_world(), key=lambda r: (r["end"], r["id"]))
    split = 1_770_000_000 + 30 * 5000 + 17
    queries = [(r["id"], r["start"], r["ra"], r["da"]) for r in results if r["start"] >= split]
    assert len(queries) > 10
    monkeypatch.setattr(r_module, "CUT", split)
    resumed = r_module.rating_input(results, queries, table=_reference_state(results, split))
    monkeypatch.setattr(r_module, "CUT", 0)
    full = r_module.rating_input(results, queries, table=({}, {}))
    assert max(abs(resumed[q[0]] - full[q[0]]) for q in queries) < 1e-12
    leaky = _reference_state(results, split + 20 * 5000)  # planted: 2026 results inside the table
    monkeypatch.setattr(r_module, "CUT", split)
    wrong = r_module.rating_input(results, queries, table=leaky)
    assert max(abs(wrong[q[0]] - full[q[0]]) for q in queries) > 1e-3


def test_cut_and_poison_leave_the_input_unchanged(r_module, monkeypatch):
    results = sorted(_world(seed=5), key=lambda r: (r["end"], r["id"]))
    split = 1_770_000_000 + 10 * 5000
    monkeypatch.setattr(r_module, "CUT", split)
    state = _reference_state(results, split)
    target = results[40]
    query = [(target["id"], target["start"], target["ra"], target["da"])]
    base = r_module.rating_input(results, query, table=state)[target["id"]]
    cut_copy = [r for r in results if r["end"] < target["start"]]
    assert r_module.rating_input(cut_copy, query, table=state)[target["id"]] == base
    rng = np.random.default_rng(1)
    poisoned = [
        dict(r, radiant_win=int(rng.integers(0, 2)), ra=list(rng.permutation(31)[:5] + 100))
        if r["end"] >= target["start"]
        else r
        for r in results
    ]
    assert r_module.rating_input(poisoned, query, table=state)[target["id"]] == base


def test_twelve_minute_match_through_the_whole_path(r_module, monkeypatch):
    """Ratings set by hand: Radiant 1532, Dire 1468, so f = 64 ln(10) / 400 on every row."""
    match, players = make_match(1, minutes=12, start_time=r_module.CUT + 7200)
    table = CandidateTable.build([(match, players)], {}, load_items())
    assert len(table.rows["t"]) == 12
    radiant = [p["account_id"] for p in players if p["player_slot"] < 128]
    dire = [p["account_id"] for p in players if p["player_slot"] >= 128]
    rating = {a: 1532.0 for a in radiant} | {a: 1468.0 for a in dire}
    monkeypatch.setattr(r_module, "load_table", lambda: (rating, {a: 1 for a in rating}))
    coef = np.zeros(13)
    coef[11] = 1.0  # design columns: 100 u, ten A1 inputs, f, f x u
    state = {"mu_u": 0.0, "sd_u": 1.0, "mu": np.zeros(12), "sd": np.ones(12), "coef": coef,
             "intercept": 0.5}  # fmt: skip
    ctx = CandidateCtx(table, np.arange(12))
    p = r_module.predict(state, ctx, ctx.rows)
    f = 64 * SCALE
    assert p.shape == (12,)
    assert np.allclose(p, 1 / (1 + math.exp(-(0.5 + f))), atol=1e-15)
