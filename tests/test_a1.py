import math
import random

import pytest

from vexil.features.a1 import A1_INPUTS, a1_at_minute, a1_table
from vexil.leakcheck import poison
from vexil.matchdata import cut, n_rows

from .synthetic import make_match

CHECKS = [(seed, t) for seed in range(40) for t in random.Random(seed).sample(range(30), 3)]


def close(got: dict, want: dict) -> bool:
    return all(abs(got[k] - want[k]) <= 1e-3 * max(1.0, abs(want[k])) for k in A1_INPUTS)


def table_row(match, players, t):
    return {name: float(values[t]) for name, values in a1_table(match, players).items()}


def hand_built_match():
    """Twelve minute rows, small enough to compute every input by hand."""
    match = {
        "match_id": 1,
        "duration": 780,
        "radiant_gold_adv": [100 * i for i in range(13)],
        "radiant_xp_adv": [50 * i for i in range(13)],
        "objectives": [
            {"type": "building_kill", "key": "npc_dota_badguys_tower1_mid", "time": 180},
            {"type": "CHAT_MESSAGE_ROSHAN_KILL", "team": 2, "time": 300},
            {"type": "CHAT_MESSAGE_AEGIS", "player_slot": 1, "time": 305},
            {"type": "building_kill", "key": "npc_dota_goodguys_tower1_top", "time": 420},
            {"type": "CHAT_MESSAGE_AEGIS_STOLEN", "player_slot": 128, "time": 480},
            {"type": "building_kill", "key": "npc_dota_badguys_melee_rax_mid", "time": 600},
            {"type": "building_kill", "key": "npc_dota_badguys_range_rax_mid", "time": 610},
            {"type": "building_kill", "key": "npc_dota_goodguys_melee_rax_top", "time": 661},
            {"type": "building_kill", "key": "npc_dota_badguys_fort", "time": 700},
        ],
    }

    def player(slot, gold_per_minute, kill_times):
        return {
            "player_slot": slot,
            "gold_t": [gold_per_minute * i for i in range(13)],
            "kills_log": [{"time": when} for when in kill_times],
        }

    players = [
        player(0, 200, [100, 170, 400]),
        player(1, 100, []),
        player(128, 150, [200, 650]),
        player(129, 50, []),
    ]
    return match, players


# Expected values worked out by hand from the events above.
HAND_VALUES = {
    3: dict(kill_diff=2, towers_diff=1, xp_adv_pm=37.5, gold_adv_dec=300 * math.exp(-0.15),
            aegis_team=0, gold_adv_d5=300, gold_rank1_diff=150, rax_lanes_diff=0, roshan_diff=0,
            kill_diff_last3=2),
    7: dict(kill_diff=2, towers_diff=0, xp_adv_pm=43.75, gold_adv_dec=700 * math.exp(-0.35),
            aegis_team=1, gold_adv_d5=500, gold_rank1_diff=350, rax_lanes_diff=0, roshan_diff=1,
            kill_diff_last3=1),
    11: dict(kill_diff=1, towers_diff=0, xp_adv_pm=550 / 12, gold_adv_dec=1100 * math.exp(-0.55),
             aegis_team=-1, gold_adv_d5=500, gold_rank1_diff=550, rax_lanes_diff=1, roshan_diff=1,
             kill_diff_last3=-1),
}  # fmt: skip


@pytest.mark.parametrize("t", [3, 7, 11])
def test_hand_built_match(t):
    match, players = hand_built_match()
    assert n_rows(match) == 12
    assert a1_at_minute(match, players, t) == pytest.approx(HAND_VALUES[t])
    assert table_row(match, players, t) == pytest.approx(HAND_VALUES[t], rel=1e-6)


@pytest.mark.parametrize("seed", range(40))
def test_table_equals_reference_at_every_minute(seed):
    match, players = make_match(seed)
    table = a1_table(match, players)
    for t in range(n_rows(match)):
        assert close({k: float(v[t]) for k, v in table.items()}, a1_at_minute(match, players, t))


@pytest.mark.parametrize("seed,t", CHECKS)
def test_cut_test(seed, t):
    """Inputs recomputed on data cut at minute t equal the table values."""
    match, players = make_match(seed)
    assert close(a1_at_minute(*cut(match, players, t), t), table_row(match, players, t))


@pytest.mark.parametrize("seed,t", CHECKS)
def test_poison_test(seed, t):
    """Replacing everything after minute t with random values leaves the inputs unchanged."""
    match, players = make_match(seed)
    poisoned = poison(match, players, t, random.Random(1000 * seed + t))
    assert close(a1_at_minute(*poisoned, t), table_row(match, players, t))


def test_planted_bug_is_caught():
    """A copy of the code that reads minute t + 1 must fail both leak tests."""

    def leaky(match, players, t):
        return a1_at_minute(match, players, t + 1)

    caught_by_cut = caught_by_poison = 0
    for seed, t in CHECKS:
        match, players = make_match(seed)
        want = table_row(match, players, t)
        try:
            caught_by_cut += not close(leaky(*cut(match, players, t), t), want)
        except IndexError:
            caught_by_cut += 1
        poisoned = poison(match, players, t, random.Random(1000 * seed + t))
        caught_by_poison += not close(leaky(*poisoned, t), want)
    assert caught_by_cut == len(CHECKS)
    assert caught_by_poison == len(CHECKS)
