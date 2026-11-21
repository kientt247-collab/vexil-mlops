"""The ten in-game inputs of the frozen models (named "A1" in phase 1 and phase 2).

Two implementations of the same definition:

* ``a1_at_minute`` computes one minute at a time and is written for clarity. The leak
  tests run it on cut and poisoned copies of a match.
* ``a1_table`` computes all minutes of a match at once and is what the pipeline uses.

Stored inputs are 32-bit floats. The frozen models were fitted on 32-bit values, so the
cast is part of the model contract and must not be removed.
"""

from __future__ import annotations

import math
import re

import numpy as np

from vexil.matchdata import Match, Player, is_radiant, n_rows

A1_INPUTS = (
    "kill_diff",
    "towers_diff",
    "xp_adv_pm",
    "gold_adv_dec",
    "aegis_team",
    "gold_adv_d5",
    "gold_rank1_diff",
    "rax_lanes_diff",
    "roshan_diff",
    "kill_diff_last3",
)

_TOWER = re.compile(r"^npc_dota_(goodguys|badguys)_tower[1-4](_top|_mid|_bot)?$")
_RAX = re.compile(r"^npc_dota_(goodguys|badguys)_(melee_rax|range_rax)_(top|mid|bot)$")
_AEGIS = ("CHAT_MESSAGE_AEGIS", "CHAT_MESSAGE_AEGIS_STOLEN")
_ROSHAN = "CHAT_MESSAGE_ROSHAN_KILL"
_RADIANT_TEAM, _DIRE_TEAM = 2, 3


def _objectives(match: Match) -> list[dict]:
    """Objectives without the destruction of the ancient, which ends the match."""
    return [
        o for o in (match.get("objectives") or []) if not str(o.get("key", "")).endswith("_fort")
    ]


def a1_at_minute(match: Match, players: list[Player], t: int) -> dict[str, float]:
    """The ten inputs at minute t (reference implementation)."""
    now = 60 * t
    gold, xp = match["radiant_gold_adv"], match["radiant_xp_adv"]

    def sign(player: Player) -> int:
        return 1 if is_radiant(player) else -1

    kills = [(sign(p), e["time"]) for p in players for e in (p.get("kills_log") or [])]
    kill_diff = sum(s for s, when in kills if when <= now)
    kill_diff_last3 = sum(s for s, when in kills if 60 * (t - 3) < when <= now)

    objectives = _objectives(match)
    towers, radiant_lanes_lost, dire_lanes_lost = 0, set(), set()
    for o in objectives:
        if o.get("type") != "building_kill" or o["time"] > now:
            continue
        key = o["key"]
        if _TOWER.match(key):
            towers += 1 if "_badguys_" in key else -1
        rax = _RAX.match(key)
        if rax:
            (dire_lanes_lost if rax.group(1) == "badguys" else radiant_lanes_lost).add(rax.group(3))

    roshan = sum(
        1 if o.get("team") == _RADIANT_TEAM else -1 if o.get("team") == _DIRE_TEAM else 0
        for o in objectives
        if o.get("type") == _ROSHAN and o["time"] <= now
    )
    aegis_events = [
        (o["time"], i, o)
        for i, o in enumerate(objectives)
        if o.get("type") in _AEGIS and o["time"] <= now
    ]
    aegis = 0
    if aegis_events:
        last = max(aegis_events, key=lambda item: (item[0], item[1]))[2]
        aegis = 1 if last["player_slot"] < 128 else -1

    richest_radiant = max(p["gold_t"][t] for p in players if is_radiant(p))
    richest_dire = max(p["gold_t"][t] for p in players if not is_radiant(p))
    return {
        "kill_diff": kill_diff,
        "towers_diff": towers,
        "xp_adv_pm": xp[t] / (t + 1),
        "gold_adv_dec": gold[t] * math.exp(-t / 20),
        "aegis_team": aegis,
        "gold_adv_d5": gold[t] - gold[max(t - 5, 0)],
        "gold_rank1_diff": richest_radiant - richest_dire,
        "rax_lanes_diff": len(dire_lanes_lost) - len(radiant_lanes_lost),
        "roshan_diff": roshan,
        "kill_diff_last3": kill_diff_last3,
    }


def _count_until(times: list[float], limits: np.ndarray) -> np.ndarray:
    """For each limit, how many of the times are at or before it."""
    return np.searchsorted(np.sort(np.asarray(times, dtype=float)), limits, side="right")


def a1_table(match: Match, players: list[Player]) -> dict[str, np.ndarray]:
    """The ten inputs for every minute row of a match, as 32-bit float arrays."""
    n = n_rows(match)
    t = np.arange(n)
    now = 60.0 * t
    gold = np.asarray(match["radiant_gold_adv"][:n], dtype=float)
    xp = np.asarray(match["radiant_xp_adv"][:n], dtype=float)

    radiant_kills = [e["time"] for p in players if is_radiant(p) for e in p.get("kills_log") or []]
    dire_kills = [e["time"] for p in players if not is_radiant(p) for e in p.get("kills_log") or []]
    kill_diff = _count_until(radiant_kills, now) - _count_until(dire_kills, now)
    kill_diff_3_ago = _count_until(radiant_kills, now - 180) - _count_until(dire_kills, now - 180)

    objectives = _objectives(match)
    buildings = [o for o in objectives if o.get("type") == "building_kill"]
    dire_towers = [
        o["time"] for o in buildings if _TOWER.match(o["key"]) and "_badguys_" in o["key"]
    ]
    radiant_towers = [
        o["time"] for o in buildings if _TOWER.match(o["key"]) and "_badguys_" not in o["key"]
    ]
    towers = _count_until(dire_towers, now) - _count_until(radiant_towers, now)

    first_rax_loss: dict[tuple[str, str], float] = {}
    for o in buildings:
        rax = _RAX.match(o["key"])
        if rax:
            lane = (rax.group(1), rax.group(3))
            first_rax_loss[lane] = min(first_rax_loss.get(lane, float("inf")), o["time"])
    dire_lanes = [when for (side, _), when in first_rax_loss.items() if side == "badguys"]
    radiant_lanes = [when for (side, _), when in first_rax_loss.items() if side == "goodguys"]
    rax_lanes = _count_until(dire_lanes, now) - _count_until(radiant_lanes, now)

    def roshan_times(team: int) -> list[float]:
        return [o["time"] for o in objectives if o.get("type") == _ROSHAN and o.get("team") == team]

    roshan = _count_until(roshan_times(_RADIANT_TEAM), now) - _count_until(
        roshan_times(_DIRE_TEAM), now
    )

    aegis_events = sorted(
        (o["time"], i, 1 if o["player_slot"] < 128 else -1)
        for i, o in enumerate(objectives)
        if o.get("type") in _AEGIS
    )
    aegis = np.zeros(n)
    if aegis_events:
        times = np.array([event[0] for event in aegis_events], dtype=float)
        holders = np.array([event[2] for event in aegis_events])
        seen = np.searchsorted(times, now, side="right")
        aegis = np.where(seen > 0, holders[np.maximum(seen - 1, 0)], 0)

    radiant_gold = np.array([p["gold_t"][:n] for p in players if is_radiant(p)], dtype=float)
    dire_gold = np.array([p["gold_t"][:n] for p in players if not is_radiant(p)], dtype=float)

    columns = {
        "kill_diff": kill_diff,
        "towers_diff": towers,
        "xp_adv_pm": xp / (t + 1),
        "gold_adv_dec": gold * np.exp(-t / 20),
        "aegis_team": aegis,
        "gold_adv_d5": gold - gold[np.maximum(t - 5, 0)],
        "gold_rank1_diff": radiant_gold.max(0) - dire_gold.max(0),
        "rax_lanes_diff": rax_lanes,
        "roshan_diff": roshan,
        "kill_diff_last3": kill_diff - kill_diff_3_ago,
    }
    return {name: np.asarray(columns[name], dtype=np.float32) for name in A1_INPUTS}
