"""Synthetic matches for tests. No real match data lives in this repository's tests."""

from __future__ import annotations

import random

TOWER_KEYS = [
    f"npc_dota_{side}_tower{tier}_{lane}"
    for side in ("goodguys", "badguys")
    for tier in (1, 2, 3)
    for lane in ("top", "mid", "bot")
]
RAX_KEYS = [
    f"npc_dota_{side}_{kind}_{lane}"
    for side in ("goodguys", "badguys")
    for kind in ("melee_rax", "range_rax")
    for lane in ("top", "mid", "bot")
]


def make_match(
    seed: int, minutes: int = 30, start_time: int = 1_770_000_000
) -> tuple[dict, list[dict]]:
    """A random but well-formed match with `minutes` minute rows and ten players."""
    rng = random.Random(seed)
    length = minutes + 1
    gold, xp = [0], [0]
    for _ in range(length - 1):
        gold.append(gold[-1] + rng.randint(-900, 900))
        xp.append(xp[-1] + rng.randint(-700, 700))
    end = 60 * length
    players = []
    for slot in [0, 1, 2, 3, 4, 128, 129, 130, 131, 132]:
        gold_t, lh_t = [600], [0]
        for _ in range(length - 1):
            gold_t.append(gold_t[-1] + rng.randint(150, 700))
            lh_t.append(lh_t[-1] + rng.randint(0, 12))
        kills = sorted(rng.randint(30, end) for _ in range(rng.randint(0, 9)))
        players.append(
            {
                "match_id": 9_000_000_000 + seed,
                "player_slot": slot,
                "account_id": 1000 * (seed % 7) + slot,
                "gold_t": gold_t,
                "xp_t": list(gold_t),
                "lh_t": lh_t,
                "kills_log": [{"key": "npc_dota_hero_x", "time": when} for when in kills],
            }
        )
    objectives = [
        {"type": "building_kill", "key": key, "time": rng.randint(300, end)}
        for key in rng.sample(TOWER_KEYS, 8) + rng.sample(RAX_KEYS, 4)
    ]
    for _ in range(rng.randint(1, 3)):
        when = rng.randint(600, end)
        team = rng.choice([2, 3])
        objectives.append({"type": "CHAT_MESSAGE_ROSHAN_KILL", "team": team, "time": when})
        slot = rng.choice([0, 3] if team == 2 else [128, 131])
        objectives.append({"type": "CHAT_MESSAGE_AEGIS", "player_slot": slot, "time": when + 5})
    if rng.random() < 0.5:
        stolen = {"type": "CHAT_MESSAGE_AEGIS_STOLEN", "player_slot": 130, "time": end - 200}
        objectives.append(stolen)
    objectives.append({"type": "building_kill", "key": "npc_dota_badguys_fort", "time": end})
    objectives.sort(key=lambda o: o["time"])
    match = {
        "match_id": 9_000_000_000 + seed,
        "start_time": start_time,
        "leagueid": 17,
        "game_mode": 2,
        "radiant_win": bool(rng.getrandbits(1)),
        "duration": end,
        "radiant_gold_adv": gold,
        "radiant_xp_adv": xp,
        "objectives": objectives,
    }
    return match, players
