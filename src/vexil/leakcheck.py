"""Tools that prove an input does not read the future.

``poison`` keeps what was known at minute t and replaces everything later with random
values, adding random later events as well. An input function that gives the same answer
on the real match, on the cut copy and on the poisoned copy cannot depend on data after t.
"""

from __future__ import annotations

import random

from vexil.matchdata import Match, Player, cut

_TOWERS = ("npc_dota_goodguys_tower%d_mid", "npc_dota_badguys_tower%d_top")
_RAX = ("npc_dota_goodguys_melee_rax_mid", "npc_dota_badguys_range_rax_top")


def poison(
    match: Match, players: list[Player], t: int, rng: random.Random
) -> tuple[Match, list[Player]]:
    """Copy of a match that agrees with the original up to minute t and is random after it."""
    limit = 60 * t

    def later() -> int:
        return limit + rng.randint(1, 3000)

    poisoned, poisoned_players = cut(match, players, t)
    extra = len(match["radiant_gold_adv"]) - (t + 1) + rng.randint(0, 5)
    for key in ("radiant_gold_adv", "radiant_xp_adv"):
        poisoned[key] += [rng.randint(-60000, 60000) for _ in range(extra)]
    poisoned["duration"] = rng.randint(600, 6000)
    poisoned["objectives"] += [
        {"type": "building_kill", "key": rng.choice(_TOWERS) % rng.randint(1, 4), "time": later()}
        for _ in range(rng.randint(3, 12))
    ]
    poisoned["objectives"] += [
        {"type": "CHAT_MESSAGE_ROSHAN_KILL", "team": rng.choice([2, 3]), "time": later()},
        {"type": "CHAT_MESSAGE_AEGIS", "player_slot": rng.choice([0, 128]), "time": later()},
        {"type": "building_kill", "key": rng.choice(_RAX), "time": later()},
        # one event inside the next minute, so that code reading minute t + 1 must change
        {"type": "CHAT_MESSAGE_ROSHAN_KILL", "team": rng.choice([2, 3]), "time": limit + 30},
    ]
    for player in poisoned_players:
        player["kills_log"] = list(player.get("kills_log") or []) + [
            {"key": "npc_dota_hero_x", "time": later()} for _ in range(rng.randint(0, 6))
        ]
        for key in ("gold_t", "xp_t", "lh_t"):
            player[key] = list(player.get(key) or []) + [
                rng.randint(0, 90000) for _ in range(extra)
            ]
    return poisoned, poisoned_players
