"""Pure helpers on one raw OpenDota match record (a dict) and its ten player records.

Time rule used everywhere: an input at game minute t may read array values at index t or
earlier and events whose time is at most 60 * t seconds. Nothing in this module reads the
match result.
"""

from __future__ import annotations

from typing import Any

Match = dict[str, Any]
Player = dict[str, Any]

MIN_DURATION_SECONDS = 600
# Per-player event lists (each event has a "time") whose name does not end in "_log".
EVENT_LISTS = ("neutral_item_history",)
PLAYER_SERIES = ("gold_t", "xp_t", "lh_t", "dn_t", "times")
MATCH_SERIES = ("radiant_gold_adv", "radiant_xp_adv")


def is_radiant(player: Player) -> bool:
    """Player slots below 128 belong to the Radiant side."""
    return player["player_slot"] < 128


def n_rows(match: Match) -> int:
    """Minute rows of a match: t = 0 .. L-2, where L is the length of the gold-lead array.

    The last element is dropped because it can fall after the end of the match.
    """
    return max(len(match.get("radiant_gold_adv") or []) - 1, 0)


def is_usable(match: Match) -> bool:
    """Parsed (has per-minute arrays) and at least ten minutes long."""
    has_arrays = bool(match.get("radiant_gold_adv"))
    return has_arrays and (match.get("duration") or 0) >= MIN_DURATION_SECONDS


def cut(match: Match, players: list[Player], t: int) -> tuple[Match, list[Player]]:
    """Copy of a match as it was known at minute t.

    Arrays are kept up to index t, events up to 60 * t seconds, and end-of-match fields
    are removed. The leak tests recompute inputs on this copy.
    """
    limit = 60 * t
    dropped = {"duration", "objectives", "teamfights", *MATCH_SERIES}
    cut_match = {k: v for k, v in match.items() if k not in dropped}
    for key in MATCH_SERIES:
        cut_match[key] = list(match[key][: t + 1])
    cut_match["objectives"] = [
        o for o in (match.get("objectives") or []) if o.get("time", 0) <= limit
    ]
    cut_players = []
    for player in players:
        kept = {
            k: v
            for k, v in player.items()
            if not k.endswith("_log") and not k.endswith("_t") and k not in ("times", *EVENT_LISTS)
        }
        for key in PLAYER_SERIES:
            kept[key] = list((player.get(key) or [])[: t + 1])
        for key in player:
            if key.endswith("_log") or key in EVENT_LISTS:
                kept[key] = [e for e in (player.get(key) or []) if e.get("time", 0) <= limit]
        cut_players.append(kept)
    return cut_match, cut_players
