"""Shape checks for rows that come from OpenDota.

The checks are about structure, not about values: a row that passes can be read by the
input code without a type error. A match that fails is rejected and counted, so that a
change on OpenDota's side shows up as a number on the data-health page instead of a crash
deep inside the pipeline.
"""

from __future__ import annotations

from typing import Any

REQUIRED_MATCH_KEYS = ("match_id", "start_time", "leagueid", "game_mode", "radiant_win")
REQUIRED_PLAYER_KEYS = ("match_id", "player_slot", "account_id")
RADIANT_SLOTS, DIRE_SLOTS = {0, 1, 2, 3, 4}, {128, 129, 130, 131, 132}


def _is_number_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, int | float) for x in value)


def match_problems(match: dict, players: list[dict]) -> list[str]:
    """Reasons why a match cannot be used at all. An empty list means it is accepted."""
    problems = [f"match key missing: {key}" for key in REQUIRED_MATCH_KEYS if key not in match]
    if problems:
        return problems
    if not isinstance(match["match_id"], int) or not isinstance(match["start_time"], int):
        problems.append("match_id and start_time must be integers")
    if match["radiant_win"] not in (True, False, None):
        problems.append("radiant_win must be true, false or null")
    duration = match.get("duration")
    if duration is not None and not isinstance(duration, int):
        problems.append("duration must be an integer or null")
    gold, xp = match.get("radiant_gold_adv"), match.get("radiant_xp_adv")
    if gold is not None:
        if not _is_number_list(gold) or not _is_number_list(xp):
            problems.append("gold and xp lead must be lists of numbers")
        elif len(gold) != len(xp):
            problems.append("gold and xp lead differ in length")
    objectives = match.get("objectives")
    if objectives is not None and not isinstance(objectives, list):
        problems.append("objectives must be a list or null")
    for player in players:
        missing = [key for key in REQUIRED_PLAYER_KEYS if key not in player]
        if missing:
            problems.append(f"player key missing: {missing[0]}")
            break
    slots = {p.get("player_slot") for p in players}
    if gold and slots != RADIANT_SLOTS | DIRE_SLOTS:
        problems.append("a parsed match needs the ten standard player slots")
    if gold and not problems:
        for player in players:
            if not _is_number_list(player.get("gold_t")) or len(player["gold_t"]) < len(gold) - 1:
                problems.append("player gold_t is missing or shorter than the match arrays")
                break
    return problems
