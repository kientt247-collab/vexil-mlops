"""Turning a match into model inputs and predictions for every registered model."""

from __future__ import annotations

import numpy as np

from vexil.features.a1 import a1_table
from vexil.matchdata import Match, Player, is_usable, n_rows
from vexil.ratings import INPUT_NAME as RATING_INPUT
from vexil.registry import ModelEntry

JUDGED_TIERS = ("professional", "premium")
CAPTAINS_MODE = 2


def is_judged(match: Match) -> bool:
    """Only these matches enter metrics: usable, Captains Mode, judged league tier."""
    return (
        is_usable(match)
        and match.get("game_mode") == CAPTAINS_MODE
        and match.get("league_tier") in JUDGED_TIERS
    )


def match_inputs(match: Match, players: list[Player], rating_input: float) -> dict[str, np.ndarray]:
    """Every input any registered model kind may ask for, one value per minute row."""
    rows = n_rows(match)
    inputs: dict[str, np.ndarray] = {
        name: v.astype(float) for name, v in a1_table(match, players).items()
    }
    inputs["t"] = np.arange(rows, dtype=float)
    inputs["gold"] = np.asarray(match["radiant_gold_adv"][:rows], dtype=float)
    inputs[RATING_INPUT] = np.full(rows, rating_input)
    return inputs


def score_match(
    entries: tuple[ModelEntry, ...], inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Win probability of Radiant per minute row, for each model id.

    Per-match models only. Entries of kind ``frozen_train`` replay results and are scored per
    week by ``pipeline.process_week``.
    """
    if any(entry.kind == "frozen_train" for entry in entries):
        raise ValueError("frozen_train entries are scored per week, not per match")
    return {entry.id: entry.load().predict(inputs) for entry in entries}
