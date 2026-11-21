"""Challengers and the promotion gate.

A challenger is a model that is scored next to the champion without being used. Two kinds:

* refit: the champion's recipe fitted on recent matches;
* recalibration: the champion with a slope and shift on its log-odds.

``gate`` decides whether a challenger may be *proposed* for promotion after its shadow
period. It never promotes: the registry accepts a new champion only with approval fields
that a person writes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vexil.evaluate import calibration

GATE_DEFAULTS = {"min_matches": 600, "min_gain": 0.002, "slope_low": 0.9, "slope_high": 1.1}
BOOTSTRAP_DRAWS = 2000


@dataclass
class GateResult:
    passed: bool
    matches: int
    gain: float
    interval: tuple[float, float]
    slope: float
    reasons: list[str]


def paired_gain_interval(
    champion_loss: np.ndarray, challenger_loss: np.ndarray, rows: np.ndarray, groups: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS, seed: int = 0,
) -> tuple[float, tuple[float, float]]:  # fmt: skip
    """Pooled gain of the challenger over the champion and its 95% interval.

    Inputs are per match: loss sums over minutes 0-29, the number of those rows, and the
    match group. Whole groups are resampled, because games of one series are not independent.
    """
    _, group_index = np.unique(groups, return_inverse=True)
    n_groups = group_index.max() + 1
    difference = np.bincount(
        group_index, weights=champion_loss - challenger_loss, minlength=n_groups
    )
    n = np.bincount(group_index, weights=rows, minlength=n_groups)
    gain = float(difference.sum() / n.sum())
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_groups, (draws, n_groups))
    resampled = difference[picks].sum(1) / n[picks].sum(1)
    low, high = np.quantile(resampled, [0.025, 0.975])
    return gain, (float(low), float(high))


def gate(
    champion_loss: np.ndarray, challenger_loss: np.ndarray, rows: np.ndarray, groups: np.ndarray,
    challenger_p: np.ndarray, y_rows: np.ndarray, settings: dict | None = None, seed: int = 0,
) -> GateResult:  # fmt: skip
    """Apply the promotion gate to one challenger after its shadow period.

    `challenger_p` and `y_rows` are the challenger's minute rows (minutes 0-29) and their
    outcomes, used for its calibration slope.
    """
    settings = {**GATE_DEFAULTS, **(settings or {})}
    gain, interval = paired_gain_interval(champion_loss, challenger_loss, rows, groups, seed=seed)
    slope = calibration(y_rows, challenger_p)[0]
    reasons = []
    if len(rows) < settings["min_matches"]:
        reasons.append(f"only {len(rows)} judged matches, {settings['min_matches']} needed")
    if gain < settings["min_gain"]:
        reasons.append(f"gain {gain:+.4f} is below {settings['min_gain']}")
    if interval[0] <= 0:
        reasons.append(f"95% interval {interval[0]:+.4f} to {interval[1]:+.4f} includes 0")
    if not settings["slope_low"] <= slope <= settings["slope_high"]:
        reasons.append(
            f"slope {slope:.3f} is outside {settings['slope_low']} to {settings['slope_high']}"
        )
    return GateResult(not reasons, len(rows), gain, interval, slope, reasons)
