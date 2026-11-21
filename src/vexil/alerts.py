"""Alarm statistics. The same functions run in the weekly job and in the simulations that
set their limits, so a limit always belongs to exactly the statistic it was measured on.

All functions work on arrays whose last axis is the week, and broadcast over any leading
axes (the simulations pass thousands of simulated years at once).

Vocabulary: a rule is *firing* in a week when its statistic is past its limit. An *alarm*
is the week a rule starts firing after a quiet week; one alarm opens one issue.
"""

from __future__ import annotations

import numpy as np

WINDOW_WEEKS = 8


def rolling_sum(values: np.ndarray, window: int = WINDOW_WEEKS) -> np.ndarray:
    """Sum over the last `window` weeks; NaN until that many weeks exist."""
    values = np.asarray(values, dtype=float)
    total = np.cumsum(values, axis=-1)
    out = np.full(values.shape, np.nan)
    out[..., window - 1] = total[..., window - 1]
    out[..., window:] = total[..., window:] - total[..., :-window]
    return out


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise ratio that is NaN where the denominator is zero or missing."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def one_step_slope(cal_u: np.ndarray, cal_i: np.ndarray) -> np.ndarray:
    """Calibration slope after one Newton step from the ideal slope of 1.

    With x = logit(p): slope is about 1 + sum(x * (y - p)) / sum(x^2 * p * (1 - p)).
    Both sums add up over matches and weeks, which a fitted slope does not.
    """
    return 1.0 + ratio(cal_u, cal_i)


def below(statistic: np.ndarray, reference: float, limit: float) -> np.ndarray:
    """Firing when the statistic is more than `limit` below its reference. NaN never fires."""
    with np.errstate(invalid="ignore"):
        return statistic < reference - limit


def outside(statistic: np.ndarray, reference: float, limit: float) -> np.ndarray:
    """Firing when the statistic is more than `limit` away from its reference."""
    with np.errstate(invalid="ignore"):
        return np.abs(statistic - reference) > limit


def cusum(shortfall: np.ndarray, slack: float, limit: float) -> np.ndarray:
    """One-sided CUSUM. `shortfall` is the standardised amount by which a week falls short
    of its reference (positive is bad). The sum grows by shortfall minus slack, never drops
    below zero, fires when it passes `limit`, and restarts from zero after firing.
    Weeks with a missing value leave the sum unchanged.
    """
    shortfall = np.asarray(shortfall, dtype=float)
    total = np.zeros(shortfall.shape[:-1])
    firing = np.zeros(shortfall.shape, dtype=bool)
    for week in range(shortfall.shape[-1]):
        step = np.nan_to_num(shortfall[..., week] - slack, nan=0.0)
        step = np.where(np.isnan(shortfall[..., week]), 0.0, step)
        total = np.maximum(0.0, total + step)
        firing[..., week] = total > limit
        total = np.where(firing[..., week], 0.0, total)
    return firing


def alarms(firing: np.ndarray) -> np.ndarray:
    """True in the weeks a rule starts firing (the week before was quiet)."""
    firing = np.asarray(firing, dtype=bool)
    previous = np.concatenate(
        [np.zeros((*firing.shape[:-1], 1), dtype=bool), firing[..., :-1]], axis=-1
    )
    return firing & ~previous


def standardised_shortfall(
    gain: np.ndarray, rows: np.ndarray, reference: float, weekly_sd: float, typical_rows: float
) -> np.ndarray:
    """(reference - gain) in units of the weekly noise, which shrinks with more rows."""
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = weekly_sd * np.sqrt(typical_rows / np.where(rows > 0, rows, np.nan))
        return (reference - gain) / sd
