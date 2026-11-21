"""Applying the frozen rules to the weekly tables.

``evaluate_rules`` takes the scores, the match table and the data-health counts of all
final weeks and returns one row per week, alarm family and subject, saying whether the
rule is firing and whether an alarm starts that week. Four families exist:

* performance: CUSUM on the weekly gain of a model over the anchor (needs results)
* calibration: rolling 8-week one-step slope and mean of p minus outcome (needs results)
* inputs: weekly mean of four economy totals, as a percent of its development mean
* data: fixed checks on the pull (share of judged-tier matches that are usable)

The limits come from ``monitoring/rules.yaml``, which node M6 of the action map set by
simulation on the development region. This module never changes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from vexil import alerts

USABLE_SHARE_FLOOR = 0.90  # fixed check; development weeks never fell below 0.98
STATE_SCHEMA = {
    "week": pl.String, "family": pl.String, "subject": pl.String, "statistic": pl.String,
    "value": pl.Float64, "reference": pl.Float64, "limit": pl.Float64, "firing": pl.Boolean,
    "alarm": pl.Boolean, "note": pl.String,
}  # fmt: skip


@dataclass
class MonitorResult:
    states: pl.DataFrame  # one row per week, family and subject
    weekly: pl.DataFrame  # weekly sums per model, the base of every label-based statistic
    events: pl.DataFrame  # first week each patch id appears among judged matches


def weekly_sums(scores: pl.DataFrame, anchor_id: str) -> pl.DataFrame:
    """Per week and model: rows, loss sums and calibration sums, plus the anchor's loss."""
    anchor = scores.filter(pl.col("model_id") == anchor_id).select(
        "match_id", pl.col("loss").alias("anchor_loss")
    )
    return (
        scores.join(anchor, on="match_id", how="left")
        .group_by(["week", "model_id"], maintain_order=True)
        .agg(
            pl.col("match_id").n_unique().alias("matches"),
            pl.col("n").sum().alias("rows"),
            pl.col("loss").sum(),
            pl.col("anchor_loss").sum(),
            pl.col("cal_u").sum(),
            pl.col("cal_i").sum(),
            pl.col("bias_sum").sum(),
        )
        .sort(["model_id", "week"])
    )


def _series(weekly: pl.DataFrame, model_id: str, weeks: list[str], column: str) -> np.ndarray:
    """Values of one column for one model, aligned to `weeks` (NaN where the week is absent)."""
    part = weekly.filter(pl.col("model_id") == model_id)
    lookup = dict(zip(part["week"], part[column], strict=True))
    return np.array([lookup.get(week, np.nan) for week in weeks], dtype=float)


def _rows(weeks, family, subject, statistic, value, reference, limit, firing, notes=None):
    started = alerts.alarms(np.asarray(firing))
    notes = notes if notes is not None else [""] * len(weeks)
    return [
        {
            "week": week,
            "family": family,
            "subject": subject,
            "statistic": statistic,
            "value": None if np.isnan(v) else float(v),
            "reference": reference,
            "limit": limit,
            "firing": bool(f),
            "alarm": bool(a),
            "note": note,
        }  # fmt: skip
        for week, v, f, a, note in zip(weeks, value, firing, started, notes, strict=True)
    ]


def evaluate_rules(
    scores: pl.DataFrame,
    matches: pl.DataFrame,
    counts: pl.DataFrame,
    rules: dict,
    weeks: list[str],
) -> MonitorResult:
    """Apply every rule to the given final weeks, in order."""
    weeks = sorted(weeks)
    scores = scores.filter(pl.col("week").is_in(weeks))
    matches = matches.filter(pl.col("week").is_in(weeks))
    weekly = weekly_sums(scores, rules["anchor"])
    rows: list[dict] = []

    # performance: CUSUM on the standardised weekly shortfall of the gain over the anchor
    slack = rules["performance"]["cusum_slack"]
    for model_id, rule in rules["performance"]["models"].items():
        n = _series(weekly, model_id, weeks, "rows")
        gain = alerts.ratio(
            _series(weekly, model_id, weeks, "anchor_loss")
            - _series(weekly, model_id, weeks, "loss"),
            np.nan_to_num(n),
        )
        shortfall = alerts.standardised_shortfall(
            gain, n, rule["reference_gain"], rule["weekly_sd"], rule["typical_rows"]
        )
        firing, level = alerts.cusum_levels(shortfall, slack, rule["limit"])
        notes = [f"weekly gain over the anchor {g:+.4f}" if not np.isnan(g) else "" for g in gain]
        rows += _rows(
            weeks, "performance", model_id, "cusum", level, 0.0, rule["limit"], firing, notes
        )

    # calibration: rolling window of the one-step slope and of the mean of p minus outcome
    window = rules["window_weeks"]
    for model_id, rule in rules["calibration"]["models"].items():
        sums = {
            c: np.nan_to_num(_series(weekly, model_id, weeks, c))
            for c in ("cal_u", "cal_i", "bias_sum", "rows")
        }
        rolled = {c: alerts.rolling_sum(v, window) for c, v in sums.items()}
        slope = alerts.one_step_slope(rolled["cal_u"], rolled["cal_i"])
        bias = alerts.ratio(rolled["bias_sum"], rolled["rows"])
        rows += _rows(
            weeks, "calibration", model_id, "slope", slope, rule["reference_slope"], rule["slope_limit"],  # noqa: E501
            alerts.outside(slope, rule["reference_slope"], rule["slope_limit"]),
        )  # fmt: skip
        rows += _rows(
            weeks, "calibration", model_id, "bias", bias, rule["reference_bias"], rule["bias_limit"],  # noqa: E501
            alerts.outside(bias, rule["reference_bias"], rule["bias_limit"]),
        )  # fmt: skip

    # inputs: percent band on the weekly mean of the economy totals
    if "inputs" in rules:
        rule = rules["inputs"]
        columns = list(rule["development_mean"])
        by_week = matches.group_by("week").agg(
            [pl.col(c).mean().alias(c) for c in columns] + [pl.col(columns[0]).count().alias("n")]
        )
        lookup = {row["week"]: row for row in by_week.iter_rows(named=True)}
        value, notes = [], []
        for week in weeks:
            row = lookup.get(week)
            if row is None or row["n"] < rule["minimum_matches"]:
                value.append(np.nan)
                notes.append("fewer than 20 judged matches: not checked")
                continue
            deviation = {c: row[c] / rule["development_mean"][c] - 1 for c in columns}
            worst = max(deviation, key=lambda c: abs(deviation[c]))
            value.append(abs(deviation[worst]))
            notes.append(f"{worst} {deviation[worst]:+.1%} against its development mean")
        value = np.array(value)
        rows += _rows(
            weeks, "inputs", "economy", "largest relative deviation", value, 0.0, rule["relative_limit"],  # noqa: E501
            alerts.below(-value, 0.0, rule["relative_limit"]), notes,
        )  # fmt: skip

    # data: fixed check on the share of judged-tier matches that are usable
    lookup = {row["week"]: row for row in counts.iter_rows(named=True)}
    share, notes = [], []
    for week in weeks:
        row = lookup.get(week)
        if row is None or row["judged_tier"] < rules["minimum_matches"]:
            share.append(np.nan)
            notes.append("fewer than 20 matches in the judged tiers: not checked")
        else:
            share.append(row["judged_tier_usable"] / row["judged_tier"])
            notes.append(
                f"{row['judged_tier_usable']} of {row['judged_tier']} judged-tier matches usable"
            )
    share = np.array(share)
    rows += _rows(
        weeks, "data", "pull", "usable share", share, 1.0, 1 - USABLE_SHARE_FLOOR,
        alerts.below(share, 1.0, 1 - USABLE_SHARE_FLOOR), notes,
    )  # fmt: skip

    events = (
        matches.filter(pl.col("patch").is_not_null())
        .group_by("patch")
        .agg(pl.col("week").min().alias("week"), pl.len().alias("matches"))
        .sort("week")
    )
    return MonitorResult(pl.DataFrame(rows, schema=STATE_SCHEMA), weekly, events)
