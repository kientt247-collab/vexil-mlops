"""Evaluation: from predictions and results to per-match scores and weekly metrics.

Everything is built from one small table, ``match_scores``: one row per match and model
with the sums needed for log loss by minute phase. Sums add up, so a weekly or rolling
number is a group-by over this table, and pooled numbers are exactly the sum of the weekly
ones. The calibration slope does not add up; it is fitted on minute rows.

Metric: log loss over minute rows t = 0 to 29 (lower is better; a coin flip scores 0.693).
"""

from __future__ import annotations

import numpy as np
import polars as pl

PHASES = ("0-9", "10-19", "20-29", "30+")
EARLY_PHASES = PHASES[:3]  # minutes 0 to 29: the judged range
GROUP_GAP_SECONDS = 3 * 3600


def log_loss_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calibration(y: np.ndarray, p: np.ndarray, iterations: int = 50) -> tuple[float, float]:
    """Slope and intercept of a logistic fit of the outcome on logit(p), by Newton's method.

    A slope of 1 and an intercept of 0 are ideal; a slope below 1 means overconfidence.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.ones(len(p)), np.log(p / (1 - p))])
    beta = np.array([0.0, 1.0])
    for _ in range(iterations):
        mu = 1 / (1 + np.exp(-(x @ beta)))
        gradient = x.T @ (y - mu)
        hessian = (x * (mu * (1 - mu))[:, None]).T @ x
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[1]), float(beta[0])


def match_scores(predictions: pl.DataFrame, results: dict[int, int]) -> pl.DataFrame:
    """One row per (match, model): row counts and loss sums by minute phase.

    Matches without a known result are left out.
    """
    labels = pl.DataFrame(
        {"match_id": list(results), "y": list(results.values())},
        schema={"match_id": pl.Int64, "y": pl.Int8},
    )
    clipped = pl.col("p").cast(pl.Float64).clip(1e-6, 1 - 1e-6)
    rows = predictions.join(labels, on="match_id", how="inner").with_columns(
        q=clipped,
        x=(clipped / (1 - clipped)).log(),
        loss=-(
            pl.col("y") * pl.col("p").clip(1e-12, 1 - 1e-12).log()
            + (1 - pl.col("y")) * (1 - pl.col("p").clip(1e-12, 1 - 1e-12)).log()
        ),
        phase=pl.when(pl.col("t") < 10).then(pl.lit("0-9"))
        .when(pl.col("t") < 20).then(pl.lit("10-19"))
        .when(pl.col("t") < 30).then(pl.lit("20-29"))
        .otherwise(pl.lit("30+")),
    )  # fmt: skip
    aggregates = [pl.col("y").first()]
    for phase in PHASES:
        in_phase = pl.col("phase") == phase
        aggregates += [
            in_phase.sum().cast(pl.Int32).alias(f"n_{phase}"),
            pl.col("loss").filter(in_phase).sum().alias(f"loss_{phase}"),
        ]
    early = pl.col("t") < 30
    aggregates += [
        (pl.col("p") - pl.col("y")).filter(early).sum().alias("bias_sum"),
        # sums for the one-step calibration slope: 1 + cal_u / cal_i (see vexil.alerts)
        (pl.col("x") * (pl.col("y") - pl.col("q"))).filter(early).sum().alias("cal_u"),
        (pl.col("x") ** 2 * pl.col("q") * (1 - pl.col("q"))).filter(early).sum().alias("cal_i"),
    ]
    scores = rows.group_by(["week", "match_id", "model_id"], maintain_order=True).agg(aggregates)
    return scores.with_columns(
        n=sum(pl.col(f"n_{phase}") for phase in EARLY_PHASES),
        loss=sum(pl.col(f"loss_{phase}") for phase in EARLY_PHASES),
    )


def add_groups(matches: pl.DataFrame) -> pl.DataFrame:
    """Give each match a group: games of the same two sides less than 3 hours apart."""
    ordered = matches.sort(["pair", "start_time", "match_id"])
    new_group = (
        (pl.col("start_time") - pl.col("start_time").shift(1).over("pair")) >= GROUP_GAP_SECONDS
    ).fill_null(True)
    return ordered.with_columns(group=new_group.cum_sum().over("pair")).with_columns(
        group=pl.col("pair") + "#" + pl.col("group").cast(pl.String)
    )


def summarise(scores: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Log loss (minutes 0-29), loss by phase, and mean of p minus outcome, per group of rows."""
    aggregates = [
        pl.col("match_id").n_unique().alias("matches"),
        pl.col("n").sum().alias("rows"),
        (pl.col("loss").sum() / pl.col("n").sum()).alias("log_loss"),
        (pl.col("bias_sum").sum() / pl.col("n").sum()).alias("bias"),
    ]
    for phase in PHASES:
        aggregates.append(
            (pl.col(f"loss_{phase}").sum() / pl.col(f"n_{phase}").sum()).alias(f"log_loss_{phase}")
        )
    return scores.group_by(by, maintain_order=True).agg(aggregates)


def paired_gain(
    scores: pl.DataFrame, model_id: str, reference_id: str, by: list[str]
) -> pl.DataFrame:
    """Log loss of the reference minus log loss of the model, on the same matches."""
    keys = ["match_id", *by]
    model = scores.filter(pl.col("model_id") == model_id).select([*keys, "n", "loss"])
    reference = scores.filter(pl.col("model_id") == reference_id).select(
        ["match_id", pl.col("loss").alias("reference_loss")]
    )
    joined = model.join(reference, on="match_id", how="inner")
    aggregate = ((pl.col("reference_loss").sum() - pl.col("loss").sum()) / pl.col("n").sum()).alias(
        "gain"
    )
    if not by:
        return joined.select(aggregate)
    return joined.group_by(by, maintain_order=True).agg(aggregate)
