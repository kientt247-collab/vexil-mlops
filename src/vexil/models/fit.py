"""Fitting code. Kept apart from prediction so that scoring never imports scikit-learn."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_gold_time(t: np.ndarray, gold: np.ndarray, y: np.ndarray) -> dict:
    """Fit the anchor exactly as phase 2 fitted its baseline "R2".

    Almost no penalty (C = 1e6); the three columns are divided by their standard
    deviation on the fitting rows.
    """
    log_minute = np.log1p(np.asarray(t, dtype=float))
    mu_u, sd_u = float(log_minute.mean()), float(log_minute.std())
    u = (log_minute - mu_u) / sd_u
    gold = np.asarray(gold, dtype=float)
    design = np.column_stack([u, gold, gold * u])
    scale = design.std(0)
    scale[scale == 0] = 1
    model = LogisticRegression(C=1e6, max_iter=5000).fit(design / scale, y)
    return {
        "kind": "gold_time",
        "mu_u": mu_u,
        "sd_u": sd_u,
        "scale": scale.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "n_rows": int(len(y)),
    }


def fit_ridge_a1(
    inputs: dict[str, np.ndarray], y: np.ndarray, added: tuple[str, ...], c_value: float
) -> dict:
    """Fit the recipe of the phase 2 models: ridge logistic regression on the design
    ``[100 * u, z(ten A1 inputs), z(f), z(f * u) for every added input f]``.

    The result has the layout of the phase 2 model files, so ``load_model`` reads it.
    """
    from vexil.features.a1 import A1_INPUTS

    log_minute = np.log1p(np.asarray(inputs["t"], dtype=float))
    mu_u, sd_u = float(log_minute.mean()), float(log_minute.std())
    u = (log_minute - mu_u) / sd_u
    blocks = [np.column_stack([np.asarray(inputs[name], dtype=float) for name in A1_INPUTS])]
    for name in added:
        value = np.asarray(inputs[name], dtype=float)
        blocks.append(np.column_stack([value, value * u]))
    raw = np.column_stack(blocks)
    mu, sd = raw.mean(0), raw.std(0)
    sd[sd == 0] = 1
    design = np.column_stack([100 * u, (raw - mu) / sd])
    model = LogisticRegression(C=c_value, solver="newton-cholesky", max_iter=1000, tol=1e-8)
    model.fit(design, y)
    return {
        "kind": "ridge_a1",
        "features": list(A1_INPUTS),
        "added": list(added),
        "curved": [],
        "C": c_value,
        "mu_u": mu_u,
        "sd_u": sd_u,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "n_rows": int(len(y)),
    }


def fit_recalibration(log_odds: np.ndarray, y: np.ndarray) -> dict:
    """Slope and shift of a logistic fit of the outcome on a model's log-odds."""
    model = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(log_odds).reshape(-1, 1), y)
    return {
        "slope": float(model.coef_[0, 0]),
        "shift": float(model.intercept_[0]),
        "n_rows": int(len(y)),
    }
