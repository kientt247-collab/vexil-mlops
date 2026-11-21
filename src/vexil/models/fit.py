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
