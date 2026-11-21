"""Logistic models stored as plain JSON files, predicted with NumPy only.

Three kinds exist:

* ``ridge_a1``: the frozen phase 2 models (R* and the final model). Design matrix
  ``[100 * u, z(ten A1 inputs), z(f), z(f * u) for every added input f]`` where
  ``u`` is ln(1 + t) standardised with the mean and standard deviation of the fitting rows,
  and ``z`` standardises with statistics stored in the file.
* ``gold_time``: the anchor. Design matrix ``[u, gold, gold * u]`` divided by stored scales.
* ``recalibrated``: another model file plus a slope and a shift applied to its log-odds.

Prediction needs no fitting library, so a model file can be scored anywhere NumPy runs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Inputs = Mapping[str, np.ndarray]


def sigmoid(log_odds: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-log_odds))


def standardised_log_minute(t: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """u = (ln(1 + t) - mean) / sd, with mean and sd taken from the fitting rows."""
    return (np.log1p(np.asarray(t, dtype=float)) - mean) / sd


@dataclass(frozen=True)
class RidgeA1Model:
    """R* or the final model, exactly as phase 2 saved it."""

    features: tuple[str, ...]
    added: tuple[str, ...]
    mu_u: float
    sd_u: float
    mu: np.ndarray
    sd: np.ndarray
    coef: np.ndarray
    intercept: float

    @property
    def input_names(self) -> tuple[str, ...]:
        return ("t", *self.features, *self.added)

    def log_odds(self, inputs: Inputs) -> np.ndarray:
        u = standardised_log_minute(inputs["t"], self.mu_u, self.sd_u)
        blocks = [
            np.column_stack([np.asarray(inputs[name], dtype=float) for name in self.features])
        ]
        for name in self.added:
            value = np.asarray(inputs[name], dtype=float)
            blocks.append(np.column_stack([value, value * u]))
        raw = np.column_stack(blocks)
        design = np.column_stack([100 * u, (raw - self.mu) / self.sd])
        return design @ self.coef + self.intercept

    def predict(self, inputs: Inputs) -> np.ndarray:
        return sigmoid(self.log_odds(inputs))


@dataclass(frozen=True)
class GoldTimeModel:
    """The anchor: logistic regression on the gold lead, time and their product."""

    mu_u: float
    sd_u: float
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    input_names = ("t", "gold")

    def log_odds(self, inputs: Inputs) -> np.ndarray:
        u = standardised_log_minute(inputs["t"], self.mu_u, self.sd_u)
        gold = np.asarray(inputs["gold"], dtype=float)
        design = np.column_stack([u, gold, gold * u]) / self.scale
        return design @ self.coef + self.intercept

    def predict(self, inputs: Inputs) -> np.ndarray:
        return sigmoid(self.log_odds(inputs))


@dataclass(frozen=True)
class RecalibratedModel:
    """Another model with its log-odds rescaled: ``shift + slope * log-odds``."""

    base: RidgeA1Model
    slope: float
    shift: float

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.base.input_names

    def log_odds(self, inputs: Inputs) -> np.ndarray:
        return self.shift + self.slope * self.base.log_odds(inputs)

    def predict(self, inputs: Inputs) -> np.ndarray:
        return sigmoid(self.log_odds(inputs))


def load_model(path: Path) -> RidgeA1Model | GoldTimeModel | RecalibratedModel:
    """Read a model file. Files without a ``kind`` are phase 2 files (``ridge_a1``)."""
    spec = json.loads(Path(path).read_text())
    kind = spec.get("kind", "ridge_a1")
    if kind == "ridge_a1":
        if spec.get("curved") or spec.get("calibrator"):
            raise ValueError("curved inputs and calibrators are not part of the frozen models")
        return RidgeA1Model(
            features=tuple(spec["features"]),
            added=tuple(spec["added"]),
            mu_u=spec["mu_u"],
            sd_u=spec["sd_u"],
            mu=np.array(spec["mu"]),
            sd=np.array(spec["sd"]),
            coef=np.array(spec["coef"]),
            intercept=spec["intercept"],
        )
    if kind == "gold_time":
        return GoldTimeModel(
            mu_u=spec["mu_u"],
            sd_u=spec["sd_u"],
            scale=np.array(spec["scale"]),
            coef=np.array(spec["coef"]),
            intercept=spec["intercept"],
        )
    if kind == "recalibrated":
        base = load_model(Path(path).parent / spec["base_file"])
        if not isinstance(base, RidgeA1Model):
            raise ValueError("a recalibrated model needs a ridge_a1 base")
        return RecalibratedModel(base=base, slope=spec["slope"], shift=spec["shift"])
    raise ValueError(f"unknown model kind: {kind}")
