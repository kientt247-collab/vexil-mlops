import json
from pathlib import Path

import numpy as np
import pytest

from vexil.features.a1 import A1_INPUTS
from vexil.models.fit import fit_recalibration, fit_ridge_a1
from vexil.models.linear import RecalibratedModel, load_model
from vexil.retrain import gate, paired_gain_interval

REPO = Path(__file__).resolve().parents[1]


def shadow_period(true_gain, n_matches=900, seed=0):
    """Per-match loss sums for a champion and a challenger whose true gain is known."""
    rng = np.random.default_rng(seed)
    rows = np.full(n_matches, 30.0)
    groups = np.arange(n_matches) // 2
    champion = rows * (0.52 + rng.normal(0, 0.15, n_matches))
    challenger = champion - rows * (true_gain + rng.normal(0, 0.02, n_matches))
    p = rng.uniform(0.05, 0.95, 20_000)
    y = (rng.random(20_000) < p).astype(float)  # a calibrated challenger: slope near 1
    return champion, challenger, rows, groups, p, y


def test_gate_passes_a_clearly_better_challenger():
    result = gate(*shadow_period(true_gain=0.01))
    assert result.passed and result.reasons == []
    assert result.gain == pytest.approx(0.01, abs=0.002) and result.interval[0] > 0


def test_gate_rejects_and_says_why():
    no_gain = gate(*shadow_period(true_gain=0.0))
    assert not no_gain.passed and any("below 0.002" in r for r in no_gain.reasons)
    too_few = gate(*shadow_period(true_gain=0.01, n_matches=300))
    assert not too_few.passed and any("600 needed" in r for r in too_few.reasons)
    champion, challenger, rows, groups, p, y = shadow_period(true_gain=0.01)
    overconfident = np.clip(1 / (1 + np.exp(-np.log(p / (1 - p)) / 0.7)), 1e-6, 1 - 1e-6)
    bad_slope = gate(champion, challenger, rows, groups, overconfident, y)
    assert not bad_slope.passed and any("slope" in r for r in bad_slope.reasons)


def test_interval_resamples_whole_groups():
    champion, challenger, rows, groups, _, _ = shadow_period(true_gain=0.005)
    _, by_match = paired_gain_interval(champion, challenger, rows, np.arange(len(rows)))
    _, one_group = paired_gain_interval(champion, challenger, rows, np.zeros(len(rows)))
    assert by_match[1] - by_match[0] > 0
    assert one_group[1] - one_group[0] == pytest.approx(
        0, abs=1e-12
    )  # one group: nothing to resample


def test_recalibration_layer_undoes_an_overconfident_model(tmp_path):
    rng = np.random.default_rng(3)
    log_odds = rng.normal(0, 1.5, 100_000)
    y = (rng.random(100_000) < 1 / (1 + np.exp(-log_odds))).astype(int)
    layer = fit_recalibration(log_odds / 0.85, y)
    assert layer["slope"] == pytest.approx(0.85, abs=0.02) and layer["shift"] == pytest.approx(
        0, abs=0.02
    )
    spec = {"kind": "recalibrated", "base_file": "candidate.json", "slope": 0.9, "shift": 0.1}
    (tmp_path / "candidate.json").write_text((REPO / "models" / "candidate.json").read_text())
    (tmp_path / "k2.json").write_text(json.dumps(spec))
    model, base = load_model(tmp_path / "k2.json"), load_model(tmp_path / "candidate.json")
    assert isinstance(model, RecalibratedModel)
    cases = json.loads((REPO / "tests" / "data" / "model_cases.json").read_text())
    inputs = {k: np.array(v) for k, v in cases["inputs"].items()}
    assert model.log_odds(inputs) == pytest.approx(0.1 + 0.9 * base.log_odds(inputs))


def test_fit_ridge_a1_writes_a_file_the_loader_reads(tmp_path):
    rng = np.random.default_rng(5)
    n = 5000
    inputs = {name: rng.normal(0, 1, n) for name in A1_INPUTS}
    inputs["t"] = rng.integers(0, 40, n).astype(float)
    inputs["pelx"] = rng.normal(0, 0.3, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(inputs["kill_diff"] + inputs["pelx"])))).astype(int)
    spec = fit_ridge_a1(inputs, y, added=("pelx",), c_value=0.03)
    path = tmp_path / "challenger.json"
    path.write_text(json.dumps(spec))
    p = load_model(path).predict(inputs)
    assert ((p > 0.5) == (y == 1)).mean() > 0.6 and len(spec["coef"]) == 13
