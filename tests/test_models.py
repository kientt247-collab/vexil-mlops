import json
from pathlib import Path

import numpy as np
import pytest

from vexil.hashing import sha256_file
from vexil.models.fit import fit_gold_time
from vexil.models.linear import GoldTimeModel, RidgeA1Model, load_model

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
CASES = json.loads((REPO / "tests" / "data" / "model_cases.json").read_text())

# Frozen files. A changed hash means the model changed, which the project forbids.
FROZEN_SHA256 = {
    "candidate": "a24854dc32cb6e5b2e1eb37c8f2560ba0b868f10455d2cd6d8862cdbadaff072",
    "rstar": "87a22e23efe8b1a29d5573ee71cf7f031afc055bf917e80e42385a9f22238ee5",
    "anchor": "102adcb55f41036b3090568e9ad7bc8a48498101ba22d9333976935ab6671446",
}


@pytest.mark.parametrize("name", FROZEN_SHA256)
def test_model_files_are_unchanged(name):
    assert sha256_file(MODELS / f"{name}.json") == FROZEN_SHA256[name]


@pytest.mark.parametrize("name", FROZEN_SHA256)
def test_predictions_match_reference_values(name):
    inputs = {key: np.array(values) for key, values in CASES["inputs"].items()}
    got = load_model(MODELS / f"{name}.json").predict(inputs)
    assert got == pytest.approx(CASES["expected"][name], abs=1e-12)


def test_model_kinds_and_inputs():
    candidate, rstar = load_model(MODELS / "candidate.json"), load_model(MODELS / "rstar.json")
    assert isinstance(candidate, RidgeA1Model) and candidate.added == ("pelx",)
    assert isinstance(rstar, RidgeA1Model) and rstar.added == ()
    assert candidate.input_names[0] == "t" and len(rstar.input_names) == 11
    assert isinstance(load_model(MODELS / "anchor.json"), GoldTimeModel)


def test_unknown_kind_is_rejected(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"kind": "mystery"}))
    with pytest.raises(ValueError):
        load_model(path)


def test_fit_gold_time_recovers_a_planted_relation():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 40, 20000).astype(float)
    gold = rng.normal(0, 5000, 20000)
    p = 1 / (1 + np.exp(-(0.0004 * gold)))
    y = (rng.random(20000) < p).astype(int)
    spec = fit_gold_time(t, gold, y)
    model = GoldTimeModel(
        spec["mu_u"],
        spec["sd_u"],
        np.array(spec["scale"]),
        np.array(spec["coef"]),
        spec["intercept"],
    )
    predicted = model.predict({"t": t, "gold": gold})
    assert np.abs(predicted - p).mean() < 0.02
