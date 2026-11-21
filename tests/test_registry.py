import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from vexil.registry import RegistryError, load_registry
from vexil.scoring import is_judged, match_inputs, score_match

from .synthetic import make_match

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    """A copy of the models folder, so that tests can edit the registry file."""
    shutil.copytree(REPO / "models", tmp_path / "models")
    return tmp_path


def edit(workspace, change):
    path = workspace / "models" / "registry.yaml"
    raw = yaml.safe_load(path.read_text())
    change(raw["models"])
    path.write_text(yaml.safe_dump(raw))
    return load_registry(path)


def score(registry):
    match, players = make_match(3)
    return score_match(registry.active(), match_inputs(match, players, rating_input=0.2))


def test_the_registry_in_the_repository_is_valid():
    registry = load_registry(REPO / "models" / "registry.yaml")
    assert [e.id for e in registry.active()] == ["final-model", "rstar", "anchor"]
    assert registry.one("champion").id == "final-model"
    assert registry.one("anchor").id == "anchor"


def test_a_model_goes_in_and_out_by_editing_the_registry_file_only(workspace):
    """Success condition 4: no file other than registry.yaml changes."""
    before = score(load_registry(workspace / "models" / "registry.yaml"))
    assert set(before) == {"final-model", "rstar", "anchor"}

    rstar = next(
        m
        for m in yaml.safe_load((REPO / "models" / "registry.yaml").read_text())["models"]
        if m["id"] == "rstar"
    )
    newcomer = {**rstar, "id": "rstar-copy", "role": "challenger"}
    with_newcomer = score(edit(workspace, lambda models: models.append(newcomer)))
    assert set(with_newcomer) == {"final-model", "rstar", "anchor", "rstar-copy"}
    assert np.array_equal(with_newcomer["rstar-copy"], with_newcomer["rstar"])

    def retire(models):
        next(m for m in models if m["id"] == "rstar-copy")["role"] = "retired"

    assert set(score(edit(workspace, retire))) == {"final-model", "rstar", "anchor"}


def test_a_changed_model_file_stops_the_run(workspace):
    (workspace / "models" / "rstar.json").write_text("{}")
    with pytest.raises(RegistryError, match="SHA-256"):
        load_registry(workspace / "models" / "registry.yaml")


def test_the_champion_role_needs_approval_fields(workspace):
    def promote_without_approval(models):
        next(m for m in models if m["id"] == "final-model")["role"] = "retired"
        next(m for m in models if m["id"] == "rstar")["role"] = "champion"

    with pytest.raises(RegistryError, match="approval"):
        edit(workspace, promote_without_approval)

    def promote_with_approval(models):
        next(m for m in models if m["id"] == "rstar")["approval"] = {
            "by": "Kien",
            "date": "2027-01-01",
        }

    assert edit(workspace, promote_with_approval).one("champion").id == "rstar"


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda models: models.append(dict(models[1])), "unique"),
        (lambda models: models[1].update(role="boss"), "unknown role"),
        (lambda models: models[2].update(role="baseline"), "exactly one anchor"),
        (lambda models: models[1].pop("sha256"), "missing sha256"),
    ],
)
def test_inconsistent_registries_are_rejected(workspace, change, message):
    with pytest.raises(RegistryError, match=message):
        edit(workspace, change)


def test_judged_matches():
    match, _ = make_match(1)
    assert not is_judged(match)  # no league tier yet
    assert is_judged({**match, "league_tier": "professional"})
    assert not is_judged({**match, "league_tier": "excluded"})
    assert not is_judged({**match, "league_tier": "premium", "game_mode": 22})
    assert not is_judged({**match, "league_tier": "premium", "duration": 500})
