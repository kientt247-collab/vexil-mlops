"""The model registry: which models are scored, and in which role.

The registry is one YAML file. The pipeline loops over its entries and never names a
model in code, which is what makes adding or retiring a model a one-file change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from vexil.candidates import FrozenTrainModel, load_candidate
from vexil.hashing import sha256_file
from vexil.models.linear import GoldTimeModel, RecalibratedModel, RidgeA1Model, load_model

ROLES = ("champion", "baseline", "anchor", "challenger", "retired")


class RegistryError(ValueError):
    """The registry file is inconsistent; the run must stop."""


@dataclass(frozen=True)
class ModelEntry:
    id: str
    role: str
    kind: str
    file: Path
    sha256: str
    description: str = ""
    approval: dict | None = None
    scored_from: str | None = None  # first week a challenger is scored, if set

    def load(self) -> RidgeA1Model | GoldTimeModel | RecalibratedModel | FrozenTrainModel:
        if self.kind == "frozen_train":
            return load_candidate(self.file)
        return load_model(self.file)


@dataclass(frozen=True)
class Registry:
    entries: tuple[ModelEntry, ...]

    def active(self) -> tuple[ModelEntry, ...]:
        """Every entry that is scored: all roles except retired."""
        return tuple(e for e in self.entries if e.role != "retired")

    def by_role(self, role: str) -> tuple[ModelEntry, ...]:
        return tuple(e for e in self.entries if e.role == role)

    def one(self, role: str) -> ModelEntry:
        (entry,) = self.by_role(role)
        return entry


def load_registry(path: Path, root: Path | None = None) -> Registry:
    """Read and check the registry. `root` is the folder that model file paths refer to."""
    path = Path(path)
    root = Path(root) if root else path.parent.parent
    raw = yaml.safe_load(path.read_text()) or {}
    entries = []
    for item in raw.get("models", []):
        known = {k: item.get(k) for k in ("id", "role", "kind", "file", "sha256")}
        missing = [k for k, v in known.items() if not v]
        if missing:
            raise RegistryError(f"entry {item.get('id')!r}: missing {', '.join(missing)}")
        entries.append(
            ModelEntry(
                **{**known, "file": root / known["file"]},
                description=item.get("description", ""),
                approval=item.get("approval"),
                scored_from=item.get("scored_from"),
            )
        )
    _check(entries)
    return Registry(tuple(entries))


def _check(entries: list[ModelEntry]) -> None:
    ids = [e.id for e in entries]
    if len(set(ids)) != len(ids):
        raise RegistryError("model ids must be unique")
    for entry in entries:
        if entry.role not in ROLES:
            raise RegistryError(f"{entry.id}: unknown role {entry.role!r}")
        if entry.role == "retired":
            continue
        if not entry.file.exists():
            raise RegistryError(f"{entry.id}: file {entry.file} does not exist")
        if sha256_file(entry.file) != entry.sha256:
            raise RegistryError(f"{entry.id}: file does not match the registered SHA-256")
    for role in ("champion", "anchor"):
        if sum(e.role == role for e in entries) != 1:
            raise RegistryError(f"the registry needs exactly one {role}")
    if sum(e.role == "baseline" for e in entries) > 1:
        raise RegistryError("the registry allows at most one baseline")
    champion = next(e for e in entries if e.role == "champion")
    approval = champion.approval or {}
    if not approval.get("by") or not approval.get("date"):
        raise RegistryError(f"{champion.id}: the champion needs approval fields 'by' and 'date'")


def append_entry(path: Path, entry: dict) -> None:
    """Add a model to the registry file as text, so that comments in the file survive."""
    lines = ["", f"  - id: {entry['id']}"]
    for key in ("role", "kind", "file", "sha256"):
        lines.append(f"    {key}: {entry[key]}")
    lines.append(
        f"    description: {json.dumps(entry['description'])}"
    )  # quoted: it may hold colons
    if entry.get("scored_from"):
        lines.append(f"    scored_from: {json.dumps(str(entry['scored_from']))}")
    path = Path(path)
    path.write_text(path.read_text().rstrip("\n") + "\n" + "\n".join(lines) + "\n")


def set_role(path: Path, model_id: str, role: str) -> None:
    """Change the role of one entry in the registry file (text edit; the file is re-checked on load)."""  # noqa: E501
    if role not in ROLES:
        raise RegistryError(f"unknown role {role!r}")
    lines = Path(path).read_text().split("\n")
    inside = False
    for i, line in enumerate(lines):
        if line.strip().startswith("- id:"):
            inside = line.split("- id:")[1].strip() == model_id
        elif inside and line.strip().startswith("role:"):
            lines[i] = line.split("role:")[0] + f"role: {role}"
            Path(path).write_text("\n".join(lines))
            return
    raise RegistryError(f"{model_id}: no such entry")
