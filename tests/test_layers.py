"""The import rule of docs/guide.md, checked by reading the source.

Layer 1 is pure logic, layer 2 holds single steps, layer 3 touches disk, network or GitHub.
A module may import modules of its own layer or of a lower number, never a higher one.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "vexil"
LAYER = {
    "weeks": 1, "matchdata": 1, "hashing": 1, "alerts": 1, "evaluate": 1, "ratings": 1,
    "features.a1": 1, "models.linear": 1,
    "scoring": 2, "registry": 2, "monitor": 2, "investigate": 2, "retrain": 2, "leakcheck": 2,
    "models.fit": 2,
    "data.schema": 3, "data.store": 3, "data.opendota": 3, "data.pull": 3, "data.remote": 3,
    "labels": 3, "pipeline": 3, "run": 3, "issues": 3, "job": 3, "dashboard": 3, "cli": 3,
    "__main__": 3,
}  # fmt: skip


def module_name(path: Path) -> str:
    return ".".join(path.relative_to(PACKAGE).with_suffix("").parts)


def imported_modules(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vexil"):
            base = node.module.removeprefix("vexil").lstrip(".")
            for alias in node.names:
                candidate = f"{base}.{alias.name}".lstrip(".")
                found.add(candidate if candidate in LAYER else base)
    return {name for name in found if name}


def test_every_module_has_a_layer():
    modules = {module_name(p) for p in PACKAGE.rglob("*.py") if p.name != "__init__.py"}
    assert modules == set(LAYER)


def test_no_module_imports_from_a_higher_layer():
    problems = []
    for path in PACKAGE.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        name = module_name(path)
        for target in imported_modules(path):
            if LAYER[target] > LAYER[name]:
                problems.append(
                    f"{name} (layer {LAYER[name]}) imports {target} (layer {LAYER[target]})"
                )
    assert problems == []
