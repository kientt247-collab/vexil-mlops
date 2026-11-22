"""P1 (time budget): cost of the weekly job with the four candidates, on real development weeks.

Copies models/ to a temporary root, registers the four candidates as challengers, rebuilds the
result history from the first stored week (the cold start of a job without a snapshot), then
scores the last three development weeks. Reports seconds per step and the size of the snapshot.
Labels come through vexil.labels, purpose "P1-timing". Budget: weekly job under 15 minutes.

Usage: uv run python scripts/p1_timing.py STORE_ROOT
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from vexil.candidates import ResultHistory
from vexil.data.store import Store
from vexil.job import Layout, _history, _rating_state
from vexil.labels import LabelLoader
from vexil.registry import append_entry, load_registry
from vexil.run import score_weeks

TAGS = ("A1", "A3", "A4", "A5")


def main(store_root: str) -> None:
    root = Path(tempfile.mkdtemp())
    shutil.copytree("models", root / "models")
    for tag in TAGS:
        file = root / "models" / "candidates" / tag / "model.json"
        append_entry(
            root / "models" / "registry.yaml",
            {
                "id": f"cand-{tag}", "role": "challenger", "kind": "frozen_train",
                "file": f"models/candidates/{tag}/model.json",
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "description": "timing",
            },
        )  # fmt: skip
    registry = load_registry(root / "models" / "registry.yaml", root)
    store = Store(Path(store_root), Path("data/manifest.json"))
    layout = Layout(root, store.root if hasattr(store, "root") else Path(store_root))
    loader = LabelLoader(store, Path("/tmp/p1_label_reads.log"), Path("lock.json"))
    weeks = [w for w in store.weeks() if w <= "2026-07-27"]
    last3 = weeks[-3:]
    out = {}
    t = time.time()
    history = _history(layout, store, loader, last3[0])
    out["cold_history_seconds"] = round(time.time() - t, 1)
    t = time.time()
    state = _rating_state(layout, store, loader, last3[0])
    out["cold_rating_seconds"] = round(time.time() - t, 1)
    per_week = {}
    for week in last3:
        t = time.time()
        outputs, _ = score_weeks(store, registry, loader, state, [week], "P1-timing", history)
        per_week[week] = {
            "seconds": round(time.time() - t, 1),
            "judged_matches": outputs[week].matches.height,
            "prediction_rows": outputs[week].predictions.height,
        }
    out["weeks"] = per_week
    path = root / "h.json.gz"
    history.save(path)
    out["history_snapshot_bytes"] = path.stat().st_size
    out["history_matches"] = len(history.meta)
    print(json.dumps(out, indent=1))
    Path("docs/evidence/p1_timing.json").write_text(json.dumps(out, indent=1) + "\n")
    assert isinstance(history, ResultHistory)


if __name__ == "__main__":
    main(sys.argv[1])
