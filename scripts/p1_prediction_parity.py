"""P1 (b): predictions of the candidates run through the package against the loop's evaluator.

For each candidate (and the champion, which is the baseline train.py of the loop) and each of the
five blocks B1..B5: fit on the block's training matches and predict its test rows, with the audited
train.py unchanged, on a table built from the vexil store. Compare with the evaluator's own run
(prepare.evaluate on the evaluator bundle, seed 0): largest absolute difference of the probabilities.
Development weeks only. Dev labels are read through vexil.labels (logged, purpose P1-parity).

Usage: uv run python scripts/p1_prediction_parity.py STORE_ROOT TAG [TAG ...]   (TAG in BASE A1 A3 A4 A5)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/claude/vauto/evaluator")
import prepare as P  # noqa: E402

from vexil.candidates import CandidateCtx, CandidateTable, load_train  # noqa: E402
from vexil.data.store import Store  # noqa: E402
from vexil.features.extra import load_items  # noqa: E402
from vexil.labels import LabelLoader  # noqa: E402

PATHS = {"BASE": "/home/claude/vauto/agent_template/train.py"} | {
    t: f"/home/claude/vauto/agents/{t}/train.py" for t in ("A1", "A3", "A4", "A5")
}
DEV_END = 1785456000


def main(store_root: str, tags: list[str]) -> None:
    store = Store(Path(store_root), Path("data/manifest.json"))
    loader = LabelLoader(store, Path("/tmp/p1_label_reads.log"), Path("lock.json"))
    d = P.Data()
    order = list(dict.fromkeys(d.rows["match_id"].tolist()))
    weeks = [w for w in store.weeks() if w <= "2026-07-27"]
    results: dict[int, int] = {}
    for w in weeks:
        results.update(loader.week_labels(w, "P1-parity"))
    results = {m: r for m, r in results.items() if m in d.meta}

    def matches():
        for w in weeks:
            for match, players in store.read_week(w):
                if match["start_time"] < DEV_END:
                    yield match, players

    table = CandidateTable.build(matches(), results, load_items(), order=order)
    assert (table.rows["match_id"] == d.rows["match_id"]).all() and (
        table.rows["t"] == d.rows["t"]
    ).all()
    report = {}
    for tag in tags:
        path = PATHS[tag]
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        mod = load_train(Path(path), sha)
        ref = P.evaluate(path, data=d, seeds=1)["p0"]
        worst, off, t0 = 0.0, 0, time.time()
        per_block = []
        for _b, tr, te in d.blocks:
            ids = sorted({int(m) for m in table.rows["match_id"][tr]})
            state = mod.fit(CandidateCtx(table, tr, label_ids=ids), ids, 0)
            cte = CandidateCtx(table, te)
            p = np.asarray(mod.predict(state, cte, cte.rows), dtype=float)
            diff = float(np.max(np.abs(p - ref[off : off + len(te)])))
            per_block.append(diff)
            worst = max(worst, diff)
            off += len(te)
        report[tag] = {
            "sha256": sha[:12],
            "largest_abs_diff_probability": worst,
            "per_block": per_block,
            "seconds": round(time.time() - t0, 1),
        }
        print(tag, json.dumps(report[tag]), flush=True)
    Path("docs/evidence").mkdir(parents=True, exist_ok=True)
    out = Path("docs/evidence/p1_prediction_parity_" + "_".join(tags) + ".json")
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
