"""P1 (final fit parity): the written model files against a fit by the loop's own evaluator code.

For each candidate: fit the audited train.py on all development rows with the evaluator's Ctx
(prepare.Ctx, seed 0), load our model file, and compare the probabilities on all 170,519
development rows. Largest absolute difference is written to docs/evidence.

Usage: uv run python scripts/p1_final_parity.py TAG [TAG ...]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/claude/vauto/evaluator")
import prepare as P  # noqa: E402

from vexil.candidates import CandidateTable, load_candidate, load_train  # noqa: E402

FOLDER = Path("models/candidates")


def main(tags: list[str]) -> None:
    d = P.Data()
    everything = np.arange(len(d.rows["match_id"]))
    ids = sorted({int(m) for m in d.rows["match_id"]})
    report = {}
    for tag in tags:
        train = FOLDER / tag / "train.py"
        sha = hashlib.sha256(train.read_bytes()).hexdigest()
        mod = load_train(train, sha)
        state = mod.fit(P.Ctx(d, everything, label_ids=ids), ids, 0)
        ctx = P.Ctx(d, everything)
        ref = np.asarray(mod.predict(state, ctx, ctx.rows), dtype=float)
        model = load_candidate(FOLDER / tag / "model.json")
        # our table is rebuilt from the evaluator bundle's rows, so only the model file differs
        table = _table_from_bundle(d)
        ours = model.predict_rows(table, np.arange(len(table.rows["match_id"])))
        report[tag] = {
            "rows": int(len(ref)),
            "largest_abs_diff_probability": float(np.max(np.abs(ours - ref))),
        }
        print(tag, json.dumps(report[tag]), flush=True)
    out = Path("docs/evidence", "p1_final_parity_" + "_".join(tags) + ".json")
    out.write_text(json.dumps(report, indent=1) + "\n")


def _table_from_bundle(d) -> CandidateTable:
    """CandidateTable view of the evaluator bundle (rows and match facts as the loop had them)."""
    table = CandidateTable()
    table.rows = {k: np.asarray(v) for k, v in d.rows.items()}
    table.meta = {int(k): v for k, v in d.meta.items()}
    for v in table.meta.values():
        v.setdefault("leagueid", 0)
        v.setdefault("tier", "professional")
        v.setdefault("game_mode", 2)
    table.start = {k: v["start"] for k, v in table.meta.items()}
    table.results = {int(k): int(v) for k, v in d.results.items()}
    ends = sorted((v["end"], k) for k, v in table.meta.items())
    table.end_sorted = np.array([e for e, _ in ends], dtype=np.int64)
    table.end_ids = [i for _, i in ends]
    return table


if __name__ == "__main__":
    main(sys.argv[1:])
