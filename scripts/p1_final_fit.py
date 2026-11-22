"""P1 (final fit): fit each candidate once on all development matches and write its model file.

Development = matches that start before 2026-07-31 00:00 UTC (the loop's development split).
Same settings as in the loop (seed 0, no new tuning). Labels come through vexil.labels with the
purpose "P1-final-fit". Each model file is written twice into temporary files first: the two
must be byte identical, else the script stops.

Usage: uv run python scripts/p1_final_fit.py STORE_ROOT TAG [TAG ...]   (TAG in A1 A3 A4 A5)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from vexil.candidates import CandidateTable, fit_candidate
from vexil.data.store import Store
from vexil.features.extra import load_items
from vexil.labels import LabelLoader
from vexil.matchdata import is_usable
from vexil.scoring import JUDGED_TIERS

DEV_END = 1785456000  # 2026-07-31 00:00 UTC
FOLDER = Path("models/candidates")


def main(store_root: str, tags: list[str]) -> None:
    store = Store(Path(store_root), Path("data/manifest.json"))
    loader = LabelLoader(store, Path("/tmp/p1_label_reads.log"), Path("lock.json"))
    weeks = [w for w in store.weeks() if w <= "2026-07-27"]
    results: dict[int, int] = {}
    for w in weeks:
        results.update(loader.week_labels(w, "P1-final-fit"))

    fit_ids: set[int] = set()

    def matches():
        for w in weeks:
            for match, players in store.read_week(w):
                if match["start_time"] < DEV_END:
                    if is_usable(match) and match.get("league_tier") in JUDGED_TIERS:
                        fit_ids.add(match["match_id"])
                    yield match, players

    table = CandidateTable.build(matches(), results, load_items())
    # The fitting set of the loop and of the champion: usable matches of a judged tier that
    # started before DEV_END and have a result (4,470 matches, 170,519 rows).
    ids = sorted(fit_ids & set(results))
    n_rows = int(np.isin(table.rows["match_id"], ids).sum())
    print(f"fitting set: {len(ids)} matches, {n_rows} rows")
    if (len(ids), n_rows) != (4470, 170519):
        raise SystemExit("the fitting set is not the one of the loop (4,470 matches, 170,519 rows)")
    report = {}
    for tag in tags:
        train = FOLDER / tag / "train.py"
        sha = hashlib.sha256(train.read_bytes()).hexdigest()
        t0 = time.time()
        specs = [fit_candidate(train, sha, table, ids, seed=0) for _ in range(2)]
        texts = [json.dumps(s, separators=(",", ":"), sort_keys=True) for s in specs]
        if texts[0] != texts[1]:
            raise SystemExit(f"{tag}: two fits with seed 0 gave different state files")
        out = FOLDER / tag / "model.json"
        out.write_text(texts[0] + "\n")
        report[tag] = {
            "train_sha256": sha,
            "model_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "n_matches": specs[0]["n_matches"],
            "n_rows": specs[0]["n_rows"],
            "model_bytes": out.stat().st_size,
            "two_fits_identical": True,
            "seconds_for_two_fits": round(time.time() - t0, 1),
        }
        print(tag, json.dumps(report[tag]), flush=True)
    Path("docs/evidence").mkdir(parents=True, exist_ok=True)
    name = "p1_final_fit_" + "_".join(tags) + ".json"
    Path("docs/evidence", name).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
