"""P1 (a), step 2: the twenty extra inputs built from the vexil store against the evaluator bundle.

Development weeks only, no labels. Usage: uv run python scripts/p1_extra_parity.py STORE_ROOT BUNDLE_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from vexil.data.store import Store
from vexil.features.extra import EXTRA_INPUTS, extra_table, load_items
from vexil.matchdata import n_rows

DEV_END = 1785456000


def main(store_root: str, bundle: str) -> None:
    store = Store(Path(store_root), Path("data/manifest.json"))
    items = load_items()
    z = np.load(Path(bundle) / "rows.npz")
    mid_col, rows_t = z["match_id"], z["t"]
    start_of = {}
    for i, m in enumerate(mid_col.tolist()):
        start_of.setdefault(m, i)
    worst = {k: 0.0 for k in EXTRA_INPUTS}
    n_rows_checked = n_bad = 0
    for week in [w for w in store.weeks() if w <= "2026-07-27"]:
        for match, players in store.read_week(week):
            if (
                match["start_time"] >= DEV_END
                or not n_rows(match)
                or match["match_id"] not in start_of
            ):
                continue
            idx = np.flatnonzero(mid_col == match["match_id"])
            assert (rows_t[idx] == np.arange(len(idx))).all()
            tab = extra_table(match, players, items)
            for k in EXTRA_INPUTS:
                d = float(np.max(np.abs(tab[k].astype(float) - z[k][idx].astype(float))))
                worst[k] = max(worst[k], d)
                n_bad += d > 0
            n_rows_checked += len(idx)
    print(
        json.dumps(
            {
                "rows_checked": n_rows_checked,
                "column_match_pairs_with_difference": n_bad,
                "largest_abs_diff_by_column": worst,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
