"""P1 (a), step 1: the fields the candidates read, built from the vexil store, against the evaluator bundle.

Reads development weeks only (start before 2026-07-31 00:00 UTC). No labels are read here.
Usage: uv run python scripts/p1_input_parity.py STORE_ROOT BUNDLE_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from vexil.data.store import Store
from vexil.features.a1 import A1_INPUTS, a1_table
from vexil.matchdata import n_rows

DEV_END = 1785456000  # 2026-07-31 00:00 UTC


def main(store_root: str, bundle: str) -> None:
    store = Store(Path(store_root), Path("data/manifest.json"))
    meta = {m["id"]: m for m in json.load(open(Path(bundle) / "meta.json"))}
    z = np.load(Path(bundle) / "rows.npz")
    rows_mid, rows_t = z["match_id"], z["t"]
    first = {}
    for i, m in enumerate(rows_mid.tolist()):
        first.setdefault(m, i)
    out = {
        "matches_store_dev": 0,
        "meta_equal": 0,
        "meta_diff": [],
        "a1_rows_checked": 0,
        "a1_max_abs_diff": 0.0,
        "missing_in_store": [],
        "extra_in_store_usable": 0,
    }
    seen = set()
    for week in [w for w in store.weeks() if w <= "2026-07-27"]:
        for match, players in store.read_week(week):
            if match["start_time"] >= DEV_END:
                continue
            out["matches_store_dev"] += 1
            mid = match["match_id"]
            seen.add(mid)
            if mid not in meta:
                out["extra_in_store_usable"] += 1 if n_rows(match) else 0
                continue
            ordered = sorted(players, key=lambda p: p["player_slot"])
            rad = [p for p in ordered if p["player_slot"] < 128]
            dire = [p for p in ordered if p["player_slot"] >= 128]
            mine = {
                "start": match["start_time"],
                "end": match["start_time"] + (match.get("duration") or 0),
                "ra": [p.get("account_id") for p in rad],
                "da": [p.get("account_id") for p in dire],
                "rh": [p.get("hero_id") for p in rad],
                "dh": [p.get("hero_id") for p in dire],
            }
            ref = meta[mid]
            bad = [k for k in mine if mine[k] != ref[k]]
            if bad:
                out["meta_diff"].append((mid, bad))
            else:
                out["meta_equal"] += 1
            if mid in first and n_rows(match):
                tab = a1_table(match, players)
                idx = np.flatnonzero(rows_mid == mid)
                assert len(idx) == n_rows(match) and (rows_t[idx] == np.arange(len(idx))).all(), mid
                for name in A1_INPUTS:
                    d = float(np.max(np.abs(tab[name].astype(float) - z[name][idx].astype(float))))
                    out["a1_max_abs_diff"] = max(out["a1_max_abs_diff"], d)
                out["a1_rows_checked"] += len(idx)
    out["missing_in_store"] = sorted(set(meta) - seen)[:20]
    out["n_missing_in_store"] = len(set(meta) - seen)
    out["meta_diff"] = out["meta_diff"][:20]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
