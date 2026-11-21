"""Node M2 checks that need OpenDota and the seeded store. Runs in a session.

    uv run python scripts/m2_checks.py --store _store

1. Three past weeks pulled again from OpenDota are compared with the stored weeks.
2. A pull written twice to a scratch store changes nothing the second time.
3. Parse lag: share of usable matches by age at the time of the phase 2 pull.
No match result is used: the result field is left out of every comparison.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from vexil.data.opendota import OpenDotaClient, pull_week
from vexil.data.pull import check_rows
from vexil.data.store import Store
from vexil.matchdata import is_usable
from vexil.weeks import CUTOFF

REPO = Path(__file__).resolve().parents[1]
WEEKS = ("2026-03-02", "2026-08-10", "2026-11-02")
MATCH_ARRAYS = ("radiant_gold_adv", "radiant_xp_adv", "objectives")
PLAYER_ARRAYS = ("gold_t", "lh_t", "kills_log")
JUDGED_TIERS = ("professional", "premium")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=REPO / "_store")
    args = parser.parse_args()
    store = Store(args.store, REPO / "data" / "manifest.json")
    client = OpenDotaClient()
    out: dict = {"weeks pulled again": {}}

    for week in WEEKS:
        matches, players, checks = check_rows(*pull_week(client.explorer, week))
        old_matches, old_players = store._read_raw(week)
        old = {m["match_id"]: m for m in old_matches}
        new = {m["match_id"]: m for m in matches}

        def by_slot(rows):
            return {(p["match_id"], p["player_slot"]): p for p in rows}

        old_p, new_p = by_slot(old_players), by_slot(players)
        common = sorted(set(old) & set(new))
        same_arrays = 0
        tier_or_patch_changed = 0
        for match_id in common:
            same = all(old[match_id].get(k) == new[match_id].get(k) for k in MATCH_ARRAYS)
            slots = [key for key in old_p if key[0] == match_id]
            same = same and all(
                key in new_p and all(old_p[key].get(k) == new_p[key].get(k) for k in PLAYER_ARRAYS)
                for key in slots
            )
            same_arrays += same
            tier_or_patch_changed += any(
                old[match_id].get(k) != new[match_id].get(k) for k in ("patch", "league_tier")
            )
        out["weeks pulled again"][week] = {
            "matches stored": len(old),
            "matches now": len(new),
            "only stored": len(set(old) - set(new)),
            "only now": len(set(new) - set(old)),
            "match-id difference, share": round(len(set(old) ^ set(new)) / len(old), 4),
            "arrays equal, share of common matches": round(same_arrays / len(common), 4),
            "patch or tier changed": tier_or_patch_changed,
            "usable stored": sum(is_usable(m) for m in old_matches),
            "usable now": sum(is_usable(m) for m in matches),
            "rejected by shape checks": checks["rejected"],
        }
        if week == WEEKS[0]:
            with tempfile.TemporaryDirectory() as scratch:
                scratch_store = Store(Path(scratch) / "s", Path(scratch) / "manifest.json")
                first = scratch_store.write_week(week, matches, players, source="opendota")
                before = scratch_store.manifest_path.read_text()
                second = scratch_store.write_week(week, matches, players, source="opendota")
                out["writing the same pull twice"] = {
                    "first write stored": first,
                    "second write stored": second,
                    "manifest unchanged": before == scratch_store.manifest_path.read_text(),
                }
    out["explorer calls"] = client.calls

    # parse lag, from the store as pulled at the cutoff: usable share by age of the match
    buckets = {"0-7 days": [0, 0], "7-14 days": [0, 0], "14-21 days": [0, 0], "21-28 days": [0, 0]}
    older = [0, 0]
    for week in store.weeks():
        for match in store._read_raw(week)[0]:
            if match.get("league_tier") not in JUDGED_TIERS:
                continue
            age_days = (CUTOFF - match["start_time"]) / 86400
            target = older
            for label in buckets:
                low, high = (int(x) for x in label.split(" ")[0].split("-"))
                if low <= age_days < high:
                    target = buckets[label]
            target[0] += is_usable(match)
            target[1] += 1
    out["usable share of judged-tier matches by age at the phase 2 pull"] = {
        **{
            k: {"usable": v[0], "matches": v[1], "share": round(v[0] / max(v[1], 1), 4)}
            for k, v in buckets.items()
        },
        "older": {"usable": older[0], "matches": older[1], "share": round(older[0] / older[1], 4)},
    }
    (REPO / "docs" / "evidence" / "m2_checks.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
