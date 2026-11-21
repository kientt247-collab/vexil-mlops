"""Build the weekly store from the phase 2 data store. Runs once, in a session.

    uv run python scripts/seed_from_phase2.py --phase2 /path/to/v2 --store _store

The phase 2 store holds every match from 2026-01-01 up to 2026-11-20 00:00 UTC in files of
200 matches. This script regroups them by week, adds patch and league fields to each match
row, and writes revision 1 of every week. No match result is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vexil.data.schema import match_problems
from vexil.data.store import Store, read_jsonl_gz
from vexil.weeks import week_label

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=REPO / "_store")
    args = parser.parse_args()
    source = args.phase2 / "data" / "opendota"
    store = Store(args.store, REPO / "data" / "manifest.json")

    patches = {r["match_id"]: r["patch"] for r in read_jsonl_gz(source / "patch.jsonl.gz")}
    leagues = {r["leagueid"]: r for r in read_jsonl_gz(source / "leagues.jsonl.gz")}
    chunk_files = sorted(source.glob("matches_*.jsonl.gz"))

    # pass 1: which chunk files hold matches of which week
    files_of_week: dict[str, set[Path]] = {}
    for path in chunk_files:
        for match in read_jsonl_gz(path):
            files_of_week.setdefault(week_label(match["start_time"]), set()).add(path)

    # pass 2: one week at a time
    summary = {"weeks": 0, "matches": 0, "shape problems": 0}
    for week in sorted(files_of_week):
        matches, players = [], []
        for path in sorted(files_of_week[week]):
            ids = set()
            for match in read_jsonl_gz(path):
                if week_label(match["start_time"]) == week:
                    league = leagues.get(match["leagueid"], {})
                    match["patch"] = patches.get(match["match_id"])
                    match["league_tier"] = league.get("tier")
                    match["league_name"] = league.get("name")
                    matches.append(match)
                    ids.add(match["match_id"])
            player_path = path.with_name(path.name.replace("matches_", "players_"))
            players += [row for row in read_jsonl_gz(player_path) if row["match_id"] in ids]
        by_match: dict[int, list[dict]] = {}
        for row in players:
            by_match.setdefault(row["match_id"], []).append(row)
        summary["shape problems"] += sum(
            bool(match_problems(m, by_match.get(m["match_id"], []))) for m in matches
        )
        store.write_week(week, matches, players, source="phase2-seed")
        summary["weeks"] += 1
        summary["matches"] += len(matches)
    summary["usable"] = sum(entry["usable"] for entry in store.manifest["weeks"].values())
    summary["megabytes"] = round(
        sum(sum(entry["bytes"].values()) for entry in store.manifest["weeks"].values()) / 1e6, 1
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
