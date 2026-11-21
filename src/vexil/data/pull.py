"""The weekly pull: fetch open weeks, check them, and store what changed."""

from __future__ import annotations

from vexil.data.opendota import Explorer, pull_week
from vexil.data.schema import match_problems
from vexil.data.store import Store
from vexil.weeks import CUTOFF

MAX_REJECTED_SHARE = 0.05


def check_rows(matches: list[dict], players: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Drop matches that fail the shape checks. Returns kept rows and a report."""
    by_match: dict[int, list[dict]] = {}
    for row in players:
        by_match.setdefault(row.get("match_id"), []).append(row)
    kept, rejected = [], {}
    for match in matches:
        problems = match_problems(match, by_match.get(match.get("match_id"), []))
        if problems:
            rejected[match.get("match_id")] = problems
        else:
            kept.append(match)
    if matches and len(rejected) / len(matches) > MAX_REJECTED_SHARE:
        sample = next(iter(rejected.items()))
        raise RuntimeError(
            f"{len(rejected)} of {len(matches)} matches fail the shape checks, e.g. {sample}; "
            "OpenDota may have changed its tables"
        )
    kept_ids = {m["match_id"] for m in kept}
    kept_players = [p for p in players if p.get("match_id") in kept_ids]
    return kept, kept_players, {"rejected": len(rejected), "reasons": rejected}


def pull_weeks(store: Store, explorer: Explorer, weeks: list[str]) -> dict:
    """Pull each week and store it if its content changed.

    Rows of matches that start before the phase 2 cutoff are history: if the store already
    holds them, they are kept as stored and only later matches are taken from the new pull.
    That keeps every number computed on 2026 data up to the cutoff reproducible.
    """
    report = {}
    for week in weeks:
        matches, players, checks = check_rows(*pull_week(explorer, week))
        if week in store.manifest["weeks"]:
            old_matches, old_players = store._read_raw(week)
            frozen_ids = {m["match_id"] for m in old_matches if m["start_time"] < CUTOFF}
            new_ids = {m["match_id"] for m in matches if m["start_time"] >= CUTOFF}
            matches = [m for m in old_matches if m["match_id"] in frozen_ids] + [
                m for m in matches if m["match_id"] in new_ids
            ]
            players = [p for p in old_players if p["match_id"] in frozen_ids] + [
                p for p in players if p["match_id"] in new_ids
            ]
        changed = store.write_week(week, matches, players, source="opendota")
        report[week] = {"matches": len(matches), "changed": changed, **checks}
    return report
