"""Calendar rules of the project, in one place.

A week runs from Monday 00:00 UTC to the next Monday and is named by its start date
("2026-11-16"). A match belongs to the week of its ``start_time``. A week becomes final
14 days after it ends; until then it is provisional and is pulled again by every run.

Three data regions exist. They differ in what a number computed on them means:
development (the frozen models were fitted on it), the phase 2 holdout (labels were read
once on 2026-11-21, so new numbers there are marked POST-HOC), and prospective (clean).
Two short gaps are never scored: 2026-07-31, and 2026-11-20 to 2026-11-22.
"""

from __future__ import annotations

import datetime as dt

WEEK_SECONDS = 7 * 86400
FINAL_AFTER_DAYS = 14


def _ts(year: int, month: int, day: int) -> int:
    return int(dt.datetime(year, month, day, tzinfo=dt.UTC).timestamp())


DATA_START = _ts(2026, 1, 1)
DEVELOPMENT_END = _ts(2026, 7, 31)  # development: start_time before this
HOLDOUT_START = _ts(2026, 8, 1)
CUTOFF = _ts(2026, 11, 20)  # end of the phase 2 data; labels after it need the go-live lock
PROSPECTIVE_START = _ts(2026, 11, 23)
_MONDAY_ZERO = _ts(1970, 1, 5)  # the first Monday after the Unix epoch


def week_start(timestamp: int) -> int:
    """Unix time of the Monday 00:00 UTC that opens the week of `timestamp`."""
    return timestamp - (timestamp - _MONDAY_ZERO) % WEEK_SECONDS


def week_label(timestamp: int) -> str:
    """Name of the week of `timestamp`: the ISO date of its Monday."""
    return dt.datetime.fromtimestamp(week_start(timestamp), dt.UTC).strftime("%Y-%m-%d")


def week_bounds(label: str) -> tuple[int, int]:
    """Start (inclusive) and end (exclusive) Unix times of a named week."""
    start = int(dt.datetime.strptime(label, "%Y-%m-%d").replace(tzinfo=dt.UTC).timestamp())
    if week_start(start) != start:
        raise ValueError(f"{label} is not a Monday")
    return start, start + WEEK_SECONDS


def weeks_between(first: str, last: str) -> list[str]:
    """All week names from `first` to `last`, both included."""
    start, _ = week_bounds(first)
    end, _ = week_bounds(last)
    return [week_label(t) for t in range(start, end + 1, WEEK_SECONDS)]


def is_final(label: str, now: int) -> bool:
    """A week is final once 14 days have passed since it ended."""
    _, end = week_bounds(label)
    return now >= end + FINAL_AFTER_DAYS * 86400


def open_weeks(now: int) -> list[str]:
    """Weeks a run at time `now` must pull: every week that has ended and is not final
    before this run (the week that becomes final in this run is included once more),
    plus nothing from the running week."""
    current = week_start(now)
    span = FINAL_AFTER_DAYS * 86400 // WEEK_SECONDS + 1
    return [week_label(current - k * WEEK_SECONDS) for k in range(span, 0, -1)]


def region(start_time: int) -> str:
    """Region of a match: development, holdout, prospective, or gap (never scored)."""
    if start_time < DEVELOPMENT_END:
        return "development"
    if start_time < HOLDOUT_START:
        return "gap"
    if start_time < CUTOFF:
        return "holdout"
    if start_time < PROSPECTIVE_START:
        return "gap"
    return "prospective"
