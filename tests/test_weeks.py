import datetime as dt

import pytest

from vexil import weeks


def ts(*args):
    return int(dt.datetime(*args, tzinfo=dt.UTC).timestamp())


def test_week_is_named_by_its_monday():
    assert weeks.week_label(ts(2026, 11, 21, 9)) == "2026-11-16"  # a Saturday
    assert weeks.week_label(ts(2026, 11, 23)) == "2026-11-23"  # Monday 00:00 opens a week
    assert weeks.week_label(ts(2026, 11, 22, 23, 59, 59)) == "2026-11-16"
    assert weeks.week_label(ts(2026, 1, 1)) == "2025-12-29"


def test_week_bounds_and_range():
    start, end = weeks.week_bounds("2026-11-23")
    assert (start, end) == (ts(2026, 11, 23), ts(2026, 11, 30))
    with pytest.raises(ValueError):
        weeks.week_bounds("2026-11-24")
    assert len(weeks.weeks_between("2025-12-29", "2026-11-16")) == 47


def test_a_week_is_final_14_days_after_it_ends():
    assert not weeks.is_final("2026-11-23", ts(2026, 12, 13, 23))
    assert weeks.is_final("2026-11-23", ts(2026, 12, 14))


def test_a_run_pulls_the_three_most_recent_ended_weeks():
    assert weeks.open_weeks(ts(2026, 12, 14, 3, 17)) == ["2026-11-23", "2026-11-30", "2026-12-07"]


@pytest.mark.parametrize(
    "when,expected",
    [
        (ts(2026, 7, 30, 23), "development"),
        (ts(2026, 7, 31, 12), "gap"),
        (ts(2026, 8, 1), "holdout"),
        (ts(2026, 11, 19, 23), "holdout"),
        (ts(2026, 11, 20), "gap"),
        (ts(2026, 11, 22, 23), "gap"),
        (ts(2026, 11, 23), "prospective"),
    ],
)
def test_regions(when, expected):
    assert weeks.region(when) == expected


def test_history_weeks_are_final_once_the_prospective_region_starts():
    assert not weeks.is_final("2026-11-09", ts(2026, 11, 22, 23))
    assert weeks.is_final("2026-11-16", ts(2026, 11, 23))  # frozen history, no 14-day wait
    assert not weeks.is_final("2026-11-23", ts(2026, 12, 7))
