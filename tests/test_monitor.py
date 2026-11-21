"""Planted cases for the alarm rules, the recommendation and the issue channel.

The tables are built by hand, so that each case contains exactly one fault.
"""

from pathlib import Path

import polars as pl
import pytest
import yaml

from vexil.investigate import recommend, report
from vexil.issues import LogChannel, issue_title, sync
from vexil.monitor import evaluate_rules
from vexil.weeks import WEEK_SECONDS, week_bounds, week_label

REPO = Path(__file__).resolve().parents[1]
RULES = yaml.safe_load((REPO / "monitoring" / "rules.yaml").read_text())
FIRST = week_bounds("2027-01-04")[0]
WEEKS = [week_label(FIRST + k * WEEK_SECONDS) for k in range(30)]
FAULT = 15  # index of the week in which a planted fault starts
MATCHES, ROWS = 60, 30
CHAMPION = "final-model"


def build(
    gain_drop=0.0, slope=None, gold_factor=1.0, usable=MATCHES, patch="7.42", fault_only_week=False
):
    """Weekly tables for 30 weeks of 60 matches; a fault starts in week 15."""
    score_rows, match_rows, count_rows = [], [], []
    means = RULES["inputs"]["development_mean"]
    for index, week in enumerate(WEEKS):
        faulty = index == FAULT if fault_only_week else index >= FAULT
        for k in range(MATCHES):
            match_id = 1000 * index + k
            anchor_loss = 0.55 * ROWS
            for model_id, performance in RULES["performance"]["models"].items():
                calibration = RULES["calibration"]["models"][model_id]
                drop = gain_drop if faulty and model_id == CHAMPION else 0.0
                target = (
                    slope
                    if faulty and slope and model_id == CHAMPION
                    else calibration["reference_slope"]
                )
                score_rows.append(
                    {
                        "week": week,
                        "match_id": match_id,
                        "model_id": model_id,
                        "n": ROWS,
                        "loss": anchor_loss - ROWS * (performance["reference_gain"] - drop),
                        "cal_u": (target - 1) * 0.2 * ROWS,
                        "cal_i": 0.2 * ROWS,
                        "bias_sum": calibration["reference_bias"] * ROWS,
                    }  # fmt: skip
                )
            score_rows.append(
                {
                    "week": week,
                    "match_id": match_id,
                    "model_id": "anchor",
                    "n": ROWS,
                    "loss": anchor_loss,
                    "cal_u": 0.0,
                    "cal_i": 0.2 * ROWS,
                    "bias_sum": 0.0,
                }  # fmt: skip
            )
            factor = gold_factor if faulty else 1.0
            match_rows.append(
                {
                    "week": week,
                    "match_id": match_id,
                    "region": "prospective",
                    "league_tier": "professional",
                    "league_name": f"League {k % 3}",
                    "patch": patch if faulty else "7.41",
                    "new_player_share": 0.1,
                    **{c: m * (factor if "gold" in c else 1.0) for c, m in means.items()},
                }  # fmt: skip
            )
        count_rows.append(
            {
                "week": week,
                "judged_tier": MATCHES,
                "judged_tier_usable": usable if faulty else MATCHES,
            }
        )
    scores = pl.DataFrame(score_rows)
    for phase in ("0-9", "10-19", "20-29", "30+"):
        scores = scores.with_columns(
            (pl.col("loss") / 3).alias(f"loss_{phase}"), pl.lit(10).alias(f"n_{phase}")
        )
    facts = pl.DataFrame(match_rows)
    scores = scores.join(
        facts.select("match_id", "region", "league_tier", "league_name", "patch"), on="match_id"
    )
    return scores, facts, pl.DataFrame(count_rows)


def alarms_of(result):
    return (
        result.states.filter(pl.col("alarm"))
        .select("week", "family", "subject", "statistic")
        .rows()
    )


def first_alarm(result, family):
    alarm = result.states.filter(pl.col("alarm") & (pl.col("family") == family)).row(0, named=True)
    return alarm, recommend(result.states, alarm, CHAMPION, result.events)


def test_a_clean_year_raises_no_alarm():
    result = evaluate_rules(*build(), RULES, WEEKS)
    assert alarms_of(result) == []
    assert result.states.height == len(WEEKS) * (
        2 + 4 + 1 + 1
    )  # 2 performance, 4 calibration, inputs, data


def test_case_slope_drift_with_the_gain_intact_recommends_recalibration():
    result = evaluate_rules(*build(slope=0.80), RULES, WEEKS)
    families = {family for _, family, _, _ in alarms_of(result)}
    assert families == {"calibration"}
    alarm, (code, reason) = first_alarm(result, "calibration")
    assert alarm["subject"] == CHAMPION and alarm["statistic"] == "slope"
    assert WEEKS.index(alarm["week"]) - FAULT < 8  # found before the window has fully turned over
    assert code == "recalibrate" and "performance alarm stayed quiet" in reason


def test_case_input_shift_after_a_new_patch_names_the_input_and_recommends_a_refit():
    scores, matches, counts = build(gold_factor=1.10, patch="7.99")
    result = evaluate_rules(scores, matches, counts, RULES, WEEKS)
    alarm, (code, reason) = first_alarm(result, "inputs")
    assert alarm["week"] == WEEKS[FAULT]  # label-free: fires in the first week of the fault
    assert "total_gold" in alarm["note"] and "+10.0%" in alarm["note"]
    assert code == "refit" and "7.99" in reason
    text = report(alarm, (code, reason), result.states, scores, matches, CHAMPION, "anchor")
    assert "Recommendation: refit" in text and "By patch" in text and "7.99" in text


def test_case_surge_of_unparsed_matches_is_a_data_alarm_with_no_model_action():
    result = evaluate_rules(*build(usable=30, gain_drop=0.05, fault_only_week=True), RULES, WEEKS)
    alarm, (code, reason) = first_alarm(result, "data")
    assert alarm["week"] == WEEKS[FAULT] and "30 of 60" in alarm["note"]
    assert code == "data"
    # a performance alarm in the same weeks is not acted on either: the data come first
    performance, (code, _) = first_alarm(result, "performance")
    assert performance["week"] == WEEKS[FAULT] and code == "data"


def test_a_lasting_drop_of_the_champion_raises_the_performance_alarm_and_recommends_a_refit():
    result = evaluate_rules(*build(gain_drop=0.02), RULES, WEEKS)
    alarm, (code, _) = first_alarm(result, "performance")
    assert alarm["subject"] == CHAMPION and 0 <= WEEKS.index(alarm["week"]) - FAULT <= 3
    assert code == "refit"


def test_weeks_with_few_matches_are_not_checked_by_the_input_and_data_rules():
    scores, matches, counts = build(gold_factor=1.5)
    few = matches.filter(pl.col("match_id") % 1000 < 10)  # 10 matches a week
    result = evaluate_rules(scores, few, counts.with_columns(judged_tier=pl.lit(10)), RULES, WEEKS)
    quiet = result.states.filter(pl.col("family").is_in(["inputs", "data"]))
    assert not quiet["firing"].any() and quiet["value"].null_count() == quiet.height


def test_one_alarm_keeps_one_issue_until_four_quiet_weeks(tmp_path):
    result = evaluate_rules(*build(gain_drop=0.02), RULES, WEEKS)
    channel = LogChannel(tmp_path / "issues.jsonl")
    title = issue_title("performance", CHAMPION)
    actions = []
    for week in WEEKS:
        actions += [(week, action) for action in sync(result.states, week, {}, channel)]
    kinds = [action.split(":")[0] for _, action in actions if title in action]
    assert kinds[0] == "open" and "comment" in kinds and kinds.count("open") == 1
    assert title in channel.open_titles()  # the drop lasts, so the issue stays open

    recovered = evaluate_rules(*build(gain_drop=0.05, fault_only_week=True), RULES, WEEKS)
    channel = LogChannel(tmp_path / "issues2.jsonl")
    log = [action for week in WEEKS for action in sync(recovered.states, week, {}, channel)]
    assert log == [f"open: {title}", f"close: {title}"]
    assert channel.open_titles() == set()


def test_rules_file_matches_the_registry():
    registry = yaml.safe_load((REPO / "models" / "registry.yaml").read_text())
    ids = {m["id"] for m in registry["models"]}
    assert set(RULES["performance"]["models"]) <= ids and RULES["anchor"] in ids
    assert RULES["performance"]["arm"] == "B" and RULES["inputs"][
        "relative_limit"
    ] == pytest.approx(0.0418, abs=1e-3)
