"""What to say when an alarm starts: where the problem sits and what to do about it.

``recommend`` turns the alarm states around an alarm into one of four actions. The rules
are fixed and short on purpose, so that a person can check them:

* data        the pull is unreliable; fix the data first, take no model action
* refit       the inputs or the patch changed, or performance dropped: fit a challenger
              with the champion's recipe on recent matches
* recalibrate predictions are off in confidence but still rank matches well: fit a
              recalibration layer on top of the champion
* watch       nothing to do yet (used for the baseline and the anchor)

``report`` writes the text of the issue: the alarm, the recommendation, and breakdowns of
the last 8 final weeks by minute phase, patch, league tier and league.
"""

from __future__ import annotations

import polars as pl

from vexil.evaluate import PHASES, summarise

LOOKBACK_WEEKS = 8
ACTIONS = {
    "data": "Data problem. Metrics of the affected weeks are unreliable. No model action; "
    "check the OpenDota pull and wait for the next run.",
    "refit": "Fit a refit challenger: the champion's recipe on all matches up to now. It is "
    "scored next to the champion and replaces it only through the promotion gate and "
    "with approval.",
    "recalibrate": "Fit a recalibration challenger: the champion with a slope and shift fitted "
    "on the last 12 final weeks. Ranking is intact, confidence is off.",
    "watch": "No action. This model is a yardstick, not the model in use.",
}


def recommend(
    states: pl.DataFrame, alarm: dict, champion_id: str, events: pl.DataFrame
) -> tuple[str, str]:
    """Action code and one sentence of reasoning for an alarm (a row of the states table)."""
    weeks = sorted(states["week"].unique())
    position = weeks.index(alarm["week"])
    recent = weeks[max(0, position - LOOKBACK_WEEKS + 1) : position + 1]
    window = states.filter(pl.col("week").is_in(recent) & pl.col("firing"))

    def fired(family: str, subject: str | None = None, weeks_back: int = LOOKBACK_WEEKS) -> bool:
        part = window.filter(pl.col("family") == family)
        part = part.filter(pl.col("week").is_in(recent[-weeks_back:]))
        if subject:
            part = part.filter(pl.col("subject") == subject)
        return part.height > 0

    new_patch = events.filter(pl.col("week").is_in(recent))
    patch_note = (
        f" Patch {new_patch['patch'][-1]} first appeared in week {new_patch['week'][-1]}."
        if new_patch.height
        else ""
    )
    if alarm["family"] == "data" or fired("data", weeks_back=3):
        return (
            "data",
            "The share of usable matches is below its floor in this or the last two weeks.",
        )
    if alarm["family"] == "inputs":
        return "refit", f"Model inputs changed scale ({alarm['note']}).{patch_note}"
    if alarm["subject"] != champion_id:
        return "watch", f"The alarm is on {alarm['subject']}, which is not the champion."
    if alarm["family"] == "calibration":
        if fired("performance", champion_id):
            return (
                "refit",
                "Calibration and performance alarms both fired within 8 weeks." + patch_note,
            )
        return (
            "recalibrate",
            "The calibration alarm fired while the performance alarm stayed quiet.",
        )
    if fired("inputs"):
        return "refit", "Performance dropped after the input alarm fired." + patch_note
    return "refit", "Performance dropped against the anchor." + patch_note


def _table(frame: pl.DataFrame, columns: list[str]) -> str:
    """A small markdown table."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in frame.select(columns).iter_rows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _breakdown(scores: pl.DataFrame, by: str, model_id: str, anchor_id: str, top: int = 6) -> str:
    """Log loss of the model and of the anchor, and the gain, per value of one column."""
    model = summarise(scores.filter(pl.col("model_id") == model_id), [by]).select(
        by, "matches", "rows", pl.col("log_loss").alias("model")
    )
    anchor = summarise(scores.filter(pl.col("model_id") == anchor_id), [by]).select(
        by, pl.col("log_loss").alias("anchor")
    )
    table = (
        model.join(anchor, on=by)
        .with_columns(gain=pl.col("anchor") - pl.col("model"))
        .sort("rows", descending=True)
        .head(top)
    )
    return _table(table, [by, "matches", "rows", "model", "anchor", "gain"])


def report(
    alarm: dict,
    action: tuple[str, str],
    states: pl.DataFrame,
    scores: pl.DataFrame,
    matches: pl.DataFrame,
    champion_id: str,
    anchor_id: str,
) -> str:
    """Markdown text of the issue for one alarm."""
    weeks = sorted(states["week"].unique())
    position = weeks.index(alarm["week"])
    recent = weeks[max(0, position - LOOKBACK_WEEKS + 1) : position + 1]
    model_id = (
        alarm["subject"] if alarm["family"] in ("performance", "calibration") else champion_id
    )
    window = scores.filter(pl.col("week").is_in(recent))
    window_matches = matches.filter(pl.col("week").is_in(recent))
    code, reason = action
    value = "n/a" if alarm["value"] is None else f"{alarm['value']:.4f}"
    lines = [
        f"## Alarm: {alarm['family']} / {alarm['subject']} in week {alarm['week']}",
        "",
        f"- Statistic: {alarm['statistic']} = {value} (reference {alarm['reference']:.4f}, "
        f"limit {alarm['limit']:.4f})",
        f"- Detail: {alarm['note'] or 'none'}",
        f"- Regions in the last {len(recent)} final weeks: "
        + ", ".join(sorted(window_matches["region"].unique())),
        "",
        f"### Recommendation: {code}",
        "",
        f"{reason} {ACTIONS[code]}",
        "",
        "### Other rules firing in the last 8 final weeks",
        "",
    ]
    others = states.filter(pl.col("week").is_in(recent) & pl.col("firing")).select(
        "week", "family", "subject", "statistic", "note"
    )
    lines.append(_table(others, others.columns) if others.height else "None.")
    if window.filter(pl.col("model_id") == model_id).height:
        phase_rows = []
        for model in (model_id, anchor_id):
            pooled = summarise(window.filter(pl.col("model_id") == model), ["model_id"])
            phase_rows.append({"model": model, **{p: pooled[f"log_loss_{p}"][0] for p in PHASES}})
        lines += [
            "",
            f"### Log loss by minute phase, last {len(recent)} final weeks",
            "",
            _table(pl.DataFrame(phase_rows), ["model", *PHASES]),
        ]
        for by in ("patch", "league_tier", "league_name"):
            lines += [
                "",
                f"### By {by.replace('_', ' ')} (minutes 0-29)",
                "",
                _breakdown(window, by, model_id, anchor_id),
            ]
    if window_matches.height:
        coverage = window_matches["new_player_share"].mean()
        lines += [
            "",
            "### Inputs and rating coverage",
            "",
            f"- Judged matches in the window: {window_matches.height}",
            "- Mean share of player slots held by players with fewer than 20 rated matches: "
            f"{coverage:.2f}",
        ]
    return "\n".join(lines) + "\n"
