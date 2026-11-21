"""One week through the pipeline: ratings, inputs and predictions for every model.

``process_week`` is the unit of work of both the weekly job and the replay. It reads a
week from the store, advances the rating state with the results the caller passes in, and
returns two tables: one row per match, and one row per (match, minute, model) with the
predicted probability. It does not evaluate anything; that is ``vexil.evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from vexil.data.store import Store
from vexil.matchdata import is_usable
from vexil.ratings import MatchMeta, RatingState
from vexil.registry import Registry
from vexil.scoring import JUDGED_TIERS, is_judged, match_inputs, score_match
from vexil.weeks import region

MATCH_SCHEMA = {
    "week": pl.String, "match_id": pl.Int64, "start_time": pl.Int64, "region": pl.String,
    "league_tier": pl.String, "league_name": pl.String, "patch": pl.String, "pair": pl.String,
    "rows": pl.Int32, "rating_input": pl.Float64, "new_player_share": pl.Float64,
    "abs_gold_10": pl.Float64, "abs_gold_20": pl.Float64, "abs_xp_pm_10": pl.Float64,
    "abs_xp_pm_20": pl.Float64, "abs_kill_diff_20": pl.Float64,
    "total_gold_10": pl.Float64, "total_gold_20": pl.Float64, "total_xp_10": pl.Float64,
    "total_xp_20": pl.Float64, "total_kills_20": pl.Float64,
}  # fmt: skip
# Size of the economy: totals over the ten players. They barely depend on which leagues
# play in a week, unlike the leads above, so a patch that changes gold or experience
# income shows up here. The input alarm watches these columns.
ECONOMY_COLUMNS = ("total_gold_10", "total_gold_20", "total_xp_10", "total_xp_20", "total_kills_20")
# Inputs watched by the input alarm: (column, input name, minute). Absolute values, because
# a signed lead averages to zero over matches and would hide a change of scale.
WATCHED_INPUTS = (
    ("abs_gold_10", "gold", 10), ("abs_gold_20", "gold", 20), ("abs_xp_pm_10", "xp_adv_pm", 10),
    ("abs_xp_pm_20", "xp_adv_pm", 20), ("abs_kill_diff_20", "kill_diff", 20),
)  # fmt: skip
PREDICTION_SCHEMA = {
    "week": pl.String, "match_id": pl.Int64, "t": pl.Int16, "model_id": pl.String,
    "p": pl.Float64,
}  # fmt: skip


@dataclass
class WeekOutput:
    matches: pl.DataFrame  # judged matches outside the gaps, one row each
    predictions: pl.DataFrame  # one row per match, minute and model
    counts: dict  # data-health counts for the week


def economy_totals(players: list[dict]) -> dict[str, float | None]:
    """Totals over the ten players at minutes 10 and 20 (None if the match is shorter)."""
    out: dict[str, float | None] = {}
    for minute in (10, 20):
        for name, series in (("gold", "gold_t"), ("xp", "xp_t")):
            values = [p.get(series) or [] for p in players]
            complete = all(len(v) > minute for v in values)
            out[f"total_{name}_{minute}"] = (
                float(sum(v[minute] for v in values)) if complete else None
            )
    long_enough = all(len(p.get("gold_t") or []) > 20 for p in players)
    kills = sum(1 for p in players for e in (p.get("kills_log") or []) if e["time"] <= 1200)
    out["total_kills_20"] = float(kills) if long_enough else None
    return out


def pair_key(match: dict, meta: MatchMeta) -> str:
    """The two sides of a match, in an order-free form (team ids, else account sets)."""
    radiant_team, dire_team = match.get("radiant_team_id"), match.get("dire_team_id")
    if radiant_team and dire_team:
        low, high = sorted((radiant_team, dire_team))
        return f"t:{low}-{high}"
    sides = sorted(
        ",".join(str(a) for a in sorted(side, key=str)) for side in (meta.radiant, meta.dire)
    )
    return "a:" + "|".join(sides)


def process_week(
    store: Store, week: str, registry: Registry, state: RatingState, results: dict[int, int]
) -> WeekOutput:
    """Score one week. `results` are the match results the rating may use (from the loader)."""
    rows = list(store.read_week(week))
    metas = {match["match_id"]: MatchMeta.from_rows(match, players) for match, players in rows}
    shares = {}
    rating_inputs = {}
    for meta in sorted(metas.values(), key=lambda m: (m.start, m.match_id)):
        rating_inputs[meta.match_id] = state.read(meta)
        shares[meta.match_id] = state.new_player_share(meta)
        if meta.match_id in results:
            state.add_result(meta, results[meta.match_id])

    entries = registry.active()
    match_rows, prediction_frames = [], []
    counts = {
        "matches": len(rows),
        "judged_tier": 0,
        "judged_tier_usable": 0,
        "judged": 0,
        "gap": 0,
    }
    for match, players in rows:
        in_tier = match.get("league_tier") in JUDGED_TIERS
        counts["judged_tier"] += in_tier
        counts["judged_tier_usable"] += in_tier and is_usable(match)
        if not is_judged(match):
            continue
        match_region = region(match["start_time"])
        if match_region == "gap":
            counts["gap"] += 1
            continue
        counts["judged"] += 1
        match_id = match["match_id"]
        inputs = match_inputs(match, players, rating_inputs[match_id])
        match_rows.append(
            {
                "week": week,
                "match_id": match_id,
                "start_time": match["start_time"],
                "region": match_region,
                "league_tier": match.get("league_tier"),
                "league_name": match.get("league_name"),
                "patch": match.get("patch"),
                "pair": pair_key(match, metas[match_id]),
                "rows": len(inputs["t"]),
                "rating_input": rating_inputs[match_id],
                "new_player_share": shares[match_id],
                **{
                    column: abs(float(inputs[name][minute])) if len(inputs["t"]) > minute else None
                    for column, name, minute in WATCHED_INPUTS
                },
                **economy_totals(players),
            }  # fmt: skip
        )
        for model_id, p in score_match(entries, inputs).items():
            prediction_frames.append(
                pl.DataFrame(
                    {
                        "week": week,
                        "match_id": match_id,
                        "t": inputs["t"].astype("int16"),
                        "model_id": model_id,
                        "p": p,
                    },
                    schema=PREDICTION_SCHEMA,
                )  # fmt: skip
            )
    matches = pl.DataFrame(match_rows, schema=MATCH_SCHEMA)
    predictions = (
        pl.concat(prediction_frames)
        if prediction_frames
        else pl.DataFrame(schema=PREDICTION_SCHEMA)
    )
    return WeekOutput(matches, predictions, counts)
