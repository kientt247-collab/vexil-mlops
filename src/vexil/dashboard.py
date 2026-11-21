"""The dashboard: one static page built from the monitoring tables.

``build_data`` collects every number the page shows into one dictionary; ``build`` writes
that dictionary into the HTML template. The page has no server and no external script, so
it can be hosted anywhere and opened from disk. Tests compare ``build_data`` with the
tables, which is how "the page shows what the tables say" is checked.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from vexil import alerts
from vexil.evaluate import paired_gain, summarise
from vexil.job import Layout
from vexil.monitor import weekly_sums
from vexil.registry import load_registry
from vexil.run import effective_scores, read_matches
from vexil.weeks import region, week_bounds

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
REGION_NOTES = {
    "development": "out-of-fold predictions; the models were fitted on these matches",
    "holdout": "POST-HOC: labels already read by phase 2 on 2026-11-21",
    "prospective": "clean: scored after the rules were locked",
}


def _clean(values) -> list:
    """Floats for JSON: NaN and infinities become null, numbers are rounded."""
    return [None if v is None or not np.isfinite(v) else round(float(v), 6) for v in values]


def _series(weekly: pl.DataFrame, model_id: str, weeks: list[str], column: str) -> np.ndarray:
    part = weekly.filter(pl.col("model_id") == model_id)
    lookup = dict(zip(part["week"], part[column], strict=True))
    return np.array([lookup.get(week, np.nan) for week in weeks], dtype=float)


def build_data(layout: Layout) -> dict:
    """Everything the page shows, as plain data."""
    folder = layout.monitoring
    rules = yaml.safe_load(layout.rules.read_text())
    gate = yaml.safe_load(layout.gate.read_text())
    registry = load_registry(layout.registry, layout.root)
    states = pl.read_parquet(folder / "alert_states.parquet")
    weeks = sorted(states["week"].unique())
    scores = effective_scores(folder).filter(pl.col("week").is_in(weeks))
    matches = read_matches(folder).filter(pl.col("week").is_in(weeks))
    counts = pl.read_parquet(folder / "weekly_counts.parquet").filter(pl.col("week").is_in(weeks))
    weekly = weekly_sums(scores, rules["anchor"])
    window = rules["window_weeks"]

    def state_values(family: str, subject: str, statistic: str, column: str = "value") -> list:
        part = states.filter(
            (pl.col("family") == family)
            & (pl.col("subject") == subject)
            & (pl.col("statistic") == statistic)
        )
        lookup = dict(zip(part["week"], part[column], strict=True))
        return [lookup.get(week) for week in weeks]

    models = {}
    for model_id, rule in rules["performance"]["models"].items():
        rows = _series(weekly, model_id, weeks, "rows")
        gain_sum = _series(weekly, model_id, weeks, "anchor_loss") - _series(
            weekly, model_id, weeks, "loss"
        )
        filled = {
            c: np.nan_to_num(_series(weekly, model_id, weeks, c))
            for c in ("cal_u", "cal_i", "bias_sum")
        }
        rolled_rows = alerts.rolling_sum(np.nan_to_num(rows), window)
        calibration = rules["calibration"]["models"][model_id]
        models[model_id] = {
            "matches": _clean(_series(weekly, model_id, weeks, "matches")),
            "gain_week": _clean(alerts.ratio(gain_sum, np.nan_to_num(rows))),
            "gain_rolling": _clean(
                alerts.ratio(alerts.rolling_sum(np.nan_to_num(gain_sum), window), rolled_rows)
            ),
            "slope_rolling": _clean(
                alerts.one_step_slope(
                    alerts.rolling_sum(filled["cal_u"], window),
                    alerts.rolling_sum(filled["cal_i"], window),
                )
            ),
            "bias_rolling": _clean(
                alerts.ratio(alerts.rolling_sum(filled["bias_sum"], window), rolled_rows)
            ),
            "cusum": _clean(
                [np.nan if v is None else v for v in state_values("performance", model_id, "cusum")]
            ),
            "reference_gain": rule["reference_gain"],
            "cusum_limit": rule["limit"],
            "reference_slope": calibration["reference_slope"],
            "slope_limit": calibration["slope_limit"],
            "reference_bias": calibration["reference_bias"],
            "bias_limit": calibration["bias_limit"],
        }

    by_region = []
    facts = scores.select(
        "match_id",
        "model_id",
        "region",
        "n",
        "loss",
        "bias_sum",
        "week",
        *[c for c in scores.columns if c.startswith(("n_", "loss_"))],
    )
    for name in ("development", "holdout", "prospective"):
        part = facts.filter(pl.col("region") == name)
        if not part.height:
            continue
        pooled = {r["model_id"]: r for r in summarise(part, ["model_id"]).iter_rows(named=True)}
        champion, baseline = registry.one("champion").id, registry.one("baseline").id
        by_region.append(
            {
                "region": name,
                "note": REGION_NOTES[name],
                "matches": int(pooled[champion]["matches"]),
                "log_loss": {m: round(pooled[m]["log_loss"], 4) for m in pooled},
                "gain_champion_over_baseline": round(
                    paired_gain(part, champion, baseline, [])["gain"][0], 4
                ),
                "gain_champion_over_anchor": round(
                    paired_gain(part, champion, rules["anchor"], [])["gain"][0], 4
                ),
            }
        )

    count_of = {row["week"]: row for row in counts.iter_rows(named=True)}
    inputs = {}
    if "inputs" in rules:
        means = rules["inputs"]["development_mean"]
        by_week = matches.group_by("week").agg(
            [pl.col(c).mean().alias(c) for c in means]
            + [pl.col(next(iter(means))).count().alias("n")]
        )
        lookup = {row["week"]: row for row in by_week.iter_rows(named=True)}
        for column, mean in means.items():
            inputs[column] = _clean(
                [
                    lookup[w][column] / mean - 1
                    if w in lookup and lookup[w]["n"] >= rules["inputs"]["minimum_matches"]
                    else np.nan
                    for w in weeks
                ]
            )
    alarms = [
        {k: row[k] for k in ("week", "family", "subject", "statistic", "value", "limit", "note")}
        for row in states.filter(pl.col("alarm")).sort("week").iter_rows(named=True)
    ]
    issue_log = folder / "replay_issues.jsonl"
    issue_actions = []
    if issue_log.exists():
        for line in issue_log.read_text().splitlines():
            record = json.loads(line)
            first = next((ln for ln in record["body"].splitlines() if ln.strip()), "")
            issue_actions.append(
                {"action": record["action"], "title": record["title"], "summary": first[:200]}
            )
    events = pl.read_parquet(folder / "patch_events.parquet")
    last_run = (
        json.loads((folder / "last_run.json").read_text())
        if (folder / "last_run.json").exists()
        else {}
    )
    lock = json.loads(layout.lock.read_text()) if layout.lock.exists() else None
    challengers = json.loads(layout.challengers.read_text()) if layout.challengers.exists() else []
    return {
        "built_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "weeks": weeks,
        "week_region": [region(week_bounds(w)[0] + 86400 * 3) for w in weeks],
        "models": models,
        "champion": registry.one("champion").id,
        "baseline": registry.one("baseline").id,
        "anchor": rules["anchor"],
        "regions": by_region,
        "inputs": inputs,
        "input_limit": rules.get("inputs", {}).get("relative_limit"),
        "input_alarm": _clean(
            [
                np.nan if v is None else v
                for v in state_values("inputs", "economy", "largest relative deviation")
            ]
        ),
        "usable_share": _clean(
            [np.nan if v is None else v for v in state_values("data", "pull", "usable share")]
        ),
        "judged_matches": [count_of.get(w, {}).get("judged") for w in weeks],
        "firing": {
            f"{family} / {subject}": bool(part["firing"][-1])
            for (family, subject), part in states.sort("week").group_by(
                ["family", "subject"], maintain_order=True
            )
        },
        "alarms": alarms,
        "issue_actions": issue_actions,
        "patches": [{"patch": p, "week": w, "matches": m} for p, w, m in events.iter_rows()],
        "registry": [
            {"id": e.id, "role": e.role, "kind": e.kind, "description": e.description}
            for e in registry.entries
        ],
        "challengers": challengers,
        "rules": {
            "arm": rules["performance"]["arm"],
            "window_weeks": window,
            "detectable_loss_increase": rules["performance"].get("detectable_loss_increase"),
            "minimum_matches": rules["minimum_matches"],
        },
        "gate": {
            k: gate[k]
            for k in ("shadow_weeks", "min_matches", "min_gain", "slope_low", "slope_high")
        },
        "last_run": {k: last_run.get(k) for k in ("now", "open_weeks", "final_weeks")},
        "lock": lock,
    }


def build(layout: Layout, out_dir: Path) -> Path:
    """Write ``index.html`` and ``data.json`` into `out_dir`."""
    data = build_data(layout)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    (out_dir / "data.json").write_text(json.dumps(data, indent=1) + "\n")
    page = TEMPLATE.read_text().replace("/*__DATA__*/null", payload)
    (out_dir / "index.html").write_text(page)
    return out_dir / "index.html"
