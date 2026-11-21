"""Pivot node M6b: an input alarm on the size of the economy, then write rules.yaml.

    uv run python scripts/m6b_input_alarm.py

Reads the development rows of ``monitoring/matches.parquet`` (no labels). The card of this
node is in the experiment log and was written before this script was run. If the alarm
passes its keep rule it is added to the rules; either way ``monitoring/rules.yaml`` is
written from the performance and calibration limits that node M6 measured.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m6_alert_rules import BUDGET, HORIZON, WEEKS, YEARS, detection, find_limit, rate, simulate  # noqa: E402

from vexil import alerts  # noqa: E402
from vexil.pipeline import ECONOMY_COLUMNS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FACTORS = (1.05, 1.10, 1.15)
GOLD = [i for i, name in enumerate(ECONOMY_COLUMNS) if "gold" in name]


def main() -> int:
    started = time.time()
    rng = np.random.default_rng(2026112102)
    dev = (
        pl.read_parquet(REPO / "monitoring" / "matches.parquet")
        .filter(pl.col("region") == "development")
        .sort("match_id")
    )
    weeks = sorted(dev["week"].unique())
    week_of = np.array([weeks.index(w) for w in dev["week"]])
    values = np.stack([dev[c].to_numpy().astype(float) for c in ECONOMY_COLUMNS])
    present = ~np.isnan(values)
    mean, sd = np.nanmean(values, axis=1), np.nanstd(values, axis=1, ddof=1)
    arrays = np.concatenate([np.where(present, values, 0.0), present.astype(float)])
    k = len(ECONOMY_COLUMNS)

    def worst_z(sim, factor=1.0):
        scale = np.ones(k)
        scale[GOLD] = factor
        sums, counts = sim[..., :k] * scale, sim[..., k:]
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (sums / counts - mean) / (sd / np.sqrt(counts))
        return np.nan_to_num(np.nanmax(np.abs(np.nan_to_num(z, nan=0.0)), axis=-1))

    real = np.stack([arrays[:, week_of == w].sum(1) for w in range(len(weeks))])
    with np.errstate(invalid="ignore", divide="ignore"):
        weekly_ratio = real[:, :k] / real[:, k:] / mean
    counts = np.bincount(week_of)
    sims = {v: simulate(v, arrays, week_of, dev["match_id"].to_numpy(), rng) for v in ("N1", "N2")}
    per_version = {
        v: find_limit(lambda h, v=v: worst_z(sims[v]) > h, BUDGET, high=10.0) for v in sims
    }
    limit = max(per_version.values())
    cross = {
        "false alarms per 52 weeks under N2 with the N1 limit": rate(
            worst_z(sims["N2"]) > per_version["N1"]
        ),
        "false alarms per 52 weeks under N1 with the N2 limit": rate(
            worst_z(sims["N1"]) > per_version["N2"]
        ),
    }
    start = rng.integers(16, 37, YEARS)
    planted = {}
    for factor in FACTORS:
        planted[f"gold totals x {factor}"] = {}
        for v, sim in sims.items():
            z = np.where(np.arange(WEEKS) >= start[:, None], worst_z(sim, factor), worst_z(sim))
            planted[f"gold totals x {factor}"][v] = detection(z > limit, start)
    found = min(r["found within 12 weeks"] for r in planted["gold totals x 1.15"].values())
    passed = max(cross.values()) <= 3 and found >= 0.8
    big = counts >= 20
    out = {
        "statistics": list(ECONOMY_COLUMNS),
        "matches": dev.height,
        "weeks": len(weeks),
        "development mean": dict(zip(ECONOMY_COLUMNS, mean.round(1).tolist(), strict=True)),
        "between-match sd": dict(zip(ECONOMY_COLUMNS, sd.round(1).tolist(), strict=True)),
        "weekly mean over the development mean, weeks with at least 20 matches (min, max)": {
            c: [
                round(float(np.nanmin(weekly_ratio[big, i])), 3),
                round(float(np.nanmax(weekly_ratio[big, i])), 3),
            ]
            for i, c in enumerate(ECONOMY_COLUMNS)
        },
        "largest standardised shift in the real weeks": [round(float(x), 2) for x in worst_z(real)],
        "limits per noise version": per_version,
        "limit": limit,
        **cross,
        "false alarms per 52 weeks at the final limit": {
            v: round(rate(worst_z(s) > limit), 3) for v, s in sims.items()
        },
        "planted cases": planted,
        f"keep rule passed (cross-version rate at most 3, x 1.15 found within {HORIZON} weeks in 80%)": bool(
            passed
        ),
        "seconds": round(time.time() - started, 1),
    }
    (REPO / "docs" / "evidence" / "m6b_input_alarm.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )

    # ---------------------------------------------------------------- pivot node M6c
    band_columns = [i for i, name in enumerate(ECONOMY_COLUMNS) if "kills" not in name]
    band_gold = [j for j, i in enumerate(band_columns) if i in GOLD]

    def worst_deviation(sim, factor=1.0):
        scale = np.ones(len(band_columns))
        scale[band_gold] = factor
        sums, counts = sim[..., band_columns] * scale, sim[..., [k + i for i in band_columns]]
        with np.errstate(invalid="ignore", divide="ignore"):
            deviation = np.abs(sums / counts / mean[band_columns] - 1)
        enough = sim[..., k + band_columns[0]] >= 20  # matches long enough for minute 10
        return np.where(
            enough, np.nan_to_num(np.nanmax(np.nan_to_num(deviation, nan=0.0), axis=-1)), 0.0
        )

    n2_limit = find_limit(lambda h: worst_deviation(sims["N2"]) > h, BUDGET, high=0.5)
    band_limit = 1.5 * n2_limit
    band = {
        "statistics": [ECONOMY_COLUMNS[i] for i in band_columns],
        "largest relative deviation in the real weeks": [
            round(float(x), 4) for x in worst_deviation(real)
        ],
        "limit under N2 (1 false alarm per 52 weeks)": n2_limit,
        "limit under N1, reported only": find_limit(
            lambda h: worst_deviation(sims["N1"]) > h, BUDGET, high=0.5
        ),
        "final limit (N2 limit x 1.5)": band_limit,
        "false alarms per 52 weeks at the final limit": {
            v: round(rate(worst_deviation(sim) > band_limit), 4) for v, sim in sims.items()
        },
        "planted cases": {},
    }
    for factor in FACTORS:
        band["planted cases"][f"gold totals x {factor}"] = {
            v: detection(
                np.where(
                    np.arange(WEEKS) >= start[:, None],
                    worst_deviation(sim, factor),
                    worst_deviation(sim),
                )
                > band_limit,
                start,
            )
            for v, sim in sims.items()
        }
    band_found = min(
        r["found within 12 weeks"] for r in band["planted cases"]["gold totals x 1.15"].values()
    )
    band_passed = (
        band_found >= 0.8
        and max(band["false alarms per 52 weeks at the final limit"].values()) <= 1
    )
    band["keep rule passed"] = bool(band_passed)
    (REPO / "docs" / "evidence" / "m6c_input_band.json").write_text(
        json.dumps(band, indent=1) + "\n"
    )
    print(json.dumps(band, indent=1))

    # rules.yaml from the M6 evidence, plus the input alarm if it passed
    m6 = json.loads((REPO / "docs" / "evidence" / "m6_alert_rules.json").read_text())
    rules: dict = {
        "version": 1,
        "window_weeks": alerts.WINDOW_WEEKS,
        "minimum_matches": 20,
        "anchor": "anchor",
        "performance": {"arm": m6["keep rule"]["kept arm"], "cusum_slack": 0.5, "models": {}},
        "calibration": {"models": {}},
    }
    for model_id, info in m6["models"].items():
        rules["performance"]["models"][model_id] = {
            "reference_gain": info["reference"]["gain"],
            "weekly_sd": info["weekly gain over the anchor, sd over full weeks"],
            "typical_rows": info["typical rows per week"],
            "limit": info["limits"][m6["keep rule"]["kept arm"]],
        }
        rules["calibration"]["models"][model_id] = {
            "reference_slope": info["reference"]["slope"],
            "slope_limit": info["limits"]["slope"],
            "reference_bias": info["reference"]["bias"],
            "bias_limit": info["limits"]["bias"],
        }
    rules["performance"]["detectable_loss_increase"] = m6["keep rule"][
        "smallest planted size found within 12 weeks in at least 80% (kept arm)"
    ]
    if band_passed:
        rules["inputs"] = {
            "relative_limit": band_limit,
            "minimum_matches": 20,
            "development_mean": {ECONOMY_COLUMNS[i]: float(mean[i]) for i in band_columns},
            "displayed_only": {
                "total_kills_20": float(mean[ECONOMY_COLUMNS.index("total_kills_20")])
            },
        }
    (REPO / "monitoring" / "rules.yaml").write_text(yaml.safe_dump(rules, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
