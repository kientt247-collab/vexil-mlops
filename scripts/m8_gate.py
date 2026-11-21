"""Node M8: measure the promotion gate on the development region and write gate.yaml.

    uv run python scripts/m8_gate.py --store _store

The card of this node (challenger kinds, gate arms, placebo, power, keep rule) is in the
experiment log and was written before this script was run. Labels are read through the
loader, development weeks only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from vexil.data.store import Store
from vexil.evaluate import add_groups, log_loss_rows
from vexil.features.a1 import A1_INPUTS
from vexil.labels import LabelLoader
from vexil.matchdata import is_usable
from vexil.models.fit import fit_recalibration, fit_ridge_a1
from vexil.models.linear import RidgeA1Model, load_model
from vexil.pipeline import pair_key
from vexil.ratings import MatchMeta, RatingState
from vexil.retrain import GATE_DEFAULTS, gate
from vexil.scoring import JUDGED_TIERS, is_judged, match_inputs
from vexil.weeks import DEVELOPMENT_END, WEEK_SECONDS, week_bounds

REPO = Path(__file__).resolve().parents[1]
RECIPE = {"added": ("pelx",), "c_value": 0.03}
TRIGGERS = ("2026-03-02", "2026-03-24", "2026-03-30", "2026-04-27", "2026-05-25")
ARMS = (8, 12)
PLACEBOS = 20
RECALIBRATION_WEEKS = 12
MIN_TRAINING_MATCHES = 1500
DAY = 86400


def timestamp(date: str) -> int:
    return int(dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=dt.UTC).timestamp())


def build_table(store: Store, loader: LabelLoader):
    """Minute rows of every usable development match of the judged tiers, with outcomes."""
    state = RatingState()
    columns = {name: [] for name in ("t", "pelx", *A1_INPUTS)}
    row_match, facts = [], []
    for week in store.weeks():
        if week_bounds(week)[0] >= DEVELOPMENT_END:
            break
        results = loader.week_labels(week, purpose="M8 training table (development)")
        rows = list(store.read_week(week))
        metas = {m["match_id"]: MatchMeta.from_rows(m, p) for m, p in rows}
        ratings = state.process(list(metas.values()), results)
        for match, players in rows:
            usable_tier = is_usable(match) and match.get("league_tier") in JUDGED_TIERS
            if not usable_tier or match["start_time"] >= DEVELOPMENT_END:
                continue
            inputs = match_inputs(match, players, ratings[match["match_id"]])
            for name in columns:
                columns[name].append(inputs[name])
            row_match.append(np.full(len(inputs["t"]), len(facts)))
            facts.append(
                {
                    "index": len(facts),
                    "match_id": match["match_id"],
                    "start_time": match["start_time"],
                    "judged": is_judged(match),
                    "y": results[match["match_id"]],
                    "pair": pair_key(match, metas[match["match_id"]]),
                }
            )
    table = {name: np.concatenate(parts) for name, parts in columns.items()}
    table["match"] = np.concatenate(row_match)
    # add_groups reorders the matches; "index" restores the order the rows were built in
    facts = add_groups(pl.DataFrame(facts)).sort("index")
    return table, facts


def model_from(spec: dict) -> RidgeA1Model:
    return RidgeA1Model(
        tuple(spec["features"]), tuple(spec["added"]), spec["mu_u"], spec["sd_u"],
        np.array(spec["mu"]), np.array(spec["sd"]), np.array(spec["coef"]), spec["intercept"],
    )  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=REPO / "_store")
    args = parser.parse_args()
    started = time.time()
    rng = np.random.default_rng(2026112108)
    store = Store(args.store, REPO / "data" / "manifest.json")
    loader = LabelLoader(store, REPO / "_work" / "label_reads.log", REPO / "lock.json")
    table, facts = build_table(store, loader)
    n_matches = facts.height
    start = facts["start_time"].to_numpy()
    judged = facts["judged"].to_numpy()
    y_match = facts["y"].to_numpy()
    groups = facts["group"].to_numpy()
    row_match = table["match"]
    out: dict = {
        "development matches of the judged tiers": n_matches,
        "minute rows": int(len(row_match)),
    }
    return_code = run(out, table, facts, start, judged, y_match, groups, row_match, rng)
    out["seconds"] = round(time.time() - started, 1)
    (REPO / "docs" / "evidence" / "m8_gate.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return return_code


def run(out, table, facts, start, judged, y_match, groups, row_match, rng) -> int:
    y_rows = y_match[row_match]
    early = table["t"] < 30

    def rows_of(match_mask):
        return match_mask[row_match]

    def fit(match_mask):
        rows = rows_of(match_mask)
        return model_from(
            fit_ridge_a1({k: v[rows] for k, v in table.items()}, y_rows[rows], **RECIPE)
        )

    def losses(p, match_mask):
        """Per shadow match: loss sum over minutes 0-29 and the number of those rows."""
        rows = rows_of(match_mask) & early
        index = np.flatnonzero(match_mask)
        position = np.full(len(match_mask), -1)
        position[index] = np.arange(len(index))
        where = position[row_match[rows]]
        loss = np.bincount(
            where, weights=log_loss_rows(y_rows[rows], p[rows]), minlength=len(index)
        )
        n = np.bincount(where, minlength=len(index)).astype(float)
        return loss, n, rows

    def predict(model, match_mask):
        rows = rows_of(match_mask)
        p = np.full(len(row_match), np.nan)
        p[rows] = model.predict({k: v[rows] for k, v in table.items()})
        return p

    # proof of done: the fitting code reproduces candidate.json on the phase 2 fitting rows
    everything = np.ones(len(start), bool)
    refit = fit(everything)
    frozen = load_model(REPO / "models" / "candidate.json")
    out["proof of done: refit on all development rows against candidate.json"] = {
        "fitting rows": int(len(row_match)),
        "fitting rows of candidate.json": 170519,
        "largest coefficient difference": float(np.abs(refit.coef - frozen.coef).max()),
        "largest difference in predictions on all rows": float(
            np.abs(refit.predict(table) - frozen.predict(table)).max()
        ),
        "note": "phase 2 fitted with the rating stored as 32-bit floats; here it is 64-bit",
    }

    placebo = {arm: [] for arm in ARMS}
    placebo_gains = {arm: [] for arm in ARMS}
    per_date = {}
    for date in TRIGGERS:
        trigger = timestamp(date)
        train = start <= trigger - DAY
        info = {"training matches": int(train.sum())}
        if train.sum() < MIN_TRAINING_MATCHES:
            info["skipped"] = "fewer than 1,500 training matches"
            per_date[date] = info
            continue
        champion = fit(train)
        train_groups = np.unique(groups[train])
        for arm in ARMS:
            end = trigger + arm * WEEK_SECONDS
            if end > DEVELOPMENT_END:
                info[f"{arm} weeks"] = "skipped: shadow period would pass 2026-07-30"
                continue
            shadow = judged & (start >= trigger) & (start < end)
            champion_p = predict(champion, shadow)
            champion_loss, n, rows = losses(champion_p, shadow)
            passes, gains = 0, []
            for _ in range(PLACEBOS):
                kept = rng.choice(train_groups, int(0.8 * len(train_groups)), replace=False)
                challenger = fit(train & np.isin(groups, kept))
                p = predict(challenger, shadow)
                loss, _, _ = losses(p, shadow)
                result = gate(
                    champion_loss,
                    loss,
                    n,
                    groups[shadow],
                    p[rows],
                    y_rows[rows],
                    seed=int(rng.integers(1 << 30)),
                )
                passes += result.passed
                gains.append(result.gain)
            placebo[arm].append(passes)
            placebo_gains[arm] += gains
            info[f"{arm} weeks"] = {
                "shadow matches": int(shadow.sum()),
                "placebo challengers passing": f"{passes} of {PLACEBOS}",
                "placebo gain, min / median / max": [
                    round(float(x), 5) for x in (min(gains), np.median(gains), max(gains))
                ],
            }
        per_date[date] = info
    out["placebo by trigger date"] = per_date
    false_rate = {arm: sum(placebo[arm]) / (PLACEBOS * len(placebo[arm])) for arm in ARMS}
    out["false-promotion rate"] = {
        f"{arm} weeks": {"rate": false_rate[arm], "trials": PLACEBOS * len(placebo[arm])}
        for arm in ARMS
    }

    # power: an overconfident champion (out-of-fold log-odds / 0.85) against K2
    oof = pl.read_parquet(REPO / "monitoring" / "oof_phase2.parquet").filter(
        (pl.col("model_id") == "final-model") & (pl.col("t") < 30)
    )
    fact_of = {m: i for i, m in enumerate(facts["match_id"])}
    oof_match = np.array([fact_of[m] for m in oof["match_id"]])
    p = np.clip(oof["p"].to_numpy().astype(float), 1e-6, 1 - 1e-6)
    bad_logit = np.log(p / (1 - p)) / 0.85
    oof_y = y_match[oof_match]
    oof_start = start[oof_match]
    power = {arm: [] for arm in ARMS}
    power_detail = {}
    for date in TRIGGERS:
        trigger = timestamp(date)
        before = (oof_start >= trigger - RECALIBRATION_WEEKS * WEEK_SECONDS) & (oof_start < trigger)
        layer = fit_recalibration(bad_logit[before], oof_y[before])
        power_detail[date] = {
            "fitted slope": round(layer["slope"], 3),
            "fitting rows": layer["n_rows"],
        }
        for arm in ARMS:
            end = trigger + arm * WEEK_SECONDS
            if end > DEVELOPMENT_END:
                continue
            shadow_rows = (oof_start >= trigger) & (oof_start < end)
            ids, where = np.unique(oof_match[shadow_rows], return_inverse=True)
            champion_p = 1 / (1 + np.exp(-bad_logit[shadow_rows]))
            challenger_p = 1 / (
                1 + np.exp(-(layer["shift"] + layer["slope"] * bad_logit[shadow_rows]))
            )
            ys = oof_y[shadow_rows]
            champion_loss = np.bincount(where, weights=log_loss_rows(ys, champion_p))
            challenger_loss = np.bincount(where, weights=log_loss_rows(ys, challenger_p))
            n = np.bincount(where).astype(float)
            result = gate(champion_loss, challenger_loss, n, groups[ids], challenger_p, ys)
            power[arm].append(result.passed)
            power_detail[date][f"{arm} weeks"] = {
                "passed": result.passed, "gain": round(result.gain, 5),
                "interval": [round(x, 5) for x in result.interval], "slope": round(result.slope, 3),
                "matches": result.matches, "reasons": result.reasons,
            }  # fmt: skip
    out["power: overconfident champion against a recalibration challenger"] = {
        "by trigger date": power_detail,
        "passed": {f"{arm} weeks": f"{sum(power[arm])} of {len(power[arm])}" for arm in ARMS},
    }

    # information: real refits against a champion fitted at the first trigger date
    first = timestamp(TRIGGERS[0])
    old_champion = fit(start <= first - DAY)
    real = {}
    for date in TRIGGERS[2:]:
        trigger = timestamp(date)
        challenger = fit(start <= trigger - DAY)
        shadow = judged & (start >= trigger) & (start < trigger + 8 * WEEK_SECONDS)
        champion_p, challenger_p = predict(old_champion, shadow), predict(challenger, shadow)
        champion_loss, n, rows = losses(champion_p, shadow)
        challenger_loss, _, _ = losses(challenger_p, shadow)
        result = gate(
            champion_loss, challenger_loss, n, groups[shadow], challenger_p[rows], y_rows[rows]
        )
        real[date] = {
            "passed": result.passed, "gain": round(result.gain, 5),
            "interval": [round(x, 5) for x in result.interval], "slope": round(result.slope, 3),
            "matches": result.matches, "reasons": result.reasons,
        }  # fmt: skip
    out["information: real refits against the champion fitted at 2026-03-02, 8-week shadow"] = real

    # keep rule
    passing = [arm for arm in ARMS if false_rate[arm] <= 0.05]
    share = {arm: sum(power[arm]) / max(len(power[arm]), 1) for arm in ARMS}
    settings = dict(GATE_DEFAULTS)
    if passing:
        best = max(share[arm] for arm in passing)
        kept = next(arm for arm in passing if share[arm] >= best - 0.05)
    else:
        kept = 12
        gains = np.sort(placebo_gains[12])
        settings["min_gain"] = float(np.ceil(np.quantile(gains, 0.95) * 1e4) / 1e4)
    out["keep rule"] = {
        "arms within the 5% line": passing,
        "power share": share,
        "kept shadow weeks": kept,
        "gate": settings,
    }
    gate_file = {
        "version": 1,
        "shadow_weeks": kept,
        **settings,
        "bootstrap_draws": 2000,
        "recalibration_weeks": RECALIBRATION_WEEKS,
        "refit_recipe": {"added": list(RECIPE["added"]), "c_value": RECIPE["c_value"]},
        "new_patch_matches": 300,
    }
    (REPO / "monitoring" / "gate.yaml").write_text(yaml.safe_dump(gate_file, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
