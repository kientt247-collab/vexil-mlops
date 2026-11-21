"""Node M6: set the alarm limits by simulation and measure what they can detect.

    uv run python scripts/m6_alert_rules.py --store _store

Reads the development region only: the out-of-fold predictions imported from phase 2 and,
for two planted cases, in-sample predictions of the frozen models on the same matches.
Writes ``monitoring/rules.yaml`` and ``docs/evidence/m6_alert_rules.json``. The card of
this node (arms, noise versions, planted cases, keep rule) is in the experiment log and
was written before this script was run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from vexil import alerts
from vexil.data.store import Store
from vexil.evaluate import add_groups, log_loss_rows
from vexil.labels import LabelLoader
from vexil.pipeline import WATCHED_INPUTS
from vexil.ratings import MatchMeta, RatingState
from vexil.registry import load_registry
from vexil.scoring import is_judged, match_inputs, score_match
from vexil.weeks import DEVELOPMENT_END, week_bounds

REPO = Path(__file__).resolve().parents[1]
YEARS, WEEKS, BLOCK = 20_000, 52, 4
MODELS = ("final-model", "rstar")
SIZES = (0.005, 0.010, 0.020, 0.030, 0.050)
DRAWS = 4  # noise draws kept per match and planted size
SLACK = 0.5
BUDGET = 1.0  # false alarms per 52 weeks, per model and family
HORIZON = 12
FREEZE_RATINGS_AT = "2026-03-02"  # a Monday; ratings stop moving from this week on
GOLD_SCALE = 1.15
GOLD_INPUTS = ("gold", "gold_adv_dec", "gold_adv_d5", "gold_rank1_diff")
INPUT_STATS = (*[column for column, _, _ in WATCHED_INPUTS], "abs_rating_input", "new_player_share")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- data
def load_development():
    """Per-match arrays for the out-of-fold matches, in one fixed order."""
    folder = REPO / "monitoring"
    scores = pl.read_parquet(folder / "match_scores_oof.parquet")
    matches = add_groups(
        pl.read_parquet(folder / "matches.parquet").filter(pl.col("region") == "development")
    )
    ids = np.array(sorted(scores["match_id"].unique()))
    index = {m: i for i, m in enumerate(ids)}
    meta = matches.filter(pl.col("match_id").is_in(ids)).sort("match_id")
    assert meta["match_id"].to_list() == ids.tolist()
    per_model = {}
    for model_id in (*MODELS, "anchor"):
        part = scores.filter(pl.col("model_id") == model_id).sort("match_id")
        per_model[model_id] = {
            k: part[k].to_numpy().astype(float) for k in ("n", "loss", "bias_sum", "cal_u", "cal_i")
        }
    rows = pl.read_parquet(folder / "oof_phase2.parquet").filter(pl.col("t") < 30)
    y_of = dict(zip(scores["match_id"], scores["y"], strict=False))
    row_data = {}
    for model_id in MODELS:
        part = rows.filter(pl.col("model_id") == model_id)
        p = np.clip(part["p"].to_numpy().astype(float), 1e-6, 1 - 1e-6)
        row_data[model_id] = {
            "match": np.array([index[m] for m in part["match_id"]]),
            "logit": np.log(p / (1 - p)),
            "y": np.array([y_of[m] for m in part["match_id"]], dtype=float),
        }
    return ids, meta, per_model, row_data, matches


def per_match(values, match, n_matches):
    return np.bincount(match, weights=values, minlength=n_matches)


def degraded_by_noise(rows, n_matches, clean_loss, n_rows, rng):
    """Loss sums per match after adding match-level noise to the log-odds, per planted size."""
    eps = rng.standard_normal((DRAWS, n_matches))
    out, sds = np.empty((len(SIZES), DRAWS, n_matches)), []

    def losses(sd):
        return np.stack(
            [
                per_match(
                    log_loss_rows(rows["y"], sigmoid(rows["logit"] + sd * e[rows["match"]])),
                    rows["match"],
                    n_matches,
                )
                for e in eps
            ]
        )

    for k, size in enumerate(SIZES):
        low, high = 0.0, 3.0
        for _ in range(30):
            mid = (low + high) / 2
            increase = (losses(mid).mean(0).sum() - clean_loss.sum()) / n_rows.sum()
            low, high = (mid, high) if increase < size else (low, mid)
        sds.append(high)
        out[k] = losses(high)
    return out, sds


def degraded_by_slope(rows, n_matches, factor=0.85):
    """Sums per match when the log-odds are divided by `factor` (an overconfident model)."""
    x = rows["logit"] / factor
    p = sigmoid(x)
    q = np.clip(p, 1e-6, 1 - 1e-6)
    xq = np.log(q / (1 - q))
    return {
        "loss": per_match(log_loss_rows(rows["y"], p), rows["match"], n_matches),
        "cal_u": per_match(xq * (rows["y"] - q), rows["match"], n_matches),
        "cal_i": per_match(xq**2 * q * (1 - q), rows["match"], n_matches),
        "bias_sum": per_match(p - rows["y"], rows["match"], n_matches),
    }


def in_sample_deltas(store, ids):
    """Loss change per match (rows 0-29) of the frozen models under two planted input faults:
    ratings frozen from a date on, and gold inputs scaled. Measured in-sample."""
    index = {m: i for i, m in enumerate(ids)}
    loader = LabelLoader(store, REPO / "_work" / "label_reads.log", REPO / "lock.json")
    entries = [e for e in load_registry(REPO / "models" / "registry.yaml").active()]
    live, frozen = RatingState(), RatingState()
    freeze_start = week_bounds(FREEZE_RATINGS_AT)[0]
    out = {
        case: {e.id: np.zeros(len(ids)) for e in entries}
        for case in ("frozen ratings", "gold scaled")
    }
    for week in store.weeks():
        if week_bounds(week)[0] >= DEVELOPMENT_END:
            break
        results = loader.week_labels(week, purpose="M6 planted input faults (development)")
        rows = list(store.read_week(week))
        metas = [MatchMeta.from_rows(m, p) for m, p in rows]
        live_inputs = live.process(metas, results)
        stale = {} if week_bounds(week)[0] >= freeze_start else results
        frozen_inputs = frozen.process(metas, stale)
        for match, players in rows:
            if match["match_id"] not in index or not is_judged(match):
                continue
            i = index[match["match_id"]]
            y = results[match["match_id"]]
            clean = match_inputs(match, players, live_inputs[match["match_id"]])
            early = clean["t"] < 30
            variants = {
                "frozen ratings": {
                    **clean,
                    "pelx": np.full_like(clean["pelx"], frozen_inputs[match["match_id"]]),
                },
                "gold scaled": {**clean, **{k: clean[k] * GOLD_SCALE for k in GOLD_INPUTS}},
            }
            base = {
                k: log_loss_rows(y, p[early]).sum()
                for k, p in score_match(tuple(entries), clean).items()
            }
            for case, inputs in variants.items():
                for k, p in score_match(tuple(entries), inputs).items():
                    out[case][k][i] = log_loss_rows(y, p[early]).sum() - base[k]
    return out


# ---------------------------------------------------------------- simulation
def simulate(version, arrays, unit_week, unit_group, rng, years=YEARS):
    """Weekly sums of every array for simulated years. Returns [years, 52, n_arrays].

    N1: each week draws whole match groups with replacement (as many as a random real week).
    N2: blocks of 4 consecutive real weeks, drawn with replacement.
    """
    full_weeks = np.unique(unit_week)[1:-1]  # the first and last development weeks are partial
    if version == "N2":
        real = np.stack([arrays[:, unit_week == w].sum(1) for w in full_weeks])  # [weeks, arrays]
        starts = rng.integers(0, len(full_weeks) - BLOCK + 1, (years, WEEKS // BLOCK))
        weeks = (starts[:, :, None] + np.arange(BLOCK)).reshape(years, WEEKS)
        return real[weeks]
    groups, group_of = np.unique(unit_group, return_inverse=True)
    by_group = np.stack([np.bincount(group_of, weights=a, minlength=len(groups)) for a in arrays])
    counts = np.array([len(np.unique(group_of[unit_week == w])) for w in full_weeks])
    out = np.empty((years * WEEKS, len(arrays)), dtype=np.float32)
    pvals = np.full(len(groups), 1 / len(groups))
    table = by_group.T.astype(np.float32)
    chunk = max(1_000, int(2e7 / len(groups)))
    for start in range(0, years * WEEKS, chunk):
        stop = min(start + chunk, years * WEEKS)
        k = rng.choice(counts, stop - start)
        out[start:stop] = rng.multinomial(k, pvals).astype(np.float32) @ table
    return out.reshape(years, WEEKS, len(arrays)).astype(float)


def rate(firing):
    """Mean number of alarms per simulated year."""
    return float(alerts.alarms(firing).sum(-1).mean())


def find_limit(firing_at, budget, low=0.0, high=1.0):
    """Smallest limit whose false-alarm rate is within the budget (rate falls as the limit grows)."""
    while rate(firing_at(high)) > budget:
        high *= 2
    for _ in range(40):
        mid = (low + high) / 2
        low, high = (mid, high) if rate(firing_at(mid)) > budget else (low, mid)
    return high


def detection(firing, start):
    """Share of years in which the rule fires within HORIZON weeks of the planted start,
    and the median number of weeks until it first fires (inf if it never does)."""
    week = np.arange(WEEKS)
    after = firing & (week >= start[:, None])
    first = np.where(after.any(1), after.argmax(1), np.inf)
    delay = first - start + 1
    return {
        "found within 12 weeks": round(float((delay <= HORIZON).mean()), 3),
        "median delay, weeks": float(np.median(delay)),
    }


def splice(clean, degraded, start):
    """Clean weekly values before the planted start week, degraded ones from it on."""
    return np.where(np.arange(WEEKS) >= start[:, None], degraded, clean)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=REPO / "_store")
    args = parser.parse_args()
    started = time.time()
    rng = np.random.default_rng(20261121)
    ids, meta, per_model, row_data, all_dev = load_development()
    n_matches = len(ids)
    week_names = sorted(meta["week"].unique())
    unit_week = np.array([week_names.index(w) for w in meta["week"]])
    unit_group = meta["group"].to_numpy()
    n_rows = per_model["anchor"]["n"]
    deltas = in_sample_deltas(Store(args.store, REPO / "data" / "manifest.json"), ids)

    out: dict = {
        "development matches": n_matches,
        "match groups": len(set(unit_group)),
        "weeks": [week_names[0], week_names[-1]],
        "full weeks used": len(week_names) - 2,
        "simulated years per noise version": YEARS,
        "models": {},
    }
    rules: dict = {
        "version": 1,
        "window_weeks": alerts.WINDOW_WEEKS,
        "minimum_matches": 20,
        "performance": {},
        "calibration": {},
    }
    hard_stop = []

    for model_id in MODELS:
        own, anchor = per_model[model_id], per_model["anchor"]
        gain_sum = anchor["loss"] - own["loss"]  # per match: rows * (anchor loss - model loss)
        noisy, noise_sds = degraded_by_noise(
            row_data[model_id], n_matches, own["loss"], n_rows, rng
        )
        slope_case = degraded_by_slope(row_data[model_id], n_matches)
        # arrays, in one fixed order
        names = ["n", "gain", "cal_u", "cal_i", "bias"]
        stack = [n_rows, gain_sum, own["cal_u"], own["cal_i"], own["bias_sum"]]
        for k, size in enumerate(SIZES):
            for j in range(DRAWS):
                names.append(f"gain|{size}|{j}")
                stack.append(anchor["loss"] - noisy[k, j])
        names += ["gain|slope", "cal_u|slope", "cal_i|slope", "bias|slope"]
        stack += [
            anchor["loss"] - slope_case["loss"],
            slope_case["cal_u"],
            slope_case["cal_i"],
            slope_case["bias_sum"],
        ]
        for case in ("frozen ratings", "gold scaled"):
            names.append(f"gain|{case}")
            stack.append(gain_sum - (deltas[case][model_id] - deltas[case]["anchor"]))
        arrays = np.stack(stack)
        col = {name: i for i, name in enumerate(names)}

        full = np.isin(unit_week, np.unique(unit_week)[1:-1])
        reference = {
            "gain": float(gain_sum.sum() / n_rows.sum()),
            "slope": float(alerts.one_step_slope(own["cal_u"].sum(), own["cal_i"].sum())),
            "bias": float(own["bias_sum"].sum() / n_rows.sum()),
        }
        weekly_real = np.array(
            [
                gain_sum[unit_week == w].sum() / n_rows[unit_week == w].sum()
                for w in np.unique(unit_week[full])
            ]
        )
        weekly_sd = float(weekly_real.std(ddof=1))
        typical_rows = float(
            np.mean([n_rows[unit_week == w].sum() for w in np.unique(unit_week[full])])
        )

        sims = {v: simulate(v, arrays, unit_week, unit_group, rng) for v in ("N1", "N2")}

        def stats(sim, gain_name="gain", suffix=""):
            n = sim[..., col["n"]]
            gain_week = alerts.ratio(sim[..., col[gain_name]], n)
            gain_roll = alerts.ratio(
                alerts.rolling_sum(sim[..., col[gain_name]]), alerts.rolling_sum(n)
            )
            shortfall = alerts.standardised_shortfall(
                gain_week, n, reference["gain"], weekly_sd, typical_rows
            )
            slope = alerts.one_step_slope(
                alerts.rolling_sum(sim[..., col["cal_u" + suffix]]),
                alerts.rolling_sum(sim[..., col["cal_i" + suffix]]),
            )
            bias = alerts.ratio(
                alerts.rolling_sum(sim[..., col["bias" + suffix]]), alerts.rolling_sum(n)
            )
            return {
                "week": gain_week,
                "roll": gain_roll,
                "shortfall": shortfall,
                "slope": slope,
                "bias": bias,
            }

        null = {v: stats(sims[v]) for v in sims}
        rule_of = {
            "D": lambda s, h: alerts.below(s["week"], reference["gain"], h),
            "A": lambda s, h: alerts.below(s["roll"], reference["gain"], h),
            "B": lambda s, h: alerts.cusum(s["shortfall"], SLACK, h),
            "slope": lambda s, h: alerts.outside(s["slope"], reference["slope"], h),
            "bias": lambda s, h: alerts.outside(s["bias"], reference["bias"], h),
        }
        budgets = {
            "D": BUDGET,
            "A": BUDGET,
            "B": BUDGET,
            "A half": BUDGET / 2,
            "B half": BUDGET / 2,
            "slope": BUDGET / 2,
            "bias": BUDGET / 2,
        }
        limits, cross = {}, {}
        for name, budget in budgets.items():
            rule = rule_of[name.split(" ")[0]]
            high = 50.0 if name.startswith("B") else 1.0
            per_version = {
                v: find_limit(lambda h, v=v: rule(null[v], h), budget, high=high) for v in null
            }
            limits[name] = max(per_version.values())
            cross[name] = {
                "limit from N1": per_version["N1"],
                "limit from N2": per_version["N2"],
                "false alarms per year under N2 with the N1 limit": rate(
                    rule(null["N2"], per_version["N1"])
                ),
                "false alarms per year under N1 with the N2 limit": rate(
                    rule(null["N1"], per_version["N2"])
                ),
            }
            worst = max(
                cross[name]["false alarms per year under N2 with the N1 limit"],
                cross[name]["false alarms per year under N1 with the N2 limit"],
            )
            if worst > 3:
                hard_stop.append(
                    f"{model_id} {name}: {worst:.2f} false alarms per year across noise versions"
                )

        def firing(arm, s):
            if arm == "C":
                return rule_of["A"](s, limits["A half"]) | rule_of["B"](s, limits["B half"])
            return rule_of[arm](s, limits[arm])

        false_alarms = {
            arm: {v: round(rate(firing(arm, null[v])), 3) for v in null} for arm in "DABC"
        }
        false_alarms["calibration"] = {
            v: round(
                rate(
                    rule_of["slope"](null[v], limits["slope"])
                    | rule_of["bias"](null[v], limits["bias"])
                ),
                3,
            )
            for v in null
        }

        # planted cases
        found: dict = {}
        start = rng.integers(16, 37, YEARS)
        for v, sim in sims.items():
            found[v] = {}
            cases = {f"loss +{size}": [f"gain|{size}|{j}" for j in range(DRAWS)] for size in SIZES}
            cases.update(
                {
                    "slope 0.85": ["gain|slope"],
                    "frozen ratings": ["gain|frozen ratings"],
                    "gold scaled": ["gain|gold scaled"],
                }
            )
            for case, gain_names in cases.items():
                pick = rng.integers(0, len(gain_names), YEARS)
                planted = sim.copy()
                degraded_gain = np.choose(pick[:, None], [sim[..., col[g]] for g in gain_names])
                planted[..., col["gain"]] = splice(sim[..., col["gain"]], degraded_gain, start)
                if case == "slope 0.85":
                    for k in ("cal_u", "cal_i", "bias"):
                        planted[..., col[k]] = splice(
                            sim[..., col[k]], sim[..., col[k + "|slope"]], start
                        )
                s = stats(planted)
                found[v][case] = {arm: detection(firing(arm, s), start) for arm in "DABC"}
                found[v][case]["calibration"] = detection(
                    rule_of["slope"](s, limits["slope"]) | rule_of["bias"](s, limits["bias"]), start
                )

        # keep rule: shortest median delay for loss +0.010 on the final model, worse of the two versions
        out["models"][model_id] = {
            "reference": reference,
            "weekly gain over the anchor, sd over full weeks": weekly_sd,
            "typical rows per week": typical_rows,
            "noise sd on the log-odds per planted size": dict(
                zip(map(str, SIZES), noise_sds, strict=True)
            ),
            "limits": limits,
            "limits per noise version": cross,
            "false alarms per 52 weeks at the final limits": false_alarms,
            "planted cases": found,
        }
        rules["calibration"][model_id] = {
            "reference_slope": reference["slope"], "slope_limit": limits["slope"],
            "reference_bias": reference["bias"], "bias_limit": limits["bias"],
        }  # fmt: skip
        rules["performance"][model_id] = {
            "reference_gain": reference["gain"], "weekly_sd": weekly_sd, "typical_rows": typical_rows,
            "limits": {"D": limits["D"], "A": limits["A"], "B": limits["B"], "A_half": limits["A half"], "B_half": limits["B half"]},
        }  # fmt: skip

    # choose the arm on the final model, case loss +0.010
    final = out["models"]["final-model"]["planted cases"]
    worst_delay = {
        arm: max(final[v]["loss +0.01"][arm]["median delay, weeks"] for v in final)
        for arm in "DABC"
    }
    worst_found = {
        arm: min(final[v]["loss +0.01"][arm]["found within 12 weeks"] for v in final)
        for arm in "DABC"
    }
    best = min(worst_delay.values())
    kept = next(arm for arm in "DABC" if worst_delay[arm] <= best + 1)
    detectable = next(
        (
            size
            for size in SIZES
            if min(final[v][f"loss +{size}"][kept]["found within 12 weeks"] for v in final) >= 0.8
        ),
        None,
    )
    out["keep rule"] = {
        "median delay for loss +0.010, worse noise version": worst_delay,
        "found within 12 weeks for loss +0.010, worse noise version": worst_found,
        "kept arm": kept,
        "target met (0.010 found within 12 weeks in at least 80%)": bool(worst_found[kept] >= 0.8),
        "smallest planted size found within 12 weeks in at least 80% (kept arm)": detectable,
    }
    rules["performance"]["arm"] = kept
    rules["performance"]["cusum_slack"] = SLACK
    rules["performance"]["detectable_loss_increase"] = detectable

    # ------------------------------------------------------------ input alarm
    dev = all_dev.with_columns(abs_rating_input=pl.col("rating_input").abs()).sort("match_id")
    dev_weeks = sorted(dev["week"].unique())
    week_of = np.array([dev_weeks.index(w) for w in dev["week"]])
    values = np.stack([dev[s].to_numpy().astype(float) for s in INPUT_STATS])
    present = ~np.isnan(values)
    mean = np.nanmean(values, axis=1)
    sd = np.nanstd(values, axis=1, ddof=1)
    filled = np.where(present, values, 0.0)
    arrays = np.concatenate([filled, present.astype(float)])
    k = len(INPUT_STATS)
    scale = np.ones(k)
    scale[[INPUT_STATS.index("abs_gold_10"), INPUT_STATS.index("abs_gold_20")]] = GOLD_SCALE

    def worst_z(sim, factor=None):
        sums, counts = sim[..., :k], sim[..., k:]
        if factor is not None:
            sums = sums * factor
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (sums / counts - mean) / (sd / np.sqrt(counts))
        return np.nanmax(np.abs(z), axis=-1)

    real_z = worst_z(np.stack([arrays[:, week_of == w].sum(1) for w in range(len(dev_weeks))]))
    input_sims = {
        v: simulate(v, arrays, week_of, dev["match_id"].to_numpy(), rng) for v in ("N1", "N2")
    }
    per_version = {
        v: find_limit(lambda h, v=v: worst_z(input_sims[v]) > h, BUDGET, high=10.0)
        for v in input_sims
    }
    input_limit = max(per_version.values())
    start = rng.integers(16, 37, YEARS)
    gold_case = {}
    for v, sim in input_sims.items():
        z = np.where(np.arange(WEEKS) >= start[:, None], worst_z(sim, scale), worst_z(sim))
        gold_case[v] = detection(z > input_limit, start)
    cross_rates = {
        "false alarms per year under N2 with the N1 limit": rate(
            worst_z(input_sims["N2"]) > per_version["N1"]
        ),
        "false alarms per year under N1 with the N2 limit": rate(
            worst_z(input_sims["N1"]) > per_version["N2"]
        ),
    }
    if max(cross_rates.values()) > 3:
        hard_stop.append(
            f"input alarm: {max(cross_rates.values()):.2f} false alarms per year across noise versions"
        )
    out["input alarm"] = {
        "statistics": list(INPUT_STATS),
        "matches": dev.height,
        "weeks": len(dev_weeks),
        "limit on the largest standardised weekly shift": input_limit,
        "limits per noise version": per_version,
        **cross_rates,
        "largest standardised shift in the real development weeks": [
            round(float(x), 2) for x in real_z
        ],
        "planted case, gold inputs x 1.15": gold_case,
    }
    rules["inputs"] = {
        "limit": input_limit,
        "statistics": {
            s: {"mean": float(m), "sd": float(d)}
            for s, m, d in zip(INPUT_STATS, mean, sd, strict=True)
        },
    }
    out["hard stop"] = hard_stop
    out["seconds"] = round(time.time() - started, 1)
    (REPO / "docs" / "evidence" / "m6_alert_rules.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )
    if not hard_stop:
        (REPO / "monitoring" / "rules.yaml").write_text(yaml.safe_dump(rules, sort_keys=False))
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
