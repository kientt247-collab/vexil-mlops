"""Baseline of the loop: the frozen final model of Vexil phase 2 (candidate.json), rewritten against `ctx`.

Ridge logistic regression (L2, C = 0.03) on [100 x u, z(A1 features), z(f), z(f x u)], where u = standardised ln(1 + t) and
f = the player rating input "pelx": Elo of the ten players (start 1500, K = 32, K = 64 for a player's first 20 rated matches),
results of every development match applied in the order matches end, the value for a match uses only matches that ended
strictly before it started; f = (mean rating Radiant - mean rating Dire) x ln(10) / 400.

Interface:  fit(ctx, train_match_ids, seed) -> state ;  predict(state, ctx, rows) -> numpy array of P(Radiant wins).
"""
import math

import numpy as np
from sklearn.linear_model import LogisticRegression

A1 = ["kill_diff", "towers_diff", "xp_adv_pm", "gold_adv_dec", "aegis_team", "gold_adv_d5", "gold_rank1_diff", "rax_lanes_diff",
      "roshan_diff", "kill_diff_last3"]
C_RIDGE = 0.03
K, K_NEW, N_NEW = 32.0, 64.0, 20
SCALE = math.log(10) / 400
HS = 0.10


def _sides(m):
    ra, da = m["ra"], m["da"]
    return (ra, da) if len(ra) == 5 and len(da) == 5 and all(ra) and all(da) else None


def _apply(rating, games, m, result):
    sides = _sides(m)
    if sides is None:
        return
    rad, dire = sides
    r = sum(rating.get(a, 1500.0) for a in rad) / 5
    d = sum(rating.get(a, 1500.0) for a in dire) / 5
    surprise = result - 1.0 / (1.0 + 10 ** ((d - r) / 400))
    for sign, side in ((1.0, rad), (-1.0, dire)):
        for a in side:
            kk = K_NEW if games.get(a, 0) < N_NEW else K
            rating[a] = rating.get(a, 1500.0) + sign * kk * surprise
            games[a] = games.get(a, 0) + 1


def _read(rating, ra, da):
    side = lambda acc: float(np.mean([rating.get(a, 1500.0) if a else 1500.0 for a in acc])) if acc else 1500.0
    return (side(ra) - side(da)) * SCALE


def rating_input(results, queries):
    """results: dicts sorted by end time (as ctx.results_before returns them); queries: (match_id, start, ra, da).
    Returns {match_id: input}. A query reads the ratings after every result that ended strictly before its start."""
    rating, games, out, i = {}, {}, {}, 0
    for mid, start, ra, da in sorted(queries, key=lambda q: (q[1], q[0])):
        while i < len(results) and results[i]["end"] < start:
            _apply(rating, games, results[i], results[i]["radiant_win"])
            i += 1
        out[mid] = _read(rating, ra, da)
    return out


def _inputs(ctx, rows):
    ids = sorted(set(int(m) for m in rows["match_id"]))
    pre = {m: ctx.pregame(m) for m in ids}
    f = rating_input(ctx.results_before(ctx.now), [(m, pre[m]["start"], pre[m]["ra"], pre[m]["da"]) for m in ids])
    return np.array([f[int(m)] for m in rows["match_id"]]), [(pre[int(m)]["rh"], pre[int(m)]["dh"]) for m in rows["match_id"]]


def _heroes(rows, picks, heroes):
    idx = {h: i for i, h in enumerate(heroes)}
    H = np.zeros((len(rows), len(heroes)))
    for r, (rh, dh) in enumerate(picks):
        for h in rh or []:
            if h in idx:
                H[r, idx[h]] += 1.0
        for h in dh or []:
            if h in idx:
                H[r, idx[h]] -= 1.0
    return H * (HS * np.ones(len(rows)))[:, None]


def _design(rows, f, mu_u, sd_u):
    u = (np.log1p(rows["t"].to_numpy(float)) - mu_u) / sd_u
    a = rows[A1].to_numpy(float)
    x = np.column_stack([a, np.sign(a) * np.sqrt(np.abs(a)), f, f * u, a * u[:, None]])
    return u, x


def fit(ctx, train_match_ids, seed):
    rows = ctx.rows
    y = np.array([ctx.label(m) for m in rows["match_id"]])
    f, picks = _inputs(ctx, rows)
    heroes = sorted({h for rh, dh in picks for h in list(rh or []) + list(dh or []) if h})
    logt = np.log1p(rows["t"].to_numpy(float))
    mu_u, sd_u = float(logt.mean()), float(logt.std())
    u, x = _design(rows, f, mu_u, sd_u)
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1
    clf = LogisticRegression(C=C_RIDGE, solver="newton-cholesky", max_iter=1000, tol=1e-8).fit(np.column_stack([100 * u, (x - mu) / sd, _heroes(rows, picks, heroes)]), y)
    return {"heroes": heroes, "mu_u": mu_u, "sd_u": sd_u, "mu": mu, "sd": sd, "coef": clf.coef_[0], "intercept": float(clf.intercept_[0])}


def predict(state, ctx, rows):
    f, picks = _inputs(ctx, rows)
    u, x = _design(rows, f, state["mu_u"], state["sd_u"])
    z = np.column_stack([100 * u, (x - state["mu"]) / state["sd"], _heroes(rows, picks, state["heroes"])]) @ state["coef"] + state["intercept"]
    return 1.0 / (1.0 + np.exp(-z))
