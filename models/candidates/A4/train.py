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
COLS = A1 + ["lh_diff", "kd60", "kd120", "kd300", "t1_diff", "t2_diff", "t3_diff", "t4_diff",
             "rax_count_diff", "aegis_active", "fb_team", "courier_diff", "tormentor_diff", "xp_adv_d5", "gold_rank2_diff",
             "gold_rank45_diff", "top3_share_diff", "xp_min_diff", "xp_adv_dec", "item_nw_diff"]
C_RIDGE = 0.03
K_HERO = 16.0
K, K_NEW, N_NEW = 32.0, 64.0, 20
SCALE = math.log(10) / 400


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


def _apply_hero(hero, m, result):
    rh, dh = m.get("rh"), m.get("dh")
    if not rh or not dh:
        return
    r = sum(hero.get(h, 1500.0) for h in rh) / len(rh)
    d = sum(hero.get(h, 1500.0) for h in dh) / len(dh)
    surprise = result - 1.0 / (1.0 + 10 ** ((d - r) / 400))
    for sign, side in ((1.0, rh), (-1.0, dh)):
        for h in side:
            hero[h] = hero.get(h, 1500.0) + sign * K_HERO * surprise


def _read_hero(hero, rh, dh):
    if not rh or not dh:
        return 0.0
    return (np.mean([hero.get(h, 1500.0) for h in rh]) - np.mean([hero.get(h, 1500.0) for h in dh])) * SCALE


def _read(rating, ra, da):
    side = lambda acc: float(np.mean([rating.get(a, 1500.0) if a else 1500.0 for a in acc])) if acc else 1500.0
    return (side(ra) - side(da)) * SCALE


def rating_input(results, queries):
    """results: dicts sorted by end time (as ctx.results_before returns them); queries: (match_id, start, ra, da).
    Returns {match_id: input}. A query reads the ratings after every result that ended strictly before its start."""
    rating, games, hero, out, i = {}, {}, {}, {}, 0
    for mid, start, ra, da, rh, dh in sorted(queries, key=lambda q: (q[1], q[0])):
        while i < len(results) and results[i]["end"] < start:
            _apply(rating, games, results[i], results[i]["radiant_win"])
            _apply_hero(hero, results[i], results[i]["radiant_win"])
            i += 1
        out[mid] = (_read(rating, ra, da), _read_hero(hero, rh, dh))
    return out


def _inputs(ctx, rows):
    ids = sorted(set(int(m) for m in rows["match_id"]))
    pre = {m: ctx.pregame(m) for m in ids}
    f = rating_input(ctx.results_before(ctx.now), [(m, pre[m]["start"], pre[m]["ra"], pre[m]["da"], pre[m]["rh"], pre[m]["dh"]) for m in ids])
    return np.array([f[int(m)] for m in rows["match_id"]])  # (n, 2): player input, hero input


def _design(rows, f, mu_u, sd_u):
    u = (np.log1p(rows["t"].to_numpy(float)) - mu_u) / sd_u
    g = rows[COLS].to_numpy(float)
    x = np.column_stack([g, g * u[:, None], f, f[:, :1] * u[:, None], f[:, 1:] * u[:, None]])
    return u, x


def fit(ctx, train_match_ids, seed):
    rows = ctx.rows
    y = np.array([ctx.label(m) for m in rows["match_id"]])
    f = _inputs(ctx, rows)
    logt = np.log1p(rows["t"].to_numpy(float))
    mu_u, sd_u = float(logt.mean()), float(logt.std())
    u, x = _design(rows, f, mu_u, sd_u)
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1
    clf = LogisticRegression(C=C_RIDGE, solver="newton-cholesky", max_iter=1000, tol=1e-8).fit(np.column_stack([100 * u, (x - mu) / sd]), y)
    return {"mu_u": mu_u, "sd_u": sd_u, "mu": mu, "sd": sd, "coef": clf.coef_[0], "intercept": float(clf.intercept_[0])}


def predict(state, ctx, rows):
    f = _inputs(ctx, rows)
    u, x = _design(rows, f, state["mu_u"], state["sd_u"])
    z = np.column_stack([100 * u, (x - state["mu"]) / state["sd"]]) @ state["coef"] + state["intercept"]
    return 1.0 / (1.0 + np.exp(-z))
