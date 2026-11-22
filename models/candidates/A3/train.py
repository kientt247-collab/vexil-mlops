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


def _sides(m):
    ra, da = m["ra"], m["da"]
    return (ra, da) if len(ra) == 5 and len(da) == 5 and all(ra) and all(da) else None


def _apply(rating, games, m, result, offset=0.0):
    sides = _sides(m)
    if sides is None:
        return
    rad, dire = sides
    r = sum(rating.get(a, 1500.0) for a in rad) / 5
    d = sum(rating.get(a, 1500.0) for a in dire) / 5
    surprise = result - 1.0 / (1.0 + 10 ** ((d - r - offset) / 400))
    for sign, side in ((1.0, rad), (-1.0, dire)):
        for a in side:
            kk = K_NEW if games.get(a, 0) < N_NEW else K
            rating[a] = rating.get(a, 1500.0) + sign * kk * surprise
            games[a] = games.get(a, 0) + 1


HERO_KS = (10.0,)


def _pdiff(rating, m):
    sides = _sides(m)
    if sides is None:
        return 0.0
    rad, dire = sides
    return sum(rating.get(a, 1500.0) for a in rad) / 5 - sum(rating.get(a, 1500.0) for a in dire) / 5


def _apply_h(heroes, m, result, pdiff):
    rh, dh = m.get("rh"), m.get("dh")
    if not rh or not dh or len(rh) != 5 or len(dh) != 5:
        return
    for hero, kh in zip(heroes, HERO_KS):
        r = sum(hero.get(h, 1500.0) for h in rh) / 5
        d = sum(hero.get(h, 1500.0) for h in dh) / 5
        surprise = result - 1.0 / (1.0 + 10 ** (-(pdiff + r - d) / 400))
        for sign, side in ((1.0, rh), (-1.0, dh)):
            for h in side:
                hero[h] = hero.get(h, 1500.0) + sign * kh * surprise


def _hdiff(hero, m):
    rh, dh = m.get("rh"), m.get("dh")
    if not rh or not dh or len(rh) != 5 or len(dh) != 5:
        return 0.0
    return sum(hero.get(h, 1500.0) for h in rh) / 5 - sum(hero.get(h, 1500.0) for h in dh) / 5


def _read_h(heroes, rh, dh):
    if not rh or not dh:
        return [0.0] * len(heroes)
    side = lambda hero, hs: float(np.mean([hero.get(h, 1500.0) for h in hs]))
    return [(side(hero, rh) - side(hero, dh)) * SCALE for hero in heroes]


def _read(rating, ra, da):
    side = lambda acc: float(np.mean([rating.get(a, 1500.0) if a else 1500.0 for a in acc])) if acc else 1500.0
    return (side(ra) - side(da)) * SCALE


def rating_input(results, queries):
    """results: dicts sorted by end time (as ctx.results_before returns them); queries: (match_id, start, ra, da).
    Returns {match_id: input}. A query reads the ratings after every result that ended strictly before its start."""
    rating, games, heroes, out, i = {}, {}, [{} for _ in HERO_KS], {}, 0
    for mid, start, ra, da, rh, dh in sorted(queries, key=lambda q: (q[1], q[0])):
        while i < len(results) and results[i]["end"] < start:
            hd = _hdiff(heroes[0], results[i])
            _apply_h(heroes, results[i], results[i]["radiant_win"], _pdiff(rating, results[i]))
            _apply(rating, games, results[i], results[i]["radiant_win"], hd)
            i += 1
        out[mid] = [_read(rating, ra, da)] + _read_h(heroes, rh, dh)
    return out


def _inputs(ctx, rows):
    ids = sorted(set(int(m) for m in rows["match_id"]))
    pre = {m: ctx.pregame(m) for m in ids}
    f = rating_input(ctx.results_before(ctx.now), [(m, pre[m]["start"], pre[m]["ra"], pre[m]["da"], pre[m]["rh"], pre[m]["dh"]) for m in ids])
    return np.array([f[int(m)] for m in rows["match_id"]])


def _design(rows, f, mu_u, sd_u):
    u = (np.log1p(rows["t"].to_numpy(float)) - mu_u) / sd_u
    a = rows[A1].to_numpy(float)
    x = np.column_stack([a, f, f * u[:, None], a * u[:, None]])
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
