"""cand-R (P3 of action map v3): the champion's design with a warm-started player rating.

Ridge logistic regression (L2, C = 0.03) on [100 x u, z(ten A1 inputs), z(f), z(f x u)], u = standardised ln(1 + t),
f = player rating input "pelx": Elo of the ten players (K = 64 for a player's first 20 rated matches, then 32), as in the champion.
Difference from the champion: the rating does not start at 1500 on the first stored week. It starts from a frozen table
(initial_ratings_2026-01-01.csv: rating and rated-match count per account, the state after every rated match that started on
or after 2024-01-01 and ended before 2026-01-01 00:00 UTC), checked by SHA-256. Only results of matches that ended at or after the
cut are then applied, in the order matches end (ties by match id); a match reads the ratings after every result that ended
strictly before it started. The table holds no result from the cut on.
Fit: Newton with step halving on sum of log loss + ||w||^2 / (2C) (intercept not penalised), rows t = 0..29 as given by ctx.

Interface:  fit(ctx, train_match_ids, seed) -> state ;  predict(state, ctx, rows) -> numpy array of P(Radiant wins).
"""
import csv
import hashlib
import math
from pathlib import Path

import numpy as np

A1 = ["kill_diff", "towers_diff", "xp_adv_pm", "gold_adv_dec", "aegis_team", "gold_adv_d5", "gold_rank1_diff", "rax_lanes_diff",
      "roshan_diff", "kill_diff_last3"]
C_RIDGE = 0.03
K, K_NEW, N_NEW = 32.0, 64.0, 20
SCALE = math.log(10) / 400
T_MAX = 30
TABLE_FILE = "initial_ratings_2026-01-01.csv"
TABLE_SHA256 = "fac4d6e5a428c394430ba564db87f16bd4e9c75d8a45f7f9d3a521be2b29b7a2"
CUT = 1767225600  # 2026-01-01 00:00 UTC


def load_table():
    path = Path(__file__).with_name(TABLE_FILE)
    if hashlib.sha256(path.read_bytes()).hexdigest() != TABLE_SHA256:
        raise ValueError("initial ratings table differs from the registered hash")
    rating, games = {}, {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["state_time_utc"]) != CUT:
                raise ValueError("table state time is not the cut")
            rating[int(row["account_id"])] = float(row["rating"])
            games[int(row["account_id"])] = int(row["n_rated"])
    return rating, games


def _sides(ra, da):
    return (ra, da) if len(ra) == 5 and len(da) == 5 and all(ra) and all(da) else None


def _apply(rating, games, rad, dire, result):
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


def rating_input(results, queries, table=None):
    """results: dicts sorted by end time (as ctx.results_before returns them); queries: (match_id, start, ra, da).
    Returns {match_id: input}. A query reads the ratings after every result that ended at or after CUT and strictly before its start."""
    rating, games = (dict(x) for x in (table or load_table()))
    out, i = {}, 0
    for mid, start, ra, da in sorted(queries, key=lambda q: (q[1], q[0])):
        while i < len(results) and results[i]["end"] < start:
            r = results[i]
            i += 1
            sides = _sides(r["ra"], r["da"])
            if r["end"] >= CUT and sides is not None:
                _apply(rating, games, *sides, r["radiant_win"])
        out[mid] = _read(rating, ra, da)
    return out


def _inputs(ctx, rows):
    ids = sorted(set(int(m) for m in rows["match_id"]))
    pre = {m: ctx.pregame(m) for m in ids}
    f = rating_input(ctx.results_before(ctx.now), [(m, pre[m]["start"], pre[m]["ra"], pre[m]["da"]) for m in ids])
    return np.array([f[int(m)] for m in rows["match_id"]])


def _logloss_terms(z, y):
    return np.logaddexp(0.0, z) - y * z


def _objective(X, y, w, b, C):
    return _logloss_terms(X @ w + b, y).sum() + 0.5 / C * float(w @ w)


def fit_ridge(X, y, C, max_iter=200, tol=1e-12):
    n, p = X.shape
    A = np.column_stack([X, np.ones(n)])
    reg = np.diag(np.r_[np.full(p, 1.0 / C), 0.0])
    theta = np.zeros(p + 1)
    f = _objective(X, y, theta[:p], theta[p], C)
    for _ in range(max_iter):
        z = A @ theta
        s = 1.0 / (1.0 + np.exp(-z))
        g = A.T @ (s - y) + reg @ theta
        if np.max(np.abs(g)) / n < tol:
            break
        H = (A * (s * (1 - s))[:, None]).T @ A + reg
        step = np.linalg.solve(H, g)
        lam = 1.0
        while True:
            th2 = theta - lam * step
            f2 = _objective(X, y, th2[:p], th2[p], C)
            if f2 <= f + 1e-12 * abs(f) or lam < 1e-8:
                break
            lam *= 0.5
        done = abs(f - f2) <= 1e-14 * abs(f)
        theta, f = th2, f2
        if done:
            break
    return theta[:p], float(theta[p])


def _design(t, X, f, st):
    u = (np.log1p(t) - st["mu_u"]) / st["sd_u"]
    x = np.column_stack([X, f, f * u])
    return np.column_stack([100.0 * u, (x - st["mu"]) / st["sd"]])


def fit_core(t, X, f, y, C=C_RIDGE):
    """Pure fit on arrays (t, ten A1 inputs, rating input, outcome). Statistics with ddof 0 from the fitting rows."""
    lt = np.log1p(t)
    mu_u, sd_u = float(lt.mean()), float(lt.std())
    u = (lt - mu_u) / sd_u
    x = np.column_stack([X, f, f * u])
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    st = {"mu_u": mu_u, "sd_u": sd_u, "mu": mu, "sd": sd}
    w, b = fit_ridge(_design(t, X, f, st), y.astype(np.float64), C)
    return {"mu_u": mu_u, "sd_u": sd_u, "mu": mu, "sd": sd, "coef": w, "intercept": b}


def predict_core(state, t, X, f):
    z = _design(t, X, f, state) @ state["coef"] + state["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def fit(ctx, train_match_ids, seed):
    rows = ctx.rows
    rows = rows[rows["t"] < T_MAX]
    y = np.array([ctx.label(m) for m in rows["match_id"]])
    f = _inputs(ctx, rows)
    return fit_core(rows["t"].to_numpy(float), rows[A1].to_numpy(float), f, y)


def predict(state, ctx, rows):
    f = _inputs(ctx, rows)
    return predict_core(state, rows["t"].to_numpy(float), rows[A1].to_numpy(float), f)
