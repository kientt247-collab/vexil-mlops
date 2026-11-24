"""P3 (build, map v3): frozen table of initial ratings and its resume path. Device-side part; the vexil-mlops package part is not built here.
Table = state of the player rating after all rated matches that started at or after 2024-01-01 and ENDED before 2026-01-01 00:00 UTC
(arm R-B2 of E1, the arm chosen by the arm rule), K 64 for an account's first 20 rated matches then 32."""
import csv, hashlib, math
import numpy as np
from e2_common import *


def build_table(R, w0=HISTORY_START['B2'], cut=JAN1):
    sel = R['end'] < cut
    res = rating_pass(R['acc'][sel], R['start'][sel], R['end'][sel], R['y'][sel], R['match_id'][sel], w0, np.array([], np.int64))
    return res['ua'], res['R'], res['cnt']


def write_table(path, ua, Rt, cnt, cut=JAN1):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['account_id', 'rating', 'n_rated', 'state_time_utc'])
        for a, r, c in zip(ua, Rt, cnt): w.writerow([int(a), repr(float(r)), int(c), int(cut)])


def read_table(path):
    rows = list(csv.DictReader(open(path)))
    return (np.array([int(r['account_id']) for r in rows], np.int64), np.array([float(r['rating']) for r in rows]), np.array([int(r['n_rated']) for r in rows], np.int64), int(rows[0]['state_time_utc']))


def rating_pass_resume(table, acc, start, end, y, mid, read_idx, w0=HISTORY_START['B2'], k_old=K_OLD, k_new=K_NEW, n_new=N_NEW, _plant=None):
    """Rating pass that starts from the table state at `cut`: only rated matches with END at or after cut are applied (a match that started before cut and
    ended after it is applied at its end). Reads at start time, as rating_pass. _plant 'start': selects updates by START >= cut instead of END (defect)."""
    ua0, R0, c0, cut = table
    upd = np.nonzero((start >= cut) if _plant == 'start' else (end >= cut))[0]
    rd = np.asarray(read_idx, np.int64)
    ua = np.unique(np.concatenate([ua0, acc[upd].ravel(), acc[rd].ravel()]))
    A = np.searchsorted(ua, acc)
    R = np.full(len(ua), 1500.0); cnt = np.zeros(len(ua), np.int64)
    j = np.searchsorted(ua, ua0); R[j] = R0; cnt[j] = c0
    time = np.r_[start[rd].astype(np.float64), end[upd].astype(np.float64)]
    typ = np.r_[np.zeros(len(rd), np.int8), np.ones(len(upd), np.int8)]
    who = np.r_[rd, upd]; pos = np.r_[np.arange(len(rd)), np.full(len(upd), -1)]
    order = np.lexsort((mid[who], typ, time)); f = np.zeros(len(rd))
    for e in order:
        i = who[e]; a = A[i]; r = R[a]
        if typ[e] == 0: f[pos[e]] = (r[:5].mean() - r[5:].mean()) * LN10_400
        else:
            ex = 1.0 / (1.0 + 10.0 ** (-(r[:5].mean() - r[5:].mean()) / 400.0))
            d = np.where(cnt[a] < n_new, k_new, k_old) * (y[i] - ex)
            np.add.at(R, a[:5], d[:5]); np.add.at(R, a[5:], -d[5:]); np.add.at(cnt, a, 1)
    return {'f': f, 'R': R, 'cnt': cnt, 'ua': ua}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''): h.update(ch)
    return h.hexdigest()
