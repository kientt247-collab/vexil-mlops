"""P3 proof items (d): hand fixture with a straddling match, planted defects that must be caught, resume equals full pass on real matches.
usage: python3 -B test_p3.py [real]   (label-free: ratings use results that are already in the rating pass of B/E1; scores nothing)"""
import sys, math, numpy as np
from p3_common import *

def fx():
    ids = np.arange(1, 11, dtype=np.int64)
    acc = np.tile(ids, (3, 1))
    start = np.array([JAN1 - 3000, JAN1 - 1000, JAN1 + 2000], np.int64)
    end = np.array([JAN1 - 1000, JAN1 + 1000, JAN1 + 3000], np.int64)
    y = np.array([1.0, 1.0, 1.0]); mid = np.array([1, 2, 3], np.int64)
    return acc, start, end, y, mid

def hand_f():
    r1 = 64 / 2.0; ra, rb = 1500 + r1, 1500 - r1                      # after m0: winner side +32, K 64, expected 0.5
    ex = 1 / (1 + 10 ** (-(ra - rb) / 400)); d = 64 * (1 - ex)       # m1 ends after the cut, applied once at its end
    ra, rb = ra + d, rb - d
    return (ra - rb) * math.log(10) / 400                              # m2 read at its start

def test_fixture():
    acc, start, end, y, mid = fx(); full = rating_pass(acc, start, end, y, mid, 0, np.array([2]))['f'][0]
    sel = end < JAN1
    ua, Rt, cnt = (lambda r: (r['ua'], r['R'], r['cnt']))(rating_pass(acc[sel], start[sel], end[sel], y[sel], mid[sel], 0, np.array([], np.int64)))
    assert abs(Rt[0] - 1532) < 1e-9 and cnt[0] == 1, (Rt[:2], cnt[:2])
    res = rating_pass_resume((ua, Rt, cnt, JAN1), acc, start, end, y, mid, np.array([2]), w0=0)['f'][0]
    print('fixture: hand', hand_f(), '| full pass', full, '| resume', res)
    assert abs(res - hand_f()) < 1e-12 and abs(full - hand_f()) < 1e-12
    bad = rating_pass_resume((ua, Rt, cnt, JAN1), acc, start, end, y, mid, np.array([2]), w0=0, _plant='start')['f'][0]
    print('planted start-selection on the fixture: f', bad, '| difference', abs(bad - hand_f()))
    assert abs(bad - hand_f()) > 1e-3, 'planted start-selection NOT caught'

def test_real():
    from b_data import load_all, f_column
    R, ctx, mism = load_all(); assert mism == 0
    f_full = f_column(R, ctx, HISTORY_START['B2'])
    ua, Rt, cnt = build_table(R)
    r = rating_pass_resume((ua, Rt, cnt, JAN1), R['acc'], R['start'], R['end'], R['y'], R['match_id'], ctx['ridx'])
    d = float(np.abs(r['f'] - f_full).max()); print('real: resume equals full pass, max |diff|', d); assert d < 1e-9
    cut2 = JAN1 + 30 * 86400                                           # planted: 30 days of 2026 results inside the table
    ua2, R2, c2 = build_table(R, cut=cut2)
    r2 = rating_pass_resume((ua2, R2, c2, JAN1), R['acc'], R['start'], R['end'], R['y'], R['match_id'], ctx['ridx'])
    d2 = float(np.abs(r2['f'] - f_full).max()); print('real, planted 2026 results in the table: max |diff|', d2); assert d2 > 1e-3, 'planted leak NOT caught'
    # table must not move when every result at or after the cut is replaced by noise
    rng = np.random.default_rng(20261124); R_n = dict(R); y_n = R['y'].copy(); late = R['end'] >= JAN1; y_n[late] = rng.integers(0, 2, late.sum()).astype(float); R_n['y'] = y_n
    ua3, R3, c3 = build_table(R_n); same = (ua3 == ua).all() and (R3 == Rt).all() and (c3 == cnt).all(); print('table unchanged when results at or after the cut are random:', bool(same)); assert same
    # earliest and latest source time in the table build
    print('latest END used by the table:', int(R['end'][R['end'] < JAN1].max()), '< cut', JAN1)

test_fixture()
if len(sys.argv) > 1 and sys.argv[1] == 'real': test_real()
print('ALL PASS')
