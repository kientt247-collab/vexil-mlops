"""P3 items (b) and (c), device side: cand-R through the vexil package (train.py loaded by hash, CandidateTable built from the weekly store)
against E1's arm R-B2 on the same five test blocks. Reads results of development matches only (start < 2026-07-31), taken from the zip results
cache of study B (no holdout match, no label of the weekly store is read). LABEL LOOK 1.
usage (python 3.12 venv of /tmp/lock-v4):  cd /tmp/lock-v4 && python -B /tmp/b/code/run_p3_package.py STORE_ROOT MANIFEST OUT.json"""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/tmp/b/code')
from e1_common import *                       # E1 data, scheme and fit (numpy only)
from e1_common import _design
from b_data import load_all, f_column
from vexil.candidates import CandidateTable, CandidateCtx, match_meta, load_train, fit_candidate, decode_state
from vexil.data.store import Store
from vexil.features.extra import load_items
from vexil.matchdata import n_rows
store_root, manifest, out = sys.argv[1], sys.argv[2], sys.argv[3]
DEV_END = 1785456000
TRAIN = Path('/tmp/lock-v4/models/candidates/R/train.py'); SHA = hashlib.sha256(TRAIN.read_bytes()).hexdigest()
t0 = time.time()
sm, tabs = load_store(); R, ctx, mism = load_all(); assert mism == 0
sc = build_scheme(ctx); pop = population(ctx, sm); assert int(pop.sum()) == 4231
rw = build_rows(ctx, sm, tabs, pop)
f_zip = f_column(R, ctx, HISTORY_START['B2'])                       # E1's R-B2 input per ctx match
pop_ids = set(int(m) for m in ctx['match_id'][pop])
res_zip = dict(zip(*[np.load('/tmp/b/cache/results_rw.npz')[k].tolist() for k in ('match_id', 'y')]))
store = Store(Path(store_root), Path(manifest))
weeks = [w for w in store.weeks() if w <= '2026-07-27']
other_meta = {}; n_pop = [0]
def gen():
    for w in weeks:
        for m, p in store.read_week(w):
            if m['start_time'] >= DEV_END: continue                    # never a match on or after 2026-07-31
            if m['match_id'] in pop_ids: n_pop[0] += 1; yield m, p
            else: other_meta[m['match_id']] = match_meta(m, p)
table = CandidateTable.build(gen(), {}, load_items(), order=sorted(pop_ids))
print('store dev matches: population', n_pop[0], 'others', len(other_meta), '| seconds', round(time.time() - t0, 1), flush=True)
for mid, v in other_meta.items(): table.meta.setdefault(mid, v)
table.results = {mid: int(res_zip[mid]) for mid in table.meta if mid in res_zip}
table.start = {m: v['start'] for m, v in table.meta.items()}
ends = sorted((v['end'], m) for m, v in table.meta.items()); table.end_sorted = np.array([e for e, _ in ends], np.int64); table.end_ids = [i for _, i in ends]
no_res = sum(1 for mid, v in table.meta.items() if mid not in table.results)
print('table: meta', len(table.meta), 'results', len(table.results), 'meta without result', no_res, '| rows', len(table.rows['match_id']), flush=True)
mt = table.rows['match_id'].astype(np.int64); tt = table.rows['t'].astype(np.int64)
keep = tt < T_MAX
# --- align package rows with E1 rows by (match_id, t)
key_pkg = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(mt, tt)) if b < T_MAX}
mid_e1 = ctx['match_id'][rw['ci']]; te1 = rw['t'].astype(np.int64)
miss = [i for i in range(len(mid_e1)) if (int(mid_e1[i]), int(te1[i])) not in key_pkg]
print('E1 rows without a package row:', len(miss), '| package rows (t<30) without an E1 row:', len(key_pkg) - (len(mid_e1) - len(miss)), flush=True)
align = np.array([key_pkg[(int(mid_e1[i]), int(te1[i]))] for i in range(len(mid_e1)) if i not in set(miss)], np.int64) if not miss else None
x_cols = [k for k in A1]
Xp = np.column_stack([table.rows[k][align].astype(np.float64) for k in x_cols])
dX = float(np.abs(Xp - rw['X']).max()); print('max |A1 input package - E1 store tables| =', dX, flush=True)
module = load_train(TRAIN, SHA)
rep = {'train_py_sha256': SHA, 'max_abs_dX': dX, 'blocks': []}
n = len(ctx['start']); y_row = ctx['y'][rw['ci']]; f_row = f_zip[rw['ci']]
tot = {'e1': [0.0, 0], 'pkg': [0.0, 0], 'pkg_zipf': [0.0, 0]}
for k in range(5):
    idx, nb, ne = train_idx(ctx, sc, k, 'A0'); assert ne == 0
    tr_ci = match_mask(n, idx); te_ci = match_mask(n, sc['chunks'][k + 1])
    tr, te = tr_ci[rw['ci']], te_ci[rw['ci']]
    mod = fit_e1(rw, tr, f_row, y_row)
    z = _design(rw['t'][te], rw['X'][te], f_row[te], mod['st']) @ mod['w'] + mod['b']; p_e1 = 1 / (1 + np.exp(-z))
    # (b1) cand-R's pure fit on E1's arrays (zip rating input)
    st = module.fit_core(rw['t'][tr], rw['X'][tr], f_row[tr], y_row[tr].astype(float))
    p_core = module.predict_core(st, rw['t'][te], rw['X'][te], f_row[te])
    # (b2) cand-R through the package: store rows, results history, hashed train.py
    tr_ids = sorted(set(int(m) for m in ctx['match_id'][tr_ci & pop])); te_ids = sorted(set(int(m) for m in ctx['match_id'][te_ci & pop]))
    spec = fit_candidate(TRAIN, SHA, table, tr_ids, seed=0); state = decode_state(spec['state'])
    te_rows = np.flatnonzero(np.isin(mt, te_ids) & keep)
    cctx = CandidateCtx(table, te_rows); rows = cctx.rows
    p_pkg = np.asarray(module.predict(state, cctx, rows), float)
    f_pkg = module._inputs(cctx, rows)
    # compare on E1's test rows
    pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rows['match_id'].to_numpy(), rows['t'].to_numpy()))}
    sel = np.array([pos[(int(a), int(b))] for a, b in zip(mid_e1[te], te1[te])], np.int64)
    d_f = np.abs(f_pkg[sel] - f_row[te]); d_p = np.abs(p_pkg[sel] - p_e1)
    y_te = y_row[te].astype(float)
    ll = lambda p: float(np.mean(-(y_te * np.log(p) + (1 - y_te) * np.log(1 - p))))
    b = {'block': k + 1, 'train_matches': len(tr_ids), 'test_rows': int(te.sum()), 'core_vs_e1_max_abs_dp': float(np.abs(p_core - p_e1).max()),
         'pkg_vs_e1_max_abs_dp': float(d_p.max()), 'pkg_vs_e1_max_abs_df': float(d_f.max()), 'share_rows_df_above_0.01': float((d_f > 0.01).mean()),
         'loss_e1': ll(p_e1), 'loss_pkg': ll(p_pkg[sel]), 'loss_core': ll(p_core)}
    rep['blocks'].append(b); print(json.dumps(b), flush=True)
    for nm, p in (('e1', p_e1), ('pkg', p_pkg[sel]), ):
        tot[nm][0] += float(-(y_te * np.log(p) + (1 - y_te) * np.log(1 - p)).sum()); tot[nm][1] += len(y_te)
rep['pooled_loss_e1'] = tot['e1'][0] / tot['e1'][1]; rep['pooled_loss_pkg'] = tot['pkg'][0] / tot['pkg'][1]
rep['pooled_diff'] = rep['pooled_loss_pkg'] - rep['pooled_loss_e1']
rep['max_core_dp'] = max(b['core_vs_e1_max_abs_dp'] for b in rep['blocks']); rep['max_pkg_dp'] = max(b['pkg_vs_e1_max_abs_dp'] for b in rep['blocks'])
rep['max_df'] = max(b['pkg_vs_e1_max_abs_df'] for b in rep['blocks']); rep['seconds'] = round(time.time() - t0, 1)
json.dump(rep, open(out, 'w'), indent=1); print('DONE', json.dumps({k: v for k, v in rep.items() if k != 'blocks'}))
