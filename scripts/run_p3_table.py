"""P3 (a), (c-device), (d), (e): build the table, hash it, check that table + later matches equals the full pass for all 4,236 judged 2026 matches.
LABEL LOOK 1 (rated results before the cut and 2026 results enter the rating pass, as in B and E1). usage: python3 -B run_p3_table.py <out_dir>"""
import sys, os, json, numpy as np
from p3_common import *
from b_data import load_all, f_column
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
R, ctx, mism = load_all(); assert mism == 0
ua, Rt, cnt = build_table(R)
path = f'{out}/initial_ratings_2026-01-01.csv'; write_table(path, ua, Rt, cnt)
ua2, R2, c2, cut = read_table(path)
assert (ua2 == ua).all() and (R2 == Rt).all() and (c2 == cnt).all(), 'csv round trip changed the table'
print('table: accounts', len(ua), '| accounts with 20 or more rated matches', int((cnt >= 20).sum()), '| sha256', sha256_file(path)[:16])
straddle = int(((R['start'] < JAN1) & (R['end'] >= JAN1) & (R['start'] >= HISTORY_START['B2'])).sum())
print('matches that started before the cut and ended after it:', straddle)
f_full = f_column(R, ctx, HISTORY_START['B2'])
res = rating_pass_resume((ua2, R2, c2, cut), R['acc'], R['start'], R['end'], R['y'], R['match_id'], ctx['ridx'])
d = float(np.abs(res['f'] - f_full).max()); print('resume equals full pass on', len(f_full), 'judged matches: max |difference|', d)
ua3, R3, c3 = build_table(R, cut=JAN1 + 30 * 86400)
res_bad = rating_pass_resume((ua3, R3, c3, JAN1), R['acc'], R['start'], R['end'], R['y'], R['match_id'], ctx['ridx'])
bad = float(np.abs(res_bad['f'] - f_full).max())
print('planted (30 days of 2026 results inside the table): max |difference|', bad)
json.dump({'table_sha256': sha256_file(path), 'accounts': int(len(ua)), 'straddling_matches': straddle, 'resume_max_abs_diff': d,
           'planted_2026_in_table_max_abs_diff': bad, 'code_sha256': {n: sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), n)) for n in ('p3_common.py', 'run_p3_table.py', 'test_p3.py', 'b_common.py', 'e2_common.py')},
           'label_looks': 1}, open(f'{out}/table_record.json', 'w'), indent=1)
assert d < 1e-9
