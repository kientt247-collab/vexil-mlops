"""P3 item (f), device side: fit cand-R once on all development matches through the vexil package and write models/candidates/R/model.json.
Development = matches that start before 2026-07-31 00:00 UTC. Fitting set = usable matches of a judged tier with a result (as p1_final_fit.py).
Results: study B zip results cache plus the 4,470 development labels of E1 label look 1 (phase 2 loader); no label of the weekly store is read and no match
on or after 2026-07-31 is read. The file is fitted twice and the two texts must be identical. LABEL LOOK 1.
usage: cd /tmp/lock-v4 && python -B /tmp/b/code/run_p3_final_fit.py STORE_ROOT MANIFEST DEV_LABELS_JSON OUT_REPORT.json"""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/tmp/b/code')
from vexil.candidates import CandidateTable, fit_candidate, match_meta
from vexil.data.store import Store
from vexil.features.extra import load_items
from vexil.matchdata import is_usable
from vexil.scoring import JUDGED_TIERS
store_root, manifest, dev_labels, out = sys.argv[1:5]
DEV_END = 1785456000
TRAIN = Path('models/candidates/R/train.py'); SHA = hashlib.sha256(TRAIN.read_bytes()).hexdigest()
t0 = time.time()
z = np.load('/tmp/b/cache/results_rw.npz'); results = dict(zip(z['match_id'].tolist(), z['y'].tolist()))
dl = {int(k): int(v) for k, v in json.load(open(dev_labels)).items()}
clash = sum(1 for k, v in dl.items() if k in results and results[k] != v)
results.update(dl)
store = Store(Path(store_root), Path(manifest)); weeks = [w for w in store.weeks() if w <= '2026-07-27']
fit_ids, other_meta, n_late = set(), {}, 0
def gen():
    global n_late
    for w in weeks:
        for m, p in store.read_week(w):
            if m['start_time'] >= DEV_END: n_late += 1; continue
            if is_usable(m) and m.get('league_tier') in JUDGED_TIERS and m['match_id'] in results:
                fit_ids.add(m['match_id']); yield m, p
            else: other_meta[m['match_id']] = match_meta(m, p)
table = CandidateTable.build(gen(), {}, load_items())
for mid, v in other_meta.items(): table.meta.setdefault(mid, v)
table.results = {mid: int(results[mid]) for mid in table.meta if mid in results}
table.start = {m: v['start'] for m, v in table.meta.items()}
ends = sorted((v['end'], m) for m, v in table.meta.items()); table.end_sorted = np.array([e for e, _ in ends], np.int64); table.end_ids = [i for _, i in ends]
ids = sorted(fit_ids); print('fitting set:', len(ids), 'matches | matches skipped as on/after 2026-07-31 (start time only, not read further):', n_late, '| label clashes zip vs dev labels:', clash, flush=True)
specs = [fit_candidate(TRAIN, SHA, table, ids, seed=0) for _ in range(2)]
texts = [json.dumps(s, separators=(',', ':'), sort_keys=True) for s in specs]
assert texts[0] == texts[1], 'two fits with seed 0 gave different state files'
o = Path('models/candidates/R/model.json'); o.write_text(texts[0] + '\n')
rep = {'train_sha256': SHA, 'model_sha256': hashlib.sha256(o.read_bytes()).hexdigest(), 'n_matches': specs[0]['n_matches'], 'n_rows': specs[0]['n_rows'], 'model_bytes': o.stat().st_size,
       'two_fits_identical': True, 'fit_matches_of_loop': 4470, 'label_clashes': clash, 'seconds': round(time.time() - t0, 1)}
json.dump(rep, open(out, 'w'), indent=1); print(json.dumps(rep))
