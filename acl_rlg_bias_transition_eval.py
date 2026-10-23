"""
Evaluation for acl_rlg_bias_vs_transition_PREREG.md. Reads acl_rlg_pilot_out/<tag>/<year>/<mode>.npz
(written by acl_rlg_bias_transition.py), scores every query, averages the three annotator sets per
tutorial, and (1) validates the new solver against the stored temporal means, (2) tests
bias vs transition per centrality family with the pre-registered protocol.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='bt')
args = ap.parse_args()

RLG_BASE = Path(r"E:\0000000-Thesis\1403-new approach\ACL-rlg\aclanthology-tutorial-reading-lists")
ACL_DATASET_PATH = RLG_BASE / "experiments/classic_methods/acl_anthology_dataset/acl_anthology_dataset.json"
READING_LISTS_CSV = RLG_BASE / "reading_lists.csv"
ANNOT_DIR = RLG_BASE / "annotations"
OUT = Path(r"E:\0000000-Thesis\1405\thesis-1405\thesis-1405\acl_rlg_pilot_out")
YEARS = [2016, 2019, 2020, 2021, 2022, 2023, 2024]
BIAS = {'d': 'd_modified_bias', 'cl': 'cl_modified_bias', 'bw': 'bw_modified_bias', 'kz': 'kz_modified_bias'}
TRANS = {'d': 'd_modified_transition', 'cl': 'cl_modified_transition', 'bw': 'bw_modified_transition',
         'kz': 'kz_modified_transition'}
FAMILIES = ['d', 'cl', 'bw', 'kz']
NAMES = {'d': 'degree', 'cl': 'closeness', 'bw': 'betweenness', 'kz': 'katz'}


def clean_string(s):
    return re.sub(r'\W+', '', s or '').lower()


def recall(t, p): return len([x for x in t if x in p]) / len(t) if t else 0.0


def ndcg(t, p):
    rel = [1 if x in t else 0 for x in p]
    dcg = sum((2 ** r - 1) / np.log2(i + 1) for i, r in enumerate(rel, 1))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, len(t) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(t, p):
    for i, x in enumerate(p, 1):
        if x in t:
            return 1.0 / i
    return 0.0


with open(ACL_DATASET_PATH, encoding='utf-8') as f:
    ds = json.load(f)
acl_ids = sorted(ds)
years = np.array([ds[a]['year'] for a in acl_ids])
clean_titles = np.array([clean_string(ds[a].get('title', '')) for a in acl_ids])
dfg = pd.read_csv(READING_LISTS_CSV)
dfg['reading_list'] = dfg['reading_list'].apply(ast.literal_eval)
trues = {r['id']: [clean_string(p['title']) for p in r['reading_list']
                   if str(p.get('in_acl', 'False')) == 'True' and p.get('title')] for _, r in dfg.iterrows()}

# map (annotation set, tutorial) -> year
qyear = {}
for ai in (1, 2, 3):
    da = pd.read_csv(ANNOT_DIR / f'annotation_{ai}.csv')
    for _, r in da.iterrows():
        qyear[(ai, r['id'])] = int(r['year'])

modes_all = sorted({p.stem for y in YEARS for p in (OUT / args.tag / str(y)).glob('*.npz')})
rows = {}            # mode -> {(annot, tid): (recall, ndcg, mrr)}
conv = {}            # mode -> list of (year, nonconverged, median iters, max iters, mean ratio)
tq_nonconv = {}      # mode -> {(annot, tid): share of theta_q weight on nonconverged topics}
for mode in modes_all:
    rows[mode] = {}
    conv[mode] = []
    tq_nonconv[mode] = {}
    for y in YEARS:
        p = OUT / args.tag / str(y) / f'{mode}.npz'
        if not p.exists():
            continue
        z = np.load(p)
        S, qs, theta_q, ok = z['S'], z['queries'], z['theta_q'], z['converged']
        conv[mode].append((y, int((~ok).sum()), int(np.median(z['iterations'])), int(z['iterations'].max())))
        for qi, q in enumerate(qs):
            a, tid = q.split('|', 1)
            a = int(a)
            true = trues[tid]
            score = np.where(years <= y, S[qi], -1)
            ranked = np.argsort(-score)[:20]
            preds = clean_titles[ranked].tolist()
            rows[mode][(a, tid)] = (recall(true, preds) * 100, ndcg(true, preds) * 100, mrr(true, preds) * 100)
            tq_nonconv[mode][(a, tid)] = float(theta_q[qi][~ok].sum())


def per_tutorial(mode, metric_idx):
    by = {}
    for (a, tid), v in rows[mode].items():
        by.setdefault(tid, []).append(v[metric_idx])
    return {tid: float(np.mean(v)) for tid, v in by.items()}


# --------------------------------------------------------- validation vs stored means
stored = json.load(open(OUT / 'final_results_temporal.json'))['mean']
print('VALIDATION against stored temporal means (mean over the 3 annotation sets, as in the paper)')
ok_all = True
for mode, key in (('personalized', 'personalized'), ('themed', 'themed'), ('bw_modified_bias', 'bw_modified_bias')):
    if mode not in rows:
        continue
    mean = [np.mean([v[i] for v in rows[mode].values()]) for i in range(3)]
    # the paper's means are per-annotation-set means averaged over 3 sets; equal sets sizes -> same as mean over queries
    st = [stored[key]['Recall@20'], stored[key]['NDCG@20'], stored[key]['MRR@20']]
    diff = max(abs(a - b) for a, b in zip(mean, st))
    ok_all &= diff < 0.01
    print(f'  {mode:<18} new R/N/M = {mean[0]:.2f} {mean[1]:.2f} {mean[2]:.2f}   stored = {st[0]:.2f} {st[1]:.2f} {st[2]:.2f}   max|diff| = {diff:.4f}')
print('  VALIDATION', 'PASSED' if ok_all else 'FAILED (do not use the new modes until explained)')

# --------------------------------------------------------- convergence
print('\nCONVERGENCE (year: non-converged topics of 200 / median / max iterations)')
for mode in modes_all:
    print(f'  {mode:<24}', '  '.join(f'{y}:{n}/{m}/{mx}' for y, n, m, mx in conv[mode]))

# --------------------------------------------------------- descriptives
tids = sorted(per_tutorial('bw_modified_bias', 1))
print(f'\n{len(tids)} tutorials with a non-empty gold list')
print('\nMEANS over tutorials (R@20 / NDCG@20 / MRR@20)')
for mode in modes_all:
    m = [np.mean(list(per_tutorial(mode, i).values())) for i in range(3)]
    print(f'  {mode:<24} {m[0]:6.2f} {m[1]:6.2f} {m[2]:6.2f}')

# --------------------------------------------------------- pre-registered tests
rng = np.random.default_rng(0)


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


results = {}
for mi, mname in ((1, 'NDCG@20'), (2, 'MRR@20'), (0, 'Recall@20')):
    pvals, rec = [], []
    for f in FAMILIES:
        b = per_tutorial(BIAS[f], mi)
        t = per_tutorial(TRANS[f], mi)
        D = np.array([b[x] - t[x] for x in tids])
        try:
            p = wilcoxon(D, alternative='two-sided', zero_method='wilcox').pvalue
        except ValueError:
            p = 1.0
        boots = np.array([D[rng.integers(0, len(D), len(D))].mean() for _ in range(10000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rec.append({'family': NAMES[f], 'mean_diff': float(D.mean()), 'median_diff': float(np.median(D)),
                    'ci95': [float(lo), float(hi)], 'bias_better': int((D > 0).sum()),
                    'transition_better': int((D < 0).sum()), 'ties': int((D == 0).sum()), 'p_raw': float(p),
                    'mean_bias': float(np.mean([b[x] for x in tids])), 'mean_transition': float(np.mean([t[x] for x in tids]))})
        pvals.append(p)
    adj = holm(np.array(pvals))
    for r, a in zip(rec, adj):
        r['p_holm_within_metric'] = float(a)
    results[mname] = rec

# conservative sensitivity: Holm over the 8 confirmatory tests
all8 = np.array([r['p_raw'] for m in ('NDCG@20', 'MRR@20') for r in results[m]])
adj8 = holm(all8)
k = 0
for m in ('NDCG@20', 'MRR@20'):
    for r in results[m]:
        r['p_holm_over_8'] = float(adj8[k])
        k += 1

for mname in ('NDCG@20', 'MRR@20', 'Recall@20'):
    tag = {'NDCG@20': 'PRIMARY', 'MRR@20': 'SECONDARY (confirmatory)', 'Recall@20': 'DESCRIPTIVE ONLY'}[mname]
    print(f'\n{mname}  [{tag}]   D = bias - transition, n = {len(tids)} tutorials')
    print(f'  {"family":<12}{"bias":>7}{"trans":>7}{"mean D":>8}{"median":>8}{"95% CI":>18}{"b>t/t>b/tie":>13}{"p raw":>9}{"p Holm4":>9}{"p Holm8":>9}')
    for r in results[mname]:
        print(f'  {r["family"]:<12}{r["mean_bias"]:7.2f}{r["mean_transition"]:7.2f}{r["mean_diff"]:8.2f}{r["median_diff"]:8.2f}'
              f'  [{r["ci95"][0]:6.2f},{r["ci95"][1]:6.2f}]{r["bias_better"]:>5}/{r["transition_better"]}/{r["ties"]:<3}'
              f'{r["p_raw"]:9.4f}{r["p_holm_within_metric"]:9.4f}{r.get("p_holm_over_8", float("nan")):9.4f}')

# --------------------------------------------------------- sensitivity: non-converged topics
print('\nSENSITIVITY: weight of each tutorial on non-converged topics')
for mode in modes_all:
    w = np.array([tq_nonconv[mode].get(k, 0.0) for k in rows[mode]])
    if w.max() > 0:
        print(f'  {mode:<24} mean share {w.mean():.4f}   max {w.max():.4f}   tutorials affected {int((w>0).sum())}')
if all(all(c[1] == 0 for c in conv[m]) for m in modes_all):
    print('  all solves converged for all modes and years; sensitivity analysis not needed')

(OUT / f'bias_vs_transition_{args.tag}.json').write_text(json.dumps(
    {'n_tutorials': len(tids), 'results': results,
     'convergence': {m: conv[m] for m in modes_all}}, indent=2), encoding='utf-8')
print(f'\nsaved {OUT / ("bias_vs_transition_" + args.tag + ".json")}')
