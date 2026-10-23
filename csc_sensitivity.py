"""
Sensitivity of the AAN CSC results to the CSC parameters.

Jardine (2014, Definition 4.1) fixes the substitution score as 1/C[Ei,Sj]
with a path-length cap. The paper keeps these fixed (cap = 10 edges, 1/c decay).
This script re-scores the SAME cached rankings under other cap/decay choices and
checks whether the paper's qualitative conclusions survive:

  (i)   bw_modified_bias is the top-ranked of the 19 PageRank variants;
  (ii)  *_bias > *_transition in every one of the five weighting families;
  (iii) how many of the 19 variants score above BM25;
  (iv)  rank correlation of the 19-variant ordering with the paper's setting.

No PageRank is recomputed: it reads the same cached pageranks_*.bin files and the
same query-aware ranking code as csc_evaluation.py, so the (cap=10, 1/c) row
must reproduce Table 2 exactly (used here as a sanity check).

Writes: components/resources/gs/csc_sensitivity_per_variant.csv
        components/resources/gs/csc_sensitivity_summary.json
"""

import json
import math
import os
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path

LDA_NO_TOPICS = int(os.getenv('LDA_NO_TOPICS', '200'))
K = 10
PR_OUTPUT = Path('components/output/pr') / str(LDA_NO_TOPICS) / str(LDA_NO_TOPICS)
GS_PATH = Path('components/resources/gs/golden_standard.json')
GS_OUT = Path('components/resources/gs')

from components.prmodes import PRModes
from components.database import setup as db_setup
from models.tables import Paper, PaperCitation
db_setup()
from components.lda import setup as lda_setup
from components.query import rank_pagerank_for_query

lda_results = lda_setup(LDA_NO_TOPICS, 0, fresh=False)
ALL_MODES = [m.value for m in PRModes.get_evaluation_modes()]

# ---- graph / gold standard (identical construction to csc_evaluation.py) ----
code_to_docid = {p.code: p.id - 1 for p in Paper.select(Paper.id, Paper.code).iterator()}
undirected = defaultdict(set)
for c in PaperCitation.select().iterator():
    a, b = c.citing_paper_id - 1, c.cited_paper_id - 1
    if a != b:
        undirected[a].add(b)
        undirected[b].add(a)

gs_raw = json.load(GS_PATH.open())
golden = {q: {code_to_docid[c] for c in codes if c in code_to_docid}
          for q, codes in gs_raw.items()}
golden = {q: g for q, g in golden.items() if g}

MAX_CAP = 20
_cache = {}


def dists(src):
    """BFS edge distances up to MAX_CAP (filtered per variant later)."""
    if src in _cache:
        return _cache[src]
    d = {src: 0}
    fr = deque([src])
    while fr:
        n = fr.popleft()
        if d[n] >= MAX_CAP:
            continue
        for m in undirected.get(n, ()):
            if m not in d:
                d[m] = d[n] + 1
                fr.append(m)
    _cache[src] = d
    return d


def csc(pred, gold, cap, decay):
    f = (lambda x: 1.0 / x) if decay == 'inv' else (lambda x: 1.0 / (x * x))
    top = pred[:K]
    top_set = set(top)
    fs = []
    for e in gold:
        if e in top_set:
            fs.append(1.0)
            continue
        dd = dists(e)
        fs.append(max([f(dd[s]) for s in top if dd.get(s, 0) > 0 and dd[s] <= cap] or [0.0]))
    rs = []
    for s in top:
        if s in gold:
            rs.append(1.0)
            continue
        dd = dists(s)
        rs.append(max([f(dd[e]) for e in gold if dd.get(e, 0) > 0 and dd[e] <= cap] or [0.0]))
    fcsc, rcsc = sum(fs) / len(fs), sum(rs) / len(rs)
    return fcsc, rcsc, (math.sqrt(fcsc * rcsc) if fcsc > 0 and rcsc > 0 else 0.0)


def load_pr(mode):
    p = PR_OUTPUT / f'pageranks_{mode}.bin'
    if not p.exists():
        return None
    res = pickle.load(p.open('rb'))
    ag = {}
    for paper in Paper.select(Paper.id, Paper.year).iterator():
        d = paper.id - 1
        if d in res:
            yd = max(1, 2011 - paper.year)
            ag[d] = {t: v / yd for t, v in res[d].items()}
    return ag


# ---- rankings (computed once, then re-scored under each variant) ----
rankings = {}
for mode in ALL_MODES:
    pr = load_pr(mode)
    if pr is None:
        print(f'[MISSING] {mode}')
        continue
    rankings[mode] = {q: rank_pagerank_for_query(q, pr, lda_results) for q in golden}
    print(f'ranked {mode}')

from components.baselines import BM25Baseline
bm25 = BM25Baseline()
bm25.build_index(fresh=False)
# BM25Baseline.search returns [(paper_id_1based, score), ...]; convert to 0-based doc ids
# exactly as csc_with_baselines.py::eval_query_results does.
rankings['BM25'] = {q: [pid - 1 for pid, _ in bm25.search(q, top_k=K)[:K]] for q in golden}

FAMILIES = {
    'author': ('modified_bias', 'modified_transition'),
    'degree': ('d_modified_bias', 'd_modified_transition'),
    'closeness': ('cl_modified_bias', 'cl_modified_transition'),
    'betweenness': ('bw_modified_bias', 'bw_modified_transition'),
    'katz': ('kz_modified_bias', 'kz_modified_transition'),
}


def spearman(a, b):
    ks = sorted(a)
    ra = {k: i for i, k in enumerate(sorted(ks, key=lambda k: -a[k]))}
    rb = {k: i for i, k in enumerate(sorted(ks, key=lambda k: -b[k]))}
    n = len(ks)
    return 1 - 6 * sum((ra[k] - rb[k]) ** 2 for k in ks) / (n * (n * n - 1))


variants = [(cap, decay) for decay in ('inv', 'inv2') for cap in (3, 5, 10, 20)]
per_variant = {}
rows = []
for cap, decay in variants:
    key = f'cap{cap}_{decay}'
    means = {}
    for mode, qr in rankings.items():
        vals = [csc(qr[q], golden[q], cap, decay) for q in golden]
        means[mode] = {
            'fcsc': sum(v[0] for v in vals) / len(vals),
            'rcsc': sum(v[1] for v in vals) / len(vals),
            'ocsc': sum(v[2] for v in vals) / len(vals),
        }
        rows.append((key, mode, means[mode]['fcsc'], means[mode]['rcsc'], means[mode]['ocsc']))
    per_variant[key] = means

ref = {m: per_variant['cap10_inv'][m]['ocsc'] for m in ALL_MODES if m in rankings}
summary = {}
for cap, decay in variants:
    key = f'cap{cap}_{decay}'
    means = per_variant[key]
    oc = {m: means[m]['ocsc'] for m in ALL_MODES if m in means}
    best = max(oc, key=oc.get)
    fam = {f: oc[b] > oc[t] for f, (b, t) in FAMILIES.items()}
    summary[key] = {
        'best_of_19': best,
        'best_ocsc': round(oc[best], 4),
        'bm25_ocsc': round(means['BM25']['ocsc'], 4),
        'n_above_bm25': sum(1 for m in oc if oc[m] > means['BM25']['ocsc']),
        'bias_gt_transition_by_family': fam,
        'families_bias_gt_transition': sum(fam.values()),
        'spearman_vs_cap10_inv': round(spearman(oc, ref), 3),
    }

with (GS_OUT / 'csc_sensitivity_per_variant.csv').open('w', encoding='utf-8') as f:
    f.write('variant,mode,fcsc,rcsc,ocsc\n')
    for r in rows:
        f.write(f'{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f}\n')
(GS_OUT / 'csc_sensitivity_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

print('\n=== sanity: cap10_inv must reproduce Table 2 ===')
for m in ('bw_modified_bias', 'normal', 'themed', 'BM25'):
    print(f"  {m:<20} OCSC={per_variant['cap10_inv'][m]['ocsc']:.4f}")
print('\n=== sensitivity summary ===')
for key, s in summary.items():
    print(f"{key:<14} best={s['best_of_19']:<18} ({s['best_ocsc']})  BM25={s['bm25_ocsc']}  "
          f"above_BM25={s['n_above_bm25']:>2}/19  bias>trans={s['families_bias_gt_transition']}/5  "
          f"rho={s['spearman_vs_cap10_inv']}")
