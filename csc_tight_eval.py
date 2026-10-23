"""
CSC evaluation (AAN; primary setting: 10-edge cutoff, 1/c decay, K=10, aged scores) using
the PageRank vectors computed with the tight stopping rule
(components/output/pr/<T>/<T>_tight/, from pagerank_tight_rerun.py), side by side with the
production vectors the paper reports.

Writes components/resources/gs/csc_tight_vs_production.csv and csc_tight_summary.json
"""
import json
import math
import os
import pickle
from collections import defaultdict, deque
from pathlib import Path

os.environ.setdefault('LDA_NO_TOPICS', '200')
T = int(os.environ['LDA_NO_TOPICS'])
PR_PROD = Path('components/output/pr') / str(T) / str(T)
PR_TIGHT = Path('components/output/pr') / str(T) / f'{T}_tight'
GS_PATH = Path('components/resources/gs/golden_standard.json')
GS_OUT = Path('components/resources/gs')
K, CAP = 10, 10

from components.prmodes import PRModes
from components.database import setup as db_setup
from models.tables import Paper, PaperCitation
db_setup()
from components.lda import setup as lda_setup
from components.query import rank_pagerank_for_query
lda_results = lda_setup(T, 0, fresh=False)
ALL_MODES = [m.value for m in PRModes.get_evaluation_modes()]

code_to_docid = {p.code: p.id - 1 for p in Paper.select(Paper.id, Paper.code).iterator()}
years = {p.id - 1: p.year for p in Paper.select(Paper.id, Paper.year).iterator()}
adj = defaultdict(set)
for c in PaperCitation.select().iterator():
    a, b = c.citing_paper_id - 1, c.cited_paper_id - 1
    if a != b:
        adj[a].add(b)
        adj[b].add(a)
gs = json.load(GS_PATH.open())
golden = {q: {code_to_docid[c] for c in cs if c in code_to_docid} for q, cs in gs.items()}
golden = {q: g for q, g in golden.items() if g}

_cache = {}


def dists(src):
    if src in _cache:
        return _cache[src]
    d = {src: 0}
    fr = deque([src])
    while fr:
        n = fr.popleft()
        if d[n] >= CAP:
            continue
        for m in adj.get(n, ()):
            if m not in d:
                d[m] = d[n] + 1
                fr.append(m)
    _cache[src] = d
    return d


def csc(pred, gold):
    top = pred[:K]
    ts = set(top)
    fs = []
    for e in gold:
        if e in ts:
            fs.append(1.0)
            continue
        dd = dists(e)
        fs.append(max([1.0 / dd[s] for s in top if dd.get(s, 0) > 0] or [0.0]))
    rs = []
    for s in top:
        if s in gold:
            rs.append(1.0)
            continue
        dd = dists(s)
        rs.append(max([1.0 / dd[e] for e in gold if dd.get(e, 0) > 0] or [0.0]))
    f, r = sum(fs) / len(fs), sum(rs) / len(rs)
    return f, r, (math.sqrt(f * r) if f > 0 and r > 0 else 0.0)


def load(path):
    """Apply the aging factor exactly as csc_evaluation.py does: iterate Paper.select() so the
    resulting dict has the same key order (ties in the final ranking are broken by that order)."""
    res = pickle.load(path.open('rb'))
    agified = {}
    for paper in Paper.select(Paper.id, Paper.year).iterator():
        d = paper.id - 1
        if d in res:
            yd = max(1, 2011 - paper.year)
            agified[d] = {t: v / yd for t, v in res[d].items()}
    return agified


def evaluate(ranker):
    vals = [csc(ranker(q), golden[q]) for q in golden]
    return tuple(sum(v[i] for v in vals) / len(vals) for i in range(3))


out = {}
for mode in ALL_MODES:
    row = {}
    for label, d in (('prod', PR_PROD), ('tight', PR_TIGHT)):
        p = d / f'pageranks_{mode}.bin'
        if not p.exists():
            row[label] = None
            continue
        pr = load(p)
        row[label] = evaluate(lambda q: rank_pagerank_for_query(q, pr, lda_results))
    out[mode] = row
    fmt = lambda v: 'n/a' if v is None else f'{v[2]:.4f}'
    print(f'{mode:<24} prod OCSC {fmt(row["prod"])}   tight OCSC {fmt(row["tight"])}', flush=True)

from components.baselines import BM25Baseline
bm = BM25Baseline()
bm.build_index(fresh=False)
bm25 = evaluate(lambda q: [pid - 1 for pid, _ in bm.search(q, top_k=K)[:K]])
print('BM25 OCSC', round(bm25[2], 4))

with (GS_OUT / 'csc_tight_vs_production.csv').open('w') as f:
    f.write('mode,prod_fcsc,prod_rcsc,prod_ocsc,tight_fcsc,tight_rcsc,tight_ocsc\n')
    for m, r in out.items():
        a = r['prod'] or (None,) * 3
        b = r['tight'] or (None,) * 3
        f.write(f'{m},' + ','.join('' if v is None else f'{v:.6f}' for v in (*a, *b)) + '\n')

FAM = {'author': ('modified_bias', 'modified_transition'),
       'degree': ('d_modified_bias', 'd_modified_transition'),
       'closeness': ('cl_modified_bias', 'cl_modified_transition'),
       'betweenness': ('bw_modified_bias', 'bw_modified_transition'),
       'katz': ('kz_modified_bias', 'kz_modified_transition')}
summ = {'bm25_ocsc': round(bm25[2], 4)}
for label in ('prod', 'tight'):
    oc = {m: r[label][2] for m, r in out.items() if r[label]}
    if len(oc) < len(ALL_MODES):
        summ[label] = f'incomplete ({len(oc)}/{len(ALL_MODES)} modes)'
        continue
    best = max(oc, key=oc.get)
    summ[label] = {'best': best, 'best_ocsc': round(oc[best], 4),
                   'n_above_bm25': sum(v > bm25[2] for v in oc.values()),
                   'families_bias_gt_transition': {f: oc[b] > oc[t] for f, (b, t) in FAM.items()},
                   'ocsc': {m: round(v, 4) for m, v in oc.items()}}
(GS_OUT / 'csc_tight_summary.json').write_text(json.dumps(summ, indent=2), encoding='utf-8')
print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'ocsc'} if isinstance(v, dict) else v)
                  for k, v in summ.items()}, indent=2))
