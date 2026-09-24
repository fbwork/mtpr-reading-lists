"""
Cheap significance check using ONLY already-computed artifacts (no new
PageRank/LDA runs): Wilcoxon signed-rank tests on the 8 paired per-query
OCSC values for (1) bw_modified_bias vs BM25, and (2) bias vs transition
within each of the five weighting families. BM25's per-query OCSC was
not previously saved to disk (only the 8-query average was), so this
script recomputes it by querying the already-built, cached BM25 index
(components/output/baselines/bm25_index.pkl) -- no retraining, no new
experiments, just extracting a value that was already computed once per
query internally but not persisted.
"""
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path

from scipy.stats import wilcoxon

from components.database import setup as db_setup
db_setup()

from models.tables import Paper, PaperCitation
from components.baselines import BM25Baseline

GS_PATH = Path('components/resources/gs/golden_standard.json')
PERQUERY_CSV = Path('components/resources/gs/csc_per_query.csv')
MAX_HOPS = 10

print("Building paper id maps and citation graph for CSC...", flush=True)
code_to_docid = {}
for p in Paper.select(Paper.id, Paper.code).iterator():
    code_to_docid[p.code] = p.id - 1

outgoing = defaultdict(set)
for c in PaperCitation.select().iterator():
    outgoing[c.citing_paper_id - 1].add(c.cited_paper_id - 1)

undirected_adj = defaultdict(set)
for src, dsts in outgoing.items():
    for dst in dsts:
        if src == dst:
            continue
        undirected_adj[src].add(dst)
        undirected_adj[dst].add(src)

_bfs_cache = {}


def bfs_distances(source):
    cached = _bfs_cache.get(source)
    if cached is not None:
        return cached
    dist = {source: 0}
    frontier = deque([source])
    while frontier:
        node = frontier.popleft()
        if dist[node] >= MAX_HOPS:
            continue
        for nb in undirected_adj.get(node, ()):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                frontier.append(nb)
    _bfs_cache[source] = dist
    return dist


def evaluate_list(predicted_ids, gold_set, k=10):
    top_k = predicted_ids[:k]
    top_k_set = set(top_k)
    if not top_k:
        return 0.0
    fcsc_scores = []
    for e in gold_set:
        if e in top_k_set:
            fcsc_scores.append(1.0)
            continue
        dist = bfs_distances(e)
        best = 0.0
        for s in top_k:
            d = dist.get(s)
            if d is not None and d > 0:
                best = max(best, 1.0 / d)
        fcsc_scores.append(best)
    rcsc_scores = []
    for s in top_k:
        if s in gold_set:
            rcsc_scores.append(1.0)
            continue
        dist = bfs_distances(s)
        best = 0.0
        for e in gold_set:
            d = dist.get(e)
            if d is not None and d > 0:
                best = max(best, 1.0 / d)
        rcsc_scores.append(best)
    fcsc = sum(fcsc_scores) / len(fcsc_scores) if fcsc_scores else 0.0
    rcsc = sum(rcsc_scores) / len(rcsc_scores) if rcsc_scores else 0.0
    return math.sqrt(fcsc * rcsc) if (fcsc > 0 and rcsc > 0) else 0.0


print("Loading golden standard...", flush=True)
with GS_PATH.open() as f:
    gs_raw = json.load(f)
queries = {}
for q, codes in gs_raw.items():
    doc_ids = {code_to_docid[c] for c in codes if c in code_to_docid}
    if doc_ids:
        queries[q] = doc_ids
print(f"  {len(queries)} queries", flush=True)

print("Loading cached BM25 index (no retraining)...", flush=True)
bm25 = BM25Baseline()
bm25.build_index(fresh=False)

bm25_per_query = {}
for q, gold in queries.items():
    results = bm25.search(q, top_k=10)
    predicted_doc_ids = [pid - 1 for pid, _ in results]
    bm25_per_query[q] = evaluate_list(predicted_doc_ids, gold, k=10)

print("\nBM25 per-query OCSC (recomputed from cached index):", flush=True)
for q, v in bm25_per_query.items():
    print(f"  {q:<40s} {v:.4f}", flush=True)
print(f"  mean = {sum(bm25_per_query.values())/len(bm25_per_query):.4f} "
      f"(reported aggregate: 0.5568)", flush=True)

print("\nLoading per-query PageRank OCSC from csc_per_query.csv...", flush=True)
pr_per_query = defaultdict(dict)
with open(PERQUERY_CSV, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        pr_per_query[row['mode']][row['query']] = float(row['ocsc'])

query_order = list(queries.keys())

# ── Test 1: bw_modified_bias vs BM25 ────────────────────────────────────────
bw_bias_vals = [pr_per_query['bw_modified_bias'][q] for q in query_order]
bm25_vals = [bm25_per_query[q] for q in query_order]
diffs = [a - b for a, b in zip(bw_bias_vals, bm25_vals)]
print(f"\n=== bw_modified_bias vs BM25 (n=8 paired queries) ===", flush=True)
print(f"  per-query diffs: {[f'{d:+.3f}' for d in diffs]}", flush=True)
try:
    stat, p = wilcoxon(bw_bias_vals, bm25_vals)
    print(f"  Wilcoxon signed-rank: W={stat:.2f}, p={p:.4f}", flush=True)
except ValueError as e:
    print(f"  Wilcoxon test could not be computed: {e}", flush=True)

# ── Test 2: bias vs transition within each family ───────────────────────────
families = ['modified', 'd_modified', 'cl_modified', 'bw_modified', 'kz_modified']
print(f"\n=== Bias vs transition, per family (n=8 paired queries each) ===", flush=True)
for fam in families:
    bias_vals = [pr_per_query[f'{fam}_bias'][q] for q in query_order]
    trans_vals = [pr_per_query[f'{fam}_transition'][q] for q in query_order]
    diffs = [a - b for a, b in zip(bias_vals, trans_vals)]
    n_pos = sum(1 for d in diffs if d > 0)
    try:
        stat, p = wilcoxon(bias_vals, trans_vals)
        print(f"  {fam:<15s} bias>trans in {n_pos}/8 queries, W={stat:.2f}, p={p:.4f}", flush=True)
    except ValueError as e:
        print(f"  {fam:<15s} bias>trans in {n_pos}/8 queries, test failed: {e}", flush=True)

# ── Test 3: pooled across all 5 families (n=40 paired observations) ────────
print(f"\n=== Pooled bias vs transition across all 5 families (n=40) ===", flush=True)
all_bias, all_trans = [], []
for fam in families:
    for q in query_order:
        all_bias.append(pr_per_query[f'{fam}_bias'][q])
        all_trans.append(pr_per_query[f'{fam}_transition'][q])
stat, p = wilcoxon(all_bias, all_trans)
print(f"  W={stat:.2f}, p={p:.6f}", flush=True)

print("\nDone.", flush=True)
