"""
Complete CSC Evaluation: All 30 PageRank modes + All baselines
Outputs the final comparison table for the paper.

Baselines evaluated:
  - BM25 (Okapi BM25 on technical terms)
  - Doc2Vec (Gensim document embeddings)
  - SPECTER (allenai/specter citation-aware BERT)
  - SciBERT (allenai/scibert_scivocab_uncased)
  - CF (Collaborative Filtering via SVD on citation matrix)

PageRank modes: 30 variants (normal, bw_modified, hybrid, GAT, etc.)

CSC metric (Jardine 2014, UCAM-CL-TR-848, Definition 4.1): for each expert
(gold) paper Ei, FCSC = max over top-k system papers Sj of 1/C[Ei,Sj], where
C is the shortest UNDIRECTED citation-graph path length in edges (capped at
MAX_HOPS); RCSC is the symmetric max over gold papers for each system paper.
OCSC = sqrt(FCSC_system * RCSC_system).
"""

import json
import math
import os
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
LDA_NO_TOPICS = int(os.getenv('LDA_NO_TOPICS', '200'))
NO_TOPICS = LDA_NO_TOPICS  # PageRank output dir is keyed by the same topic count.
K = 10
AGIFY = True
CSC_CACHE_VERSION = 'query-aware-v2'

# Single source of truth: components/prmodes.py::PRModes.get_evaluation_modes().
# Toggle PRModes.INCLUDE_EXPERIMENTAL_MODES there to add GAT/Flow Matching back in.
from components.prmodes import PRModes
PR_MODES = [m.value for m in PRModes.get_evaluation_modes()]

PR_OUTPUT = Path('components/output/pr') / str(NO_TOPICS) / str(NO_TOPICS)
GS_PATH   = Path('components/resources/gs/golden_standard.json')
GS_OUT    = Path('components/resources/gs')

# ── Database + mappings ───────────────────────────────────────────────────────
print("Connecting to database...")
from components.database import setup as db_setup
from models.tables import Paper, PaperCitation
db_setup()

from components.lda import setup as lda_setup
from components.query import rank_pagerank_for_query
print(f"Loading LDA topic model for query-aware PageRank scoring ({LDA_NO_TOPICS} topics)...")
lda_results = lda_setup(LDA_NO_TOPICS, 0, fresh=False)
if not lda_results:
    raise RuntimeError("No LDA model is available for query-aware CSC evaluation.")

print("Building paper mappings...")
code_to_docid = {}  # ACL code -> 0-based doc_id
docid_to_1based = {}  # 0-based -> 1-based paper_id
paper1based_to_docid = {}  # 1-based -> 0-based

for p in Paper.select(Paper.id, Paper.code).iterator():
    doc_id = p.id - 1
    code_to_docid[p.code] = doc_id
    docid_to_1based[doc_id] = p.id
    paper1based_to_docid[p.id] = doc_id
print(f"  {len(code_to_docid)} papers")

# ── Build citation graph ─────────────────────────────────────────────────────
print("Building citation graph...")
outgoing = defaultdict(set)   # doc_id -> set of doc_ids it cites
incoming = defaultdict(set)   # doc_id -> set of doc_ids that cite it

for c in PaperCitation.select().iterator():
    src = c.citing_paper_id - 1
    dst = c.cited_paper_id - 1
    outgoing[src].add(dst)
    incoming[dst].add(src)
print(f"  {sum(len(v) for v in outgoing.values())} edges")

# ── Golden standard ───────────────────────────────────────────────────────────
print("Loading golden standard...")
with GS_PATH.open() as f:
    gs_raw = json.load(f)

queries = {}  # query_text -> set of gold doc_ids
for q, codes in gs_raw.items():
    doc_ids = {code_to_docid[c] for c in codes if c in code_to_docid}
    if doc_ids:
        queries[q] = doc_ids

print(f"  {len(queries)} queries, "
      f"{sum(len(v) for v in queries.values())} total gold doc_ids")


# ── CSC functions ─────────────────────────────────────────────────────────────
MAX_HOPS = 10

_undirected_adj = defaultdict(set)
for _citing, _cited_set in outgoing.items():
    for _cited in _cited_set:
        if _citing == _cited:
            continue  # ignore self-citation edges
        _undirected_adj[_citing].add(_cited)
        _undirected_adj[_cited].add(_citing)

_bfs_cache: dict = {}


def _bfs_distances(source: int, max_hops: int = MAX_HOPS) -> dict:
    """Cached BFS edge-distances from ``source``, capped at ``max_hops``."""
    cached = _bfs_cache.get(source)
    if cached is not None:
        return cached
    dist = {source: 0}
    frontier = deque([source])
    while frontier:
        node = frontier.popleft()
        if dist[node] >= max_hops:
            continue
        for neighbor in _undirected_adj.get(node, ()):
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                frontier.append(neighbor)
    _bfs_cache[source] = dist
    return dist


def evaluate_list(predicted_ids: list, gold_set: set, k: int = 10) -> dict:
    """FCSC/RCSC/OCSC + Precision for a predicted list vs gold set (Jardine Def. 4.1)."""
    top_k = predicted_ids[:k]
    top_k_set = set(top_k)
    if not top_k:
        return {'precision': 0, 'fcsc': 0, 'rcsc': 0, 'ocsc': 0}

    fcsc_scores = []
    for e in gold_set:
        if e in top_k_set:
            fcsc_scores.append(1.0)
            continue
        dist = _bfs_distances(e)
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
        dist = _bfs_distances(s)
        best = 0.0
        for e in gold_set:
            d = dist.get(e)
            if d is not None and d > 0:
                best = max(best, 1.0 / d)
        rcsc_scores.append(best)

    exact = len(top_k_set & gold_set)
    fcsc = sum(fcsc_scores) / len(fcsc_scores) if fcsc_scores else 0.0
    rcsc = sum(rcsc_scores) / len(rcsc_scores) if rcsc_scores else 0.0
    ocsc = math.sqrt(fcsc * rcsc) if (fcsc > 0 and rcsc > 0) else 0.0

    return {
        'precision': exact / len(top_k),
        'fcsc': fcsc,
        'rcsc': rcsc,
        'ocsc': ocsc,
    }


def avg_over_queries(query_rankings: dict, k: int = 10) -> dict:
    """Average CSC over per-query rankings; never reuse a global ranking."""
    all_q = [
        evaluate_list(query_rankings.get(query, []), gold_set, k)
        for query, gold_set in queries.items()
    ]
    if not all_q:
        return {'precision': 0, 'fcsc': 0, 'rcsc': 0, 'ocsc': 0}
    return {m: sum(q[m] for q in all_q) / len(all_q) for m in ['precision', 'fcsc', 'rcsc', 'ocsc']}


# ── PageRank evaluation ───────────────────────────────────────────────────────
def load_pr(mode: str):
    """Load PageRank results, apply aging if AGIFY."""
    path = PR_OUTPUT / f'pageranks_{mode}.bin'
    if not path.exists():
        return None
    with path.open('rb') as f:
        results = pickle.load(f)
    if AGIFY:
        try:
            ag = {}
            for paper in Paper.select(Paper.id, Paper.year).iterator():
                d = paper.id - 1
                if d in results:
                    yd = max(1, 2011 - paper.year)
                    ag[d] = {t: v / yd for t, v in results[d].items()}
            return ag
        except Exception:
            pass
    return results


def rank_from_pr(pr: dict, query: str) -> list:
    """Rank one PageRank result set using this query's LDA topic weights."""
    return rank_pagerank_for_query(query, pr, lda_results)


print(f"\n{'Method':<30} {'P@10':>8} {'FCSC':>8} {'RCSC':>8} {'OCSC':>8}")
print("=" * 65)

all_results = {}

for mode in PR_MODES:
    pr = load_pr(mode)
    if pr is None:
        print(f"  {mode:<28}  [MISSING]")
        continue
    query_rankings = {query: rank_from_pr(pr, query) for query in queries}
    m = avg_over_queries(query_rankings, K)
    all_results[f'PR:{mode}'] = m
    print(f"  {mode:<28} {m['precision']:>8.4f} {m['fcsc']:>8.4f} {m['rcsc']:>8.4f} {m['ocsc']:>8.4f}")


# ── Baseline evaluation ───────────────────────────────────────────────────────
print("\n--- Baselines ---")
from components.baselines import BM25Baseline, Doc2VecBaseline, BERTBaseline, CollaborativeFilteringBaseline
from components.database import setup as db_setup

BASELINE_PATH = Path('components/output/baselines')

# Helper: query-averaged CSC from a {query_text -> [(paper_id_1based, score)]} dict
def eval_query_results(query_results: dict) -> dict:
    """
    query_results: {query_text: [(paper_1based_id, score), ...]}
    Converts 1-based paper_ids to 0-based doc_ids, then evaluates CSC.
    """
    all_q = []
    for query, results in query_results.items():
        if query not in queries:
            continue
        gold_set = queries[query]
        pred_docids = [paper1based_to_docid.get(pid, pid - 1) for pid, _ in results[:K]]
        all_q.append(evaluate_list(pred_docids, gold_set, K))
    if not all_q:
        return {'precision': 0, 'fcsc': 0, 'rcsc': 0, 'ocsc': 0}
    return {m: sum(q[m] for q in all_q) / len(all_q) for m in ['precision', 'fcsc', 'rcsc', 'ocsc']}


# ── BM25 ─────────────────────────────────────────────────────────────────────
try:
    bm25 = BM25Baseline()
    bm25.build_index(fresh=False)
    qr = {q: bm25.search(q, top_k=K) for q in queries}
    m = eval_query_results(qr)
    all_results['BM25'] = m
    print(f"  {'BM25':<28} {m['precision']:>8.4f} {m['fcsc']:>8.4f} {m['rcsc']:>8.4f} {m['ocsc']:>8.4f}")
except Exception as e:
    print(f"  {'BM25':<28}  [ERROR: {e}]")

# ── Doc2Vec ───────────────────────────────────────────────────────────────────
try:
    d2v = Doc2VecBaseline()
    d2v.build_model(fresh=False)
    qr = {q: d2v.search(q, top_k=K) for q in queries}
    m = eval_query_results(qr)
    all_results['Doc2Vec'] = m
    print(f"  {'Doc2Vec':<28} {m['precision']:>8.4f} {m['fcsc']:>8.4f} {m['rcsc']:>8.4f} {m['ocsc']:>8.4f}")
except Exception as e:
    print(f"  {'Doc2Vec':<28}  [ERROR: {e}]")

# ── SPECTER / SciBERT ─────────────────────────────────────────────────────────
# A document-embedding cache contains no query encoder.  Ranking it by vector
# norm was global and invalid for CSC; do not report it as a baseline result.
for baseline_name in ('SPECTER', 'SciBERT'):
    print(f"  {baseline_name:<28}  [SKIPPED: cached embeddings lack a query encoder]")


# ── Final ranked table ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL RANKING BY OCSC:\n")

ranked = sorted(all_results.items(), key=lambda x: x[1]['ocsc'], reverse=True)
for rank, (name, m) in enumerate(ranked[:15], 1):
    label = name.replace('PR:', '')
    print(f"  {rank:>2}. {label:<28} OCSC={m['ocsc']:.4f}  "
          f"FCSC={m['fcsc']:.4f}  RCSC={m['rcsc']:.4f}  P@10={m['precision']:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
metadata_out = GS_OUT / 'full_csc_results.meta.json'
metadata_out.write_text(
    json.dumps({
        'cache_version': CSC_CACHE_VERSION,
        'query_aware': True,
        'lda_no_topics': LDA_NO_TOPICS,
        'no_topics': NO_TOPICS,
        'k': K,
    }, indent=2),
    encoding='utf-8'
)
json_out = GS_OUT / 'full_csc_results.json'
with json_out.open('w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nJSON: {json_out}")

csv_out = GS_OUT / 'full_csc_summary.csv'
with csv_out.open('w') as f:
    f.write("method,precision@10,fcsc,rcsc,ocsc\n")
    for name, m in sorted(all_results.items(), key=lambda x: x[1]['ocsc'], reverse=True):
        f.write(f"{name.replace('PR:','')},{m['precision']:.6f},{m['fcsc']:.6f},{m['rcsc']:.6f},{m['ocsc']:.6f}\n")
print(f"CSV: {csv_out}")

# LaTeX — only top rows + all baselines
tex_lines = [
    r"\begin{table*}[htbp]",
    r"\centering",
    r"\caption{Comparison of PageRank variants and baseline methods using CSC evaluation (K=10). "
    r"FCSC = Forward CSC, RCSC = Reverse CSC, OCSC = Overall CSC (geometric mean). "
    r"Bold = best per column.}",
    r"\label{tab:full_csc}",
    r"\begin{tabular}{|l|c|c|c|c|}",
    r"\hline",
    r"\textbf{Method} & \textbf{P@10} & \textbf{FCSC} & \textbf{RCSC} & \textbf{OCSC} \\ \hline",
    r"\multicolumn{5}{|c|}{\textit{Baseline Methods}} \\ \hline",
]

baseline_names = ['BM25', 'Doc2Vec', 'SPECTER', 'SciBERT']
for name in baseline_names:
    if name in all_results:
        m = all_results[name]
        tex_lines.append(
            f"{name} & {m['precision']:.4f} & {m['fcsc']:.4f} & {m['rcsc']:.4f} & {m['ocsc']:.4f} \\\\"
        )

tex_lines.append(r"\hline")
tex_lines.append(r"\multicolumn{5}{|c|}{\textit{Proposed PageRank Variants (Top 10 by OCSC)}} \\ \hline")

pr_ranked = [(k, v) for k, v in ranked if k.startswith('PR:')]
for name, m in pr_ranked[:10]:
    label = name.replace('PR:', '').replace('_', r'\_')
    tex_lines.append(
        f"{label} & {m['precision']:.4f} & {m['fcsc']:.4f} & {m['rcsc']:.4f} & {m['ocsc']:.4f} \\\\"
    )

tex_lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]

tex_out = GS_OUT / 'full_csc_table.tex'
with tex_out.open('w') as f:
    f.write("\n".join(tex_lines))
print(f"LaTeX: {tex_out}")
print("\nDone.")