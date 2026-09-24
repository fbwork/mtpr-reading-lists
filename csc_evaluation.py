"""
Citation Substitution Coefficient (CSC) Evaluation
Based on: Jardine, J.G. (2014). UCAM-CL-TR-848
          "Measuring the Similarity of Reading Lists"

CSC evaluates recommended reading lists by giving partial credit to papers
that are citation-graph-neighbors of gold standard papers, rather than
requiring exact matches. This is more appropriate for academic paper
recommendation because citation-adjacent papers often cover similar content.

Two variants (TR-848 Definition 4.1 / Sec 3.3.1, implemented exactly, not
approximated by direct-citation overlap):
  FCSC (Forward): for each expert (gold) paper Ei, max_j{1/C[Ei,Sj]} over the
                  top-k system papers Sj; C[Ei,Sj] = shortest UNDIRECTED
                  citation-graph path length in edges, capped at MAX_HOPS.
  RCSC (Reverse): for each system paper Sj, max_i{1/C[Ei,Sj]} over the gold
                  papers Ei.
  OCSC (Overall): geometric mean of FCSC_system and RCSC_system.

  FCSC_system = mean over ALL gold papers of their FCSC score.
  RCSC_system = mean over the top-k predicted papers of their RCSC score.
  An exact match (paper present in both lists) scores 1. Unconnected pairs,
  or pairs farther than MAX_HOPS edges apart, score 0.
"""

import json
import os
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
LDA_NO_TOPICS = int(os.getenv('LDA_NO_TOPICS', '200'))
NO_TOPICS = LDA_NO_TOPICS  # PageRank output dir is keyed by the same topic count.
K = 10
AGIFY = True
CSC_CACHE_VERSION = 'query-aware-v2'

# Single source of truth: components/prmodes.py::PRModes.get_evaluation_modes().
# Toggle PRModes.INCLUDE_EXPERIMENTAL_MODES there to add GAT/Flow Matching back in.
from components.prmodes import PRModes
ALL_MODES = [m.value for m in PRModes.get_evaluation_modes()]

PR_OUTPUT = Path('components/output/pr') / str(NO_TOPICS) / str(NO_TOPICS)
GS_PATH   = Path('components/resources/gs/golden_standard.json')
GS_OUT    = Path('components/resources/gs')

# ── Database setup ───────────────────────────────────────────────────────────
print("Connecting to database...")
from components.database import setup as db_setup
from models.tables import Paper, PaperCitation
db_setup()

from components.lda import setup as lda_setup
from components.query import rank_pagerank_for_query
print(f"Loading LDA topic model for query-aware scoring ({LDA_NO_TOPICS} topics)...")
lda_results = lda_setup(LDA_NO_TOPICS, 0, fresh=False)
if not lda_results:
    raise RuntimeError("No LDA model is available for query-aware CSC evaluation.")

# ── Build code <-> doc_id mappings ───────────────────────────────────────────
print("Building paper mappings...")
code_to_docid = {}
docid_to_code = {}
for p in Paper.select(Paper.id, Paper.code).iterator():
    code_to_docid[p.code] = p.id - 1   # 0-based doc_id
    docid_to_code[p.id - 1] = p.code
print(f"  {len(code_to_docid)} papers")

# ── Build citation graph (outgoing and incoming) ─────────────────────────────
print("Building citation graph from database...")
# outgoing[d] = set of doc_ids that d cites
# incoming[d] = set of doc_ids that cite d
outgoing = defaultdict(set)
incoming = defaultdict(set)

for citation in PaperCitation.select().iterator():
    citing_id = citation.citing_paper_id - 1   # 0-based
    cited_id  = citation.cited_paper_id  - 1   # 0-based
    outgoing[citing_id].add(cited_id)
    incoming[cited_id].add(citing_id)

print(f"  {sum(len(v) for v in outgoing.values())} citations loaded")

# ── Load golden standard ─────────────────────────────────────────────────────
print("Loading golden standard...")
with GS_PATH.open() as f:
    gs_raw = json.load(f)

# Convert codes to doc_ids
golden_standard = {}
for query, codes in gs_raw.items():
    doc_ids = [code_to_docid[c] for c in codes if c in code_to_docid]
    if doc_ids:
        golden_standard[query] = set(doc_ids)
    else:
        print(f"  [WARN] Query '{query}': no matching papers in DB")

print(f"  {len(golden_standard)} queries, "
      f"{sum(len(v) for v in golden_standard.values())} total gold papers")


# ── CSC functions ────────────────────────────────────────────────────────────

# Jardine (2014, Sec 3.3.1 / Definition 4.1): C[Ei,Sj] is the number of edges
# in the shortest (undirected) citation-graph path between an expert paper and
# a system paper. Up to MAX_HOPS edges are followed; beyond that, score is 0.
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


def csc_list(predicted_ids: list, gold_set: set, k: int = 10) -> dict:
    """
    Compute FCSC, RCSC, OCSC for a predicted list against a gold set, following
    Jardine (2014) Definition 4.1 exactly (shortest undirected citation-graph
    path, inverse-linear decay, capped at MAX_HOPS edges) rather than a
    direct-citation-only approximation.

    FCSC_system = mean over ALL gold papers Ei of max_j{1/C[Ei,Sj]} (Sj in top-k).
    RCSC_system = mean over the top-k system papers Sj of max_i{1/C[Ei,Sj]}.
    OCSC = sqrt(FCSC_system * RCSC_system).
    """
    top_k = predicted_ids[:k]
    top_k_set = set(top_k)

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

    exact_matches = len(top_k_set & gold_set)
    n = len(top_k)
    if n == 0:
        return {'precision': 0, 'fcsc': 0, 'rcsc': 0, 'ocsc': 0, 'exact': 0}

    precision = exact_matches / n
    fcsc = sum(fcsc_scores) / len(fcsc_scores) if fcsc_scores else 0.0
    rcsc = sum(rcsc_scores) / len(rcsc_scores) if rcsc_scores else 0.0
    import math
    ocsc = math.sqrt(fcsc * rcsc) if (fcsc > 0 and rcsc > 0) else 0.0

    return {
        'precision': precision,
        'fcsc': fcsc,
        'rcsc': rcsc,
        'ocsc': ocsc,
        'exact': exact_matches
    }


# ── Load PageRank results ─────────────────────────────────────────────────────

def load_pagerank(mode: str):
    """Load cached PageRank results and optionally apply aging."""
    path = PR_OUTPUT / f'pageranks_{mode}.bin'
    if not path.exists():
        return None
    with path.open('rb') as f:
        results = pickle.load(f)

    if AGIFY:
        try:
            agified = {}
            for paper in Paper.select(Paper.id, Paper.year).iterator():
                doc_id = paper.id - 1
                if doc_id in results:
                    year_diff = max(1, 2011 - paper.year)
                    agified[doc_id] = {
                        t: v / year_diff for t, v in results[doc_id].items()
                    }
            return agified
        except Exception as e:
            print(f"  [WARN] aging failed: {e}")
            return results
    return results


def rank_papers_for_query(pr_results: dict, query: str) -> list:
    """Rank PageRank output with the LDA topic distribution of this query."""
    return rank_pagerank_for_query(query, pr_results, lda_results)


# ── Main evaluation loop ──────────────────────────────────────────────────────
print(f"\nCSC Evaluation: {len(ALL_MODES)} modes, {len(golden_standard)} queries, K={K}\n")
print(f"{'Mode':<28} {'P@10':>8} {'FCSC':>8} {'RCSC':>8} {'OCSC':>8} {'Status'}")
print("-" * 75)

all_results = {}
per_query_results = {}  # mode -> list of per-query score dicts (supplementary material)

for mode in ALL_MODES:
    pr = load_pagerank(mode)
    if pr is None:
        print(f"  {mode:<26}  [MISSING]")
        continue

    query_scores = []

    for query, gold_set in golden_standard.items():
        ranked = rank_papers_for_query(pr, query)
        scores = csc_list(ranked, gold_set, k=K)
        scores['query'] = query
        query_scores.append(scores)

    if not query_scores:
        continue

    per_query_results[mode] = query_scores

    avg = {
        metric: sum(s[metric] for s in query_scores) / len(query_scores)
        for metric in ['precision', 'fcsc', 'rcsc', 'ocsc']
    }
    all_results[mode] = avg

    print(f"  {mode:<26} {avg['precision']:>8.4f} {avg['fcsc']:>8.4f} "
          f"{avg['rcsc']:>8.4f} {avg['ocsc']:>8.4f}")

# ── Sort by OCSC ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("RANKED BY OCSC (Overall Citation Substitution Coefficient):\n")
ranked_modes = sorted(all_results.items(), key=lambda x: x[1]['ocsc'], reverse=True)
for rank, (mode, m) in enumerate(ranked_modes[:10], 1):
    print(f"  {rank:>2}. {mode:<28} OCSC={m['ocsc']:.4f}  FCSC={m['fcsc']:.4f}  RCSC={m['rcsc']:.4f}  P@10={m['precision']:.4f}")

# ── Save results ─────────────────────────────────────────────────────────────

# JSON. The version marker makes previous global-ranking CSC artifacts invalid.
metadata_out = GS_OUT / 'csc_results.meta.json'
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
json_out = GS_OUT / 'csc_results.json'
with json_out.open('w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nJSON saved: {json_out}")

# CSV
csv_out = GS_OUT / 'csc_evaluation_summary.csv'
with csv_out.open('w') as f:
    f.write("mode,precision@10,fcsc,rcsc,ocsc\n")
    for mode, m in all_results.items():
        f.write(f"{mode},{m['precision']:.6f},{m['fcsc']:.6f},"
                f"{m['rcsc']:.6f},{m['ocsc']:.6f}\n")
print(f"CSV saved: {csv_out}")

# Per-query CSV (supplementary material; supports paired bias-vs-transition
# comparisons per query without re-running PageRank).
per_query_csv_out = GS_OUT / 'csc_per_query.csv'
with per_query_csv_out.open('w', encoding='utf-8') as f:
    f.write("mode,query,precision@10,fcsc,rcsc,ocsc\n")
    for mode, scores_list in per_query_results.items():
        for s in scores_list:
            f.write(f"{mode},{s['query']},{s['precision']:.6f},{s['fcsc']:.6f},"
                    f"{s['rcsc']:.6f},{s['ocsc']:.6f}\n")
print(f"Per-query CSV saved: {per_query_csv_out}")

# LaTeX table
tex_lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{CSC Evaluation of PageRank Modes (K=10)}",
    r"\label{tab:csc_eval}",
    r"\begin{tabular}{|l|c|c|c|c|}",
    r"\hline",
    r"Method & P@10 & FCSC & RCSC & OCSC \\ \hline",
]
for mode, m in sorted(all_results.items(), key=lambda x: x[1]['ocsc'], reverse=True):
    mode_esc = mode.replace('_', r'\_')
    tex_lines.append(
        f"{mode_esc} & {m['precision']:.4f} & {m['fcsc']:.4f} & "
        f"{m['rcsc']:.4f} & {m['ocsc']:.4f} \\\\"
    )
tex_lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

tex_out = GS_OUT / 'csc_comparison_table.tex'
with tex_out.open('w') as f:
    f.write("\n".join(tex_lines))
print(f"LaTeX saved: {tex_out}")

# Best results
if all_results:
    best_ocsc = max(all_results, key=lambda m: all_results[m]['ocsc'])
    best_fcsc = max(all_results, key=lambda m: all_results[m]['fcsc'])
    best_rcsc = max(all_results, key=lambda m: all_results[m]['rcsc'])
    print(f"\nBest OCSC : {best_ocsc}  ({all_results[best_ocsc]['ocsc']:.4f})")
    print(f"Best FCSC : {best_fcsc}  ({all_results[best_fcsc]['fcsc']:.4f})")
    print(f"Best RCSC : {best_rcsc}  ({all_results[best_rcsc]['rcsc']:.4f})")

print("\nDone.")