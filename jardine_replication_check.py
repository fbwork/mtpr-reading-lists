"""
Jardine Replication Check
==========================
Standalone validation script -- NOT part of the MTPR paper's own evaluation
pipeline (which deliberately uses different parameters: K=10, its own query
mechanism; see Reports/*.tex). This script exists only to answer one
question: does our `themed` PageRank mode (= Jardine's TPR, topic bias +
topic-aware transition, no author score) reproduce Jardine (2014)'s reported
numbers when run under his exact experimental setup?

Matched to Jardine (UCAM-CL-TR-848) Table 5 / Section 3.1.3 / Section 5.2.1:
  - D = 20 recommended papers (NOT K=10, which the MTPR paper's own scripts use)
  - Query topic distribution xi: Lucene-style TF-IDF full-text search over
    (title + body) -> top 20 most similar papers -> average THEIR LDA topic
    distributions theta_d. ("unknown words or technical terms" mechanism,
    the one Jardine states was used for this exact experiment.)
  - The 8 gold-standard queries (95 gold papers total)
  - PageRank mode: themed, with linear age adjustment (PR / max(1, 2011-year))
  - Metrics: FCSC, RCSC, OCSC (Definition 4.1: shortest undirected citation
    path, inverse-linear decay, capped at 10 edges), plus Precision/Recall/F1/MAP

Target to reproduce (Jardine Table 5, TPR row):
    RCSC=0.456   FCSC=0.563   MAP=0.043   F=0.128   (RCP not computed here)

Usage:
    LDA_NO_TOPICS=200 python jardine_replication_check.py
    (defaults to 200, matching Jardine; use a smaller value like 5 for a
    quick sanity pass if 200-topic themed PageRank hasn't been computed yet)
"""
import json
import math
import os
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
LDA_NO_TOPICS = int(os.getenv('LDA_NO_TOPICS', '200'))
D = 20  # Jardine Table 5: 20 recommended papers, not K=10
NEIGHBOR_K = 20  # "20 highest scoring papers" averaged for the query topic distribution
MAX_HOPS = 10  # Jardine Definition 4.1 CSC cap

PR_OUTPUT = Path('components/output/pr') / str(LDA_NO_TOPICS) / str(LDA_NO_TOPICS)
GS_PATH = Path('components/resources/gs/golden_standard.json')

# ── Database + LDA setup ────────────────────────────────────────────────────
print("Connecting to database...")
from components.database import setup as db_setup
from models.tables import Paper, PaperCitation
db_setup()

from components.lda import setup as lda_setup
print(f"Loading LDA topic model ({LDA_NO_TOPICS} topics)...")
lda_results = lda_setup(LDA_NO_TOPICS, 0, fresh=False)
if not lda_results:
    raise RuntimeError(
        f"No {LDA_NO_TOPICS}-topic LDA model available. Run:\n"
        f"  python main.py lda --lda-no-topics {LDA_NO_TOPICS}"
    )
lda = lda_results[0]

pr_path = PR_OUTPUT / 'pageranks_themed.bin'
if not pr_path.exists():
    raise RuntimeError(
        f"No themed PageRank at {LDA_NO_TOPICS} topics. Run:\n"
        f"  python main.py pagerank --topic-model lda --pr-mode themed "
        f"--no-topics {LDA_NO_TOPICS} --lda-no-topics {LDA_NO_TOPICS}"
    )
print(f"Loading themed PageRank from {pr_path}...")
with pr_path.open('rb') as f:
    raw_pr = pickle.load(f)

# Linear age adjustment (Jardine's TPR default -- "AGE-LINEAR"): PR / max(1, 2011-year).
print("Applying linear age adjustment...")
themed_pr = {}
for paper in Paper.select(Paper.id, Paper.year).iterator():
    doc_id = paper.id - 1
    if doc_id in raw_pr:
        year_diff = max(1, 2011 - paper.year)
        themed_pr[doc_id] = {t: v / year_diff for t, v in raw_pr[doc_id].items()}

# ── Golden standard ──────────────────────────────────────────────────────────
print("Loading golden standard...")
with GS_PATH.open() as f:
    gs_raw = json.load(f)

code_to_docid = {p.code: p.id - 1 for p in Paper.select(Paper.id, Paper.code).iterator()}
golden_standard = {}
for query, codes in gs_raw.items():
    doc_ids = {code_to_docid[c] for c in codes if c in code_to_docid}
    missing = [c for c in codes if c not in code_to_docid]
    if missing:
        print(f"  [WARN] '{query}': {len(missing)} codes not found: {missing}")
    if doc_ids:
        golden_standard[query] = doc_ids
print(f"  {len(golden_standard)} queries, {sum(len(v) for v in golden_standard.values())} gold papers")

# ── Full-text Lucene-classic TF-IDF index (Jardine used Lucene.NET v2.9.2) ──
# sklearn's TfidfVectorizer uses smoothed-idf + L2-normalized vectors, which is
# NOT what Lucene's default TFIDFSimilarity computes. We implement Lucene's
# actual classic scoring formula instead (this was Lucene's default Similarity
# through the version Jardine used, and is also what the "TFIDF" ablation
# baseline in the thesis's own Section 5.1.6 relies on):
#   tf(t,d)    = sqrt(freq(t,d))
#   idf(t)     = 1 + ln(N / (df(t) + 1))
#   norm(d)    = 1 / sqrt(total token count in d)
#   coord(q,d) = |query terms found in d| / |query terms|
#   score(q,d) = coord(q,d) * norm(d) * sum_{t in q} tf(t,d) * idf(t)^2
print("Building full-text Lucene-classic TF-IDF index over (title + body)...")
from sklearn.feature_extraction.text import CountVectorizer

all_papers = list(Paper.select(Paper.id, Paper.title, Paper.body).order_by(Paper.id).iterator())
doc_texts = [(p.title or '') + ' ' + (p.body or '') for p in all_papers]
docid_order = [p.id - 1 for p in all_papers]  # row i of the matrix -> doc_id docid_order[i]

count_vectorizer = CountVectorizer(lowercase=True, stop_words='english', max_features=100_000)
tf_counts = count_vectorizer.fit_transform(doc_texts).tocsr()  # raw term counts, shape (N, V)
N = tf_counts.shape[0]

doc_freq = np.asarray((tf_counts > 0).sum(axis=0)).ravel()
idf = 1.0 + np.log(N / (doc_freq + 1.0))
idf_sq = idf ** 2

tf_sqrt = tf_counts.astype(np.float64)
tf_sqrt.data = np.sqrt(tf_sqrt.data)

# Per-term weight tf(t,d) * idf(t)^2; document-length norm applied at scoring time.
weighted = tf_sqrt.multiply(idf_sq).tocsr()

num_tokens = np.asarray(tf_counts.sum(axis=1)).ravel()
doc_norm = 1.0 / np.sqrt(np.maximum(num_tokens, 1))
print(f"  Term-document matrix: {weighted.shape}")


def query_topic_distribution(query_text: str) -> dict:
    """Jardine's 'unknown words or technical terms' query mechanism (Sec 3.1.3):
    Lucene-classic TF-IDF search -> top NEIGHBOR_K similar papers -> average
    their theta_d."""
    query_vec = count_vectorizer.transform([query_text])
    query_terms = query_vec.indices
    if len(query_terms) == 0:
        return {}

    matched = weighted[:, query_terms]  # (N, |query_terms|) sparse
    raw_scores = np.asarray(matched.sum(axis=1)).ravel()
    terms_matched = np.asarray((matched > 0).sum(axis=1)).ravel()
    coord = terms_matched / len(query_terms)
    scores = raw_scores * coord * doc_norm

    top_rows = np.argsort(scores)[::-1][:NEIGHBOR_K]
    top_doc_ids = [docid_order[r] for r in top_rows if scores[r] > 0]
    if not top_doc_ids:
        return {}
    xi = defaultdict(float)
    for doc_id in top_doc_ids:
        for t in range(LDA_NO_TOPICS):
            xi[t] += lda.topic_doc_score(doc_id, t)
    n = len(top_doc_ids)
    return {t: v / n for t, v in xi.items()}


def rank_papers_for_query(xi: dict) -> list:
    """TPR(q,d) = sum_t xi_t * themed_PR(t,d), descending."""
    scores = {
        doc_id: sum(xi.get(t, 0.0) * topic_scores.get(t, 0.0) for t in xi)
        for doc_id, topic_scores in themed_pr.items()
    }
    return sorted(scores, key=scores.get, reverse=True)


# ── Jardine-exact CSC (Definition 4.1): shortest undirected path, capped ────
print("Building undirected citation graph for CSC...")
outgoing = defaultdict(set)
for c in PaperCitation.select().iterator():
    citing, cited = c.citing_paper_id - 1, c.cited_paper_id - 1
    if citing != cited:
        outgoing[citing].add(cited)

undirected_adj = defaultdict(set)
for citing, cited_set in outgoing.items():
    for cited in cited_set:
        undirected_adj[citing].add(cited)
        undirected_adj[cited].add(citing)

_bfs_cache = {}


def bfs_distances(source, max_hops=MAX_HOPS):
    cached = _bfs_cache.get(source)
    if cached is not None:
        return cached
    dist = {source: 0}
    frontier = deque([source])
    while frontier:
        node = frontier.popleft()
        if dist[node] >= max_hops:
            continue
        for neighbor in undirected_adj.get(node, ()):
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                frontier.append(neighbor)
    _bfs_cache[source] = dist
    return dist


def csc_scores(top_d: list, gold_set: set) -> dict:
    top_set = set(top_d)
    fcsc_scores = []
    for e in gold_set:
        if e in top_set:
            fcsc_scores.append(1.0)
            continue
        dist = bfs_distances(e)
        best = max((1.0 / dist[s] for s in top_d if s in dist and dist[s] > 0), default=0.0)
        fcsc_scores.append(best)

    rcsc_scores = []
    for s in top_d:
        if s in gold_set:
            rcsc_scores.append(1.0)
            continue
        dist = bfs_distances(s)
        best = max((1.0 / dist[e] for e in gold_set if e in dist and dist[e] > 0), default=0.0)
        rcsc_scores.append(best)

    fcsc = sum(fcsc_scores) / len(fcsc_scores) if fcsc_scores else 0.0
    rcsc = sum(rcsc_scores) / len(rcsc_scores) if rcsc_scores else 0.0
    ocsc = math.sqrt(fcsc * rcsc) if fcsc > 0 and rcsc > 0 else 0.0
    return {'fcsc': fcsc, 'rcsc': rcsc, 'ocsc': ocsc}


def standard_ir_metrics(top_d: list, gold_set: set) -> dict:
    """Precision/Recall/F1/MAP@D, standard definitions (matches run_eval_all_fast.py)."""
    hits = set(top_d) & gold_set
    precision = len(hits) / len(top_d) if top_d else 0.0
    recall = len(hits) / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ap = 0.0
    for i, pid in enumerate(top_d, 1):
        if pid in gold_set:
            ap += len(set(top_d[:i]) & gold_set) / i
    map_score = ap / len(hits) if hits else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'map': map_score}


# ── Main loop ────────────────────────────────────────────────────────────────
print(f"\nRunning Jardine replication (D={D}, {LDA_NO_TOPICS} topics)...\n")
per_query = {}
for query, gold_set in golden_standard.items():
    xi = query_topic_distribution(query)
    if not xi:
        print(f"  [WARN] '{query}': no TF-IDF matches, skipping")
        continue
    ranked = rank_papers_for_query(xi)
    top_d = ranked[:D]
    metrics = {**csc_scores(top_d, gold_set), **standard_ir_metrics(top_d, gold_set)}
    per_query[query] = metrics
    print(f"  {query:<40} RCSC={metrics['rcsc']:.3f} FCSC={metrics['fcsc']:.3f} "
          f"OCSC={metrics['ocsc']:.3f} P={metrics['precision']:.3f} "
          f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f} MAP={metrics['map']:.3f}")

avg = {k: sum(m[k] for m in per_query.values()) / len(per_query) for k in
       ['rcsc', 'fcsc', 'ocsc', 'precision', 'recall', 'f1', 'map']}

print("\n" + "=" * 70)
print(f"{'Metric':<12} {'Ours':>10} {'Jardine Table 5':>18}")
print("-" * 70)
print(f"{'RCSC':<12} {avg['rcsc']:>10.3f} {0.456:>18.3f}")
print(f"{'FCSC':<12} {avg['fcsc']:>10.3f} {0.563:>18.3f}")
print(f"{'OCSC':<12} {avg['ocsc']:>10.3f} {'(not reported)':>18}")
print(f"{'MAP':<12} {avg['map']:>10.3f} {0.043:>18.3f}")
print(f"{'F-score':<12} {avg['f1']:>10.3f} {0.128:>18.3f}")
print(f"{'Precision':<12} {avg['precision']:>10.3f} {'(RCP not computed)':>18}")
print("=" * 70)

out_path = Path('components/resources/gs/jardine_replication_results.json')
with out_path.open('w') as f:
    json.dump({'per_query': per_query, 'average': avg, 'no_topics': LDA_NO_TOPICS, 'D': D}, f, indent=2)
print(f"\nSaved: {out_path}")
