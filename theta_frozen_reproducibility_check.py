"""
Validate the frozen-theta fix end to end, BEFORE committing to the expensive
full 19-mode rebuild: with GensimLDA.get_doc_topics() now reading from the
frozen artifact instead of calling gensim's get_document_topics()/inference()
at runtime, two independent PageRank passes for the same (mode, topic) should
now be EXACTLY byte-identical (not just close), since there is no remaining
source of runtime randomness in the classic pipeline.

Runs a SMALL sample (a few modes x a few topics, including a centrality mode
and an author-score mode) with all lru_caches cleared between passes -- same
methodology as theta_stability_check2.py, but checking for exact equality
instead of measuring a diff.
"""
import pickle
import time
from pathlib import Path

import networkx as nx

from components.database import setup as db_setup
db_setup()

from components.lda import setup as lda_setup
from components.base_pagerank import pr_output_path
from components.prmodes import PRModes
from components.pagerank import PaperRankClassic, transition_star
from components.query import query_topic_weights

ALPHA = 0.15
MAX_ITERATIONS = 100
EPSILON = 1.0e-6

print("Loading citation graph...", flush=True)
with open(pr_output_path.joinpath('graph.bin'), 'rb') as f:
    graph: nx.DiGraph = pickle.load(f)
W = nx.stochastic_graph(graph, weight='weight')
N = W.number_of_nodes()

print("Loading LDA (T=200)...", flush=True)
lda_results = lda_setup(200, 0, fresh=False)
lda = list(lda_results.values())[0]

SAMPLE_TOPICS = [0, 50, 100, 150]
SAMPLE_MODES = [PRModes.THEMED, PRModes.MODIFIED, PRModes.D_MODIFIED, PRModes.HYBRID]


def clear_all_caches():
    for attr in ('topic_doc_score', 'topic_docs_score', 'get_doc_topics',
                 'author_topic_scores', 'topic_authors_score', 'doc_authors_score'):
        fn = getattr(lda, attr, None)
        if fn is not None and hasattr(fn, 'cache_clear'):
            fn.cache_clear()
    transition_star.cache_clear()


def run_topic(mode, t):
    x = {d: PaperRankClassic.bias(mode, t, d, W, lda) for d in W}
    total = sum(x.values())
    x = {k: v / total for k, v in x.items()} if total > 0 else {d: 1.0 / N for d in W}
    biases = {**x}
    for _ in range(MAX_ITERATIONS):
        x_last = x
        x = dict.fromkeys(x_last.keys(), 0)
        dangling_mass = sum(x_last[d_prime] for d_prime in W if W.out_degree(d_prime) == 0)
        for d in x:
            random_jump = ALPHA * biases[d]
            dangling_component = (1 - ALPHA) * dangling_mass * biases[d]
            walk_component = (1 - ALPHA) * sum(
                PaperRankClassic.transition(mode, t, d, d_prime, W, lda) * x_last[d_prime]
                for d_prime, _ in W.in_edges(d)
            )
            x[d] = random_jump + dangling_component + walk_component
        total_x = sum(x.values())
        if total_x > 0:
            x = {k: v / total_x for k, v in x.items()}
        diff = sum(abs(x[n] - x_last[n]) for n in x)
        if diff < N * EPSILON:
            break
    return x


print(f"\nRunning {len(SAMPLE_MODES)} modes x {len(SAMPLE_TOPICS)} topics, TWICE each "
      f"(all caches cleared between passes), checking for EXACT equality...\n", flush=True)

all_exact = True
t_start = time.time()
for mode in SAMPLE_MODES:
    for t in SAMPLE_TOPICS:
        clear_all_caches()
        x_a = run_topic(mode, t)
        clear_all_caches()
        x_b = run_topic(mode, t)
        exact = x_a == x_b
        max_diff = max(abs(x_a[d] - x_b[d]) for d in x_a)
        all_exact = all_exact and exact
        print(f"  mode={mode.value:<22} topic={t:>4}  exact_equal={str(exact):<5}  max_diff={max_diff:.3e}",
              flush=True)

print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. ALL EXACT: {all_exact}")

# Query-side determinism: two independent calls, cache cleared, must be identical.
print("\nChecking query_topic_weights determinism...", flush=True)
for query in ['machine learning', 'graph neural networks', 'natural language processing']:
    lda.get_doc_topics.cache_clear()
    w_a = query_topic_weights(query, lda_results)
    lda.get_doc_topics.cache_clear()
    w_b = query_topic_weights(query, lda_results)
    print(f"  '{query}': exact_equal={w_a == w_b}, n_topics={len(w_a)}")
