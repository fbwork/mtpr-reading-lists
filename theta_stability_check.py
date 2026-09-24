"""
Cheap empirical test: does PR(d,t) itself (not just the query side) actually
change between two independent fresh computations, given that
get_document_topics() re-infers theta_d non-deterministically for ~130
sparse (1-3 ATR term) documents out of 15,666?

Runs the SAME page_rank() logic used in production (copied verbatim, same
per-iteration renormalization) for a sample of topics, TWICE, in two fully
separate calls with lru_cache cleared between them (forcing fresh inference
each time -- no reuse of in-memory theta_d). Tests THEMED (pure topic-based,
no centrality/author confound) since it isolates exactly the effect in
question. Reports max/mean absolute difference in the converged PageRank
vector between run A and run B, for each sampled topic.
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
from components.pagerank import PaperRankClassic

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

# Find topics that actually include at least one of the sparse (1-3 term) docs,
# to bias the sample toward the highest-risk case rather than random topics.
sparse_doc_ids = {i for i, bow in enumerate(lda.bag_of_words) if 1 <= len(bow) <= 3}
print(f"Sparse (1-3 ATR term) documents in corpus: {len(sparse_doc_ids)}", flush=True)

SAMPLE_TOPICS = list(range(0, 200, 10))  # 20 topics, spread across the range


def run_topic(t):
    x = {d: PaperRankClassic.bias(PRModes.THEMED, t, d, W, lda) for d in W}
    total = sum(x.values())
    x = {k: v / total for k, v in x.items()} if total > 0 else {d: 1.0 / N for d in W}
    biases = {**x}

    for _iteration in range(MAX_ITERATIONS):
        x_last = x
        x = dict.fromkeys(x_last.keys(), 0)
        dangling_mass = sum(x_last[d_prime] for d_prime in W if W.out_degree(d_prime) == 0)
        for d in x:
            random_jump = ALPHA * biases[d]
            dangling_component = (1 - ALPHA) * dangling_mass * biases[d]
            walk_component = (1 - ALPHA) * sum(
                PaperRankClassic.transition(PRModes.THEMED, t, d, d_prime, W, lda) * x_last[d_prime]
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


def clear_caches():
    for attr in ('topic_doc_score', 'topic_docs_score', 'get_doc_topics'):
        fn = getattr(lda, attr, None)
        if fn is not None and hasattr(fn, 'cache_clear'):
            fn.cache_clear()


print(f"\nRunning THEMED PageRank for {len(SAMPLE_TOPICS)} sample topics, TWICE each "
      f"(cache cleared between the two runs)...\n", flush=True)

results_a = {}
results_b = {}
t_start = time.time()
for t in SAMPLE_TOPICS:
    clear_caches()
    results_a[t] = run_topic(t)
for t in SAMPLE_TOPICS:
    clear_caches()
    results_b[t] = run_topic(t)
print(f"Both runs done in {(time.time() - t_start) / 60:.1f} min\n", flush=True)

print(f"{'topic':>6s} {'max_abs_diff':>14s} {'mean_abs_diff':>14s} {'sum_a':>10s} {'sum_b':>10s} {'identical':>10s}")
overall_max = 0.0
for t in SAMPLE_TOPICS:
    a, b = results_a[t], results_b[t]
    diffs = [abs(a[d] - b[d]) for d in a]
    max_d = max(diffs)
    mean_d = sum(diffs) / len(diffs)
    overall_max = max(overall_max, max_d)
    print(f"{t:>6d} {max_d:>14.3e} {mean_d:>14.3e} {sum(a.values()):>10.6f} {sum(b.values()):>10.6f} {str(a==b):>10s}")

print(f"\nOverall max abs diff across all sampled topics: {overall_max:.3e}")
