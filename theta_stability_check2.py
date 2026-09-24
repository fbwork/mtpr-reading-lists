"""
Corrected re-run of theta_stability_check.py: the first pass missed clearing
transition_star's own module-level @lru_cache, so pass B's transition step
could have silently reused pass A's cached T* values (biasing the two runs
to look MORE similar than a truly independent recomputation would be).
This version clears every relevant cache (get_doc_topics, topic_doc_score,
topic_docs_score, transition_star) and adds the diagnostics requested:
iterations-to-convergence, final L1 residual, and top-10 overlap between the
two runs' rankings for each topic (using raw score order as a proxy, since
this is a single-topic PageRank vector, not a full query-aware ranking).
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

ALPHA = 0.15
MAX_ITERATIONS = 100
EPSILON = 1.0e-6

print("Loading citation graph...", flush=True)
with open(pr_output_path.joinpath('graph.bin'), 'rb') as f:
    graph: nx.DiGraph = pickle.load(f)
W = nx.stochastic_graph(graph, weight='weight')
N = W.number_of_nodes()
print(f"  graph identity check: {W.number_of_nodes()} nodes, {W.number_of_edges()} edges "
      f"(held fixed across both passes -- built once, before either pass runs)", flush=True)

print("Loading LDA (T=200)...", flush=True)
lda_results = lda_setup(200, 0, fresh=False)
lda = list(lda_results.values())[0]

SAMPLE_TOPICS = list(range(0, 200, 10))  # same 20 topics as before, for direct comparability


def clear_all_caches():
    for attr in ('topic_doc_score', 'topic_docs_score', 'get_doc_topics'):
        fn = getattr(lda, attr, None)
        if fn is not None and hasattr(fn, 'cache_clear'):
            fn.cache_clear()
    transition_star.cache_clear()


def run_topic(t):
    x = {d: PaperRankClassic.bias(PRModes.THEMED, t, d, W, lda) for d in W}
    total = sum(x.values())
    x = {k: v / total for k, v in x.items()} if total > 0 else {d: 1.0 / N for d in W}
    biases = {**x}

    n_iterations = 0
    final_diff = None
    for iteration in range(MAX_ITERATIONS):
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
        n_iterations = iteration + 1
        final_diff = diff
        # exact production convergence criterion: diff < N * epsilon
        if diff < N * EPSILON:
            break
    return x, n_iterations, final_diff


print(f"\nRunning THEMED PageRank for {len(SAMPLE_TOPICS)} sample topics, TWICE each "
      f"(ALL relevant caches -- including transition_star -- cleared between the two passes)...\n",
      flush=True)

results_a, meta_a = {}, {}
results_b, meta_b = {}, {}
t_start = time.time()
for t in SAMPLE_TOPICS:
    clear_all_caches()
    x, n_iter, diff = run_topic(t)
    results_a[t] = x
    meta_a[t] = (n_iter, diff)
for t in SAMPLE_TOPICS:
    clear_all_caches()
    x, n_iter, diff = run_topic(t)
    results_b[t] = x
    meta_b[t] = (n_iter, diff)
print(f"Both runs done in {(time.time() - t_start) / 60:.1f} min\n", flush=True)

print(f"{'topic':>6s} {'max_diff':>12s} {'mean_diff':>12s} {'iters_A':>8s} {'iters_B':>8s} "
      f"{'resid_A':>10s} {'resid_B':>10s} {'top10_overlap':>14s}")
overall_max = 0.0
for t in SAMPLE_TOPICS:
    a, b = results_a[t], results_b[t]
    diffs = [abs(a[d] - b[d]) for d in a]
    max_d = max(diffs)
    mean_d = sum(diffs) / len(diffs)
    overall_max = max(overall_max, max_d)
    top10_a = set(sorted(a, key=a.get, reverse=True)[:10])
    top10_b = set(sorted(b, key=b.get, reverse=True)[:10])
    overlap = len(top10_a & top10_b)
    ia, ra = meta_a[t]
    ib, rb = meta_b[t]
    print(f"{t:>6d} {max_d:>12.3e} {mean_d:>12.3e} {ia:>8d} {ib:>8d} {ra:>10.3e} {rb:>10.3e} {overlap:>10d}/10")

print(f"\nOverall max abs diff across all sampled topics: {overall_max:.3e}")
