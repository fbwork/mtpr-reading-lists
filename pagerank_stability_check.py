"""
Stability of the renormalised power iteration with respect to the stopping rule.

The production loop (components/pagerank.py::page_rank) stops when
sum|x_k - x_{k-1}| < N * epsilon (epsilon = 1e-6, i.e. an L1 change of about
0.0157 for N = 15,666 nodes, the same rule as networkx.pagerank) or after
max_iterations = 100, whichever comes first, and renormalises x to sum to 1
after every update. This script asks whether the RANKING depends on that rule.

For a sample of topics (the highest-weight topic of each AAN gold query) and a
set of modes (including the per-target-normalised ones for which the
unnormalised iteration is known to diverge), it runs the identical update
(same bias(), transition(), dangling handling and renormalisation as
page_rank) with a much tighter rule and compares

  - the production stopping point (first iterate with L1 change < N*1e-6, capped at
    100 iterations; verified against the cached pageranks_*.bin) with
  - the final iterate under epsilon = 1e-12 (or max_iter, whichever first).

Reported per (mode, topic): iterations used by the production rule, whether it
converged or hit the cap, L1 distance and Spearman correlation between the two
score vectors over all nodes, top-20 / top-100 overlap, and the largest rank change
among the production top-20.

Writes components/resources/gs/pagerank_stability.json
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

# The project's .env sets LDA_NO_TOPICS for ad-hoc runs; the paper's main setting is T = 200.
# Pin it BEFORE importing components (dotenv does not override an existing variable).
os.environ.setdefault('LDA_NO_TOPICS', '200')

import networkx as nx

from components.database import setup as db_setup
from components.lda import setup as lda_setup
from components.pagerank import PaperRankClassic
from components.prmodes import PRModes
from components.query import query_topic_weights

ap = argparse.ArgumentParser()
ap.add_argument('--modes', default='themed,bw_modified,bw_modified_bias,normal')
ap.add_argument('--n-topics-per-query', type=int, default=1)
ap.add_argument('--max-topics', type=int, default=0, help='limit number of topics (0 = all selected)')
ap.add_argument('--tight-eps', type=float, default=1e-12)
ap.add_argument('--tight-max-iter', type=int, default=1000)
ap.add_argument('--out', default='components/resources/gs/pagerank_stability.json')
args = ap.parse_args()

NO_TOPICS = int(os.getenv('LDA_NO_TOPICS', '200'))
ALPHA = 0.15
PROD_EPS, PROD_MAXIT = 1e-6, 100
PR_CACHE = Path('components/output/pr') / str(NO_TOPICS) / str(NO_TOPICS)

db_setup()
lda_results = lda_setup(NO_TOPICS, 0, fresh=False)
lda = lda_results[0]

gold = json.load(open('components/resources/gs/golden_standard.json'))
topics = []
for q in gold:
    w = query_topic_weights(q, lda_results)
    for t, _ in sorted(w.items(), key=lambda kv: -kv[1])[:args.n_topics_per_query]:
        if t not in topics:
            topics.append(t)
if args.max_topics:
    topics = topics[:args.max_topics]
print('topics:', topics, flush=True)


def run(mode, t, W, tight_eps, tight_max):
    """Identical update to components.pagerank.page_rank; returns production + tight iterates."""
    N = W.number_of_nodes()
    x = {d: PaperRankClassic.bias(mode, t, d, W, lda) for d in W}
    tot = sum(x.values())
    x = {k: v / tot for k, v in x.items()} if tot > 0 else {d: 1.0 / N for d in W}
    biases = {**x}
    dangling = [d for d in W if W.out_degree(d) == 0]
    prod = None
    prod_iters = None
    prod_converged = False
    diffs = []
    for it in range(1, tight_max + 1):
        x_last = x
        x = dict.fromkeys(x_last.keys(), 0)
        dmass = sum(x_last[d] for d in dangling)
        for d in x:
            walk = sum(PaperRankClassic.transition(mode, t, d, dp, W, lda) * x_last[dp]
                       for dp, _ in W.in_edges(d))
            x[d] = ALPHA * biases[d] + (1 - ALPHA) * dmass * biases[d] + (1 - ALPHA) * walk
        s = sum(x.values())
        if s > 0:
            x = {k: v / s for k, v in x.items()}
        diff = sum(abs(x[n] - x_last[n]) for n in x)
        diffs.append(diff)
        if prod is None and (diff < N * PROD_EPS or it == PROD_MAXIT):
            prod, prod_iters, prod_converged = dict(x), it, diff < N * PROD_EPS
        if diff < N * tight_eps:
            break
    return prod, prod_iters, prod_converged, x, it, diffs


def ranks(v):
    order = sorted(v, key=v.get, reverse=True)
    return {d: i for i, d in enumerate(order)}, order


def spearman(a, b):
    ra, _ = ranks(a)
    rb, _ = ranks(b)
    n = len(ra)
    return 1 - 6 * sum((ra[d] - rb[d]) ** 2 for d in ra) / (n * (n * n - 1))


results = []
for mode_name in args.modes.split(','):
    mode = PRModes(mode_name)
    inst = PaperRankClassic(lda_results, mode, NO_TOPICS)
    W = nx.stochastic_graph(inst.graph, weight='weight')
    cache = pickle.load((PR_CACHE / f'pageranks_{mode_name}.bin').open('rb'))
    for t in topics:
        t0 = time.time()
        prod, p_it, p_conv, tight, t_it, diffs = run(mode, t, W, args.tight_eps, args.tight_max_iter)
        cached = {d: cache[d].get(t, 0.0) for d in cache}
        match = max(abs(prod[d] - cached[d]) for d in prod)
        ra, order_p = ranks(prod)
        rb, order_t = ranks(tight)
        row = {
            'mode': mode_name, 'topic': t,
            'prod_iterations': p_it, 'prod_converged': p_conv, 'prod_final_l1_change': diffs[p_it - 1],
            'tight_iterations': t_it, 'tight_final_l1_change': diffs[-1],
            'matches_cached_max_abs_diff': match,
            'l1_prod_vs_tight': sum(abs(prod[d] - tight[d]) for d in prod),
            'spearman_prod_vs_tight': spearman(prod, tight),
            'top20_overlap': len(set(order_p[:20]) & set(order_t[:20])),
            'top100_overlap': len(set(order_p[:100]) & set(order_t[:100])),
            'max_rank_shift_in_prod_top20': max(abs(ra[d] - rb[d]) for d in order_p[:20]),
            'seconds': round(time.time() - t0, 1),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding='utf-8')
print('done', flush=True)
