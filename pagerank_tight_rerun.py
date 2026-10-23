"""
Recompute the per-topic PageRank vectors with a much tighter stopping rule.

Background. The production loop (components/pagerank.py::page_rank) stops when
sum|x_k - x_{k-1}| < N * 1e-6 (about 0.0157 for N = 15,666) or after 100 iterations,
and renormalises x to sum to 1 after every update. pagerank_stability_check.py found
that for the per-target-normalised modes this rule fires after only 6-10 iterations
while the iteration needs ~100-280 to converge: the per-iteration change is small
because convergence is slow, not because the iterate is close to the fixed point. The
L1 distance of the production iterate from the converged vector is ~0.5 for
d_modified_transition.

This script reruns the IDENTICAL update for every topic and every mode, only faster:
the transition weights trans(t, d, d') are a fixed function of the mode, topic and edge,
so they are assembled once per topic (vectorised with numpy from the same quantities the
production code uses: P(t|d), the out-neighbour sums, the source-node multiplier and the
per-target normalisation) and the iteration is then a sparse matrix-vector product.
The vectorised weights are checked against components.pagerank.PaperRankClassic.transition
on the first VALIDATE_TOPICS topics of every mode (max |difference| in the meta file). It writes

  components/output/pr/<T>/<T>_tight/pageranks_<mode>.bin    converged vectors, same
        {doc_id: {topic: score}} layout as the production files
  components/output/pr/<T>/<T>_tight/pageranks_<mode>.meta.json  iteration statistics

Validation (done for every mode, first VALIDATE_TOPICS topics): the iterate at which
the PRODUCTION rule would have stopped must equal the cached production vector
(max |difference| reported in the meta file).

Usage: python pagerank_tight_rerun.py --modes themed,normal [--eps 1e-10]
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault('LDA_NO_TOPICS', '200')

import networkx as nx
import numpy as np
import scipy.sparse as sp

from components.database import setup as db_setup
from components.lda import setup as lda_setup
import components.pagerank as pr_module
from components.pagerank import PaperRankClassic
from components.prmodes import PRModes

ap = argparse.ArgumentParser()
ap.add_argument('--modes', required=True)
ap.add_argument('--eps', type=float, default=1e-10)
ap.add_argument('--max-iter', type=int, default=3000)
ap.add_argument('--validate-topics', type=int, default=5)
args = ap.parse_args()

NO_TOPICS = int(os.getenv('LDA_NO_TOPICS', '200'))
ALPHA = 0.15
PROD_EPS, PROD_MAXIT = 1e-6, 100
PR_DIR = Path('components/output/pr') / str(NO_TOPICS)
OUT = PR_DIR / f'{NO_TOPICS}_tight'
OUT.mkdir(parents=True, exist_ok=True)

db_setup()
lda_results = lda_setup(NO_TOPICS, 0, fresh=False)
lda = lda_results[0]


def clear_caches():
    """Same per-topic cache clearing as page_rank()."""
    for name in ('topic_docs_score', 'topic_doc_score', 'topic_authors_score',
                 'author_topic_scores', 'doc_authors_score'):
        fn = getattr(lda, name, None)
        if hasattr(fn, 'cache_clear'):
            fn.cache_clear()
    for fn in (getattr(pr_module, 'transition_star', None),
               getattr(pr_module, 'normalized_transition_weights', None)):
        if hasattr(fn, 'cache_clear'):
            fn.cache_clear()


PER_TARGET = {PRModes.THEMED, PRModes.MODIFIED, PRModes.MODIFIED_TRANSITION,
              PRModes.D_MODIFIED, PRModes.CL_MODIFIED, PRModes.BW_MODIFIED, PRModes.KZ_MODIFIED,
              PRModes.D_MODIFIED_TRANSITION, PRModes.CL_MODIFIED_TRANSITION,
              PRModes.BW_MODIFIED_TRANSITION, PRModes.KZ_MODIFIED_TRANSITION}
_edge_cache = {}
validation = {'trans': 0.0, 'bias': 0.0}


CENTRALITY = {  # name -> (batch method, total method)
    'degree': ('topic_degree_centrality_scores_batch', 'topic_degree_centrality_score'),
    'closeness': ('topic_closeness_centrality_scores_batch', 'topic_closeness_centrality_score'),
    'betweenness': ('topic_betweenness_centrality_scores_batch', 'topic_betweenness_centrality_score'),
    'katz': ('topic_katz_centrality_scores_batch', 'topic_katz_centrality_score'),
}
MODE_CENTRALITY = {}
for _name, _prefix in (('degree', 'D'), ('closeness', 'CL'), ('betweenness', 'BW'), ('katz', 'KZ')):
    for _suffix in ('MODIFIED', 'MODIFIED_BIAS', 'MODIFIED_TRANSITION'):
        MODE_CENTRALITY[PRModes[f'{_prefix}_{_suffix}']] = _name


def centrality_vector(name, t, W, nodes):
    """c[d] / total for one centrality measure; the batch dict is read once (the per-document
    accessors in components/lda.py re-read the cache pickle on every call)."""
    batch_fn, total_fn = CENTRALITY[name]
    batch = getattr(lda, batch_fn)(W, t)
    total = getattr(lda, total_fn)(W, t) or 1
    return np.array([batch.get(d, 0.0) for d in nodes], dtype=float) / total


def fast_bias(mode, t, W, nodes):
    """Same quantity as PaperRankClassic.bias for the centrality-based modes; others fall back."""
    if mode in MODE_CENTRALITY and not mode.value.endswith('_transition'):
        p = np.array([lda.topic_doc_score(d, t) for d in nodes], dtype=float)
        return centrality_vector(MODE_CENTRALITY[mode], t, W, nodes) * p
    if mode == PRModes.HYBRID:
        p = np.array([lda.topic_doc_score(d, t) for d in nodes], dtype=float)
        c = sum(0.25 * centrality_vector(n, t, W, nodes) for n in ('degree', 'closeness', 'betweenness', 'katz'))
        return p * c
    return np.array([PaperRankClassic.bias(mode, t, d, W, lda) for d in nodes], dtype=float)


def edge_arrays(W, nodes, index):
    """Edge d' -> d as (src=d', dst=d), cached per graph."""
    key = id(W)
    if key not in _edge_cache:
        src = np.fromiter((index[u] for u, _ in W.edges()), dtype=np.int64)
        dst = np.fromiter((index[v] for _, v in W.edges()), dtype=np.int64)
        outdeg = np.array([W.out_degree(d) for d in nodes], dtype=float)
        _edge_cache[key] = (src, dst, outdeg)
    return _edge_cache[key]


def source_multiplier(mode, t, W, nodes):
    """Factor multiplying T* at the SOURCE node d' (transition_star's mode branches)."""
    if mode == PRModes.MODIFIED_TRANSITION:
        tot = lda.topic_authors_score(t) or 1
        return np.array([lda.doc_authors_score(t, d) / tot for d in nodes])
    if mode in MODE_CENTRALITY and mode.value.endswith('_transition'):
        return centrality_vector(MODE_CENTRALITY[mode], t, W, nodes)
    return None


def build_transition_matrix(mode, t, W, nodes, index):
    """Sparse M with M[d, d'] = trans(t, d, d')."""
    N = len(nodes)
    src, dst, outdeg = edge_arrays(W, nodes, index)
    if mode not in PER_TARGET:                      # uniform, column-stochastic: 1/out_degree(d')
        return sp.csr_matrix((1.0 / outdeg[src], (dst, src)), shape=(N, N))

    p = np.array([lda.topic_doc_score(d, t) for d in nodes], dtype=float)
    total_topic = lda.topic_docs_score(t)
    A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(N, N))      # A[d', d] = 1 if d' cites d
    out_sum = np.asarray(A @ p).ravel()                                    # sum_{d in link_out(d')} P(t|d)
    ok = (total_topic > 0) & (out_sum[src] > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = (p[src] / total_topic) * (p[dst] / out_sum[src])
    raw = np.where(ok, np.sqrt(np.maximum(0.0, term)), 0.0)
    mult = source_multiplier(mode, t, W, nodes)
    if mult is not None:
        raw = raw * mult[src]
    tot = np.bincount(dst, weights=raw, minlength=N)
    indeg = np.bincount(dst, minlength=N).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(tot[dst] > 0, raw / tot[dst], 1.0 / indeg[dst])       # uniform fallback if all-zero
    return sp.csr_matrix((w, (dst, src)), shape=(N, N))


def solve_topic(mode, t, W, nodes, index, dangling_idx):
    """Return (production-stop iterate, its, converged), (tight iterate, its, converged, final diff)."""
    N = len(nodes)
    b = fast_bias(mode, t, W, nodes)
    if t < args.validate_topics:
        b_ref = np.array([PaperRankClassic.bias(mode, t, d, W, lda) for d in nodes], dtype=float)
        validation['bias'] = max(validation['bias'], float(np.abs(b - b_ref).max()))
    tot = b.sum()
    b = b / tot if tot > 0 else np.full(N, 1.0 / N)

    M = build_transition_matrix(mode, t, W, nodes, index)
    if t < args.validate_topics:
        rows, cols, vals = [], [], []
        for d in nodes:
            i = index[d]
            for dp, _ in W.in_edges(d):
                rows.append(i)
                cols.append(index[dp])
                vals.append(PaperRankClassic.transition(mode, t, d, dp, W, lda))
        M_ref = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        validation['trans'] = max(validation['trans'], abs(M - M_ref).max())

    x = b.copy()
    prod = None
    for it in range(1, args.max_iter + 1):
        x_last = x
        dmass = x_last[dangling_idx].sum() if len(dangling_idx) else 0.0
        x = ALPHA * b + (1 - ALPHA) * dmass * b + (1 - ALPHA) * (M @ x_last)
        s = x.sum()
        if s > 0:
            x = x / s
        diff = float(np.abs(x - x_last).sum())
        if prod is None and (diff < N * PROD_EPS or it == PROD_MAXIT):
            prod = (x.copy(), it, diff < N * PROD_EPS)
        if diff < N * args.eps:
            return prod, (x, it, True, diff)
    return prod, (x, args.max_iter, False, diff)


for mode_name in args.modes.split(','):
    final = OUT / f'pageranks_{mode_name}.bin'
    if final.exists():
        print(f'[skip] {mode_name} already complete', flush=True)
        continue
    t0 = time.time()
    mode = PRModes(mode_name)
    inst = PaperRankClassic(lda_results, mode, NO_TOPICS)
    W = nx.stochastic_graph(inst.graph, weight='weight')
    nodes = list(W.nodes())
    index = {d: i for i, d in enumerate(nodes)}
    dangling_idx = np.array([index[d] for d in nodes if W.out_degree(d) == 0], dtype=int)
    cached = pickle.load((PR_DIR / PR_DIR.name / f'pageranks_{mode_name}.bin').open('rb'))

    validation['trans'] = 0.0
    validation['bias'] = 0.0
    tight_vecs, stats, max_val_diff = {}, [], 0.0
    for t in range(NO_TOPICS):
        prod, tight = solve_topic(mode, t, W, nodes, index, dangling_idx)
        if t < args.validate_topics:
            ref = np.array([cached[d].get(t, 0.0) for d in nodes])
            max_val_diff = max(max_val_diff, float(np.abs(prod[0] - ref).max()))
        tight_vecs[t] = tight[0]
        stats.append({'topic': t, 'prod_iterations': prod[1], 'prod_converged': prod[2],
                      'tight_iterations': tight[1], 'tight_converged': tight[2],
                      'tight_final_l1_change': tight[3],
                      'l1_prod_vs_tight': float(np.abs(prod[0] - tight[0]).sum())})
        clear_caches()
        if (t + 1) % 20 == 0:
            print(f'{mode_name}: {t+1}/{NO_TOPICS}  {time.time()-t0:.0f}s  '
                  f'validation max|diff|: snapshot-vs-cache {max_val_diff:.2e}, '
                  f'trans {validation["trans"]:.2e}, bias {validation["bias"]:.2e}', flush=True)

    results = {d: {t: float(tight_vecs[t][index[d]]) for t in range(NO_TOPICS)} for d in nodes}
    pickle.dump(results, final.open('wb'))
    its = sorted(s['tight_iterations'] for s in stats)
    meta = {
        'mode': mode_name, 'eps': args.eps, 'stopping_rule': 'L1 change < N * eps',
        'validation_max_abs_diff_prod_snapshot_vs_cached': max_val_diff,
        'validation_max_abs_diff_vectorised_vs_production_transition': float(validation['trans']),
        'validation_max_abs_diff_fast_vs_production_bias': float(validation['bias']),
        'all_tight_converged': all(s['tight_converged'] for s in stats),
        'prod_rule_converged_all': all(s['prod_converged'] for s in stats),
        'prod_iterations_median': sorted(s['prod_iterations'] for s in stats)[NO_TOPICS // 2],
        'tight_iterations_min_median_max': [its[0], its[NO_TOPICS // 2], its[-1]],
        'median_l1_prod_vs_tight': sorted(s['l1_prod_vs_tight'] for s in stats)[NO_TOPICS // 2],
        'max_l1_prod_vs_tight': max(s['l1_prod_vs_tight'] for s in stats),
        'seconds': round(time.time() - t0),
        'per_topic': stats,
    }
    (OUT / f'pageranks_{mode_name}.meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'[done] {mode_name}: {meta}', flush=True)
print('all done', flush=True)
