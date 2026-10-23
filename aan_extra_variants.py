"""
AAN sensitivity analyses that need new PageRank runs (all reuse the production update:
bias, dangling mass redistributed through the bias vector, per-iteration renormalization,
init = bias, stopping rule L1 change < N * 1e-10 or 3000 iterations; vectorised with sparse
matrices exactly as in pagerank_tight_rerun.py):

 (1) In-degree instead of out-degree for the degree centrality:   id_modified_bias / id_modified_transition
 (2) Topic-membership threshold tau for the degree-based centralities (tau in 0.005, 0.01, 0.02, 0.05)
 (3) Hybrid weights (degree, closeness, betweenness, Katz): equal, four 'one-heavy' settings, and no-betweenness

Validation (must hold before the new numbers are used): the rebuilt out-degree centrality equals the cached
degree batch for sampled topics, and the rebuilt d_modified_bias / hybrid(equal) reproduce Table 2 (0.5972 / 0.6291).

Writes components/resources/gs/aan_extra_variants.json and .csv
"""
import json
import math
import os
import pickle
import sys

os.environ.setdefault('LDA_NO_TOPICS', '200')
src = open('csc_tight_eval.py', encoding='utf-8').read()
exec(compile(src[:src.index("out = {}")], 'csc_head', 'exec'))   # db, lda_results, golden, years, csc(), evaluate()

import numpy as np
import scipy.sparse as sp
import networkx as nx
from components.pagerank import PaperRankClassic
from components.theta_artifact import load_theta_artifact

ALPHA = 0.15
TOL_EPS, MAX_IT = 1e-10, 3000
lda = lda_results[0]
theta, _ = load_theta_artifact(T, cluster=0)
N = theta.shape[0]
inst = PaperRankClassic(lda_results, PRModes.NORMAL, T)
G = inst.graph
nodes = list(range(N))
assert set(G.nodes()) == set(nodes), 'node ids are not 0..N-1'
E = list(G.edges())
src_e = np.array([u for u, _ in E], dtype=np.int64)       # citing paper d'
tgt_e = np.array([v for _, v in E], dtype=np.int64)       # cited paper d
outdeg = np.bincount(src_e, minlength=N).astype(float)
indeg = np.bincount(tgt_e, minlength=N).astype(float)
dangling = np.where(outdeg == 0)[0]
print(f'graph: {N} nodes, {len(E)} edges, {len(dangling)} dangling', flush=True)


def degree_vectors(t, tau):
    """Out- and in-degree restricted to the topic subgraph {d : theta[d,t] >= tau} (nodes outside get 0)."""
    mask = theta[:, t] >= tau
    both = mask[src_e] & mask[tgt_e]
    out = np.bincount(src_e[both], minlength=N).astype(float)
    inn = np.bincount(tgt_e[both], minlength=N).astype(float)
    out[~mask] = 0.0
    inn[~mask] = 0.0
    return out, inn


def tstar(t):
    th = theta[:, t]
    total = th.sum() or 1.0
    out_sum = np.bincount(src_e, weights=th[tgt_e], minlength=N)
    with np.errstate(divide='ignore', invalid='ignore'):
        row = np.sqrt(np.where(out_sum > 0, th / (total * out_sum), 0.0))
    return row[src_e] * np.sqrt(np.maximum(th, 0.0))[tgt_e]


def uniform_matrix():
    return sp.csr_matrix((1.0 / outdeg[src_e], (tgt_e, src_e)), shape=(N, N))


UNI = uniform_matrix()


def per_target_matrix(w):
    rs = np.bincount(tgt_e, weights=w, minlength=N)
    with np.errstate(divide='ignore', invalid='ignore'):
        wn = np.where(rs[tgt_e] > 0, w / rs[tgt_e], 1.0 / indeg[tgt_e])
    return sp.csr_matrix((wn, (tgt_e, src_e)), shape=(N, N))


def solve(M, bias):
    x = bias.copy()
    for it in range(1, MAX_IT + 1):
        xn = (1 - ALPHA) * (M @ x) + (1 - ALPHA) * x[dangling].sum() * bias + ALPHA * bias
        s = xn.sum()
        if s > 0:
            xn = xn / s
        diff = float(np.abs(xn - x).sum())
        x = xn
        if diff < N * TOL_EPS:
            return x, it, True
    return x, MAX_IT, False


def norm_bias(raw):
    s = raw.sum()
    if s > 0:
        return raw / s
    return np.full(N, 1.0 / N)


def run_variant(name, bias_fn, trans_fn=None):
    """bias_fn(t) -> raw bias vector; trans_fn(t) -> matrix (None = uniform)."""
    nonconv = 0
    iters = []
    res = {d: {} for d in nodes}
    X = np.zeros((N, T))
    for t in range(T):
        b = norm_bias(bias_fn(t))
        M = UNI if trans_fn is None else trans_fn(t)
        x, it, ok = solve(M, b)
        X[:, t] = x
        nonconv += (not ok)
        iters.append(it)
    yd = np.array([max(1, 2011 - years[d]) for d in nodes], dtype=float)
    agified = {d: {t: X[d, t] / yd[d] for t in range(T)} for d in nodes}
    scores = evaluate(lambda q: rank_pagerank_for_query(q, agified, lda_results))
    print(f'{name:<34} OCSC {scores[2]:.4f}  (F {scores[0]:.3f} R {scores[1]:.3f})  nonconv {nonconv}/200  median it {int(np.median(iters))}', flush=True)
    return {'fcsc': scores[0], 'rcsc': scores[1], 'ocsc': scores[2], 'nonconverged': nonconv, 'median_iterations': int(np.median(iters))}


results = {}

# ---------- validation: degree centrality against the cached batch, then Table 2
for t in (0, 7, 29, 140):
    cached = lda.topic_degree_centrality_scores_batch(G, t)
    out, _ = degree_vectors(t, 0.01)
    mine = {d: int(out[d]) for d in np.where(theta[:, t] >= 0.01)[0]}
    cd = {d: v for d, v in cached.items()}
    same = all(mine.get(d, 0) == cd.get(d, 0) for d in set(mine) | set(cd))
    print(f'validate degree topic {t}: rebuilt == cached: {same}', flush=True)
    assert same, 'out-degree centrality does not match the cache'

tot_cache = {}
def cent_norm(kind, t):
    batch_fn = {'closeness': lda.topic_closeness_centrality_scores_batch,
                'betweenness': lda.topic_betweenness_centrality_scores_batch,
                'katz': lda.topic_katz_centrality_scores_batch}[kind]
    d = batch_fn(G, t)
    v = np.zeros(N)
    for k, x in d.items():
        v[k] = x
    s = v.sum()
    return v / s if s > 0 else v


results['d_modified_bias (tau=0.01, out)'] = run_variant(
    'd_modified_bias (tau=0.01, out)', lambda t: theta[:, t] * degree_vectors(t, 0.01)[0])


def hybrid(wd, wc, wb, wk):
    def f(t):
        out, _ = degree_vectors(t, 0.01)
        cs = wd * (out / out.sum() if out.sum() > 0 else out)
        if wc: cs = cs + wc * cent_norm('closeness', t)
        if wb: cs = cs + wb * cent_norm('betweenness', t)
        if wk: cs = cs + wk * cent_norm('katz', t)
        return theta[:, t] * cs
    return f


HYB = {'equal (.25,.25,.25,.25)': (.25, .25, .25, .25), 'degree-heavy (.7,.1,.1,.1)': (.7, .1, .1, .1),
       'closeness-heavy (.1,.7,.1,.1)': (.1, .7, .1, .1), 'betweenness-heavy (.1,.1,.7,.1)': (.1, .1, .7, .1),
       'katz-heavy (.1,.1,.1,.7)': (.1, .1, .1, .7), 'no betweenness (1/3,1/3,0,1/3)': (1 / 3, 1 / 3, 0, 1 / 3)}
for name, w in HYB.items():
    results['hybrid ' + name] = run_variant('hybrid ' + name, hybrid(*w))

# ---------- (1) in-degree
results['id_modified_bias (tau=0.01, in)'] = run_variant(
    'id_modified_bias (tau=0.01, in)', lambda t: theta[:, t] * degree_vectors(t, 0.01)[1])
results['d_modified_transition (out)'] = run_variant(
    'd_modified_transition (out)', lambda t: theta[:, t],
    lambda t: per_target_matrix(tstar(t) * degree_vectors(t, 0.01)[0][src_e]))
results['id_modified_transition (in)'] = run_variant(
    'id_modified_transition (in)', lambda t: theta[:, t],
    lambda t: per_target_matrix(tstar(t) * degree_vectors(t, 0.01)[1][src_e]))

# ---------- (2) tau for the degree-based centralities
for tau in (0.005, 0.02, 0.05):
    for kind, idx in (('out', 0), ('in', 1)):
        n = f'd{"" if kind == "out" else "(in)"}_modified_bias tau={tau}'
        results[n] = run_variant(n, lambda t, tau=tau, idx=idx: theta[:, t] * degree_vectors(t, tau)[idx])

bm = BM25Baseline() if 'BM25Baseline' in globals() else None
from components.baselines import BM25Baseline
bm = BM25Baseline()
bm.build_index(fresh=False)
bm25 = evaluate(lambda q: [pid - 1 for pid, _ in bm.search(q, top_k=K)[:K]])
results['BM25'] = {'fcsc': bm25[0], 'rcsc': bm25[1], 'ocsc': bm25[2]}
print('BM25', round(bm25[2], 4))

json.dump(results, open('components/resources/gs/aan_extra_variants.json', 'w'), indent=2)
with open('components/resources/gs/aan_extra_variants.csv', 'w') as f:
    f.write('variant,fcsc,rcsc,ocsc,nonconverged,median_iterations\n')
    for k, v in results.items():
        f.write(f'"{k}",{v["fcsc"]:.6f},{v["rcsc"]:.6f},{v["ocsc"]:.6f},{v.get("nonconverged","")},{v.get("median_iterations","")}\n')
print('saved')
