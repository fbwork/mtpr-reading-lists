"""
ACL-rlg: bias-vs-transition placement of centrality signals (see acl_rlg_bias_vs_transition_PREREG.md).

For ONE tutorial year this script computes, for a set of modes and all 200 topics, the
query-level score  S[q, d] = sum_t theta_q[q, t] * PR(d, t)  for every query (annotation set x
tutorial) of that year, WITHOUT keeping the full PR(d, t) matrix: each solved topic vector is
folded into S immediately. Per-(mode, topic) convergence statistics are stored.

Modes
  personalized, themed          legacy ACL-rlg solvers (validation: must reproduce the stored means)
  themed_fb                     themed with the production semantics (uniform fallback, init = bias)
  bw_modified_bias              validation of the uniform-transition solver
  d_/cl_/kz_modified_bias       centrality x theta bias, uniform transition
  d_/cl_/bw_/kz_modified_transition  bias = theta, transition = T* weighted by the SOURCE node's
                                centrality, normalised per target, uniform fallback

Output: acl_rlg_pilot_out/bt/<year>/<mode>.npz   (S, query ids, convergence arrays)
"""
import argparse
import ast
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, r"E:\0000000-Thesis\1405\thesis-1405\thesis-1405")

RLG_BASE = Path(r"E:\0000000-Thesis\1403-new approach\ACL-rlg\aclanthology-tutorial-reading-lists")
EVAL_DATA = Path(r"E:\0000000-Thesis\1405\thesis-1405\acl-rlg-eval\data")
ACL_DATASET_PATH = RLG_BASE / "experiments/classic_methods/acl_anthology_dataset/acl_anthology_dataset.json"
ACL_GRAPH_PATH = EVAL_DATA / "acl_full_citation_graph.json"
READING_LISTS_CSV = RLG_BASE / "reading_lists.csv"
ANNOT_DIR = RLG_BASE / "annotations"
OUT_ROOT = Path(r"E:\0000000-Thesis\1405\thesis-1405\thesis-1405\acl_rlg_pilot_out")

CLUSTER, T, ALPHA, NEIGHBOR_K = 99, 200, 0.15, 20
FULL_YEAR = 2024

ap = argparse.ArgumentParser()
ap.add_argument('--year', type=int, required=True)
ap.add_argument('--modes', required=True)
ap.add_argument('--tol', type=float, default=1e-8)
ap.add_argument('--max-iter', type=int, default=3000)
ap.add_argument('--max-topics', type=int, default=T)
ap.add_argument('--tag', default='bt')
args = ap.parse_args()

CENT = {'d': 'degree', 'cl': 'closeness', 'bw': 'betweenness', 'kz': 'katz'}


def clean_string(s):
    return re.sub(r'\W+', '', s or '').lower()


# ---------------------------------------------------------------- data
with open(ACL_DATASET_PATH, encoding='utf-8') as f:
    acl_dataset = json.load(f)
acl_ids = sorted(acl_dataset.keys())
N = len(acl_ids)
aid2idx = {a: i for i, a in enumerate(acl_ids)}
years = np.array([acl_dataset[a]['year'] for a in acl_ids])

with open(ACL_GRAPH_PATH, encoding='utf-8') as f:
    raw_graph = json.load(f)

from components.theta_artifact import load_theta_artifact
theta, _meta = load_theta_artifact(T, cluster=CLUSTER)

YEAR = args.year
eligible_mask = years <= YEAR
src_l, tgt_l = [], []
for s, tg in raw_graph.items():
    si = aid2idx.get(s)
    if si is None or not eligible_mask[si]:
        continue
    for t_ in tg:
        ti = aid2idx.get(t_)
        if ti is not None and eligible_mask[ti]:
            src_l.append(si)
            tgt_l.append(ti)
edge_src = np.array(src_l, dtype=np.int64)
edge_tgt = np.array(tgt_l, dtype=np.int64)
out_degree = np.bincount(edge_src, minlength=N)
in_degree = np.bincount(edge_tgt, minlength=N).astype(float)
dangling = out_degree == 0
cent_dir = OUT_ROOT / ('centrality' if YEAR == FULL_YEAR else f'centrality_y{YEAR}')
print(f'year {YEAR}: {eligible_mask.sum():,} papers, {len(edge_src):,} edges, centrality dir {cent_dir.name}', flush=True)

# queries of this year (same construction as acl_rlg_final_eval_temporal.py)
corpus_texts = [f"{acl_dataset[a].get('title', '')} {acl_dataset[a].get('abstract', '') or ''}" for a in acl_ids]
vectorizer = TfidfVectorizer(stop_words="english", max_features=30_000, sublinear_tf=True)
tfidf = vectorizer.fit_transform(corpus_texts)
df_gold = pd.read_csv(READING_LISTS_CSV)
df_gold['reading_list'] = df_gold['reading_list'].apply(ast.literal_eval)
trues = {row['id']: [clean_string(p['title']) for p in row['reading_list']
                     if str(p.get('in_acl', 'False')) == 'True' and p.get('title')]
         for _, row in df_gold.iterrows()}
queries = []          # (annotation set, tutorial id, query text)
for ai in (1, 2, 3):
    da = pd.read_csv(ANNOT_DIR / f'annotation_{ai}.csv')
    for _, row in da.iterrows():
        if int(row['year']) == YEAR and trues.get(row['id']):
            queries.append((ai, row['id'], str(row['query_keywords'])))
nq = len(queries)
theta_q = np.zeros((nq, T))
for qi, (_, _, qtext) in enumerate(queries):
    sims = cosine_similarity(vectorizer.transform([qtext]), tfidf)[0]
    elig = np.where(years <= YEAR)[0]
    top = elig[np.argsort(-sims[elig])[:NEIGHBOR_K]]
    tq = theta[top].mean(axis=0)
    s = tq.sum()
    theta_q[qi] = tq / s if s > 1e-12 else np.full(T, 1.0 / T)
print(f'{nq} queries in {YEAR}', flush=True)

normal_bias = np.full(N, 1.0 / N)


def load_cent(kind, t):
    with open(cent_dir / f'{CENT[kind]}_{t}.pkl', 'rb') as f:
        d = pickle.load(f)
    return np.fromiter((d.get(i, 0.0) for i in range(N)), dtype=float, count=N)


# ---------------------------------------------------------------- solvers
def run_iteration(step, x0, bias):
    """x_{k+1} = renorm(step(x_k)); returns x, iterations, converged, final diff, last ratio."""
    x = x0
    prev = None
    ratio = float('nan')
    for it in range(1, args.max_iter + 1):
        x_new = step(x)
        s = x_new.sum()
        if s > 0:
            x_new = x_new / s
        diff = float(np.abs(x_new - x).sum())
        if prev is not None and prev > 0:
            ratio = diff / prev
        prev = diff
        x = x_new
        if diff < args.tol:
            return x, it, True, diff, ratio
    return x, args.max_iter, False, diff, ratio


def uniform_solver(bias):
    """Legacy uniform-transition solver of acl_rlg_final_eval_temporal.py (init 1/N, no renorm)."""
    M = csr_matrix((np.ones(len(edge_src)), (edge_tgt, edge_src)), shape=(N, N))
    cs = np.asarray(M.sum(axis=0)).flatten()
    Mn = M.multiply(1.0 / np.where(cs == 0, 1.0, cs))

    def step(x):
        return (1 - ALPHA) * Mn.dot(x) + (1 - ALPHA) * float(x[dangling].sum()) * bias + ALPHA * bias
    x = np.full(N, 1.0 / N)
    prev = None
    ratio = float('nan')
    for it in range(1, args.max_iter + 1):
        x_new = step(x)
        diff = float(np.abs(x_new - x).sum())
        if prev is not None and prev > 0:
            ratio = diff / prev
        prev = diff
        x = x_new
        if diff < args.tol:
            return x, it, True, diff, ratio
    return x, args.max_iter, False, diff, ratio


def tstar_weights(t):
    th = theta[:, t]
    total = th.sum() or 1.0
    out_sum = np.bincount(edge_src, weights=th[edge_tgt], minlength=N)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_factor = np.sqrt(np.where(out_sum > 0, th / (total * out_sum), 0.0))
    col_factor = np.sqrt(np.maximum(th, 0.0))
    return row_factor[edge_src] * col_factor[edge_tgt]


def per_target_matrix(w, fallback):
    rs = np.bincount(edge_tgt, weights=w, minlength=N)
    with np.errstate(divide='ignore', invalid='ignore'):
        wn = np.where(rs[edge_tgt] > 0, w / rs[edge_tgt], (1.0 / in_degree[edge_tgt]) if fallback else 0.0)
    return csr_matrix((wn, (edge_tgt, edge_src)), shape=(N, N))


def transition_solver(Mn, bias, init):
    def step(x):
        return (1 - ALPHA) * Mn.dot(x) + (1 - ALPHA) * float(x[dangling].sum()) * bias + ALPHA * bias
    return run_iteration(step, init, bias)


def solve(mode, t):
    th = theta[:, t]
    s = th.sum()
    theta_bias = th / s if s > 1e-12 else normal_bias
    if mode == 'personalized':
        return uniform_solver(theta_bias)
    if mode == 'themed':                                   # legacy: zero rows, init 1/N, renorm
        Mn = per_target_matrix(tstar_weights(t), fallback=False)
        return transition_solver(Mn, theta_bias, np.full(N, 1.0 / N))
    if mode == 'themed_fb':                                # production semantics
        Mn = per_target_matrix(tstar_weights(t), fallback=True)
        return transition_solver(Mn, theta_bias, theta_bias.copy())
    kind, _, suffix = mode.partition('_modified_')
    c = load_cent(kind, t)
    if suffix == 'bias':
        raw = th * c
        sm = raw.sum()
        return uniform_solver(raw / sm if sm > 1e-12 else normal_bias)
    if suffix == 'transition':
        Mn = per_target_matrix(tstar_weights(t) * c[edge_src], fallback=True)
        return transition_solver(Mn, theta_bias, theta_bias.copy())
    raise ValueError(mode)


# ---------------------------------------------------------------- run
out_dir = OUT_ROOT / args.tag / str(YEAR)
out_dir.mkdir(parents=True, exist_ok=True)
for mode in args.modes.split(','):
    dest = out_dir / f'{mode}.npz'
    if dest.exists():
        print(f'[skip] {mode}', flush=True)
        continue
    t0 = time.time()
    S = np.zeros((nq, N))
    its = np.zeros(T, dtype=int)
    conv = np.zeros(T, dtype=bool)
    fin = np.zeros(T)
    rat = np.full(T, np.nan)
    for t in range(args.max_topics):
        x, it, ok, diff, ratio = solve(mode, t)
        S += np.outer(theta_q[:, t], x)
        its[t], conv[t], fin[t], rat[t] = it, ok, diff, ratio
    done = args.max_topics
    np.savez_compressed(dest, S=S, queries=np.array([f'{a}|{i}' for a, i, _ in queries]),
                        iterations=its[:done], converged=conv[:done], final_diff=fin[:done],
                        ratio=rat[:done], theta_q=theta_q, tol=args.tol, max_iter=args.max_iter)
    print(f'[done] {mode}: {time.time()-t0:.0f}s  nonconverged {int((~conv[:done]).sum())}/{done}  '
          f'iters median {int(np.median(its[:done]))} max {int(its[:done].max())}', flush=True)
print('all done', flush=True)
