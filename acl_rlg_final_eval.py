"""
Final full-LDA-based MTPR evaluation on ACL-rlg: normal, personalized,
themed, bw_modified_bias, hybrid (modified_bias skipped -- no author
metadata available in the candidate corpus, per user decision), using the
frozen Theta artifact (cluster=99) and the centrality caches already
computed by acl_rlg_centrality_worker.py (4 parallel terminals, ~9.5h).

Configurations were fixed a priori from the AAN ablation; nothing here is
tuned on ACL-rlg results. Evaluated with the official Recall@20, NDCG@20,
MRR@20 (matching acl-rlg-eval/run_eval.py's implementation) across all
three annotation sets, mean reported. Query construction: deterministic
TF-IDF top-20 retrieval (temporally filtered, no future-paper leakage)
-> mean of their fixed Theta rows -> theta_q; Score(d,q) = sum_t
theta_q[t] * PR(d,t). Final ranking is temporally filtered to year <=
tutorial year, matching the official BM25 protocol exactly.
"""
import ast
import json
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

OUT_DIR = Path(r"E:\0000000-Thesis\1405\thesis-1405\thesis-1405\acl_rlg_pilot_out")
CENTRALITY_DIR = OUT_DIR / "centrality"
CLUSTER = 99
T = 200
ALPHA = 0.15
NEIGHBOR_K = 20
TAU = 0.01
MODES = ['normal', 'personalized', 'themed', 'bw_modified_bias', 'hybrid']


def clean_string(s):
    return re.sub(r'\W+', '', s or '').lower()


def recall(trues, preds):
    return len([t for t in trues if t in preds]) / len(trues) if trues else 0.0


def ndcg(trues, preds):
    rel = [1 if p in trues else 0 for p in preds]
    dcg = sum((2 ** r - 1) / np.log2(i + 1) for i, r in enumerate(rel, 1))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, len(trues) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(trues, preds):
    for i, p in enumerate(preds, 1):
        if p in trues:
            return 1.0 / i
    return 0.0


def main():
    t0 = time.time()
    print("Loading candidate corpus, graph, theta...", flush=True)
    with open(ACL_DATASET_PATH, encoding='utf-8') as f:
        acl_dataset = json.load(f)
    acl_ids = sorted(acl_dataset.keys())
    N = len(acl_ids)
    aid2idx = {aid: i for i, aid in enumerate(acl_ids)}
    years = np.array([acl_dataset[aid]['year'] for aid in acl_ids])

    with open(ACL_GRAPH_PATH, encoding='utf-8') as f:
        raw_graph = json.load(f)
    src_list, tgt_list = [], []
    for src, tgts in raw_graph.items():
        si = aid2idx.get(src)
        if si is None:
            continue
        for tgt in tgts:
            ti = aid2idx.get(tgt)
            if ti is not None:
                src_list.append(si)
                tgt_list.append(ti)
    edge_src = np.array(src_list, dtype=np.int64)
    edge_tgt = np.array(tgt_list, dtype=np.int64)
    out_degree = np.bincount(edge_src, minlength=N)
    dangling_mask = out_degree == 0
    print(f"  {N:,} nodes, {len(edge_src):,} edges", flush=True)

    from components.theta_artifact import load_theta_artifact
    theta, meta = load_theta_artifact(T, cluster=CLUSTER)
    print(f"  theta: {theta.shape}, sha256={meta['artifact_sha256'][:16]}...", flush=True)

    def uniform_transition_pagerank(bias_vec, max_iter=200, tol=1e-9):
        """Standard PageRank with uniform (out-degree) transition and a given
        personalization/dangling bias vector -- matches NORMAL/PERSONALIZED/
        BW_MODIFIED_BIAS/HYBRID's transition() dispatch (base_transition =
        1/out_degree for all four)."""
        w = np.ones(len(edge_src))
        M = csr_matrix((w, (edge_tgt, edge_src)), shape=(N, N))
        col_sums = np.asarray(M.sum(axis=0)).flatten()
        col_sums_safe = np.where(col_sums == 0, 1.0, col_sums)
        M_norm = M.multiply(1.0 / col_sums_safe)
        x = np.full(N, 1.0 / N)
        for _ in range(max_iter):
            dangling_contrib = (1 - ALPHA) * float(x[dangling_mask].sum()) * bias_vec
            x_new = (1 - ALPHA) * M_norm.dot(x) + dangling_contrib + ALPHA * bias_vec
            if np.abs(x_new - x).sum() < tol:
                x = x_new
                break
            x = x_new
        return x

    def themed_pagerank(t, max_iter=200, tol=1e-9):
        """Jardine T* transition (Eq. eq:tstar), normalized per-target over
        citers (Eq. eq:transnorm), bias = P(t|d) normalized -- matches
        THEMED's bias()/transition() dispatch exactly, vectorized via the
        row/col-separable form of sqrt(a*b) = sqrt(a)*sqrt(b)."""
        theta_t = theta[:, t]
        total = theta_t.sum()
        if total <= 0:
            total = 1.0
        out_sum = np.bincount(edge_src, weights=theta_t[edge_tgt], minlength=N)
        with np.errstate(divide='ignore', invalid='ignore'):
            row_factor = np.sqrt(np.where(out_sum > 0, theta_t / (total * out_sum), 0.0))
        col_factor = np.sqrt(np.maximum(theta_t, 0.0))
        w = row_factor[edge_src] * col_factor[edge_tgt]

        M = csr_matrix((w, (edge_tgt, edge_src)), shape=(N, N))
        row_sums = np.asarray(M.sum(axis=1)).flatten()  # per-target normalization over citers
        row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
        inv_row = 1.0 / row_sums_safe
        M_norm = M.multiply(inv_row.reshape(-1, 1))
        # rows with zero citers (row_sums==0) fall back to uniform over citers,
        # but if there are no citers there's nothing to normalize -- those
        # rows just get zero inflow, matching normalized_transition_weights()
        # returning {} for citers=[].

        bias_raw = theta_t.copy()
        bias_sum = bias_raw.sum()
        bias_vec = bias_raw / bias_sum if bias_sum > 1e-12 else np.full(N, 1.0 / N)

        x = np.full(N, 1.0 / N)
        for _ in range(max_iter):
            dangling_contrib = (1 - ALPHA) * float(x[dangling_mask].sum()) * bias_vec
            x_new = (1 - ALPHA) * M_norm.dot(x) + dangling_contrib + ALPHA * bias_vec
            # per-iteration renormalization: T*'s transition matrix is not
            # guaranteed row/column-stochastic in a way that keeps the
            # unnormalized recurrence bounded (documented AAN finding,
            # Section "Numerical Stability" -- without this, THEMED's
            # scores diverge to ~1e15+ instead of summing to 1).
            total = x_new.sum()
            if total > 0:
                x_new = x_new / total
            if np.abs(x_new - x).sum() < tol:
                x = x_new
                break
            x = x_new
        return x

    print("\nComputing PR(d,t) for all modes/topics...", flush=True)
    pr = {mode: np.zeros((T, N)) for mode in MODES}

    t_normal = time.time()
    normal_bias = np.full(N, 1.0 / N)
    normal_pr = uniform_transition_pagerank(normal_bias)
    for t in range(T):
        pr['normal'][t] = normal_pr  # query-independent, same for every topic
    print(f"  normal: {time.time()-t_normal:.1f}s (query-independent, computed once)", flush=True)

    t_pers = time.time()
    for t in range(T):
        theta_t = theta[:, t]
        s = theta_t.sum()
        bias = theta_t / s if s > 1e-12 else normal_bias
        pr['personalized'][t] = uniform_transition_pagerank(bias)
    print(f"  personalized: {(time.time()-t_pers)/60:.1f} min ({T} topics)", flush=True)

    t_themed = time.time()
    for t in range(T):
        pr['themed'][t] = themed_pagerank(t)
    print(f"  themed: {(time.time()-t_themed)/60:.1f} min ({T} topics)", flush=True)

    t_bw = time.time()
    for t in range(T):
        with open(CENTRALITY_DIR / f"betweenness_{t}.pkl", 'rb') as f:
            bw = pickle.load(f)
        bw_arr = np.array([bw.get(i, 0.0) for i in range(N)])
        theta_t = theta[:, t]
        raw_bias = theta_t * bw_arr
        s = raw_bias.sum()
        bias = raw_bias / s if s > 1e-12 else normal_bias
        pr['bw_modified_bias'][t] = uniform_transition_pagerank(bias)
    print(f"  bw_modified_bias: {(time.time()-t_bw)/60:.1f} min ({T} topics)", flush=True)

    t_hyb = time.time()
    for t in range(T):
        with open(CENTRALITY_DIR / f"degree_{t}.pkl", 'rb') as f:
            deg = pickle.load(f)
        with open(CENTRALITY_DIR / f"closeness_{t}.pkl", 'rb') as f:
            clo = pickle.load(f)
        with open(CENTRALITY_DIR / f"betweenness_{t}.pkl", 'rb') as f:
            bet = pickle.load(f)
        with open(CENTRALITY_DIR / f"katz_{t}.pkl", 'rb') as f:
            katz = pickle.load(f)

        theta_t = theta[:, t]
        deg_arr = np.array([deg.get(i, 0.0) for i in range(N)])
        clo_arr = np.array([clo.get(i, 0.0) for i in range(N)])
        bet_arr = np.array([bet.get(i, 0.0) for i in range(N)])
        katz_arr = np.array([katz.get(i, 0.0) for i in range(N)])

        def norm_sum(a):
            s = a.sum()
            return a / s if s > 1e-12 else np.zeros_like(a)

        centrality_score = 0.25 * (norm_sum(deg_arr) + norm_sum(clo_arr) + norm_sum(bet_arr) + norm_sum(katz_arr))
        raw_bias = theta_t * centrality_score
        s = raw_bias.sum()
        bias = raw_bias / s if s > 1e-12 else normal_bias
        pr['hybrid'][t] = uniform_transition_pagerank(bias)
    print(f"  hybrid: {(time.time()-t_hyb)/60:.1f} min ({T} topics)", flush=True)

    with open(OUT_DIR / "pr_all_modes.pkl", 'wb') as f:
        pickle.dump({'pr': pr, 'acl_ids': acl_ids}, f)
    print(f"\nAll PR(d,t) saved. Total so far: {(time.time()-t0)/60:.1f} min", flush=True)

    # ── Query construction + evaluation over all tutorials/annotations ─────
    print("\nBuilding TF-IDF index for query neighbors...", flush=True)
    corpus_texts = [f"{acl_dataset[aid].get('title', '')} {acl_dataset[aid].get('abstract', '') or ''}" for aid in acl_ids]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=30_000, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)

    df_gold = pd.read_csv(READING_LISTS_CSV)
    df_gold['reading_list'] = df_gold['reading_list'].apply(ast.literal_eval)
    trues = {}
    for _, row in df_gold.iterrows():
        refs = [clean_string(p['title']) for p in row['reading_list']
                if str(p.get('in_acl', 'False')) == 'True' and p.get('title')]
        trues[row['id']] = refs

    clean_titles = np.array([clean_string(acl_dataset[aid].get('title', '')) for aid in acl_ids])

    all_results = {mode: {} for mode in MODES}
    for annot_i in [1, 2, 3]:
        annot_path = ANNOT_DIR / f"annotation_{annot_i}.csv"
        df_annot = pd.read_csv(annot_path)
        print(f"\nEvaluating annotation set {annot_i} ({len(df_annot)} tutorials)...", flush=True)

        scores = {mode: {'Recall@20': [], 'NDCG@20': [], 'MRR@20': []} for mode in MODES}
        for qi, row in df_annot.iterrows():
            tid = row['id']
            query = row['query_keywords']
            year = int(row['year'])
            true = trues.get(tid, [])
            if not true:
                continue

            qvec = vectorizer.transform([str(query)])
            sims = cosine_similarity(qvec, tfidf_matrix)[0]
            eligible = np.where(years <= year)[0]
            if len(eligible) == 0:
                continue
            elig_sims = sims[eligible]
            top20_local = eligible[np.argsort(-elig_sims)[:NEIGHBOR_K]]

            theta_q = theta[top20_local].mean(axis=0)
            s = theta_q.sum()
            theta_q = theta_q / s if s > 1e-12 else np.full(T, 1.0 / T)

            for mode in MODES:
                score = theta_q @ pr[mode]  # (T,) @ (T,N) -> (N,)
                score_masked = np.where(years <= year, score, -1)
                ranked = np.argsort(-score_masked)[:20]
                preds = clean_titles[ranked].tolist()
                scores[mode]['Recall@20'].append(recall(true, preds) * 100)
                scores[mode]['NDCG@20'].append(ndcg(true, preds) * 100)
                scores[mode]['MRR@20'].append(mrr(true, preds) * 100)

            if (qi + 1) % 15 == 0:
                print(f"    [{qi+1}/{len(df_annot)}] done", flush=True)

        for mode in MODES:
            all_results[mode][f'A{annot_i}'] = {k: float(np.mean(v)) if v else 0.0 for k, v in scores[mode].items()}

    print("\n" + "=" * 70, flush=True)
    print("FINAL RESULTS (mean across 3 annotation sets)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Method':<20} {'Recall@20':>10} {'NDCG@20':>10} {'MRR@20':>10}", flush=True)
    final = {}
    for mode in MODES:
        mean_r = np.mean([all_results[mode][f'A{i}']['Recall@20'] for i in [1, 2, 3]])
        mean_n = np.mean([all_results[mode][f'A{i}']['NDCG@20'] for i in [1, 2, 3]])
        mean_m = np.mean([all_results[mode][f'A{i}']['MRR@20'] for i in [1, 2, 3]])
        final[mode] = {'Recall@20': mean_r, 'NDCG@20': mean_n, 'MRR@20': mean_m}
        print(f"{mode:<20} {mean_r:>10.2f} {mean_n:>10.2f} {mean_m:>10.2f}", flush=True)

    with open(OUT_DIR / "final_results.json", 'w') as f:
        json.dump({'per_annotation': all_results, 'mean': final}, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'final_results.json'}", flush=True)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
