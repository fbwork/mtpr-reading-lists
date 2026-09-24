"""
Temporally-consistent version of acl_rlg_final_eval.py: fixes the
citation-graph leakage in the original evaluation, where centrality and
PR(d,t) were computed once on the FULL (2024) citation graph and reused
for every tutorial regardless of its year, even though the final
candidate list was already year-filtered. Here, each of the 7 distinct
ACL-rlg tutorial years gets its own PR(d,t), computed on a citation
graph truncated to papers published no later than that year, using the
centrality already computed per year by acl_rlg_centrality_worker_temporal.py.
2024 is the corpus's own max year, so it equals the full graph -- the
existing pr_all_modes.pkl from the original (non-temporal) run is
reused unchanged for 2024 tutorials rather than recomputed.

theta (P(t|d)) and the query-side TF-IDF vectorizer are NOT rebuilt per
year: both come from each paper's own fixed text, not from the
citation graph, so they are not the leakage vector this fixes. This is
a documented scope decision, not an oversight (Limitations).
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
CLUSTER = 99
T = 200
ALPHA = 0.15
NEIGHBOR_K = 20
TAU = 0.01
MODES = ['normal', 'personalized', 'themed', 'bw_modified_bias', 'hybrid']
TRUNCATED_YEARS = [2016, 2019, 2020, 2021, 2022, 2023]
FULL_YEAR = 2024


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


def compute_pr_for_graph(edge_src, edge_tgt, N, theta, centrality_dir):
    """Compute PR(d,t) for all 5 modes/200 topics on one (possibly
    temporally-truncated) graph, using centrality pickles from centrality_dir."""
    out_degree = np.bincount(edge_src, minlength=N)
    dangling_mask = out_degree == 0

    def uniform_transition_pagerank(bias_vec, max_iter=200, tol=1e-9):
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
        row_sums = np.asarray(M.sum(axis=1)).flatten()
        row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
        inv_row = 1.0 / row_sums_safe
        M_norm = M.multiply(inv_row.reshape(-1, 1))
        bias_raw = theta_t.copy()
        bias_sum = bias_raw.sum()
        bias_vec = bias_raw / bias_sum if bias_sum > 1e-12 else np.full(N, 1.0 / N)
        x = np.full(N, 1.0 / N)
        for _ in range(max_iter):
            dangling_contrib = (1 - ALPHA) * float(x[dangling_mask].sum()) * bias_vec
            x_new = (1 - ALPHA) * M_norm.dot(x) + dangling_contrib + ALPHA * bias_vec
            total = x_new.sum()
            if total > 0:
                x_new = x_new / total
            if np.abs(x_new - x).sum() < tol:
                x = x_new
                break
            x = x_new
        return x

    pr = {mode: np.zeros((T, N)) for mode in MODES}
    normal_bias = np.full(N, 1.0 / N)
    normal_pr = uniform_transition_pagerank(normal_bias)
    for t in range(T):
        pr['normal'][t] = normal_pr

    for t in range(T):
        theta_t = theta[:, t]
        s = theta_t.sum()
        bias = theta_t / s if s > 1e-12 else normal_bias
        pr['personalized'][t] = uniform_transition_pagerank(bias)

    for t in range(T):
        pr['themed'][t] = themed_pagerank(t)

    for t in range(T):
        with open(centrality_dir / f"betweenness_{t}.pkl", 'rb') as f:
            bw = pickle.load(f)
        bw_arr = np.array([bw.get(i, 0.0) for i in range(N)])
        theta_t = theta[:, t]
        raw_bias = theta_t * bw_arr
        s = raw_bias.sum()
        bias = raw_bias / s if s > 1e-12 else normal_bias
        pr['bw_modified_bias'][t] = uniform_transition_pagerank(bias)

    for t in range(T):
        with open(centrality_dir / f"degree_{t}.pkl", 'rb') as f:
            deg = pickle.load(f)
        with open(centrality_dir / f"closeness_{t}.pkl", 'rb') as f:
            clo = pickle.load(f)
        with open(centrality_dir / f"betweenness_{t}.pkl", 'rb') as f:
            bet = pickle.load(f)
        with open(centrality_dir / f"katz_{t}.pkl", 'rb') as f:
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

    return pr


def main():
    t0 = time.time()
    print("Loading candidate corpus, full graph, theta...", flush=True)
    with open(ACL_DATASET_PATH, encoding='utf-8') as f:
        acl_dataset = json.load(f)
    acl_ids = sorted(acl_dataset.keys())
    N = len(acl_ids)
    aid2idx = {aid: i for i, aid in enumerate(acl_ids)}
    years = np.array([acl_dataset[aid]['year'] for aid in acl_ids])

    with open(ACL_GRAPH_PATH, encoding='utf-8') as f:
        raw_graph = json.load(f)

    from components.theta_artifact import load_theta_artifact
    theta, meta = load_theta_artifact(T, cluster=CLUSTER)
    print(f"  theta: {theta.shape}", flush=True)

    # ── PR(d,t) per year: 2024 reused from the original full-graph run;
    # the other 6 years computed fresh on their truncated graphs. ──────────
    pr_by_year = {}

    print(f"\nLoading existing full-graph (2024) PR(d,t)...", flush=True)
    with open(OUT_DIR / "pr_all_modes.pkl", 'rb') as f:
        pr_by_year[FULL_YEAR] = pickle.load(f)['pr']

    for cutoff in TRUNCATED_YEARS:
        cache_path = OUT_DIR / f"pr_all_modes_y{cutoff}.pkl"
        if cache_path.exists():
            print(f"\nLoading cached PR(d,t) for year {cutoff}...", flush=True)
            with open(cache_path, 'rb') as f:
                pr_by_year[cutoff] = pickle.load(f)
            continue

        print(f"\nComputing PR(d,t) for year {cutoff}...", flush=True)
        t_y = time.time()
        eligible_mask = years <= cutoff
        src_list, tgt_list = [], []
        for src, tgts in raw_graph.items():
            si = aid2idx.get(src)
            if si is None or not eligible_mask[si]:
                continue
            for tgt in tgts:
                ti = aid2idx.get(tgt)
                if ti is not None and eligible_mask[ti]:
                    src_list.append(si)
                    tgt_list.append(ti)
        edge_src = np.array(src_list, dtype=np.int64)
        edge_tgt = np.array(tgt_list, dtype=np.int64)
        print(f"  {eligible_mask.sum():,} eligible papers, {len(edge_src):,} edges", flush=True)

        centrality_dir = OUT_DIR / f"centrality_y{cutoff}"
        pr = compute_pr_for_graph(edge_src, edge_tgt, N, theta, centrality_dir)
        pr_by_year[cutoff] = pr
        with open(cache_path, 'wb') as f:
            pickle.dump(pr, f)
        print(f"  done in {(time.time()-t_y)/60:.1f} min", flush=True)

    print(f"\nAll years' PR(d,t) ready. Total so far: {(time.time()-t0)/60:.1f} min", flush=True)

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

            pr = pr_by_year[year]  # <-- the temporal fix: year-specific PR(d,t)

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
                score = theta_q @ pr[mode]
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
    print("FINAL RESULTS -- TEMPORALLY-CONSISTENT (mean across 3 annotation sets)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Method':<20} {'Recall@20':>10} {'NDCG@20':>10} {'MRR@20':>10}", flush=True)
    final = {}
    for mode in MODES:
        mean_r = np.mean([all_results[mode][f'A{i}']['Recall@20'] for i in [1, 2, 3]])
        mean_n = np.mean([all_results[mode][f'A{i}']['NDCG@20'] for i in [1, 2, 3]])
        mean_m = np.mean([all_results[mode][f'A{i}']['MRR@20'] for i in [1, 2, 3]])
        final[mode] = {'Recall@20': mean_r, 'NDCG@20': mean_n, 'MRR@20': mean_m}
        print(f"{mode:<20} {mean_r:>10.2f} {mean_n:>10.2f} {mean_m:>10.2f}", flush=True)

    with open(OUT_DIR / "final_results_temporal.json", 'w') as f:
        json.dump({'per_annotation': all_results, 'mean': final}, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'final_results_temporal.json'}", flush=True)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
