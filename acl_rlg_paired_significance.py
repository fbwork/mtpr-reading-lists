"""
Paired significance test, per tutorial, for bw_modified_bias vs. BM25 on
ACL-rlg (temporally-corrected PR(d,t)). Reuses cached PR(d,t) per year
(no new PageRank/centrality computation) and the official BM25
predictions already saved under the ACL-rlg repo's experiments folder.

For each of the 85 tutorials, scores are averaged over the 3 annotator
query sets first (one score per tutorial per method), to avoid treating
3 annotator-level observations of the same tutorial as independent
samples. A two-sided Wilcoxon signed-rank test is then run on the 85
paired (bw_modified_bias, BM25) differences, for NDCG@20 and MRR@20.
"""
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, r"E:\0000000-Thesis\1405\thesis-1405\thesis-1405")

RLG_BASE = Path(r"E:\0000000-Thesis\1403-new approach\ACL-rlg\aclanthology-tutorial-reading-lists")
EVAL_DATA = Path(r"E:\0000000-Thesis\1405\thesis-1405\acl-rlg-eval\data")
ACL_DATASET_PATH = RLG_BASE / "experiments/classic_methods/acl_anthology_dataset/acl_anthology_dataset.json"
READING_LISTS_CSV = RLG_BASE / "reading_lists.csv"
ANNOT_DIR = RLG_BASE / "annotations"
BM25_PREDS_DIR = RLG_BASE / "experiments/classic_methods/preds/bm25"

OUT_DIR = Path(r"E:\0000000-Thesis\1405\thesis-1405\thesis-1405\acl_rlg_pilot_out")
CLUSTER = 99
T = 200
NEIGHBOR_K = 20
MODE = 'bw_modified_bias'
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


def main():
    print("Loading candidate corpus, theta, cached PR(d,t) per year...", flush=True)
    with open(ACL_DATASET_PATH, encoding='utf-8') as f:
        acl_dataset = json.load(f)
    acl_ids = sorted(acl_dataset.keys())
    N = len(acl_ids)
    years = np.array([acl_dataset[aid]['year'] for aid in acl_ids])
    clean_titles = np.array([clean_string(acl_dataset[aid].get('title', '')) for aid in acl_ids])

    from components.theta_artifact import load_theta_artifact
    theta, meta = load_theta_artifact(T, cluster=CLUSTER)

    import pickle
    pr_by_year = {}
    with open(OUT_DIR / "pr_all_modes.pkl", 'rb') as f:
        pr_by_year[FULL_YEAR] = pickle.load(f)['pr']
    for cutoff in TRUNCATED_YEARS:
        with open(OUT_DIR / f"pr_all_modes_y{cutoff}.pkl", 'rb') as f:
            pr_by_year[cutoff] = pickle.load(f)

    print("Building TF-IDF index for query neighbors (title+abstract)...", flush=True)
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

    # tutorial_id -> {'ndcg': [per-annotator MTPR], 'mrr': [...]}, same for bm25
    mtpr_scores = {}
    bm25_scores = {}

    for annot_i in [1, 2, 3]:
        annot_path = ANNOT_DIR / f"annotation_{annot_i}.csv"
        df_annot = pd.read_csv(annot_path)

        with open(BM25_PREDS_DIR / f"preds_annot{annot_i}.json", encoding='utf-8') as f:
            bm25_preds = json.load(f)

        print(f"Evaluating annotation set {annot_i} ({len(df_annot)} tutorials)...", flush=True)
        for qi, row in df_annot.iterrows():
            tid = row['id']
            query = row['query_keywords']
            year = int(row['year'])
            true = trues.get(tid, [])
            if not true:
                continue

            pr = pr_by_year[year]
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

            score = theta_q @ pr[MODE]
            score_masked = np.where(years <= year, score, -1)
            ranked = np.argsort(-score_masked)[:20]
            mtpr_preds = clean_titles[ranked].tolist()

            mtpr_scores.setdefault(tid, {'ndcg': [], 'mrr': []})
            mtpr_scores[tid]['ndcg'].append(ndcg(true, mtpr_preds) * 100)
            mtpr_scores[tid]['mrr'].append(mrr(true, mtpr_preds) * 100)

            bm25_entry = bm25_preds.get(tid)
            bm25_preds_titles = [clean_string(p.get('title', '')) for p in (bm25_entry or [])][:20]
            bm25_scores.setdefault(tid, {'ndcg': [], 'mrr': []})
            bm25_scores[tid]['ndcg'].append(ndcg(true, bm25_preds_titles) * 100)
            bm25_scores[tid]['mrr'].append(mrr(true, bm25_preds_titles) * 100)

    # Aggregate to one value per tutorial (mean over the annotators that covered it)
    common_tids = sorted(set(mtpr_scores) & set(bm25_scores))
    print(f"\n{len(common_tids)} tutorials with both MTPR and BM25 scores (of 85 total).", flush=True)

    mtpr_ndcg = np.array([np.mean(mtpr_scores[t]['ndcg']) for t in common_tids])
    bm25_ndcg = np.array([np.mean(bm25_scores[t]['ndcg']) for t in common_tids])
    mtpr_mrr = np.array([np.mean(mtpr_scores[t]['mrr']) for t in common_tids])
    bm25_mrr = np.array([np.mean(bm25_scores[t]['mrr']) for t in common_tids])

    print(f"\nMean NDCG@20: bw_modified_bias={mtpr_ndcg.mean():.3f}  BM25={bm25_ndcg.mean():.3f}", flush=True)
    print(f"Mean MRR@20:  bw_modified_bias={mtpr_mrr.mean():.3f}  BM25={bm25_mrr.mean():.3f}", flush=True)

    diff_ndcg = mtpr_ndcg - bm25_ndcg
    diff_mrr = mtpr_mrr - bm25_mrr
    print(f"\nMedian paired diff NDCG@20 (MTPR-BM25): {np.median(diff_ndcg):.3f}", flush=True)
    print(f"Median paired diff MRR@20  (MTPR-BM25): {np.median(diff_mrr):.3f}", flush=True)
    n_nonzero_ndcg = np.sum(diff_ndcg != 0)
    n_nonzero_mrr = np.sum(diff_mrr != 0)
    print(f"Non-zero diffs: NDCG={n_nonzero_ndcg}/{len(diff_ndcg)}  MRR={n_nonzero_mrr}/{len(diff_mrr)}", flush=True)

    p_n, p_m = None, None
    try:
        stat_n, p_n = wilcoxon(mtpr_ndcg, bm25_ndcg)
        print(f"\nWilcoxon signed-rank, NDCG@20: W={stat_n:.2f}, p={p_n:.5f}", flush=True)
    except Exception as e:
        print(f"NDCG Wilcoxon failed: {e}", flush=True)

    try:
        stat_m, p_m = wilcoxon(mtpr_mrr, bm25_mrr)
        print(f"Wilcoxon signed-rank, MRR@20:  W={stat_m:.2f}, p={p_m:.5f}", flush=True)
    except Exception as e:
        print(f"MRR Wilcoxon failed: {e}", flush=True)

    with open(OUT_DIR / "paired_significance_results.json", 'w') as f:
        json.dump({
            'n_tutorials': len(common_tids),
            'mean_ndcg_mtpr': float(mtpr_ndcg.mean()),
            'mean_ndcg_bm25': float(bm25_ndcg.mean()),
            'mean_mrr_mtpr': float(mtpr_mrr.mean()),
            'mean_mrr_bm25': float(bm25_mrr.mean()),
            'median_diff_ndcg': float(np.median(diff_ndcg)),
            'median_diff_mrr': float(np.median(diff_mrr)),
            'wilcoxon_ndcg_p': float(p_n) if p_n is not None else None,
            'wilcoxon_mrr_p': float(p_m) if p_m is not None else None,
        }, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'paired_significance_results.json'}", flush=True)


if __name__ == '__main__':
    main()
