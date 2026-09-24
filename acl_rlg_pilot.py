"""
Pilot for full LDA-based MTPR on ACL-rlg (per user + ChatGPT plan):
ATR -> LDA (T=200) -> frozen Theta -> normal PageRank -> bw_modified_bias
for ONE pilot query, to validate ID mapping, query construction, non-zero
scores, and get a realistic per-topic betweenness timing estimate on the
much larger (97,440-node) ACL Anthology graph before committing to the
full multi-day run.

Deliberately standalone / does NOT touch components/output/lda or
components/output/pr: those caches (especially centrality) are keyed only
by (mode, topic_id), NOT by corpus/cluster, so reusing them here would
silently corrupt the AAN artifacts rebuilt this session. LDA model
training/saving uses cluster=99 (cluster-safe paths, verified in
theta_artifact.py/lda.py), but centrality/PageRank are computed here with
plain networkx/numpy into a private directory instead.

Wrapped in main()/__main__ guard: LdaMulticore uses Python multiprocessing
with spawn on Windows, which re-imports this file in each worker unless
guarded -- an earlier unguarded version crashed with exactly this error.
"""
import ast
import json
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RLG_BASE = Path(r"E:\0000000-Thesis\1403-new approach\ACL-rlg\aclanthology-tutorial-reading-lists")
EVAL_DATA = Path(r"E:\0000000-Thesis\1405\thesis-1405\acl-rlg-eval\data")
ACL_DATASET_PATH = RLG_BASE / "experiments/classic_methods/acl_anthology_dataset/acl_anthology_dataset.json"
ACL_GRAPH_PATH = EVAL_DATA / "acl_full_citation_graph.json"
READING_LISTS_CSV = RLG_BASE / "reading_lists.csv"
ANNOTATION_1_CSV = RLG_BASE / "annotations/annotation_1.csv"

OUT_DIR = Path(r"E:\0000000-Thesis\1405\thesis-1405\thesis-1405\acl_rlg_pilot_out")

CLUSTER = 99  # isolated from AAN's cluster=0 artifacts
T = 200
ALPHA = 0.15
NEIGHBOR_K = 20
TOP_TOPICS_FOR_PILOT = 15  # only compute betweenness/PageRank for the top-N topics by theta_q weight (pilot shortcut)

sys.path.insert(0, r"E:\0000000-Thesis\1405\thesis-1405\thesis-1405")


def title_ngrams(title, clean_title, clean_gram):
    grams = []
    for part in clean_title(title or ""):
        tokens = [t for g in part.split() if (t := clean_gram(g))]
        n = len(tokens)
        if n < 3:
            continue
        for i in range(1, n):
            for j in range(len(tokens) - i + 1):
                seq = tokens[j:j + i]
                if any((not v or v.isdigit() or '&' in v) for v in seq):
                    continue
                grams.append(' '.join(seq))
    return grams


def clean_string(s):
    return re.sub(r'\W+', '', s or '').lower()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print("Loading candidate corpus...", flush=True)
    with open(ACL_DATASET_PATH, encoding='utf-8') as f:
        acl_dataset = json.load(f)
    acl_ids = sorted(acl_dataset.keys())
    N = len(acl_ids)
    aid2idx = {aid: i for i, aid in enumerate(acl_ids)}
    print(f"  {N:,} papers", flush=True)

    print("Extracting ATR n-gram terms from titles...", flush=True)
    from components.atr import clean_title, clean_gram
    from nltk.corpus import stopwords
    STOP = set(stopwords.words("english"))

    def trim_stopword_boundary(g):
        parts = g.split()
        return parts[-1] not in STOP and parts[0] not in STOP

    doc_grams = []
    gram_doc_freq = Counter()
    for aid in acl_ids:
        grams = title_ngrams(acl_dataset[aid].get('title', ''), clean_title, clean_gram)
        doc_grams.append(grams)
        gram_doc_freq.update(set(grams))

    vocab = {g for g, c in gram_doc_freq.items() if c >= 2}
    vocab = {g for g in vocab if trim_stopword_boundary(g)}
    print(f"  ATR vocabulary: {len(vocab):,} terms", flush=True)

    docs = [[g for g in grams if g in vocab] for grams in doc_grams]
    n_empty = sum(1 for d in docs if not d)
    print(f"  {n_empty:,} / {N:,} papers with empty bag-of-words (title too short / no surviving terms)", flush=True)

    print(f"\nTraining LDA (T={T})... this is the big one, timing it", flush=True)
    from components.lda import GensimMcLDA

    t_lda = time.time()
    dictionary = Dictionary(docs)
    lda = GensimMcLDA(lda_no_topics=T, docs=docs, dictionary=dictionary, cluster=CLUSTER)
    lda.model = lda.generate_model(lda.bag_of_words, lda.dictionary)
    lda.save_model(cluster=CLUSTER)
    print(f"  LDA trained in {(time.time() - t_lda) / 60:.1f} min", flush=True)

    print("\nBuilding frozen Theta artifact...", flush=True)
    from components.theta_artifact import build_theta_artifact, THETA_EXTRACTION_SEED

    t_theta = time.time()
    meta = build_theta_artifact(lda, cluster=CLUSTER, seed=THETA_EXTRACTION_SEED)
    print(f"  Theta built in {(time.time() - t_theta) / 60:.1f} min, shape matches {N} docs x {T} topics", flush=True)
    print(f"  artifact sha256: {meta['artifact_sha256'][:16]}...", flush=True)

    theta = lda.theta_artifact  # (N, T)
    assert theta.shape == (N, T)

    print("\nLoading citation graph...", flush=True)
    with open(ACL_GRAPH_PATH, encoding='utf-8') as f:
        raw_graph = json.load(f)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    n_edges = 0
    for src, tgts in raw_graph.items():
        si = aid2idx.get(src)
        if si is None:
            continue
        for tgt in tgts:
            ti = aid2idx.get(tgt)
            if ti is not None:
                G.add_edge(si, ti, weight=1.0)
                n_edges += 1
    print(f"  {N:,} nodes, {n_edges:,} edges", flush=True)

    print("\nComputing 'normal' PageRank (uniform bias/transition)...", flush=True)
    t_normal = time.time()
    pr_normal = nx.pagerank(G, alpha=1 - ALPHA, max_iter=200, tol=1e-9)
    print(f"  done in {time.time() - t_normal:.1f}s", flush=True)
    with open(OUT_DIR / "pr_normal.pkl", 'wb') as f:
        pickle.dump(pr_normal, f)

    print("\nBuilding TF-IDF index over corpus (title+abstract) for query neighbors...", flush=True)
    corpus_texts = [f"{acl_dataset[aid].get('title', '')} {acl_dataset[aid].get('abstract', '') or ''}" for aid in acl_ids]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=30_000, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)

    df_annot = pd.read_csv(ANNOTATION_1_CSV)
    pilot_row = df_annot.iloc[0]
    pilot_tid = pilot_row['id']
    pilot_query = pilot_row['query_keywords']
    pilot_year = int(pilot_row['year'])
    print(f"\nPilot query: tutorial={pilot_tid!r} year={pilot_year} query={pilot_query!r}", flush=True)

    qvec = vectorizer.transform([pilot_query])
    sims = cosine_similarity(qvec, tfidf_matrix)[0]

    years_arr = np.array([acl_dataset[aid]['year'] for aid in acl_ids])
    eligible = np.where(years_arr <= pilot_year)[0]
    elig_sims = sims[eligible]
    top20_local = eligible[np.argsort(-elig_sims)[:NEIGHBOR_K]]
    print(f"  top-{NEIGHBOR_K} TF-IDF neighbors (temporally eligible): "
          f"{[acl_ids[i] for i in top20_local[:5]]} ...", flush=True)

    theta_q = theta[top20_local].mean(axis=0)
    theta_q = theta_q / theta_q.sum()
    top_topics = np.argsort(-theta_q)[:TOP_TOPICS_FOR_PILOT]
    print(f"  theta_q mass on top-{TOP_TOPICS_FOR_PILOT} topics: {theta_q[top_topics].sum():.3f} "
          f"(of 1.0 total) -- topics: {top_topics.tolist()}", flush=True)

    print(f"\nComputing bw_modified_bias PageRank for top-{TOP_TOPICS_FOR_PILOT} pilot topics "
          f"(betweenness centrality is the expensive part -- timing this closely)...", flush=True)

    TAU = 0.01
    pr_scores_per_topic = {}
    bw_times = []
    for t in top_topics:
        t = int(t)
        topic_membership = theta[:, t] >= TAU
        topic_nodes = set(np.where(topic_membership)[0].tolist())
        t_bw = time.time()
        if len(topic_nodes) < 2:
            bw = {n: 0.0 for n in range(N)}
        else:
            bw = nx.betweenness_centrality_subset(G, sources=topic_nodes, targets=topic_nodes, normalized=True)
        bw_elapsed = time.time() - t_bw
        bw_times.append(bw_elapsed)
        bw_arr = np.array([bw.get(i, 0.0) for i in range(N)])

        raw_bias = theta[:, t] * bw_arr
        bias_sum = raw_bias.sum()
        bias = raw_bias / bias_sum if bias_sum > 1e-12 else np.full(N, 1.0 / N)
        bias_dict = {i: float(bias[i]) for i in range(N)}

        pr_t = nx.pagerank(G, alpha=1 - ALPHA, max_iter=200, tol=1e-9, personalization=bias_dict, dangling=bias_dict)
        pr_scores_per_topic[t] = pr_t
        print(f"    topic {t}: |topic_nodes|={len(topic_nodes)}, betweenness took {bw_elapsed:.1f}s", flush=True)

    mean_bw_time = np.mean(bw_times)
    print(f"\n  Mean betweenness time per topic: {mean_bw_time:.1f}s "
          f"-> extrapolated full 200-topic cost: {mean_bw_time * 200 / 3600:.1f} hours "
          f"(this graph has {N:,} nodes vs AAN's 15,666)", flush=True)

    score = np.zeros(N)
    mass_covered = 0.0
    for t in top_topics:
        t = int(t)
        pr_t = pr_scores_per_topic[t]
        pr_arr = np.array([pr_t.get(i, 0.0) for i in range(N)])
        score += theta_q[t] * pr_arr
        mass_covered += theta_q[t]
    print(f"\n(Pilot approximation: Score(d,q) computed over only the top-{TOP_TOPICS_FOR_PILOT} "
          f"topics, covering {mass_covered:.3f} of theta_q's probability mass -- "
          f"NOT the exact full-200-topic score, just a plumbing/sanity check.)", flush=True)

    score_masked = np.where(years_arr <= pilot_year, score, -1)
    ranked = np.argsort(-score_masked)[:20]

    print(f"\nTop-10 bw_modified_bias (pilot-approx) recommendations for {pilot_tid!r}:", flush=True)
    for rank, idx in enumerate(ranked[:10], 1):
        title = acl_dataset[acl_ids[idx]].get('title', '')[:70]
        print(f"    {rank:>2}. [{acl_ids[idx]}] {title}", flush=True)

    df_gold = pd.read_csv(READING_LISTS_CSV)
    df_gold['reading_list'] = df_gold['reading_list'].apply(ast.literal_eval)
    gold_row = df_gold[df_gold['id'] == pilot_tid].iloc[0]
    gold_titles = {clean_string(p['title']) for p in gold_row['reading_list'] if p.get('title')}
    print(f"\nGold reading list has {len(gold_titles)} papers.", flush=True)

    pred_titles = [clean_string(acl_dataset[acl_ids[i]].get('title', '')) for i in ranked]
    matches = [t for t in pred_titles if t in gold_titles]
    print(f"Exact title matches in top-20 (pilot-approx ranking): {len(matches)} / {len(gold_titles)}", flush=True)

    recall = len(matches) / len(gold_titles) if gold_titles else 0.0
    print(f"Recall@20 (pilot-approx, single query): {recall * 100:.2f}%", flush=True)

    normal_top10 = sorted(pr_normal.items(), key=lambda x: -x[1])[:10]
    print(f"\nTop-10 by 'normal' PageRank (query-independent, sanity check):", flush=True)
    for rank, (idx, sc) in enumerate(normal_top10, 1):
        title = acl_dataset[acl_ids[idx]].get('title', '')[:70]
        print(f"    {rank:>2}. [{acl_ids[idx]}] {title} (score={sc:.2e})", flush=True)

    print(f"\nTotal pilot time: {(time.time() - t0) / 60:.1f} min", flush=True)
    print("Pilot done.", flush=True)


if __name__ == '__main__':
    main()
