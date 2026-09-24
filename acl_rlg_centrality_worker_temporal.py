"""
Temporal-leakage fix for the ACL-rlg external evaluation: computes
per-topic centrality (degree, closeness, betweenness, Katz) on
citation graphs truncated to each of the tutorial years actually
present in ACL-rlg, instead of the single full (2024) graph.

Only 7 distinct tutorial years exist: 2016, 2019, 2020, 2021, 2022,
2023, 2024. The corpus itself tops out at 2024, so the 2024 cutoff IS
the existing full graph -- already computed, reused as-is, no new work.
This script computes the other 6 (smaller, therefore cheaper) cutoffs.

For a fixed topic range, a worker loops over all 6 cutoff years
(building each truncated graph once, reusing it across that range's
topics) so a single terminal can be given one topic range and left to
work through every year -- 4 terminals total, not 4x6=24.

theta (P(t|d)) is NOT recomputed per cutoff: it comes from each paper's
own fixed title/abstract text, which does not change over time, so
reusing the single frozen Theta artifact across all cutoffs is not a
temporal-leakage vector. Only the citation-graph-derived centrality is
cutoff-sensitive, which is exactly what this script fixes. (LDA
training itself used the full corpus's vocabulary; we do not retrain
per cutoff -- a smaller, documented, separate scope decision.)

Usage (same start/end split as acl_rlg_centrality_worker.py, but each
invocation now works through all 6 non-2024 cutoffs for that range):

    .venv\\Scripts\\python.exe acl_rlg_centrality_worker_temporal.py --start 0   --end 50
    .venv\\Scripts\\python.exe acl_rlg_centrality_worker_temporal.py --start 50  --end 100
    .venv\\Scripts\\python.exe acl_rlg_centrality_worker_temporal.py --start 100 --end 150
    .venv\\Scripts\\python.exe acl_rlg_centrality_worker_temporal.py --start 150 --end 200
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx

sys.path.insert(0, r"E:\0000000-Thesis\1405\thesis-1405\thesis-1405")

EVAL_DATA = Path(r"E:\0000000-Thesis\1405\thesis-1405\acl-rlg-eval\data")
RLG_BASE = Path(r"E:\0000000-Thesis\1403-new approach\ACL-rlg\aclanthology-tutorial-reading-lists")
ACL_DATASET_PATH = RLG_BASE / "experiments/classic_methods/acl_anthology_dataset/acl_anthology_dataset.json"
ACL_GRAPH_PATH = EVAL_DATA / "acl_full_citation_graph.json"

OUT_DIR = Path(r"E:\0000000-Thesis\1405\thesis-1405\thesis-1405\acl_rlg_pilot_out")
CLUSTER = 99
T = 200
TAU = 0.01

# 2024 == full corpus == already computed in acl_rlg_pilot_out/centrality/, skipped here.
CUTOFF_YEARS = [2016, 2019, 2020, 2021, 2022, 2023]


def load_corpus_and_years():
    with open(ACL_DATASET_PATH, encoding='utf-8') as f:
        acl_dataset = json.load(f)
    acl_ids = sorted(acl_dataset.keys())
    N = len(acl_ids)
    aid2idx = {aid: i for i, aid in enumerate(acl_ids)}
    years = np.array([acl_dataset[aid]['year'] for aid in acl_ids])
    return acl_ids, aid2idx, years, N


def build_truncated_graph(aid2idx, years, cutoff, N_full):
    """Citation graph restricted to papers with year <= cutoff. Node ids
    are kept as the ORIGINAL full-corpus indices (0..N_full-1) so they
    align directly with the single shared Theta artifact -- nodes for
    papers published after cutoff simply have no edges and are never
    topic-eligible members practically, but keeping indices aligned
    avoids a second id-remapping layer end to end."""
    with open(ACL_GRAPH_PATH, encoding='utf-8') as f:
        raw_graph = json.load(f)

    eligible_mask = years <= cutoff
    G = nx.DiGraph()
    G.add_nodes_from(np.where(eligible_mask)[0].tolist())
    for src, tgts in raw_graph.items():
        si = aid2idx.get(src)
        if si is None or not eligible_mask[si]:
            continue
        for tgt in tgts:
            ti = aid2idx.get(tgt)
            if ti is not None and eligible_mask[ti]:
                G.add_edge(si, ti, weight=1.0)
    return G


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    args = parser.parse_args()

    print(f"Worker for topics [{args.start}, {args.end}), all {len(CUTOFF_YEARS)} temporal cutoffs", flush=True)
    from components.theta_artifact import load_theta_artifact
    theta, meta = load_theta_artifact(T, cluster=CLUSTER)
    print(f"  theta loaded: {theta.shape}", flush=True)

    acl_ids, aid2idx, years, N_full = load_corpus_and_years()

    for cutoff in CUTOFF_YEARS:
        cutoff_dir = OUT_DIR / f"centrality_y{cutoff}"
        cutoff_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== cutoff year {cutoff} ===", flush=True)
        t_g = time.time()
        G = build_truncated_graph(aid2idx, years, cutoff, N_full)
        n_eligible = sum(1 for y in years if y <= cutoff)
        print(f"  {n_eligible:,} eligible papers, {G.number_of_edges():,} edges "
              f"(built in {time.time()-t_g:.1f}s)", flush=True)

        for t in range(args.start, args.end):
            paths = {m: cutoff_dir / f"{m}_{t}.pkl" for m in ('degree', 'closeness', 'betweenness', 'katz')}
            if all(p.exists() for p in paths.values()):
                print(f"  topic {t}: already cached, skipping", flush=True)
                continue

            # topic membership restricted to papers eligible under this cutoff
            topic_nodes_list = [i for i in range(N_full) if years[i] <= cutoff and theta[i, t] >= TAU]
            topic_nodes = set(topic_nodes_list)
            t0 = time.time()

            if not paths['degree'].exists():
                degree = {d: sum(1 for tgt in G.successors(d) if tgt in topic_nodes) for d in topic_nodes}
                with open(paths['degree'], 'wb') as f:
                    pickle.dump(degree, f)

            if not paths['closeness'].exists():
                if topic_nodes_list:
                    sub = G.subgraph(topic_nodes_list)
                    closeness = nx.closeness_centrality(sub, distance=None, wf_improved=True)
                else:
                    closeness = {}
                with open(paths['closeness'], 'wb') as f:
                    pickle.dump(closeness, f)

            if not paths['katz'].exists():
                if topic_nodes_list:
                    sub = G.subgraph(topic_nodes_list)
                    try:
                        katz = nx.katz_centrality(sub, alpha=0.1, beta=1.0, max_iter=1000)
                    except Exception:
                        katz = {n: 0.0 for n in topic_nodes_list}
                else:
                    katz = {}
                with open(paths['katz'], 'wb') as f:
                    pickle.dump(katz, f)

            t_bw = time.time()
            if not paths['betweenness'].exists():
                if len(topic_nodes) >= 2:
                    betweenness = nx.betweenness_centrality_subset(G, sources=topic_nodes, targets=topic_nodes, normalized=True)
                else:
                    betweenness = {n: 0.0 for n in topic_nodes}
                with open(paths['betweenness'], 'wb') as f:
                    pickle.dump(betweenness, f)
            bw_elapsed = time.time() - t_bw

            print(f"  topic {t}: |nodes|={len(topic_nodes)}, total={time.time()-t0:.1f}s "
                  f"(betweenness={bw_elapsed:.1f}s)", flush=True)

    print(f"\nWorker for [{args.start}, {args.end}) done, all cutoffs.", flush=True)


if __name__ == '__main__':
    main()
