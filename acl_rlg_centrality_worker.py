"""
Standalone worker: computes all four topic-subgraph centrality measures
(degree, closeness, betweenness, Katz) for a range of topics on the
ACL-rlg citation graph, using the already-built frozen Theta artifact
(cluster=99, T=200). Writes one pickle per (measure, topic) to a shared
cache directory so multiple copies of this script -- run in separate
terminals over disjoint topic ranges -- can work in parallel without
recomputing each other's topics or touching the AAN pipeline's caches.

Betweenness is the bottleneck (pilot measurement: ~12 min/topic average
on this 97,440-node graph); degree/closeness/Katz are comparatively
cheap and computed alongside it per topic so the same worker range
serves both bw_modified_bias and hybrid without a second pass.

Usage (run in as many separate terminals as you like, splitting 0..199
into disjoint ranges -- order/overlap doesn't matter, each topic's
result is cached to its own file and skipped if already present):

    .venv\\Scripts\\python.exe acl_rlg_centrality_worker.py --start 0   --end 50
    .venv\\Scripts\\python.exe acl_rlg_centrality_worker.py --start 50  --end 100
    .venv\\Scripts\\python.exe acl_rlg_centrality_worker.py --start 100 --end 150
    .venv\\Scripts\\python.exe acl_rlg_centrality_worker.py --start 150 --end 200
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
CENTRALITY_DIR = OUT_DIR / "centrality"
CLUSTER = 99
T = 200
TAU = 0.01


def build_graph():
    with open(ACL_DATASET_PATH, encoding='utf-8') as f:
        acl_dataset = json.load(f)
    acl_ids = sorted(acl_dataset.keys())
    N = len(acl_ids)
    aid2idx = {aid: i for i, aid in enumerate(acl_ids)}

    with open(ACL_GRAPH_PATH, encoding='utf-8') as f:
        raw_graph = json.load(f)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for src, tgts in raw_graph.items():
        si = aid2idx.get(src)
        if si is None:
            continue
        for tgt in tgts:
            ti = aid2idx.get(tgt)
            if ti is not None:
                G.add_edge(si, ti, weight=1.0)
    return G, N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    args = parser.parse_args()

    CENTRALITY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Worker for topics [{args.start}, {args.end})", flush=True)
    from components.theta_artifact import load_theta_artifact
    theta, meta = load_theta_artifact(T, cluster=CLUSTER)
    print(f"  theta loaded: {theta.shape}, artifact sha256={meta['artifact_sha256'][:16]}...", flush=True)

    print("  building citation graph...", flush=True)
    t_g = time.time()
    G, N = build_graph()
    print(f"  graph: {N:,} nodes, {G.number_of_edges():,} edges (built in {time.time()-t_g:.1f}s)", flush=True)

    for t in range(args.start, args.end):
        paths = {m: CENTRALITY_DIR / f"{m}_{t}.pkl" for m in ('degree', 'closeness', 'betweenness', 'katz')}
        if all(p.exists() for p in paths.values()):
            print(f"  topic {t}: already cached, skipping", flush=True)
            continue

        topic_nodes_list = [i for i in range(N) if theta[i, t] >= TAU]
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

    print(f"Worker for [{args.start}, {args.end}) done.", flush=True)


if __name__ == '__main__':
    main()
