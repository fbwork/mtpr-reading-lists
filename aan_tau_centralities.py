"""
Sensitivity of the headline AAN result to the topic-membership threshold tau (default 0.01) for the
graph-based centralities. For a given tau it recomputes, with the same networkx calls as components/lda.py,
the closeness (topic-induced subgraph), betweenness (subset betweenness, sources = targets = topic nodes, full
graph) and Katz (topic subgraph, alpha=0.1, beta=1, max_iter=1000) centralities for all 200 topics, then runs the
*_modified_bias variants (centrality x P(t|d) bias, uniform transition) and scores them with CSC (primary setting).

Usage: python aan_tau_centralities.py --tau 0.02
Output: components/resources/gs/aan_tau_<tau>.json ; centrality cache in components/output/tau_cache/<tau>/
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path
os.environ.setdefault('LDA_NO_TOPICS', '200')
ap = argparse.ArgumentParser(); ap.add_argument('--tau', type=float, required=True)
ap.add_argument('--skip-betweenness', action='store_true')
cli = ap.parse_args()
src = open('aan_extra_variants.py', encoding='utf-8').read()
exec(compile(src[:src.index("results = {}")], 'extra_head', 'exec'))
TAU = cli.tau
cache = Path('components/output/tau_cache') / str(TAU)
cache.mkdir(parents=True, exist_ok=True)


def centrality(kind, t):
    p = cache / f'{kind}_{t}.pkl'
    if p.exists():
        return pickle.load(p.open('rb'))
    nodes_t = [int(d) for d in np.where(theta[:, t] >= TAU)[0]]
    if not nodes_t:
        r = {}
    elif kind == 'closeness':
        r = nx.closeness_centrality(G.subgraph(nodes_t), distance=None, wf_improved=True)
    elif kind == 'betweenness':
        r = nx.betweenness_centrality_subset(G, sources=nodes_t, targets=nodes_t, normalized=True)
    elif kind == 'katz':
        try:
            r = nx.katz_centrality(G.subgraph(nodes_t), alpha=0.1, beta=1.0, max_iter=1000)
        except Exception:
            r = {n: 0.0 for n in nodes_t}
    pickle.dump(r, p.open('wb'))
    return r


def vec(kind, t):
    d = centrality(kind, t)
    v = np.zeros(N)
    for k, x in d.items():
        v[k] = x
    s = v.sum()
    return v / s if s > 0 else v


out = {}
kinds = ['closeness', 'katz'] + ([] if cli.skip_betweenness else ['betweenness'])
for kind in kinds:
    t0 = time.time()
    for t in range(T):
        centrality(kind, t)
        if t % 20 == 0:
            print(f'tau {TAU} {kind} topic {t}  {time.time()-t0:.0f}s', flush=True)
    key = {'closeness': 'cl', 'katz': 'kz', 'betweenness': 'bw'}[kind] + f'_modified_bias tau={TAU}'
    out[key] = run_variant(key, lambda t, kind=kind: theta[:, t] * vec(kind, t))
json.dump(out, open(f'components/resources/gs/aan_tau_{TAU}.json', 'w'), indent=2)
print('saved')
