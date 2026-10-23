"""
Corrects the four centrality *_transition variants on AAN.

Finding: components/pagerank.py::PaperRankClassic.bias() has no branch for D/CL/BW/KZ_MODIFIED_TRANSITION, so
these four variants were run with a UNIFORM bias (the `else` branch), whereas Table 1 of the paper defines
their bias as P(t|d) (only MODIFIED_TRANSITION was handled). This script recomputes them with bias = P(t|d), with
(a) the production stopping rule (L1 change < N*1e-6, <= 100 iterations) and (b) a tight rule (N*1e-10, <= 3000),
and reproduces the OLD (uniform-bias) numbers first as a validation of the solver and of the stopping-rule emulation.

Writes pageranks_<mode>.bin in components/output/pr/200/200_corrected (production rule) and 200_corrected_tight,
plus per-query scores in components/resources/gs/csc_per_query_corrected_{prod,tight}.csv
"""
import json, os, pickle, sys
os.environ.setdefault('LDA_NO_TOPICS', '200')
src = open('aan_extra_variants.py', encoding='utf-8').read()
exec(compile(src[:src.index("results = {}")], 'extra_head', 'exec'))

OUTP = Path('components/output/pr') / str(T)
DIRS = {'prod': OUTP / f'{T}_corrected', 'tight': OUTP / f'{T}_corrected_tight'}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


def solve_rule(M, bias, rule):
    eps, mx = (1e-6, 100) if rule == 'prod' else (1e-10, 3000)
    x = bias.copy()
    for it in range(1, mx + 1):
        xn = (1 - ALPHA) * (M @ x) + (1 - ALPHA) * x[dangling].sum() * bias + ALPHA * bias
        s = xn.sum()
        if s > 0:
            xn = xn / s
        diff = float(np.abs(xn - x).sum())
        x = xn
        if diff < N * eps:
            return x, it, True
    return x, mx, False



def cent_norm(kind, t):
    fn = {'closeness': lda.topic_closeness_centrality_scores_batch,
          'betweenness': lda.topic_betweenness_centrality_scores_batch,
          'katz': lda.topic_katz_centrality_scores_batch}[kind]
    v = np.zeros(N)
    for k, x in fn(G, t).items():
        v[k] = x
    sm = v.sum()
    return v / sm if sm > 0 else v


def cent_vec(kind, t):
    if kind == 'degree':
        out, _ = degree_vectors(t, 0.01)
        return out / out.sum() if out.sum() > 0 else out
    return cent_norm(kind, t)


MODES = {'d_modified_transition': 'degree', 'cl_modified_transition': 'closeness',
         'bw_modified_transition': 'betweenness', 'kz_modified_transition': 'katz'}
yd = np.array([max(1, 2011 - years[d]) for d in nodes], dtype=float)


def per_query(agified):
    rows = []
    for q, g in golden.items():
        f, r, o = csc(rank_pagerank_for_query(q, agified, lda_results)[:K], g)
        rows.append((q, f, r, o))
    return rows


summary = {}
per_query_rows = {'prod': {}, 'tight': {}}
for mode, kind in MODES.items():
    for label, uniform_bias in (('OLD uniform bias', True), ('CORRECTED theta bias', False)):
        for rule in ('prod', 'tight'):
            X = np.zeros((N, T)); nonconv = 0; its = []
            for t in range(T):
                th = theta[:, t]
                b = np.full(N, 1.0 / N) if uniform_bias else norm_bias(th)
                M = per_target_matrix(tstar(t) * cent_vec(kind, t)[src_e])
                x, it, ok = solve_rule(M, b, rule)
                X[:, t] = x; nonconv += (not ok); its.append(it)
            agified = {d: {t: X[d, t] / yd[d] for t in range(T)} for d in nodes}
            rows = per_query(agified)
            o = float(np.mean([r[3] for r in rows]))
            f_ = float(np.mean([r[1] for r in rows])); r_ = float(np.mean([r[2] for r in rows]))
            print(f'{mode:<24} {label:<22} {rule:<5} OCSC {o:.4f}  F {f_:.3f} R {r_:.3f}  nonconv {nonconv:>3}  median it {int(np.median(its))}', flush=True)
            summary[f'{mode}|{label}|{rule}'] = {'ocsc': o, 'fcsc': f_, 'rcsc': r_, 'nonconv': nonconv, 'median_it': int(np.median(its))}
            if not uniform_bias:
                per_query_rows[rule][mode] = rows
                pickle.dump(agified if False else {d: {t: float(X[d, t]) for t in range(T)} for d in nodes},
                            (DIRS[rule] / f'pageranks_{mode}.bin').open('wb'))

for rule in ('prod', 'tight'):
    with (GS_OUT / f'csc_per_query_corrected_{rule}.csv').open('w', encoding='utf-8') as f:
        f.write('mode,query,fcsc,rcsc,ocsc\n')
        for mode, rows in per_query_rows[rule].items():
            for q, fc, rc, oc in rows:
                f.write(f'{mode},{q},{fc:.6f},{rc:.6f},{oc:.6f}\n')
json.dump(summary, open(GS_OUT / 'aan_corrected_transition_summary.json', 'w'), indent=2)
print('saved')
