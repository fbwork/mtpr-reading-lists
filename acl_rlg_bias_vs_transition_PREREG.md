# Pre-registration: bias vs. transition placement on ACL-rlg

Written 2026-10-23, BEFORE any of the new ACL-rlg runs described below were executed.
Nothing in this file may be edited after the first full run; deviations go in a
"Deviations" section appended at the end, with the date.

## Question
On AAN (8 queries) the configuration that places an extra centrality signal in the
*bias* with a uniform transition (`*_modified_bias`) scored higher than the one that
places it in the *transition* (`*_modified_transition`) in all five families, but the
differences are not significant after Holm-Bonferroni. Does the same ordering hold on
ACL-rlg (73 tutorials with a non-empty gold list), which is independent and larger?

## Families (4; the author family is excluded because ACL-rlg has no author field)
degree (`d_`), closeness (`cl_`), betweenness (`bw_`), Katz (`kz_`).
For family f: bias_f = `f_modified_bias`; transition_f = `f_modified_transition`.

## Fixed settings (identical to the paper's ACL-rlg evaluation; none tuned on ACL-rlg)
T=200, alpha=0.15, tau=0.01, frozen Theta of the ACL-rlg corpus, per-tutorial-year
truncated citation graph and per-year centrality caches, neighbour retrieval K_q=20 by
TF-IDF over title+abstract restricted to papers up to the tutorial year, ranking
restricted to papers up to the tutorial year, metrics at 20.

## Unit of analysis
One tutorial = one paired observation. For each tutorial the metric is averaged over the
three annotator query sets first. n = 73 tutorials. Annotators are NOT treated as
independent observations.

## Metrics and error control
- NDCG@20: primary. MRR@20: secondary confirmatory. Recall@20: descriptive only (effect
  sizes reported, no confirmatory claim).
- Test: two-sided Wilcoxon signed-rank on the 73 paired differences
  D_i = M_i,bias - M_i,transition (zero differences dropped, as in scipy).
- Holm-Bonferroni across the 4 families, applied separately within NDCG@20 and within
  MRR@20 (each its own family of 4 tests, alpha = 0.05). As a conservative sensitivity
  analysis, Holm over all 8 tests is also reported.
- For every family and metric: mean and median of D_i, share of tutorials with D_i>0 /
  <0 / =0, and a 95% percentile bootstrap CI for the mean difference (10,000 paired
  resamples of tutorials, seed 0).

## Convergence protocol (decided before running)
- Every PageRank solve uses the production update (bias, dangling-mass redistribution
  through the bias vector, per-iteration renormalization) with stopping rule
  L1 change < 1e-8 (vectors sum to 1) or 3000 iterations.
- For every (mode, year, topic) the iteration count, a convergence flag, the final L1
  change and the final ratio of consecutive changes (an estimate of the contraction
  rate) are stored.
- Non-converged solves are NOT dropped and NOT reported as converged. Primary analysis
  uses all topics (non-converged ones at their final iterate). Sensitivity analysis: the
  share of each tutorial's query-topic weight (theta_q) that falls on non-converged
  topics is reported, and the analysis is repeated on tutorials whose weight on
  non-converged topics is below the median.
- Transition modes use the production semantics: if the weights into a target sum to 0
  the transition falls back to uniform over its citers. (The legacy ACL-rlg `themed` run
  used zero rows instead; `themed` is therefore also rerun with the production
  semantics, and the old run reproduced as a validation of the new code.)

## Validation of the new code before using any new mode
The new solver must reproduce, to the second decimal, the stored temporal ACL-rlg means
of `personalized`, `themed` (legacy semantics) and `bw_modified_bias`
(`acl_rlg_pilot_out/final_results_temporal.json`). If it does not, the new modes are not
used until the discrepancy is explained.

## Decision rule (what the result will be called)
Let "supports" mean the bias configuration has the higher mean NDCG@20 in a family.
- All 4 families support AND Holm-adjusted p < 0.05 for NDCG@20 in all 4: report as an
  independent confirmation of the bias-over-transition ordering on ACL-rlg.
- All 4 support in direction but not all Holm-significant: report as a consistent but not
  confirmed descriptive pattern, now with n=73 (state the number of significant families).
- Any family reversed, or mixed directions: report that the ordering is not confirmed on
  ACL-rlg, and restrict the paper's claim to "on AAN, descriptively". The Abstract and
  Contributions are then rewritten accordingly. This outcome is reported as found.
- If non-converged solves affect the conclusion (sensitivity analysis changes the
  direction or the Holm decision), the claim is limited to "the operational algorithms
  at a fixed iteration cap" and the paper says so.
