"""
Follow-up to build_theta_artifact.py's regeneration-stability check: that
check found max_abs_diff=0.9355 between two independent inference passes --
far larger than the ~1e-8-relative noise characterized earlier for query-side
and training-doc-side inference. This script identifies WHICH documents
drive that gap and whether it correlates with bag-of-words sparsity, so the
finding can be reported precisely rather than as a single scary number.
"""
import numpy as np

from components.database import setup as db_setup
db_setup()

from components.lda import setup as lda_setup
from components.theta_artifact import compute_theta_matrix

print("Loading LDA (T=200)...", flush=True)
lda = list(lda_setup(200, 0, fresh=False).values())[0]

print("Computing theta matrix, pass A...", flush=True)
theta_a = compute_theta_matrix(lda)
print("Computing theta matrix, pass B...", flush=True)
theta_b = compute_theta_matrix(lda)

diff = np.abs(theta_a - theta_b)
per_doc_max = diff.max(axis=1)
bow_lens = np.array([len(bow) for bow in lda.bag_of_words])

order = np.argsort(-per_doc_max)
print(f"\n{'doc_id':>8} {'max_diff':>10} {'bow_len':>8} {'top_topic_A':>12} {'top_topic_B':>12}")
for i in order[:30]:
    top_a = int(np.argmax(theta_a[i]))
    top_b = int(np.argmax(theta_b[i]))
    print(f"{i:>8} {per_doc_max[i]:>10.4f} {bow_lens[i]:>8} {top_a:>12} {top_b:>12}")

for thresh in [0.5, 0.1, 0.05, 0.01, 0.001]:
    n = int((per_doc_max > thresh).sum())
    print(f"docs with max_diff > {thresh}: {n} / {len(per_doc_max)} ({100*n/len(per_doc_max):.2f}%)")

print(f"\nCorrelation(max_diff, bow_len): {np.corrcoef(per_doc_max, bow_lens)[0,1]:.3f}")
print(f"mean bow_len overall: {bow_lens.mean():.2f}")
print(f"mean bow_len among docs with max_diff > 0.05: {bow_lens[per_doc_max > 0.05].mean() if (per_doc_max > 0.05).any() else float('nan'):.2f}")

top_10_diff_flips = sum(1 for i in order[:len(order)] if int(np.argmax(theta_a[i])) != int(np.argmax(theta_b[i])))
print(f"\nDocs where the ARGMAX topic itself flips between passes: {top_10_diff_flips} / {len(per_doc_max)}")
