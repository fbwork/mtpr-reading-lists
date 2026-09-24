"""
Cheap diagnostic (per explicit instructions) to decide whether multi-pass
averaging of theta_d would be worth adopting, BEFORE committing to it:

  - select 50 highest-instability documents (by natural, un-reseeded
    per-call inference variance) and 50 random documents;
  - run inference 5 times per document (natural stochasticity: model's
    random_state is NOT reset between calls, same as gensim's normal
    behavior / the original bug);
  - compare the single-pass theta (first of the 5) with the 5-pass mean:
    dominant-topic flips, mean/max distribution differences, and changes in
    topic-subgraph membership under tau=0.01 (TOPIC_MEMBERSHIP_MIN_PROBABILITY).

Do NOT adopt multi-pass averaging unless this shows a clear benefit worth
the ~5x extraction cost. Averaging reduces initialization sensitivity but
does not by itself prove a more correct posterior estimate -- gensim's
variational inference, given random initialization and the current
iteration budget, converges to different local solutions for a subset of
documents; that is the precise, conservative description, not "the true
posterior is multimodal".
"""
import random

import numpy as np

from components.database import setup as db_setup
db_setup()

from components.lda import setup as lda_setup, TOPIC_MEMBERSHIP_MIN_PROBABILITY
from components.theta_artifact import compute_theta_matrix

N_PASSES = 5
TAU = TOPIC_MEMBERSHIP_MIN_PROBABILITY

print("Loading LDA (T=200)...", flush=True)
lda = list(lda_setup(200, 0, fresh=False).values())[0]
num_docs = lda.num_docs

print("Ranking documents by natural (un-reseeded) instability across 2 quick passes...", flush=True)
theta_a = compute_theta_matrix(lda, seed=None)
theta_b = compute_theta_matrix(lda, seed=None)
per_doc_max = np.abs(theta_a - theta_b).max(axis=1)
top50 = list(np.argsort(-per_doc_max)[:50])

random.seed(0)
remaining = [d for d in range(num_docs) if d not in set(top50)]
random50 = random.sample(remaining, 50)

sample_docs = top50 + random50
labels = ['high_instability'] * 50 + ['random'] * 50

print(f"\nRunning {N_PASSES} un-reseeded inference passes for {len(sample_docs)} sample docs...", flush=True)
passes = []  # list of {doc_id: theta_row}
for p in range(N_PASSES):
    pass_result = {}
    for doc_id in sample_docs:
        gamma, _ = lda.model.inference([lda.bag_of_words[doc_id]])
        pass_result[doc_id] = gamma[0] / gamma[0].sum()
    passes.append(pass_result)

print(f"\n{'doc_id':>8} {'label':<18} {'single_top':>11} {'mean_top':>9} {'flip':>5} "
      f"{'mean_L1_diff':>13} {'max_L1_diff':>12} {'tau_membership_changed':>23}")

flips = 0
mean_l1_diffs = []
membership_changes = 0
for doc_id, label in zip(sample_docs, labels):
    rows = np.stack([passes[p][doc_id] for p in range(N_PASSES)])
    single = rows[0]
    mean = rows.mean(axis=0)

    single_top = int(np.argmax(single))
    mean_top = int(np.argmax(mean))
    flip = single_top != mean_top
    flips += int(flip)

    l1_diff = np.abs(single - mean).sum()
    mean_l1_diffs.append(l1_diff)

    single_membership = set(np.where(single >= TAU)[0].tolist())
    mean_membership = set(np.where(mean >= TAU)[0].tolist())
    membership_changed = single_membership != mean_membership
    membership_changes += int(membership_changed)

    print(f"{doc_id:>8} {label:<18} {single_top:>11} {mean_top:>9} {str(flip):>5} "
          f"{l1_diff:>13.4f} {np.abs(single-mean).max():>12.4f} {str(membership_changed):>23}")

print(f"\nSummary over {len(sample_docs)} docs ({N_PASSES} passes each):")
print(f"  dominant-topic flips (single vs 5-pass mean): {flips} / {len(sample_docs)}")
print(f"  tau={TAU} membership changed:                  {membership_changes} / {len(sample_docs)}")
print(f"  mean L1(single, mean) across sample:          {np.mean(mean_l1_diffs):.4f}")
print(f"  max  L1(single, mean) across sample:           {np.max(mean_l1_diffs):.4f}")

# Same breakdown split by group, since high-instability docs are the
# adversarial case and random docs are the "typical corpus" case.
for label in ['high_instability', 'random']:
    idxs = [i for i, l in enumerate(labels) if l == label]
    sub_flips = sum(1 for i in idxs if
                    int(np.argmax(np.stack([passes[p][sample_docs[i]] for p in range(N_PASSES)])[0])) !=
                    int(np.argmax(np.stack([passes[p][sample_docs[i]] for p in range(N_PASSES)]).mean(axis=0))))
    print(f"  [{label}] flips: {sub_flips} / {len(idxs)}")
