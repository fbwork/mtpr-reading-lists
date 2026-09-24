"""
Build the frozen, versioned theta_d artifact (components/theta_artifact.py)
for a given LDA topic count, and empirically check whether regenerating it
from scratch reproduces the same numbers.

This does NOT touch any PageRank cache. It only freezes P(t|d) for every
document into components/output/lda/theta_artifact/theta_T{n}_cluster{c}.npy,
which components/lda.py's GensimLDA.get_doc_topics() now reads from instead
of calling gensim's get_document_topics()/inference() at runtime.

Usage:
    python build_theta_artifact.py --lda-no-topics 200
"""
import argparse
import json

from components.database import setup as db_setup
db_setup()

from components.lda import setup as lda_setup
from components.theta_artifact import build_theta_artifact, check_theta_regeneration_stability

parser = argparse.ArgumentParser()
parser.add_argument('--lda-no-topics', type=int, default=200)
args = parser.parse_args()

print(f"Loading LDA (T={args.lda_no_topics})...", flush=True)
lda_results = lda_setup(args.lda_no_topics, 0, fresh=False)
if not lda_results:
    raise RuntimeError(f"No {args.lda_no_topics}-topic LDA model available.")

for cluster, lda in lda_results.items():
    print(f"\n[cluster {cluster}] Building theta artifact ({lda.num_docs} docs x {lda.lda_no_topics} topics)...",
          flush=True)
    metadata = build_theta_artifact(lda, cluster=cluster)
    print(f"[cluster {cluster}] Artifact SHA-256: {metadata['artifact_sha256']}", flush=True)

    print(f"[cluster {cluster}] Checking regeneration stability (two independent inference passes, "
          f"NOT saved, comparison only)...", flush=True)
    stability = check_theta_regeneration_stability(lda, cluster=cluster)
    print(f"[cluster {cluster}] identical_bitwise={stability['identical_bitwise']} "
          f"max_abs_diff={stability['max_abs_diff']:.3e} mean_abs_diff={stability['mean_abs_diff']:.3e}",
          flush=True)

    # Record the empirical answer in the artifact's own metadata -- this is
    # the honest basis for whatever wording the paper uses, not an assumption.
    from components.theta_artifact import _artifact_paths
    _npy_path, meta_path = _artifact_paths(lda.lda_no_topics, cluster)
    metadata['regeneration_reproducible'] = stability
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

print("\nDone.")
