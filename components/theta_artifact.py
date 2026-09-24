"""
Fixed, versioned document-topic-distribution artifact.

Gensim's LdaModel.get_document_topics()/inference() re-runs variational
inference on every call, from a non-reseeded internal state -- confirmed
empirically (see theta_stability_check2.py) to return slightly different
theta_d values (~1e-8 relative) across repeated calls with identical input,
and confirmed to amplify into 0.1-0.6% PageRank score differences after the
100-iteration power-iteration. This module generates theta_d for every
document ONCE, from a single inference pass, and freezes it to disk so every
downstream consumer (bias, transition_star, centrality subgraph membership,
author scores, query construction) reads the SAME numbers every time.

Terminology note (deliberate): this is a fixed, versioned document-topic-
distribution artifact generated from one inference pass -- NOT a claim that
Gensim's inference is "inherently deterministic" or that regenerating it from
scratch would reproduce the same array bitwise. See the artifact's own
metadata field `regeneration_reproducible` for an empirical answer to that
separate question (populated by check_theta_regeneration_stability()).
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import gensim
import gensim.utils as gensim_utils
import numpy as np

from components.lda import GensimLDA, lda_output_path

theta_artifact_dir = lda_output_path.joinpath('theta_artifact')

# Canonical protocol for the artifact that actually ships: gensim's
# inference() draws its initial gamma from model.random_state.gamma(...),
# and that RandomState object advances its internal state on every call --
# so successive inference() calls on a fixed, already-trained model draw
# DIFFERENT initializations even though the model itself was trained with
# random_state=0. Resetting model.random_state to a freshly-seeded
# RandomState immediately before a full, fixed-order (ascending doc_id,
# i.e. ascending paper id) pass over every document makes the resulting
# theta matrix bitwise reproducible on regeneration -- verified empirically
# (see theta_regeneration_diagnosis.py's follow-up check): same seed + same
# order => identical output, every time.
THETA_EXTRACTION_SEED = 0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _artifact_paths(lda_no_topics: int, cluster: int) -> Tuple[Path, Path]:
    theta_artifact_dir.mkdir(parents=True, exist_ok=True)
    base = theta_artifact_dir.joinpath(f'theta_T{lda_no_topics}_cluster{cluster}')
    return base.with_suffix('.npy'), base.with_suffix('.meta.json')


def compute_theta_matrix(lda: GensimLDA, seed: Optional[int] = None) -> np.ndarray:
    """Run gensim's variational inference EXACTLY ONCE per document, in
    ascending doc_id order (= ascending paper id), and return the dense
    (num_docs, num_topics) theta matrix.

    Reproduces get_document_topics()'s own normalization
    (gamma / sum(gamma)) directly from model.inference(), rather than going
    through get_document_topics()'s minimum_probability-filtered sparse
    output and re-densifying it -- this is the same computation gensim
    performs internally (see LdaModel.get_document_topics source), just
    capturing the full dense row instead of a filtered list.

    If ``seed`` is given, lda.model.random_state is reset to a freshly-seeded
    RandomState immediately before the pass (canonical protocol -- makes the
    whole pass, and hence the resulting matrix, bitwise reproducible on
    regeneration with the same seed and doc order). If ``seed`` is None, the
    model's existing random_state is used as-is, advancing across calls as
    gensim normally does -- this is the mode used by
    check_theta_regeneration_stability() / theta_regeneration_diagnosis.py to
    characterize the model's natural per-call instability, deliberately NOT
    the canonical reproducible protocol.
    """
    if seed is not None:
        lda.model.random_state = gensim_utils.get_random_state(seed)
    num_docs = lda.num_docs
    num_topics = lda.lda_no_topics
    theta = np.zeros((num_docs, num_topics), dtype=np.float64)
    for doc_id in range(num_docs):
        bow = lda.bag_of_words[doc_id]
        gamma, _ = lda.model.inference([bow])
        theta[doc_id] = gamma[0] / gamma[0].sum()
    return theta


def _corpus_order_hash(lda: GensimLDA) -> str:
    """Hash of the exact document sequence (content + order) theta was
    extracted over -- detects any accidental reordering or corpus drift on
    a later regeneration attempt."""
    import pickle as _pickle
    return _sha256_bytes(_pickle.dumps(lda.bag_of_words))


def build_theta_artifact(lda: GensimLDA, cluster: int = 0, seed: int = THETA_EXTRACTION_SEED) -> dict:
    """Build and persist the frozen theta artifact for this LDA model/cluster,
    using the canonical deterministic protocol: model.random_state reset to
    a fixed seed immediately before extraction, documents processed in
    ascending doc_id (= ascending paper id) order, single documented pass.

    Returns the metadata dict that was written alongside the .npy file.
    """
    npy_path, meta_path = _artifact_paths(lda.lda_no_topics, cluster)

    corpus_order_hash = _corpus_order_hash(lda)
    theta = compute_theta_matrix(lda, seed=seed)
    np.save(npy_path, theta)

    model_full_path = GensimLDA.full_path(lda.lda_no_topics, cluster)
    model_file_hashes = {}
    for suffix in ('', '.expElogbeta.npy', '.state', '.id2word'):
        p = Path(str(model_full_path['model']) + suffix)
        if p.exists():
            model_file_hashes[p.name] = _sha256_file(p)
    dictionary_hash = _sha256_file(model_full_path['dictionary']) if model_full_path['dictionary'].exists() else None
    docs_hash = _sha256_file(model_full_path['docs']) if model_full_path['docs'].exists() else None

    metadata = {
        'description': (
            'Fixed, versioned document-topic inference artifact. This freezes '
            'theta_d for reproducibility of all downstream PageRank/query '
            'computations. Produced by a canonical deterministic protocol: '
            'model.random_state reset to `theta_extraction_seed` immediately '
            'before extraction, documents processed in ascending doc_id '
            '(= ascending paper id) order, single documented pass -- verified '
            'empirically to make regeneration bitwise-reproducible under this '
            'exact protocol (same seed, same order, same model). This is NOT '
            'a claim that Gensim variational inference is inherently '
            'deterministic in general (ad hoc repeated calls without resetting '
            'random_state are NOT reproducible -- that non-determinism is what '
            'motivated freezing this artifact in the first place); it is a '
            'claim about this specific, documented extraction protocol only.'
        ),
        'lda_no_topics': lda.lda_no_topics,
        'cluster': cluster,
        'num_docs': lda.num_docs,
        'num_topics': lda.lda_no_topics,
        'alpha': lda.alpha,
        'training_random_state': 0,
        'theta_extraction_seed': seed,
        'doc_processing_order': 'ascending doc_id (= ascending paper.id - 1)',
        'corpus_order_hash': corpus_order_hash,
        'gensim_version': gensim.__version__,
        'python_version': sys.version,
        'model_class': type(lda.model).__name__,
        'model_file_sha256': model_file_hashes,
        'dictionary_sha256': dictionary_hash,
        'docs_sha256': docs_hash,
        'artifact_sha256': _sha256_file(npy_path),
        'git_commit_hash': _git_commit_hash(),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'regeneration_reproducible': None,  # filled in by check_theta_regeneration_stability()
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def load_theta_artifact(lda_no_topics: int, cluster: int = 0) -> Tuple[np.ndarray, dict]:
    npy_path, meta_path = _artifact_paths(lda_no_topics, cluster)
    if not npy_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No frozen theta artifact for lda_no_topics={lda_no_topics}, cluster={cluster} "
            f"at {npy_path}. Build it first: "
            f"python build_theta_artifact.py --lda-no-topics {lda_no_topics}"
        )
    theta = np.load(npy_path)
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    if theta.shape[1] != lda_no_topics:
        raise ValueError(
            f"Theta artifact at {npy_path} has {theta.shape[1]} topic columns, "
            f"expected {lda_no_topics}"
        )
    return theta, metadata


def check_theta_regeneration_stability(lda: GensimLDA, cluster: int = 0,
                                       seed: Optional[int] = THETA_EXTRACTION_SEED) -> dict:
    """Build the theta matrix TWICE independently (two fresh inference passes
    over the same loaded model) and compare. With ``seed`` set (the default),
    this checks regeneration under the CANONICAL protocol actually used by
    build_theta_artifact() -- i.e. whether the shipped artifact is genuinely
    regeneratable. Pass ``seed=None`` instead to characterize the model's
    natural, un-reseeded per-call instability (what
    theta_regeneration_diagnosis.py uses). Does NOT overwrite the persisted
    artifact.
    """
    theta_a = compute_theta_matrix(lda, seed=seed)
    theta_b = compute_theta_matrix(lda, seed=seed)
    diff = np.abs(theta_a - theta_b)
    identical = np.array_equal(theta_a, theta_b)
    result = {
        'identical_bitwise': bool(identical),
        'max_abs_diff': float(diff.max()),
        'mean_abs_diff': float(diff.mean()),
        'sha256_a': _sha256_bytes(theta_a.tobytes()),
        'sha256_b': _sha256_bytes(theta_b.tobytes()),
    }
    return result
