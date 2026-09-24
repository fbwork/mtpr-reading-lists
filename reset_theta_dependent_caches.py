"""
Delete every on-disk cache whose values depend on P(t|d), for a given
lda_no_topics -- so that rebuilding the 19 MTPR PageRank modes actually
recomputes from the frozen theta artifact instead of silently reusing values
computed under the old, non-deterministic get_document_topics() path.

IMPORTANT: none of these cache paths are namespaced by lda_no_topics (see
components/lda.py's get_cache_path_and_value -- cache_root is a bare
directory, not cache_root/str(lda_no_topics)), so this wipes ALL cached
topic_docs/centrality/author_scores/topic_authors entries regardless of which
topic count produced them. That is safe here because the project only has
one live topic count (200) in active use, but it means this script should
NOT be run casually if multiple topic counts are ever in concurrent use.

Does NOT touch: the LDA model files, the frozen theta artifact itself, the
citation graph, or the golden standard.

Only completion-flag files are left in place deliberately -- the caches this
script deletes are gated purely by file existence (see
get_cache_path_and_value: `if cache_path.exists() and flag.is_module_done(...)`),
so deleting the file alone is sufficient to force recomputation; stale flags
are harmless since the corresponding file no longer exists.

Usage:
    python reset_theta_dependent_caches.py --no-topics 200 --yes
"""
import argparse
import shutil
from pathlib import Path

from components.lda import lda_output_path
from components.prmodes import PRModes

parser = argparse.ArgumentParser()
parser.add_argument('--no-topics', type=int, default=200)
parser.add_argument('--yes', action='store_true', help='Actually delete (default: dry run).')
args = parser.parse_args()

pr_output_root = Path('components/output/pr') / str(args.no_topics) / str(args.no_topics)

targets = [
    lda_output_path / 'topic_docs',
    lda_output_path / 'centrality',
    lda_output_path / 'author_scores',
    lda_output_path / 'topic_authors',
    lda_output_path / 'author_scores.pkl',
    lda_output_path / 'authors_score.pkl',
]

pr_file_targets = []
for mode in PRModes.get_mtpr_modes():
    pr_file_targets.append(pr_output_root / f'pageranks_{mode.value}.bin')
    pr_file_targets.append(pr_output_root / f'pageranks_{mode.value}.meta.json')
    pr_file_targets.append(pr_output_root / 'values' / mode.value)

print(f"{'DRY RUN -- ' if not args.yes else ''}Targets for lda_no_topics={args.no_topics}:\n")
for t in targets + pr_file_targets:
    exists = t.exists()
    kind = 'dir' if t.is_dir() else ('file' if t.exists() else '???')
    print(f"  [{'EXISTS' if exists else 'absent'}] {kind:<5} {t}")

if not args.yes:
    print("\nDry run only -- pass --yes to actually delete.")
else:
    print("\nDeleting...")
    for t in targets + pr_file_targets:
        if t.is_dir():
            shutil.rmtree(t)
        elif t.exists():
            t.unlink()
    print("Done.")
