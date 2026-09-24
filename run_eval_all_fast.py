"""
Fast evaluation of ALL 30 PageRank modes in a single Python session.
- Loads LDA/BERTopic ONCE, then evaluates each mode quickly using cached files.
- Prints a comparison table to stdout and saves LaTeX + CSV to components/resources/gs/

KEY: Golden standard uses ACL anthology paper codes (e.g. 'H05-1042').
     PageRank uses 0-based integer doc_ids (paper.id - 1).
     We convert golden codes → doc_ids using Paper.code2id() via the database.
"""
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
NO_TOPICS = 5
LDA_NO_TOPICS = 200
BERTOPIC_NO_TOPICS = 0
K = 10   # Precision/Recall/MAP @ K
AGIFY = True

# Single source of truth: components/prmodes.py::PRModes.get_evaluation_modes().
# Toggle PRModes.INCLUDE_EXPERIMENTAL_MODES there to add GAT/Flow Matching back in.
from components.prmodes import PRModes
ALL_MODES = [m.value for m in PRModes.get_evaluation_modes()]

PR_OUTPUT = Path('components/output/pr') / str(NO_TOPICS) / str(NO_TOPICS)
GS_PATH   = Path('components/resources/gs/golden_standard.json')
GS_OUT    = Path('components/resources/gs')
GS_OUT.mkdir(parents=True, exist_ok=True)

# ── Database setup (needed for code → doc_id mapping) ───────────────────────
print("Connecting to database...")
from components.database import setup as db_setup
from models.tables import Paper
db_setup()

# Build code → 0-based doc_id mapping
# Paper.id is 1-based; PageRank uses (paper.id - 1) as doc_id
print("Building ACL code -> doc_id mapping...")
code_to_docid = {p.code: p.id - 1 for p in Paper.select(Paper.id, Paper.code).iterator()}
print(f"  {len(code_to_docid)} papers mapped")

# ── Golden Standard ──────────────────────────────────────────────────────────
print("Loading golden standard...")
with GS_PATH.open() as f:
    golden_standard_raw = json.load(f)
print(f"  {len(golden_standard_raw)} queries loaded")

# Convert codes to doc_ids
golden_standard = {}
for query, codes in golden_standard_raw.items():
    doc_ids = []
    missing = []
    for code in codes:
        if code in code_to_docid:
            doc_ids.append(code_to_docid[code])
        else:
            missing.append(code)
    if missing:
        print(f"  [WARN] Query '{query}': {len(missing)} codes not in DB: {missing[:3]}")
    golden_standard[query] = doc_ids

# Summary
total_gs = sum(len(v) for v in golden_standard.values())
print(f"  {total_gs} relevant doc_ids across {len(golden_standard)} queries")


def evaluate_recommendations(ranked_ids, relevant_set, k=10):
    top_k = ranked_ids[:k]
    hits = set(top_k) & set(relevant_set)
    precision = len(hits) / k if k > 0 else 0
    recall    = len(hits) / len(relevant_set) if relevant_set else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    ap = 0
    for i, pid in enumerate(top_k, 1):
        if pid in relevant_set:
            ap += len(set(top_k[:i]) & set(relevant_set)) / i
    map_score = ap / len(hits) if hits else 0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'map': map_score}


def load_pagerank(mode):
    """Load the saved pageranks_<mode>.bin and apply aging if requested."""
    path = PR_OUTPUT / f'pageranks_{mode}.bin'
    if not path.exists():
        return None
    with path.open('rb') as f:
        results = pickle.load(f)
    
    if AGIFY:
        # Apply aging: divide by max(1, 2011 - year)
        try:
            from models.tables import Paper
            from components.database import setup as database_setup
            database_setup()
            agified = {}
            for paper in Paper.select(Paper.id, Paper.year):
                doc_id = paper.id - 1
                if doc_id in results:
                    year_diff = max(1, 2011 - paper.year)
                    agified[doc_id] = {
                        t: v / year_diff for t, v in results[doc_id].items()
                    }
            return agified
        except Exception as e:
            print(f"  [WARN] aging failed ({e}), using raw scores")
            return results
    return results


def rank_from_pagerank(pr_results):
    """Rank all papers by sum of topic scores."""
    paper_scores = {
        doc_id: sum(topic_scores.values())
        for doc_id, topic_scores in pr_results.items()
    }
    return sorted(paper_scores, key=lambda d: paper_scores[d], reverse=True)


# ── Main Evaluation Loop ─────────────────────────────────────────────────────
all_metrics = {}   # mode -> {precision, recall, f1, map}

print(f"\nEvaluating {len(ALL_MODES)} modes against {len(golden_standard)} queries (K={K})...\n")
print(f"{'Mode':<28} {'P@10':>7} {'R@10':>7} {'F1':>7} {'MAP':>7} {'Status'}")
print("-" * 70)

for mode in ALL_MODES:
    pr = load_pagerank(mode)
    if pr is None:
        print(f"  {mode:<26} {'—':>7} {'—':>7} {'—':>7} {'—':>7}  [MISSING FILE]")
        continue
    
    ranked = rank_from_pagerank(pr)
    
    query_metrics = []
    for query_text, relevant_ids in golden_standard.items():
        m = evaluate_recommendations(ranked, set(relevant_ids), k=K)
        query_metrics.append(m)
    
    if not query_metrics:
        print(f"  {mode:<26}  no golden standard queries")
        continue
    
    avg = {k: sum(m[k] for m in query_metrics) / len(query_metrics)
           for k in ['precision', 'recall', 'f1', 'map']}
    all_metrics[mode] = avg
    
    print(f"  {mode:<26} {avg['precision']:>7.4f} {avg['recall']:>7.4f} "
          f"{avg['f1']:>7.4f} {avg['map']:>7.4f}")

# ── Save Results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)

# JSON
json_out = GS_OUT / 'golden_results.json'
with json_out.open('w') as f:
    json.dump(all_metrics, f, indent=2)
print(f"JSON saved: {json_out}")

# LaTeX table
latex_lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Evaluation of PageRank Modes (Precision@10, Recall@10, F1, MAP)}",
    r"\label{tab:pagerank_eval}",
    r"\begin{tabular}{|l|c|c|c|c|}",
    r"\hline",
    r"Method & P@10 & R@10 & F1 & MAP \\ \hline",
]
for mode, m in all_metrics.items():
    mode_escaped = mode.replace('_', r'\_')
    latex_lines.append(
        f"{mode_escaped} & {m['precision']:.4f} & "
        f"{m['recall']:.4f} & {m['f1']:.4f} & {m['map']:.4f} \\\\"
    )
latex_lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
latex_str = "\n".join(latex_lines)

tex_out = GS_OUT / 'comparison_table_golden.tex'
with tex_out.open('w') as f:
    f.write(latex_str)
print(f"LaTeX saved: {tex_out}")

# CSV
csv_out = GS_OUT / 'evaluation_summary.csv'
with csv_out.open('w') as f:
    f.write("mode,precision,recall,f1,map\n")
    for mode, m in all_metrics.items():
        f.write(f"{mode},{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},{m['map']:.6f}\n")
print(f"CSV saved: {csv_out}")

# Best mode
if all_metrics:
    best = max(all_metrics, key=lambda m: all_metrics[m]['map'])
    print(f"\nBest mode by MAP: {best}  (MAP={all_metrics[best]['map']:.4f})")
    
    best_f1 = max(all_metrics, key=lambda m: all_metrics[m]['f1'])
    print(f"Best mode by F1 : {best_f1}  (F1={all_metrics[best_f1]['f1']:.4f})")

print("\nDone.")