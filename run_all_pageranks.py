# --- File: run_all_pageranks.py ---
import os
import sys
import json
import pickle
import signal
import traceback
import argparse
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime
from collections import defaultdict

import numpy as np
import psutil
import matplotlib.pyplot as plt
from tqdm import tqdm

from components.lda import setup as lda_setup
from components.pagerank import setup as pagerank_setup
from components.prmodes import PRModes
from components.topic_model_modes import TopicModelModes

# Define paths
gs_resource_path = Path(__file__).parent.joinpath('components/resources/gs')
output_path = Path("output/experiments")
log_path = output_path.joinpath("logs")
checkpoint_path = output_path.joinpath("checkpoints")
log_path.mkdir(parents=True, exist_ok=True)
checkpoint_path.mkdir(parents=True, exist_ok=True)


class CheckpointManager:
    """Manages checkpoints for resumable experiments"""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.completed_modes: Set[str] = set()
        self.load_checkpoint()

    def load_checkpoint(self):
        """Load completed modes from checkpoint file"""
        checkpoint_file = self.checkpoint_path.joinpath("completed_modes.json")
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    self.completed_modes = set(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load checkpoint: {e}")
                self.completed_modes = set()

    def save_checkpoint(self, mode_key: str):
        """Save completed mode to checkpoint file"""
        self.completed_modes.add(mode_key)
        checkpoint_file = self.checkpoint_path.joinpath("completed_modes.json")
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(list(self.completed_modes), f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save checkpoint: {e}")

    def is_completed(self, mode_key: str) -> bool:
        """Check if a mode has been completed"""
        return mode_key in self.completed_modes

    def get_remaining_modes(self, all_modes: List[str]) -> List[str]:
        """Get list of modes that haven't been completed"""
        return [m for m in all_modes if not self.is_completed(m)]


class ResourceMonitor:
    """Monitors system resource usage during experiments"""

    def __init__(self):
        self.start_time = None
        self.cpu_percentages = []
        self.memory_percentages = []
        self.disk_io_start = None
        self.disk_io_end = None

    def start(self):
        """Start monitoring"""
        self.start_time = datetime.now()
        self.cpu_percentages = []
        self.memory_percentages = []
        self.disk_io_start = psutil.disk_io_counters()

    def update(self):
        """Update resource statistics"""
        self.cpu_percentages.append(psutil.cpu_percent(interval=0.1))
        self.memory_percentages.append(psutil.virtual_memory().percent)

    def stop(self):
        """Stop monitoring and return statistics"""
        self.disk_io_end = psutil.disk_io_counters()
        duration = (datetime.now() - self.start_time).total_seconds()

        return {
            'duration': duration,
            'cpu_mean': np.mean(self.cpu_percentages) if self.cpu_percentages else 0,
            'cpu_std': np.std(self.cpu_percentages) if self.cpu_percentages else 0,
            'memory_mean': np.mean(self.memory_percentages) if self.memory_percentages else 0,
            'memory_std': np.std(self.memory_percentages) if self.memory_percentages else 0,
            'disk_read_mb': (self.disk_io_end.read_bytes - self.disk_io_start.read_bytes) / 1024 / 1024,
            'disk_write_mb': (self.disk_io_end.write_bytes - self.disk_io_start.write_bytes) / 1024 / 1024
        }


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n\n⚠️  Received interrupt signal. Saving checkpoint...")
    print("Progress has been saved. You can resume by running the script again.")
    sys.exit(0)


def load_golden_standard() -> Dict[str, List[str]]:
    """Load golden standard reading lists: {query_name: [ACL paper code, ...]}.

    NOTE: keys are query names (e.g. "domain adaptation"), not LDA topic
    indices -- this is the same golden_standard.json used by csc_evaluation.py
    / csc_with_baselines.py. The per-LDA-topic-index NDCG loop below can never
    match against these string keys, so it always reports 0.0; it is left in
    place only so the run doesn't crash on it, not as a meaningful metric.
    Use csc_evaluation.py or csc_with_baselines.py after this script for the
    real query-aware CSC/MAP/F evaluation.
    """
    golden_path = gs_resource_path.joinpath('golden_standard.json')
    if not golden_path.exists():
        print(f"⚠️  Warning: Golden standard file not found at {golden_path}")
        return {}

    with open(golden_path, 'r') as f:
        return json.load(f)


def calculate_ndcg(ranked_list: List[int], golden_list: List[int], k: int = 10) -> float:
    """
    Calculate NDCG@k for a ranked list compared to golden standard.

    Args:
        ranked_list: List of paper IDs in ranked order
        golden_list: Ground truth relevant paper IDs
        k: Cutoff position

    Returns:
        NDCG@k score
    """
    if not golden_list:
        return 0.0

    # Create relevance vector
    relevance = [1 if pid in golden_list else 0 for pid in ranked_list[:k]]

    # Calculate DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))

    # Calculate ideal DCG
    ideal_relevance = [1] * min(k, len(golden_list))
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def plot_results(results: Dict, resource_stats: Dict):
    """Plot comprehensive comparison of results"""
    if not results or not resource_stats:
        print("⚠️  Warning: No data to plot")
        return

    # Create figure with 2x3 subplots for better organization
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. NDCG Comparison (Top-left, spanning 2 rows)
    ax1 = fig.add_subplot(gs[0:2, 0])
    modes = list(results.keys())
    ndcg_means = [results[m]['ndcg_mean'] for m in modes]
    ndcg_stds = [results[m]['ndcg_std'] for m in modes]

    sorted_indices = np.argsort(ndcg_means)[::-1]
    modes_sorted = [modes[i] for i in sorted_indices]
    ndcg_means_sorted = [ndcg_means[i] for i in sorted_indices]
    ndcg_stds_sorted = [ndcg_stds[i] for i in sorted_indices]

    colors = plt.cm.viridis(np.linspace(0, 1, len(modes_sorted)))
    bars = ax1.barh(range(len(modes_sorted)), ndcg_means_sorted, xerr=ndcg_stds_sorted,
                    capsize=3, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(modes_sorted)))
    ax1.set_yticklabels(modes_sorted, fontsize=8)
    ax1.set_xlabel('NDCG@10', fontsize=12)
    ax1.set_title('NDCG@10 Comparison (All Modes)', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax1.axvline(x=np.mean(ndcg_means_sorted), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(ndcg_means_sorted):.4f}')
    ax1.legend()

    # 2. Top-10 Best Performers (Top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    top_10_modes = modes_sorted[:10]
    top_10_ndcg = ndcg_means_sorted[:10]

    ax2.bar(range(10), top_10_ndcg, color='green', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(10))
    ax2.set_xticklabels([m.split('_')[0][:10] for m in top_10_modes],
                        rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('NDCG@10', fontsize=10)
    ax2.set_title('Top-10 Best Performers', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 3. Execution Time (Middle-right)
    ax3 = fig.add_subplot(gs[1, 1])
    resource_modes = list(resource_stats.keys())
    durations = [resource_stats[m]['duration'] / 60 for m in resource_modes]  # Convert to minutes

    sorted_time_indices = np.argsort(durations)
    resource_modes_sorted = [resource_modes[i] for i in sorted_time_indices]
    durations_sorted = [durations[i] for i in sorted_time_indices]

    ax3.barh(range(len(resource_modes_sorted)), durations_sorted, color='orange',
             edgecolor='black', linewidth=0.5)
    ax3.set_yticks(range(len(resource_modes_sorted)))
    ax3.set_yticklabels([m.split('_')[0][:15] for m in resource_modes_sorted], fontsize=7)
    ax3.set_xlabel('Time (minutes)', fontsize=10)
    ax3.set_title('Execution Time per Mode', fontsize=12, fontweight='bold')
    ax3.grid(True, axis='x', linestyle='--', alpha=0.5)

    # 4. CPU Usage (Bottom-left)
    ax4 = fig.add_subplot(gs[2, 0])
    cpu_means = [resource_stats[m]['cpu_mean'] for m in resource_modes]

    ax4.bar(range(len(resource_modes)), cpu_means, color='skyblue',
            edgecolor='black', linewidth=0.5)
    ax4.set_xticks(range(len(resource_modes)))
    ax4.set_xticklabels([m.split('_')[0][:8] for m in resource_modes],
                        rotation=90, ha='right', fontsize=7)
    ax4.set_ylabel('CPU Usage (%)', fontsize=10)
    ax4.set_title('Average CPU Usage', fontsize=12, fontweight='bold')
    ax4.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax4.axhline(y=np.mean(cpu_means), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(cpu_means):.1f}%')
    ax4.legend()

    # 5. Memory Usage (Bottom-right)
    ax5 = fig.add_subplot(gs[2, 1])
    memory_means = [resource_stats[m]['memory_mean'] for m in resource_modes]

    ax5.bar(range(len(resource_modes)), memory_means, color='salmon',
            edgecolor='black', linewidth=0.5)
    ax5.set_xticks(range(len(resource_modes)))
    ax5.set_xticklabels([m.split('_')[0][:8] for m in resource_modes],
                        rotation=90, ha='right', fontsize=7)
    ax5.set_ylabel('Memory Usage (%)', fontsize=10)
    ax5.set_title('Average Memory Usage', fontsize=12, fontweight='bold')
    ax5.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax5.axhline(y=np.mean(memory_means), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(memory_means):.1f}%')
    ax5.legend()

    # Add overall title
    fig.suptitle('Comprehensive PageRank Variants Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    # Save plots
    plot_file = output_path.joinpath('comparison_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Plots saved to {plot_file}")


def run_all_pageranks(no_topics: int, fresh: bool = False,
                      modes_subset: List[PRModes] = None):
    """
    Run comprehensive PageRank comparison.

    Args:
        no_topics: Number of topics for LDA
        fresh: Force recomputation
        modes_subset: Subset of modes to run (None = all modes)
    """
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n" + "=" * 80)
    print("🚀 Starting Comprehensive PageRank Comparison")
    print("=" * 80 + "\n")

    # Step 1: Run LDA
    print("📊 Step 1/4: Running LDA to generate topics...")
    try:
        lda_results = lda_setup(no_topics, bertopic_no_topics=0, verbose=False, fresh=fresh)
        if not lda_results:
            raise ValueError("LDA setup failed to return results")
        print(f"✅ LDA completed! Generated {len(lda_results)} clusters\n")
    except Exception as e:
        print(f"❌ Error in LDA setup: {str(e)}")
        traceback.print_exc()
        return

    # Step 2: Load golden standard
    print("📚 Step 2/4: Loading golden standard...")
    try:
        golden_standard = load_golden_standard()
        print(f"✅ Loaded golden standard: {len(golden_standard)} queries")
        print("   (NDCG below is per-LDA-topic-index and will read 0.0 for all "
              "modes -- it can't match query-name keys. Run csc_evaluation.py "
              "or csc_with_baselines.py afterward for real query-aware metrics.)\n")
    except Exception as e:
        print(f"❌ Error loading golden standard: {str(e)}")
        traceback.print_exc()
        return

    # Step 3: Define modes to run
    print("🎯 Step 3/4: Defining PageRank modes...")
    if modes_subset is None:
        # Defaults to the 19 MTPR paper variants; flip
        # PRModes.INCLUDE_EXPERIMENTAL_MODES (components/prmodes.py) or pass
        # --modes explicitly to include GAT/Flow Matching.
        pr_modes = PRModes.get_evaluation_modes()
    else:
        pr_modes = modes_subset

    print(f"Will run {len(pr_modes)} modes × 2 (agify/raw) = {len(pr_modes) * 2} experiments")
    print(f"Modes: {[m.value for m in pr_modes[:5]]}... (showing first 5)\n")

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_path)

    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path.joinpath(f"run_log_{timestamp}.txt")

    # Load existing results
    results = {}
    resource_stats = {}
    results_file = output_path.joinpath('pagerank_comparison.bin')

    if results_file.exists() and not fresh:
        try:
            with open(results_file, 'rb') as f:
                data = pickle.load(f)
                results = data.get('results', {})
                resource_stats = data.get('resource_stats', {})
            print(f"📂 Loaded {len(results)} existing results from cache\n")
        except Exception as e:
            print(f"⚠️  Could not load existing results: {e}")
            results = {}
            resource_stats = {}

    # Step 4: Run experiments
    print("🔬 Step 4/4: Running experiments...")
    print("=" * 80 + "\n")

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Experiment started at: {datetime.now()}\n")
        f.write(f"Parameters: no_topics={no_topics}, fresh={fresh}\n")
        f.write(f"Total modes to run: {len(pr_modes)}\n")
        f.write("=" * 80 + "\n\n")

        # Generate all mode keys
        all_mode_keys = []
        for mode in pr_modes:
            for agify in [True, False]:
                all_mode_keys.append(f"{mode.value}_{'agify' if agify else 'raw'}")

        # Get remaining modes
        remaining_modes = checkpoint_manager.get_remaining_modes(all_mode_keys)
        f.write(f"✅ Completed: {len(checkpoint_manager.completed_modes)} modes\n")
        f.write(f"⏳ Remaining: {len(remaining_modes)} modes\n\n")

        # Run each mode
        total_experiments = len(remaining_modes)
        completed_count = 0

        for agify in [True, False]:
            agify_str = "with agify" if agify else "without agify"

            for i, mode in enumerate(pr_modes):
                mode_key = f"{mode.value}_{'agify' if agify else 'raw'}"

                # Skip if already completed (unless --fresh forces recomputation)
                if not fresh and checkpoint_manager.is_completed(mode_key):
                    continue

                completed_count += 1
                print(f"\n[{completed_count}/{total_experiments}] Running: {mode_key}")
                f.write(f"\n{'=' * 80}\n")
                f.write(f"[{completed_count}/{total_experiments}] Mode: {mode_key}\n")
                f.write(f"{'=' * 80}\n")

                monitor = ResourceMonitor()

                try:
                    # Start monitoring
                    monitor.start()

                    # Setup PageRank
                    f.write(f"⏳ Setting up PageRank...\n")
                    pr = pagerank_setup(
                        log_msg=f"run_all_pageranks: mode={mode.value}, no_topics={no_topics}",
                        lda_results=lda_results,
                        topic_model=TopicModelModes.LDA,
                        no_topics=no_topics,
                        pr_mode=mode,
                        fresh=fresh
                    )

                    if pr is None:
                        raise ValueError(f"PageRank setup returned None")

                    # Update resources periodically
                    monitor.update()

                    # Get results
                    f.write(f"⏳ Computing results...\n")
                    rankings = pr.results(agify=agify)

                    if not rankings:
                        raise ValueError(f"No rankings returned")

                    monitor.update()

                    # Calculate NDCG for each topic
                    f.write(f"⏳ Calculating NDCG scores...\n")
                    topic_ndcgs = {}

                    for topic_id in range(no_topics):
                        if topic_id not in golden_standard:
                            continue

                        # Get ranked list for this topic
                        paper_scores = []
                        for paper_id, scores in rankings.items():
                            score = scores.get(topic_id, 0)
                            paper_scores.append((paper_id, score))

                        paper_scores.sort(key=lambda x: x[1], reverse=True)
                        ranked_papers = [p[0] for p in paper_scores]

                        # Calculate NDCG
                        ndcg = calculate_ndcg(ranked_papers, golden_standard[topic_id], k=10)
                        topic_ndcgs[topic_id] = ndcg

                    # Stop monitoring
                    stats = monitor.stop()
                    resource_stats[mode_key] = stats

                    # Store results
                    if topic_ndcgs:
                        results[mode_key] = {
                            'ndcg_mean': np.mean(list(topic_ndcgs.values())),
                            'ndcg_std': np.std(list(topic_ndcgs.values())),
                            'topic_ndcgs': topic_ndcgs,
                            'num_topics_evaluated': len(topic_ndcgs)
                        }
                    else:
                        results[mode_key] = {
                            'ndcg_mean': 0.0,
                            'ndcg_std': 0.0,
                            'topic_ndcgs': {},
                            'num_topics_evaluated': 0
                        }

                    # Save checkpoint
                    checkpoint_manager.save_checkpoint(mode_key)

                    # Log results
                    f.write(f"\n✅ Completed {mode_key}!\n")
                    f.write(f"   NDCG@10: {results[mode_key]['ndcg_mean']:.4f} ± {results[mode_key]['ndcg_std']:.4f}\n")
                    f.write(f"   Topics evaluated: {results[mode_key]['num_topics_evaluated']}/{no_topics}\n")
                    f.write(f"   Duration: {stats['duration']:.2f} seconds ({stats['duration'] / 60:.2f} min)\n")
                    f.write(f"   CPU: {stats['cpu_mean']:.1f}% ± {stats['cpu_std']:.1f}%\n")
                    f.write(f"   Memory: {stats['memory_mean']:.1f}% ± {stats['memory_std']:.1f}%\n")
                    f.write(f"   Disk Read: {stats['disk_read_mb']:.1f} MB\n")
                    f.write(f"   Disk Write: {stats['disk_write_mb']:.1f} MB\n")

                    print(f"   ✅ NDCG@10: {results[mode_key]['ndcg_mean']:.4f}")
                    print(f"   ⏱️  Time: {stats['duration'] / 60:.2f} min")

                    # Save intermediate results
                    with open(results_file, 'wb') as f2:
                        pickle.dump({
                            'results': results,
                            'resource_stats': resource_stats,
                            'timestamp': datetime.now().isoformat()
                        }, f2)

                except Exception as e:
                    f.write(f"\n❌ Error in {mode_key}: {str(e)}\n")
                    f.write("Traceback:\n")
                    traceback.print_exc(file=f)
                    f.write("\n⏭️  Continuing with next mode...\n")

                    print(f"   ❌ Error: {str(e)}")
                    print(f"   ⏭️  Skipping to next mode...")
                    continue

        # Final summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("📊 FINAL RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        if results:
            sorted_results = sorted(results.items(),
                                    key=lambda x: x[1]['ndcg_mean'],
                                    reverse=True)

            f.write(f"{'Rank':<6} {'Mode':<35} {'Agify':<10} {'NDCG@10':<12} {'Std':<12} {'Topics':<10}\n")
            f.write("-" * 95 + "\n")

            for rank, (mode_key, result) in enumerate(sorted_results, 1):
                mode_parts = mode_key.rsplit('_', 1)
                mode_name = mode_parts[0]
                agify_status = mode_parts[1]

                f.write(f"{rank:<6} {mode_name:<35} {agify_status:<10} "
                        f"{result['ndcg_mean']:<12.4f} {result['ndcg_std']:<12.4f} "
                        f"{result['num_topics_evaluated']:<10}\n")

            # Statistics
            f.write("\n" + "=" * 80 + "\n")
            f.write("📈 STATISTICS\n")
            f.write("=" * 80 + "\n")

            all_ndcg = [r['ndcg_mean'] for r in results.values()]
            f.write(f"Best NDCG@10: {max(all_ndcg):.4f} ({sorted_results[0][0]})\n")
            f.write(f"Worst NDCG@10: {min(all_ndcg):.4f}\n")
            f.write(f"Mean NDCG@10: {np.mean(all_ndcg):.4f} ± {np.std(all_ndcg):.4f}\n")
            f.write(f"Median NDCG@10: {np.median(all_ndcg):.4f}\n")

            all_durations = [r['duration'] for r in resource_stats.values()]
            f.write(f"\nTotal execution time: {sum(all_durations) / 3600:.2f} hours\n")
            f.write(f"Average time per mode: {np.mean(all_durations) / 60:.2f} minutes\n")

    # Plot results
    print("\n📊 Generating visualization...")
    try:
        plot_results(results, resource_stats)
    except Exception as e:
        print(f"❌ Error plotting results: {str(e)}")
        traceback.print_exc()

    # Final output
    print("\n" + "=" * 80)
    print("✅ EXPERIMENT COMPLETED!")
    print("=" * 80)
    print(f"📁 Results saved to: {results_file}")
    print(f"📝 Log file: {log_file}")
    print(f"📊 Plots: {output_path.joinpath('comparison_plots.png')}")
    print(f"💾 Checkpoints: {checkpoint_path}")
    print("=" * 80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive PageRank comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all modes
  python run_all_pageranks.py --no-topics 50

  # Run specific modes
  python run_all_pageranks.py --no-topics 50 --modes normal,themed,hybrid

  # Force fresh computation
  python run_all_pageranks.py --no-topics 50 --fresh
        """
    )

    parser.add_argument(
        "--no-topics",
        type=int,
        required=True,
        help="Number of topics for LDA"
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force fresh computation (ignore cache)"
    )

    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Comma-separated list of modes to run (e.g., 'normal,hybrid,gat')"
    )

    args = parser.parse_args()

    # Parse modes if specified
    modes_subset = None
    if args.modes:
        try:
            mode_names = [m.strip().upper() for m in args.modes.split(',')]
            modes_subset = [PRModes[name] for name in mode_names]
        except KeyError as e:
            print(f"❌ Invalid mode name: {e}")
            print(f"Available modes: {[m.value for m in PRModes.get_all_modes()]}")
            sys.exit(1)

    run_all_pageranks(args.no_topics, args.fresh, modes_subset)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)