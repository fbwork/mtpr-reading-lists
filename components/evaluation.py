import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import flag

from .bertopic import evaluate_flow_matching_paths
from .database import setup as database_setup, get_relevant_papers
from .lda import setup as lda_setup
from .pagerank import setup as pagerank_setup
from .prmodes import PRModes
from .query import QueryManager, rank_pagerank_for_query
from .topic_model_modes import TopicModelModes

gs_resource_path = Path(__file__).parent.joinpath('resources', 'gs')
eval_log_path = Path(os.getenv('LOG_PATH', default='logs')).joinpath('eval.log')


def setup_logger() -> logging.Logger:
    """
    Initialize logger with file output.

    :return: Configured logger
    """
    try:
        eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=eval_log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger()
    except OSError as e:
        logging.error(f"Failed to set up logger at {eval_log_path}: {e}")
        raise


class Evaluation:
    def __init__(self, results_by_mode: Dict, output_path: Path):
        """
        Initialize Evaluation with results and output path.

        :param results_by_mode: Dictionary of results by mode
        :param output_path: Path for saving evaluation outputs
        """
        self.results_by_mode = results_by_mode
        self.pagerank_output_path = output_path

    def evaluate(self, query: str, recommended_papers: List[int], relevant_papers: Set[int], k: int = 10) -> Dict:
        """
        Evaluate recommendation performance for a query.

        :param query: Search query
        :param recommended_papers: List of recommended paper IDs
        :param relevant_papers: Set of relevant paper IDs
        :param k: Number of top papers to evaluate
        :return: Dictionary of evaluation metrics
        """
        return self.evaluate_recommendations(recommended_papers, relevant_papers, k)

    def evaluate_paths(self, paths: List[int], query: str, k: int = 10) -> Dict:
        """
        Evaluate Flow Matching paths for a query.

        :param paths: List of paper IDs in the predicted paths
        :param query: Search query
        :param k: Number of top papers to evaluate
        :return: Dictionary of evaluation metrics
        """
        logger = setup_logger()
        try:
            metrics = evaluate_flow_matching_paths(paths, get_relevant_papers(query), k)
            output_file = self.pagerank_output_path.joinpath('flow_matching_metrics.json')
            with output_file.open('w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Flow Matching metrics saved to {output_file}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating paths for query '{query}': {e}")
            return {'precision': 0, 'recall': 0, 'f1': 0, 'map': 0}

    def visualize_comparison(self):
        """
        Visualize comparison of metrics across modes.
        """
        logger = setup_logger()
        try:
            if not self.results_by_mode:
                logger.warning("No results available for visualization.")
                return

            plt.figure(figsize=(14, 10))
            metrics = ['precision', 'recall', 'f_score', 'avg_precision']
            modes = list(self.results_by_mode.keys())

            data = []
            for mode in modes:
                for result in self.results_by_mode.get(mode, []):
                    for metric in metrics:
                        if metric in result:
                            data.append({
                                'Mode': mode,
                                'Metric': metric.replace('_', ' ').title(),
                                'Value': result[metric],
                                'Cluster': result.get('cluster', 'N/A')
                            })

            if not data:
                logger.warning("No valid data for visualization.")
                return

            df = pd.DataFrame(data)
            summary_data = df.groupby(['Mode', 'Metric'])['Value'].agg(['mean', 'std']).reset_index()

            fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
            axes = axes.flatten()

            for idx, metric in enumerate(metrics):
                metric_data = summary_data[summary_data['Metric'] == metric.replace('_', ' ').title()]
                if metric_data.empty:
                    logger.warning(f"No data for metric {metric}.")
                    continue
                sns.barplot(data=metric_data, x='Mode', y='mean', ax=axes[idx], palette='Set2')
                axes[idx].errorbar(x=metric_data['Mode'], y=metric_data['mean'], yerr=metric_data['std'],
                                   fmt='none', c='black', capsize=5)
                axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
                axes[idx].set_xlabel('Mode')
                axes[idx].set_ylabel('Value')
                axes[idx].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            output_path = gs_resource_path.joinpath('comparison_plot_detailed.png')
            gs_resource_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)
            plt.close()
            logger.info(f"Detailed comparison plot saved to {output_path}")

            chart_data = {
                'labels': modes,
                'datasets': [
                    {
                        'label': metric.replace('_', ' ').title(),
                        'data': summary_data[summary_data['Metric'] == metric.replace('_', ' ').title()][
                            'mean'].tolist(),
                        'backgroundColor': f'rgba({idx * 50 + 50}, {255 - idx * 50}, 100, 0.5)',
                        'borderColor': f'rgba({idx * 50 + 50}, {255 - idx * 50}, 100, 1)',
                        'borderWidth': 1
                    } for idx, metric in enumerate(metrics)
                    if not summary_data[summary_data['Metric'] == metric.replace('_', ' ').title()].empty
                ]
            }

            output_json = gs_resource_path.joinpath('comparison_chart.json')
            with output_json.open('w') as f:
                json.dump({
                    'type': 'bar',
                    'data': chart_data,
                    'options': {
                        'scales': {
                            'y': {'beginAtZero': True, 'title': {'display': True, 'text': 'Metric Value'}},
                            'x': {'title': {'display': True, 'text': 'Mode'}}
                        },
                        'plugins': {
                            'legend': {'display': True, 'position': 'top'},
                            'title': {'display': True, 'text': 'Comparison of PageRank and Flow Matching Modes'}
                        }
                    }
                }, f, indent=2)
            logger.info(f"Chart JSON saved to {output_json}")
        except Exception as e:
            logger.error(f"Error in visualize_comparison: {e}")
            raise

    @staticmethod
    def evaluate_recommendations(recommended_papers: List[int], relevant_papers: Set[int], k: int = 10) -> Dict:
        """
        Evaluate recommendation performance.

        :param recommended_papers: List of recommended paper IDs
        :param relevant_papers: Set of relevant paper IDs
        :param k: Number of top papers to evaluate
        :return: Dictionary of evaluation metrics
        """
        relevant_set = set(recommended_papers[:k]).intersection(relevant_papers)
        precision = len(relevant_set) / k if k > 0 else 0
        recall = len(relevant_set) / len(relevant_papers) if relevant_papers else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        ap = 0
        for i, paper in enumerate(recommended_papers[:k], 1):
            if paper in relevant_papers:
                ap += len(set(recommended_papers[:i]).intersection(relevant_papers)) / i
        map_score = ap / len(relevant_set) if relevant_set else 0
        return {'precision': precision, 'recall': recall, 'f1': f1, 'map': map_score}

    @staticmethod
    def plot_comparison(metrics: Dict, methods: List[str], metric_name: str = 'precision') -> None:
        """
        Plot comparison of a specific metric across methods.

        :param metrics: Dictionary of metrics by method
        :param methods: List of method names
        :param metric_name: Name of the metric to plot
        """
        logger = setup_logger()
        try:
            values = [metrics[m][metric_name] for m in methods if metric_name in metrics[m]]
            if not values:
                logger.warning(f"No valid data for metric {metric_name}.")
                return

            plt.figure(figsize=(10, 6))
            sns.barplot(x=methods[:len(values)], y=values, palette='Set2')
            plt.ylabel(f'{metric_name.capitalize()} (%)')
            plt.title(f'Comparison of Recommendation Methods ({metric_name.capitalize()})')
            plt.xticks(rotation=45)
            plt.grid(True)
            output_path = gs_resource_path.joinpath(f'{metric_name}_comparison.png')
            gs_resource_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)
            plt.close()
            logger.info(f"Comparison plot for {metric_name} saved to {output_path}")
        except Exception as e:
            logger.error(f"Error in plot_comparison for metric {metric_name}: {e}")
            raise

    @staticmethod
    def generate_latex_table(metrics: Dict, methods: List[str]) -> str:
        """
        Generate a LaTeX table comparing recommendation methods.

        :param metrics: Dictionary of metrics by method
        :param methods: List of method names
        :return: LaTeX table string
        """
        logger = setup_logger()
        try:
            latex = "\\begin{table}[h]\n\\centering\n\\caption{Comparison of Recommendation Methods}\n"
            latex += "\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n"
            latex += "Method & Precision@10 & Recall@10 & F1-score & MAP \\\\\n\\hline\n"
            for method in methods:
                m = metrics.get(method, {})
                precision = m.get('precision', 0.0)
                recall = m.get('recall', 0.0)
                f1 = m.get('f1', 0.0)
                map_score = m.get('map', 0.0)
                latex += f"{method} & {precision:.2f} & {recall:.2f} & {f1:.2f} & {map_score:.2f} \\\\\n"
            latex += "\\hline\n\\end{tabular}\n\\end{table}"
            output_path = gs_resource_path.joinpath('comparison_table.tex')
            gs_resource_path.mkdir(parents=True, exist_ok=True)
            with output_path.open('w') as f:
                f.write(latex)
            logger.info(f"LaTeX table saved to {output_path}")
            return latex
        except Exception as e:
            logger.error(f"Error generating LaTeX table: {e}")
            raise


def setup(no_topics: int, pr_mode: PRModes, agify_results: bool, fresh: bool = False,
          lda_no_topics: int = 200, bertopic_no_topics: int = 0) -> 'Evaluation':
    """
    Set up evaluation pipeline.

    :param no_topics: Number of topics for PageRank
    :param pr_mode: PageRank mode
    :param agify_results: Whether to use age-weighted results
    :param fresh: Whether to recompute results
    :param lda_no_topics: Number of topics for LDA
    :param bertopic_no_topics: Number of topics for BERTopic
    :return: Evaluation instance
    """
    logger = setup_logger()
    # Evaluation version 2 is query-aware; the former global-ranking cache is
    # methodologically invalid for per-query metrics.
    evaluation_cache_version = 'query-aware-v2'
    cache_path = Path('evaluation_results.pkl')
    metadata_path = Path('evaluation_results.meta.json')
    cache_is_current = False
    if not fresh and cache_path.exists() and metadata_path.exists():
        try:
            cache_is_current = (
                json.loads(metadata_path.read_text(encoding='utf-8')).get('cache_version')
                == evaluation_cache_version
            )
        except (OSError, ValueError):
            cache_is_current = False
    if not fresh and flag.is_module_done('evaluation_setup') and cache_is_current:
        with cache_path.open('rb') as f:
            return pickle.load(f)
    flag.clear_module_flag('evaluation_setup')
    try:
        database_setup()
        lda_results = lda_setup(lda_no_topics, bertopic_no_topics, fresh=fresh)
        log_msg = f"[EVALUATION] mode={pr_mode}, no_topics={no_topics}"
        pr = pagerank_setup(
            log_msg=log_msg,
            lda_results=lda_results,
            topic_model=TopicModelModes.LDA,
            no_topics=no_topics,
            pr_mode=pr_mode,
            bertopic_no_topics=bertopic_no_topics,
            fresh=fresh
        )
        qm = QueryManager(pr, lda_results)

        # Load golden standard queries
        gs_file = gs_resource_path.joinpath('golden_standard.json')
        if gs_file.exists():
            with gs_file.open() as f:
                golden_standard = json.load(f)
        else:
            golden_standard = {}
            logger.warning("No golden_standard.json found — evaluation metrics will be empty")

        # Compute a separate, LDA-topic-weighted ranking for every query.
        pr_results = pr.results(agify=agify_results)
        mode_metrics = []
        for query_text, relevant_ids in golden_standard.items():
            relevant_set = set(relevant_ids)
            ranked_papers = rank_pagerank_for_query(query_text, pr_results, lda_results)
            metrics = Evaluation.evaluate_recommendations(ranked_papers, relevant_set, k=10)
            metrics['query'] = query_text
            metrics['cluster'] = 0
            mode_metrics.append(metrics)

        results_by_mode = {pr_mode.value: mode_metrics}
        evaluation = Evaluation(results_by_mode, pr.pagerank_output_path)

        if mode_metrics:
            evaluation.visualize_comparison()
            # Build metrics dict for LaTeX table
            avg_metrics = {
                pr_mode.value: {
                    k: sum(m[k] for m in mode_metrics) / len(mode_metrics)
                    for k in ['precision', 'recall', 'f1', 'map']
                }
            }
            evaluation.generate_latex_table(avg_metrics, list(avg_metrics.keys()))
            # Print summary to stdout
            m = avg_metrics[pr_mode.value]
            print(f"\n=== Evaluation Results for mode: {pr_mode.value} ===")
            print(f"  Precision@10 : {m['precision']:.4f}")
            print(f"  Recall@10    : {m['recall']:.4f}")
            print(f"  F1-score     : {m['f1']:.4f}")
            print(f"  MAP          : {m['map']:.4f}")
            print(f"  Queries      : {len(mode_metrics)}")
        else:
            logger.warning("No golden standard queries — skipping visualization")

        with cache_path.open('wb') as f:
            pickle.dump(evaluation, f)
        metadata_path.write_text(
            json.dumps({'cache_version': evaluation_cache_version}, indent=2),
            encoding='utf-8'
        )
        flag.mark_module_done('evaluation_setup')
        return evaluation
    except Exception as e:
        logger.error(f"Error in evaluation setup: {e}")
        raise


def define_subparser(subparser):
    """
    Define subparser for evaluation command-line arguments.

    :param subparser: ArgumentParser subparser
    :return: Name of the subparser
    """
    evaluation = subparser.add_parser(subparser_name := 'evaluation')
    evaluation.add_argument('--no-topics', dest='no_topics', type=int, help='Number of topics', required=True)
    evaluation.add_argument('--pr-mode', dest='pr_mode', type=str, choices=[mode.value for mode in PRModes],
                            help='Algorithm mode to be used for customizing page rank.', required=True)
    evaluation.add_argument('--agify', dest='agify_results', action='store_true',
                            help='Whether the papers should weight based on their publish year.')
    evaluation.add_argument('--lda-no-topics', dest='lda_no_topics', type=int, default=200,
                            help='Number of topics for LDA')
    evaluation.add_argument('--bertopic-no-topics', dest='bertopic_no_topics', type=int, default=0,
                            help='Number of topics for BERTopic')
    evaluation.add_argument('--fresh', dest='fresh', action='store_true',
                            help='Force recomputation, ignoring cached results')
    return subparser_name
