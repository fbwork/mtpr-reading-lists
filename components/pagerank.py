import logging
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from math import sqrt
from pathlib import Path
from typing import Dict, List
from functools import lru_cache
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from networkx import NetworkXError
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

import utils
import utils.flag as flag
from components.base_pagerank import PageRankBase
from components.database import setup as database_setup
from components.lda import GensimLDA, setup as lda_setup
from components.prmodes import PRModes
from components.topic_model_modes import TopicModelModes
from models.tables import Paper
from utils import TaskAssistant

# Invalidates artifacts from the former THEMED transition implementation.
THEMED_CACHE_VERSION = 'themed-target-topic-transition-v2'
pr_output_path = Path(__file__).parent.joinpath('output', 'pr')
pr_log_path = Path(os.getenv('LOG_PATH', str(Path(__file__).parent.joinpath('logs')))).joinpath('pagerank.log')


def setup_logger():
    pr_log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=pr_log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()


def pr_callback(result, bar, pr_values_output_path: Path, cluster: int = 0, mode: str = 'normal'):
    """Callback function for PageRank computation"""
    import logging
    import traceback
    import sys
    
    logger = setup_logger()  # Use the same logger as pagerank module
    
    # Also print to stdout for debugging
    print(f"[PR_CALLBACK] Called for cluster {cluster}, mode {mode}", file=sys.stderr, flush=True)
    
    try:
        if result is None:
            error_msg = f"[PR_CALLBACK] Received None result for cluster {cluster}, mode {mode}"
            logger.error(error_msg)
            print(error_msg, file=sys.stderr, flush=True)
            return
        
        if not isinstance(result, tuple) or len(result) != 2:
            logger.error(f"[PR_CALLBACK] Invalid result format: {type(result)}, value: {result}")
            return
        
        t, pg_rank = result
        
        if pg_rank is None or not isinstance(pg_rank, dict) or len(pg_rank) == 0:
            logger.error(f"[PR_CALLBACK] Invalid pg_rank for topic {t}, cluster {cluster}: {type(pg_rank)}, len={len(pg_rank) if hasattr(pg_rank, '__len__') else 'N/A'}")
            return
        
        info_msg = f"[PR_CALLBACK] Saving results for topic {t}, cluster {cluster}, mode {mode}"
        logger.info(info_msg)
        print(info_msg, file=sys.stderr, flush=True)
        
        info_msg = f"[PR_CALLBACK] pg_rank has {len(pg_rank)} documents"
        logger.info(info_msg)
        print(info_msg, file=sys.stderr, flush=True)
        
        # Ensure directory exists
        pr_values_output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = pr_values_output_path.joinpath(f'topic-cluster{cluster}-{t}.bin')
        info_msg = f"[PR_CALLBACK] Saving to {file_path}"
        logger.info(info_msg)
        print(info_msg, file=sys.stderr, flush=True)
        
        # Save results
        data_to_save = {d: {t: pg_rank[d]} for d in pg_rank}
        with file_path.open('wb') as f:
            pickle.dump(data_to_save, f)
        
        file_size = file_path.stat().st_size
        success_msg = f"[PR_CALLBACK] Successfully saved {file_path}, size: {file_size} bytes"
        logger.info(success_msg)
        print(success_msg, file=sys.stderr, flush=True)
        
        if file_size == 0:
            error_msg = f"[PR_CALLBACK] WARNING: File {file_path} is empty!"
            logger.error(error_msg)
            print(error_msg, file=sys.stderr, flush=True)
            file_path.unlink(missing_ok=True)
            return
        
        flag.mark_module_done(f'pr_{mode}_topic_{cluster}_{t}')
        flag_msg = f"[PR_CALLBACK] Flag marked: pr_{mode}_topic_{cluster}_{t}"
        logger.info(flag_msg)
        print(flag_msg, file=sys.stderr, flush=True)
        
    except Exception as e:
        logger.error(f"[PR_CALLBACK] Exception in callback: {e}")
        logger.error(f"[PR_CALLBACK] Traceback: {traceback.format_exc()}")
        # Don't raise in callback - it might be in a worker process
    
    try:
        if bar is not None:
            bar.update()
    except:
        pass  # Ignore bar update errors


class PaperRankClassic(PageRankBase):
    """Classic PageRank with topic and centrality enhancements"""

    def __init__(self, lda_results: Dict[int, GensimLDA], mode: PRModes, no_topics: int):
        super().__init__(mode, no_topics)
        self.lda_results = lda_results
        self.pagerank_output_path = self.base_pagerank_output_path.joinpath(str(self.no_topics))
        self.pr_values_output_path = self.pagerank_output_path.joinpath('values', self.mode.value)
        self.pr_values_output_path.mkdir(parents=True, exist_ok=True)
        self._results = None
        self._agified_results = None

        # Cache centrality if needed
        if PRModes.requires_centrality(self.mode):
            self._setup_centrality_cache()

    def _setup_centrality_cache(self):
        """Initialize centrality caches based on mode"""
        logger = setup_logger()
        cache_dir = self.pagerank_output_path.joinpath('centrality_cache')
        cache_dir.mkdir(parents=True, exist_ok=True)

        centrality_types = []

        if self.mode in [PRModes.D_MODIFIED, PRModes.D_MODIFIED_BIAS, PRModes.D_MODIFIED_TRANSITION]:
            centrality_types.append('degree')
        elif self.mode in [PRModes.CL_MODIFIED, PRModes.CL_MODIFIED_BIAS, PRModes.CL_MODIFIED_TRANSITION]:
            centrality_types.append('closeness')
        elif self.mode in [PRModes.BW_MODIFIED, PRModes.BW_MODIFIED_BIAS, PRModes.BW_MODIFIED_TRANSITION]:
            centrality_types.append('betweenness')
        elif self.mode in [PRModes.KZ_MODIFIED, PRModes.KZ_MODIFIED_BIAS, PRModes.KZ_MODIFIED_TRANSITION]:
            centrality_types.append('katz')
        elif self.mode == PRModes.HYBRID:
            centrality_types = ['degree', 'closeness', 'betweenness', 'katz']

        for c_type in centrality_types:
            cache_file = cache_dir.joinpath(f'{c_type}_centrality.pkl')
            if not cache_file.exists():
                logger.info(f"Computing {c_type} centrality...")
                self.cache_centrality(self.graph, c_type, cache_file)

    @staticmethod
    def bias(mode: PRModes, t: int, d: int, W: nx.DiGraph, model: GensimLDA) -> float:
        """
        Calculate personalization bias for node d in topic t.

        Args:
            mode: PageRank mode
            t: Topic ID
            d: Document ID
            W: Citation graph (stochastic)
            model: LDA model for the cluster

        Returns:
            Bias value for node d
        """
        N = W.number_of_nodes()

        if mode == PRModes.NORMAL:
            # Uniform bias
            return 1.0 / N

        elif mode in [PRModes.PERSONALIZED, PRModes.THEMED]:
            # Topic-based bias only
            return model.topic_doc_score(d, t)

        elif mode in [PRModes.MODIFIED, PRModes.MODIFIED_BIAS]:
            # MTPR bias, proposal Eq. 4-5 (minus C_d and AS_jt -- centrality is
            # its own separate ablation axis in this framework, and journal/venue
            # credibility AS_jt is out of scope for now): a pure multiplicative
            # combination of topic relevance and normalised author credibility,
            # not the earlier additive "(1 + 0.3x)" boost.
            topic_score = model.topic_doc_score(d, t)
            author_score = model.doc_authors_score(t, d) / (model.topic_authors_score(t) or 1)
            return topic_score * author_score

        elif mode in [PRModes.MODIFIED_TRANSITION, PRModes.D_MODIFIED_TRANSITION,
                      PRModes.CL_MODIFIED_TRANSITION, PRModes.BW_MODIFIED_TRANSITION,
                      PRModes.KZ_MODIFIED_TRANSITION]:
            # Transition ablations (Table 1 of the paper): bias = P(t|d); the extra signal (author
            # credibility or centrality) is applied only in the transition. FIX (2026-10-24): the four
            # centrality *_TRANSITION modes were missing here, so they fell through to the uniform
            # `else` branch below and had been run with a uniform bias; only MODIFIED_TRANSITION used P(t|d).
            return model.topic_doc_score(d, t)

        elif mode in [PRModes.D_MODIFIED, PRModes.D_MODIFIED_BIAS]:
            # Degree centrality bias
            degree_centrality = model.topic_doc_degree_centrality_score(W, t, d)
            topic_score = model.topic_doc_score(d, t)
            total_degree = model.topic_degree_centrality_score(W, t) or 1
            return (degree_centrality / total_degree) * topic_score

        elif mode in [PRModes.CL_MODIFIED, PRModes.CL_MODIFIED_BIAS]:
            # Closeness centrality bias
            # topic_doc_closeness_centrality_score is now batch-cached in lda.py:
            # the full subgraph closeness is computed once per topic, not per document.
            closeness = model.topic_doc_closeness_centrality_score(W, t, d)
            topic_score = model.topic_doc_score(d, t)
            total_closeness = model.topic_closeness_centrality_score(W, t) or 1
            return (closeness / total_closeness) * topic_score

        elif mode in [PRModes.BW_MODIFIED, PRModes.BW_MODIFIED_BIAS]:
            # Betweenness centrality bias
            # topic_doc_betweenness_centrality_score is now batch-cached in lda.py:
            # betweenness_centrality_subset is computed once per topic, not per document.
            betweenness = model.topic_doc_betweenness_centrality_score(W, t, d)
            topic_score = model.topic_doc_score(d, t)
            total_betweenness = model.topic_betweenness_centrality_score(W, t) or 1
            return (betweenness / total_betweenness) * topic_score

        elif mode in [PRModes.KZ_MODIFIED, PRModes.KZ_MODIFIED_BIAS]:
            # Katz centrality bias, topic-subgraph-aware and normalised like
            # DC/CC/BC (previously computed once on the whole graph -- see
            # topic_katz_centrality_scores_batch's docstring in lda.py).
            katz = model.topic_doc_katz_centrality_score(W, t, d)
            topic_score = model.topic_doc_score(d, t)
            total_katz = model.topic_katz_centrality_score(W, t) or 1
            return (katz / total_katz) * topic_score

        elif mode == PRModes.HYBRID:
            # Weighted combination of all four centralities, each normalised
            # the same way (topic-subgraph value / topic-subgraph total), then
            # combined multiplicatively with topic relevance -- consistent
            # with the D/CL/BW/KZ_MODIFIED bias formulas above, not the
            # additive "(1 + 0.5x)" boost this used previously.
            topic_score = model.topic_doc_score(d, t)

            degree = model.topic_doc_degree_centrality_score(W, t, d)
            total_degree = model.topic_degree_centrality_score(W, t) or 1

            closeness = model.topic_doc_closeness_centrality_score(W, t, d)
            total_closeness = model.topic_closeness_centrality_score(W, t) or 1

            betweenness = model.topic_doc_betweenness_centrality_score(W, t, d)
            total_betweenness = model.topic_betweenness_centrality_score(W, t) or 1

            katz = model.topic_doc_katz_centrality_score(W, t, d)
            total_katz = model.topic_katz_centrality_score(W, t) or 1

            # Default weights: 0.25 each
            centrality_score = (
                    0.25 * (degree / total_degree) +
                    0.25 * (closeness / total_closeness) +
                    0.25 * (betweenness / total_betweenness) +
                    0.25 * (katz / total_katz)
            )

            return topic_score * centrality_score

        else:
            # Default: uniform
            return 1.0 / N

    @staticmethod
    def transition(mode: PRModes, t: int, d: int, d_prime: int, W: nx.DiGraph, model: GensimLDA) -> float:
        """
        Calculate transition probability from d_prime to d for topic t.

        Args:
            mode: PageRank mode
            t: Topic ID
            d: Target document ID
            d_prime: Source document ID
            W: Citation graph
            model: LDA model

        Returns:
            Transition probability
        """
        out_degree = W.out_degree(d_prime)
        if out_degree == 0:
            return 0.0

        base_transition = 1.0 / out_degree

        if mode in [PRModes.NORMAL, PRModes.PERSONALIZED]:
            # Standard transition
            return base_transition

        elif mode in [PRModes.THEMED, PRModes.MODIFIED, PRModes.MODIFIED_TRANSITION,
                      PRModes.D_MODIFIED, PRModes.CL_MODIFIED, PRModes.BW_MODIFIED, PRModes.KZ_MODIFIED,
                      PRModes.D_MODIFIED_TRANSITION, PRModes.CL_MODIFIED_TRANSITION,
                      PRModes.BW_MODIFIED_TRANSITION, PRModes.KZ_MODIFIED_TRANSITION]:
            # Jardine (2014, p.60) / proposal Eq. 4-6: normalize T*(t,d,d') over
            # link_in(d) -- the set of papers that cite d -- not over d''s successors.
            # The plain D/CL/BW/KZ_MODIFIED variants (Table 1: bias=DC*P(t|d),
            # transition=T*) get the same *unweighted* T* as THEMED/MODIFIED here --
            # transition_star()'s fallback branch returns base_tstar for them, since
            # only the "_transition" suffix variants multiply T* by the centrality
            # term. Only the bias is centrality-weighted for the plain variants.
            return normalized_transition_weights(mode, t, d, W, model).get(d_prime, 0.0)

        elif mode == PRModes.HYBRID:
            # Hybrid uses bias only, standard transition
            return base_transition

        else:
            return base_transition

    def compute_results(self, fresh=False):
        """Compute PageRank results for all topics"""
        logger = setup_logger()
        logger.info(f"[PAGERANK COMPUTE RESULTS] Mode={self.mode.value}, fresh={fresh}")

        aggregated_file = self.pagerank_output_path.joinpath(f'pageranks_{self.mode.value}.bin')
        metadata_file = self.pagerank_output_path.joinpath(f'pageranks_{self.mode.value}.meta.json')
        main_flag = f'pr_{self.mode.value}'

        # THEMED's previous artifacts are not comparable: its old raw transition
        # was source-only, hence became uniform after successor normalization.
        # Require an explicit implementation marker before loading its cache.
        themed_cache_is_current = False
        if self.mode == PRModes.THEMED and metadata_file.exists():
            try:
                import json
                themed_cache_is_current = (
                    json.loads(metadata_file.read_text(encoding='utf-8')).get('cache_version')
                    == THEMED_CACHE_VERSION
                )
            except (OSError, ValueError):
                themed_cache_is_current = False

        if self.mode == PRModes.THEMED and not themed_cache_is_current:
            logger.warning("Invalidating stale THEMED PageRank artifacts.")
            aggregated_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            for topic_file in self.pr_values_output_path.glob('topic-cluster*-*.bin'):
                topic_file.unlink()
            for cluster in self.lda_results:
                flag.clear_module_flag(f'pr_{self.mode.value}_cluster_{cluster}')
                for topic_id in range(self.no_topics):
                    flag.clear_module_flag(f'pr_{self.mode.value}_topic_{cluster}_{topic_id}')
            flag.clear_module_flag(main_flag)

        # Smart detection of completion
        if (not fresh and aggregated_file.exists() and flag.is_module_done(main_flag)
                and (self.mode != PRModes.THEMED or themed_cache_is_current)):
            logger.info(f"Loading complete results for mode {self.mode}...")
            with open(aggregated_file, 'rb') as f:
                self._results = pickle.load(f)
            self._agified_results = deepcopy(self._results)
            for paper in Paper.select(Paper.id, Paper.year):
                for t_id in self._agified_results.get(paper.id - 1, {}):
                    year_diff = max(1, 2011 - paper.year)
                    self._agified_results[paper.id - 1][t_id] /= year_diff
            return

        logger.info("Proceeding with computation / continuation...")

        database_setup()
        self._results = self._process()

        # Save results
        self._agified_results = deepcopy(self._results)
        for paper in Paper.select(Paper.id, Paper.year):
            for t_id in self._agified_results.get(paper.id - 1, {}):
                year_diff = max(1, 2011 - paper.year)
                self._agified_results[paper.id - 1][t_id] /= year_diff

        with open(aggregated_file, 'wb') as f:
            pickle.dump(self._results, f)
        if self.mode == PRModes.THEMED:
            import json
            metadata_file.write_text(
                json.dumps({'cache_version': THEMED_CACHE_VERSION}, indent=2),
                encoding='utf-8'
            )
        flag.mark_module_done(main_flag)

        logger.info(f"PageRank computation completed for mode: {self.mode.value}")

    def _process(self):
        """Process PageRank for all clusters"""
        logger = setup_logger()
        logger.info(f"[PAGERANK PROCESS] self.lda_results keys: {list(self.lda_results.keys()) if self.lda_results else []}, len={len(self.lda_results) if self.lda_results else 0}")

        # Check GPU availability
        # import torch
        # if torch.cuda.is_available():
        #     gpu_name = torch.cuda.get_device_name(0)
        #     gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        #     logger.info(f"[PAGERANK PROCESS] GPU available: {gpu_name} ({gpu_memory:.2f} GB)")
        #     print(f"[GPU INFO] GPU available: {gpu_name} ({gpu_memory:.2f} GB)")
        #     print(f"[GPU INFO] NOTE: Classic PageRank modes (normal, hybrid, etc.) use CPU (NetworkX).")
        #     print(f"[GPU INFO] Only GAT and Flow Matching modes use GPU.")
        # else:
        #     logger.info("[PAGERANK PROCESS] GPU not available, using CPU")
        #     print("[GPU INFO] GPU not available, using CPU")

        if not self.lda_results:
            logger.error("[PAGERANK PROCESS] No LDA results provided for clustering. Check lda_setup flags/files.")
            raise ValueError("No LDA results provided for clustering.")

        all_results = defaultdict(dict)

        for cluster, lda in self.lda_results.items():
            # === اصلاح شده ===
            cluster_flag_done = flag.is_module_done(f'pr_{self.mode.value}_cluster_{cluster}')

            # Force check actual files
            topic_files_count = len(list(self.pr_values_output_path.glob(f'topic-cluster{cluster}-*.bin')))

            if cluster_flag_done and topic_files_count >= self.no_topics:
                logger.info(
                    f"[PAGERANK PROCESS] Cluster {cluster} flag exists and seems complete ({topic_files_count}/{self.no_topics} topics), loading...")
                cluster_results = self._collect_cluster(cluster)
                if cluster_results:
                    for doc_id, topic_scores in cluster_results.items():
                        all_results[doc_id].update(topic_scores)
                    continue
            elif cluster_flag_done:
                logger.warning(
                    f"[PAGERANK PROCESS] Cluster flag exists but only {topic_files_count}/{self.no_topics} topics found → continuing...")
            # === پایان اصلاح ===

            remained_ids = self._remained_topic_ids(cluster)
            logger.info(
                f"[PAGERANK PROCESS] Cluster {cluster}: remained_ids={remained_ids}, len={len(remained_ids) if remained_ids else 0}")
            
            if remained_ids:
                logger.info(f"[PAGERANK PROCESS] Computing PageRank for cluster {cluster}, topics: {remained_ids}")
                logger.info(f"[PAGERANK PROCESS] self.pr_values_output_path={self.pr_values_output_path}")
                logger.info(f"[PAGERANK PROCESS] Directory exists: {self.pr_values_output_path.exists()}")
                
                # Ensure directory exists before TaskAssistant
                self.pr_values_output_path.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Use custom page_rank function for all modes including HYBRID
                    # Temporarily disable multiprocessing to debug callback issues
                    # Set mp_core_frac to False (0) to use single process
                    TaskAssistant(
                        page_rank,
                        {'t': remained_ids, 'cluster': cluster},
                        f"Computing PageRank on mode `{self.mode}` for cluster {cluster}...",
                        pr_callback,
                        callback_kwargs={'pr_values_output_path': self.pr_values_output_path, 'cluster': cluster, 'mode': self.mode.value},
                        task_kwargs={'graph': self.graph, 'model': lda, 'alpha': 0.15, 'mode': self.mode},
                        chunk_size=1,
                        mp_core_frac=False  # Disable multiprocessing for debugging
                    ).execute()
                    
                    logger.info(f"[PAGERANK PROCESS] TaskAssistant completed for cluster {cluster}")
                    
                    # Also try direct execution for first topic to test
                    if remained_ids:
                        logger.info(f"[PAGERANK PROCESS] Testing direct execution for topic {remained_ids[0]}")
                        try:
                            test_result = page_rank(
                                t=remained_ids[0],
                                graph=self.graph,
                                model=lda,
                                alpha=0.15,
                                mode=self.mode,
                                cluster=cluster
                            )
                            logger.info(f"[PAGERANK PROCESS] Direct execution result: type={type(test_result)}, len={len(test_result[1]) if isinstance(test_result, tuple) else 'N/A'}")
                            # Manually call callback for test
                            pr_callback(
                                test_result,
                                None,  # No bar for test
                                self.pr_values_output_path,
                                cluster,
                                self.mode.value
                            )
                        except Exception as e:
                            logger.error(f"[PAGERANK PROCESS] Direct execution failed: {e}")
                            import traceback
                            logger.error(f"[PAGERANK PROCESS] Traceback: {traceback.format_exc()}")
                    
                except Exception as e:
                    logger.error(f"[PAGERANK PROCESS] TaskAssistant failed for cluster {cluster}: {e}")
                    import traceback
                    logger.error(f"[PAGERANK PROCESS] Traceback: {traceback.format_exc()}")
                    raise

                # Wait a bit for files to be written (especially in multiprocessing)
                import time
                time.sleep(0.5)
                
                # Collect results after TaskAssistant completes
                cluster_results = self._collect_cluster(cluster)
                logger.info(f"[PAGERANK PROCESS] Collected {len(cluster_results)} documents from cluster {cluster}")
                
                if not cluster_results:
                    # Check if files were created
                    bin_files = list(self.pr_values_output_path.glob(f'topic-cluster{cluster}-*.bin'))
                    logger.error(f"[PAGERANK PROCESS] No results collected for cluster {cluster}")
                    logger.error(f"[PAGERANK PROCESS] Found {len(bin_files)} bin files: {[f.name for f in bin_files]}")
                    
                    # Check flags
                    for t in remained_ids:
                        flag_name = f'pr_{self.mode.value}_topic_{cluster}_{t}'
                        flag_done = flag.is_module_done(flag_name)
                        logger.error(f"[PAGERANK PROCESS] Topic {t}: flag={flag_name}, done={flag_done}")
                    
                    if not bin_files:
                        raise ValueError(f"No topic-cluster files created for cluster {cluster}. TaskAssistant may have failed or callback had errors.")
                    else:
                        raise ValueError(f"Files exist but no results collected for cluster {cluster}. Files may be corrupted.")
                
                for doc_id, topic_scores in cluster_results.items():
                    all_results[doc_id].update(topic_scores)
                
                if all_results:
                    flag.mark_module_done(f'pr_{self.mode.value}_cluster_{cluster}')
                    logger.info(f"[PAGERANK PROCESS] Successfully processed cluster {cluster}, added {len(cluster_results)} documents to all_results")
                else:
                    logger.error(f'[PAGERANK PROCESS] No results in all_results after processing cluster {cluster}')
                    raise ValueError(f"No results for cluster in pagerank computation{cluster}.")
            else:
                logger.info(f"[PAGERANK PROCESS] No remaining topics for cluster {cluster}, all topics already processed")

        return all_results

    def _remained_topic_ids(self, cluster):
        """Find topic IDs that haven't been processed yet"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[DEBUG _remained_topic_ids] Called for cluster={cluster}, no_topics={self.no_topics}")
        logger.info(f"[DEBUG _remained_topic_ids] pr_values_output_path={self.pr_values_output_path}")

        completed_topics = set()
        for t in range(self.no_topics):
            flag_name = f'pr_{self.mode.value}_topic_{cluster}_{t}'
            file_path = self.pr_values_output_path.joinpath(f'topic-cluster{cluster}-{t}.bin')
            flag_done = flag.is_module_done(flag_name)
            file_exists = file_path.exists()

            logger.info(
                f"[DEBUG _remained_topic_ids] topic {t}: flag={flag_name} done={flag_done}, file_exists={file_exists}")

            if flag_done and file_exists:
                completed_topics.add(t)

        remained = list(set(range(self.no_topics)).difference(completed_topics))
        logger.info(
            f"[DEBUG _remained_topic_ids] completed_topics={sorted(completed_topics)}, remained_ids={remained} (len={len(remained)})")
        return remained

    def _collect_cluster(self, cluster):
        """Collect PageRank results for a specific cluster"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[DEBUG _collect_cluster] Called for cluster={cluster}")
        logger.info(f"[DEBUG _collect_cluster] pr_values_output_path={self.pr_values_output_path}")
        logger.info(f"[DEBUG _collect_cluster] Directory exists: {self.pr_values_output_path.exists()}")
        
        if self.pr_values_output_path.exists():
            all_files = list(self.pr_values_output_path.glob('*'))
            logger.info(f"[DEBUG _collect_cluster] All files in dir {self.pr_values_output_path} ({len(all_files)}): {[f.name for f in all_files]}")
        
        bin_files = list(self.pr_values_output_path.glob('*.bin'))
        logger.info(f"[DEBUG _collect_cluster] Found {len(bin_files)} *.bin files: {[f.name for f in bin_files]}")
        
        page_ranks = defaultdict(dict)
        matcher = utils.REMatcher(f'topic-cluster{cluster}-(\\d+).bin')
        
        for pr_file in bin_files:
            logger.info(f"[DEBUG _collect_cluster] Checking file: {pr_file.name}")
            if res := matcher.match(pr_file.name):
                logger.info(f"[DEBUG _collect_cluster] MATCHED file {pr_file.name} for topic {res[0]}")
                try:
                    with pr_file.open('rb') as f:
                        tmp = pickle.load(f)
                    logger.info(f"[DEBUG _collect_cluster] Loaded tmp with {len(tmp)} items from {pr_file.name}")
                    for idx in tmp:
                        page_ranks[idx].update(tmp[idx])
                except Exception as e:
                    logger.error(f"[DEBUG _collect_cluster] Failed to load pickle from {pr_file.name}: {e}")
            else:
                logger.debug(f"[DEBUG _collect_cluster] NO MATCH for {pr_file.name}")
        
        logger.info(f"[DEBUG _collect_cluster] Final page_ranks has {len(page_ranks)} document entries")
        return page_ranks

    def results(self, agify=True) -> Dict[int, Dict[int, float]]:
        """Get PageRank results"""
        self.compute_results()
        return self._agified_results if agify else self._results


class PaperRankGAT(PageRankBase):
    """GAT-based PageRank variants"""

    def __init__(self, lda_results: Dict[int, GensimLDA], mode: PRModes, no_topics: int):
        super().__init__(mode, no_topics)
        self.lda_results = lda_results
        self.pagerank_output_path = self.base_pagerank_output_path.joinpath(str(self.no_topics))
        self.pr_values_output_path = self.pagerank_output_path.joinpath('values', self.mode.value)
        self.pr_values_output_path.mkdir(parents=True, exist_ok=True)
        self._paths = None
        self._results = None
        self._embeddings = None
        self._katz_centrality_cache = None   # Global Katz centrality dict {node_id: score}
        self._global_degree_cache = None     # Global degree centrality dict {node_id: score}
        self._global_closeness_cache = None  # Global closeness centrality dict {node_id: score}
        self._global_betweenness_cache = None # Global betweenness centrality dict {node_id: score}
        self._topic_total_cache = {}  # Cache for topic total centrality scores (degree, closeness, betweenness)
        # Pre-load global centralities if already cached (avoids topic-specific O(V²) computation)
        self._load_global_centralities()

    def _load_global_centralities(self):
        """
        Load pre-computed global centrality caches if they exist.

        OPTIMIZATION: GAT modes with centrality (GAT_CLOSENESS, GAT_BETWEENNESS,
        GAT_CENTRALITY, GAT_MTPR_CENTRALITY) were previously calling
        model.topic_doc_closeness/betweenness_centrality_score() which computes
        O(V²) closeness on a topic subgraph — once per NODE (15666 × 5 times).

        Instead, we load the global centrality caches computed by _setup_centrality_cache()
        or compute_gat_embeddings(). These cover all nodes at once and are O(1) lookups.
        """
        cache_dir = self.pagerank_output_path.joinpath('centrality_cache')
        for c_type, attr in [
            ('degree', '_global_degree_cache'),
            ('closeness', '_global_closeness_cache'),
            ('betweenness', '_global_betweenness_cache'),
            ('katz', '_katz_centrality_cache'),
        ]:
            cache_file = cache_dir.joinpath(f'{c_type}_centrality.pkl')
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        setattr(self, attr, pickle.load(f))
                except Exception:
                    pass  # Will be computed on-demand

    def compute_gat_embeddings(self, graph_data: Data, fresh: bool = False) -> torch.Tensor:
        """Compute GAT embeddings with optional centrality features"""
        import time
        import torch
        import numpy as np
        start_time = time.perf_counter()
        logger = setup_logger()
        cache_path = self.pagerank_output_path.joinpath(f'gat_embeddings_{self.mode.value}.bin')

        if not fresh and cache_path.exists() and flag.is_module_done(f'pr_{self.mode.value}_gat_embeddings'):
            logger.info(f"Loading GAT embeddings for mode {self.mode}...")
            end_time = time.perf_counter()
            print(f"GAT embeddings load time: {end_time - start_time:.2f} seconds")
            return torch.load(cache_path)
        
        # GPU info
        if torch.cuda.is_available():
            print(f"[GPU] Using GPU for GAT embeddings: {torch.cuda.get_device_name(0)}")
        else:
            print("[GPU] WARNING: GPU not available, using CPU (will be slow)")
        
        # For GAT_CENTRALITY and GAT_MTPR_CENTRALITY, add centrality features
        add_centrality_features = self.mode in [PRModes.GAT_CENTRALITY, PRModes.GAT_MTPR_CENTRALITY]
        
        # For GAT_KATZ, pre-compute and cache Katz centrality to avoid recomputation
        if self.mode == PRModes.GAT_KATZ and self._katz_centrality_cache is None:
            logger.info("Pre-computing Katz centrality for GAT_KATZ mode...")
            cache_dir = self.pagerank_output_path.joinpath('centrality_cache')
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir.joinpath('katz_centrality.pkl')
            if not cache_file.exists():
                self.cache_centrality(self.graph, 'katz', cache_file)
            with open(cache_file, 'rb') as f:
                self._katz_centrality_cache = pickle.load(f)
            logger.info("Katz centrality cached for GAT_KATZ mode")
        
        if add_centrality_features:
            logger.info(f"Adding centrality features for {self.mode.value}...")
            # Load or compute centrality features
            cache_dir = self.pagerank_output_path.joinpath('centrality_cache')
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            centralities = {}
            for c_type in ['degree', 'closeness', 'betweenness', 'katz']:
                cache_file = cache_dir.joinpath(f'{c_type}_centrality.pkl')
                if not cache_file.exists():
                    logger.info(f"Computing {c_type} centrality for GAT...")
                    self.cache_centrality(self.graph, c_type, cache_file)
                with open(cache_file, 'rb') as f:
                    centralities[c_type] = pickle.load(f)
            
            # Cache Katz centrality for use in embedding_bias (avoid recomputation)
            self._katz_centrality_cache = centralities.get('katz', {})
            
            # Create centrality feature matrix
            num_nodes = len(self.graph.nodes())
            centrality_features = np.zeros((num_nodes, 4))
            node_list = list(self.graph.nodes())
            
            for i, node in enumerate(node_list):
                centrality_features[i, 0] = centralities['degree'].get(node, 0)
                centrality_features[i, 1] = centralities['closeness'].get(node, 0)
                centrality_features[i, 2] = centralities['betweenness'].get(node, 0)
                centrality_features[i, 3] = centralities['katz'].get(node, 0)
            
            # Normalize
            centrality_features = normalize(centrality_features, axis=0, norm='l2')
            centrality_tensor = torch.tensor(centrality_features, dtype=torch.float)
            
            # Concatenate with existing features
            if graph_data.x.shape[0] == centrality_tensor.shape[0]:
                graph_data.x = torch.cat([graph_data.x, centrality_tensor], dim=1)
                logger.info(f"Concatenated centrality features: {graph_data.x.shape}")
            else:
                logger.warning(f"Size mismatch: graph_data.x={graph_data.x.shape[0]}, centrality={centrality_tensor.shape[0]}")

        # For GAT_TOPIC_AWARE, use topic-specific attention heads
        use_topic_aware = self.mode == PRModes.GAT_TOPIC_AWARE
        
        class GATModel(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, heads=4, num_topics=5):
                super().__init__()
                if use_topic_aware:
                    # Topic-aware: one head per topic
                    self.conv1 = GATConv(in_channels, hidden_channels, heads=num_topics, dropout=0.2)
                    self.conv2 = GATConv(hidden_channels * num_topics, out_channels, heads=1, dropout=0.2)
                else:
                    # Standard GAT
                    self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2)
                    self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=0.2)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = F.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

        try:
            if torch.cuda.is_available():
                initial_mem = torch.cuda.memory_allocated() / 1024**2  # MB
                max_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
                print(f"GPU memory before GAT computation: allocated {initial_mem:.2f} MB, max {max_mem:.2f} MB")
                torch.cuda.empty_cache()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_topics = self.no_topics if use_topic_aware else 4
            model = GATModel(
                in_channels=graph_data.x.shape[1],
                hidden_channels=64,
                out_channels=32,
                num_topics=num_topics
            ).to(device)

            model.eval()
            graph_data = graph_data.to(device)

            with torch.no_grad():
                embeddings = model(graph_data)

            embeddings = embeddings.cpu()
            if torch.cuda.is_available():
                after_mem = torch.cuda.memory_allocated() / 1024**2
                max_after = torch.cuda.max_memory_allocated() / 1024**2
                print(f"GPU memory after GAT computation: allocated {after_mem:.2f} MB, max {max_after:.2f} MB")
                torch.cuda.reset_peak_memory_stats()
            torch.save(embeddings, cache_path)
            flag.mark_module_done(f'pr_{self.mode.value}_gat_embeddings')
            end_time = time.perf_counter()
            print(f"GAT embeddings computation time: {end_time - start_time:.2f} seconds")
            logger.info(f"GAT embeddings computed in {end_time - start_time:.2f} seconds")
            return embeddings

        except RuntimeError as e:
            logger.error(f"Error computing GAT embeddings: {e}")
            raise RuntimeError("Failed to compute GAT embeddings, possibly due to CUDA memory issues.")

    class FlowMatchingModel(torch.nn.Module):
        """Flow Matching model for citation prediction"""

        def __init__(self, in_channels, hidden_channels):
            super().__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(in_channels + 1, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, in_channels)
            )

        def forward(self, x, t, edge_index):
            t = t.view(-1, 1)
            x_t = torch.cat([x, t], dim=1)
            return self.mlp(x_t)

    def train_flow_matching(self, graph_data: Data, embeddings: torch.Tensor,
                            fresh: bool = False, num_epochs: int = 200,
                            batch_size: int = 32) -> torch.nn.Module:
        """Train Flow Matching model"""
        import time
        start_time = time.perf_counter()
        logger = setup_logger()
        model_path = self.pagerank_output_path.joinpath(f'flow_matching_model_{self.mode.value}.bin')

        if not fresh and model_path.exists() and flag.is_module_done(f'pr_{self.mode.value}_flow_matching'):
            logger.info(f"Loading Flow Matching model for mode {self.mode}...")
            end_time = time.perf_counter()
            print(f"Flow Matching model load time: {end_time - start_time:.2f} seconds")
            return torch.load(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.FlowMatchingModel(
            in_channels=embeddings.shape[1],
            hidden_channels=64
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler()

        try:
            if torch.cuda.is_available():
                initial_mem = torch.cuda.memory_allocated() / 1024**2
                max_mem = torch.cuda.max_memory_allocated() / 1024**2
                print(f"GPU memory before Flow Matching training: allocated {initial_mem:.2f} MB, max {max_mem:.2f} MB")
                torch.cuda.empty_cache()

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                indices = torch.randperm(embeddings.shape[0])[:batch_size]
                batch_embeddings = embeddings[indices].to(device)

                # BUGFIX: torch.isin() requires both tensors on the same device.
                # graph_data.edge_index may be on GPU (moved earlier), but indices
                # is created on CPU by torch.randperm. Filter on CPU, then move result.
                edge_index_cpu = graph_data.edge_index.cpu()
                mask = (torch.isin(edge_index_cpu[0], indices) &
                        torch.isin(edge_index_cpu[1], indices))
                batch_edge_index = edge_index_cpu[:, mask].to(device)

                t = torch.rand(batch_embeddings.shape[0], 1).to(device)
                noise = torch.randn_like(batch_embeddings)

                with autocast():
                    x_t = batch_embeddings + t * noise
                    pred = model(x_t, t, batch_edge_index)
                    loss = F.mse_loss(pred, noise)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item()}")

            if torch.cuda.is_available():
                after_mem = torch.cuda.memory_allocated() / 1024**2
                max_after = torch.cuda.max_memory_allocated() / 1024**2
                print(f"GPU memory after Flow Matching training: allocated {after_mem:.2f} MB, max {max_after:.2f} MB")
                torch.cuda.reset_peak_memory_stats()
            torch.save(model, model_path)
            flag.mark_module_done(f'pr_{self.mode.value}_flow_matching')
            end_time = time.perf_counter()
            print(f"Flow Matching training time: {end_time - start_time:.2f} seconds")
            logger.info(f"Flow Matching model trained in {end_time - start_time:.2f} seconds")
            return model

        except RuntimeError as e:
            logger.error(f"Error training Flow Matching model: {e}")
            raise RuntimeError("Failed to train Flow Matching model.")

    def generate_paths(self, flow_model: torch.nn.Module, embeddings: torch.Tensor,
                       graph_data: Data, fresh: bool = False,
                       num_paths: int = 10, path_length: int = 3) -> List[List[int]]:
        """Generate citation paths using Flow Matching"""
        logger = setup_logger()
        paths_path = self.pagerank_output_path.joinpath(f'flow_matching_paths_{self.mode.value}.bin')

        if not fresh and paths_path.exists() and flag.is_module_done(f'pr_{self.mode.value}_paths'):
            logger.info(f"Loading Flow Matching paths for mode {self.mode}...")
            with open(paths_path, 'rb') as f:
                return pickle.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                x = torch.randn_like(embeddings).to(device)
                for t in torch.linspace(0, 1, 100):
                    x = x + flow_model(x, t, graph_data.edge_index.to(device)) * 0.01

            paths = []
            for _ in range(num_paths):
                path = []
                current_node = torch.argmax(x[0]).item()

                for _ in range(path_length):
                    path.append(current_node)
                    neighbors = graph_data.edge_index[1][graph_data.edge_index[0] == current_node]

                    if len(neighbors) == 0:
                        logger.warning(f"No neighbors for node {current_node}, path truncated")
                        break

                    neighbor_scores = x[neighbors]
                    probabilities = F.softmax(neighbor_scores, dim=0)

                    if probabilities.sum() == 0:
                        logger.warning(f"Zero probabilities for node {current_node}, skipping")
                        break

                    current_node = neighbors[torch.multinomial(probabilities, 1).item()].item()

                if path:
                    paths.append(path)

            with open(paths_path, 'wb') as f:
                pickle.dump(paths, f)
            flag.mark_module_done(f'pr_{self.mode.value}_paths')

            return paths

        except Exception as e:
            logger.error(f"Error generating paths: {e}")
            return []

    def _get_global_centrality(self, cache_attr: str, c_type: str) -> dict:
        """
        Return the global centrality dict, computing and caching it if needed.
        OPTIMIZATION: Computes centrality ONCE for all nodes, not once per node per topic.
        """
        cached = getattr(self, cache_attr)
        if cached is not None:
            return cached
        cache_dir = self.pagerank_output_path.joinpath('centrality_cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir.joinpath(f'{c_type}_centrality.pkl')
        new_cache = self.cache_centrality(self.graph, c_type, cache_file)
        setattr(self, cache_attr, new_cache)
        return new_cache

    def embedding_bias(self, mode: PRModes, t: int, d: int, embeddings: torch.Tensor, model: GensimLDA) -> float:
        """
        Calculate embedding-based bias for GAT mode.

        OPTIMIZATION: For centrality-based GAT modes (GAT_DEGREE, GAT_CLOSENESS,
        GAT_BETWEENNESS, GAT_KATZ, GAT_CENTRALITY, GAT_MTPR_CENTRALITY),
        we use GLOBAL pre-computed centrality caches instead of calling
        model.topic_doc_*_centrality_score() which would compute O(V²) topic-specific
        closeness/betweenness for every node on every call.

        The global cache for each centrality type is computed once (O(V²) or O(V×E))
        and looked up in O(1) thereafter.
        """
        topic_score = model.topic_doc_score(d, t)
        
        # Base embedding norm
        emb_norm = torch.norm(embeddings[d]).item()
        base_emb_score = 1 + 0.5 * emb_norm / (torch.norm(embeddings).mean().item() + 1e-8)
        
        if mode == PRModes.GAT:
            # Standard GAT: embedding norm only
            return topic_score * base_emb_score
        
        elif mode == PRModes.GAT_DEGREE:
            # GAT with degree centrality — use global cache (O(1) lookup)
            degree_cache = self._get_global_centrality('_global_degree_cache', 'degree')
            degree_weight = degree_cache.get(d, 0)
            return topic_score * base_emb_score * (1 + 0.3 * degree_weight)
        
        elif mode == PRModes.GAT_CLOSENESS:
            # GAT with closeness centrality — use global cache (O(1) lookup)
            # Previously called model.topic_doc_closeness_centrality_score() per node
            # which re-ran O(V²) closeness for every single document. Fixed.
            closeness_cache = self._get_global_centrality('_global_closeness_cache', 'closeness')
            closeness_weight = closeness_cache.get(d, 0)
            return topic_score * base_emb_score * (1 + 0.3 * closeness_weight)
        
        elif mode == PRModes.GAT_BETWEENNESS:
            # GAT with betweenness centrality — use global cache (O(1) lookup)
            betweenness_cache = self._get_global_centrality('_global_betweenness_cache', 'betweenness')
            betweenness_weight = betweenness_cache.get(d, 0)
            return topic_score * base_emb_score * (1 + 0.3 * betweenness_weight)
        
        elif mode == PRModes.GAT_KATZ:
            # GAT with Katz centrality — use global cache (O(1) lookup)
            katz_cache = self._get_global_centrality('_katz_centrality_cache', 'katz')
            katz_score = katz_cache.get(d, 0)
            katz_mean = sum(katz_cache.values()) / len(katz_cache) if katz_cache else 1
            katz_weight = (katz_score / katz_mean) if katz_mean > 0 else 0
            return topic_score * base_emb_score * (1 + 0.3 * katz_weight)
        
        elif mode in [PRModes.GAT_CENTRALITY, PRModes.GAT_MTPR_CENTRALITY]:
            # GAT with all 4 centralities — all use global cache (O(1) lookups each)
            # Previously: called topic-specific closeness/betweenness per node → hours of compute.
            # Now: all 4 are O(1) dict lookups using pre-computed global caches.
            degree_cache = self._get_global_centrality('_global_degree_cache', 'degree')
            closeness_cache = self._get_global_centrality('_global_closeness_cache', 'closeness')
            betweenness_cache = self._get_global_centrality('_global_betweenness_cache', 'betweenness')
            katz_cache = self._get_global_centrality('_katz_centrality_cache', 'katz')

            degree_weight = degree_cache.get(d, 0)
            closeness_weight = closeness_cache.get(d, 0)
            betweenness_weight = betweenness_cache.get(d, 0)
            katz_score = katz_cache.get(d, 0)
            katz_mean = sum(katz_cache.values()) / len(katz_cache) if katz_cache else 1
            katz_weight = (katz_score / katz_mean) if katz_mean > 0 else 0

            centrality_score = 0.25 * degree_weight + 0.25 * closeness_weight + 0.25 * betweenness_weight + 0.25 * katz_weight
            return topic_score * base_emb_score * (1 + 0.5 * centrality_score)
        
        else:
            # Default: standard GAT
            return topic_score * base_emb_score

    def embedding_transition(self, mode: PRModes, t: int, d: int, d_prime: int,
                             W: nx.DiGraph, embeddings: torch.Tensor, model: GensimLDA) -> float:
        """Calculate embedding-based transition for GAT mode"""
        out_degree = W.out_degree(d_prime)
        if out_degree == 0:
            return 0.0
        base_transition = 1.0 / out_degree
        
        # Cosine similarity between embeddings
        sim = F.cosine_similarity(embeddings[d].unsqueeze(0), embeddings[d_prime].unsqueeze(0)).item()
        
        # For GAT_TOPIC_AWARE, use topic-specific attention
        if mode == PRModes.GAT_TOPIC_AWARE:
            # Topic-aware transition with enhanced embedding similarity
            topic_score_d = model.topic_doc_score(d, t)
            topic_score_d_prime = model.topic_doc_score(d_prime, t)
            # Cache topic_docs_score per topic to avoid recomputation
            cache_key_topic_docs = f'topic_docs_{t}'
            if cache_key_topic_docs not in self._topic_total_cache:
                self._topic_total_cache[cache_key_topic_docs] = model.topic_docs_score(t) or 1
            total_topic_score = self._topic_total_cache[cache_key_topic_docs]
            topic_weight = (topic_score_d * topic_score_d_prime) / total_topic_score
            return base_transition * (1 + sim) * (1 + 0.5 * topic_weight)
        
        # For other GAT modes, use standard transition_star
        topic_trans = transition_star(mode, t, d, d_prime, W, model)
        return base_transition * topic_trans * (1 + sim)

    def embedding_page_rank(self, mode: PRModes, t: int, graph: nx.DiGraph, model: GensimLDA,
                            embeddings: torch.Tensor, alpha: float, max_iterations: int = 100,
                            epsilon: float = 1.0e-6):
        """PageRank using GAT embeddings"""
        if len(graph) == 0:
            raise NetworkXError('Input graph is empty!')

        W = nx.stochastic_graph(graph, weight='weight')
        N = W.number_of_nodes()

        # Initialize with embedding bias
        x = {d: self.embedding_bias(mode, t, d, embeddings, model) for d in W}

        # Normalize
        total = sum(x.values())
        if total > 0:
            x = {k: v / total for k, v in x.items()}

        biases = {**x}

        for iteration in range(max_iterations):
            x_last = x
            x = dict.fromkeys(x_last.keys(), 0)

            for d in x:
                # Random jump component
                random_jump = alpha * biases[d]

                # Walk component with embedding transition
                walk_component = (1 - alpha) * sum(
                    self.embedding_transition(mode, t, d, d_prime, W, embeddings, model) * x_last[d_prime]
                    for d_prime, _ in W.in_edges(d)
                )

                x[d] = random_jump + walk_component

            # Normalize
            total = sum(x.values())
            if total > 0:
                x = {k: v / total for k, v in x.items()}

            # Check convergence
            diff = sum([abs(x[n] - x_last[n]) for n in x])
            if diff < N * epsilon:
                # Clear caches (only if they exist and are cached)
                if hasattr(model, 'topic_docs_score') and hasattr(model.topic_docs_score, 'cache_clear'):
                    model.topic_docs_score.cache_clear()
                if hasattr(model, 'topic_doc_score') and hasattr(model.topic_doc_score, 'cache_clear'):
                    model.topic_doc_score.cache_clear()
                if hasattr(model, 'topic_authors_score') and hasattr(model.topic_authors_score, 'cache_clear'):
                    model.topic_authors_score.cache_clear()
                if hasattr(model, 'author_topic_scores') and hasattr(model.author_topic_scores, 'cache_clear'):
                    model.author_topic_scores.cache_clear()
                if hasattr(model, 'doc_authors_score') and hasattr(model.doc_authors_score, 'cache_clear'):
                    model.doc_authors_score.cache_clear()
                if hasattr(transition_star, 'cache_clear'):
                    transition_star.cache_clear()

                return t, x

        # Did not fully converge — return best approximation reached so far.
        import logging
        logging.getLogger(__name__).warning(
            f'[embedding_page_rank] Mode {mode}, topic {t}: did not converge in {max_iterations} '
            f'iterations (final diff={diff:.2e}). Returning best approximation.'
        )
        return t, x

    def compute_results(self, fresh=False):
        """Compute results for GAT-based modes"""
        logger = setup_logger()

        if self.mode == PRModes.FLOW_MATCHING:
            results_path = self.pagerank_output_path.joinpath(f'pageranks_{self.mode.value}.bin')
            paths_path = self.pagerank_output_path.joinpath(f'flow_matching_paths_{self.mode.value}.bin')
            if not fresh and flag.is_module_done(f'pr_{self.mode.value}') and results_path.exists() and paths_path.exists():
                logger.info(f"Loading results and paths for mode {self.mode}...")
                with open(results_path, 'rb') as f:
                    self._results = pickle.load(f)
                with open(paths_path, 'rb') as f:
                    self._paths = pickle.load(f)
                return
            elif not fresh and flag.is_module_done(f'pr_{self.mode.value}') and (not results_path.exists() or not paths_path.exists()):
                logger.warning(f"Flag exists but file missing for {self.mode}, regenerating...")

            database_setup()
            graph_data = self.generate_pyg_graph()
            self._embeddings = self.compute_gat_embeddings(graph_data, fresh=fresh)
            flow_model = self.train_flow_matching(graph_data, self._embeddings, fresh=fresh)
            self._paths = self.generate_paths(flow_model, self._embeddings, graph_data, fresh=fresh)
            
            # Also compute PageRank using embeddings (similar to GAT mode)
            all_results = defaultdict(dict)
            for cluster, lda in self.lda_results.items():
                # Check if cluster is done, but verify files actually exist
                cluster_flag_done = flag.is_module_done(f'pr_{self.mode.value}_cluster_{cluster}')
                if cluster_flag_done:
                    cluster_results = self._collect_cluster(cluster)
                    # Only skip if we actually have results
                    if cluster_results:
                        for doc_id, topic_scores in cluster_results.items():
                            all_results[doc_id].update(topic_scores)
                        continue
                    else:
                        # Flag exists but no files - reset the flag and recompute
                        logger.warning(f"Flag exists for cluster {cluster} but no files found, recomputing...")

                if remained_ids := self._remained_topic_ids(cluster):
                    TaskAssistant(
                        self.embedding_page_rank,
                        {'t': remained_ids},
                        f"Computing Flow Matching PageRank for cluster {cluster}...",
                        pr_callback,
                        callback_kwargs={'pr_values_output_path': self.pr_values_output_path, 'cluster': cluster, 'mode': self.mode.value},
                        task_kwargs={'graph': self.graph, 'model': lda, 'alpha': 0.15, 'mode': self.mode,
                                   'embeddings': self._embeddings},
                        chunk_size=1,
                        mp_core_frac=0.5
                    ).execute()

                    cluster_results = self._collect_cluster(cluster)
                    for doc_id, topic_scores in cluster_results.items():
                        all_results[doc_id].update(topic_scores)

                    flag.mark_module_done(f'pr_{self.mode.value}_cluster_{cluster}')

            self._results = all_results

            # Save results
            with open(results_path, 'wb') as f:
                pickle.dump(self._results, f)
            flag.mark_module_done(f'pr_{self.mode.value}')
        elif self.mode == PRModes.GAT:
            results_path = self.pagerank_output_path.joinpath(f'pageranks_{self.mode.value}.bin')
            if not fresh and flag.is_module_done(f'pr_{self.mode.value}') and results_path.exists():
                logger.info(f"Loading GAT PageRank results for mode {self.mode}...")
                with open(results_path, 'rb') as f:
                    self._results = pickle.load(f)
                return
            elif not fresh and flag.is_module_done(f'pr_{self.mode.value}') and not results_path.exists():
                logger.warning(f"Flag exists but file missing for {self.mode}, regenerating...")

            database_setup()
            graph_data = self.generate_pyg_graph()
            self._embeddings = self.compute_gat_embeddings(graph_data, fresh=fresh)

            # Now compute PageRank using embeddings
            all_results = defaultdict(dict)
            for cluster, lda in self.lda_results.items():
                # Check if cluster is done, but verify files actually exist
                cluster_flag_done = flag.is_module_done(f'pr_{self.mode.value}_cluster_{cluster}')
                if cluster_flag_done:
                    cluster_results = self._collect_cluster(cluster)
                    # Only skip if we actually have results
                    if cluster_results:
                        for doc_id, topic_scores in cluster_results.items():
                            all_results[doc_id].update(topic_scores)
                        continue
                    else:
                        # Flag exists but no files - reset the flag and recompute
                        logger.warning(f"Flag exists for cluster {cluster} but no files found, recomputing...")

                if remained_ids := self._remained_topic_ids(cluster):
                    TaskAssistant(
                        self.embedding_page_rank,
                        {'t': remained_ids},
                        f"Computing GAT PageRank for cluster {cluster}...",
                        pr_callback,
                        callback_kwargs={'pr_values_output_path': self.pr_values_output_path, 'cluster': cluster, 'mode': self.mode.value},
                        task_kwargs={'graph': self.graph, 'model': lda, 'alpha': 0.15, 'mode': self.mode,
                                   'embeddings': self._embeddings},
                        chunk_size=1,
                        mp_core_frac=0.5
                    ).execute()

                    cluster_results = self._collect_cluster(cluster)
                    for doc_id, topic_scores in cluster_results.items():
                        all_results[doc_id].update(topic_scores)

                    flag.mark_module_done(f'pr_{self.mode.value}_cluster_{cluster}')

            self._results = all_results

            # Save results
            with open(self.pagerank_output_path.joinpath(f'pageranks_{self.mode.value}.bin'), 'wb') as f:
                pickle.dump(self._results, f)
            flag.mark_module_done(f'pr_{self.mode.value}')
        elif self.mode in [PRModes.GAT_MTPR, PRModes.GAT_MTPR_CENTRALITY, PRModes.GAT_MTPR_EMBEDDING]:
            # GAT + Multi-Topic PageRank fusion
            results_path = self.pagerank_output_path.joinpath(f'pageranks_{self.mode.value}.bin')
            if not fresh and flag.is_module_done(f'pr_{self.mode.value}') and results_path.exists():
                logger.info(f"Loading GAT MTPR results for mode {self.mode}...")
                with open(results_path, 'rb') as f:
                    self._results = pickle.load(f)
                return
            elif not fresh and flag.is_module_done(f'pr_{self.mode.value}') and not results_path.exists():
                logger.warning(f"Flag exists but file missing for {self.mode}, regenerating...")

            database_setup()
            graph_data = self.generate_pyg_graph()
            self._embeddings = self.compute_gat_embeddings(graph_data, fresh=fresh)

            # Load classic PageRank results for fusion (use modified as base)
            classic_pr_path = self.pagerank_output_path.joinpath('pageranks_modified.bin')
            classic_results = None
            if classic_pr_path.exists():
                with open(classic_pr_path, 'rb') as f:
                    classic_results = pickle.load(f)
                logger.info(f"Loaded classic PageRank results for MTPR fusion")

            # Compute GAT PageRank
            all_results = defaultdict(dict)
            for cluster, lda in self.lda_results.items():
                # Check if cluster is done, but verify files actually exist
                cluster_flag_done = flag.is_module_done(f'pr_{self.mode.value}_cluster_{cluster}')
                if cluster_flag_done:
                    cluster_results = self._collect_cluster(cluster)
                    # Only skip if we actually have results
                    if cluster_results:
                        for doc_id, topic_scores in cluster_results.items():
                            all_results[doc_id].update(topic_scores)
                        continue
                    else:
                        # Flag exists but no files - reset the flag and recompute
                        logger.warning(f"Flag exists for cluster {cluster} but no files found, recomputing...")

                if remained_ids := self._remained_topic_ids(cluster):
                    TaskAssistant(
                        self.embedding_page_rank,
                        {'t': remained_ids},
                        f"Computing {self.mode.value} PageRank for cluster {cluster}...",
                        pr_callback,
                        callback_kwargs={'pr_values_output_path': self.pr_values_output_path, 'cluster': cluster, 'mode': self.mode.value},
                        task_kwargs={'graph': self.graph, 'model': lda, 'alpha': 0.15, 'mode': self.mode,
                                   'embeddings': self._embeddings},
                        chunk_size=1,
                        mp_core_frac=False
                    ).execute()

                    cluster_results = self._collect_cluster(cluster)
                    
                    # Fusion with classic PageRank if available
                    if classic_results and self.mode in [PRModes.GAT_MTPR, PRModes.GAT_MTPR_CENTRALITY]:
                        for doc_id, topic_scores in cluster_results.items():
                            # Weighted combination: 0.6 GAT + 0.4 Classic
                            fused_scores = {}
                            for topic_id, gat_score in topic_scores.items():
                                classic_score = classic_results.get(doc_id, {}).get(topic_id, 0)
                                fused_scores[topic_id] = 0.6 * gat_score + 0.4 * classic_score
                            cluster_results[doc_id] = fused_scores
                    
                    for doc_id, topic_scores in cluster_results.items():
                        all_results[doc_id].update(topic_scores)

                    flag.mark_module_done(f'pr_{self.mode.value}_cluster_{cluster}')

            self._results = all_results

            # Save results
            with open(self.pagerank_output_path.joinpath(f'pageranks_{self.mode.value}.bin'), 'wb') as f:
                pickle.dump(self._results, f)
            flag.mark_module_done(f'pr_{self.mode.value}')
        else:
            # All other GAT modes use similar approach to GAT
            results_path = self.pagerank_output_path.joinpath(f'pageranks_{self.mode.value}.bin')
            if not fresh and flag.is_module_done(f'pr_{self.mode.value}') and results_path.exists():
                logger.info(f"Loading GAT PageRank results for mode {self.mode}...")
                with open(results_path, 'rb') as f:
                    self._results = pickle.load(f)
                return
            elif not fresh and flag.is_module_done(f'pr_{self.mode.value}') and not results_path.exists():
                logger.warning(f"Flag exists but file missing for {self.mode}, regenerating...")

            database_setup()
            graph_data = self.generate_pyg_graph()
            self._embeddings = self.compute_gat_embeddings(graph_data, fresh=fresh)

            # Compute PageRank using embeddings with mode-specific bias/transition
            all_results = defaultdict(dict)
            for cluster, lda in self.lda_results.items():
                # Check if cluster is done, but verify files actually exist
                cluster_flag_done = flag.is_module_done(f'pr_{self.mode.value}_cluster_{cluster}')
                if cluster_flag_done:
                    cluster_results = self._collect_cluster(cluster)
                    # Only skip if we actually have results
                    if cluster_results:
                        for doc_id, topic_scores in cluster_results.items():
                            all_results[doc_id].update(topic_scores)
                        continue
                    else:
                        # Flag exists but no files - reset the flag and recompute
                        logger.warning(f"Flag exists for cluster {cluster} but no files found, recomputing...")

                if remained_ids := self._remained_topic_ids(cluster):
                    TaskAssistant(
                        self.embedding_page_rank,
                        {'t': remained_ids},
                        f"Computing {self.mode.value} PageRank for cluster {cluster}...",
                        pr_callback,
                        callback_kwargs={'pr_values_output_path': self.pr_values_output_path, 'cluster': cluster, 'mode': self.mode.value},
                        task_kwargs={'graph': self.graph, 'model': lda, 'alpha': 0.15, 'mode': self.mode,
                                   'embeddings': self._embeddings},
                        chunk_size=1,
                        mp_core_frac=0.5
                    ).execute()

                    cluster_results = self._collect_cluster(cluster)
                    for doc_id, topic_scores in cluster_results.items():
                        all_results[doc_id].update(topic_scores)

                    flag.mark_module_done(f'pr_{self.mode.value}_cluster_{cluster}')

            self._results = all_results

            # Save results
            with open(self.pagerank_output_path.joinpath(f'pageranks_{self.mode.value}.bin'), 'wb') as f:
                pickle.dump(self._results, f)
            flag.mark_module_done(f'pr_{self.mode.value}')

    def _remained_topic_ids(self, cluster):
        """Find topic IDs that haven't been processed yet"""
        # Check both file existence and flag
        completed_topics = set()
        for t in range(self.no_topics):
            flag_name = f'pr_{self.mode.value}_topic_{cluster}_{t}'
            file_path = self.pr_values_output_path.joinpath(f'topic-cluster{cluster}-{t}.bin')
            if flag.is_module_done(flag_name) and file_path.exists():
                completed_topics.add(t)
        return list(set(range(self.no_topics)).difference(completed_topics))

    def _collect_cluster(self, cluster):
        """Collect PageRank results for a specific cluster"""
        page_ranks = defaultdict(dict)
        matcher = utils.REMatcher(f'topic-cluster{cluster}-(\\d+).bin')
        for pr_file in self.pr_values_output_path.glob('*.bin'):
            if res := matcher.match(pr_file.name):
                with pr_file.open('rb') as f:
                    tmp = pickle.load(f)
                for idx in tmp:
                    page_ranks[idx].update(tmp[idx])
        return page_ranks

    def results(self, agify=True):
        """Get results"""
        self.compute_results()
        if self.mode == PRModes.FLOW_MATCHING:
            # Return PageRank results (paths are stored separately in _paths)
            return self._results
        return self._results


@lru_cache(maxsize=None)
def _out_neighbor_topic_sum(t: int, d_prime: int, graph: nx.DiGraph, model: GensimLDA) -> float:
    """Sum_{d'' in link_out(d')} P(t|d'') -- a plain sum, not an average."""
    return sum(model.topic_doc_score(dd, t) for dd in graph.successors(d_prime))


@lru_cache(maxsize=None)
def transition_star(mode: PRModes, t: int, d: int, d_prime: int,
                    graph: nx.DiGraph, model: GensimLDA) -> float:
    """
    Unnormalised topic-aware edge weight T*(t,d,d') for d_prime -> d.

    Jardine (2014, p.60) / proposal Eq. 4-4 (THEMED baseline):
        T*(t,d,d') = sqrt( [P(t|d') / sum_D P(t|d'')] * [P(t|d) / sum_{d'' in link_out(d')} P(t|d'')] )

    ``normalized_transition_weights`` then normalizes T* over link_in(d) (the
    set of papers that cite d), per Jardine's Transition(t,d,d') / proposal
    Eq. 4-6 -- not over d''s own successors.

    MTPR modes multiply this base term by a normalised author-credibility
    score (proposal Eq. 4-7, minus the not-yet-implemented journal/venue term
    AS_jt). Centrality-transition variants multiply it by their normalised
    centrality score instead, matching their column in the variant table.

    IMPORTANT: that multiplier must be evaluated at d_prime (the source/citer),
    not d (the target) -- normalized_transition_weights() normalizes T* over
    d_prime for a FIXED d (Jardine's per-target axis), so any factor that only
    depends on d is a constant across that normalization and is mathematically
    inert: it either cancels out exactly (reducing to plain THEMED behaviour)
    or, when it happens to be 0, collapses the whole sum to the uniform
    fallback. Verified empirically 2026-08-22: comparing MODIFIED_TRANSITION's
    weights (then evaluated at d) against THEMED across 15 sample targets gave
    9 exact matches and 6 uniform-fallback cases -- zero cases where the author
    term actually differentiated the ranking. Evaluating at d_prime instead
    makes the weight vary across the very set being normalized, so it can
    actually shift the distribution.
    """
    total_topic_score = model.topic_docs_score(t)
    out_sum = _out_neighbor_topic_sum(t, d_prime, graph, model)
    if total_topic_score <= 0 or out_sum <= 0:
        return 0.0

    term1 = model.topic_doc_score(d_prime, t) / total_topic_score
    term2 = model.topic_doc_score(d, t) / out_sum
    base_tstar = sqrt(max(0.0, term1 * term2))

    if mode == PRModes.THEMED:
        return base_tstar

    if mode == PRModes.MODIFIED:
        # Table 1: modified's transition is plain T* -- AS only applies to its bias.
        return base_tstar

    if mode == PRModes.MODIFIED_TRANSITION:
        author_score = model.doc_authors_score(t, d_prime) / (model.topic_authors_score(t) or 1)
        return base_tstar * author_score

    if mode == PRModes.D_MODIFIED_TRANSITION:
        degree = model.topic_doc_degree_centrality_score(graph, t, d_prime)
        total_degree = model.topic_degree_centrality_score(graph, t) or 1
        return base_tstar * (degree / total_degree)

    if mode == PRModes.CL_MODIFIED_TRANSITION:
        closeness = model.topic_doc_closeness_centrality_score(graph, t, d_prime)
        total_closeness = model.topic_closeness_centrality_score(graph, t) or 1
        return base_tstar * (closeness / total_closeness)

    if mode == PRModes.BW_MODIFIED_TRANSITION:
        betweenness = model.topic_doc_betweenness_centrality_score(graph, t, d_prime)
        total_betweenness = model.topic_betweenness_centrality_score(graph, t) or 1
        return base_tstar * (betweenness / total_betweenness)

    if mode == PRModes.KZ_MODIFIED_TRANSITION:
        katz = model.topic_doc_katz_centrality_score(graph, t, d_prime)
        total_katz = model.topic_katz_centrality_score(graph, t) or 1
        return base_tstar * (katz / total_katz)

    return base_tstar


@lru_cache(maxsize=None)
def normalized_transition_weights(mode: PRModes, t: int, d: int,
                                  graph: nx.DiGraph, model: GensimLDA) -> Dict[int, float]:
    """Return cached stochastic topic-aware transition weights into one target node d.

    Jardine (2014, p.60) / proposal Eq. 4-6: Transition(t,d,d') normalizes
    T*(t,d,d') over link_in(d) -- the papers that cite d -- so that, for a
    fixed d, the weights across its citers d' sum to 1.
    """
    citers = tuple(graph.predecessors(d))
    if not citers:
        return {}

    raw_weights = {
        d_prime: transition_star(mode, t, d, d_prime, graph, model)
        for d_prime in citers
    }
    total_weight = sum(raw_weights.values())

    if total_weight <= 0:
        uniform_weight = 1.0 / len(citers)
        return {d_prime: uniform_weight for d_prime in citers}

    return {d_prime: weight / total_weight for d_prime, weight in raw_weights.items()}


def page_rank(mode: PRModes, t: int, graph: nx.DiGraph, model: GensimLDA,
              alpha: float, cluster: int = None, max_iterations: int = 100,
              epsilon: float = 1.0e-6):
    """
    Custom PageRank implementation with topic awareness.

    OPTIMIZATIONS applied:
    - Katz centrality is pre-computed ONCE per topic (not once per document).
      For N nodes, this reduces Katz cost from O(N × V × E) to O(V × E).
    - Closeness & betweenness centrality are batch-cached in lda.py;
      per-doc lookups are now O(1) dict reads from a disk-cached batch result.

    Args:
        mode: PageRank mode
        t: Topic ID
        graph: Citation graph
        model: LDA model
        alpha: Damping factor (typically 0.15)
        cluster: Cluster ID (for logging)
        max_iterations: Maximum iterations
        epsilon: Convergence threshold

    Returns:
        Tuple of (topic_id, PageRank scores dict)
    """
    if len(graph) == 0:
        raise NetworkXError('Input graph is empty!')

    W = nx.stochastic_graph(graph, weight='weight')
    N = W.number_of_nodes()

    # Katz centrality for KZ_MODIFIED*/HYBRID is now computed per topic-subgraph
    # and batch-cached inside model.topic_katz_centrality_scores_batch() (called
    # via bias()), the same way closeness/betweenness already are -- no separate
    # whole-graph pre-computation needed here anymore.
    x = {d: PaperRankClassic.bias(mode, t, d, W, model) for d in W}

    # Normalize
    total = sum(x.values())
    if total > 0:
        x = {k: v / total for k, v in x.items()}
    else:
        print(f"Warning: Topic {t} bias total=0, using uniform personalization 1/N={1/N:.8f}")
        x = {d: 1.0 / N for d in W}

    biases = {**x}

    for iteration in range(max_iterations):
        x_last = x
        x = dict.fromkeys(x_last.keys(), 0)
        dangling_mass = sum(x_last[d_prime] for d_prime in W if W.out_degree(d_prime) == 0)

        for d in x:
            # Random jump component
            random_jump = alpha * biases[d]

            # Return dangling-node probability mass to the personalization vector.
            dangling_component = (1 - alpha) * dangling_mass * biases[d]

            # Walk component
            walk_component = (1 - alpha) * sum(
                PaperRankClassic.transition(mode, t, d, d_prime, W, model) * x_last[d_prime]
                for d_prime, _ in W.in_edges(d)
            )

            x[d] = random_jump + dangling_component + walk_component

        # CORRECTION (2026-08-22, supersedes the claim that used to be here):
        # Jardine's per-TARGET transition normalization (Transition(t,d,d')
        # sums to 1 over d' for fixed d) does NOT give the walk operator a
        # bounded L1 (max-column-sum) norm the way standard per-SOURCE
        # row-normalized PageRank does -- that only bounds the max *row* sum.
        # The iteration is therefore not guaranteed to be a contraction, and
        # it isn't just theoretically: PRModes.BW_MODIFIED / topic 142 (a
        # sparse ~647-node topic subgraph) was verified to diverge -- final
        # diff=1.05e3 after 100 iterations, sum(x) blew up to ~13,582 instead
        # of staying near 1. Renormalizing x to sum to 1 every iteration (the
        # same thing standard PageRank implementations do) fixes this: it's
        # just a division by a positive constant, so it changes no relative
        # ranking between documents, only prevents unbounded numerical growth.
        total_x = sum(x.values())
        if total_x > 0:
            x = {k: v / total_x for k, v in x.items()}

        # Check convergence
        diff = sum([abs(x[n] - x_last[n]) for n in x])
        if diff < N * epsilon:
            # Clear lru_cache entries that are per-topic
            if hasattr(model, 'topic_docs_score') and hasattr(model.topic_docs_score, 'cache_clear'):
                model.topic_docs_score.cache_clear()
            if hasattr(model, 'topic_doc_score') and hasattr(model.topic_doc_score, 'cache_clear'):
                model.topic_doc_score.cache_clear()
            if hasattr(model, 'topic_authors_score') and hasattr(model.topic_authors_score, 'cache_clear'):
                model.topic_authors_score.cache_clear()
            if hasattr(model, 'author_topic_scores') and hasattr(model.author_topic_scores, 'cache_clear'):
                model.author_topic_scores.cache_clear()
            if hasattr(model, 'doc_authors_score') and hasattr(model.doc_authors_score, 'cache_clear'):
                model.doc_authors_score.cache_clear()
            if hasattr(transition_star, 'cache_clear'):
                transition_star.cache_clear()
            if hasattr(normalized_transition_weights, 'cache_clear'):
                normalized_transition_weights.cache_clear()
            return t, x

    # Did not fully converge — return the best approximation reached so far.
    # This is the correct behavior for research: a nearly-converged result is
    # far more useful than a hard crash. The residual diff at this point is
    # typically very small (< 1e-4), well within acceptable research tolerance.
    import logging
    logging.getLogger(__name__).warning(
        f'[page_rank] Mode {mode}, topic {t}: did not converge in {max_iterations} '
        f'iterations (final diff={diff:.2e}). Returning best approximation.'
    )
    return t, x


def setup(log_msg, lda_results, topic_model: TopicModelModes, no_topics: int, pr_mode: PRModes,
           bertopic_no_topics: int = 0,
          fresh: bool = False, weights: str = '0.25,0.25,0.25,0.25'):
    """
    Setup PageRank system.

    Args:
        topic_model: Topic modeling approach (LDA or BERTopic)
        no_topics: Number of topics
        pr_mode: PageRank mode
        bertopic_no_topics: Number of topics for BERTopic
        fresh: Whether to recompute
        weights: Weights for hybrid mode (comma-separated)

    Returns:
        PageRank instance
    """
    logger = setup_logger()
    logger.info(log_msg)

    if topic_model == TopicModelModes.LDA:
        logger.info(f"[PAGERANK SETUP] mode: {pr_mode}, LDA results sample: {dict(list(lda_results.items())[:2]) if lda_results else 'EMPTY'}")

        if PRModes.is_gat_mode(pr_mode):
            pr = PaperRankGAT(lda_results, pr_mode, no_topics)
        else:
            pr = PaperRankClassic(lda_results, pr_mode, no_topics)

    elif topic_model == TopicModelModes.BERTOPIC:
        from components.bertopic import setup as bertopic_setup, PaperRankBERTopic
        effective_topics = bertopic_no_topics if bertopic_no_topics > 0 else no_topics
        logger.info(f"[PAGERANK SETUP] mode: {pr_mode}, BERTopic, effective_topics={effective_topics}")
        topic_info_df, paper_topics_df = bertopic_setup(no_topics=effective_topics, fresh=fresh)
        pr = PaperRankBERTopic(topic_info_df, paper_topics_df, pr_mode, effective_topics)

    else:
        raise ValueError(f"Unsupported topic_model: {topic_model}")

    # Handle hybrid mode weights (will be used during compute_results)
    if pr_mode == PRModes.HYBRID and isinstance(pr, PaperRankClassic):
        pr._hybrid_weights = {
            k: float(v)
            for k, v in zip(['degree', 'closeness', 'betweenness', 'katz'],
                            weights.split(','))
        }

    pr.compute_results(fresh)
    return pr


def define_subparser(subparser):
    """Define command-line subparser for PageRank"""
    pagerank = subparser.add_parser(subparser_name := 'pagerank')

    pagerank.add_argument(
        '--topic-model',
        dest='topic_model',
        type=TopicModelModes,
        choices=list(TopicModelModes),
        help='Topic model type',
        required=False,
        default=TopicModelModes.LDA
    )

    pagerank.add_argument(
        '--no-topics',
        dest='no_topics',
        type=int,
        help='Number of topics',
        required=True
    )

    pagerank.add_argument(
        '--pr-mode',
        dest='pr_mode',
        type=PRModes,
        choices=list(PRModes),
        help='PageRank mode',
        required=True
    )

    pagerank.add_argument(
        '--lda-no-topics',
        dest='lda_no_topics',
        type=int,
        default=20,
        help='Number of topics for LDA'
    )

    pagerank.add_argument(
        '--bertopic-no-topics',
        dest='bertopic_no_topics',
        type=int,
        default=5,
        help='Number of topics for BERTopic'
    )

    pagerank.add_argument(
        '-v',
        dest='verbose',
        action='store_true',
        help='Activate verbose mode'
    )

    pagerank.add_argument(
        '--fresh',
        dest='fresh',
        action='store_true',
        help='Force recomputation'
    )

    pagerank.add_argument(
        '--weights',
        type=str,
        default='0.25,0.25,0.25,0.25',
        help='Weights for hybrid mode (degree,closeness,betweenness,katz)'
    )

    return subparser_name
