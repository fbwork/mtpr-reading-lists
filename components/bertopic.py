import ast
import logging
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, List

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bertopic import BERTopic
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from utils import flag, TaskAssistant
from .base_pagerank import PageRankBase
from .database import setup as database_setup, Paper, PaperCitation
from .lda import TopicScoreModel
from .pagerank import page_rank
from .prmodes import PRModes

bertopic_output_path = Path(__file__).parent.joinpath('output', 'bertopic')
bertopic_pr_output_path = Path(__file__).parent.joinpath('output', 'pr', 'bertopic')
bertopic_log_path = Path(os.getenv('LOG_PATH', default='logs')).joinpath('bertopic.log')


def setup_logger():
    """Initialize logger with file output."""
    os.makedirs(bertopic_log_path.parent, exist_ok=True)
    logging.basicConfig(
        filename=bertopic_log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()


def pr_callback(result, bar, pr_values_output_path: Path, mode: str = 'normal'):
    t, pg_rank = result
    file_path = pr_values_output_path.joinpath(f'topic-{t}.bin')
    with file_path.open('wb') as f:
        pickle.dump({d: {t: pg_rank[d]} for d in pg_rank}, f)
    flag.mark_module_done(f'pr_{mode}_topic_{t}')
    bar.update()


def evaluate_flow_matching_paths(paths: list, relevant_papers: set, k: int = 10) -> Dict[str, float]:
    """
    Evaluate Flow Matching paths for precision, recall, and F1 score.

    :param paths: List of paper IDs in the predicted paths
    :param relevant_papers: Set of relevant paper IDs
    :param k: Number of top papers to evaluate
    :return: Dictionary with precision, recall, and F1 score
    """
    relevant_set = set(paths[:k]).intersection(relevant_papers)
    precision = len(relevant_set) / k if k > 0 else 0
    recall = len(relevant_set) / len(relevant_papers) if relevant_papers else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def setup(no_topics: int, fresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import time
    start_time = time.perf_counter()
    logger = setup_logger()
    topic_model_path = bertopic_output_path.joinpath(f'bertopic_model_{no_topics}.bin')
    paper_topics_path = bertopic_output_path.joinpath(f'paper_topics_{no_topics}.csv')
    topic_info_path = bertopic_output_path.joinpath(f'topic_info_{no_topics}.csv')
    docs_path = bertopic_output_path.joinpath(f'documents_{no_topics}.bin')
    topics_probs_path = bertopic_output_path.joinpath(f'topics_probs_{no_topics}.bin')

    # Always get current paper count for consistency
    current_paper_ids = [p.id for p in Paper.select().iterator()]
    current_n_papers = len(current_paper_ids)
    logger.info(f"Current number of papers in database: {current_n_papers}")

    # Always initialize topic_model
    topic_model = BERTopic(nr_topics=no_topics)
    fitted = False

    # Try to load model if not fresh
    if not fresh and topic_model_path.is_file():
        try:
            loaded_model = BERTopic.load(str(topic_model_path))
            # Check if loaded model is fitted
            if hasattr(loaded_model, 'topics_') and loaded_model.topics_ is not None:
                topic_model = loaded_model
                fitted = True
                logger.info("Loaded fitted cached BERTopic model.")
            else:
                logger.warning("Loaded model is not fitted, will fit new.")
        except Exception as e:
            logger.warning(f"Failed to load cached model: {e}. Will fit new model.")

    # Load or fetch documents
    try:
        if not fresh and docs_path.is_file() and flag.is_module_done('bertopic_fetch_docs'):
            logger.info("Loading cached documents...")
            with open(docs_path, 'rb') as f:
                documents = pickle.load(f)
            if len(documents) != current_n_papers:
                logger.warning(f"Cached documents length {len(documents)} != current papers {current_n_papers}. Refetching.")
                raise ValueError("Cached documents length mismatch.")
        else:
            documents = [paper.body for paper in Paper.select().iterator()]
            if len(documents) != current_n_papers:
                logger.error(f"Fetched {len(documents)} documents, but expected {current_n_papers} papers.")
                raise ValueError("Fetched documents length mismatch.")
            if not documents:
                raise ValueError("No documents found in Paper database.")
            with open(docs_path, 'wb') as f:
                pickle.dump(documents, f)
            flag.mark_module_done('bertopic_fetch_docs')
            logger.info(f"Fetched and cached {len(documents)} documents.")
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise FileNotFoundError("Failed to fetch documents from Paper database.")

    # Load or compute topics and probs
    topics = None
    probs = None
    if not fresh and topics_probs_path.is_file() and flag.is_module_done('bertopic_fit_transform') and fitted:
        logger.info("Loading cached topics and probabilities...")
        with open(topics_probs_path, 'rb') as f:
            topics, probs = pickle.load(f)
        if len(topics) != current_n_papers or (hasattr(probs, '__len__') and len(probs) != current_n_papers):
            logger.warning(f"Cached topics/probs lengths {len(topics)}/{len(probs) if hasattr(probs, '__len__') else 'N/A'} != current papers {current_n_papers}. Recomputing.")
            topics = None
            probs = None
        else:
            logger.info(f"Loaded cached topics shape/len: {len(topics) if hasattr(topics, '__len__') else 'N/A'}")
            logger.info(f"Loaded cached probs type: {type(probs)}, len/shape: {probs.shape if hasattr(probs, 'shape') else len(probs) if hasattr(probs, '__len__') else 'scalar?'}")
            for i in range(min(5, len(probs) if hasattr(probs, '__len__') else 0)):
                prob = probs[i] if hasattr(probs, '__getitem__') else probs
                logger.info(f"Sample prob {i}: type={type(prob)}, value={prob}")
    if topics is None or probs is None:
        logger.info("Computing topics and probabilities...")
        topics, probs = topic_model.fit_transform(documents)
        logger.info(f"Computed fresh topics/probs: len(topics)={len(topics)}, probs type={type(probs)}, len/probs={probs.shape if hasattr(probs, 'shape') else len(probs) if hasattr(probs, '__len__') else 'N/A'}")
        fitted = True
        with open(topics_probs_path, 'wb') as f:
            pickle.dump((topics, probs), f)
        flag.mark_module_done('bertopic_fit_transform')

    # Ensure topics match current paper count
    if len(topics) != current_n_papers:
        logger.warning(f"Topics length {len(topics)} != current papers {current_n_papers}. Padding/truncating topics.")
        if len(topics) < current_n_papers:
            topics += [-1] * (current_n_papers - len(topics))  # Pad with outlier topic
        else:
            topics = topics[:current_n_papers]

    print(f"DEBUG: len(current_paper_ids) = {len(current_paper_ids)}")
    print(f"DEBUG: len(topics) = {len(topics)}")
    if hasattr(probs, '__len__'):
        print(f"DEBUG: len(probs) = {len(probs)}")
    else:
        print(f"DEBUG: probs is not len-able, type={type(probs)}")
    if hasattr(probs, '__getitem__'):
        print(f"DEBUG: type(probs[0]) = {type(probs[0])}")
        print(f"DEBUG: probs[0] = {probs[0]}")

    # Process probs to match length and ensure all are lists of exactly no_topics length
    probs_list = []
    expected_len = current_n_papers
    default_prob = [0.0] * no_topics

    # Handle if probs is 1D array of scalars (common issue)
    if hasattr(probs, 'shape') and len(probs.shape) == 1:
        # probs is 1D, treat each as scalar probability for the assigned topic, but for consistency, set full vector with 1.0 at topic position
        for i in range(expected_len):
            topic = topics[i]
            prob_vector = [0.0] * no_topics
            if 0 <= topic < no_topics:
                prob_vector[topic] = 1.0  # Or use the scalar if available, but since scalar is overall prob, perhaps set to uniform or default
            probs_list.append(prob_vector)
        logger.info("Detected 1D probs array; assigned one-hot vectors based on topics.")
    else:
        # General case
        if hasattr(probs, '__len__') and len(probs) != expected_len:
            logger.warning(f"Probs length {len(probs)} != expected {expected_len}. Adjusting.")
            if len(probs) < expected_len:
                probs = list(probs) + [default_prob] * (expected_len - len(probs))
            else:
                probs = list(probs)[:expected_len]
        else:
            probs = list(probs) if hasattr(probs, '__iter__') else [default_prob] * expected_len

        for i, prob in enumerate(probs):
            prob_list = []
            try:
                if hasattr(prob, '__iter__') and not isinstance(prob, (str, bytes)):
                    prob_list = list(prob)
                else:
                    # Scalar or non-iterable
                    logger.warning(f"Non-iterable prob at index {i}: type={type(prob)}, value={prob}. Using default vector.")
                    prob_list = default_prob.copy()
            except Exception as e:
                logger.error(f"Error converting prob at index {i}: {e}. Using default.")
                prob_list = default_prob.copy()

            # Ensure length is exactly no_topics
            if len(prob_list) < no_topics:
                prob_list += [0.0] * (no_topics - len(prob_list))
            elif len(prob_list) > no_topics:
                prob_list = prob_list[:no_topics]

            probs_list.append(prob_list)

    # Final check and force uniform length for all
    for i in range(len(probs_list)):
        if len(probs_list[i]) != no_topics:
            logger.warning(f"Forcing uniform length for probs_list[{i}]: {len(probs_list[i])} -> {no_topics}")
            probs_list[i] = default_prob.copy()

    print(f"DEBUG: len(probs_list) = {len(probs_list)}")
    print(f"DEBUG: Sample lengths: {[len(p) for p in probs_list[:3]]}")

    # Now create DataFrame - all should be uniform
    paper_topics_df = pd.DataFrame({
        'paper_id': current_paper_ids,
        'topic': topics,
        'probabilities': probs_list
    })
    logger.info(f"Created DataFrame with {len(paper_topics_df)} rows, all probabilities length {no_topics}")

    # Get topic_info only if model is fitted
    if fitted and topic_model is not None:
        try:
            topic_info_df = topic_model.get_topic_info()
        except ValueError as e:
            if "not fitted" in str(e):
                logger.warning("Model not fitted, generating topic_info from topics.")
                from collections import Counter
                topic_counts = Counter(t for t in topics if t >= 0)
                topic_info_df = pd.DataFrame({
                    'Topic': list(topic_counts.keys()),
                    'Count': list(topic_counts.values()),
                    'Name': [f"Topic {t}" for t in topic_counts.keys()]
                })
            else:
                raise
    else:
        logger.warning("Model not fitted, generating topic_info from topics.")
        from collections import Counter
        topic_counts = Counter(t for t in topics if t >= 0)
        topic_info_df = pd.DataFrame({
            'Topic': list(topic_counts.keys()),
            'Count': list(topic_counts.values()),
            'Name': [f"Topic {t}" for t in topic_counts.keys()]
        })

    try:
        os.makedirs(bertopic_output_path, exist_ok=True)
        if topic_model is not None:
            topic_model.save(str(topic_model_path))
        paper_topics_df.to_csv(paper_topics_path, index=False)
        topic_info_df.to_csv(topic_info_path, index=False)
        flag.mark_module_done('bertopic_save')
        logger.info("BERTopic model and data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving BERTopic model or data: {e}")
        raise ValueError(f"Failed to save BERTopic model or data: {e}")

    end_time = time.perf_counter()
    print(f"BERTopic setup time: {end_time - start_time:.2f} seconds")
    logger.info(f"BERTopic setup completed in {end_time - start_time:.2f} seconds")

    return topic_info_df, paper_topics_df


class FlowMatchingModel(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels + 1, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, in_channels)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        x_t = torch.cat([x, t], dim=1)
        return self.mlp(x_t)


class BERTopicScoreModel(TopicScoreModel):
    """BERTopic-backed implementation of TopicScoreModel.

    Wraps BERTopic's paper-topic probability matrix (paper_topics_df) behind
    the same get_doc_topics()/num_docs/cache_root interface GensimLDA uses.
    This lets PaperRankBERTopic feed the same page_rank() -- and therefore the
    same mode-specific bias()/transition_star() MTPR logic (centrality,
    author-credibility, per-target transition normalization) -- as
    PaperRankClassic uses for LDA. Previously PaperRankBERTopic ignored
    ``mode`` entirely and always ran plain personalized PageRank regardless of
    which of the 19 variants was requested; every "BERTopic <mode>" result
    reported before this fix was actually just that one generic computation.
    """

    def __init__(self, paper_topics_df: pd.DataFrame, no_topics: int, verbose: bool = False):
        self.verbose = verbose
        self.no_topics = no_topics
        self.lda_no_topics = no_topics  # shared name used by TopicScoreModel.author_scores
        self._doc_topics: Dict[int, Dict[int, float]] = {}
        for _, row in paper_topics_df.iterrows():
            doc_id = row['paper_id'] - 1
            probs = (ast.literal_eval(row['probabilities']) if isinstance(row['probabilities'], str)
                     else row['probabilities'])
            if isinstance(probs, (list, tuple)) and len(probs) == no_topics:
                self._doc_topics[doc_id] = {t: float(p) for t, p in enumerate(probs) if p > 0}
        self._num_docs = Paper.select().count()

    @property
    def num_docs(self) -> int:
        return self._num_docs

    @property
    def cache_root(self) -> Path:
        return bertopic_output_path.joinpath('scores', str(self.no_topics))

    @lru_cache(maxsize=None)
    def get_doc_topics(self, doc_id: int, minimum_probability: float = 0.0) -> Dict[int, float]:
        topics = self._doc_topics.get(doc_id, {})
        if minimum_probability > 0.0:
            return {t: p for t, p in topics.items() if p >= minimum_probability}
        return topics


class PaperRankBERTopic(PageRankBase):
    def __init__(self, topic_info_df: pd.DataFrame, paper_topics_df: pd.DataFrame, mode: PRModes, no_topics: int):
        """Initialize PaperRankBERTopic with topic data and mode."""
        super().__init__(mode, no_topics)
        self.topic_info_df = topic_info_df
        self.paper_topics_df = paper_topics_df
        self.pagerank_output_path = bertopic_pr_output_path.joinpath(str(self.no_topics), str(self.no_topics))
        self.pr_values_output_path = self.pagerank_output_path.joinpath('values', self.mode.value)
        self.pr_values_output_path.mkdir(parents=True, exist_ok=True)
        self._results = None
        self._agified_results = None
        self._paths = None

    def evaluate_paths(self, paths: list, query: str, k: int = 10) -> Dict[str, float]:
        """
        Evaluate the generated paths based on relevance to the query.

        :param paths: List of paper IDs in the predicted paths
        :param query: Search query for relevance evaluation
        :param k: Number of top papers to evaluate
        :return: Dictionary with precision, recall, and F1 score
        """
        logger = setup_logger()
        try:
            relevant_papers = set(p.id - 1 for p in Paper.select().where(Paper.title.contains(query)).limit(k))
            metrics = evaluate_flow_matching_paths(paths, relevant_papers, k)
            logger.info(f"Evaluation metrics for query '{query}': {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating paths for query '{query}': {e}")
            return {'precision': 0, 'recall': 0, 'f1': 0}

    def compute_gat_embeddings(self, graph_data: Data, fresh: bool = False) -> torch.Tensor:
        import time
        start_time = time.perf_counter()
        logger = setup_logger()
        cache_path = self.pagerank_output_path.joinpath('gat_embeddings.bin')
        if not fresh and cache_path.is_file() and flag.is_module_done(f'pr_{self.mode.value}_gat_embeddings'):
            logger.info(f"Loading GAT embeddings for mode {self.mode}...")
            end_time = time.perf_counter()
            print(f"GAT embeddings load time: {end_time - start_time:.2f} seconds")
            return torch.load(cache_path)

        class GATModel(torch.nn.Module):
            def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4):
                super().__init__()
                self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
                self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

            def forward(self, data: Data) -> torch.Tensor:
                x, edge_index = data.x, data.edge_index
                x = F.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

        try:
            if torch.cuda.is_available():
                initial_mem = torch.cuda.memory_allocated() / 1024**2  # MB
                max_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
                print(f"GPU memory before GAT: allocated {initial_mem:.2f} MB, max {max_mem:.2f} MB")
                torch.cuda.empty_cache()
            model = GATModel(
                in_channels=graph_data.x.shape[1],
                hidden_channels=64,
                out_channels=32
            ).cuda()
            model.eval()
            with torch.no_grad(), autocast():
                embeddings = model(graph_data.cuda())
            embeddings = embeddings.cpu()
            if torch.cuda.is_available():
                after_mem = torch.cuda.memory_allocated() / 1024**2
                max_after = torch.cuda.max_memory_allocated() / 1024**2
                print(f"GPU memory after GAT: allocated {after_mem:.2f} MB, max {max_after:.2f} MB")
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

    def train_flow_matching(self, graph_data: Data, embeddings: torch.Tensor, fresh: bool = False,
                            num_epochs: int = 200, batch_size: int = 32) -> torch.nn.Module:
        import time
        start_time = time.perf_counter()
        logger = setup_logger()
        cache_path = self.pagerank_output_path.joinpath('flow_matching_model.bin')
        if not fresh and cache_path.is_file() and flag.is_module_done(f'pr_{self.mode.value}_flow_matching'):
            logger.info(f"Loading Flow Matching model for mode {self.mode}...")
            end_time = time.perf_counter()
            print(f"Flow Matching model load time: {end_time - start_time:.2f} seconds")
            return torch.load(cache_path)

        model = FlowMatchingModel(in_channels=embeddings.shape[1], hidden_channels=64).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler()
        try:
            if torch.cuda.is_available():
                initial_mem = torch.cuda.memory_allocated() / 1024**2
                max_mem = torch.cuda.max_memory_allocated() / 1024**2
                print(f"GPU memory before Flow Matching training: allocated {initial_mem:.2f} MB, max {max_mem:.2f} MB")
                torch.cuda.empty_cache()
            paths = [(p.citing_paper_id - 1, p.cited_paper_id - 1) for p in PaperCitation.select().iterator()]
            edge_index = torch.tensor(paths, dtype=torch.long).t().contiguous().cuda()
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                indices = torch.randperm(embeddings.shape[0])[:min(batch_size, embeddings.shape[0])]
                batch_embeddings = embeddings[indices].cuda()
                batch_edge_index = edge_index[:, torch.isin(edge_index[0], indices) & torch.isin(edge_index[1], indices)]
                t = torch.rand(batch_embeddings.shape[0], 1).cuda()
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
            torch.save(model, cache_path)
            flag.mark_module_done(f'pr_{self.mode.value}_flow_matching')
            end_time = time.perf_counter()
            print(f"Flow Matching training time: {end_time - start_time:.2f} seconds")
            logger.info(f"Flow Matching model trained in {end_time - start_time:.2f} seconds")
            return model
        except RuntimeError as e:
            logger.error(f"Error training Flow Matching model: {e}")
            raise RuntimeError("Failed to train Flow Matching model, possibly due to CUDA memory issues.")

    def generate_paths(self, flow_model: torch.nn.Module, embeddings: torch.Tensor, graph_data: Data,
                       fresh: bool = False, num_paths: int = 10, path_length: int = 3) -> List[List[int]]:
        import time
        start_time = time.perf_counter()
        logger = setup_logger()
        cache_path = self.pagerank_output_path.joinpath('flow_matching_paths.bin')
        if not fresh and cache_path.is_file() and flag.is_module_done(f'pr_{self.mode.value}_paths'):
            logger.info(f"Loading Flow Matching paths for mode {self.mode}...")
            end_time = time.perf_counter()
            print(f"Flow Matching paths load time: {end_time - start_time:.2f} seconds")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        try:
            if torch.cuda.is_available():
                initial_mem = torch.cuda.memory_allocated() / 1024**2
                max_mem = torch.cuda.max_memory_allocated() / 1024**2
                print(f"GPU memory before path generation: allocated {initial_mem:.2f} MB, max {max_mem:.2f} MB")
                torch.cuda.empty_cache()
            with torch.no_grad():
                x = torch.randn_like(embeddings).cuda()
                for t in torch.linspace(0, 1, 100):
                    x = x + flow_model(x, t, graph_data.edge_index.cuda()) * 0.01
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
                    current_node = neighbors[torch.multinomial(probabilities, 1).item()]
                if path:
                    paths.append(path)
            if torch.cuda.is_available():
                after_mem = torch.cuda.memory_allocated() / 1024**2
                max_after = torch.cuda.max_memory_allocated() / 1024**2
                print(f"GPU memory after path generation: allocated {after_mem:.2f} MB, max {max_after:.2f} MB")
                torch.cuda.reset_peak_memory_stats()
            with open(cache_path, 'wb') as f:
                pickle.dump(paths, f)
            flag.mark_module_done(f'pr_{self.mode.value}_paths')
            end_time = time.perf_counter()
            print(f"Flow Matching path generation time: {end_time - start_time:.2f} seconds")
            logger.info(f"Paths generated in {end_time - start_time:.2f} seconds")
            return paths
        except Exception as e:
            logger.error(f"Error generating paths: {e}")
            return []

    def compute_results(self, fresh: bool = False):
        logger = setup_logger()
        if self.mode == PRModes.FLOW_MATCHING:
            if self._paths and not fresh:
                logger.info(f"Using in-memory Flow Matching paths for mode {self.mode}...")
                return
            file_path = self.pagerank_output_path.joinpath('flow_matching_paths.bin')
            if not fresh and file_path.is_file() and flag.is_module_done(f'pr_{self.mode.value}_paths'):
                logger.info(f"Loading Flow Matching paths for mode {self.mode}...")
                with open(file_path, 'rb') as f:
                    self._paths = pickle.load(f)
            else:
                database_setup()
                graph_data = self.generate_pyg_graph()
                embeddings = self.compute_gat_embeddings(graph_data, fresh=fresh)
                flow_model = self.train_flow_matching(graph_data, embeddings, fresh=fresh)
                self._paths = self.generate_paths(flow_model, embeddings, graph_data, fresh=fresh)
                with open(file_path, 'wb') as f:
                    pickle.dump(self._paths, f)
                flag.mark_module_done(f'pr_{self.mode.value}_paths')
        else:
            if self._results and not fresh:
                logger.info(f"Using in-memory results for mode {self.mode}...")
                return
            file_path = self.pagerank_output_path.joinpath(f'pageranks_{self.mode.value}.bin')
            if not fresh and file_path.is_file() and flag.is_module_done(f'pr_{self.mode.value}'):
                logger.info(f"Loading results for mode {self.mode}...")
                with open(file_path, 'rb') as f:
                    self._results = pickle.load(f)
            else:
                database_setup()
                required_columns = ['paper_id', 'topic', 'probabilities']
                if not all(col in self.paper_topics_df.columns for col in required_columns):
                    raise ValueError(f"Missing required columns in paper_topics_df: {required_columns}")

                # Same MTPR engine as LDA (bias()/transition_star() in pagerank.py):
                # mode-specific behavior (centrality, author-credibility, per-target
                # transition normalization) now applies to BERTopic too, not just a
                # single mode-blind personalized PageRank for every requested mode.
                score_model = BERTopicScoreModel(self.paper_topics_df, self.no_topics, verbose=False)

                self._results = defaultdict(dict)
                for topic in range(self.no_topics):
                    bin_file = self.pr_values_output_path.joinpath(f'topic-{topic}.bin')
                    if not fresh and flag.is_module_done(f'pr_{self.mode.value}_topic_{topic}') and bin_file.exists():
                        logger.info(f"Topic {topic} already processed, loading...")
                        with open(bin_file, 'rb') as f:
                            topic_results = pickle.load(f)
                        for doc_id, scores in topic_results.items():
                            self._results[doc_id].update(scores)
                    else:
                        logger.info(f"Computing BERTopic MTPR PageRank for topic {topic}, mode {self.mode.value}...")
                        _, pr_scores = page_rank(self.mode, topic, self.graph, score_model, alpha=0.15)
                        data = {d: {topic: pr_scores[d]} for d in pr_scores}
                        bin_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(bin_file, 'wb') as f:
                            pickle.dump(data, f)
                        flag.mark_module_done(f'pr_{self.mode.value}_topic_{topic}')
                        for doc_id, scores in data.items():
                            self._results[doc_id].update(scores)

                with open(file_path, 'wb') as f:
                    pickle.dump(dict(self._results), f)
                flag.mark_module_done(f'pr_{self.mode.value}')
            self._agified_results = deepcopy(self._results)
            for paper in Paper.select(Paper.id, Paper.year).iterator():
                for t_id in self._agified_results[paper.id - 1]:
                    year_diff = max(1, 2011 - paper.year)
                    self._agified_results[paper.id - 1][t_id] /= year_diff

    def results(self, agify: bool = True) -> Dict[int, Dict[int, float]]:
        self.compute_results()
        if self.mode == PRModes.FLOW_MATCHING:
            return self._paths
        return self._agified_results if agify else self._results


def define_subparser(subparser):
    bertopic = subparser.add_parser(subparser_name := 'bertopic')
    bertopic.add_argument('--no-topics', dest='no_topics', type=int, help='Number of topics', required=True)
    bertopic.add_argument('--fresh', dest='fresh', action='store_true',
                          help='Execute BERTopic even if results are cached.')
    return subparser_name
