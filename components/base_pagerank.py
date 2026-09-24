from __future__ import annotations

import logging
import pickle
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data

from components.prmodes import PRModes
from models import Paper, Author, AuthorCitation, PaperCitation, PaperTechnicalTerm, TechnicalTerm

pr_output_path = Path(__file__).parent.joinpath('output', 'pr')



class PageRankBase:
    GRAPH_FILE_NAME = 'graph.bin'
    AUTHOR_CITATION_GRAPH_FILE_NAME = 'author_citation_graph.bin'

    def __init__(self, mode: PRModes, no_topics: int):
        """Initialize PageRankBase with only the graphs required by ``mode``."""
        self.no_topics = no_topics
        self.mode = mode
        self.graph = self.generate_citation_graph()
        self.author_graph = (
            self.generate_author_citation_graph()
            if mode in [PRModes.MODIFIED, PRModes.MODIFIED_BIAS, PRModes.MODIFIED_TRANSITION]
            else None
        )
        self.base_pagerank_output_path = pr_output_path.joinpath(str(self.no_topics))

    @staticmethod
    def cache_centrality(graph: nx.DiGraph, centrality_type: str, cache_file: Path, k: int = 100) -> Dict:
        """
        Compute or load cached centrality measures for a graph.

        :param graph: Directed graph of papers or authors
        :param centrality_type: Type of centrality ('degree', 'closeness', 'betweenness', 'katz')
        :param cache_file: Path to cache file for storing centrality results
        :param k: Number of nodes to sample for betweenness centrality (default: 100)
        :return: Dictionary of centrality scores
        """
        cache_path = Path(cache_file)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        centrality_funcs = {
            'degree': nx.degree_centrality,
            'closeness': nx.closeness_centrality,
            'betweenness': lambda g: nx.betweenness_centrality(g, k=min(k, len(g.nodes()) // 2)),
            'katz': lambda g: nx.katz_centrality(g, alpha=0.1, beta=1.0)
        }

        if centrality_type not in centrality_funcs:
            raise ValueError(
                f"Invalid centrality_type: {centrality_type}. Must be one of {list(centrality_funcs.keys())}")

        centrality = centrality_funcs[centrality_type](graph)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(centrality, f)
        return centrality

    def generate_citation_graph(self) -> nx.DiGraph:
        """Generate NetworkX directed graph from PaperCitation."""
        graph_file_path = pr_output_path.joinpath(self.GRAPH_FILE_NAME)
        if graph_file_path.is_file():
            with open(graph_file_path, 'rb') as f:
                graph = pickle.load(f)
        else:
            graph = nx.DiGraph()
            for paper in Paper.select().iterator():
                graph.add_node(paper.id - 1)
                for reference in paper.references():
                    if reference.id == paper.id:
                        continue  # ignore self-citation edges
                    graph.add_edge(paper.id - 1, reference.id - 1)
            pr_output_path.mkdir(parents=True, exist_ok=True)
            with open(graph_file_path, 'wb') as f:
                pickle.dump(graph, f)
        return graph

    def generate_author_citation_graph(self) -> nx.DiGraph:
        """Generate NetworkX directed graph for author citations."""
        graph_file_path = pr_output_path.joinpath(self.AUTHOR_CITATION_GRAPH_FILE_NAME)
        if graph_file_path.is_file():
            with open(graph_file_path, 'rb') as f:
                graph = pickle.load(f)
        else:
            graph = nx.DiGraph()
            for author in Author.select():
                graph.add_node(author.id - 1)
            for citation in AuthorCitation.select():
                graph.add_edge(
                    citation.citing_author_id - 1,
                    citation.cited_author_id - 1,
                    weight=citation.count
                )
            pr_output_path.mkdir(parents=True, exist_ok=True)
            with open(graph_file_path, 'wb') as f:
                pickle.dump(graph, f)
        return graph

    @staticmethod
    def generate_pyg_graph() -> Data:
        """Generate PyTorch Geometric graph with node features from BERTopic and technical terms."""
        graph_file_path = pr_output_path.joinpath('pyg_graph.bin')
        if graph_file_path.is_file():
            return torch.load(graph_file_path)

        edges = [(c.citing_paper_id - 1, c.cited_paper_id - 1) for c in PaperCitation.select()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        num_papers = Paper.select().count()
        
        # Build term frequencies first to get accurate paper count
        term_frequencies = [[] for _ in range(num_papers)]
        for doc_term in PaperTechnicalTerm.select(PaperTechnicalTerm, TechnicalTerm).join(TechnicalTerm):
            term_frequencies[doc_term.paper_id - 1].append(doc_term.frequency)
        
        # Pad term frequencies to consistent length
        max_terms = max(len(lst) for lst in term_frequencies) if any(term_frequencies) else 1
        for lst in term_frequencies:
            lst.extend([0.0] * (max_terms - len(lst)))
        term_frequencies_tensor = torch.tensor(term_frequencies, dtype=torch.float)
        
        # Load topic probabilities
        try:
            paper_topics_df = pd.read_csv("components/resources/brt/paper_topics.csv")
            # Safe parsing of probabilities column
            import ast
            topic_probs = []
            for row in paper_topics_df.itertuples():
                if isinstance(row.probabilities, str):
                    try:
                        probs = ast.literal_eval(row.probabilities)
                        topic_probs.append(probs)
                    except (ValueError, SyntaxError):
                        logging.warning(f"Could not parse probabilities: {row.probabilities}")
                        topic_probs.append([0.0])
                elif isinstance(row.probabilities, (list, tuple)):
                    topic_probs.append(list(row.probabilities))
                else:
                    logging.warning(f"Unexpected probabilities type: {type(row.probabilities)}")
                    topic_probs.append([0.0])
            
            topic_probs_tensor = torch.tensor(topic_probs, dtype=torch.float)
            
            # Ensure sizes match (trim or pad to num_papers)
            if topic_probs_tensor.shape[0] > num_papers:
                logging.warning(f"Topic probs has {topic_probs_tensor.shape[0]} rows, trimming to {num_papers}")
                topic_probs_tensor = topic_probs_tensor[:num_papers]
            elif topic_probs_tensor.shape[0] < num_papers:
                logging.warning(f"Topic probs has {topic_probs_tensor.shape[0]} rows, padding to {num_papers}")
                padding = torch.zeros((num_papers - topic_probs_tensor.shape[0], topic_probs_tensor.shape[1]))
                topic_probs_tensor = torch.cat([topic_probs_tensor, padding], dim=0)
            
            x = topic_probs_tensor
        except Exception as e:
            logging.error(f"Error loading paper_topics.csv: {e}. Using zero probabilities.")
            x = torch.zeros((num_papers, 1), dtype=torch.float)
        
        # Concatenate features
        x = torch.cat([x, term_frequencies_tensor], dim=1)
        graph_data = Data(x=x, edge_index=edge_index)
        pr_output_path.mkdir(parents=True, exist_ok=True)
        torch.save(graph_data, graph_file_path)
        return graph_data

    def calculate_pagerank(self, graph: nx.DiGraph, mode: str = 'normal', centrality_type: Optional[str] = None,
                           weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Calculate PageRank scores based on the specified mode.

        :param graph: Directed graph of papers or authors
        :param mode: PageRank mode ('normal', 'modified', 'hybrid', etc.)
        :param centrality_type: Type of centrality for 'modified' mode ('degree', 'closeness', 'betweenness', 'katz')
        :param weights: Weights for centrality types in 'hybrid' mode
        :return: Dictionary of PageRank scores
        """
        G = nx.DiGraph(graph)
        if mode == 'hybrid':
            if weights is None:
                weights = {'degree': 0.25, 'closeness': 0.25, 'betweenness': 0.25, 'katz': 0.25}
            cache_dir = self.base_pagerank_output_path.joinpath('centrality_cache')
            cache_dir.mkdir(parents=True, exist_ok=True)
            centralities = {
                c: self.cache_centrality(G, c, cache_dir.joinpath(f'{c}_centrality.pkl'))
                for c in weights
            }
            hybrid_centrality = {node: sum(w * centralities[c].get(node, 0) for c, w in weights.items()) for node in
                                 G.nodes}
            return nx.pagerank(G, personalization=hybrid_centrality)
        elif mode == 'modified' and centrality_type:
            cache_file = self.base_pagerank_output_path.joinpath(f'centrality_cache/{centrality_type}_centrality.pkl')
            centrality = self.cache_centrality(G, centrality_type, cache_file)
            return nx.pagerank(G, personalization=centrality)
        else:
            return nx.pagerank(G)
