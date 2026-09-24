import logging
import time
import os
import pickle
from collections import defaultdict
from functools import cached_property
from functools import lru_cache
from multiprocessing import Pool
from typing import Union, Optional, List, Dict, Tuple, Any
from pathlib import Path
import networkx as nx
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore, LdaModel
from tqdm import tqdm
from models import *
from utils import flag
from .database import setup as database_setup

lda_output_path = Path(__file__).parent.joinpath('output', 'lda')
lda_log_path = Path(os.getenv('LOG_PATH', 'logs')).joinpath('lda.log')

# LDA's Dirichlet smoothing gives every document a technically-nonzero
# probability for every topic, so filtering topic subgraphs with
# minimum_probability=0.0 (i.e. no filtering) makes the "topic subgraph"
# equal to the *entire* corpus for every topic -- at 200 topics this made
# betweenness/closeness/degree centrality run on all ~15,666 documents per
# topic instead of the documents meaningfully associated with it, which is
# both conceptually wrong (the paper defines the topic subgraph as documents
# with P(t|d) > 0 in a meaningful sense, not a smoothing artifact) and the
# reason centrality-based PageRank modes were projected to take *weeks*.
# Used only for topic-subgraph membership (centrality, author enumeration),
# never for the Sum_{d in D} P(t|d) normalizations in bias()/transition_star(),
# which must keep summing over the whole corpus per Jardine's definition.
TOPIC_MEMBERSHIP_MIN_PROBABILITY = 0.01

_author_reference_sets_cache: Optional[Dict[int, frozenset]] = None


def author_reference_sets() -> Dict[int, frozenset]:
    """Each author's aggregate reference set: the union, over every paper the
    author co-wrote, of the papers that paper cites (0-based ids). Used by
    author_bibliographic_coupling() to compute ABC edge weights (proposal
    Sec. 4.2.2) -- built once from PaperAuthor + PaperCitation and cached to
    disk; independent of topic count and of no_topics, so it is not part of
    any per-topic cache key.
    """
    global _author_reference_sets_cache
    if _author_reference_sets_cache is not None:
        return _author_reference_sets_cache
    cache_path = lda_output_path.joinpath('author_scores', 'author_reference_sets.pkl')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            _author_reference_sets_cache = pickle.load(f)
        return _author_reference_sets_cache
    ref_sets: Dict[int, set] = defaultdict(set)
    query = (PaperAuthor
             .select(PaperAuthor.author, PaperCitation.cited_paper)
             .join(PaperCitation, on=(PaperCitation.citing_paper == PaperAuthor.paper))
             .tuples())
    for author_id, cited_paper_id in query.iterator():
        ref_sets[author_id - 1].add(cited_paper_id - 1)
    _author_reference_sets_cache = {k: frozenset(v) for k, v in ref_sets.items()}
    with open(cache_path, 'wb') as f:
        pickle.dump(_author_reference_sets_cache, f)
    return _author_reference_sets_cache


def author_bibliographic_coupling(refs_a: frozenset, refs_b: frozenset) -> float:
    """ABC(a,b) = |Refs(a) cap Refs(b)| / |Refs(a) cup Refs(b)|
    (proposal Sec. 4.2.2 / Zhao & Strotmann 2008): Jaccard overlap of two
    authors' aggregate reference sets. Symmetric; 0 if both sets are empty.
    """
    if not refs_a and not refs_b:
        return 0.0
    union_size = len(refs_a | refs_b)
    if union_size == 0:
        return 0.0
    return len(refs_a & refs_b) / union_size


def setup_logger() -> logging.Logger:
    """Configure and return a logger for the LDA module."""
    lda_log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=lda_log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()


class TopicScoreModel:
    """Shared topic-scoring interface: caching, centrality, and author-credibility
    logic that is identical regardless of which topic model produced P(t|d).

    Subclasses provide the topic model itself by overriding get_doc_topics(),
    num_docs, and cache_root -- everything else here (bias/transition_star's
    entire MTPR engine in pagerank.py runs against this interface) is generic.
    GensimLDA (below) is the LDA-backed implementation; BERTopicScoreModel
    (components/bertopic.py) is the BERTopic-backed one -- both feed the same
    page_rank() so a mode behaves identically under either topic model.
    """

    verbose: bool = False

    # 'raw': author-citation edge weight w_ba = raw citation count (default,
    #   matches all prior cached results). 'abc': w_ba = Author Bibliographic
    #   Coupling (proposal, Sec. 4.2.2) -- Jaccard overlap of the two authors'
    #   aggregate reference sets, replacing the raw count. Set on a fresh
    #   model instance before any author_topic_scores() call; do not flip
    #   this attribute mid-process on an instance that has already computed
    #   scores under the other scheme (author_topic_scores is lru_cache'd per
    #   instance and won't detect the change).
    author_weighting: str = 'raw'

    @property
    def num_docs(self) -> int:
        """Total number of documents in the corpus. Override in subclasses."""
        raise NotImplementedError

    @property
    def cache_root(self) -> Path:
        """Base directory for this model's on-disk caches. Override in subclasses."""
        raise NotImplementedError

    def get_doc_topics(self, doc_id: int, minimum_probability: float = 0.0) -> Dict[int, float]:
        """Retrieve topic distribution for a specific document. Override in subclasses."""
        raise NotImplementedError

    def get_cache_path_and_value(self, pr_mode, cache_name, done_flag, topic_id=None, doc_id=None):
        logger = setup_logger()
        cache_path = self.cache_root.joinpath(pr_mode, cache_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and flag.is_module_done(done_flag):
            try:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
                return cache_path, value
            except Exception as e:
                if doc_id and topic_id:
                    logger.warning(f"Error loading cached {cache_name} for topic {topic_id}, doc {doc_id}: {e}")
                elif topic_id:
                    logger.warning(f"Error loading cached {cache_name} for topic {topic_id}: {e}")
                else:
                    logger.warning(f"Error loading cached {cache_name}: {e}")
                os.remove(cache_path)  # Clean up corrupted cache
        return cache_path, None

    def verbose_time_log(self, log_title, start_t, diff, topic_id=None, doc_id=None):
        logger = setup_logger()
        if topic_id and doc_id:
            log = f"{log_title}({topic_id}, {doc_id}): {(time.time() - start_t):.5f} seconds"
        elif topic_id:
            log = f"{log_title}({topic_id}): {(time.time() - start_t):.5f} seconds"
        else:
            log = f"{log_title}: {(time.time() - start_t):.5f} seconds"
        if self.verbose and (time.time() - start_t) > diff:
            logger.info(log)

    @lru_cache(maxsize=None)
    def topic_doc_score(self, doc_id: int, topic_id: int, minimum_probability: float = 0.0) -> float:
        """Get the probability of a topic for a specific document."""
        start_t = time.time()
        score = self.get_doc_topics(doc_id, minimum_probability=minimum_probability).get(topic_id, 0.0)
        self.verbose_time_log('topic_doc_score', start_t, 0.3)
        return score

    @lru_cache(maxsize=None)
    def topic_docs_score(self, topic_id: int) -> float:
        """Aggregate topic distribution probabilities across all documents."""
        start_t = time.time()
        score = sum(self.get_topic_docs(topic_id).values())
        self.verbose_time_log('topic_docs_score', start_t, 0.3)
        return score

    def topic_degree_centrality_scores_batch(self, graph: nx.DiGraph, topic_id: int) -> Dict[int, int]:
        """
        Compute degree centrality (out-degree restricted to the topic subgraph) for
        ALL documents in a topic in ONE call, batch-cached -- mirrors
        topic_closeness/betweenness/katz_centrality_scores_batch.

        Previously this was computed per-document via a DB query
        (Paper.paper_references) and cached one file PER (topic_id, doc_id) pair --
        up to 200 x 15,666 = 3.1 million tiny cache/flag files on disk, the only one
        of the four centrality measures not already batched. This uses the
        already-loaded citation graph directly instead of a DB round trip per doc.
        """
        cache_path, cached_value = self.get_cache_path_and_value(
            'centrality', f'degree_batch_{topic_id}.pkl',
            f'degree_batch_{topic_id}', topic_id)
        if cached_value is not None:
            return cached_value
        start_t = time.time()
        topic_documents = self.get_topic_docs(topic_id, minimum_probability=TOPIC_MEMBERSHIP_MIN_PROBABILITY)
        topic_nodes = set(p for p in range(self.num_docs) if topic_documents.get(p, 0) > 0)
        centrality_dict = {
            doc_id: sum(1 for target in graph.successors(doc_id) if target in topic_nodes)
            for doc_id in topic_nodes if graph.has_node(doc_id)
        }
        self.verbose_time_log('topic_degree_centrality_scores_batch', start_t, 0.1, topic_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(centrality_dict, f)
        flag.mark_module_done(f'degree_batch_{topic_id}')
        return centrality_dict

    def topic_doc_degree_centrality_score(self, graph: nx.DiGraph, topic_id: int, doc_id: int) -> int:
        """Degree centrality for a single document within a topic subgraph. O(1) batch lookup."""
        batch = self.topic_degree_centrality_scores_batch(graph, topic_id)
        return batch.get(doc_id, 0)

    def topic_closeness_centrality_scores_batch(self, graph: nx.DiGraph, topic_id: int) -> Dict[int, float]:
        """
        OPTIMIZED: Compute closeness centrality for ALL documents in a topic in ONE call.
        
        Previously, topic_doc_closeness_centrality_score() called nx.closeness_centrality()
        once per document, recomputing it for the ENTIRE subgraph each time. This was
        O(N × V²) instead of O(V²). Now we compute once and cache the full dict.
        """
        cache_path, cached_value = self.get_cache_path_and_value(
            'centrality', f'closeness_batch_{topic_id}.pkl',
            f'closeness_batch_{topic_id}', topic_id)
        if cached_value is not None:
            return cached_value
        start_t = time.time()
        topic_documents = self.get_topic_docs(topic_id, minimum_probability=TOPIC_MEMBERSHIP_MIN_PROBABILITY)
        topic_nodes = [p for p in range(Paper.select().count()) if topic_documents.get(p, 0) > 0]
        if not topic_nodes:
            result = {}
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            flag.mark_module_done(f'closeness_batch_{topic_id}')
            return result
        topic_subgraph = graph.subgraph(topic_nodes)
        # ONE call computes for all nodes at once
        centrality_dict = nx.closeness_centrality(topic_subgraph, distance=None, wf_improved=True)
        self.verbose_time_log('topic_closeness_centrality_scores_batch', start_t, 0.1, topic_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(centrality_dict, f)
        flag.mark_module_done(f'closeness_batch_{topic_id}')
        return centrality_dict

    def topic_doc_closeness_centrality_score(self, graph: nx.DiGraph, topic_id: int, doc_id: int) -> float:
        """
        Calculate closeness centrality for a document within a topic.
        
        OPTIMIZED: Delegates to topic_closeness_centrality_scores_batch() which computes
        the full topic subgraph closeness in one shot and caches it. Looking up a single
        value from the batch cache is O(1).
        """
        batch = self.topic_closeness_centrality_scores_batch(graph, topic_id)
        return batch.get(doc_id, 0.0)

    def topic_betweenness_centrality_scores_batch(self, graph: nx.DiGraph, topic_id: int) -> Dict[int, float]:
        """
        OPTIMIZED: Compute betweenness centrality for ALL documents in a topic in ONE call.
        
        Previously, topic_doc_betweenness_centrality_score() called
        nx.betweenness_centrality_subset() once per document. That function computes
        the betweenness for ALL nodes anyway — we were just discarding all but one value
        and then recomputing it again for the next document. This was catastrophically slow:
        O(N × V × E) instead of O(V × E). This fix gives a speedup proportional to N
        (the number of papers, e.g. 2000×).
        """
        cache_path, cached_value = self.get_cache_path_and_value(
            'centrality', f'betweenness_batch_{topic_id}.pkl',
            f'betweenness_batch_{topic_id}', topic_id)
        if cached_value is not None:
            return cached_value
        start_t = time.time()
        topic_documents = self.get_topic_docs(topic_id, minimum_probability=TOPIC_MEMBERSHIP_MIN_PROBABILITY)
        topic_nodes = [p for p in range(Paper.select().count()) if topic_documents.get(p, 0) > 0]
        if not topic_nodes:
            result = {}
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            flag.mark_module_done(f'betweenness_batch_{topic_id}')
            return result
        # ONE call computes for all nodes at once
        centrality_dict = nx.betweenness_centrality_subset(
            graph, sources=topic_nodes, targets=topic_nodes, normalized=True)
        self.verbose_time_log('topic_betweenness_centrality_scores_batch', start_t, 0.1, topic_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(centrality_dict, f)
        flag.mark_module_done(f'betweenness_batch_{topic_id}')
        return centrality_dict

    def topic_doc_betweenness_centrality_score(self, graph: nx.DiGraph, topic_id: int, doc_id: int) -> float:
        """
        Calculate betweenness centrality for a document within a topic.
        
        OPTIMIZED: Delegates to topic_betweenness_centrality_scores_batch() which computes
        the full betweenness in one shot. Looking up a single value is O(1).
        """
        batch = self.topic_betweenness_centrality_scores_batch(graph, topic_id)
        return batch.get(doc_id, 0.0)

    def topic_degree_centrality_score(self, graph: nx.DiGraph, topic_id: int) -> int:
        """Calculate the sum of degree centrality values for all documents in a topic."""
        cache_path, cached_value = self.get_cache_path_and_value('centrality', f'total_degree_{topic_id}.pkl',
                                                                 f'total_degree_centrality_{topic_id}', topic_id)
        if cached_value is not None:
            return cached_value
        start_t = time.time()
        batch = self.topic_degree_centrality_scores_batch(graph, topic_id)
        total = sum(batch.values())
        self.verbose_time_log('topic_degree_centrality_score', start_t, 0.3, topic_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(total, f)
        flag.mark_module_done(f'total_degree_centrality_{topic_id}')
        return total

    def topic_closeness_centrality_score(self, graph: nx.DiGraph, topic_id: int) -> float:
        """Calculate the sum of closeness centrality values for all documents in a topic."""
        cache_path, cached_value = self.get_cache_path_and_value('centrality', f'total_closeness_{topic_id}.pkl',
                                                                 f'total_closeness_centrality_{topic_id}', topic_id)
        if cached_value is not None:
            return cached_value
        start_t = time.time()
        # OPTIMIZED: use batch computation instead of per-doc calls
        batch = self.topic_closeness_centrality_scores_batch(graph, topic_id)
        total = sum(batch.values())
        self.verbose_time_log('topic_closeness_centrality_score', start_t, 0.3, topic_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(total, f)
        flag.mark_module_done(f'total_closeness_centrality_{topic_id}')
        return total

    def topic_betweenness_centrality_score(self, graph: nx.DiGraph, topic_id: int) -> float:
        """Calculate the sum of betweenness centrality values for all documents in a topic."""
        cache_path, cached_value = self.get_cache_path_and_value('centrality', f'total_betweenness_{topic_id}.pkl',
                                                                 f'total_betweenness_centrality_{topic_id}', topic_id)
        if cached_value is not None:
            return cached_value
        start_t = time.time()
        # OPTIMIZED: use batch computation instead of per-doc calls
        batch = self.topic_betweenness_centrality_scores_batch(graph, topic_id)
        total = sum(batch.values())
        self.verbose_time_log('topic_betweenness_centrality_score', start_t, 0.3, topic_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(total, f)
        flag.mark_module_done(f'total_betweenness_centrality_{topic_id}')
        return total

    def topic_katz_centrality_scores_batch(self, graph: nx.DiGraph, topic_id: int) -> Dict[int, float]:
        """
        Compute Katz centrality for ALL documents in a topic subgraph in ONE call,
        batch-cached -- mirrors topic_closeness/betweenness_centrality_scores_batch.

        Previously KZ_MODIFIED computed Katz centrality on the WHOLE citation graph
        (not the topic subgraph), the only one of the four centrality measures that
        wasn't topic-restricted, so its centrality signal was identical across all
        200 topics -- only the P(t|d) factor varied. This makes it topic-aware like
        DC/CC/BC.
        """
        cache_path, cached_value = self.get_cache_path_and_value(
            'centrality', f'katz_batch_{topic_id}.pkl',
            f'katz_batch_{topic_id}', topic_id)
        if cached_value is not None:
            return cached_value
        start_t = time.time()
        topic_documents = self.get_topic_docs(topic_id, minimum_probability=TOPIC_MEMBERSHIP_MIN_PROBABILITY)
        topic_nodes = [p for p in range(self.num_docs) if topic_documents.get(p, 0) > 0]
        if not topic_nodes:
            result = {}
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            flag.mark_module_done(f'katz_batch_{topic_id}')
            return result
        topic_subgraph = graph.subgraph(topic_nodes)
        try:
            centrality_dict = nx.katz_centrality(topic_subgraph, alpha=0.1, beta=1.0, max_iter=1000)
        except Exception:
            centrality_dict = {n: 0.0 for n in topic_nodes}
        self.verbose_time_log('topic_katz_centrality_scores_batch', start_t, 0.1, topic_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(centrality_dict, f)
        flag.mark_module_done(f'katz_batch_{topic_id}')
        return centrality_dict

    def topic_doc_katz_centrality_score(self, graph: nx.DiGraph, topic_id: int, doc_id: int) -> float:
        """Katz centrality for a single document within a topic subgraph. O(1) batch lookup."""
        batch = self.topic_katz_centrality_scores_batch(graph, topic_id)
        return batch.get(doc_id, 0.0)

    def topic_katz_centrality_score(self, graph: nx.DiGraph, topic_id: int) -> float:
        """Calculate the sum of Katz centrality values for all documents in a topic."""
        cache_path, cached_value = self.get_cache_path_and_value('centrality', f'total_katz_{topic_id}.pkl',
                                                                 f'total_katz_centrality_{topic_id}', topic_id)
        if cached_value is not None:
            return cached_value
        start_t = time.time()
        batch = self.topic_katz_centrality_scores_batch(graph, topic_id)
        total = sum(batch.values())
        self.verbose_time_log('topic_katz_centrality_score', start_t, 0.3, topic_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(total, f)
        flag.mark_module_done(f'total_katz_centrality_{topic_id}')
        return total

    def get_topic_docs(self, topic_id: int, minimum_probability: float = 0.0) -> Dict[int, float]:
        """Collect distribution values for a given topic across all documents, with caching.

        Cache key includes minimum_probability: without this, a call with the
        default (0.0, e.g. from topic_docs_score()'s Sum_D P(t|d'') -- which
        must stay unfiltered) would get cached, and a later call for the same
        topic_id with a higher threshold (e.g. centrality's
        TOPIC_MEMBERSHIP_MIN_PROBABILITY) would silently return that stale,
        unfiltered cache instead of actually filtering -- exactly the kind of
        cache collision that made centrality subgraphs include the whole
        corpus regardless of threshold.
        """
        mp_key = f'{minimum_probability:.4f}'
        cache_path, cached_value = self.get_cache_path_and_value(
            'topic_docs', f'topic_{topic_id}_mp{mp_key}.pkl',
            f'get_topic_docs_{topic_id}_mp{mp_key}', topic_id)
        if cached_value is not None:
            return cached_value
        start_t = time.time()
        # Use .get(topic_id, 0.0) instead of [topic_id] — gensim's get_document_topics()
        # returns a sparse dict; topic_id may be absent when probability rounds to zero.
        # Using direct indexing would raise KeyError for those documents.
        topic_docs = {
            doc_id: self.get_doc_topics(doc_id, minimum_probability=minimum_probability).get(topic_id, 0.0)
            for doc_id in range(self.num_docs)
        }
        self.verbose_time_log('get_topic_docs', start_t, 0.3, topic_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(topic_docs, f)
        flag.mark_module_done(f'get_topic_docs_{topic_id}')
        return topic_docs

    def get_topic_authors(self, topic_id: int) -> List[Author]:
        """Retrieve authors who published at least one document related to a topic."""
        cache_path, cached_value = self.get_cache_path_and_value('topic_authors', f'topic_{topic_id}.pkl',
                                                                 f'get_topic_authors_{topic_id}', topic_id)
        if cached_value is not None:
            author_ids = cached_value
            authors = list(Author.select().where(Author.id << author_ids))
            return authors
        start_t = time.time()
        topic_documents = self.get_topic_docs(topic_id, minimum_probability=TOPIC_MEMBERSHIP_MIN_PROBABILITY)
        relevant_docs = [doc_id for doc_id, p in topic_documents.items() if p > 0]
        # document_authors() filters on the real (1-based) Paper.id / PaperAuthor.paper
        # FK, but doc_id here is our internal 0-based convention (doc_id = paper.id - 1)
        # -- passing doc_ids directly looked up the wrong (off-by-one) paper for every
        # entry, silently returning the wrong (or empty) author set for every topic.
        authors = Author.document_authors([doc_id + 1 for doc_id in relevant_docs])
        self.verbose_time_log('get_topic_authors', start_t, 0.3, topic_id)
        with open(cache_path, 'wb') as f:
            pickle.dump([author.id for author in authors], f)
        flag.mark_module_done(f'get_topic_authors_{topic_id}')
        return authors

    @cached_property
    def author_scores(self) -> Dict[int, Dict[int, float]]:
        """Retrieve author credits across all topics, with caching."""
        cache_path, cached_value = self.get_cache_path_and_value('', 'author_scores.pkl', 'author_scores')
        if cached_value is not None:
            return cached_value
        start_t = time.time()
        topic_author_scores = {topic_id: self.author_topic_scores(topic_id) for topic_id in range(self.lda_no_topics)}
        self.verbose_time_log('author_scores', start_t, 0.3)
        with open(cache_path, 'wb') as f:
            pickle.dump(topic_author_scores, f)
        flag.mark_module_done('author_scores')
        return topic_author_scores

    @cached_property
    def authors_score(self) -> float:
        """Calculate the total score for all authors across all topics."""
        cache_path, cached_value = self.get_cache_path_and_value('', 'authors_score.pkl', 'authors_score')
        if cached_value is not None:
            return cached_value
        start_t = time.time()
        total = sum(sum(v.values()) for v in self.author_scores.values())
        self.verbose_time_log('authors_score', start_t, 0.3)
        with open(cache_path, 'wb') as f:
            pickle.dump(total, f)
        flag.mark_module_done('authors_score')
        return total

    @lru_cache(maxsize=None)
    def author_topic_scores(self, topic_id: int) -> Dict[int, float]:
        """Calculate scores for authors who published documents in a topic.

        In-memory-cached: bias()/transition_star() call doc_authors_score()
        once per document/edge (tens of thousands of times), and without this
        cache each of those calls re-reads and re-deserializes the on-disk
        cache for this topic's full author-score dict from scratch.
        """
        weighting = self.author_weighting
        cache_path, cached_value = self.get_cache_path_and_value(
            'author_scores', f'topic_{topic_id}_{weighting}.pkl',
            f'author_topic_scores_{topic_id}_{weighting}', topic_id)
        if cached_value is not None:
            return cached_value
        start_t = time.time()
        author_topic_scores = defaultdict(lambda: 0)
        authors = self.get_topic_authors(topic_id)
        for author in authors:
            author_topic_scores[author.id - 1] = sum(
                self.topic_doc_score(doc.id - 1, topic_id) for doc in author.papers)
        initial_scores = dict(author_topic_scores)
        if weighting == 'abc':
            ref_sets = author_reference_sets()
        for author in authors:
            # citing_author.incited_authors() is not restricted to this topic's
            # author set (unlike `authors` above, which IS filtered by
            # TOPIC_MEMBERSHIP_MIN_PROBABILITY) -- a citing author with no
            # meaningful presence in topic_id won't be a key in initial_scores,
            # so this must default to 0.0 rather than a bare [] lookup (which
            # raised KeyError once initial_scores stopped containing every author).
            if weighting == 'abc':
                # proposal Sec. 4.2.2: author-network edges weighted by Author
                # Bibliographic Coupling (shared-reference overlap) rather than
                # raw citation count; edge direction/existence is unchanged
                # (still comes from the citation graph via incited_authors()),
                # only the weight w_ba is replaced.
                a_refs = ref_sets.get(author.id - 1, frozenset())
                tmp = [
                    (author_bibliographic_coupling(a_refs, ref_sets.get(citing_author.id - 1, frozenset())),
                     initial_scores.get(citing_author.id - 1, 0.0))
                    for citing_author in author.incited_authors()
                ]
                tmp = [t for t in tmp if t[0] > 0]
            else:
                tmp = [(citing_author.citations, initial_scores.get(citing_author.id - 1, 0.0))
                       for citing_author in author.incited_authors() if citing_author.citations]
            neighbour_score = sum(t[0] * t[1] for t in tmp) / sum(t[0] for t in tmp) if tmp else 0
            # Proposal step "امتیاز اعتبار نهایی": weighted neighbour score PLUS
            # the author's own initial (topic-based) score, not a replacement.
            author_topic_scores[author.id - 1] = neighbour_score + initial_scores[author.id - 1]
        self.verbose_time_log('author_topic_scores', start_t, 0.3)
        with open(cache_path, 'wb') as f:
            pickle.dump(dict(author_topic_scores), f)
        flag.mark_module_done(f'author_topic_scores_{topic_id}_{weighting}')
        return author_topic_scores

    @lru_cache(maxsize=None)
    def topic_authors_score(self, topic_id: int) -> float:
        """Calculate the total score for authors in a topic."""
        return sum(self.author_topic_scores(topic_id).values())

    def topic_author_score(self, topic_id: int, author_id: int) -> float:
        """Retrieve the score for a specific author in a topic.

        author_topic_scores(topic_id) is only keyed by the topic-filtered
        author set (TOPIC_MEMBERSHIP_MIN_PROBABILITY); callers like
        doc_authors_score() query arbitrary authors of a specific document,
        who may have no meaningful presence in this topic at all -- treat
        that as a credibility score of 0 rather than a bare-lookup KeyError.
        """
        return self.author_topic_scores(topic_id).get(author_id, 0.0)

    @lru_cache(maxsize=None)
    def doc_authors_score(self, topic_id: int, doc_id: int) -> float:
        """Calculate the aggregated author scores for a document in a topic."""
        return sum(self.topic_author_score(topic_id, author.id - 1) for author in Paper.paper_authors(doc_id + 1))


class GensimLDA(TopicScoreModel):
    """LDA-backed implementation of TopicScoreModel, using the Gensim library."""

    def __init__(self, lda_no_topics: int, docs: list = None, dictionary: Dictionary = None,
                 model: Optional[Union[LdaMulticore, LdaModel]] = None, verbose: bool = False,
                 cluster: int = 0):
        self.model = model
        self.docs = docs
        self.verbose = verbose
        self.dictionary = dictionary if dictionary is not None else Dictionary([])
        self.lda_no_topics = lda_no_topics
        self.alpha = 2 / lda_no_topics
        # Identifies which frozen theta artifact (components/theta_artifact.py)
        # this instance reads P(t|d) from -- see get_doc_topics().
        self.cluster = cluster

    @property
    def num_docs(self) -> int:
        return len(self.bag_of_words)

    @property
    def cache_root(self) -> Path:
        return lda_output_path

    @classmethod
    def file_name(cls, lda_no_topics: int, cluster: int = None) -> Dict[str, str]:
        """Generate file names for storable elements of a specific cluster."""
        cluster_suffix = f"-cluster{cluster}" if cluster is not None else ""
        base_name = f"topic{cluster_suffix}-{lda_no_topics}"
        return {
            'model': f"{base_name}.bin",
            'dictionary': f"{base_name}.dict",
            'docs': f"{base_name}.docs.pkl"
        }

    @classmethod
    def restore_model(cls, lda_no_topics: int, cluster: int) -> Optional['GensimLDA']:
        """Restore an LDA model for a specific cluster from stored files."""
        loaded = cls.load_model(lda_no_topics, cluster)
        if loaded and all(value is not None for value in loaded.values()):
            return cls(lda_no_topics=lda_no_topics, cluster=cluster, **loaded)
        return None

    def process(self, force: bool = False) -> 'GensimLDA':
        """Generate and store the LDA model, using cache if available."""
        if not self.model or force:
            self.dictionary = Dictionary(self.docs)
            self.model = self.generate_model(self.bag_of_words, self.dictionary)
            self.save_model()
        return self

    def get_term_topics(self, term: str, minimum_probability: float = 0.0) -> Dict[int, float]:
        """Retrieve topics associated with a given term."""
        return dict(self.model.get_term_topics(self.word2id[term], minimum_probability=minimum_probability))

    @cached_property
    def word2id(self) -> dict:
        """Generate a dictionary mapping words to their IDs in the model."""
        return {value: key for key, value in self.model.id2word.items()}

    @cached_property
    def bag_of_words(self) -> list:
        """Convert documents to bag-of-words format."""
        return [self.dictionary.doc2bow(doc) for doc in self.docs]

    @cached_property
    def theta_artifact(self) -> np.ndarray:
        """The frozen (num_docs, num_topics) theta matrix for this model/cluster
        -- see components/theta_artifact.py. Loaded once per instance lifetime.
        """
        from components.theta_artifact import load_theta_artifact
        theta, _metadata = load_theta_artifact(self.lda_no_topics, self.cluster)
        if theta.shape[0] != self.num_docs:
            raise ValueError(
                f"Theta artifact has {theta.shape[0]} rows, but this model has "
                f"{self.num_docs} documents (lda_no_topics={self.lda_no_topics}, "
                f"cluster={self.cluster}) -- artifact is stale, rebuild it."
            )
        return theta

    @lru_cache(maxsize=None)
    def get_doc_topics(self, doc_id: int, minimum_probability: float = 0.0) -> Dict[int, float]:
        """Retrieve topic distribution for a specific document from the frozen
        theta artifact (NOT live gensim inference -- get_document_topics()
        re-runs non-reseeded variational inference on every call, which was
        confirmed to make PR(d,t) itself irreproducible; see
        components/theta_artifact.py). Filtering matches gensim's own
        get_document_topics() semantics exactly: topics with probability
        >= max(minimum_probability, 1e-8) are kept.
        """
        start_t = time.time()
        threshold = max(minimum_probability, 1e-8)
        row = self.theta_artifact[doc_id]
        topics = {int(topic_id): float(p) for topic_id, p in enumerate(row) if p >= threshold}
        self.verbose_time_log('get_doc_topics', start_t, 0.3)
        return topics

    def get_coherence_model(self, c_type: str = 'c_v') -> CoherenceModel:
        """Create a coherence model for evaluating topic quality."""
        return CoherenceModel(model=self.model, texts=self.docs, dictionary=self.model.id2word, coherence=c_type)

    def save_model(self, cluster: int = None):
        """Save the LDA model and its associated data to disk."""
        full_path = self.full_path(self.lda_no_topics, cluster)
        os.makedirs(self.path_to_file(), exist_ok=True)
        with open(full_path['docs'], 'wb') as file:
            pickle.dump(self.docs, file)
        self.dictionary.save(str(full_path['dictionary']))
        self.model.save(str(full_path['model']))

    @classmethod
    def load_model(cls, lda_no_topics: int, cluster: int = None) -> dict[
        str, Union[LdaMulticore, LdaModel, list, None]]:
        """Load an LDA model and its associated data from disk."""
        loaded = {}
        full_path = cls.full_path(lda_no_topics, cluster)
        loaded['model'] = cls._load_model(full_path['model']) if full_path['model'].is_file() else None
        loaded['dictionary'] = Dictionary.load(str(full_path['dictionary'])) if full_path[
            'dictionary'].is_file() else None
        if full_path['docs'].is_file():
            with open(full_path['docs'], 'rb') as file:
                loaded['docs'] = pickle.load(file)
        else:
            loaded['docs'] = None
        return loaded

    @classmethod
    def full_path(cls, lda_no_topics: int, cluster: int = None) -> dict[str, Path]:
        """Generate file paths for storing model and associated data."""
        full_path = {}
        for title, file_name in cls.file_name(lda_no_topics, cluster).items():
            full_path[title] = cls.path_to_file().joinpath(file_name)
        return full_path

    @classmethod
    def path_to_file(cls) -> Path:
        """Get the base directory for storing LDA models."""
        return Path(__file__).parent.joinpath('output', 'lda', 'gensim')

    def generate_model(self, bag_of_words: list, id2word: Dictionary) -> Union[LdaMulticore, LdaModel]:
        """Generate the Gensim LDA model (implemented by subclasses)."""
        raise NotImplementedError

    def _load_model(self, path: Path) -> Union[LdaMulticore, LdaModel]:
        """Load a Gensim LDA model from a file (implemented by subclasses)."""
        raise NotImplementedError


class GensimScLDA(GensimLDA):
    """Single-core LDA model generation using the Gensim library."""

    @classmethod
    def path_to_file(cls) -> Path:
        """Get the storage directory for single-core LDA models."""
        return super().path_to_file().joinpath('single-core')

    def generate_model(self, bag_of_words: list, id2word: Dictionary) -> LdaModel:
        """Generate a single-core LDA model."""
        return LdaModel(bag_of_words, num_topics=self.lda_no_topics, id2word=id2word,
                        eval_every=1, alpha=self.alpha, random_state=0)

    @classmethod
    def _load_model(cls, file_path: Path) -> LdaModel:
        """Load a single-core LDA model from a file."""
        return LdaModel.load(str(file_path))


class GensimMcLDA(GensimLDA):
    """Multi-core LDA model generation using the Gensim library."""

    @classmethod
    def path_to_file(cls) -> Path:
        """Get the storage directory for multi-core LDA models."""
        return super().path_to_file().joinpath('multi-core')

    def generate_model(self, doc_term_matrix: list, id2word: Dictionary, is_parallel: bool = False) -> Union[
        LdaModel, LdaMulticore]:
        """Generate LDA model, using single-core if in parallel context to avoid nesting."""
        if is_parallel:
            # Use single-core to avoid daemonic process issues
            return LdaModel(doc_term_matrix, num_topics=self.lda_no_topics, id2word=id2word,
                            eval_every=1, alpha=self.alpha, random_state=0)
        else:
            return LdaMulticore(doc_term_matrix, num_topics=self.lda_no_topics, id2word=id2word,
                                workers=os.cpu_count() - 1, alpha=self.alpha, random_state=0)

    @classmethod
    def _load_model(cls, file_path: Path) -> Union[LdaModel, LdaMulticore]:
        """Load LDA model (detect type from file)."""
        try:
            return LdaMulticore.load(str(file_path))
        except:
            return LdaModel.load(str(file_path))


class CustomLDA:
    """Custom LDA implementation using NumPy for topic modeling."""
    FILE_NAME = 'a={a},b={b},mi={mi},nt={nt}'
    FILE_EXTENSION = 'lda'

    def __init__(self, docs: list, id2word: dict, word2id: dict, lda_no_topics: int, max_iterations: int, alpha: float,
                 beta: float):
        """Initialize the custom LDA model."""
        self.docs = docs
        self.id2word = id2word
        self.word2id = word2id
        self.lda_no_topics = lda_no_topics
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.zz = []
        D = len(docs)
        W = len(word2id)
        self.theta = np.zeros([D, self.lda_no_topics]) + alpha
        self.phi = np.zeros([self.lda_no_topics, W]) + beta
        self.nz = np.zeros([self.lda_no_topics]) + W * beta

    @staticmethod
    def load_from(file_path: str, alpha: float, beta: float, max_iterations: int, lda_no_topics: int) -> Optional[
        'CustomLDA']:
        """Load a custom LDA model from a file."""
        full_path = Path(__file__).parent.joinpath(file_path,
                                                   CustomLDA.FILE_NAME.format(a=alpha, b=beta, mi=max_iterations,
                                                                              nt=lda_no_topics))
        if not full_path.exists():
            raise FileNotFoundError(f"Model file not found: {full_path}")
        try:
            with open(full_path, 'rb') as file:
                return pickle.load(file)
        except pickle.PickleError as e:
            print(f"Error loading model from {full_path}: {e}")
            return None

    def save_to(self, file_path: str):
        """Save the custom LDA model to a file."""
        full_path = Path(__file__).parent.joinpath(file_path,
                                                   f'{CustomLDA.FILE_NAME.format(a=self.alpha, b=self.beta, mi=self.max_iterations, nt=self.lda_no_topics)}.{CustomLDA.FILE_EXTENSION}')
        with open(full_path, 'wb') as file:
            pickle.dump(self, file)


def generate_bow() -> list[list[str]]:
    """Generate a list of bag-of-words for each paper, with caching."""
    logger = setup_logger()
    cache_path = lda_output_path.joinpath('bow.pkl')
    if cache_path.exists() and flag.is_module_done('generate_bow'):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached BOW: {e}")
    docs = [[] for _ in range(Paper.select().count())]
    for doc_term in tqdm(PaperTechnicalTerm.select(PaperTechnicalTerm, TechnicalTerm).join(TechnicalTerm).iterator(),
                         desc="Preparing initial corpus..."):
        docs[doc_term.paper_id - 1] += [doc_term.technical_term.text] * doc_term.frequency
    with open(cache_path, 'wb') as f:
        pickle.dump(docs, f)
    flag.mark_module_done('generate_bow')
    return docs


def define_subparser(subparser):
    """Define command-line arguments for the LDA module."""
    lda = subparser.add_parser(subparser_name := 'lda')
    lda.add_argument('--lda-no-topics', dest='lda_no_topics', type=int, default=20, help='Number of topics for LDA')
    lda.add_argument('--bertopic-no-topics', dest='bertopic_no_topics', type=int, default=5,
                     help='Number of topics for BERTopic')
    lda.add_argument('--no-topics', dest='bertopic_no_topics', type=int, default=5,
                     help='Number of topics for BERTopic (alias)')
    lda.add_argument('-v', dest='verbose', action='store_true', help='Activate verbose mode.')
    lda.add_argument('--verbose', dest='verbose', action='store_true', help='Activate verbose mode.')
    lda.add_argument('--fresh', dest='fresh', action='store_true', help='Force recomputation, ignoring cached results.')
    return subparser_name


def generate_bow_for_cluster(docs: List[List[str]]) -> list:
    """Generate bag-of-words for a specific cluster of documents."""
    dictionary = Dictionary(docs)
    return [dictionary.doc2bow(doc) for doc in docs]


def train_lda_cluster(args: tuple) -> tuple[int, Optional[GensimLDA]]:
    """Train an LDA model for a specific cluster."""
    cluster, docs_in_cluster, lda_no_topics, verbose, fresh = args
    logger = setup_logger()
    cluster_flag = f'lda_cluster_{cluster}_{lda_no_topics}'
    if not fresh and flag.is_module_done(cluster_flag):
        logger.info(f"Loading cached LDA model for cluster {cluster} ({lda_no_topics} topics)")
        restored = GensimMcLDA.restore_model(lda_no_topics, cluster)
        if restored is not None:
            return cluster, restored
        logger.warning(f"Flag {cluster_flag} was set but model files are missing; retraining")
    if not docs_in_cluster or len(docs_in_cluster) < 10:
        logger.warning(f"Skipping empty or small cluster {cluster}")
        return cluster, None
    dictionary = Dictionary(docs_in_cluster)
    lda = GensimMcLDA(lda_no_topics, docs=docs_in_cluster, dictionary=dictionary, verbose=verbose, cluster=cluster)
    # Pass is_parallel=True since running in outer Pool
    lda.model = lda.generate_model(lda.bag_of_words, dictionary, is_parallel=True)
    lda_results = lda.process(fresh)
    lda.save_model(cluster)
    flag.mark_module_done(cluster_flag)
    logger.info(f"LDA model for cluster {cluster} saved")
    return cluster, lda_results


def _load_cached_lda_results(bertopic_no_topics: int, lda_no_topics: int) -> Dict[int, GensimLDA]:
    """Try to restore fully-cached LDA models for either mode. Returns {} on any cache miss.

    Flag names are scoped by lda_no_topics (below) so that a cached model at
    one topic count is never mistaken for a cache hit at a different topic
    count -- a bare 'lda_cluster_0' flag would be indistinguishable between,
    say, a 5-topic and a 200-topic run.
    """
    logger = setup_logger()
    # Flat mode has a single cluster (0); hierarchical has one per BERTopic topic.
    check_flag = f'lda_cluster_0_{lda_no_topics}' if bertopic_no_topics == 0 else f'lda_setup_{lda_no_topics}'
    if not flag.is_module_done(check_flag):
        return {}

    clusters_to_load = [0] if bertopic_no_topics == 0 else range(bertopic_no_topics)
    lda_results = {}
    for cluster in clusters_to_load:
        if flag.is_module_done(f'lda_cluster_{cluster}_{lda_no_topics}'):
            loaded_model = GensimMcLDA.restore_model(lda_no_topics, cluster)
            if loaded_model is not None:
                lda_results[cluster] = loaded_model
            else:
                logger.warning(f"Could not load cluster {cluster}, will regenerate")
    return lda_results


def _cluster_docs_by_bertopic_topic(docs: List[List[str]], paper_topics_df) -> Dict[int, list]:
    """Group each document's bag-of-words by its assigned BERTopic topic (outliers, topic -1, dropped)."""
    clustered_docs = defaultdict(list)
    topics = paper_topics_df['topic'].tolist()
    for i, doc in enumerate(docs):
        if topics[i] != -1:
            clustered_docs[topics[i]].append(doc)
    return clustered_docs


def _setup_flat(lda_no_topics: int, verbose: bool, fresh: bool) -> Dict[int, GensimLDA]:
    """Train a single LDA model over the whole corpus. No BERTopic involved at all."""
    logger = setup_logger()
    database_setup()
    docs = generate_bow()
    logger.info("Flat mode: training a single LDA model over the whole corpus")
    _, model = train_lda_cluster((0, docs, lda_no_topics, verbose, fresh))
    flag.mark_module_done(f'lda_setup_{lda_no_topics}')
    return {0: model} if model is not None else {}


def _setup_hierarchical(lda_no_topics: int, bertopic_no_topics: int, verbose: bool, fresh: bool) -> Dict[
        int, GensimLDA]:
    """Cluster the corpus with BERTopic first, then train one LDA model per cluster."""
    from components.bertopic import setup as bertopic_setup  # lazy: only hierarchical mode needs it

    logger = setup_logger()
    database_setup()
    docs = generate_bow()

    # Cache path and flag are scoped by bertopic_no_topics -- a bare filename
    # here would silently reuse another topic count's clustering.
    cache_path = lda_output_path.joinpath(f'bertopic_results_{bertopic_no_topics}.pkl')
    bertopic_flag = f'bertopic_setup_{bertopic_no_topics}'
    if cache_path.exists() and flag.is_module_done(bertopic_flag) and not fresh:
        try:
            with open(cache_path, 'rb') as f:
                topic_info_df, paper_topics_df = pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached BERTopic results: {e}")
            os.remove(cache_path)
            topic_info_df, paper_topics_df = bertopic_setup(bertopic_no_topics, fresh=fresh)
    else:
        topic_info_df, paper_topics_df = bertopic_setup(bertopic_no_topics, fresh=fresh)
        with open(cache_path, 'wb') as f:
            pickle.dump((topic_info_df, paper_topics_df), f)
        flag.mark_module_done(bertopic_flag)

    if topic_info_df.empty or paper_topics_df.empty:
        logger.error("BERTopic failed to generate topics")
        return {}

    clustered_docs_cache = lda_output_path.joinpath(f'clustered_docs_{bertopic_no_topics}.pkl')
    clustered_docs_flag = f'clustered_docs_{bertopic_no_topics}'
    if clustered_docs_cache.exists() and flag.is_module_done(clustered_docs_flag) and not fresh:
        try:
            with open(clustered_docs_cache, 'rb') as f:
                clustered_docs = pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached clustered docs: {e}")
            os.remove(clustered_docs_cache)
            clustered_docs = _cluster_docs_by_bertopic_topic(docs, paper_topics_df)
    else:
        clustered_docs = _cluster_docs_by_bertopic_topic(docs, paper_topics_df)
        with open(clustered_docs_cache, 'wb') as f:
            pickle.dump(clustered_docs, f)
        flag.mark_module_done(clustered_docs_flag)

    # Run serially (not via an outer Pool) to avoid nesting issues with LdaMulticore's own Pool.
    logger.info("Hierarchical mode: running LDA clusters serially to avoid daemonic process errors")
    lda_results = {}
    for cluster, docs_in_cluster in tqdm(clustered_docs.items(), desc="Training LDA on clusters"):
        lda_results[cluster] = train_lda_cluster((cluster, docs_in_cluster, lda_no_topics, verbose, fresh))[1]
    flag.mark_module_done(f'lda_setup_{lda_no_topics}')
    return {k: v for k, v in lda_results.items() if v is not None}


def setup(lda_no_topics: int, bertopic_no_topics: int, verbose: bool = False, fresh: bool = False) -> Dict[
    int, GensimLDA]:
    """
    Initialize LDA topic models, with full caching.

    ``bertopic_no_topics == 0`` selects flat mode: a single LDA model trained
    over the whole corpus, with no dependency on BERTopic whatsoever (it is
    not imported, and no BERTopic run is triggered).

    ``bertopic_no_topics > 0`` selects hierarchical mode: BERTopic first
    clusters the corpus, then one LDA model is trained per BERTopic cluster.
    """
    logger = setup_logger()

    if not fresh:
        cached = _load_cached_lda_results(bertopic_no_topics, lda_no_topics)
        if cached:
            logger.info(f"Loaded {len(cached)} cached LDA models (bertopic_no_topics={bertopic_no_topics})")
            print(f"[LDA] Loaded {len(cached)} cached clusters from flags")
            return cached
        logger.warning("No cached models found, regenerating...")
        print("[LDA] No cached models loaded, regenerating...")

    if bertopic_no_topics == 0:
        return _setup_flat(lda_no_topics, verbose, fresh)
    return _setup_hierarchical(lda_no_topics, bertopic_no_topics, verbose, fresh)
