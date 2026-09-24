import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from components.lda import GensimLDA
from models.tables import Paper, TechnicalTerm, db

current_path = Path(__file__).parent
query_output_path = current_path.joinpath('output', 'query')
queries = ['machine learning', 'natural language processing', 'graph neural networks']


def setup_logger() -> logging.Logger:
    """Initialize logger"""
    try:
        query_output_path.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=query_output_path.joinpath('query.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger()
    except OSError as e:
        logging.error(f"Failed to set up logger: {e}")
        raise


def _query_ngrams(query: str, max_n: int = 3) -> List[str]:
    """Tokenize a query into word n-grams (1..max_n).

    The LDA dictionary is built from PaperTechnicalTerm, whose entries are
    extracted multi-word phrases (e.g. "graph neural network"), not single
    words -- see components/atr.py:generate_n_grams and
    components/lda.py's `docs[...] += [doc_term.technical_term.text] * ...`.
    A plain query.split() only ever produces single-word tokens, which almost
    never match a multi-word dictionary entry, so doc2bow(tokens) comes back
    empty for nearly every real query. Generating the same word n-grams from
    the query lets it match phrase-level terms too.
    """
    tokens = query.lower().split()
    result = []
    for n in range(1, min(max_n, len(tokens)) + 1):
        for i in range(len(tokens) - n + 1):
            result.append(' '.join(tokens[i:i + n]))
    return result


NEIGHBOR_K = 20  # Jardine (2014, UCAM-CL-TR-848, Sec 3.1.3/5.2.1): average
                 # theta_d of the top-20 TF-IDF-retrieved neighbor papers.

_tfidf_index_cache = None


def _get_tfidf_index():
    """Lazily build the Lucene-classic TF-IDF index over (title+body), used to
    find query neighbors for Jardine's query-topic mechanism. Built once per
    process; deterministic given a fixed corpus (no randomness anywhere in
    term counting or weighting).
    """
    global _tfidf_index_cache
    if _tfidf_index_cache is not None:
        return _tfidf_index_cache

    from sklearn.feature_extraction.text import CountVectorizer

    db.connect(reuse_if_open=True)
    try:
        all_papers = list(Paper.select(Paper.id, Paper.title, Paper.body).order_by(Paper.id).iterator())
    finally:
        if not db.is_closed():
            db.close()

    doc_texts = [(p.title or '') + ' ' + (p.body or '') for p in all_papers]
    docid_order = [p.id - 1 for p in all_papers]

    # Lucene-classic TFIDFSimilarity (Jardine used Lucene.NET v2.9.2), NOT
    # sklearn's smoothed-idf/L2-normalized TfidfVectorizer:
    #   tf(t,d)    = sqrt(freq(t,d))
    #   idf(t)     = 1 + ln(N / (df(t) + 1))
    #   norm(d)    = 1 / sqrt(total token count in d)
    #   coord(q,d) = |query terms found in d| / |query terms|
    #   score(q,d) = coord(q,d) * norm(d) * sum_{t in q} tf(t,d) * idf(t)^2
    count_vectorizer = CountVectorizer(lowercase=True, stop_words='english', max_features=100_000)
    tf_counts = count_vectorizer.fit_transform(doc_texts).tocsr()
    n = tf_counts.shape[0]

    doc_freq = np.asarray((tf_counts > 0).sum(axis=0)).ravel()
    idf = 1.0 + np.log(n / (doc_freq + 1.0))
    idf_sq = idf ** 2

    tf_sqrt = tf_counts.astype(np.float64)
    tf_sqrt.data = np.sqrt(tf_sqrt.data)
    weighted = tf_sqrt.multiply(idf_sq).tocsr()

    num_tokens = np.asarray(tf_counts.sum(axis=1)).ravel()
    doc_norm = 1.0 / np.sqrt(np.maximum(num_tokens, 1))

    _tfidf_index_cache = (count_vectorizer, weighted, doc_norm, docid_order)
    return _tfidf_index_cache


def _tfidf_neighbors(query: str) -> List[int]:
    """Deterministic top-NEIGHBOR_K doc_ids by Lucene-classic TF-IDF score.
    Ties broken by ascending doc_id (stable sort over doc-id-ordered scores)
    so the result is reproducible regardless of platform/BLAS sort behavior.
    """
    count_vectorizer, weighted, doc_norm, docid_order = _get_tfidf_index()
    query_vec = count_vectorizer.transform([query])
    query_terms = query_vec.indices
    if len(query_terms) == 0:
        return []

    matched = weighted[:, query_terms]
    raw_scores = np.asarray(matched.sum(axis=1)).ravel()
    terms_matched = np.asarray((matched > 0).sum(axis=1)).ravel()
    coord = terms_matched / len(query_terms)
    scores = raw_scores * coord * doc_norm

    order = np.argsort(-scores, kind='stable')
    top_rows = order[:NEIGHBOR_K]
    return [docid_order[r] for r in top_rows if scores[r] > 0]


def query_topic_weights(query: str, lda_results: Dict[int, GensimLDA]) -> Dict[int, float]:
    """Jardine's deterministic query-topic mechanism (UCAM-CL-TR-848,
    Sec 3.1.3/5.2.1, "unknown words or technical terms"): Lucene-classic
    TF-IDF full-text retrieval over (title+body) -> top NEIGHBOR_K most
    similar papers -> average THEIR already-frozen theta_d.

    Deterministic end to end: no LDA inference is ever run on the query text
    (unlike the previous implementation, which called
    lda.model.get_document_topics() on a query bag-of-words -- confirmed to
    be non-deterministic across repeated calls with identical input). Neighbor
    theta_d values come from GensimLDA.get_doc_topics(), which reads the
    frozen theta artifact (components/theta_artifact.py), not live inference.
    """
    combined = defaultdict(float)
    top_doc_ids = _tfidf_neighbors(query)

    if top_doc_ids:
        for lda in lda_results.values():
            for doc_id in top_doc_ids:
                for topic_id, weight in lda.get_doc_topics(doc_id).items():
                    combined[topic_id] += weight
    else:
        # No TF-IDF match at all (e.g. query is entirely out-of-vocabulary or
        # stopwords): fall back to the trained model's own term-topic
        # associations. get_term_topics() is a deterministic function of the
        # fixed, already-trained word-topic matrix -- no per-call inference,
        # so this fallback introduces no non-determinism of its own.
        tokens = _query_ngrams(query)
        for lda in lda_results.values():
            for token in tokens:
                if token in lda.word2id:
                    for topic_id, weight in lda.get_term_topics(token).items():
                        combined[topic_id] += float(weight)

    total = sum(combined.values())
    return {topic_id: weight / total for topic_id, weight in combined.items()} if total > 0 else {}


def rank_pagerank_for_query(query: str, pr_results: Dict[int, Dict[int, float]],
                            lda_results: Dict[int, GensimLDA]) -> List[int]:
    """Return a query-specific PageRank ranking using LDA topic weights."""
    weights = query_topic_weights(query, lda_results)
    if not weights:
        return []

    scores = {
        doc_id: sum(weights.get(topic_id, 0.0) * score for topic_id, score in topic_scores.items())
        for doc_id, topic_scores in pr_results.items()
    }
    return sorted(scores, key=scores.get, reverse=True)


class QueryResult:
    """Class to store query results"""

    def __init__(self, query: str, papers: List[Paper],
                 standard: Optional[str], ranked_paper_ids: List[int]):
        self.query = query
        self.papers = papers
        self.standard = standard
        self.ranked_paper_ids = ranked_paper_ids


class QueryManager:
    """Manages different query/recommendation strategies"""

    def __init__(self, page_rank, lda_results: Dict[int, GensimLDA],
                 papers: Optional[List[Paper]] = None, n_components: int = 100):
        """
        Initialize QueryManager.

        Args:
            page_rank: PageRank instance (PaperRankClassic or PaperRankGAT)
            lda_results: Dictionary of LDA models by cluster
            papers: List of papers (optional)
            n_components: Number of SVD components
        """
        logger = setup_logger()

        self.page_rank = page_rank
        self.lda_results = lda_results
        self.author_graph = page_rank.author_graph if hasattr(page_rank, 'author_graph') else None
        self.papers = papers or list(Paper.select())
        self.n_components = max(n_components, 1)

        # Initialize TF-IDF and SVD
        self._tfidf_matrix = None
        self._doc_vecs = None
        self._vectorizer = None
        self._svd = None

    def _get_tfidf_matrix(self):
        """Lazy loading of TF-IDF matrix"""
        if self._tfidf_matrix is None:
            logger = setup_logger()
            try:
                db.connect(reuse_if_open=True)
                self._vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                texts = [
                    f"{paper.title or ''} {paper.body or ''}".strip()
                    for paper in self.papers
                    if paper.title or paper.body
                ]

                if texts:
                    self._tfidf_matrix = self._vectorizer.fit_transform(texts)
                    self._svd = TruncatedSVD(n_components=min(self.n_components, len(texts) - 1))
                    self._doc_vecs = self._svd.fit_transform(self._tfidf_matrix)
                    logger.info(f"TF-IDF initialized with {len(texts)} documents")
                else:
                    logger.warning("No valid texts for TF-IDF")
            except Exception as e:
                logger.error(f"Error initializing TF-IDF: {e}")
            finally:
                if not db.is_closed():
                    db.close()

        return self._tfidf_matrix

    def tfidf_recommendation(self, query: str, top_k: int = 10) -> List[Paper]:
        """TF-IDF based recommendation"""
        logger = setup_logger()
        top_k = max(top_k, 1)

        tfidf_matrix = self._get_tfidf_matrix()
        if tfidf_matrix is None or not self.papers:
            logger.warning("TF-IDF not initialized")
            return []

        try:
            query_vec = self._vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [self.papers[i] for i in top_indices if i < len(self.papers)]
        except Exception as e:
            logger.error(f"Error in TF-IDF recommendation: {e}")
            return []

    def svd_recommendation(self, query: str, top_k: int = 10) -> List[Paper]:
        """SVD/LSA based recommendation"""
        logger = setup_logger()
        top_k = max(top_k, 1)

        self._get_tfidf_matrix()  # Ensure initialized
        if self._doc_vecs is None or not self.papers:
            logger.warning("SVD not initialized")
            return []

        try:
            query_vec = self._svd.transform(self._vectorizer.transform([query]))
            similarities = cosine_similarity(query_vec, self._doc_vecs).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [self.papers[i] for i in top_indices if i < len(self.papers)]
        except Exception as e:
            logger.error(f"Error in SVD recommendation: {e}")
            return []

    def query_pagerank(self, query: str, agify_results: bool,
                       count: int, standard: Optional[str]) -> QueryResult:
        """PageRank-based recommendation"""
        logger = setup_logger()
        count = max(count, 1)

        papers, ranked_paper_ids = [], []

        try:
            db.connect(reuse_if_open=True)

            pr_results = self.page_rank.results(agify_results)
            ranked_paper_ids = rank_pagerank_for_query(
                query, pr_results, self.lda_results
            )[:count]

            if not ranked_paper_ids:
                logger.info(f"No LDA topic evidence found for query: {query}")

            for paper_id in ranked_paper_ids:
                try:
                    papers.append(Paper.get(Paper.id == paper_id + 1))
                except Paper.DoesNotExist:
                    logger.warning(f"Paper {paper_id + 1} not found")
                    continue

        except Exception as e:
            logger.error(f"Error in PageRank query: {e}")
        finally:
            if not db.is_closed():
                db.close()

        return QueryResult(query, papers, standard, ranked_paper_ids)

    def query_flow_matching(self, query: str, count: int,
                            standard: Optional[str]) -> QueryResult:
        """Flow Matching based recommendation"""
        logger = setup_logger()
        count = max(count, 1)

        papers, ranked_paper_ids = [], []

        try:
            db.connect(reuse_if_open=True)

            # Get paths from Flow Matching
            paths = self.page_rank.results(agify_results=False)

            if not paths:
                logger.info("No paths available")
                return QueryResult(query, [], standard, [])

            # Check if query is a technical term
            if TechnicalTerm.get_or_none(TechnicalTerm.text == query):
                ranking = defaultdict(float)

                for cluster, lda in self.lda_results.items():
                    term_vector = lda.get_term_topics(query)

                    # Score papers based on paths
                    for path in paths:
                        for i, paper_id in enumerate(path):
                            # Score based on:
                            # 1. Topic match
                            # 2. Position in path (earlier = better)
                            score = sum(
                                term_vector.get(t_id, 0) *
                                lda.topic_doc_score(paper_id, t_id)
                                for t_id in range(self.page_rank.no_topics)
                            )

                            # Weight by position (exponential decay)
                            position_weight = 1.0 / (i + 1)
                            ranking[paper_id] += score * position_weight

                # Sort and get top-k
                ranked_paper_ids = sorted(
                    ranking,
                    key=ranking.get,
                    reverse=True
                )[:count]

                # Retrieve papers
                for paper_id in ranked_paper_ids:
                    try:
                        papers.append(Paper.get(Paper.id == paper_id + 1))
                    except Paper.DoesNotExist:
                        logger.warning(f"Paper {paper_id + 1} not found")
                        continue
            else:
                logger.info(f"No technical term found for query: {query}")

        except Exception as e:
            logger.error(f"Error in Flow Matching query: {e}")
        finally:
            if not db.is_closed():
                db.close()

        return QueryResult(query, papers, standard, ranked_paper_ids)

    def query(self, query: str, agify_results: bool = False,
              count: int = 10, standard: Optional[str] = None,
              method: str = 'pagerank') -> QueryResult:
        """
        Main query interface.

        Args:
            query: Search query
            agify_results: Apply aging factor
            count: Number of results
            standard: Evaluation standard
            method: Recommendation method (pagerank, flow_matching, tfidf, svd)

        Returns:
            QueryResult object
        """
        if method == 'flow_matching':
            return self.query_flow_matching(query, count, standard)
        elif method == 'tfidf':
            papers = self.tfidf_recommendation(query, top_k=count)
            ranked_paper_ids = [p.id - 1 for p in papers]
            return QueryResult(query, papers, standard, ranked_paper_ids)
        elif method == 'svd':
            papers = self.svd_recommendation(query, top_k=count)
            ranked_paper_ids = [p.id - 1 for p in papers]
            return QueryResult(query, papers, standard, ranked_paper_ids)
        else:
            return self.query_pagerank(query, agify_results, count, standard)


def define_subparser(subparser) -> str:
    """Define query subparser"""
    from argparse import ArgumentTypeError

    def positive_int(value: str) -> int:
        """Validate positive integer"""
        try:
            ivalue = int(value)
            if ivalue <= 0:
                raise ValueError("Must be positive")
            return ivalue
        except ValueError as e:
            raise ArgumentTypeError(f"Invalid value: {value}. {str(e)}")

    query = subparser.add_parser('query', help='Perform recommendation queries')

    query.add_argument(
        '--query',
        type=str,
        required=True,
        help='Query string'
    )

    query.add_argument(
        '--count',
        type=positive_int,
        default=10,
        help='Number of results (default: 10)'
    )

    query.add_argument(
        '--method',
        type=str,
        choices=['pagerank', 'flow_matching', 'tfidf', 'svd'],
        default='pagerank',
        help='Recommendation method (default: pagerank)'
    )

    query.add_argument(
        '--agify',
        dest='agify_results',
        action='store_true',
        default=False,
        help='Apply aging factor (default: False)'
    )

    query.add_argument(
        '--standard',
        type=str,
        default=None,
        help='Evaluation standard (optional)'
    )

    query.add_argument(
        '--n-components',
        type=positive_int,
        default=100,
        help='SVD components (default: 100)'
    )

    return 'query'
