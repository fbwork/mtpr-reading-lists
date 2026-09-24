"""
Baseline methods for academic paper recommendation comparison.

Implements baseline methods mentioned in RESEARCH_ANALYSIS.md:
- BM25 (Okapi BM25)
- Doc2Vec (Gensim)
- Collaborative Filtering
- BERT-based (SciBERT, SPECTER)
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter

import utils.flag as flag
from models.tables import Paper, PaperCitation, PaperTechnicalTerm, TechnicalTerm

baseline_output_path = Path(__file__).parent.joinpath('output', 'baselines')
baseline_log_path = Path(__file__).parent.parent.joinpath('logs', 'baselines.log')


def _query_ngrams(query: str, max_n: int = 3) -> List[str]:
    """Tokenize a query into word n-grams (1..max_n).

    The BM25/Doc2Vec corpora are built from PaperTechnicalTerm, whose entries
    are extracted multi-word phrases (e.g. "graph neural network"), not single
    words -- see components/atr.py:generate_n_grams. A plain query.split()
    only ever produces single-word tokens, which almost never match a
    multi-word technical-term string, so every query would score near-zero
    regardless of relevance. Generating the same word n-grams from the query
    lets it match phrase-level terms too.
    """
    tokens = query.lower().split()
    result = []
    for n in range(1, min(max_n, len(tokens)) + 1):
        for i in range(len(tokens) - n + 1):
            result.append(' '.join(tokens[i:i + n]))
    return result


def setup_logger():
    baseline_log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=baseline_log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()


class BM25Baseline:
    """
    Okapi BM25 baseline for paper recommendation.
    
    Uses rank_bm25 library for efficient BM25 scoring.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 baseline.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.paper_ids = []
        self.bm25 = None
        
    def build_index(self, fresh: bool = False):
        """Build BM25 index from papers."""
        logger = setup_logger()
        cache_path = baseline_output_path.joinpath('bm25_index.pkl')
        
        if not fresh and cache_path.exists() and flag.is_module_done('bm25_index'):
            logger.info("Loading cached BM25 index...")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.corpus = data['corpus']
                self.paper_ids = data['paper_ids']
                self.bm25 = data['bm25']
            return
        
        logger.info("Building BM25 index...")
        
        # Build corpus from technical terms
        self.corpus = []
        self.paper_ids = []
        
        for paper in Paper.select().iterator():
            terms = []
            for doc_term in PaperTechnicalTerm.select(PaperTechnicalTerm, TechnicalTerm)\
                    .join(TechnicalTerm)\
                    .where(PaperTechnicalTerm.paper_id == paper.id):
                terms.extend([doc_term.technical_term.text] * doc_term.frequency)
            
            # Only add non-empty documents
            if terms:
                self.corpus.append(terms)
                self.paper_ids.append(paper.id)
        
        # Check if corpus is empty
        if not self.corpus:
            raise ValueError("No documents with technical terms found. Cannot build BM25 index.")
        
        # Create BM25 model
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
        except ImportError as e:
            logger.error("rank_bm25 not installed. Install with: pip install rank-bm25")
            print("\n" + "="*80)
            print("ERROR: Missing dependency 'rank_bm25'")
            print("="*80)
            print("To install baseline dependencies, run:")
            print("  pip install -r baseline_requirements.txt")
            print("\nOr install individually:")
            print("  pip install rank-bm25")
            print("="*80 + "\n")
            raise ImportError("rank_bm25 is required for BM25Baseline") from e
        
        # Cache
        baseline_output_path.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'corpus': self.corpus,
                'paper_ids': self.paper_ids,
                'bm25': self.bm25
            }, f)
        flag.mark_module_done('bm25_index')
        logger.info(f"BM25 index built with {len(self.corpus)} papers")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search papers using BM25.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (paper_id, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        # Tokenize query into the same word n-grams the corpus was built from
        query_terms = _query_ngrams(query)

        # Get scores
        scores = self.bm25.get_scores(query_terms)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.paper_ids[idx], scores[idx]) for idx in top_indices]
        
        return results


class Doc2VecBaseline:
    """
    Doc2Vec baseline for paper recommendation.
    
    Uses Gensim's Doc2Vec for document embeddings.
    """
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2, epochs: int = 20):
        """
        Initialize Doc2Vec baseline.
        
        Args:
            vector_size: Dimensionality of document vectors
            window: Context window size
            min_count: Minimum word frequency
            epochs: Training epochs
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
        self.paper_ids = []
        
    def build_model(self, fresh: bool = False):
        """Build Doc2Vec model from papers."""
        logger = setup_logger()
        cache_path = baseline_output_path.joinpath('doc2vec_model.bin')
        
        if not fresh and cache_path.exists() and flag.is_module_done('doc2vec_model'):
            logger.info("Loading cached Doc2Vec model...")
            from gensim.models import Doc2Vec
            self.model = Doc2Vec.load(str(cache_path))
            
            # Load paper IDs
            paper_ids_path = baseline_output_path.joinpath('doc2vec_paper_ids.pkl')
            with open(paper_ids_path, 'rb') as f:
                self.paper_ids = pickle.load(f)
            return
        
        logger.info("Building Doc2Vec model...")
        
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        
        # Prepare documents
        documents = []
        self.paper_ids = []
        
        for paper in Paper.select().iterator():
            terms = []
            for doc_term in PaperTechnicalTerm.select(PaperTechnicalTerm, TechnicalTerm)\
                    .join(TechnicalTerm)\
                    .where(PaperTechnicalTerm.paper_id == paper.id):
                terms.extend([doc_term.technical_term.text] * doc_term.frequency)
            
            if terms:
                documents.append(TaggedDocument(words=terms, tags=[str(paper.id)]))
                self.paper_ids.append(paper.id)
        
        # Check if we have documents
        if not documents:
            raise ValueError("No documents with technical terms found. Cannot build Doc2Vec model. Run 'python main.py atr --fresh --force' first.")
        
        # Train model
        self.model = Doc2Vec(
            documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=4
        )
        
        # Cache
        baseline_output_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(cache_path))
        
        paper_ids_path = baseline_output_path.joinpath('doc2vec_paper_ids.pkl')
        with open(paper_ids_path, 'wb') as f:
            pickle.dump(self.paper_ids, f)
        
        flag.mark_module_done('doc2vec_model')
        logger.info(f"Doc2Vec model trained with {len(documents)} papers")
    
    def _frozen_query_vector(self, query: str) -> np.ndarray:
        """
        Return a fixed, versioned query vector for `query`, computed once and
        cached to disk.

        Resetting model.random to a fixed seed before infer_vector() makes
        repeated calls WITHIN one process identical, but does not make
        infer_vector() reproducible ACROSS process launches (verified
        empirically: OCSC varied 0.320-0.337 across independent runs even
        with the seed reset) -- the residual noise is floating-point-level,
        not RNG-state-level, and gensim's own documentation recommends
        averaging multiple inference passes to stabilise infer_vector().
        We do that once per query (20 passes, seeded 0..19) and freeze the
        average, rather than recomputing it -- and paying its residual
        instability -- on every evaluation run.
        """
        import hashlib
        import gensim.utils as gensim_utils

        cache_path = baseline_output_path.joinpath('doc2vec_query_vectors.pkl')
        cache: Dict[str, np.ndarray] = {}
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)

        query_terms = _query_ngrams(query)
        key = hashlib.sha256(' '.join(query_terms).encode('utf-8')).hexdigest()
        if key in cache:
            return cache[key]

        n_passes = 20
        vectors = []
        for seed in range(n_passes):
            self.model.random = gensim_utils.get_random_state(seed)
            vectors.append(self.model.infer_vector(query_terms))
        averaged = np.mean(vectors, axis=0)

        cache[key] = averaged
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        return averaged

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search papers using Doc2Vec.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (paper_id, score) tuples
        """
        if self.model is None:
            raise ValueError("Doc2Vec model not built. Call build_model() first.")

        query_vector = self._frozen_query_vector(query)

        # Find most similar
        similar = self.model.dv.most_similar([query_vector], topn=top_k)
        
        # Convert to paper IDs and scores
        results = [(int(tag), score) for tag, score in similar]
        
        return results


class CollaborativeFilteringBaseline:
    """
    Collaborative Filtering baseline for paper recommendation.
    
    Uses citation co-occurrence as implicit feedback.
    """
    
    def __init__(self, n_factors: int = 50):
        """
        Initialize Collaborative Filtering baseline.
        
        Args:
            n_factors: Number of latent factors
        """
        self.n_factors = n_factors
        self.paper_factors = None
        self.paper_ids = []
        
    def build_model(self, fresh: bool = False):
        """Build CF model from citation patterns."""
        logger = setup_logger()
        cache_path = baseline_output_path.joinpath('cf_model.pkl')
        
        if not fresh and cache_path.exists() and flag.is_module_done('cf_model'):
            logger.info("Loading cached CF model...")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.paper_factors = data['paper_factors']
                self.paper_ids = data['paper_ids']
            return
        
        logger.info("Building CF model...")
        
        # Build citation matrix
        papers = list(Paper.select())
        self.paper_ids = [p.id for p in papers]
        n_papers = len(papers)
        
        # Create paper-to-index mapping
        paper_to_idx = {p.id: idx for idx, p in enumerate(papers)}
        
        # Build interaction matrix (citations as edges)
        from scipy.sparse import lil_matrix
        interaction_matrix = lil_matrix((n_papers, n_papers))
        
        for citation in PaperCitation.select():
            if citation.citing_paper_id in paper_to_idx and citation.cited_paper_id in paper_to_idx:
                citing_idx = paper_to_idx[citation.citing_paper_id]
                cited_idx = paper_to_idx[citation.cited_paper_id]
                interaction_matrix[citing_idx, cited_idx] = 1
        
        # Use SVD for matrix factorization
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.paper_factors = svd.fit_transform(interaction_matrix)
        
        # Cache
        baseline_output_path.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'paper_factors': self.paper_factors,
                'paper_ids': self.paper_ids
            }, f)
        
        flag.mark_module_done('cf_model')
        logger.info(f"CF model built with {n_papers} papers and {self.n_factors} factors")
    
    def recommend(self, paper_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Recommend papers based on citation similarity.
        
        Args:
            paper_id: Source paper ID
            top_k: Number of recommendations
            
        Returns:
            List of (paper_id, score) tuples
        """
        if self.paper_factors is None:
            raise ValueError("CF model not built. Call build_model() first.")
        
        # Find paper index
        try:
            paper_idx = self.paper_ids.index(paper_id)
        except ValueError:
            return []
        
        # Get paper vector
        paper_vector = self.paper_factors[paper_idx]
        
        # Compute similarities
        similarities = np.dot(self.paper_factors, paper_vector)
        
        # Get top-k (excluding itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        results = [(self.paper_ids[idx], similarities[idx]) for idx in top_indices]
        
        return results


class BERTBaseline:
    """
    BERT-based baseline for paper recommendation.
    
    Uses SciBERT or SPECTER for semantic embeddings.
    """
    
    def __init__(self, model_name: str = 'allenai/scibert_scivocab_uncased'):
        """
        Initialize BERT baseline.
        
        Args:
            model_name: Hugging Face model name
                - 'allenai/scibert_scivocab_uncased' (SciBERT)
                - 'allenai/specter' (SPECTER)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.embeddings = None
        self.paper_ids = []
        
    def build_embeddings(self, fresh: bool = False):
        """Build paper embeddings using BERT."""
        logger = setup_logger()
        cache_path = baseline_output_path.joinpath(f'{self.model_name.replace("/", "_")}_embeddings.pkl')
        
        if not fresh and cache_path.exists() and flag.is_module_done(f'{self.model_name}_embeddings'):
            logger.info(f"Loading cached {self.model_name} embeddings...")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.paper_ids = data['paper_ids']
            return
        
        logger.info(f"Building {self.model_name} embeddings...")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers torch")
            raise
        
        # Load model with GPU support
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU not available, using CPU (this will be slow)")
            print("[!] GPU not available, using CPU (this will be slower)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(device)
        
        # Generate embeddings
        self.embeddings = []
        self.paper_ids = []
        
        batch_size = 16
        papers = list(Paper.select())
        
        for i in range(0, len(papers), batch_size):
            batch_papers = papers[i:i+batch_size]
            texts = [p.title + ' ' + (p.body or '') for p in batch_papers]
            
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            self.embeddings.extend(batch_embeddings)
            self.paper_ids.extend([p.id for p in batch_papers])
            
            if (i // batch_size) % 10 == 0:
                progress_pct = ((i + len(batch_papers)) / len(papers)) * 100
                logger.info(f"Processed {i + len(batch_papers)}/{len(papers)} papers ({progress_pct:.1f}%)")
                print(f"  Progress: {i + len(batch_papers)}/{len(papers)} papers ({progress_pct:.1f}%)")
        
        self.embeddings = np.array(self.embeddings)
        
        # Cache
        baseline_output_path.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'paper_ids': self.paper_ids
            }, f)
        
        flag.mark_module_done(f'{self.model_name}_embeddings')
        logger.info(f"{self.model_name} embeddings built for {len(self.paper_ids)} papers")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search papers using BERT embeddings.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (paper_id, score) tuples
        """
        if self.embeddings is None:
            raise ValueError("BERT embeddings not built. Call build_embeddings() first.")
        
        try:
            import torch
        except ImportError:
            raise ImportError("torch not installed")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Encode query
        inputs = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        # Compute cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.paper_ids[idx], similarities[idx]) for idx in top_indices]
        
        return results


def setup_all_baselines(fresh: bool = False):
    """
    Setup all baseline methods.
    
    Args:
        fresh: Whether to rebuild from scratch
        
    Returns:
        Dictionary of baseline instances
    """
    logger = setup_logger()
    logger.info("Setting up all baseline methods...")
    
    baselines = {}
    errors = []
    
    # BM25
    try:
        logger.info("Setting up BM25...")
        print("\n[BM25] Setting up BM25 baseline...")
        bm25 = BM25Baseline()
        bm25.build_index(fresh=fresh)
        baselines['bm25'] = bm25
        print("[OK] BM25 setup complete")
    except (ImportError, ValueError) as e:
        errors.append(('BM25', str(e)))
        logger.warning(f"Skipping BM25: {e}")
        print(f"[!] BM25 setup failed: {e}")
        if "No documents with technical terms found" in str(e):
            print("[!] Run 'python main.py atr --fresh --force' to generate technical terms first.")
    
    # Doc2Vec
    try:
        logger.info("Setting up Doc2Vec...")
        print("\n[Doc2Vec] Setting up Doc2Vec baseline...")
        doc2vec = Doc2VecBaseline()
        doc2vec.build_model(fresh=fresh)
        baselines['doc2vec'] = doc2vec
        print("[OK] Doc2Vec setup complete")
    except (ValueError, Exception) as e:
        errors.append(('Doc2Vec', str(e)))
        logger.warning(f"Skipping Doc2Vec: {e}")
        print(f"[!] Doc2Vec setup failed: {e}")
        if "No documents with technical terms found" in str(e):
            print("[!] Run 'python main.py atr --fresh --force' to generate technical terms first.")
    
    # Collaborative Filtering
    try:
        logger.info("Setting up Collaborative Filtering...")
        print("\n[CF] Setting up Collaborative Filtering baseline...")
        cf = CollaborativeFilteringBaseline()
        cf.build_model(fresh=fresh)
        baselines['cf'] = cf
        print("[OK] Collaborative Filtering setup complete")
    except Exception as e:
        errors.append(('Collaborative Filtering', str(e)))
        logger.warning(f"Skipping CF: {e}")
    
    # BERT-based (optional, may take time)
    try:
        logger.info("Setting up SciBERT...")
        print("\n[SciBERT] Setting up SciBERT baseline (may take 30-120 min)...")
        scibert = BERTBaseline(model_name='allenai/scibert_scivocab_uncased')
        scibert.build_embeddings(fresh=fresh)
        baselines['scibert'] = scibert
        print("[OK] SciBERT setup complete")
    except Exception as e:
        errors.append(('SciBERT', str(e)))
        logger.warning(f"Skipping SciBERT: {e}")
    
    try:
        logger.info("Setting up SPECTER...")
        print("\n[SPECTER] Setting up SPECTER baseline (may take 30-120 min)...")
        specter = BERTBaseline(model_name='allenai/specter')
        specter.build_embeddings(fresh=fresh)
        baselines['specter'] = specter
        print("[OK] SPECTER setup complete")
    except Exception as e:
        errors.append(('SPECTER', str(e)))
        logger.warning(f"Skipping SPECTER: {e}")
    
    # Summary
    print("\n" + "="*80)
    print(f"Baseline setup complete: {len(baselines)}/{5} methods successfully initialized")
    if errors:
        print("\nSkipped baselines due to errors:")
        for name, error in errors:
            print(f"  [X] {name}: {error}")
        print("\nTo install missing dependencies:")
        print("  pip install -r baseline_requirements.txt")
    print("="*80 + "\n")
    
    logger.info(f"Baselines setup complete: {len(baselines)} methods")
    return baselines


def define_subparser(subparser):
    """Define command-line arguments for baseline module."""
    baselines = subparser.add_parser(subparser_name := 'baselines')
    
    baselines.add_argument(
        '--method',
        type=str,
        choices=['bm25', 'doc2vec', 'cf', 'scibert', 'specter', 'all'],
        default='all',
        help='Baseline method to setup'
    )
    
    baselines.add_argument(
        '--fresh',
        dest='fresh',
        action='store_true',
        help='Force recomputation, ignoring cached results'
    )
    
    return subparser_name