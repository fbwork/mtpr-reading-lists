from enum import Enum

# Whether evaluation scripts (csc_evaluation.py, csc_with_baselines.py,
# run_eval_all_fast.py) should also run/report the GAT and Flow Matching
# modes. These are outside the MTPR paper's 19-variant table (Table 1) and
# were previously found not to help; flip to True when revisiting them.
INCLUDE_EXPERIMENTAL_MODES = False


class PRModes(Enum):
    """
    PageRank modes for academic paper recommendation.

    Categories:
    1. Basic PageRank (3 modes)
    2. Topic-Modified PageRank (3 modes)
    3. Degree Centrality-Enhanced (3 modes)
    4. Closeness Centrality-Enhanced (3 modes)
    5. Betweenness Centrality-Enhanced (3 modes)
    6. Katz Centrality-Enhanced (3 modes)
    7. Hybrid Centrality (1 mode)
    8. Graph Neural Networks (10 modes) -- experimental, outside paper scope
    9. Flow-Based (1 mode) -- experimental, outside paper scope

    Total: 30 modes (19 MTPR variants + 11 experimental).
    See get_mtpr_modes() / get_experimental_modes() / get_evaluation_modes().
    """

    # ========== 1. Basic PageRank (3 modes) ==========
    NORMAL = 'normal'
    """Standard PageRank with uniform initialization"""

    PERSONALIZED = 'personalized'
    """Topic-based personalization vector (bias only)"""

    THEMED = 'themed'
    """Topic-aware transition probabilities"""

    # ========== 2. Topic-Modified PageRank (3 modes) ==========
    MODIFIED = 'modified'
    """Topic + Author scores in both bias and transition"""

    MODIFIED_BIAS = 'modified_bias'
    """Topic + Author scores in bias only"""

    MODIFIED_TRANSITION = 'modified_transition'
    """Topic + Author scores in transition only"""

    # ========== 3. Degree Centrality-Enhanced (3 modes) ==========
    D_MODIFIED = 'd_modified'
    """Degree centrality in both bias and transition"""

    D_MODIFIED_BIAS = 'd_modified_bias'
    """Degree centrality in bias only"""

    D_MODIFIED_TRANSITION = 'd_modified_transition'
    """Degree centrality in transition only"""

    # ========== 4. Closeness Centrality-Enhanced (3 modes) ==========
    CL_MODIFIED = 'cl_modified'
    """Closeness centrality in both bias and transition"""

    CL_MODIFIED_BIAS = 'cl_modified_bias'
    """Closeness centrality in bias only"""

    CL_MODIFIED_TRANSITION = 'cl_modified_transition'
    """Closeness centrality in transition only"""

    # ========== 5. Betweenness Centrality-Enhanced (3 modes) ==========
    BW_MODIFIED = 'bw_modified'
    """Betweenness centrality in both bias and transition"""

    BW_MODIFIED_BIAS = 'bw_modified_bias'
    """Betweenness centrality in bias only"""

    BW_MODIFIED_TRANSITION = 'bw_modified_transition'
    """Betweenness centrality in transition only"""

    # ========== 6. Katz Centrality-Enhanced (3 modes) ==========
    KZ_MODIFIED = 'kz_modified'
    """Katz centrality in both bias and transition"""

    KZ_MODIFIED_BIAS = 'kz_modified_bias'
    """Katz centrality in bias only"""

    KZ_MODIFIED_TRANSITION = 'kz_modified_transition'
    """Katz centrality in transition only"""

    # ========== 7. Hybrid Centrality (1 mode) ==========
    HYBRID = 'hybrid'
    """Weighted combination of all centrality measures"""

    # ========== 8. Graph Neural Networks (10 modes) ==========
    GAT = 'gat'
    """Standard Graph Attention Network"""

    GAT_CENTRALITY = 'gat_centrality'
    """GAT with concatenated centrality features"""

    GAT_DEGREE = 'gat_degree'
    """GAT with degree centrality weighting"""

    GAT_CLOSENESS = 'gat_closeness'
    """GAT with closeness centrality weighting"""

    GAT_BETWEENNESS = 'gat_betweenness'
    """GAT with betweenness centrality weighting"""

    GAT_KATZ = 'gat_katz'
    """GAT with Katz centrality weighting"""

    GAT_TOPIC_AWARE = 'gat_topic_aware'
    """GAT with topic-specific attention heads (TA-GAT)"""

    GAT_MTPR = 'gat_mtpr'
    """GAT + Multi-Topic PageRank fusion"""

    GAT_MTPR_CENTRALITY = 'gat_mtpr_centrality'
    """GAT + MTPR + Centrality features"""

    GAT_MTPR_EMBEDDING = 'gat_mtpr_embedding'
    """GAT + MTPR with learned PageRank embeddings"""

    # ========== 9. Flow-Based (1 mode) ==========
    FLOW_MATCHING = 'flow_matching'
    """Flow Matching for citation path prediction"""

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"PRModes.{self.name}"

    @staticmethod
    def argparse(s):
        """Parse string to PRModes enum"""
        try:
            return PRModes[s.upper()]
        except KeyError:
            return s

    @classmethod
    def get_all_modes(cls):
        """Return all available modes"""
        return list(cls)

    @classmethod
    def get_basic_modes(cls):
        """Return basic PageRank modes"""
        return [cls.NORMAL, cls.PERSONALIZED, cls.THEMED]

    @classmethod
    def get_centrality_modes(cls):
        """Return all centrality-enhanced modes"""
        return [
            # Degree
            cls.D_MODIFIED, cls.D_MODIFIED_BIAS, cls.D_MODIFIED_TRANSITION,
            # Closeness
            cls.CL_MODIFIED, cls.CL_MODIFIED_BIAS, cls.CL_MODIFIED_TRANSITION,
            # Betweenness
            cls.BW_MODIFIED, cls.BW_MODIFIED_BIAS, cls.BW_MODIFIED_TRANSITION,
            # Katz
            cls.KZ_MODIFIED, cls.KZ_MODIFIED_BIAS, cls.KZ_MODIFIED_TRANSITION,
        ]

    @classmethod
    def get_gat_modes(cls):
        """Return all GAT-based modes"""
        return [
            cls.GAT, cls.GAT_CENTRALITY, cls.GAT_DEGREE, cls.GAT_CLOSENESS,
            cls.GAT_BETWEENNESS, cls.GAT_KATZ, cls.GAT_TOPIC_AWARE,
            cls.GAT_MTPR, cls.GAT_MTPR_CENTRALITY, cls.GAT_MTPR_EMBEDDING
        ]

    @classmethod
    def is_gat_mode(cls, mode):
        """Check if mode is GAT-based"""
        return mode in cls.get_gat_modes() or mode == cls.FLOW_MATCHING

    @classmethod
    def get_mtpr_modes(cls):
        """Return the 19 MTPR variants from the paper's Table 1 (variant summary)."""
        return (
            cls.get_basic_modes()
            + [cls.MODIFIED, cls.MODIFIED_BIAS, cls.MODIFIED_TRANSITION]
            + cls.get_centrality_modes()
            + [cls.HYBRID]
        )

    @classmethod
    def get_experimental_modes(cls):
        """Return modes outside the paper's scope (GAT + Flow Matching)."""
        return cls.get_gat_modes() + [cls.FLOW_MATCHING]

    @classmethod
    def get_evaluation_modes(cls):
        """Single source of truth for which modes evaluation scripts run.

        Controlled by INCLUDE_EXPERIMENTAL_MODES above: the 19 MTPR variants
        by default, plus GAT/Flow Matching when that flag is re-enabled.
        Evaluation scripts should read this instead of hardcoding their own
        mode list.
        """
        modes = cls.get_mtpr_modes()
        if INCLUDE_EXPERIMENTAL_MODES:
            modes = modes + cls.get_experimental_modes()
        return modes

    @classmethod
    def requires_centrality(cls, mode):
        """Check if mode requires centrality computation"""
        centrality_modes = cls.get_centrality_modes() + [cls.HYBRID]
        gat_centrality = [cls.GAT_CENTRALITY, cls.GAT_DEGREE, cls.GAT_CLOSENESS,
                          cls.GAT_BETWEENNESS, cls.GAT_KATZ, cls.GAT_MTPR_CENTRALITY]
        return mode in (centrality_modes + gat_centrality)

    @classmethod
    def requires_topic_model(cls, mode):
        """Check if mode requires topic modeling (LDA/BERTopic)"""
        non_topic_modes = [cls.NORMAL]
        return mode not in non_topic_modes
