"""
Topic-Conditioned Fusion Model for TCH-PR (Topic-Conditioned Hybrid PageRank)

This module implements a lightweight learnable fusion model that combines
4 different PageRank signals based on the query's topic distribution.

Signals:
1. Topic-Sensitive PageRank (modified_transition)
2. Betweenness Centrality-enhanced PageRank (bw_modified)
3. GAT-based embedding scoring
4. Topic-aware Transition scoring

The fusion model takes a 50-dimensional topic distribution vector and
outputs 4 weights that sum to 1.0.
"""

import torch
import torch.nn as nn
from typing import Optional


class TopicConditionedFusion(nn.Module):
    """
    Lightweight Fusion Model for TCH-PR.
    
    Takes query topic distribution (50-dim) and outputs weights for 4 signals.
    Designed to be fast and GPU-friendly.
    """
    
    def __init__(self, num_topics: int = 50, num_signals: int = 4, 
                 hidden_dim: int = 64, dropout: float = 0.2):
        """
        Args:
            num_topics: Number of LDA topics (default: 50)
            num_signals: Number of PageRank signals to fuse (default: 4)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.num_topics = num_topics
        self.num_signals = num_signals
        
        self.mlp = nn.Sequential(
            nn.Linear(num_topics, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_signals),
            nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )
        
        # Initialize weights for better training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, query_topic_dist: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query_topic_dist: Topic distribution of the query [batch_size, num_topics]
                             or [num_topics] for single query
        
        Returns:
            weights: Fusion weights [batch_size, num_signals] or [num_signals]
        """
        # Handle single query case
        if query_topic_dist.dim() == 1:
            query_topic_dist = query_topic_dist.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        weights = self.mlp(query_topic_dist)
        
        if squeeze_output:
            weights = weights.squeeze(0)
        
        return weights
    
    def get_signal_names(self) -> list:
        """Return names of the 4 signals."""
        return [
            "topic_sensitive_pr",      # S1: modified_transition
            "betweenness_enhanced_pr", # S2: bw_modified  
            "gat_embedding_score",     # S3: GAT-based
            "topic_aware_transition"   # S4: transition_star
        ]


class SimpleTopicFusion(nn.Module):
    """
    Even simpler version - just a linear layer + softmax.
    Useful for faster training and inference.
    """
    
    def __init__(self, num_topics: int = 50, num_signals: int = 4):
        super().__init__()
        self.linear = nn.Linear(num_topics, num_signals)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query_topic_dist: torch.Tensor) -> torch.Tensor:
        logits = self.linear(query_topic_dist)
        return self.softmax(logits)


def create_fusion_model(num_topics: int = 50, model_type: str = "mlp") -> nn.Module:
    """
    Factory function to create fusion model.
    
    Args:
        num_topics: Number of LDA topics
        model_type: "mlp" or "simple"
    
    Returns:
        Fusion model instance
    """
    if model_type == "mlp":
        return TopicConditionedFusion(num_topics=num_topics)
    elif model_type == "simple":
        return SimpleTopicFusion(num_topics=num_topics)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")