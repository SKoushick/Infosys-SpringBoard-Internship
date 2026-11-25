"""
TED Talks Recommendation Engine
================================
A modular, content-based recommendation system for TED Talks.

Modules:
    config      - Centralised configuration management
    data        - Data loading, validation, and preprocessing
    features    - Text feature engineering and vectorisation
    recommender - Recommendation algorithms (TF-IDF, semantic, hybrid)
    evaluation  - Evaluation metrics and readiness utilities
    learning    - Pedagogical complexity and learning progression
    utils       - Shared utility functions
"""

from .learning import (
    ComplexityEstimatorConfig,
    ComplexityProfile,
    PedagogicalComplexityEstimator,
)

__version__ = "1.1.0"
__author__ = "SKoushick"

__all__ = [
    "ComplexityEstimatorConfig",
    "ComplexityProfile",
    "PedagogicalComplexityEstimator",
]

