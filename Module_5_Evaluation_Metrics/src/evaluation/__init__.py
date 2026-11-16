"""
Module 5: Evaluation Metrics Package.
Personalized TED Talks Recommendation System.
"""

from .ranking_metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    dcg_at_k,
    ndcg_at_k,
    average_precision_at_k,
    hit_rate_at_k,
    reciprocal_rank_at_k,
)
from .diversity_metrics import (
    intra_list_diversity,
    topic_shannon_entropy,
    topic_gini_simpson_index,
    duplicate_and_exclusion_integrity,
)
from .coverage_metrics import (
    catalogue_coverage,
    recommendation_novelty,
    gini_coefficient_recommendations,
)
from .evaluation_dataset import (
    EvaluationDatasetBuilder,
    EvaluationQueryRecord,
    EvaluationDatasetReport,
    parse_raw_related_talks,
)
from .benchmark import (
    UnifiedRecommenderBenchmark,
    ModelBenchmarkScorecard,
)
from .performance import (
    RecommenderPerformanceProfiler,
    get_process_memory_mb,
)
from .robustness import (
    RecommenderRobustnessHarness,
)
from .human_evaluation import (
    HumanRelevanceHarness,
    RELEVANCE_SCALE_DESCRIPTION,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "average_precision_at_k",
    "hit_rate_at_k",
    "reciprocal_rank_at_k",
    "intra_list_diversity",
    "topic_shannon_entropy",
    "topic_gini_simpson_index",
    "duplicate_and_exclusion_integrity",
    "catalogue_coverage",
    "recommendation_novelty",
    "gini_coefficient_recommendations",
    "EvaluationDatasetBuilder",
    "EvaluationQueryRecord",
    "EvaluationDatasetReport",
    "parse_raw_related_talks",
    "UnifiedRecommenderBenchmark",
    "ModelBenchmarkScorecard",
    "RecommenderPerformanceProfiler",
    "get_process_memory_mb",
    "RecommenderRobustnessHarness",
    "HumanRelevanceHarness",
    "RELEVANCE_SCALE_DESCRIPTION",
]

