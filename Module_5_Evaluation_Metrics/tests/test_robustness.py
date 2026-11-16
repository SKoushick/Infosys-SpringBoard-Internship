"""
Integration tests for Recommender Robustness and Stress Suite.
Module 5: Evaluation Metrics | Infosys Springboard 7.0
"""

import pytest
from ted_recommender.config import RecommenderConfig
from ted_recommender.data import load_dataset
from ted_recommender.features import FeaturePipeline
from ted_recommender.recommender import (
    ContentBasedRecommender,
    SemanticRecommender,
    HybridRecommender,
)
from evaluation.robustness import RecommenderRobustnessHarness


def test_robustness_suite_passes_all_checks():
    df = load_dataset(stage="features")
    config = RecommenderConfig.load()
    pipeline = FeaturePipeline(tfidf_config=config.tfidf_config, field_weights=config.field_weights)
    pipeline.load_artifacts()

    tfidf_rec = ContentBasedRecommender(df, pipeline, config)
    semantic_rec = SemanticRecommender(df, pipeline, config)
    hybrid_rec = HybridRecommender(df, pipeline, config)

    harness = RecommenderRobustnessHarness((tfidf_rec, semantic_rec, hybrid_rec), df)
    report = harness.run_all_checks()

    assert report["all_passed"] is True
    assert report["passed_checks"] == 12
    assert report["failed_checks"] == 0

