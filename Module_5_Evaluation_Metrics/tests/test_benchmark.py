"""
Integration tests for Unified Comparative Recommender Benchmark.
Module 5: Evaluation Metrics | Infosys Springboard 7.0
"""

import pytest
import numpy as np
import pandas as pd

from ted_recommender.config import RecommenderConfig
from ted_recommender.data import load_dataset
from ted_recommender.features import FeaturePipeline
from ted_recommender.recommender import (
    ContentBasedRecommender,
    SemanticRecommender,
    HybridRecommender,
)
from evaluation.benchmark import UnifiedRecommenderBenchmark
from evaluation.evaluation_dataset import EvaluationDatasetBuilder


@pytest.fixture(scope="module")
def benchmark_fixture():
    df = load_dataset(stage="features")
    config = RecommenderConfig.load()
    pipeline = FeaturePipeline(tfidf_config=config.tfidf_config, field_weights=config.field_weights)
    pipeline.load_artifacts()

    tfidf_rec = ContentBasedRecommender(df, pipeline, config)
    semantic_rec = SemanticRecommender(df, pipeline, config)
    hybrid_rec = HybridRecommender(df, pipeline, config)

    builder = EvaluationDatasetBuilder(df)
    builder.build()
    sample_queries = builder.sample_queries(sample_size=10, random_seed=42)
    ground_truth = {qid: set(builder.records[qid].relevant_talk_ids) for qid in sample_queries}

    benchmark = UnifiedRecommenderBenchmark(
        df=df,
        ground_truth=ground_truth,
        feature_matrix=pipeline.embedding_matrix,
        id_to_idx=tfidf_rec.id_to_idx,
        idx_to_id=tfidf_rec.idx_to_id,
    )
    return benchmark, tfidf_rec, semantic_rec, hybrid_rec, sample_queries


def test_unified_benchmark_execution(benchmark_fixture):
    benchmark, tfidf_rec, semantic_rec, hybrid_rec, query_ids = benchmark_fixture

    # Test TF-IDF
    sc_tfidf, df_tfidf = benchmark.evaluate_model(tfidf_rec, "tfidf", "TF-IDF", query_ids, apply_mmr=False)
    assert sc_tfidf.num_queries_evaluated == len(query_ids)
    assert sc_tfidf.catalogue_coverage_ratio > 0.0

    # Test Semantic
    sc_sem, df_sem = benchmark.evaluate_model(semantic_rec, "semantic", "Semantic", query_ids, apply_mmr=False)
    assert sc_sem.num_queries_evaluated == len(query_ids)

    # Test Hybrid without MMR
    sc_hyb, df_hyb = benchmark.evaluate_model(hybrid_rec, "hybrid", "Hybrid", query_ids, apply_mmr=False)
    assert sc_hyb.num_queries_evaluated == len(query_ids)

    # Test Hybrid with MMR
    sc_mmr, df_mmr = benchmark.evaluate_model(hybrid_rec, "hybrid_mmr", "Hybrid + MMR", query_ids, apply_mmr=True)
    assert sc_mmr.num_queries_evaluated == len(query_ids)

    # Verify MMR increases diversity (ILD)
    assert sc_mmr.intra_list_diversity_at_10 >= sc_hyb.intra_list_diversity_at_10

    # Integrity clean rate should be 1.0 (no duplicates, no self-recommendations)
    assert sc_tfidf.integrity_clean_rate == 1.0
    assert sc_sem.integrity_clean_rate == 1.0
    assert sc_hyb.integrity_clean_rate == 1.0
    assert sc_mmr.integrity_clean_rate == 1.0

