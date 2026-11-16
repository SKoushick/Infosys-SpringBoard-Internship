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
    RecommendationResult,
)
from evaluation.benchmark import UnifiedRecommenderBenchmark, ModelBenchmarkScorecard
from evaluation.evaluation_dataset import EvaluationDatasetBuilder


class FaultyRecommenderMock:
    """Mock recommender wrapping a real recommender but intentionally throwing on a specific query."""
    def __init__(self, real_rec, fail_query_id: int):
        self.real_rec = real_rec
        self.fail_query_id = fail_query_id

    def recommend_for_talk(self, talk_id: int, top_k: int = 10, apply_mmr: bool = False):
        if talk_id == self.fail_query_id:
            raise RuntimeError(f"Simulated intentional failure on query {talk_id}")
        return self.real_rec.recommend_for_talk(talk_id=talk_id, top_k=top_k, apply_mmr=apply_mmr)


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
        strict_mode=False,
    )
    return benchmark, tfidf_rec, semantic_rec, hybrid_rec, sample_queries


def test_unified_benchmark_execution(benchmark_fixture):
    benchmark, tfidf_rec, semantic_rec, hybrid_rec, query_ids = benchmark_fixture

    # Test TF-IDF
    sc_tfidf, df_tfidf = benchmark.evaluate_model(tfidf_rec, "tfidf", "TF-IDF", query_ids, apply_mmr=False)
    assert sc_tfidf.num_queries_evaluated == len(query_ids)
    assert sc_tfidf.requested_query_count == len(query_ids)
    assert sc_tfidf.failed_query_count == 0
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

    # Fair comparison verification across all 4 models
    fair_report = benchmark.verify_fair_comparison({
        "tfidf": sc_tfidf,
        "semantic": sc_sem,
        "hybrid": sc_hyb,
        "hybrid_mmr": sc_mmr,
    }, strict=True)
    assert fair_report["is_fair"] is True
    assert fair_report["common_query_count"] == len(query_ids)


def test_simulated_model_failure_strict_mode_raises(benchmark_fixture):
    """Verify that in strict mode, any failing query immediately raises RuntimeError."""
    benchmark, tfidf_rec, _, _, query_ids = benchmark_fixture
    failing_id = query_ids[0]
    faulty_model = FaultyRecommenderMock(real_rec=tfidf_rec, fail_query_id=failing_id)

    with pytest.raises(RuntimeError) as exc_info:
        benchmark.evaluate_model(
            model=faulty_model,
            model_id="faulty_model",
            display_name="Faulty Model",
            query_ids=query_ids,
            strict=True,
        )
    assert f"query {failing_id}" in str(exc_info.value)


def test_simulated_model_failure_detected_by_fair_verification(benchmark_fixture):
    """
    Verify that in non-strict mode, failing queries are recorded,
    and verify_fair_comparison detects unequal query sets and prevents false claims.
    """
    benchmark, tfidf_rec, _, _, query_ids = benchmark_fixture
    failing_id = query_ids[0]
    faulty_model = FaultyRecommenderMock(real_rec=tfidf_rec, fail_query_id=failing_id)

    # Succeeded model
    sc_good, _ = benchmark.evaluate_model(tfidf_rec, "good_model", "Good Model", query_ids, strict=False)
    assert sc_good.num_queries_evaluated == len(query_ids)
    assert sc_good.failed_query_count == 0

    # Faulty model skips 1 query
    sc_faulty, _ = benchmark.evaluate_model(faulty_model, "faulty_model", "Faulty Model", query_ids, strict=False)
    assert sc_faulty.num_queries_evaluated == len(query_ids) - 1
    assert sc_faulty.failed_query_count == 1
    assert sc_faulty.failed_query_ids == [failing_id]

    # Verify fair comparison flags mismatch
    fair_report = benchmark.verify_fair_comparison({
        "good": sc_good,
        "faulty": sc_faulty,
    }, strict=False)
    assert fair_report["is_fair"] is False
    assert "Unfair benchmark comparison detected" in fair_report["error"]

    # Verify fair comparison in strict mode raises ValueError
    with pytest.raises(ValueError) as exc_info:
        benchmark.verify_fair_comparison({
            "good": sc_good,
            "faulty": sc_faulty,
        }, strict=True)
    assert "Unfair benchmark" in str(exc_info.value)
