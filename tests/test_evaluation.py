"""Unit tests for evaluation metrics calculation and collaborative filtering audit."""

import pytest
from ted_recommender.evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    average_precision_at_k,
    hit_rate_at_k,
    intra_list_diversity,
)
from ted_recommender.collaborative import assess_collaborative_filtering_feasibility
from ted_recommender.data import load_dataset


def test_ranking_metrics_exact_cases():
    """Verify precision, recall, ndcg, and MAP with known ground truth sets."""
    rec = [10, 20, 30, 40, 50]
    rel = {20, 40, 99}  # 2 hits out of 3 relevant

    assert precision_at_k(rec, rel, k=5) == 2 / 5
    assert recall_at_k(rec, rel, k=5) == 2 / 3
    assert hit_rate_at_k(rec, rel, k=5) == 1.0

    # Test top-1
    assert precision_at_k(rec, rel, k=1) == 0.0
    assert hit_rate_at_k(rec, rel, k=1) == 0.0

    # Test NDCG and AP
    ap = average_precision_at_k(rec, rel, k=5)
    assert ap > 0.0
    ndcg = ndcg_at_k(rec, rel, k=5)
    assert 0.0 < ndcg <= 1.0


def test_collaborative_filtering_feasibility():
    """Verify collaborative filtering audit reports honest UNSUPPORTED status."""
    df = load_dataset(stage="features")
    report = assess_collaborative_filtering_feasibility(df)

    assert report.has_user_ids is False
    assert report.has_individual_ratings is False
    assert report.verdict == "UNSUPPORTED_DATA_ABSENT"
    assert "user_id" in report.future_schema_definition


def test_annotation_template_structure():
    """Verify presence and validity of human relevance annotation template."""
    from pathlib import Path
    import pandas as pd
    from ted_recommender.config import PROJECT_ROOT

    template_path = (
        PROJECT_ROOT
        / "Module_4_Recommendation_Engine"
        / "outputs"
        / "relevance_annotation_template.csv"
    )
    assert template_path.exists()
    df_ann = pd.read_csv(template_path)
    required = [
        "query_or_seed_talk_id",
        "query_or_seed_title",
        "candidate_talk_id",
        "candidate_title",
        "relevance_grade",
    ]
    for col in required:
        assert col in df_ann.columns
    assert len(df_ann) > 0


