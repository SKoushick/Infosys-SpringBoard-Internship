"""Unit tests for recommendation algorithms (TF-IDF, Semantic, Hybrid, Personalisation, MMR, and Explanations)."""

import pytest
from ted_recommender.config import RecommenderConfig
from ted_recommender.data import load_dataset
from ted_recommender.features import FeaturePipeline
from ted_recommender.recommender import (
    ContentBasedRecommender,
    SemanticRecommender,
    HybridRecommender,
)


@pytest.fixture(scope="module")
def recommender_suite():
    df = load_dataset(stage="features")
    config = RecommenderConfig()
    pipeline = FeaturePipeline(tfidf_config=config.tfidf_config, field_weights=config.field_weights)
    pipeline.load_artifacts()

    tfidf_rec = ContentBasedRecommender(df, pipeline, config)
    semantic_rec = SemanticRecommender(df, pipeline, config)
    hybrid_rec = HybridRecommender(df, pipeline, config)
    return tfidf_rec, semantic_rec, hybrid_rec, df


def test_tfidf_recommend_for_talk(recommender_suite):
    """Test TF-IDF recommender produces top-k results excluding self."""
    tfidf_rec, _, _, df = recommender_suite
    results = tfidf_rec.recommend_for_talk(talk_id=1, top_k=5)

    assert len(results) == 5
    # Verify seed talk 1 is excluded
    assert all(r.talk_id != 1 for r in results)
    # Check score ordering (descending)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # Verify explanation present
    assert results[0].explanation.rationale != ""


def test_semantic_recommend_for_talk(recommender_suite):
    """Test Semantic recommender outputs valid results."""
    _, semantic_rec, _, _ = recommender_suite
    results = semantic_rec.recommend_for_talk(talk_id=1, top_k=5)

    assert len(results) == 5
    assert all(r.talk_id != 1 for r in results)
    assert results[0].explanation.semantic_score is not None


def test_hybrid_recommendation_and_personalisation(recommender_suite):
    """Test hybrid multi-liked talks, negative exclusions, and user interests."""
    _, _, hybrid_rec, _ = recommender_suite

    results = hybrid_rec.recommend(
        liked_talk_ids=[1, 2],
        disliked_talk_ids=[3, 4],
        user_interests=["technology", "design"],
        top_k=7,
        apply_mmr=True,
    )

    assert len(results) == 7
    # Neither liked nor disliked talks should appear
    forbidden = {1, 2, 3, 4}
    assert all(r.talk_id not in forbidden for r in results)


def test_raw_query_search(recommender_suite):
    """Test text query search with TF-IDF and Semantic models."""
    tfidf_rec, semantic_rec, _, _ = recommender_suite

    res_tfidf = tfidf_rec.recommend_for_query("neuroscience brain cognition", top_k=3)
    assert len(res_tfidf) == 3

    res_sem = semantic_rec.recommend_for_query("quantum computing physics", top_k=3)
    assert len(res_sem) == 3


def test_cold_start_fallback(recommender_suite):
    """Verify popular fallback when no user preferences are supplied."""
    _, _, hybrid_rec, _ = recommender_suite
    fallback_res = hybrid_rec.recommend(top_k=5)

    assert len(fallback_res) == 5
    # Popular fallback returns high view count talks
    assert all("Cold-start" in r.explanation.rationale for r in fallback_res)


def test_edge_cases(recommender_suite):
    """Test invalid talk ID, empty query, and non-existent tag filters."""
    tfidf_rec, _, hybrid_rec, _ = recommender_suite

    # Non-existent talk_id
    with pytest.raises(KeyError):
        tfidf_rec.recommend_for_talk(talk_id=999999)

    # Empty search query
    assert tfidf_rec.recommend_for_query("   ") == []

    # Non-existent filter tag returns empty list safely
    filtered = tfidf_rec.recommend_for_talk(
        talk_id=1, filter_tags=["non_existent_impossible_tag_12345"]
    )
    assert filtered == []

