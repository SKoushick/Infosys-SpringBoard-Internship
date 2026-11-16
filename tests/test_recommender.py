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


def test_mmr_properties_and_memory_efficiency():
    """Verify MMR ranking logic, diversity tradeoff, and sparse memory efficiency."""
    import numpy as np
    from scipy import sparse
    from ted_recommender.recommender import maximal_marginal_relevance

    # Synthetic candidate setup:
    # 5 candidates with 4 features each
    # Item 0: [1, 0, 0, 0] (relevance 0.95)
    # Item 1: [0.99, 0.01, 0, 0] (near duplicate of Item 0, relevance 0.90)
    # Item 2: [0, 1, 0, 0] (completely orthogonal/diverse, relevance 0.85)
    # Item 3: [0, 0, 1, 0] (orthogonal, relevance 0.80)
    # Item 4: [0, 0, 0, 1] (orthogonal, relevance 0.75)
    dense_feats = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.99, 0.01, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    # Normalize rows
    norms = np.linalg.norm(dense_feats, axis=1, keepdims=True)
    dense_feats = dense_feats / norms

    sparse_feats = sparse.csr_matrix(dense_feats)
    cand_indices = np.array([0, 1, 2, 3, 4])
    relevance_scores = np.array([0.95, 0.90, 0.85, 0.80, 0.75], dtype=np.float32)

    # 1. Pure Relevance (lambda = 1.0): Should select [0, 1, 2] in exact descending order of relevance
    res_dense_pure_rel = maximal_marginal_relevance(
        query_vector=dense_feats[0],
        candidate_indices=cand_indices,
        candidate_features=dense_feats,
        relevance_scores=relevance_scores,
        top_k=3,
        mmr_lambda=1.0,
    )
    assert res_dense_pure_rel == [0, 1, 2], f"Expected [0, 1, 2], got {res_dense_pure_rel}"

    # 2. Balanced Diversity (lambda = 0.5):
    # After selecting 0, candidate 1 (near-duplicate) should be penalized for high similarity to 0.
    # Candidate 2 (orthogonal) should be preferred over candidate 1!
    res_dense_div = maximal_marginal_relevance(
        query_vector=dense_feats[0],
        candidate_indices=cand_indices,
        candidate_features=dense_feats,
        relevance_scores=relevance_scores,
        top_k=3,
        mmr_lambda=0.5,
    )
    assert res_dense_div[0] == 0
    assert res_dense_div[1] == 2, f"Candidate 2 should be chosen over duplicate candidate 1, got {res_dense_div}"

    # 3. Sparse CSR Execution: Exact same behavior without densification
    res_sparse_div = maximal_marginal_relevance(
        query_vector=sparse_feats.getrow(0),
        candidate_indices=cand_indices,
        candidate_features=sparse_feats,
        relevance_scores=relevance_scores,
        top_k=3,
        mmr_lambda=0.5,
    )
    assert res_sparse_div == res_dense_div, "Sparse MMR results must match dense MMR results identically"

    # 4. Edge cases
    assert maximal_marginal_relevance(dense_feats[0], cand_indices, dense_feats, relevance_scores, top_k=0) == []
    # If candidates <= top_k, returns sorted by relevance
    few_cands = np.array([3, 1])
    res_few = maximal_marginal_relevance(dense_feats[0], few_cands, dense_feats, relevance_scores[few_cands], top_k=5)
    assert res_few == [1, 3]


def test_hybrid_recommend_for_talk_polymorphism(recommender_suite):
    """Verify polymorphic recommend_for_talk method on HybridRecommender."""
    tfidf_rec, semantic_rec, hybrid_rec, _ = recommender_suite
    results = hybrid_rec.recommend_for_talk(talk_id=1, top_k=5, apply_mmr=True)

    assert len(results) == 5
    assert all(r.talk_id != 1 for r in results)
    assert results[0].score >= results[-1].score or any(r.explanation.semantic_score is not None for r in results)
    assert results[0].explanation.combined_score is not None


