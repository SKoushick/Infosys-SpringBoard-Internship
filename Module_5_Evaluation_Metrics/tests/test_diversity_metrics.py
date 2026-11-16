"""
Unit tests for Diversity, Topic Entropy, and Recommendation Integrity.
Module 5: Evaluation Metrics | Infosys Springboard 7.0
"""

import pytest
import numpy as np
from scipy import sparse

from evaluation.diversity_metrics import (
    intra_list_diversity,
    topic_shannon_entropy,
    topic_gini_simpson_index,
    duplicate_and_exclusion_integrity,
)


def test_intra_list_diversity_synthetic():
    # 3 completely orthogonal unit vectors -> pairwise cosine sim = 0 -> distance = 1.0
    orthogonal_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    ild_dense = intra_list_diversity([0, 1, 2], orthogonal_matrix)
    assert pytest.approx(ild_dense, abs=1e-4) == 1.0

    # Sparse representation
    sparse_matrix = sparse.csr_matrix(orthogonal_matrix)
    ild_sparse = intra_list_diversity([0, 1, 2], sparse_matrix)
    assert pytest.approx(ild_sparse, abs=1e-4) == 1.0

    # 3 identical vectors -> distance = 0.0
    identical_matrix = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=np.float32)
    assert pytest.approx(intra_list_diversity([0, 1, 2], identical_matrix), abs=1e-4) == 0.0

    # Edge cases
    assert intra_list_diversity([], orthogonal_matrix) == 0.0
    assert intra_list_diversity([0], orthogonal_matrix) == 0.0


def test_intra_list_diversity_bounds_and_nan_handling():
    """Verify that out-of-bounds indices and non-finite values are handled safely."""
    mat = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    # Index 99 is out of bounds (matrix has only 2 rows). Valid indices: [0, 1].
    ild = intra_list_diversity([0, 1, 99], mat)
    assert pytest.approx(ild, abs=1e-4) == 1.0

    # When all indices are out of bounds
    assert intra_list_diversity([99, 100], mat) == 0.0

    # Matrix with a zero-vector row (could cause NaN cosine sim)
    zero_mat = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
    ], dtype=np.float32)
    ild_zero = intra_list_diversity([0, 1], zero_mat)
    assert not np.isnan(ild_zero)


def test_topic_shannon_entropy():
    # Monolithic topics -> 0 entropy
    monolithic = [["tech", "tech"], ["tech"]]
    assert topic_shannon_entropy(monolithic) == 0.0

    # Uniformly distributed distinct topics -> maximum normalized entropy = 1.0
    diverse = [["tech"], ["science"], ["art"], ["business"]]
    assert pytest.approx(topic_shannon_entropy(diverse, normalize=True), abs=1e-4) == 1.0

    # Empty
    assert topic_shannon_entropy([]) == 0.0


def test_topic_gini_simpson():
    diverse = [["tech"], ["science"], ["art"], ["business"]]
    simpson = topic_gini_simpson_index(diverse)
    assert simpson > 0.5  # High diversity

    monolithic = [["tech"], ["tech"]]
    assert topic_gini_simpson_index(monolithic) == 0.0


def test_duplicate_and_exclusion_integrity():
    # Clean list
    res_clean = duplicate_and_exclusion_integrity(
        recommended_ids=[10, 20, 30],
        seed_talk_id=1,
        excluded_ids={99, 100},
    )
    assert res_clean["is_clean"] is True
    assert res_clean["duplicate_count"] == 0
    assert res_clean["has_self_recommendation"] is False
    assert res_clean["exclusion_violations_count"] == 0

    # List with duplicate
    res_dup = duplicate_and_exclusion_integrity([10, 20, 10], seed_talk_id=1)
    assert res_dup["is_clean"] is False
    assert res_dup["duplicate_count"] == 1

    # List with self-recommendation
    res_self = duplicate_and_exclusion_integrity([1, 20, 30], seed_talk_id=1)
    assert res_self["is_clean"] is False
    assert res_self["has_self_recommendation"] is True

    # List with exclusion violation
    res_excl = duplicate_and_exclusion_integrity([10, 20, 99], seed_talk_id=1, excluded_ids={99})
    assert res_excl["is_clean"] is False
    assert res_excl["exclusion_violations_count"] == 1
    assert 99 in res_excl["violation_talk_ids"]
