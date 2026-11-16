"""
Unit tests for Ranking Metrics (Precision@K, Recall@K, F1@K, NDCG@K, MAP@K, HitRate@K, MRR@K).
Module 5: Evaluation Metrics | Infosys Springboard 7.0
"""

import pytest
import numpy as np

from evaluation.ranking_metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    dcg_at_k,
    ndcg_at_k,
    average_precision_at_k,
    hit_rate_at_k,
    reciprocal_rank_at_k,
)


def test_precision_at_k_exact():
    # Recommended: [10, 20, 30, 40, 50], Relevant: {20, 40, 99}
    rec = [10, 20, 30, 40, 50]
    rel = {20, 40, 99}

    assert precision_at_k(rec, rel, k=1) == 0.0  # 10 not in rel
    assert precision_at_k(rec, rel, k=2) == 1 / 2  # 20 is hit
    assert precision_at_k(rec, rel, k=4) == 2 / 4  # 20, 40 are hits
    assert precision_at_k(rec, rel, k=5) == 2 / 5


def test_recall_at_k_exact():
    rec = [10, 20, 30, 40, 50]
    rel = {20, 40, 99}  # total 3 relevant

    assert recall_at_k(rec, rel, k=1) == 0.0
    assert recall_at_k(rec, rel, k=2) == 1 / 3
    assert recall_at_k(rec, rel, k=4) == 2 / 3
    assert recall_at_k(rec, rel, k=5) == 2 / 3


def test_f1_at_k_exact():
    rec = [10, 20, 30, 40]
    rel = {20, 40}  # Precision@4 = 2/4 = 0.5, Recall@4 = 2/2 = 1.0
    # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 2/3
    assert pytest.approx(f1_at_k(rec, rel, k=4), abs=1e-4) == 2 / 3


def test_dcg_and_ndcg_binary():
    rec = [10, 20, 30, 40]
    rel = {10, 30}  # hits at rank 1 and rank 3

    # Hand calculation:
    # rank 1: gain = (2^1 - 1) = 1, discount = log2(2) = 1.0 -> 1.0
    # rank 2: 0
    # rank 3: gain = 1, discount = log2(4) = 2.0 -> 0.5
    # actual DCG@3 = 1.5
    actual_dcg = dcg_at_k(rec, rel, k=3)
    assert pytest.approx(actual_dcg, abs=1e-4) == 1.5

    # Ideal DCG@3: hits at rank 1 and rank 2
    # rank 1: 1.0 / log2(2) = 1.0
    # rank 2: 1.0 / log2(3) = 1.0 / 1.5849625 = 0.63092975
    # IDCG@3 = 1.63092975
    # NDCG@3 = 1.5 / 1.63092975 = 0.91972
    expected_ndcg = 1.5 / (1.0 + 1.0 / np.log2(3))
    assert pytest.approx(ndcg_at_k(rec, rel, k=3), abs=1e-4) == expected_ndcg


def test_ndcg_graded_relevance():
    rec = [10, 20, 30]
    graded_rel = {10: 3.0, 20: 1.0, 30: 0.0}

    # Rank 1: item 10, rel=3 -> gain = 2^3 - 1 = 7.0 / log2(2) = 7.0
    # Rank 2: item 20, rel=1 -> gain = 2^1 - 1 = 1.0 / log2(3) = 0.6309
    # Rank 3: item 30, rel=0 -> 0
    actual_dcg = dcg_at_k(rec, graded_rel, k=3)
    assert actual_dcg > 7.0

    # Ideal order is [10, 20] which matches actual order!
    # Therefore NDCG must equal 1.0!
    assert pytest.approx(ndcg_at_k(rec, graded_rel, k=3), abs=1e-4) == 1.0


def test_average_precision_and_hit_rate():
    rec = [10, 20, 30, 40]
    rel = {20, 40}

    # Rank 1: item 10 -> miss
    # Rank 2: item 20 -> hit, precision = 1/2
    # Rank 3: item 30 -> miss
    # Rank 4: item 40 -> hit, precision = 2/4 = 1/2
    # sum_prec = 0.5 + 0.5 = 1.0
    # min(|rel|, k) = min(2, 4) = 2
    # AP@4 = 1.0 / 2 = 0.5
    assert pytest.approx(average_precision_at_k(rec, rel, k=4), abs=1e-4) == 0.5

    assert hit_rate_at_k(rec, rel, k=1) == 0.0
    assert hit_rate_at_k(rec, rel, k=2) == 1.0
    assert reciprocal_rank_at_k(rec, rel, k=4) == 0.5


def test_edge_cases_and_zero_handling():
    # Empty inputs
    assert precision_at_k([], {1, 2}, k=5) == 0.0
    assert recall_at_k([1, 2], set(), k=5) == 0.0
    assert f1_at_k([], set(), k=5) == 0.0
    assert dcg_at_k([], {1}, k=5) == 0.0
    assert ndcg_at_k([], {1}, k=5) == 0.0
    assert average_precision_at_k([1], set(), k=5) == 0.0
    assert hit_rate_at_k([], {1}, k=5) == 0.0
    assert reciprocal_rank_at_k([1], set(), k=5) == 0.0

    # Negative or zero K
    assert precision_at_k([1, 2], {1}, k=0) == 0.0
    assert recall_at_k([1, 2], {1}, k=-1) == 0.0
    assert ndcg_at_k([1, 2], {1}, k=0) == 0.0

