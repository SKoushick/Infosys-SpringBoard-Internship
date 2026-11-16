"""
Ranking and Relevance Evaluation Metrics.
Module 5: Evaluation Metrics | Infosys Springboard 7.0

Implements mathematically rigorous, edge-case resilient ranking metrics:
- Precision@K
- Recall@K
- F1@K
- Discounted Cumulative Gain (DCG@K)
- Normalized Discounted Cumulative Gain (NDCG@K) with binary or graded gains
- Average Precision (AP@K) and Mean Average Precision (MAP@K)
- HitRate@K
- Mean Reciprocal Rank (MRR@K)

Guarantees:
- Safe division (returns 0.0 when denominators are zero).
- Exact handling of empty recommendations or empty relevant sets.
- Bounds verification (0.0 <= metric <= 1.0 for normalized metrics).
"""

from typing import Dict, List, Optional, Sequence, Set, Union
import numpy as np


def precision_at_k(
    recommended_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int,
) -> float:
    """
    Precision@K: Proportion of top-K recommendations that are relevant.
    Formula: |Recommended[:k] ∩ Relevant| / k
    """
    if k <= 0 or not recommended_ids or not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return float(hits / k)


def recall_at_k(
    recommended_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int,
) -> float:
    """
    Recall@K: Proportion of known relevant items retrieved in top-K.
    Formula: |Recommended[:k] ∩ Relevant| / |Relevant|
    """
    if k <= 0 or not recommended_ids or not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return float(hits / len(relevant_ids))


def f1_at_k(
    recommended_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int,
) -> float:
    """
    F1@K: Harmonic mean of Precision@K and Recall@K.
    Formula: 2 * (Precision * Recall) / (Precision + Recall)
    """
    prec = precision_at_k(recommended_ids, relevant_ids, k)
    rec = recall_at_k(recommended_ids, relevant_ids, k)
    if (prec + rec) == 0.0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def dcg_at_k(
    recommended_ids: Sequence[int],
    relevance: Union[Set[int], Dict[int, float]],
    k: int,
    use_exponential_gain: bool = True,
) -> float:
    """
    Discounted Cumulative Gain at rank K.
    Formula: sum_{i=1}^k [ (2^{rel_i} - 1) / log2(i + 1) ]
    """
    if k <= 0 or not recommended_ids or not relevance:
        return 0.0

    top_k = recommended_ids[:k]
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if isinstance(relevance, dict):
            rel_score = float(relevance.get(item, 0.0))
        else:
            rel_score = 1.0 if item in relevance else 0.0

        if rel_score > 0.0:
            if use_exponential_gain:
                gain = (2.0 ** rel_score) - 1.0
            else:
                gain = rel_score
            discount = np.log2(rank + 1.0)
            dcg += gain / discount

    return float(dcg)


def ndcg_at_k(
    recommended_ids: Sequence[int],
    relevance: Union[Set[int], Dict[int, float]],
    k: int,
    use_exponential_gain: bool = True,
) -> float:
    """
    Normalized Discounted Cumulative Gain at rank K.
    Formula: DCG@K / IDCG@K
    Where IDCG@K is the maximum possible DCG achievable with the ground-truth set.
    """
    if k <= 0 or not recommended_ids or not relevance:
        return 0.0

    actual_dcg = dcg_at_k(recommended_ids, relevance, k, use_exponential_gain=use_exponential_gain)
    if actual_dcg == 0.0:
        return 0.0

    # Compute Ideal DCG (IDCG)
    if isinstance(relevance, dict):
        ideal_scores = sorted([float(s) for s in relevance.values() if s > 0], reverse=True)[:k]
    else:
        n_hits = min(len(relevance), k)
        ideal_scores = [1.0] * n_hits

    if not ideal_scores:
        return 0.0

    idcg = 0.0
    for rank, score in enumerate(ideal_scores, start=1):
        gain = ((2.0 ** score) - 1.0) if use_exponential_gain else score
        discount = np.log2(rank + 1.0)
        idcg += gain / discount

    if idcg <= 0.0:
        return 0.0

    return float(np.clip(actual_dcg / idcg, 0.0, 1.0))


def average_precision_at_k(
    recommended_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int,
) -> float:
    """
    Average Precision at rank K.
    Formula: sum_{i=1}^k [ P(i) * rel(i) ] / min(|Relevant|, k)
    """
    if k <= 0 or not recommended_ids or not relevant_ids:
        return 0.0

    top_k = recommended_ids[:k]
    hits = 0
    sum_precisions = 0.0

    for rank, item in enumerate(top_k, start=1):
        if item in relevant_ids:
            hits += 1
            sum_precisions += hits / rank

    if hits == 0:
        return 0.0

    denominator = min(len(relevant_ids), k)
    return float(sum_precisions / denominator)


def hit_rate_at_k(
    recommended_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int,
) -> float:
    """
    HitRate@K: Binary indicator (1.0 if at least one relevant item appears in top-K, 0.0 otherwise).
    """
    if k <= 0 or not recommended_ids or not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    return 1.0 if any(item in relevant_ids for item in top_k) else 0.0


def reciprocal_rank_at_k(
    recommended_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int,
) -> float:
    """
    Reciprocal Rank at K: 1 / rank of first relevant item in top-K, 0.0 if no hits.
    """
    if k <= 0 or not recommended_ids or not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    for rank, item in enumerate(top_k, start=1):
        if item in relevant_ids:
            return float(1.0 / rank)
    return 0.0

