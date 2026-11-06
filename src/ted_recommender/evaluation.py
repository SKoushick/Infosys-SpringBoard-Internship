"""
Evaluation Metrics & Benchmark Framework for Recommendation Systems.
Evaluation readiness for Module 4 and future Module 5 requirements.

Implements:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Mean Average Precision (MAP@K)
- Hit Rate@K
- Intra-List Diversity (ILD)
- Catalog Coverage
- Benchmark suite using TED Editorial Related Talks as proxy relevance labels
"""

import ast
import json
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def precision_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Precision@K: Fraction of top-K recommendations that are relevant."""
    if k <= 0 or not recommended_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return float(hits / k)


def recall_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Recall@K: Fraction of relevant items retrieved in top-K."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return float(hits / len(relevant_ids))


def dcg_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Discounted Cumulative Gain at K with binary relevance."""
    top_k = recommended_ids[:k]
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        rel = 1 if item in relevant_ids else 0
        dcg += (2**rel - 1) / np.log2(rank + 1)
    return float(dcg)


def ndcg_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if not relevant_ids or k <= 0:
        return 0.0
    actual_dcg = dcg_at_k(recommended_ids, relevant_ids, k)
    # Ideal DCG: all top spots filled by relevant items up to min(len(relevant_ids), k)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum((2**1 - 1) / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return float(actual_dcg / idcg)


def average_precision_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Average Precision at K."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = 0
    sum_precisions = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant_ids:
            hits += 1
            sum_precisions += hits / rank
    if not hits:
        return 0.0
    return float(sum_precisions / min(len(relevant_ids), k))


def hit_rate_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Hit Rate@K: 1.0 if at least one relevant item is in top-K, 0.0 otherwise."""
    top_k = recommended_ids[:k]
    return 1.0 if any(item in relevant_ids for item in top_k) else 0.0


def intra_list_diversity(
    recommended_indices: List[int],
    feature_matrix: np.ndarray,
) -> float:
    """
    Intra-List Diversity (ILD): Average pairwise cosine distance between recommended items.
    Values closer to 1.0 indicate higher diversity.
    """
    if len(recommended_indices) < 2:
        return 0.0

    sub = feature_matrix[recommended_indices]
    # Unit vectors dot product = cosine similarity
    sim_mat = np.dot(sub, sub.T)
    # Distance = 1 - similarity
    dist_mat = 1.0 - sim_mat

    # Average of off-diagonal elements
    n = len(recommended_indices)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_distances = dist_mat[triu_indices]
    return float(np.mean(pairwise_distances))


def parse_ground_truth_related_talks(df: pd.DataFrame) -> Dict[int, Set[int]]:
    """
    Extracts editorial related talks from the 'related_talks' column in the dataset.
    Used as an objective editorial benchmark / proxy relevance ground-truth.
    """
    ground_truth: Dict[int, Set[int]] = {}
    if "related_talks" not in df.columns:
        return ground_truth

    for _, row in df.iterrows():
        t_id = int(row["talk_id"])
        val = row["related_talks"]
        if pd.isna(val):
            continue

        parsed = None
        if isinstance(val, list):
            parsed = val
        elif isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
            except Exception:
                pass

        if parsed and isinstance(parsed, list):
            ids = set()
            for item in parsed:
                if isinstance(item, dict) and "id" in item:
                    try:
                        ids.add(int(item["id"]))
                    except ValueError:
                        pass
            if ids:
                ground_truth[t_id] = ids

    return ground_truth


class RecommendationEvaluator:
    """
    Evaluates recommender models against ground truth relevance sets across a sample of talks.
    """

    def __init__(self, ground_truth: Dict[int, Set[int]]):
        self.ground_truth = ground_truth

    def evaluate_recommender(
        self,
        recommender,
        test_talk_ids: List[int],
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate a recommender model across a set of test talk IDs.
        Returns average Precision@K, Recall@K, NDCG@K, MAP@K, and HitRate@K.
        """
        precisions = []
        recalls = []
        ndcgs = []
        maps = []
        hit_rates = []

        for talk_id in test_talk_ids:
            if talk_id not in self.ground_truth:
                continue
            relevant = self.ground_truth[talk_id]
            if not relevant:
                continue

            try:
                # Get recommendations
                results = recommender.recommend_for_talk(talk_id=talk_id, top_k=k)
                rec_ids = [r.talk_id for r in results]

                precisions.append(precision_at_k(rec_ids, relevant, k))
                recalls.append(recall_at_k(rec_ids, relevant, k))
                ndcgs.append(ndcg_at_k(rec_ids, relevant, k))
                maps.append(average_precision_at_k(rec_ids, relevant, k))
                hit_rates.append(hit_rate_at_k(rec_ids, relevant, k))
            except Exception as e:
                logger.warning(f"Failed evaluation for talk_id {talk_id}: {e}")
                continue

        if not precisions:
            return {
                f"Precision@{k}": 0.0,
                f"Recall@{k}": 0.0,
                f"NDCG@{k}": 0.0,
                f"MAP@{k}": 0.0,
                f"HitRate@{k}": 0.0,
                "n_evaluated_talks": 0,
            }

        return {
            f"Precision@{k}": round(float(np.mean(precisions)), 4),
            f"Recall@{k}": round(float(np.mean(recalls)), 4),
            f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
            f"MAP@{k}": round(float(np.mean(maps)), 4),
            f"HitRate@{k}": round(float(np.mean(hit_rates)), 4),
            "n_evaluated_talks": len(precisions),
        }

