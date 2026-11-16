"""
Diversity and List Integrity Metrics.
Module 5: Evaluation Metrics | Infosys Springboard 7.0

Implements:
- Intra-List Diversity (ILD): Pairwise cosine distance across recommended items.
- Topic Shannon Entropy: Information-theoretic spread across thematic tags.
- Gini-Simpson Topic Diversity Index: Probability of inter-topic variety.
- Recommendation List Integrity Verification: Validates absence of duplicates,
  self-recommendations, and forbidden exclusions.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np
from scipy import sparse


def intra_list_diversity(
    recommended_indices: Sequence[int],
    feature_matrix: Union[np.ndarray, sparse.spmatrix],
) -> float:
    """
    Intra-List Diversity (ILD): Average pairwise cosine distance between recommended items.
    
    Formula:
    ILD = (2 / (|R| * (|R| - 1))) * sum_{i < j} [ 1 - cos_sim(d_i, d_j) ]

    Interpretation:
    - 0.0: All recommended items have identical feature vectors (zero diversity).
    - Higher values (approaching 1.0 for orthogonal vectors) indicate diverse, non-redundant items.
    """
    if len(recommended_indices) < 2:
        return 0.0

    idx_list = list(recommended_indices)
    sub = feature_matrix[idx_list]

    if sparse.issparse(sub):
        sim_mat = (sub.dot(sub.T)).toarray()
    else:
        sim_mat = np.dot(sub, sub.T)

    # Cosine distance = 1 - cosine similarity
    dist_mat = 1.0 - sim_mat

    n = len(idx_list)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_distances = dist_mat[triu_indices]

    # Clip numerical precision artifacts to [0.0, 2.0]
    clipped = np.clip(pairwise_distances, 0.0, 2.0)
    return float(np.mean(clipped))


def topic_shannon_entropy(
    recommended_tags_list: Sequence[Sequence[str]],
    normalize: bool = True,
) -> float:
    """
    Measures the thematic spread of recommendations using Shannon Entropy.
    
    Formula:
    H = - sum_{tag} [ p(tag) * log2(p(tag)) ]
    Normalized H = H / log2(|Unique Tags|)
    """
    all_tags = []
    for tags in recommended_tags_list:
        for t in tags:
            cleaned = str(t).strip().lower()
            if cleaned:
                all_tags.append(cleaned)

    if not all_tags:
        return 0.0

    total_count = len(all_tags)
    counts = Counter(all_tags)
    num_unique = len(counts)

    if num_unique <= 1:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / total_count
        if p > 0:
            entropy -= p * np.log2(p)

    if normalize and num_unique > 1:
        max_entropy = np.log2(num_unique)
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    return float(entropy)


def topic_gini_simpson_index(
    recommended_tags_list: Sequence[Sequence[str]],
) -> float:
    """
    Gini-Simpson Diversity Index for topics:
    Probability that two randomly drawn tags from the list represent different topics.
    Formula: 1 - sum_{tag} p(tag)^2
    Range: [0.0 (monolithic), 1.0 (infinitely diverse)]
    """
    all_tags = []
    for tags in recommended_tags_list:
        for t in tags:
            cleaned = str(t).strip().lower()
            if cleaned:
                all_tags.append(cleaned)

    if len(all_tags) < 2:
        return 0.0

    total = len(all_tags)
    counts = Counter(all_tags)
    sum_sq_p = sum((c / total) ** 2 for c in counts.values())
    return float(np.clip(1.0 - sum_sq_p, 0.0, 1.0))


def duplicate_and_exclusion_integrity(
    recommended_ids: Sequence[int],
    seed_talk_id: Optional[int] = None,
    excluded_ids: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """
    Audits recommendation list for duplicate items, self-recommendations,
    and forbidden exclusion violations.
    """
    rec_list = list(recommended_ids)
    n_items = len(rec_list)
    unique_items = set(rec_list)
    duplicate_count = n_items - len(unique_items)

    has_self_recommendation = False
    if seed_talk_id is not None and seed_talk_id in unique_items:
        has_self_recommendation = True

    excluded_set = set(excluded_ids or set())
    exclusion_violations = unique_items.intersection(excluded_set)

    is_clean = (
        duplicate_count == 0
        and not has_self_recommendation
        and len(exclusion_violations) == 0
    )

    return {
        "is_clean": is_clean,
        "total_recommended": n_items,
        "unique_recommended": len(unique_items),
        "duplicate_count": duplicate_count,
        "has_self_recommendation": has_self_recommendation,
        "seed_talk_id": seed_talk_id,
        "exclusion_violations_count": len(exclusion_violations),
        "violation_talk_ids": sorted(list(exclusion_violations)),
    }

