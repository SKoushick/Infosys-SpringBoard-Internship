"""
Catalogue Coverage, Novelty, and Exposure Distribution Metrics.
Module 5: Evaluation Metrics | Infosys Springboard 7.0

Implements:
- Catalogue Coverage: Proportion of the total catalogue recommended across queries.
- Recommendation Novelty: Information-theoretic unexpectedness based on view distributions.
- Gini Coefficient of Item Exposure: Quantifies recommendation inequality / superstar bias.
"""

from collections import Counter
from typing import Any, Dict, List, Sequence, Set
import numpy as np


def catalogue_coverage(
    all_recommended_ids: Sequence[int],
    total_catalogue_size: int,
) -> Dict[str, Any]:
    """
    Measures the proportion of the catalog that appears in recommendations.
    
    Formula:
    Coverage = |bigcup_{q in Q} Rec(q)| / |Catalog|
    """
    if total_catalogue_size <= 0:
        return {
            "coverage_ratio": 0.0,
            "unique_items_recommended": 0,
            "total_catalogue_size": total_catalogue_size,
        }

    unique_items = set(all_recommended_ids)
    coverage_ratio = len(unique_items) / float(total_catalogue_size)

    return {
        "coverage_ratio": round(float(np.clip(coverage_ratio, 0.0, 1.0)), 4),
        "unique_items_recommended": len(unique_items),
        "total_catalogue_size": int(total_catalogue_size),
    }


def recommendation_novelty(
    all_recommended_ids: Sequence[int],
    item_views_map: Dict[int, int],
) -> Dict[str, Any]:
    """
    Measures information-theoretic novelty based on TED talk view counts.
    Formula:
    Novelty = (1 / |R|) * sum_{i in R} [ -log2(P(item_i)) ]
    where P(item_i) = views(item_i) / total_catalogue_views
    
    Interpretation:
    Higher values indicate the recommender introduces long-tail or niche content
    rather than simply echoing blockbuster, high-view items.
    """
    if not all_recommended_ids or not item_views_map:
        return {
            "mean_novelty_bits": 0.0,
            "median_novelty_bits": 0.0,
            "total_items_evaluated": 0,
        }

    total_views = sum(max(1, v) for v in item_views_map.values())
    if total_views <= 0:
        return {
            "mean_novelty_bits": 0.0,
            "median_novelty_bits": 0.0,
            "total_items_evaluated": 0,
        }

    novelties = []
    for item_id in all_recommended_ids:
        views = item_views_map.get(int(item_id), 1)
        prob = max(1, views) / float(total_views)
        self_information = -np.log2(prob)
        novelties.append(self_information)

    return {
        "mean_novelty_bits": round(float(np.mean(novelties)), 4),
        "median_novelty_bits": round(float(np.median(novelties)), 4),
        "min_novelty_bits": round(float(np.min(novelties)), 4),
        "max_novelty_bits": round(float(np.max(novelties)), 4),
        "total_items_evaluated": len(novelties),
    }


def gini_coefficient_recommendations(
    all_recommended_ids: Sequence[int],
    total_catalogue_size: int,
) -> float:
    """
    Calculates the Gini coefficient of recommendation frequencies across the catalogue.
    
    Range:
    - 0.0: Perfectly equal exposure (every catalog item recommended identical number of times).
    - 1.0: Extreme superstar concentration (only 1 item ever recommended).
    """
    if total_catalogue_size <= 0:
        return 0.0

    counts_map = Counter(all_recommended_ids)
    # Include zero-frequency items
    num_unrecommended = max(0, total_catalogue_size - len(counts_map))
    frequencies = np.array(list(counts_map.values()) + [0] * num_unrecommended, dtype=np.float64)

    if np.sum(frequencies) == 0:
        return 0.0

    # Sorted array
    sorted_freqs = np.sort(frequencies)
    n = len(sorted_freqs)
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_freqs) / (n * np.sum(sorted_freqs))) - ((n + 1.0) / n)
    return float(np.clip(gini, 0.0, 1.0))

