"""
Unit tests for Catalogue Coverage, Recommendation Novelty, and Gini Index.
Module 5: Evaluation Metrics | Infosys Springboard 7.0
"""

import pytest
import numpy as np

from evaluation.coverage_metrics import (
    catalogue_coverage,
    recommendation_novelty,
    gini_coefficient_recommendations,
)


def test_catalogue_coverage():
    recs = [1, 2, 3, 4, 5, 1, 2]  # 5 unique items
    cov = catalogue_coverage(recs, total_catalogue_size=10)
    assert cov["coverage_ratio"] == 0.5
    assert cov["unique_items_recommended"] == 5
    assert cov["total_catalogue_size"] == 10

    # Zero catalogue
    assert catalogue_coverage(recs, total_catalogue_size=0)["coverage_ratio"] == 0.0
    # Negative catalogue
    assert catalogue_coverage(recs, total_catalogue_size=-5)["coverage_ratio"] == 0.0
    # Empty recs
    assert catalogue_coverage([], total_catalogue_size=10)["coverage_ratio"] == 0.0


def test_recommendation_novelty():
    # Catalogue with 2 items: item 1 (views=1000), item 2 (views=10)
    # Total views = 1010
    views_map = {1: 1000, 2: 10}

    # Recommending item 2 (niche/rare) should yield much higher novelty bits than item 1
    nov_rare = recommendation_novelty([2], views_map)
    nov_pop = recommendation_novelty([1], views_map)

    assert nov_rare["mean_novelty_bits"] > nov_pop["mean_novelty_bits"]
    assert nov_rare["mean_novelty_bits"] > 5.0  # -log2(10/1010) ~ 6.65 bits

    # Empty
    assert recommendation_novelty([], views_map)["mean_novelty_bits"] == 0.0
    # Views map with zero or empty
    assert recommendation_novelty([1, 2], {})["mean_novelty_bits"] == 0.0


def test_gini_coefficient_recommendations():
    # Perfectly equal recommendations: each item recommended once
    equal_recs = [1, 2, 3, 4, 5]
    gini_equal = gini_coefficient_recommendations(equal_recs, total_catalogue_size=5)
    assert pytest.approx(gini_equal, abs=1e-2) == 0.0

    # Highly unequal: item 1 recommended 100 times, 99 items never recommended
    unequal_recs = [1] * 100
    gini_unequal = gini_coefficient_recommendations(unequal_recs, total_catalogue_size=100)
    assert gini_unequal > 0.9  # High inequality

    # Empty recommendations
    assert gini_coefficient_recommendations([], total_catalogue_size=50) == 0.0
    # Zero or negative catalogue size
    assert gini_coefficient_recommendations([1, 2], total_catalogue_size=0) == 0.0
