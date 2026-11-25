"""
Automated Unit Tests for Pedagogical Complexity Estimator.
Phase 2, Step 1: Learning Platform Extensions.
"""

import math
import pytest
import pandas as pd
import numpy as np

from ted_recommender.learning import (
    ComplexityEstimatorConfig,
    ComplexityProfile,
    PedagogicalComplexityEstimator,
)


def test_estimator_initialization_and_weights_validation():
    """Verify default initialization and strict validation of weights and thresholds."""
    estimator = PedagogicalComplexityEstimator()
    assert estimator.config.duration_weight == 0.20
    assert estimator.config.lexical_weight == 0.40
    assert estimator.config.pacing_weight == 0.20
    assert estimator.config.domain_weight == 0.20

    # Weights not summing to 1.0 must raise ValueError
    with pytest.raises(ValueError, match="Feature weights must sum to 1.0"):
        ComplexityEstimatorConfig(duration_weight=0.5, lexical_weight=0.5, pacing_weight=0.5, domain_weight=0.5)

    # Inverted thresholds must raise ValueError
    with pytest.raises(ValueError, match="Stage thresholds must satisfy"):
        ComplexityEstimatorConfig(foundational_threshold=0.80, specialized_threshold=0.30)


def test_estimate_complexity_basic_transcript():
    """Verify standard talk with full transcript produces valid profile."""
    estimator = PedagogicalComplexityEstimator()
    sample_transcript = (
        "Good morning everyone. Today I want to explore how neural networks, "
        "algorithmic architecture, and computational neuroscience intersect to "
        "shape modern machine learning paradigms. The mathematical formalisms behind "
        "backpropagation require continuous gradient descent optimization."
    ) * 15  # Replicated to simulate normal speech length

    profile = estimator.estimate_complexity(
        talk_id=101,
        duration_minutes=12.5,
        transcript=sample_transcript,
        description="A talk on computational neuroscience and neural networks.",
        tags=["neuroscience", "technology", "artificial intelligence"],
    )

    assert isinstance(profile, ComplexityProfile)
    assert profile.talk_id == 101
    assert 0.0 <= profile.complexity_score <= 1.0
    assert profile.source == "transcript"
    assert profile.is_fallback is False
    assert profile.evidence_quality in ("high", "medium")
    assert profile.learning_stage in ("foundational", "intermediate", "specialized")

    # Check component scores
    scores = profile.feature_scores
    assert "duration_score" in scores
    assert "lexical_score" in scores
    assert "pacing_score" in scores
    assert "domain_score" in scores
    for k, v in scores.items():
        assert 0.0 <= v <= 1.0

    # Explanation and raw metrics
    assert len(profile.explanation) > 10
    assert profile.raw_metrics["duration_minutes"] == 12.5
    assert profile.raw_metrics["word_count"] > 100


def test_missing_transcript_fallback_handling():
    """Verify that talks lacking a transcript use metadata fallback gracefully."""
    estimator = PedagogicalComplexityEstimator()

    profile = estimator.estimate_complexity(
        talk_id=202,
        duration_minutes=15.0,
        transcript=None,  # Missing transcript
        description="An insightful presentation discussing global environmental policy and climate economics.",
        tags=["economics", "global issues", "environment"],
    )

    assert profile.talk_id == 202
    assert 0.0 <= profile.complexity_score <= 1.0
    assert profile.source == "metadata_fallback"
    assert profile.is_fallback is True
    assert profile.evidence_quality == "low"
    assert "metadata fallback" in profile.explanation


def test_short_and_malformed_transcripts():
    """Verify handling of empty strings, whitespace, and very short transcripts."""
    estimator = PedagogicalComplexityEstimator()

    # Empty string transcript
    p_empty = estimator.estimate_complexity(
        talk_id=301,
        duration_minutes=5.0,
        transcript="   ",
        description="Short description",
        tags=["art"],
    )
    assert p_empty.is_fallback is True
    assert 0.0 <= p_empty.complexity_score <= 1.0

    # Very short transcript (< 50 words)
    p_short = estimator.estimate_complexity(
        talk_id=302,
        duration_minutes=3.0,
        transcript="Hello world, thank you very much.",
        description="A short intro.",
        tags=["culture"],
    )
    assert p_short.is_fallback is True  # Falls back because < 50 tokens
    assert 0.0 <= p_short.complexity_score <= 1.0

    # Transcript with punctuation, numbers, and symbols
    p_punct = estimator.estimate_complexity(
        talk_id=303,
        duration_minutes=8.0,
        transcript="12345 !!! ??? @@@ $$$ %%% 999 1000",
        description="Punctuation test",
        tags=["design"],
    )
    assert 0.0 <= p_punct.complexity_score <= 1.0


def test_extreme_duration_handling():
    """Verify duration clamping prevents skew from ultra-short and marathon talks."""
    estimator = PedagogicalComplexityEstimator()
    dummy_text = "This is an introductory lecture explaining basic concepts of human communication." * 20

    # Ultra-short talk (1.5 minutes)
    p_ultra_short = estimator.estimate_complexity(
        talk_id=401,
        duration_minutes=1.5,
        transcript=dummy_text,
        tags=["communication"],
    )
    assert p_ultra_short.feature_scores["duration_score"] == 0.0  # Clamped to min bound
    assert 0.0 <= p_ultra_short.complexity_score <= 1.0

    # Marathon lecture (95 minutes)
    p_marathon = estimator.estimate_complexity(
        talk_id=402,
        duration_minutes=95.0,
        transcript=dummy_text,
        tags=["physics"],
    )
    assert p_marathon.feature_scores["duration_score"] == 1.0  # Clamped to max bound (30 min)
    assert 0.0 <= p_marathon.complexity_score <= 1.0


def test_deterministic_scoring():
    """Verify exact numerical repeatability across repeated evaluations."""
    estimator = PedagogicalComplexityEstimator()
    sample = "Detailed exploration of microbiology, genetic engineering, and cellular synthesis." * 15

    p1 = estimator.estimate_complexity(501, 14.0, sample, tags=["genetics", "biology"])
    p2 = estimator.estimate_complexity(501, 14.0, sample, tags=["genetics", "biology"])

    assert p1.complexity_score == p2.complexity_score
    assert p1.learning_stage == p2.learning_stage
    assert p1.feature_scores == p2.feature_scores


def test_learning_stage_boundary_thresholds():
    """Verify that different signal profiles map logically to learning stages."""
    # Config with calibrated thresholds for synthetic testing
    config = ComplexityEstimatorConfig(
        foundational_threshold=0.30,
        specialized_threshold=0.55,
    )
    estimator = PedagogicalComplexityEstimator(config)

    # 1. Accessible, short, conversational talk -> foundational
    conversational_text = (
        "I was at home with my mom and dad and we went to a park to see the dogs run and play. "
        "It was a good day and everyone felt happy and glad to be together."
    ) * 10
    p_foundational = estimator.estimate_complexity(
        talk_id=601,
        duration_minutes=4.0,
        transcript=conversational_text,
        tags=["humor", "storytelling", "family"],
    )
    assert p_foundational.learning_stage == "foundational"
    assert p_foundational.complexity_score < config.foundational_threshold

    # 2. Standard TED presentation with domain topic -> intermediate
    standard_text = (
        "In our research team, we designed an experimental framework to analyze urban architecture, "
        "sustainable transport networks, and municipal infrastructure across modern European cities."
    ) * 15
    p_intermediate = estimator.estimate_complexity(
        talk_id=602,
        duration_minutes=14.0,
        transcript=standard_text,
        tags=["technology", "architecture", "design"],
    )
    assert p_intermediate.learning_stage == "intermediate"
    assert (
        config.foundational_threshold
        <= p_intermediate.complexity_score
        < config.specialized_threshold
    )

    # 3. Highly technical, long, specialized talk -> specialized
    dense_text = (
        "Quantum electrodynamics, non-abelian gauge fields, and topological superconductivity "
        "manifest intricate mathematical symmetries within condensed matter physics and particle interactions."
    ) * 20
    p_specialized = estimator.estimate_complexity(
        talk_id=603,
        duration_minutes=25.0,
        transcript=dense_text,
        tags=["quantum", "physics", "mathematics"],
    )
    assert p_specialized.learning_stage == "specialized"
    assert p_specialized.complexity_score >= config.specialized_threshold


def test_batch_dataframe_estimation():
    """Verify estimate_dataframe processes tabular inputs cleanly."""
    estimator = PedagogicalComplexityEstimator()
    test_df = pd.DataFrame([
        {
            "talk_id": 1,
            "duration_minutes": 18.0,
            "transcript_clean": "Sir Ken Robinson discusses educational reform and nurturing creativity." * 20,
            "description": "Talk about school creativity.",
            "tags_list": "['education', 'creativity', 'children']",
        },
        {
            "talk_id": 2,
            "duration_minutes": 22.0,
            "transcript_clean": None,  # Missing transcript test
            "description": "Al Gore presents data on climate change and carbon mitigation.",
            "tags_list": "['environment', 'global issues']",
        },
    ])

    profiles = estimator.estimate_dataframe(test_df)
    assert len(profiles) == 2
    assert profiles[0].talk_id == 1
    assert profiles[0].is_fallback is False
    assert profiles[1].talk_id == 2
    assert profiles[1].is_fallback is True


def test_short_fallback_description_ttr_protection():
    """
    Verify that a short metadata fallback description (e.g., ~24 words like Talk #590)
    cannot produce artificially inflated MATTR (> 0.60) or unconstrained lexical score (> 0.90),
    sets ttr_reliable == False, and caps learning stage at intermediate.
    """
    estimator = PedagogicalComplexityEstimator()
    # Talk #590 description with 24 words
    short_desc = (
        "At Stanford University, primatologist Robert Sapolsky offers a fascinating and "
        "funny look at human behaviors which the rest of the animal kingdom would consider bizarre."
    )
    tags = ["biology", "brain", "humanity", "life", "mind", "primates", "psychology"]

    profile = estimator.estimate_complexity(
        talk_id=590,
        duration_minutes=37.43,  # Long lecture duration
        transcript=None,  # Fallback triggered
        description=short_desc,
        tags=tags,
    )

    assert profile.is_fallback is True
    assert profile.source == "metadata_fallback"
    assert profile.evidence_quality == "low"
    # Ensure ttr_reliable is False for short texts (< 100 tokens)
    assert profile.raw_metrics.get("ttr_reliable") is False
    # Ensure raw MATTR does not blow up to 1.0; imputed conservative median
    assert profile.raw_metrics.get("mattr") <= 0.60
    # Lexical score must be moderate (~0.55), down from previous unconstrained 0.9227
    assert profile.feature_scores["lexical_score"] <= 0.65
    # Capped at intermediate even with long duration and technical tags
    assert profile.learning_stage == "intermediate"
    assert "metadata fallback" in profile.explanation


