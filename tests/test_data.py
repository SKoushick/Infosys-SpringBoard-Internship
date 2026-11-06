"""Unit tests for dataset loading, schema validation, and row alignment."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ted_recommender.config import FEATURES_CSV_PATH, RAW_DATA_PATH
from ted_recommender.data import (
    load_dataset,
    validate_dataset,
    verify_row_alignment,
    parse_tags,
    clean_text_light,
    clean_text_aggressive,
)


def test_features_dataset_loading():
    """Verify that features CSV loads properly with 2550 rows."""
    df = load_dataset(stage="features")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2550
    assert "talk_id" in df.columns
    assert "title_processed" in df.columns
    assert "description_processed" in df.columns
    assert "tags_processed" in df.columns
    assert "transcript_processed" in df.columns


def test_dataset_validation():
    """Check dataset validation reporting."""
    df = load_dataset(stage="features")
    report = validate_dataset(df)
    assert isinstance(report, dict)
    assert report["total_records"] == 2550
    assert report["duplicate_talk_ids"] == 0
    assert report["missing_titles"] == 0
    assert report["missing_descriptions"] == 0


def test_row_alignment_check():
    """Test row alignment verifier."""
    df = pd.DataFrame({"talk_id": [1, 2, 3], "title": ["A", "B", "C"]})
    assert verify_row_alignment(df, 3, "test_mat") is True

    with pytest.raises(ValueError):
        verify_row_alignment(df, 4, "test_mat")


def test_tag_parsing():
    """Check parsing of diverse stringified list formats."""
    raw_str = "['technology', 'education', 'AI']"
    tags = parse_tags(raw_str)
    assert tags == ["technology", "education", "ai"]

    # Comma-separated fallback
    raw_csv = "Science, Health, Biology"
    tags_csv = parse_tags(raw_csv)
    assert tags_csv == ["science", "health", "biology"]

    # Null handling
    assert parse_tags(None) == []
    assert parse_tags(np.nan) == []


def test_text_cleaning():
    """Verify HTML removal, URL stripping, and negation preservation."""
    sample = "Watch this <p>talk</p> at https://ted.com/talks. It was not boring!"
    light = clean_text_light(sample)
    assert "<p>" not in light
    assert "https://" not in light
    assert "not boring" in light  # Negation preserved

    aggr = clean_text_aggressive(sample)
    assert "<p>" not in aggr
    assert "https" not in aggr
    assert "not boring" in aggr

