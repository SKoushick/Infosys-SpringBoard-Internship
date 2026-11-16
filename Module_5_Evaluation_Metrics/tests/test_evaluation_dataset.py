"""
Unit tests for Evaluation Dataset Builder and Quality Validator.
Module 5: Evaluation Metrics | Infosys Springboard 7.0
"""

import pytest
import pandas as pd
import numpy as np

from evaluation.evaluation_dataset import (
    EvaluationDatasetBuilder,
    parse_raw_related_talks,
)


def test_parse_raw_related_talks():
    # Stringified list of dicts (TED format)
    raw_str = "[{'id': 865, 'title': 'Bring on the learning revolution!'}, {'id': '1738', 'title': 'Escaping death valley'}]"
    parsed = parse_raw_related_talks(raw_str)
    assert parsed == [865, 1738]

    # Simple list of ints
    assert parse_raw_related_talks([1, 2, 3]) == [1, 2, 3]

    # Null / empty
    assert parse_raw_related_talks(None) == []
    assert parse_raw_related_talks(np.nan) == []
    assert parse_raw_related_talks("[]") == []
    assert parse_raw_related_talks("invalid syntax") == []


def test_dataset_builder_validation():
    # Catalogue with 3 talks: 1, 2, 3
    # Talk 1 points to 2, 3, and 999 (out of catalogue) and 1 (self)
    df = pd.DataFrame({
        "talk_id": [1, 2, 3],
        "title": ["Talk 1", "Talk 2", "Talk 3"],
        "main_speaker": ["A", "B", "C"],
        "related_talks": [
            "[{'id': 2}, {'id': 3}, {'id': 999}, {'id': 1}]",
            "[{'id': 1}]",
            "[]",
        ],
        "has_transcript": [True, True, False],
        "tags": ["['tech']", "['science']", "['art']"],
    })

    builder = EvaluationDatasetBuilder(df)
    records, report = builder.build()

    # Talk 1 should retain only 2 and 3 (999 dropped as out of catalog, 1 dropped as self)
    assert 1 in records
    assert records[1].relevant_talk_ids == [2, 3]
    assert records[1].num_relevant == 2

    # Talk 2 should retain 1
    assert 2 in records
    assert records[2].relevant_talk_ids == [1]

    # Talk 3 has no valid links -> not included in eval records
    assert 3 not in records

    # Check report counts
    assert report.total_catalogue_talks == 3
    assert report.talks_with_valid_ground_truth == 2
    assert report.talks_without_valid_ground_truth == 1
    assert report.out_of_catalogue_links_dropped == 1  # 999
    assert report.self_references_dropped == 1  # 1


def test_deterministic_sampling():
    df = pd.DataFrame({
        "talk_id": list(range(1, 21)),
        "title": [f"Talk {i}" for i in range(1, 21)],
        "related_talks": [f"[{{'id': {(i % 20) + 1}}}]" for i in range(1, 21)],
        "has_transcript": [True] * 15 + [False] * 5,
        "tags": ["['tech']"] * 20,
    })

    builder = EvaluationDatasetBuilder(df)
    sample_1 = builder.sample_queries(sample_size=6, random_seed=42)
    sample_2 = builder.sample_queries(sample_size=6, random_seed=42)

    assert sample_1 == sample_2
    assert len(sample_1) == 6

