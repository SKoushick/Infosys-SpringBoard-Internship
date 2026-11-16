"""
Unit tests for Human Relevance Evaluation Framework and Graded Metrics.
Module 5: Evaluation Metrics | Infosys Springboard 7.0
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from evaluation.human_evaluation import (
    HumanRelevanceHarness,
    compute_graded_dcg,
    compute_graded_ndcg,
    compute_graded_ap,
    BINARY_RELEVANCE_THRESHOLD,
)


def test_hand_calculated_graded_dcg_and_ndcg():
    """
    Hand-calculated test case:
    Grades: [3.0, 0.0, 2.0] at K=3.
    DCG@3:
      Rank 1 (grade 3): (2^3 - 1) / log2(2) = 7.0 / 1.0 = 7.0
      Rank 2 (grade 0): (2^0 - 1) / log2(3) = 0.0
      Rank 3 (grade 2): (2^2 - 1) / log2(4) = 3.0 / 2.0 = 1.5
      Actual DCG = 7.0 + 0.0 + 1.5 = 8.5
    Ideal grades sorted: [3.0, 2.0, 0.0]
    IDCG@3:
      Rank 1 (grade 3): (2^3 - 1) / log2(2) = 7.0
      Rank 2 (grade 2): (2^2 - 1) / log2(3) = 3.0 / 1.5849625 = 1.89278926
      Rank 3 (grade 0): 0.0
      IDCG = 7.0 + 1.89278926 = 8.89278926
    Expected NDCG@3 = 8.5 / 8.89278926 = 0.95583057
    """
    grades = [3.0, 0.0, 2.0]
    actual_dcg = compute_graded_dcg(grades, k=3)
    assert actual_dcg == pytest.approx(8.5, rel=1e-5)

    actual_ndcg = compute_graded_ndcg(grades, k=3)
    expected_ndcg = 8.5 / (7.0 + (3.0 / np.log2(3.0)))
    assert actual_ndcg == pytest.approx(expected_ndcg, rel=1e-5)
    assert actual_ndcg == pytest.approx(0.95583, rel=1e-4)


def test_hand_calculated_graded_average_precision():
    """
    Hand-calculated test case:
    Grades: [3.0, 0.0, 2.0] at K=3, threshold=2.0.
    Rank 1: Grade 3 >= 2.0 -> Hit, Precision@1 = 1 / 1 = 1.0
    Rank 2: Grade 0 < 2.0  -> Miss
    Rank 3: Grade 2 >= 2.0 -> Hit, Precision@3 = 2 / 3 = 0.6666667
    Total relevant R = 2 (grades 3 and 2).
    Denominator = min(K=3, R=2) = 2.
    Expected AP@3 = (1.0 + 2/3) / 2 = 5/6 = 0.833333...
    """
    grades = [3.0, 0.0, 2.0]
    ap = compute_graded_ap(grades, k=3, threshold=2.0)
    assert ap == pytest.approx(5.0 / 6.0, rel=1e-5)
    assert ap == pytest.approx(0.83333, rel=1e-4)


def test_graded_metrics_edge_cases():
    # Empty list
    assert compute_graded_dcg([], k=5) == 0.0
    assert compute_graded_ndcg([], k=5) == 0.0
    assert compute_graded_ap([], k=5) == 0.0

    # K <= 0
    assert compute_graded_dcg([3, 2, 1], k=0) == 0.0
    assert compute_graded_ndcg([3, 2, 1], k=-1) == 0.0
    assert compute_graded_ap([3, 2, 1], k=0) == 0.0

    # All zeros (no relevance)
    assert compute_graded_dcg([0, 0, 0], k=3) == 0.0
    assert compute_graded_ndcg([0, 0, 0], k=3) == 0.0
    assert compute_graded_ap([0, 0, 0], k=3) == 0.0

    # Fewer items than K
    grades_short = [3.0]
    # At K=5: DCG = (2^3 - 1) / log2(2) = 7.0, IDCG = 7.0, NDCG = 1.0
    assert compute_graded_ndcg(grades_short, k=5) == 1.0
    # AP: Hit at rank 1: P@1 = 1/1 = 1.0. Denominator = min(5, 1) = 1. AP = 1.0
    assert compute_graded_ap(grades_short, k=5) == 1.0


def test_validation_rejects_invalid_grades():
    df_invalid = pd.DataFrame([
        {"query_talk_id": 1, "candidate_talk_id": 10, "model_id": "M1", "rank_position": 1, "relevance_grade": 4, "annotator_id": "ann1"},
        {"query_talk_id": 1, "candidate_talk_id": 11, "model_id": "M1", "rank_position": 2, "relevance_grade": -1, "annotator_id": "ann1"},
        {"query_talk_id": 1, "candidate_talk_id": 12, "model_id": "M1", "rank_position": 3, "relevance_grade": 2, "annotator_id": "ann1"},
    ])
    harness = HumanRelevanceHarness(pd.DataFrame({"talk_id": [1, 10, 11, 12]}))
    report = harness.validate_annotation_dataframe(df_invalid)
    assert report["is_valid"] is False
    assert report["invalid_grades_count"] == 2
    assert any("strictly in {0, 1, 2, 3}" in err for err in report["errors"])


def test_validation_detects_duplicate_records():
    df_dup = pd.DataFrame([
        {"query_talk_id": 1, "candidate_talk_id": 10, "model_id": "M1", "rank_position": 1, "relevance_grade": 2, "annotator_id": "ann1"},
        {"query_talk_id": 1, "candidate_talk_id": 10, "model_id": "M1", "rank_position": 1, "relevance_grade": 2, "annotator_id": "ann1"},
    ])
    harness = HumanRelevanceHarness(pd.DataFrame({"talk_id": [1, 10]}))
    report = harness.validate_annotation_dataframe(df_dup)
    assert report["is_valid"] is False
    assert report["duplicate_count"] == 2
    assert any("duplicate query-candidate-annotator" in err for err in report["errors"])


def test_pending_status_on_empty_and_missing_file(tmp_path):
    harness = HumanRelevanceHarness(pd.DataFrame({"talk_id": [1, 2]}))
    
    # Missing file
    missing_path = tmp_path / "nonexistent.csv"
    res_missing = harness.evaluate_annotations(missing_path)
    assert res_missing["status"] == "PENDING_HUMAN_EVALUATION"

    # Template with null grades
    template_path = tmp_path / "unlabeled_template.csv"
    df_unlabeled = pd.DataFrame([
        {"query_talk_id": 1, "candidate_talk_id": 2, "model_id": "M1", "rank_position": 1, "relevance_grade": None, "annotator_id": "ann1"},
        {"query_talk_id": 1, "candidate_talk_id": 3, "model_id": "M1", "rank_position": 2, "relevance_grade": None, "annotator_id": "ann1"},
    ])
    df_unlabeled.to_csv(template_path, index=False)
    res_unlabeled = harness.evaluate_annotations(template_path)
    assert res_unlabeled["status"] == "PENDING_HUMAN_EVALUATION"
    assert res_unlabeled["labeled_records_count"] == 0
    assert res_unlabeled["total_records_prepared"] == 2


def test_partial_and_completed_evaluation_with_per_model_breakdown(tmp_path):
    harness = HumanRelevanceHarness(pd.DataFrame({"talk_id": [1, 2, 3, 4]}))
    csv_path = tmp_path / "test_annotations.csv"

    # 4 rows: 2 for ModelA, 2 for ModelB. Only ModelA is fully labeled, ModelB is 50% labeled
    df_ann = pd.DataFrame([
        {"query_talk_id": 1, "candidate_talk_id": 2, "model_id": "ModelA", "rank_position": 1, "relevance_grade": 3, "annotator_id": "ann1"},
        {"query_talk_id": 1, "candidate_talk_id": 3, "model_id": "ModelA", "rank_position": 2, "relevance_grade": 2, "annotator_id": "ann1"},
        {"query_talk_id": 1, "candidate_talk_id": 4, "model_id": "ModelB", "rank_position": 1, "relevance_grade": 1, "annotator_id": "ann1"},
        {"query_talk_id": 1, "candidate_talk_id": 5, "model_id": "ModelB", "rank_position": 2, "relevance_grade": None, "annotator_id": "ann1"},
    ])
    df_ann.to_csv(csv_path, index=False)

    eval_partial = harness.evaluate_annotations(csv_path, k=2, threshold=2.0)
    assert eval_partial["status"] == "PARTIAL_HUMAN_EVALUATION"
    assert eval_partial["labeled_records_count"] == 3
    assert eval_partial["total_records_prepared"] == 4

    per_model = eval_partial["per_model_evaluation"]
    assert "ModelA" in per_model
    assert "ModelB" in per_model

    # ModelA: grades [3, 2] -> both >= 2 -> perfect NDCG@2 (1.0) and MAP@2 (1.0)
    assert per_model["ModelA"]["graded_ndcg_at_2"] == 1.0
    assert per_model["ModelA"]["graded_map_at_2"] == 1.0
    assert per_model["ModelA"]["query_coverage_ratio"] == 1.0
    assert per_model["ModelA"]["annotation_coverage_ratio"] == 1.0

    # ModelB: grade [1] (labeled count 1 of 2) -> grade 1 < 2 -> MAP = 0.0
    assert per_model["ModelB"]["graded_map_at_2"] == 0.0
    assert per_model["ModelB"]["annotation_coverage_ratio"] == 0.5

    # Now complete the final label
    df_ann.loc[3, "relevance_grade"] = 0
    df_ann.to_csv(csv_path, index=False)
    eval_complete = harness.evaluate_annotations(csv_path, k=2, threshold=2.0)
    assert eval_complete["status"] == "COMPLETED_HUMAN_EVALUATION"
    assert eval_complete["labeled_records_count"] == 4
    assert eval_complete["completion_percentage"] == 100.0

