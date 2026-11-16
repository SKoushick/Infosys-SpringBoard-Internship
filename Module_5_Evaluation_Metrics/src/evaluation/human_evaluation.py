"""
Human Relevance Evaluation Framework and Graded Annotation Harness.
Module 5: Evaluation Metrics | Infosys Springboard 7.0

Facilitates rigorous human-in-the-loop qualitative validation:
- 4-point graded relevance scale:
    0 = Not Relevant (completely different domain or unhelpful recommendation)
    1 = Slightly Relevant (thematic overlap or shared speaker, but poor recommendation)
    2 = Relevant (substantive topical and conceptual alignment)
    3 = Highly Relevant (ideal companion talk, direct sequel, or deep continuity)
- Preserves individual model identifiers and ranked candidate positions in annotation templates.
- Strict validation of relevance grades (must be in {0, 1, 2, 3}).
- Detects duplicate query-candidate-annotator records.
- Computes per-model Graded NDCG@K using exponential gains: (2^rel - 1) / log2(rank + 1).
- Computes Graded MAP@K under the explicitly documented convention that grades >= 2
  (Relevant / Highly Relevant) represent binary relevance hits.
- Reports per-model query coverage and annotation coverage.
- Strictly keeps status as PENDING_HUMAN_EVALUATION when no real human annotations exist.
"""

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .ranking_metrics import ndcg_at_k, precision_at_k

logger = logging.getLogger(__name__)

RELEVANCE_SCALE_DESCRIPTION = {
    0: "Not Relevant: Completely different domain, unhelpful recommendation.",
    1: "Slightly Relevant: Tangential keyword or speaker overlap, but poor thematic fit.",
    2: "Relevant: Clear shared subject matter, conceptually sound recommendation.",
    3: "Highly Relevant: Exemplary companion talk, direct sequel, or deep thematic continuity.",
}

# Explicit convention for graded MAP calculation
BINARY_RELEVANCE_THRESHOLD = 2.0  # Grade 2 and 3 are binary-relevant; 0 and 1 are non-relevant.


def compute_graded_dcg(grades: Sequence[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at rank K using standard exponential gain.
    Formula: sum_{i=1}^k [ (2^{grade_i} - 1) / log2(i + 1) ]
    """
    if k <= 0 or not grades:
        return 0.0
    top_grades = list(grades)[:k]
    dcg = 0.0
    for rank, grade in enumerate(top_grades, start=1):
        g = float(grade)
        if g > 0.0:
            gain = (2.0 ** g) - 1.0
            discount = np.log2(rank + 1.0)
            dcg += gain / discount
    return float(dcg)


def compute_graded_ndcg(grades: Sequence[float], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank K.
    Formula: DCG@K / IDCG@K
    Where IDCG@K is obtained by sorting all available candidate grades descending.
    """
    if k <= 0 or not grades:
        return 0.0
    actual_dcg = compute_graded_dcg(grades, k)
    if actual_dcg == 0.0:
        return 0.0

    ideal_grades = sorted([float(g) for g in grades if g > 0.0], reverse=True)[:k]
    idcg = compute_graded_dcg(ideal_grades, k)
    if idcg <= 0.0:
        return 0.0
    return float(np.clip(actual_dcg / idcg, 0.0, 1.0))


def compute_graded_ap(grades: Sequence[float], k: int, threshold: float = BINARY_RELEVANCE_THRESHOLD) -> float:
    """
    Compute Average Precision at rank K using a documented binary relevance threshold.
    
    Convention:
    A candidate with grade >= threshold is treated as a relevant hit (rel_i = 1),
    otherwise rel_i = 0.
    
    Formula:
    AP@K = (1 / min(k, R)) * sum_{i=1}^k [ P(i) * rel_i ]
    where R is the total number of relevant items in the candidate set for this query.
    """
    if k <= 0 or not grades:
        return 0.0

    top_grades = list(grades)[:k]
    hits = 0
    sum_precisions = 0.0

    for rank, grade in enumerate(top_grades, start=1):
        is_hit = 1 if float(grade) >= threshold else 0
        if is_hit:
            hits += 1
            sum_precisions += hits / rank

    if hits == 0:
        return 0.0

    # Total relevant candidates across all evaluated positions for this query
    total_relevant = sum(1 for g in grades if float(g) >= threshold)
    denominator = min(k, total_relevant)
    if denominator <= 0:
        return 0.0
    return float(sum_precisions / denominator)


class HumanRelevanceHarness:
    """Manages human relevance annotation templates, validation, and multi-model evaluation."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.id_to_row = {int(r["talk_id"]): r for _, r in df.iterrows()}

    def generate_annotation_template(
        self,
        recommender_models: Dict[str, Any],
        query_talk_ids: List[int],
        top_k: int = 5,
        output_csv_path: Optional[Path] = None,
        annotator_id: str = "annotator_1",
    ) -> pd.DataFrame:
        """
        Generate a multi-model candidate pool for human annotators to judge.
        Preserves individual model names and ranked positions for each query.
        """
        rows = []
        for q_id in query_talk_ids:
            if q_id not in self.id_to_row:
                continue
            q_row = self.id_to_row[q_id]
            q_title = str(q_row.get("title", ""))
            q_desc = str(q_row.get("description", ""))
            q_desc_trunc = q_desc[:250] + "..." if len(q_desc) > 250 else q_desc

            for model_name, model in recommender_models.items():
                try:
                    if hasattr(model, "recommend_for_talk"):
                        sig = inspect.signature(model.recommend_for_talk)
                        kwargs = {"talk_id": q_id, "top_k": top_k}
                        if "apply_mmr" in sig.parameters:
                            # For hybrid_mmr models, pass apply_mmr if model supports it
                            is_mmr = "mmr" in model_name.lower()
                            kwargs["apply_mmr"] = is_mmr
                        recs = model.recommend_for_talk(**kwargs)
                    else:
                        sig = inspect.signature(model.recommend)
                        kwargs = {"seed_talk_id": q_id, "top_k": top_k}
                        if "apply_mmr" in sig.parameters:
                            kwargs["apply_mmr"] = "mmr" in model_name.lower()
                        recs = model.recommend(**kwargs)

                    for rank_pos, r in enumerate(recs, start=1):
                        cid = r.talk_id
                        cand_row = self.id_to_row.get(cid)
                        c_desc = str(cand_row.get("description", "")) if cand_row is not None else ""
                        c_title = str(cand_row.get("title", "")) if cand_row is not None else ""
                        c_desc_trunc = c_desc[:250] + "..." if len(c_desc) > 250 else c_desc

                        rows.append({
                            "query_talk_id": q_id,
                            "query_title": q_title,
                            "query_description": q_desc_trunc,
                            "model_id": model_name,
                            "rank_position": rank_pos,
                            "candidate_talk_id": cid,
                            "candidate_title": c_title,
                            "candidate_description": c_desc_trunc,
                            "relevance_grade": None,  # To be filled by human judge [0, 1, 2, 3]
                            "annotator_id": annotator_id,
                            "annotator_notes": "",
                        })
                except Exception as e:
                    logger.warning(f"Error collecting candidates from {model_name} for query {q_id}: {e}")

        df_template = pd.DataFrame(rows)
        if output_csv_path:
            output_csv_path = Path(output_csv_path)
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_template.to_csv(output_csv_path, index=False)
            logger.info(f"Saved human annotation template to {output_csv_path} ({len(df_template)} candidate records).")

        return df_template

    def validate_annotation_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate schema, duplicate records, and grade values in an annotation DataFrame.
        """
        required_cols = {"query_talk_id", "candidate_talk_id", "model_id", "rank_position", "relevance_grade"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            return {
                "is_valid": False,
                "errors": [f"Missing required columns: {sorted(list(missing_cols))}"],
                "duplicate_count": 0,
                "invalid_grades_count": 0,
            }

        errors = []

        # Check duplicates
        dup_subset = ["query_talk_id", "candidate_talk_id", "model_id", "annotator_id"]
        valid_dup_subset = [c for c in dup_subset if c in df.columns]
        duplicates = df[df.duplicated(subset=valid_dup_subset, keep=False)]
        dup_count = len(duplicates)
        if dup_count > 0:
            errors.append(f"Detected {dup_count} duplicate query-candidate-annotator records.")

        # Check grade validity (grades must be null or in {0, 1, 2, 3})
        non_null_grades = df[df["relevance_grade"].notna()]["relevance_grade"]
        invalid_grades_count = 0
        valid_grade_set = {0, 1, 2, 3, 0.0, 1.0, 2.0, 3.0}

        for val in non_null_grades:
            try:
                num_val = float(val)
                if num_val not in valid_grade_set:
                    invalid_grades_count += 1
            except (ValueError, TypeError):
                invalid_grades_count += 1

        if invalid_grades_count > 0:
            errors.append(
                f"Found {invalid_grades_count} invalid relevance grades. Grades must be strictly in {{0, 1, 2, 3}}."
            )

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "duplicate_count": dup_count,
            "invalid_grades_count": invalid_grades_count,
        }

    def evaluate_annotations(
        self,
        annotation_csv_path: Path,
        k: int = 5,
        threshold: float = BINARY_RELEVANCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Evaluate recommendation models against human graded annotations.
        If annotations are incomplete, missing, or empty, honestly reports status.
        Calculates per-model Graded NDCG@K, Graded MAP@K, and coverage statistics.
        """
        annotation_csv_path = Path(annotation_csv_path)
        if not annotation_csv_path.exists():
            return {
                "status": "PENDING_HUMAN_EVALUATION",
                "message": f"Annotation file not found at {annotation_csv_path}",
                "scale": RELEVANCE_SCALE_DESCRIPTION,
                "binary_relevance_threshold": threshold,
            }

        df_ann = pd.read_csv(annotation_csv_path)
        val_report = self.validate_annotation_dataframe(df_ann)
        if not val_report["is_valid"]:
            return {
                "status": "INVALID_ANNOTATION_DATA",
                "validation_errors": val_report["errors"],
                "duplicate_count": val_report["duplicate_count"],
                "invalid_grades_count": val_report["invalid_grades_count"],
                "scale": RELEVANCE_SCALE_DESCRIPTION,
            }

        total_records = len(df_ann)
        labeled_df = df_ann[df_ann["relevance_grade"].notna()].copy()
        labeled_count = len(labeled_df)

        if labeled_count == 0:
            return {
                "status": "PENDING_HUMAN_EVALUATION",
                "total_records_prepared": total_records,
                "labeled_records_count": 0,
                "completion_percentage": 0.0,
                "message": "Template exists with prepared candidate pools, but no human relevance grades have been assigned yet.",
                "scale": RELEVANCE_SCALE_DESCRIPTION,
                "binary_relevance_threshold": threshold,
            }

        # Convert grade to float
        labeled_df["relevance_grade"] = labeled_df["relevance_grade"].astype(float)

        # Evaluate per model
        models_in_data = df_ann["model_id"].unique() if "model_id" in df_ann.columns else ["Unknown"]
        per_model_results = {}

        for m_id in models_in_data:
            m_total_df = df_ann[df_ann["model_id"] == m_id]
            m_labeled_df = labeled_df[labeled_df["model_id"] == m_id]

            total_m_queries = m_total_df["query_talk_id"].nunique()
            labeled_m_queries = m_labeled_df["query_talk_id"].nunique()
            total_m_recs = len(m_total_df)
            labeled_m_recs = len(m_labeled_df)

            query_coverage = round(float(labeled_m_queries / total_m_queries), 4) if total_m_queries > 0 else 0.0
            annotation_coverage = round(float(labeled_m_recs / total_m_recs), 4) if total_m_recs > 0 else 0.0

            ndcg_list = []
            map_list = []

            for q_id, q_group in m_labeled_df.groupby("query_talk_id"):
                q_sorted = q_group.sort_values(by="rank_position")
                grades = q_sorted["relevance_grade"].tolist()

                q_ndcg = compute_graded_ndcg(grades, k=k)
                q_ap = compute_graded_ap(grades, k=k, threshold=threshold)

                ndcg_list.append(q_ndcg)
                map_list.append(q_ap)

            mean_ndcg = round(float(np.mean(ndcg_list)), 4) if ndcg_list else 0.0
            mean_map = round(float(np.mean(map_list)), 4) if map_list else 0.0

            grade_counts = m_labeled_df["relevance_grade"].value_counts().to_dict()
            grade_dist = {str(int(g)): int(cnt) for g, cnt in grade_counts.items()}

            per_model_results[str(m_id)] = {
                "total_queries_in_template": int(total_m_queries),
                "labeled_queries_count": int(labeled_m_queries),
                "query_coverage_ratio": query_coverage,
                "total_recommendations_count": int(total_m_recs),
                "labeled_recommendations_count": int(labeled_m_recs),
                "annotation_coverage_ratio": annotation_coverage,
                f"graded_ndcg_at_{k}": mean_ndcg,
                f"graded_map_at_{k}": mean_map,
                "grade_distribution": grade_dist,
            }

        overall_status = "COMPLETED_HUMAN_EVALUATION" if labeled_count == total_records else "PARTIAL_HUMAN_EVALUATION"

        return {
            "status": overall_status,
            "total_records_prepared": total_records,
            "labeled_records_count": labeled_count,
            "completion_percentage": round(float(labeled_count / total_records * 100.0), 2),
            "binary_relevance_threshold": threshold,
            "relevance_threshold_convention": f"Grades >= {threshold} (Relevant / Highly Relevant) are treated as binary-relevant hits for Graded MAP.",
            "overall_grade_distribution": {
                str(int(g)): int(cnt) for g, cnt in labeled_df["relevance_grade"].value_counts().to_dict().items()
            },
            "per_model_evaluation": per_model_results,
            "scale": RELEVANCE_SCALE_DESCRIPTION,
        }
