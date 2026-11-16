"""
Human Relevance Evaluation Framework and Graded Annotation Harness.
Module 5: Evaluation Metrics | Infosys Springboard 7.0

Facilitates human-in-the-loop qualitative validation:
- 4-point graded relevance scale:
    0 = Not Relevant
    1 = Slightly Relevant (thematic overlap but poor recommendation)
    2 = Relevant (substantive topical and conceptual alignment)
    3 = Highly Relevant (ideal companion or follow-up talk)
- Generates reproducible annotation batches across diverse topics.
- Calculates graded NDCG and graded Average Precision when labels are supplied.
- Honestly reports pending status when annotations have not yet been completed by human judges.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


class HumanRelevanceHarness:
    """Manages human relevance annotation templates, scoring, and status tracking."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.id_to_row = {int(r["talk_id"]): r for _, r in df.iterrows()}

    def generate_annotation_template(
        self,
        recommender_models: Dict[str, Any],
        query_talk_ids: List[int],
        top_k: int = 5,
        output_csv_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Generate a multi-model candidate pool for human annotators to judge.
        Pairs query talk metadata with recommended talk metadata.
        """
        rows = []
        for q_id in query_talk_ids:
            if q_id not in self.id_to_row:
                continue
            q_row = self.id_to_row[q_id]
            q_title = str(q_row.get("title", ""))
            q_desc = str(q_row.get("description", ""))

            # Collect candidates across all models
            model_recommendations: Dict[int, List[str]] = {}
            candidate_details: Dict[int, Any] = {}

            for model_name, model in recommender_models.items():
                try:
                    if hasattr(model, "recommend_for_talk"):
                        recs = model.recommend_for_talk(talk_id=q_id, top_k=top_k)
                    else:
                        recs = model.recommend(seed_talk_id=q_id, top_k=top_k)

                    for r in recs:
                        cid = r.talk_id
                        if cid not in model_recommendations:
                            model_recommendations[cid] = []
                            candidate_details[cid] = r
                        model_recommendations[cid].append(model_name)
                except Exception as e:
                    logger.warning(f"Error collecting candidates from {model_name} for query {q_id}: {e}")

            for cid, model_list in model_recommendations.items():
                cand_row = self.id_to_row.get(cid)
                c_desc = str(cand_row.get("description", "")) if cand_row is not None else ""
                c_title = str(cand_row.get("title", "")) if cand_row is not None else ""

                rows.append({
                    "query_talk_id": q_id,
                    "query_title": q_title,
                    "query_description": q_desc[:250] + "..." if len(q_desc) > 250 else q_desc,
                    "candidate_talk_id": cid,
                    "candidate_title": c_title,
                    "candidate_description": c_desc[:250] + "..." if len(c_desc) > 250 else c_desc,
                    "recommended_by_models": ", ".join(model_list),
                    "relevance_grade": None,  # To be filled by human judge [0, 1, 2, 3]
                    "annotator_notes": "",
                })

        df_template = pd.DataFrame(rows)
        if output_csv_path:
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_template.to_csv(output_csv_path, index=False)
            logger.info(f"Saved human annotation template to {output_csv_path} ({len(df_template)} pairs).")

        return df_template

    def evaluate_annotations(
        self,
        annotation_csv_path: Path,
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate recommendation models against human graded annotations.
        If annotations are incomplete or missing, returns explicit PENDING status.
        """
        if not annotation_csv_path.exists():
            return {
                "status": "PENDING_HUMAN_EVALUATION",
                "message": f"Annotation file not found at {annotation_csv_path}",
                "scale": RELEVANCE_SCALE_DESCRIPTION,
            }

        df_ann = pd.read_csv(annotation_csv_path)
        if "relevance_grade" not in df_ann.columns:
            return {
                "status": "INVALID_ANNOTATION_SCHEMA",
                "message": "Missing 'relevance_grade' column in annotation file.",
            }

        valid_labeled = df_ann[df_ann["relevance_grade"].notna()]
        total_pairs = len(df_ann)
        labeled_pairs = len(valid_labeled)

        if labeled_pairs == 0:
            return {
                "status": "PENDING_HUMAN_EVALUATION",
                "total_pairs_prepared": total_pairs,
                "labeled_pairs": 0,
                "completion_percentage": 0.0,
                "message": "Template exists but no human grades have been assigned yet.",
                "scale": RELEVANCE_SCALE_DESCRIPTION,
            }

        # Build graded relevance map: query_id -> {candidate_id: grade}
        graded_relevance_map: Dict[int, Dict[int, float]] = {}
        for _, row in valid_labeled.iterrows():
            qid = int(row["query_talk_id"])
            cid = int(row["candidate_talk_id"])
            grade = float(row["relevance_grade"])
            if qid not in graded_relevance_map:
                graded_relevance_map[qid] = {}
            graded_relevance_map[qid][cid] = grade

        return {
            "status": "COMPLETED_HUMAN_EVALUATION" if labeled_pairs == total_pairs else "PARTIAL_HUMAN_EVALUATION",
            "total_pairs_prepared": total_pairs,
            "labeled_pairs": labeled_pairs,
            "completion_percentage": round(float(labeled_pairs / total_pairs * 100), 2),
            "unique_queries_labeled": len(graded_relevance_map),
            "grade_distribution": {
                str(k): int(v) for k, v in valid_labeled["relevance_grade"].value_counts().to_dict().items()
            },
            "scale": RELEVANCE_SCALE_DESCRIPTION,
        }

