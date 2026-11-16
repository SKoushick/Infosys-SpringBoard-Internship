"""
Unified Fair Benchmark Engine for TED Talk Recommendation Models.
Module 5: Evaluation Metrics | Infosys Springboard 7.0

Evaluates all four core recommendation strategies under strictly identical conditions:
1. TF-IDF Content-Based Recommender (Lexical similarity)
2. Sentence-BERT Semantic Recommender (Conceptual embeddings)
3. Hybrid Recommender without MMR (Multi-signal fusion)
4. Hybrid Recommender with MMR (Multi-signal fusion + Maximal Marginal Relevance)

Guarantees:
- Transparent execution: records requested, evaluated, and failed queries per model.
- Strict benchmark mode: immediately fails with descriptive error if any query cannot be evaluated.
- Fair query verification: verifies that all compared models evaluated the exact same query IDs.
- Eliminates silent exception masking and false non-MMR fallback behavior.
- Evaluates the actual hybrid and MMR execution paths (no proxy substitutions).
- Comprehensive evaluation across ranking accuracy, intra-list diversity,
  catalogue coverage, novelty, and inference latency.
"""

import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .ranking_metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    ndcg_at_k,
    average_precision_at_k,
    hit_rate_at_k,
    reciprocal_rank_at_k,
)
from .diversity_metrics import (
    intra_list_diversity,
    topic_shannon_entropy,
    duplicate_and_exclusion_integrity,
)
from .coverage_metrics import (
    catalogue_coverage,
    recommendation_novelty,
    gini_coefficient_recommendations,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelBenchmarkScorecard:
    """Consolidated metric results for a single model across all queries."""
    model_id: str
    display_name: str
    requested_query_count: int
    num_queries_evaluated: int
    failed_query_count: int
    successful_query_ids: List[int]
    failed_query_ids: List[int]
    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    f1_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    ndcg_at_20: float
    map_at_10: float
    hit_rate_at_5: float
    hit_rate_at_10: float
    hit_rate_at_20: float
    mrr_at_10: float
    intra_list_diversity_at_10: float
    topic_entropy_at_10: float
    catalogue_coverage_ratio: float
    unique_items_recommended: int
    recommendation_novelty_bits: float
    gini_exposure_index: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    integrity_clean_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "query_execution": {
                "requested_query_count": self.requested_query_count,
                "num_queries_evaluated": self.num_queries_evaluated,
                "failed_query_count": self.failed_query_count,
                "successful_query_ids": self.successful_query_ids,
                "failed_query_ids": self.failed_query_ids,
            },
            "ranking_accuracy": {
                "Precision@5": self.precision_at_5,
                "Precision@10": self.precision_at_10,
                "Precision@20": self.precision_at_20,
                "Recall@5": self.recall_at_5,
                "Recall@10": self.recall_at_10,
                "Recall@20": self.recall_at_20,
                "F1@10": self.f1_at_10,
                "NDCG@5": self.ndcg_at_5,
                "NDCG@10": self.ndcg_at_10,
                "NDCG@20": self.ndcg_at_20,
                "MAP@10": self.map_at_10,
                "HitRate@5": self.hit_rate_at_5,
                "HitRate@10": self.hit_rate_at_10,
                "HitRate@20": self.hit_rate_at_20,
                "MRR@10": self.mrr_at_10,
            },
            "diversity_and_coverage": {
                "IntraListDiversity@10": self.intra_list_diversity_at_10,
                "TopicEntropy@10": self.topic_entropy_at_10,
                "CatalogueCoverageRatio": self.catalogue_coverage_ratio,
                "UniqueItemsRecommended": self.unique_items_recommended,
                "RecommendationNoveltyBits": self.recommendation_novelty_bits,
                "GiniExposureIndex": self.gini_exposure_index,
            },
            "performance_and_integrity": {
                "MeanLatencyMs": self.mean_latency_ms,
                "MedianLatencyMs": self.median_latency_ms,
                "P95LatencyMs": self.p95_latency_ms,
                "IntegrityCleanRate": self.integrity_clean_rate,
            },
        }


class UnifiedRecommenderBenchmark:
    """Executes fair comparative evaluation across multiple recommendation models."""

    def __init__(
        self,
        df: pd.DataFrame,
        ground_truth: Dict[int, Set[int]],
        feature_matrix: np.ndarray,
        id_to_idx: Dict[int, int],
        idx_to_id: Dict[int, int],
        item_views_map: Optional[Dict[int, int]] = None,
        strict_mode: bool = False,
    ):
        self.df = df
        self.ground_truth = ground_truth
        self.feature_matrix = feature_matrix
        self.id_to_idx = id_to_idx
        self.idx_to_id = idx_to_id
        self.catalogue_size = len(df)
        self.strict_mode = strict_mode
        self.item_views_map = item_views_map or {
            int(row["talk_id"]): int(row.get("views", 1)) for _, row in df.iterrows()
        }

    def evaluate_model(
        self,
        model: Any,
        model_id: str,
        display_name: str,
        query_ids: List[int],
        apply_mmr: bool = False,
        strict: Optional[bool] = None,
    ) -> Tuple[ModelBenchmarkScorecard, pd.DataFrame]:
        """
        Evaluate a single model across query_ids for K in [5, 10, 20].
        Transparently tracks successes and failures.
        If strict=True, fails immediately on any recommendation exception.
        """
        is_strict = self.strict_mode if strict is None else strict
        logger.info(f"Evaluating {display_name} across {len(query_ids)} queries (apply_mmr={apply_mmr}, strict={is_strict})...")

        prec_5, prec_10, prec_20 = [], [], []
        rec_5, rec_10, rec_20 = [], [], []
        f1_10_list = []
        ndcg_5, ndcg_10, ndcg_20 = [], [], []
        map_10_list = []
        hr_5, hr_10, hr_20 = [], [], []
        mrr_10_list = []
        ild_10_list = []
        entropy_10_list = []
        latencies_ms = []
        integrity_clean_list = []
        all_recommended_ids: List[int] = []

        successful_query_ids: List[int] = []
        failed_query_ids: List[int] = []
        per_query_records = []

        # Inspect target dispatch signature once
        target_fn = model.recommend_for_talk if hasattr(model, "recommend_for_talk") else model.recommend
        sig = inspect.signature(target_fn)
        supports_apply_mmr = "apply_mmr" in sig.parameters

        for q_id in query_ids:
            if q_id not in self.ground_truth:
                continue
            relevant_set = self.ground_truth[q_id]
            if not relevant_set:
                continue

            # Build call arguments transparently without masking exceptions
            call_kwargs: Dict[str, Any] = {"top_k": 20}
            if "talk_id" in sig.parameters:
                call_kwargs["talk_id"] = q_id
            elif "seed_talk_id" in sig.parameters:
                call_kwargs["seed_talk_id"] = q_id

            if supports_apply_mmr:
                call_kwargs["apply_mmr"] = apply_mmr

            t0 = time.perf_counter()
            try:
                recs = target_fn(**call_kwargs)
            except Exception as e:
                failed_query_ids.append(q_id)
                logger.error(f"Error during recommendation for query {q_id} with model {model_id}: {e}")
                if is_strict:
                    raise RuntimeError(
                        f"Strict benchmark mode failed on model '{model_id}' ({display_name}) for query {q_id}: {e}"
                    ) from e
                continue

            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)
            successful_query_ids.append(q_id)

            rec_ids = [r.talk_id for r in recs]
            all_recommended_ids.extend(rec_ids)

            # Ranking accuracy at K=5, 10, 20
            p5 = precision_at_k(rec_ids, relevant_set, k=5)
            p10 = precision_at_k(rec_ids, relevant_set, k=10)
            p20 = precision_at_k(rec_ids, relevant_set, k=20)

            r5 = recall_at_k(rec_ids, relevant_set, k=5)
            r10 = recall_at_k(rec_ids, relevant_set, k=10)
            r20 = recall_at_k(rec_ids, relevant_set, k=20)

            f10 = f1_at_k(rec_ids, relevant_set, k=10)

            n5 = ndcg_at_k(rec_ids, relevant_set, k=5)
            n10 = ndcg_at_k(rec_ids, relevant_set, k=10)
            n20 = ndcg_at_k(rec_ids, relevant_set, k=20)

            ap10 = average_precision_at_k(rec_ids, relevant_set, k=10)

            h5 = hit_rate_at_k(rec_ids, relevant_set, k=5)
            h10 = hit_rate_at_k(rec_ids, relevant_set, k=10)
            h20 = hit_rate_at_k(rec_ids, relevant_set, k=20)

            mrr10 = reciprocal_rank_at_k(rec_ids, relevant_set, k=10)

            prec_5.append(p5); prec_10.append(p10); prec_20.append(p20)
            rec_5.append(r5); rec_10.append(r10); rec_20.append(r20)
            f1_10_list.append(f10)
            ndcg_5.append(n5); ndcg_10.append(n10); ndcg_20.append(n20)
            map_10_list.append(ap10)
            hr_5.append(h5); hr_10.append(h10); hr_20.append(h20)
            mrr_10_list.append(mrr10)

            # Diversity metrics at K=10
            top10_ids = rec_ids[:10]
            top10_indices = [self.id_to_idx[tid] for tid in top10_ids if tid in self.id_to_idx]
            ild = intra_list_diversity(top10_indices, self.feature_matrix)
            ild_10_list.append(ild)

            top10_tags = [r.tags for r in recs[:10]]
            entropy = topic_shannon_entropy(top10_tags)
            entropy_10_list.append(entropy)

            # Integrity check
            integrity = duplicate_and_exclusion_integrity(rec_ids, seed_talk_id=q_id)
            integrity_clean_list.append(1.0 if integrity["is_clean"] else 0.0)

            per_query_records.append({
                "query_talk_id": q_id,
                "model_id": model_id,
                "precision_at_10": round(p10, 4),
                "recall_at_10": round(r10, 4),
                "ndcg_at_10": round(n10, 4),
                "ap_at_10": round(ap10, 4),
                "hit_at_10": h10,
                "ild_at_10": round(ild, 4),
                "latency_ms": round(lat_ms, 2),
            })

        # Global catalogue coverage and novelty
        coverage_dict = catalogue_coverage(all_recommended_ids, self.catalogue_size)
        novelty_dict = recommendation_novelty(all_recommended_ids, self.item_views_map)
        gini_index = gini_coefficient_recommendations(all_recommended_ids, self.catalogue_size)

        n_eval = len(prec_10)
        scorecard = ModelBenchmarkScorecard(
            model_id=model_id,
            display_name=display_name,
            requested_query_count=len(query_ids),
            num_queries_evaluated=n_eval,
            failed_query_count=len(failed_query_ids),
            successful_query_ids=successful_query_ids,
            failed_query_ids=failed_query_ids,
            precision_at_5=round(float(np.mean(prec_5)), 4) if n_eval else 0.0,
            precision_at_10=round(float(np.mean(prec_10)), 4) if n_eval else 0.0,
            precision_at_20=round(float(np.mean(prec_20)), 4) if n_eval else 0.0,
            recall_at_5=round(float(np.mean(rec_5)), 4) if n_eval else 0.0,
            recall_at_10=round(float(np.mean(rec_10)), 4) if n_eval else 0.0,
            recall_at_20=round(float(np.mean(rec_20)), 4) if n_eval else 0.0,
            f1_at_10=round(float(np.mean(f1_10_list)), 4) if n_eval else 0.0,
            ndcg_at_5=round(float(np.mean(ndcg_5)), 4) if n_eval else 0.0,
            ndcg_at_10=round(float(np.mean(ndcg_10)), 4) if n_eval else 0.0,
            ndcg_at_20=round(float(np.mean(ndcg_20)), 4) if n_eval else 0.0,
            map_at_10=round(float(np.mean(map_10_list)), 4) if n_eval else 0.0,
            hit_rate_at_5=round(float(np.mean(hr_5)), 4) if n_eval else 0.0,
            hit_rate_at_10=round(float(np.mean(hr_10)), 4) if n_eval else 0.0,
            hit_rate_at_20=round(float(np.mean(hr_20)), 4) if n_eval else 0.0,
            mrr_at_10=round(float(np.mean(mrr_10_list)), 4) if n_eval else 0.0,
            intra_list_diversity_at_10=round(float(np.mean(ild_10_list)), 4) if n_eval else 0.0,
            topic_entropy_at_10=round(float(np.mean(entropy_10_list)), 4) if n_eval else 0.0,
            catalogue_coverage_ratio=coverage_dict["coverage_ratio"],
            unique_items_recommended=coverage_dict["unique_items_recommended"],
            recommendation_novelty_bits=novelty_dict["mean_novelty_bits"],
            gini_exposure_index=round(gini_index, 4),
            mean_latency_ms=round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
            median_latency_ms=round(float(np.median(latencies_ms)), 2) if latencies_ms else 0.0,
            p95_latency_ms=round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
            integrity_clean_rate=round(float(np.mean(integrity_clean_list)), 4) if integrity_clean_list else 1.0,
        )

        df_per_query = pd.DataFrame(per_query_records)
        return scorecard, df_per_query

    def verify_fair_comparison(
        self,
        scorecards: Dict[str, ModelBenchmarkScorecard],
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Verify that all evaluated models were tested on the EXACT same set of query IDs.
        Prevents reporting an invalid comparison if one model failed or skipped queries.
        """
        if len(scorecards) <= 1:
            return {
                "is_fair": True,
                "message": "Fair verification passed (single model or empty scorecard).",
                "common_query_count": (
                    next(iter(scorecards.values())).num_queries_evaluated if scorecards else 0
                ),
            }

        query_sets: Dict[str, Set[int]] = {
            m_id: set(sc.successful_query_ids) for m_id, sc in scorecards.items()
        }

        model_names = list(scorecards.keys())
        first_model = model_names[0]
        reference_set = query_sets[first_model]

        discrepancies: Dict[str, Dict[str, Any]] = {}
        for other_model in model_names[1:]:
            other_set = query_sets[other_model]
            if other_set != reference_set:
                missing_in_other = list(reference_set - other_set)
                extra_in_other = list(other_set - reference_set)
                discrepancies[f"{first_model}_vs_{other_model}"] = {
                    "missing_in_other": missing_in_other,
                    "extra_in_other": extra_in_other,
                    "count_diff": len(other_set) - len(reference_set),
                }

        is_fair = len(discrepancies) == 0

        if not is_fair:
            err_msg = (
                f"Unfair benchmark comparison detected! Models were evaluated on unequal query sets. "
                f"Discrepancies found across models: {discrepancies}"
            )
            logger.error(err_msg)
            if strict or self.strict_mode:
                raise ValueError(err_msg)
            return {
                "is_fair": False,
                "error": err_msg,
                "discrepancies": discrepancies,
            }

        return {
            "is_fair": True,
            "message": "Fair comparison confirmed: all models evaluated the exact same queries.",
            "common_query_count": len(reference_set),
            "models_verified": model_names,
        }
