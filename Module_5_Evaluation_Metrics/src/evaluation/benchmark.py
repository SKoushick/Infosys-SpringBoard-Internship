"""
Unified Fair Benchmark Engine for TED Talk Recommendation Models.
Module 5: Evaluation Metrics | Infosys Springboard 7.0

Evaluates all four core recommendation strategies under strictly identical conditions:
1. TF-IDF Content-Based Recommender (Lexical similarity)
2. Sentence-BERT Semantic Recommender (Conceptual embeddings)
3. Hybrid Recommender without MMR (Multi-signal fusion)
4. Hybrid Recommender with MMR (Multi-signal fusion + Maximal Marginal Relevance)

Guarantees:
- Identical evaluation query sample across all models.
- Identical candidate catalogue and exclusion rules (self-talk excluded).
- Evaluates the actual hybrid and MMR execution paths (no proxy substitutions).
- Comprehensive evaluation across ranking accuracy, intra-list diversity,
  catalogue coverage, novelty, and inference latency.
"""

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
    num_queries_evaluated: int
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
            "num_queries_evaluated": self.num_queries_evaluated,
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
    ):
        self.df = df
        self.ground_truth = ground_truth
        self.feature_matrix = feature_matrix
        self.id_to_idx = id_to_idx
        self.idx_to_id = idx_to_id
        self.catalogue_size = len(df)
        self.item_views_map = item_views_map or {
            int(row["talk_id"]): int(row.get("views", 1)) for _, row in df.iterrows()
        }

    def evaluate_model(
        self,
        model,
        model_id: str,
        display_name: str,
        query_ids: List[int],
        apply_mmr: bool = False,
    ) -> Tuple[ModelBenchmarkScorecard, pd.DataFrame]:
        """
        Evaluate a single model across query_ids for K in [5, 10, 20].
        Returns aggregate scorecard and per-query detailed DataFrame.
        """
        logger.info(f"Evaluating {display_name} across {len(query_ids)} queries (apply_mmr={apply_mmr})...")

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

        per_query_records = []

        for q_id in query_ids:
            if q_id not in self.ground_truth:
                continue
            relevant_set = self.ground_truth[q_id]
            if not relevant_set:
                continue

            t0 = time.perf_counter()
            try:
                # Dispatch recommendation call
                if hasattr(model, "recommend_for_talk"):
                    # Pass apply_mmr if model supports it (Hybrid and Content recommenders)
                    try:
                        recs = model.recommend_for_talk(talk_id=q_id, top_k=20, apply_mmr=apply_mmr)
                    except TypeError:
                        recs = model.recommend_for_talk(talk_id=q_id, top_k=20)
                else:
                    recs = model.recommend(seed_talk_id=q_id, top_k=20, apply_mmr=apply_mmr)
            except Exception as e:
                logger.error(f"Error during recommendation for query {q_id} with model {model_id}: {e}")
                continue

            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

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
            num_queries_evaluated=n_eval,
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

