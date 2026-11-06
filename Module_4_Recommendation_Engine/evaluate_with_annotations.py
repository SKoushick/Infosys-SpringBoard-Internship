"""
Evaluation Harness for Human Relevance Annotations.
Module 4 / Module 5 Evaluation Framework.

Reads graded or binary relevance labels from an annotation CSV,
evaluates Content-Based (TF-IDF), Semantic, and Hybrid recommenders,
and outputs ranking metric scorecards (NDCG@K, MAP@K, Precision@K, Recall@K).
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from ted_recommender.config import RecommenderConfig
from ted_recommender.data import load_dataset
from ted_recommender.features import FeaturePipeline
from ted_recommender.recommender import (
    ContentBasedRecommender,
    SemanticRecommender,
    HybridRecommender,
)
from ted_recommender.evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    average_precision_at_k,
    hit_rate_at_k,
)


def run_evaluation(annotation_file: Path, k: int = 5):
    if not annotation_file.exists():
        print(f"[ERROR] Annotation template not found at {annotation_file}")
        return

    print("=" * 65)
    print("EVALUATION HARNESS: HUMAN RELEVANCE JUDGMENTS")
    print("=" * 65)

    annotations = pd.read_csv(annotation_file)
    print(f"Loaded {len(annotations)} judgments across {annotations['query_or_seed_talk_id'].nunique()} unique queries.")

    # Group relevant talk IDs by query (binary relevance: grade >= 2)
    relevance_map = {}
    for q_id, group in annotations.groupby("query_or_seed_talk_id"):
        rel_ids = set(group[group["relevance_grade"] >= 2]["candidate_talk_id"])
        relevance_map[int(q_id)] = rel_ids

    # Load recommendation pipeline
    df = load_dataset(stage="features")
    config = RecommenderConfig()
    pipeline = FeaturePipeline(tfidf_config=config.tfidf_config, field_weights=config.field_weights)
    pipeline.load_artifacts()

    tfidf_rec = ContentBasedRecommender(df, pipeline, config)
    semantic_rec = SemanticRecommender(df, pipeline, config)
    hybrid_rec = HybridRecommender(df, pipeline, config)

    models = {
        "TF-IDF Content-Based": tfidf_rec,
        "Sentence-BERT Semantic": semantic_rec,
        "Hybrid Multi-Signal": hybrid_rec,
    }

    scorecard = []

    for name, model in models.items():
        precisions, recalls, ndcgs, maps, hit_rates = [], [], [], [], []

        for q_id, rel_set in relevance_map.items():
            if not rel_set:
                continue

            if name == "Hybrid Multi-Signal":
                recs = model.recommend(seed_talk_id=q_id, top_k=k, apply_mmr=True)
            else:
                recs = model.recommend_for_talk(talk_id=q_id, top_k=k)

            rec_ids = [r.talk_id for r in recs]
            precisions.append(precision_at_k(rec_ids, rel_set, k))
            recalls.append(recall_at_k(rec_ids, rel_set, k))
            ndcgs.append(ndcg_at_k(rec_ids, rel_set, k))
            maps.append(average_precision_at_k(rec_ids, rel_set, k))
            hit_rates.append(hit_rate_at_k(rec_ids, rel_set, k))

        scorecard.append({
            "Model": name,
            f"Precision@{k}": round(float(np.mean(precisions)), 4),
            f"Recall@{k}": round(float(np.mean(recalls)), 4),
            f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
            f"MAP@{k}": round(float(np.mean(maps)), 4),
            f"HitRate@{k}": round(float(np.mean(hit_rates)), 4),
        })

    scorecard_df = pd.DataFrame(scorecard)
    print("\n--- BENCHMARK RESULTS ---")
    print(scorecard_df.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    template_path = (
        PROJECT_ROOT
        / "Module_4_Recommendation_Engine"
        / "outputs"
        / "relevance_annotation_template.csv"
    )
    run_evaluation(template_path, k=5)

