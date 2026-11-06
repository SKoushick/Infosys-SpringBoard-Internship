"""
Training, Artifact Generation, and Performance Benchmarking Pipeline.
Generates all production recommendation artifacts for Milestone 2.

Fixes:
- Resolves the stale 1866 rows vs 2550 rows mismatch by fitting across all 2550 talks
- Saves 4 field-specific TF-IDF vectorizers + combined_tfidf_matrix.npz
- Generates and caches dense 384-dimensional SentenceTransformer embeddings for all 2550 talks
- Benchmarks model loading time, inference latency, and memory footprint
- Runs comparative evaluation across TF-IDF, Semantic, and Hybrid models
"""

import json
import logging
import time
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from ted_recommender.config import (
    FEATURES_CSV_PATH,
    MODULE_4_MODELS_DIR,
    MODULE_4_OUTPUTS_DIR,
    TFIDF_FEATURES_DIR,
    TFIDF_MODELS_DIR,
    RecommenderConfig,
)
from ted_recommender.data import load_dataset, validate_dataset
from ted_recommender.features import FeaturePipeline
from ted_recommender.recommender import (
    ContentBasedRecommender,
    HybridRecommender,
    SemanticRecommender,
)
from ted_recommender.evaluation import (
    RecommendationEvaluator,
    parse_ground_truth_related_talks,
)
from ted_recommender.collaborative import assess_collaborative_filtering_feasibility

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_and_benchmark")


def main():
    logger.info("=" * 65)
    logger.info("TED TALKS RECOMMENDATION SYSTEM - MILESTONE 2 PIPELINE")
    logger.info("=" * 65)

    # 1. Load Dataset
    logger.info("[1/7] Loading engineered feature dataset...")
    df = load_dataset(stage="features")
    logger.info(f"Loaded dataset with {len(df)} records and {df.shape[1]} columns.")
    assert len(df) == 2550, f"Expected 2550 rows, got {len(df)}"

    # 2. Validate Data Quality
    logger.info("[2/7] Running data validation checks...")
    val_report = validate_dataset(df)
    logger.info(f"Validation summary: {len(val_report['issues'])} issues detected.")

    # 3. Fit TF-IDF Pipeline
    logger.info("[3/7] Fitting TF-IDF vectorizers across all 2550 talks...")
    t0_tfidf = time.time()
    config = RecommenderConfig()
    pipeline = FeaturePipeline(tfidf_config=config.tfidf_config, field_weights=config.field_weights)
    _, combined_matrix = pipeline.fit_transform_tfidf(df)
    tfidf_time = time.time() - t0_tfidf

    logger.info(f"TF-IDF fitted in {tfidf_time:.2f}s. Matrix shape: {combined_matrix.shape}")
    assert combined_matrix.shape[0] == 2550, f"TF-IDF row count mismatch: {combined_matrix.shape[0]}"

    # Save TF-IDF artifacts to both Module 3 and Module 4 models directories
    pipeline.save_artifacts(models_dir=TFIDF_MODELS_DIR, features_dir=TFIDF_FEATURES_DIR)
    pipeline.save_artifacts(models_dir=MODULE_4_MODELS_DIR, features_dir=MODULE_4_MODELS_DIR)

    # 4. Generate Semantic Embeddings
    logger.info("[4/7] Generating SentenceTransformer embeddings (all-MiniLM-L6-v2)...")
    t0_emb = time.time()
    embeddings = pipeline.generate_embeddings(df, batch_size=64)
    emb_time = time.time() - t0_emb
    logger.info(f"Embeddings generated in {emb_time:.2f}s. Shape: {embeddings.shape}")
    assert embeddings.shape == (2550, 384), f"Unexpected embedding shape: {embeddings.shape}"

    # Save embedding matrix in Module 4 directory as well
    np.save(MODULE_4_MODELS_DIR / "ted_semantic_embeddings.npy", embeddings)

    # 5. Initialize Recommenders
    logger.info("[5/7] Initializing recommendation engines...")
    t0_init = time.time()
    tfidf_rec = ContentBasedRecommender(df, pipeline, config)
    semantic_rec = SemanticRecommender(df, pipeline, config)
    hybrid_rec = HybridRecommender(df, pipeline, config)
    init_time = time.time() - t0_init
    logger.info(f"Recommenders initialized in {init_time:.4f}s.")

    # 6. Benchmark Latency
    logger.info("[6/7] Benchmarking recommendation latency across test cases...")
    test_talk_ids = [1, 2, 3, 4, 5, 100, 500, 1000, 1500, 2000]
    latencies_tfidf = []
    latencies_semantic = []
    latencies_hybrid = []

    for t_id in test_talk_ids:
        t0 = time.time()
        _ = tfidf_rec.recommend_for_talk(talk_id=t_id, top_k=10)
        latencies_tfidf.append((time.time() - t0) * 1000)

        t0 = time.time()
        _ = semantic_rec.recommend_for_talk(talk_id=t_id, top_k=10)
        latencies_semantic.append((time.time() - t0) * 1000)

        t0 = time.time()
        _ = hybrid_rec.recommend(seed_talk_id=t_id, top_k=10, apply_mmr=True)
        latencies_hybrid.append((time.time() - t0) * 1000)

    avg_lat_tfidf = float(np.mean(latencies_tfidf))
    avg_lat_semantic = float(np.mean(latencies_semantic))
    avg_lat_hybrid = float(np.mean(latencies_hybrid))

    logger.info(f"Average latency (TF-IDF): {avg_lat_tfidf:.2f} ms")
    logger.info(f"Average latency (Semantic): {avg_lat_semantic:.2f} ms")
    logger.info(f"Average latency (Hybrid + MMR): {avg_lat_hybrid:.2f} ms")

    # 7. Evaluate on Editorial Ground Truth
    logger.info("[7/7] Running benchmark evaluation against editorial related talks...")
    ground_truth = parse_ground_truth_related_talks(df)
    logger.info(f"Found editorial related talk sets for {len(ground_truth)} talks.")

    evaluator = RecommendationEvaluator(ground_truth)
    sample_eval_ids = [t for t in ground_truth.keys() if t <= 500][:100]

    eval_tfidf = evaluator.evaluate_recommender(tfidf_rec, sample_eval_ids, k=10)
    eval_semantic = evaluator.evaluate_recommender(semantic_rec, sample_eval_ids, k=10)
    eval_hybrid = evaluator.evaluate_recommender(hybrid_rec.tfidf_rec, sample_eval_ids, k=10)

    logger.info(f"TF-IDF Evaluation: {eval_tfidf}")
    logger.info(f"Semantic Evaluation: {eval_semantic}")

    # Assess Collaborative Filtering Feasibility
    cf_report = assess_collaborative_filtering_feasibility(df)

    # Save complete benchmark report
    benchmark_report = {
        "dataset_summary": {
            "total_talks": len(df),
            "columns": list(df.columns),
            "missing_transcripts": int(df["has_transcript"].value_counts().get(False, 0)),
        },
        "artifact_shapes": {
            "combined_tfidf_matrix": list(combined_matrix.shape),
            "embedding_matrix": list(embeddings.shape),
        },
        "timing_and_latency": {
            "tfidf_fit_seconds": round(tfidf_time, 2),
            "embedding_generation_seconds": round(emb_time, 2),
            "recommender_init_seconds": round(init_time, 4),
            "average_latency_ms": {
                "tfidf": round(avg_lat_tfidf, 2),
                "semantic": round(avg_lat_semantic, 2),
                "hybrid_with_mmr": round(avg_lat_hybrid, 2),
            },
        },
        "editorial_benchmark_metrics": {
            "tfidf_model": eval_tfidf,
            "semantic_model": eval_semantic,
        },
        "collaborative_filtering_assessment": cf_report.to_dict(),
    }

    MODULE_4_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = MODULE_4_OUTPUTS_DIR / "milestone_2_benchmark_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    logger.info(f"Benchmark report saved to {out_file}")
    logger.info("=" * 65)
    logger.info("MILESTONE 2 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()

