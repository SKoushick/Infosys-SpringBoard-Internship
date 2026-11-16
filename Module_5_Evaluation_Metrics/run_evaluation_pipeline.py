"""
Master End-to-End Evaluation & Benchmarking Pipeline.
Module 5: Evaluation Metrics | Infosys Springboard 7.0

Executes:
1. Evaluation Dataset Preparation & Data Quality Validation
2. Multi-Model Comparative Benchmarking (TF-IDF, Semantic, Hybrid, Hybrid+MMR)
3. Ranking Metrics, Intra-List Diversity, Topic Entropy, Novelty, & Catalogue Coverage
4. Dedicated Performance & Latency Profiling (Warmup + repeated trials)
5. Robustness & Boundary Condition Stress Testing
6. Human Relevance Annotation Template & Status Verification
7. High-Resolution Publication-Quality Plots Generation
8. Consolidated Scorecards Serialization
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root and Module 5 src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "src"))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ted_recommender.config import RecommenderConfig
from ted_recommender.data import load_dataset
from ted_recommender.features import FeaturePipeline
from ted_recommender.recommender import (
    ContentBasedRecommender,
    SemanticRecommender,
    HybridRecommender,
)

from evaluation.evaluation_dataset import EvaluationDatasetBuilder
from evaluation.benchmark import UnifiedRecommenderBenchmark
from evaluation.performance import RecommenderPerformanceProfiler
from evaluation.robustness import RecommenderRobustnessHarness
from evaluation.human_evaluation import HumanRelevanceHarness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("module5_pipeline")


def generate_visualizations(
    scorecards: Dict[str, Any],
    df_per_query: pd.DataFrame,
    output_dir: Path,
) -> List[str]:
    """Generate high-resolution, publication-quality comparative evaluation charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    models = list(scorecards.keys())
    model_labels = [scorecards[m]["display_name"] for m in models]

    # 1. NDCG@K Comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(models))
    width = 0.25

    ndcg_5 = [scorecards[m]["ranking_accuracy"]["NDCG@5"] for m in models]
    ndcg_10 = [scorecards[m]["ranking_accuracy"]["NDCG@10"] for m in models]
    ndcg_20 = [scorecards[m]["ranking_accuracy"]["NDCG@20"] for m in models]

    rects1 = ax.bar(x - width, ndcg_5, width, label="NDCG@5", color="#4575b4")
    rects2 = ax.bar(x, ndcg_10, width, label="NDCG@10", color="#74add1")
    rects3 = ax.bar(x + width, ndcg_20, width, label="NDCG@20", color="#abd9e9")

    ax.set_ylabel("NDCG Score", fontsize=12, fontweight="bold")
    ax.set_title("Ranking Quality: NDCG Across Evaluation Cutoffs (K=5, 10, 20)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10, rotation=15, ha="right")
    ax.legend(frameon=True)
    ax.set_ylim(0, max(max(ndcg_20) * 1.25, 0.1))

    for rects in (rects1, rects2, rects3):
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    p1 = output_dir / "01_ndcg_comparison.png"
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()
    generated_plots.append(str(p1))

    # 2. Precision & Recall Comparison (K=10)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    prec_10 = [scorecards[m]["ranking_accuracy"]["Precision@10"] for m in models]
    rec_10 = [scorecards[m]["ranking_accuracy"]["Recall@10"] for m in models]

    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, prec_10, width, label="Precision@10", color="#313695")
    ax.bar(x + width/2, rec_10, width, label="Recall@10", color="#f46d43")

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Relevance Retrieval: Precision@10 vs Recall@10", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10, rotation=15, ha="right")
    ax.legend(frameon=True)
    ax.set_ylim(0, max(max(rec_10) * 1.25, 0.1))

    p2 = output_dir / "02_precision_recall_comparison.png"
    plt.tight_layout()
    plt.savefig(p2)
    plt.close()
    generated_plots.append(str(p2))

    # 3. Hit Rate & MAP Comparison (K=10)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    hr_10 = [scorecards[m]["ranking_accuracy"]["HitRate@10"] for m in models]
    map_10 = [scorecards[m]["ranking_accuracy"]["MAP@10"] for m in models]

    ax.bar(x - width/2, hr_10, width, label="HitRate@10", color="#2ca02c")
    ax.bar(x + width/2, map_10, width, label="MAP@10", color="#9467bd")

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Discovery Success: HitRate@10 and Mean Average Precision (MAP@10)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10, rotation=15, ha="right")
    ax.legend(frameon=True)

    p3 = output_dir / "03_hitrate_map_comparison.png"
    plt.tight_layout()
    plt.savefig(p3)
    plt.close()
    generated_plots.append(str(p3))

    # 4. Intra-List Diversity (ILD@10) Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ild_10 = [scorecards[m]["diversity_and_coverage"]["IntraListDiversity@10"] for m in models]

    bars = ax.bar(model_labels, ild_10, color=colors, width=0.55)
    ax.set_ylabel("Pairwise Cosine Distance", fontsize=12, fontweight="bold")
    ax.set_title("Recommendation Diversity: Intra-List Diversity (ILD@10)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    p4 = output_dir / "04_diversity_ild_comparison.png"
    plt.tight_layout()
    plt.savefig(p4)
    plt.close()
    generated_plots.append(str(p4))

    # 5. Catalogue Coverage Ratio
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    cov_ratio = [scorecards[m]["diversity_and_coverage"]["CatalogueCoverageRatio"] * 100 for m in models]
    unique_items = [scorecards[m]["diversity_and_coverage"]["UniqueItemsRecommended"] for m in models]

    bars = ax.bar(model_labels, cov_ratio, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], width=0.55)
    ax.set_ylabel("Catalogue Coverage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Catalogue Exposure: Percentage of Total 2,550 Talks Recommended", fontsize=13, fontweight="bold")

    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%\n({unique_items[i]} talks)", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    p5 = output_dir / "05_catalogue_coverage_comparison.png"
    plt.tight_layout()
    plt.savefig(p5)
    plt.close()
    generated_plots.append(str(p5))

    # 6. Inference Latency (Mean & P95)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    mean_lat = [scorecards[m]["performance_and_integrity"]["MeanLatencyMs"] for m in models]
    p95_lat = [scorecards[m]["performance_and_integrity"]["P95LatencyMs"] for m in models]

    ax.bar(x - width/2, mean_lat, width, label="Mean Latency (ms)", color="#3182bd")
    ax.bar(x + width/2, p95_lat, width, label="P95 Latency (ms)", color="#de2d26")

    ax.set_ylabel("Latency (milliseconds)", fontsize=12, fontweight="bold")
    ax.set_title("Inference Efficiency: Mean and 95th Percentile Latency per Query", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10, rotation=15, ha="right")
    ax.legend(frameon=True)

    for i in range(len(models)):
        ax.text(i - width/2, mean_lat[i] + 0.5, f"{mean_lat[i]:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, p95_lat[i] + 0.5, f"{p95_lat[i]:.1f}", ha="center", va="bottom", fontsize=8)

    p6 = output_dir / "06_latency_comparison.png"
    plt.tight_layout()
    plt.savefig(p6)
    plt.close()
    generated_plots.append(str(p6))

    # 7. Diversity vs NDCG Tradeoff Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for i, m in enumerate(models):
        ndcg_val = scorecards[m]["ranking_accuracy"]["NDCG@10"]
        ild_val = scorecards[m]["diversity_and_coverage"]["IntraListDiversity@10"]
        ax.scatter(ild_val, ndcg_val, s=200, color=colors[i], label=scorecards[m]["display_name"], edgecolors="black", zorder=4)
        ax.annotate(scorecards[m]["display_name"], (ild_val, ndcg_val), xytext=(8, 8), textcoords="offset points", fontsize=9, fontweight="bold")

    ax.set_xlabel("Intra-List Diversity (ILD@10)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ranking Quality (NDCG@10)", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy vs Diversity Trade-Off Frontier", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", frameon=True)

    p7 = output_dir / "07_accuracy_diversity_tradeoff.png"
    plt.tight_layout()
    plt.savefig(p7)
    plt.close()
    generated_plots.append(str(p7))

    return generated_plots


def main():
    logger.info("=" * 70)
    logger.info("STARTING MODULE 5: EVALUATION METRICS PIPELINE")
    logger.info("=" * 70)

    # 1. Load Configuration
    config_path = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "configs" / "evaluation_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        eval_config = json.load(f)
    logger.info(f"Loaded evaluation config from {config_path}")

    # 2. Load Dataset
    logger.info("[1/7] Loading engineered feature dataset (2550 talks)...")
    df = load_dataset(stage="features")
    logger.info(f"Loaded {len(df)} talks with columns: {list(df.columns)}")

    # 3. Build & Validate Evaluation Ground Truth
    logger.info("[2/7] Extracting & validating proxy ground-truth evaluation dataset...")
    builder = EvaluationDatasetBuilder(df)
    eval_records, data_report = builder.build()
    logger.info(f"Evaluation dataset built: {len(eval_records)} queries with valid ground truth.")
    logger.info(f"Dropped {data_report.out_of_catalogue_links_dropped} out-of-catalogue references and {data_report.self_references_dropped} self-references.")

    # Save prepared evaluation ground-truth and report
    data_out_path = PROJECT_ROOT / eval_config["data_paths"]["prepared_evaluation_dataset"]
    report_out_path = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "results" / "reports" / "evaluation_data_quality_report.json"
    builder.save(data_out_path, report_out_path)
    logger.info(f"Saved evaluation ground truth to {data_out_path}")

    # Deterministically sample evaluation queries
    sample_size = eval_config["evaluation_parameters"]["sample_query_count"]
    sample_query_ids = builder.sample_queries(sample_size=sample_size, random_seed=eval_config["random_seed"])
    logger.info(f"Sampled {len(sample_query_ids)} evaluation queries deterministically (seed={eval_config['random_seed']}).")

    # 4. Initialize Feature Pipeline and Recommendation Models
    logger.info("[3/7] Loading feature artifacts and initializing recommender models...")
    rec_config = RecommenderConfig.load()
    pipeline = FeaturePipeline(tfidf_config=rec_config.tfidf_config, field_weights=rec_config.field_weights)
    artifacts_loaded = pipeline.load_artifacts()
    assert artifacts_loaded, "Failed to load precomputed feature matrices!"

    tfidf_rec = ContentBasedRecommender(df, pipeline, rec_config)
    semantic_rec = SemanticRecommender(df, pipeline, rec_config)
    hybrid_rec = HybridRecommender(df, pipeline, rec_config)
    logger.info("All recommenders initialized successfully.")

    # 5. Run Unified Comparative Benchmark
    logger.info("[4/7] Running unified fair benchmark across all 4 recommendation approaches...")
    ground_truth_map = builder.load(data_out_path)
    benchmark_engine = UnifiedRecommenderBenchmark(
        df=df,
        ground_truth=ground_truth_map,
        feature_matrix=pipeline.embedding_matrix,
        id_to_idx=tfidf_rec.id_to_idx,
        idx_to_id=tfidf_rec.idx_to_id,
    )

    models_to_run = [
        ("tfidf", "TF-IDF Content-Based", tfidf_rec, False),
        ("semantic", "Sentence-BERT Semantic", semantic_rec, False),
        ("hybrid", "Hybrid Multi-Signal", hybrid_rec, False),
        ("hybrid_mmr", "Hybrid Multi-Signal + MMR", hybrid_rec, True),
    ]

    scorecards = {}
    per_query_dfs = []

    for model_id, display_name, model_inst, apply_mmr in models_to_run:
        scorecard, df_queries = benchmark_engine.evaluate_model(
            model=model_inst,
            model_id=model_id,
            display_name=display_name,
            query_ids=sample_query_ids,
            apply_mmr=apply_mmr,
        )
        scorecards[model_id] = scorecard.to_dict()
        per_query_dfs.append(df_queries)
        logger.info(f"[{display_name}] NDCG@10: {scorecard.ndcg_at_10:.4f} | Recall@10: {scorecard.recall_at_10:.4f} | ILD@10: {scorecard.intra_list_diversity_at_10:.4f} | Latency: {scorecard.mean_latency_ms:.2f}ms")

    # Serialize benchmark scorecards and per-query logs
    benchmarks_dir = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "results" / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    with open(benchmarks_dir / "unified_model_comparison_scorecard.json", "w", encoding="utf-8") as f:
        json.dump(scorecards, f, indent=2)

    df_all_queries = pd.concat(per_query_dfs, ignore_index=True)
    df_all_queries.to_csv(benchmarks_dir / "per_query_evaluation_results.csv", index=False)

    # 6. Performance Profiling
    logger.info("[5/7] Conducting performance profiling (offline loading vs online latency)...")
    profiler = RecommenderPerformanceProfiler(df)
    load_profile = profiler.benchmark_offline_loading(FeaturePipeline, rec_config)
    
    latency_profiles = {}
    for model_id, display_name, model_inst, apply_mmr in models_to_run:
        lp = profiler.benchmark_query_latency(
            recommender=model_inst,
            model_name=display_name,
            test_talk_ids=sample_query_ids[:10],
            num_warmup=3,
            num_iterations=10,
            top_k=10,
            apply_mmr=apply_mmr,
        )
        latency_profiles[model_id] = lp

    perf_report = {
        "offline_artifact_loading": load_profile,
        "online_query_latency_profiles": latency_profiles,
    }
    perf_dir = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "results" / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    with open(perf_dir / "performance_report.json", "w", encoding="utf-8") as f:
        json.dump(perf_report, f, indent=2)
    logger.info("Performance report saved.")

    # 7. Robustness and Stress Testing
    logger.info("[6/7] Running robustness and edge-case stress harness...")
    robustness_harness = RecommenderRobustnessHarness((tfidf_rec, semantic_rec, hybrid_rec), df)
    robustness_report = robustness_harness.run_all_checks()
    logger.info(f"Robustness results: {robustness_report['passed_checks']}/{robustness_report['total_checks']} checks passed.")

    reports_dir = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "robustness_report.json", "w", encoding="utf-8") as f:
        json.dump(robustness_report, f, indent=2)

    # 8. Human Relevance Annotation Setup
    logger.info("[7/7] Generating human relevance annotation template and status...")
    human_harness = HumanRelevanceHarness(df)
    template_path = reports_dir / "human_relevance_annotation_template.csv"
    _ = human_harness.generate_annotation_template(
        recommender_models={
            "TF-IDF": tfidf_rec,
            "Semantic": semantic_rec,
            "Hybrid": hybrid_rec,
        },
        query_talk_ids=sample_query_ids[:10],
        top_k=5,
        output_csv_path=template_path,
    )
    human_status = human_harness.evaluate_annotations(template_path)
    with open(reports_dir / "human_evaluation_status.json", "w", encoding="utf-8") as f:
        json.dump(human_status, f, indent=2)

    # 9. Generate Visualizations
    plots_dir = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "results" / "plots"
    generated_plots = generate_visualizations(scorecards, df_all_queries, plots_dir)
    logger.info(f"Generated {len(generated_plots)} publication-quality evaluation plots in {plots_dir}")

    # Build Summary Scorecard Table
    scorecard_rows = []
    for m in models_to_run:
        m_id = m[0]
        s = scorecards[m_id]
        scorecard_rows.append({
            "Model": s["display_name"],
            "NDCG@10": s["ranking_accuracy"]["NDCG@10"],
            "MAP@10": s["ranking_accuracy"]["MAP@10"],
            "Precision@10": s["ranking_accuracy"]["Precision@10"],
            "Recall@10": s["ranking_accuracy"]["Recall@10"],
            "HitRate@10": s["ranking_accuracy"]["HitRate@10"],
            "ILD@10 (Diversity)": s["diversity_and_coverage"]["IntraListDiversity@10"],
            "Coverage (%)": round(s["diversity_and_coverage"]["CatalogueCoverageRatio"] * 100, 2),
            "Novelty (Bits)": s["diversity_and_coverage"]["RecommendationNoveltyBits"],
            "Latency (ms)": s["performance_and_integrity"]["MeanLatencyMs"],
        })

    df_scorecard = pd.DataFrame(scorecard_rows)
    df_scorecard.to_csv(benchmarks_dir / "benchmark_summary_table.csv", index=False)

    print("\n" + "=" * 80)
    print("MODULE 5: CONSOLIDATED BENCHMARK SCORECARD")
    print("=" * 80)
    print(df_scorecard.to_string(index=False))
    print("=" * 80)
    logger.info("MODULE 5 EVALUATION PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
