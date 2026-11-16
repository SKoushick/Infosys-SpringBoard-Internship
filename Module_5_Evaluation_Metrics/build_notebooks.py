"""
Builds standard Jupyter Notebooks (schema v4) for Module 5.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.7"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().split("\n",)
    }


def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().split("\n")
    }


# Notebook 1: Evaluation Dataset Preparation
nb1_cells = [
    make_markdown_cell("""# Module 5 — Notebook 01: Evaluation Dataset Preparation & Quality Validation
### Personalized TED Talks Recommendation System | Infosys Springboard 7.0

This notebook demonstrates the extraction, validation, cleaning, and preparation of TED editorial 'related_talks' as a defensible proxy-relevance benchmark dataset.

**Key Operations:**
1. Parse raw stringified JSON dictionary arrays into integer talk IDs.
2. Filter out-of-catalogue references and self-references.
3. Validate catalogue membership against all 2,550 TED talks.
4. Stratified sampling across talks with and without transcripts.
5. Export clean evaluation ground truth and data-quality audit report.
"""),
    make_code_cell("""import sys
from pathlib import Path

# Set up paths relative to project root
PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "src"))

import pandas as pd
import numpy as np

from ted_recommender.data import load_dataset
from evaluation.evaluation_dataset import EvaluationDatasetBuilder

print("Modules imported successfully.")
"""),
    make_code_cell("""# 1. Load engineered dataset
df = load_dataset(stage="features")
print(f"Loaded {len(df)} talks with {df.shape[1]} columns.")
df[['talk_id', 'title', 'main_speaker', 'related_talks', 'has_transcript']].head(5)
"""),
    make_code_cell("""# 2. Build and Validate Evaluation Ground Truth
builder = EvaluationDatasetBuilder(df)
records, report = builder.build()

print("Evaluation Ground Truth Summary:")
for k, v in report.to_dict().items():
    print(f"  {k}: {v}")
"""),
    make_code_cell("""# 3. Deterministic Stratified Query Sampling
sample_ids = builder.sample_queries(sample_size=100, random_seed=42)
print(f"Sampled {len(sample_ids)} evaluation queries deterministically.")
print(f"First 10 sample query IDs: {sample_ids[:10]}")
"""),
    make_code_cell("""# 4. Save prepared evaluation artifacts
out_data_path = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "data" / "processed" / "evaluation_ground_truth.json"
out_report_path = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "results" / "reports" / "evaluation_data_quality_report.json"

builder.save(out_data_path, out_report_path)
print(f"Evaluation dataset persisted successfully to {out_data_path.name}")
""")
]

# Notebook 2: Ranking Metrics Implementation
nb2_cells = [
    make_markdown_cell("""# Module 5 — Notebook 02: Ranking Metrics Implementation & Verification
### Personalized TED Talks Recommendation System | Infosys Springboard 7.0

This notebook demonstrates the mathematical formulas, boundary condition handling, and hand-calculated verification for recommendation ranking metrics:
- **Precision@K** & **Recall@K**
- **F1@K**
- **Discounted Cumulative Gain (DCG@K)** & **Normalized DCG (NDCG@K)**
- **Average Precision (AP@K)** & **Mean Average Precision (MAP@K)**
- **HitRate@K** & **Mean Reciprocal Rank (MRR@K)**
"""),
    make_code_cell("""import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "src"))

import numpy as np
from evaluation.ranking_metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    dcg_at_k,
    ndcg_at_k,
    average_precision_at_k,
    hit_rate_at_k,
    reciprocal_rank_at_k,
)

print("Ranking metrics imported successfully.")
"""),
    make_code_cell("""# Hand-Calculated Verification Example
# Recommended items: [10, 20, 30, 40, 50]
# Known relevant ground truth: {20, 40, 99} (3 total relevant items)

recommended = [10, 20, 30, 40, 50]
relevant = {20, 40, 99}

print("--- Metric Scores for Hand-Calculated Example ---")
print(f"Precision@5: {precision_at_k(recommended, relevant, k=5):.4f} (Expected: 2/5 = 0.4000)")
print(f"Recall@5:    {recall_at_k(recommended, relevant, k=5):.4f} (Expected: 2/3 = 0.6667)")
print(f"F1@5:        {f1_at_k(recommended, relevant, k=5):.4f} (Expected: 0.5000)")
print(f"DCG@5:       {dcg_at_k(recommended, relevant, k=5):.4f}")
print(f"NDCG@5:      {ndcg_at_k(recommended, relevant, k=5):.4f}")
print(f"AP@5:        {average_precision_at_k(recommended, relevant, k=5):.4f}")
print(f"HitRate@5:   {hit_rate_at_k(recommended, relevant, k=5):.4f}")
print(f"MRR@5:       {reciprocal_rank_at_k(recommended, relevant, k=5):.4f} (Hit at rank 2 -> 1/2 = 0.5)")
"""),
    make_code_cell("""# Graded Relevance Demonstration (0 to 3 scale)
# Item 10: Grade 3 (Highly Relevant)
# Item 20: Grade 1 (Slightly Relevant)
# Item 30: Grade 0 (Not Relevant)

graded_relevance = {10: 3.0, 20: 1.0, 30: 0.0}
print(f"Graded NDCG@3 (Perfect Order): {ndcg_at_k([10, 20, 30], graded_relevance, k=3):.4f} (Expected: 1.0000)")
print(f"Graded NDCG@3 (Suboptimal Order [20, 10, 30]): {ndcg_at_k([20, 10, 30], graded_relevance, k=3):.4f}")
""")
]

# Notebook 3: Model Performance Evaluation
nb3_cells = [
    make_markdown_cell("""# Module 5 — Notebook 03: Model Performance & Comparative Benchmarking
### Personalized TED Talks Recommendation System | Infosys Springboard 7.0

This notebook executes the unified fair comparative benchmark across all 4 recommendation approaches:
1. **TF-IDF Content-Based Recommender**
2. **Sentence-BERT Semantic Recommender**
3. **Hybrid Recommender without MMR**
4. **Hybrid Recommender with MMR**
"""),
    make_code_cell("""import sys
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "src"))

benchmarks_dir = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "results" / "benchmarks"
scorecard_csv = benchmarks_dir / "benchmark_summary_table.csv"

df_scorecard = pd.read_csv(scorecard_csv)
df_scorecard
"""),
    make_code_cell("""# Inspect Detailed Scorecards JSON
with open(benchmarks_dir / "unified_model_comparison_scorecard.json", "r") as f:
    scorecard_data = json.load(f)

for m_id, details in scorecard_data.items():
    print(f"=== {details['display_name']} ===")
    print("Ranking Accuracy:")
    for k, v in details["ranking_accuracy"].items():
        print(f"  {k}: {v}")
    print("Diversity & Exposure:")
    for k, v in details["diversity_and_coverage"].items():
        print(f"  {k}: {v}")
    print()
""")
]

# Notebook 4: Diversity, Coverage, and Robustness
nb4_cells = [
    make_markdown_cell("""# Module 5 — Notebook 04: Diversity, Catalogue Coverage, and Robustness
### Personalized TED Talks Recommendation System | Infosys Springboard 7.0

This notebook inspects:
1. **Intra-List Diversity (ILD@10)** and the impact of MMR reranking.
2. **Catalogue Coverage** and **Recommendation Novelty**.
3. **System Robustness**: 12 boundary condition and stress checks.
4. **Offline Load vs Online Inference Latency**.
"""),
    make_code_cell("""import sys
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path.cwd().parent.parent

# Load Robustness Report
robust_path = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "results" / "reports" / "robustness_report.json"
with open(robust_path, "r") as f:
    robust_data = json.load(f)

print(f"Robustness Checks: {robust_data['passed_checks']}/{robust_data['total_checks']} Passed.\\n")
for tc in robust_data["test_cases"]:
    print(f"[{tc['status']}] {tc['case_name']}: {tc['details']}")
"""),
    make_code_cell("""# Load Performance Profiling Report
perf_path = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "results" / "performance" / "performance_report.json"
with open(perf_path, "r") as f:
    perf_data = json.load(f)

print("Offline Artifact Loading Profile:")
for k, v in perf_data["offline_artifact_loading"].items():
    print(f"  {k}: {v}")

print("\\nOnline Per-Query Latency Profiles (ms):")
for m_id, p in perf_data["online_query_latency_profiles"].items():
    print(f"  {p['model_name']}: Mean={p['latency_ms']['mean']}ms, Median={p['latency_ms']['median']}ms, P95={p['latency_ms']['p95']}ms, QPS={p['throughput_queries_per_sec']}")
""")
]

# Notebook 5: Final Evaluation Analysis
nb5_cells = [
    make_markdown_cell("""# Module 5 — Notebook 05: Final Evaluation Analysis & Visualizations
### Personalized TED Talks Recommendation System | Infosys Springboard 7.0

This notebook visualizes the comprehensive evaluation results:
- Ranking quality comparison (NDCG, MAP, Precision, Recall, HitRate).
- Intra-List Diversity and Catalogue Coverage.
- Accuracy vs. Diversity trade-off frontier.
- Human relevance evaluation workflow status.
"""),
    make_code_cell("""from pathlib import Path
from IPython.display import Image, display

PROJECT_ROOT = Path.cwd().parent.parent
plots_dir = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "results" / "plots"

plot_files = sorted(list(plots_dir.glob("*.png")))
print(f"Found {len(plot_files)} generated evaluation plots:")
for p in plot_files:
    print(f"  - {p.name}")
"""),
    make_code_cell("""# Display Accuracy vs Diversity Trade-off Plot
display(Image(filename=str(plots_dir / "07_accuracy_diversity_tradeoff.png")))
"""),
    make_code_cell("""# Display NDCG Comparison Plot
display(Image(filename=str(plots_dir / "01_ndcg_comparison.png")))
"""),
    make_code_cell("""# Display Intra-List Diversity Comparison Plot
display(Image(filename=str(plots_dir / "04_diversity_ild_comparison.png")))
"""),
    make_code_cell("""# Display Latency Comparison Plot
display(Image(filename=str(plots_dir / "06_latency_comparison.png")))
"""),
    make_code_cell("""# Human Relevance Evaluation Status
reports_dir = PROJECT_ROOT / "Module_5_Evaluation_Metrics" / "results" / "reports"
with open(reports_dir / "human_evaluation_status.json", "r") as f:
    human_status = json.load(f)

print("Human Evaluation Status:")
for k, v in human_status.items():
    if k != "scale":
        print(f"  {k}: {v}")
""")
]

# Save all notebooks
notebooks = {
    "01_Evaluation_Dataset_Preparation.ipynb": nb1_cells,
    "02_Ranking_Metrics_Implementation.ipynb": nb2_cells,
    "03_Model_Performance_Evaluation.ipynb": nb3_cells,
    "04_Diversity_Coverage_and_Robustness.ipynb": nb4_cells,
    "05_Final_Evaluation_Analysis.ipynb": nb5_cells,
}

for name, cells in notebooks.items():
    p = NOTEBOOKS_DIR / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created notebook: {p.name}")

print("All 5 Module 5 notebooks created successfully.")

