# Personalized TED Talks Recommendation System Using Machine Learning
### Infosys Springboard Internship 7.0 — Milestone 2 Deliverable

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Testing](https://img.shields.io/badge/pytest-16%20passed-success.svg)](tests/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Framework](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Embeddings](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

##  Project Overview
This repository contains the end-to-end engineering implementation of a **Personalized TED Talks Recommendation System**. Using natural language processing (NLP), sparse vectorisation, and pretrained transformer embeddings, the engine recommends highly relevant, diversified, and explainable talks from a catalog of 2,550 TED presentations.

### Key Capabilities
- **Content-Based TF-IDF Recommender:** Field-weighted term frequency / inverse document frequency matching (Title, Description, Tags, Transcript) spanning **44,613 features**.
- **Semantic Recommender:** Deep conceptual matching powered by **`sentence-transformers/all-MiniLM-L6-v2`** (384-dimensional dense vectors).
- **Multi-Signal Hybrid Recommender:** Configurable score fusion between lexical keywords, latent semantics, and topic interest boosts.
- **Session-Based Personalisation:** Aggregates multi-liked talks into a dynamic user preference profile while strictly excluding disliked or previously viewed talks.
- **Maximal Marginal Relevance (MMR):** Algorithmic re-ranking to balance topical relevance and content diversity ($\lambda = 0.70$).
- **Explainable Recommendations:** Detailed natural language justifications for every result (shared themes, speaker affinity, similarity scores).
- **Cold-Start Resilience:** Graceful fallbacks for talks lacking transcripts and users with zero interaction history.

---

##  Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd Ted_Talk_Recommendation

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Windows PowerShell
# source venv/bin/activate    # On Linux/macOS

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Generate Artifacts and Run Benchmarks
To regenerate all TF-IDF matrices, cache dense semantic embeddings, and measure system latency:
```bash
python Module_4_Recommendation_Engine/train_and_benchmark.py
```

### 3. Run Automated Tests
```bash
pytest -v
```
*(All 39 unit and integration test suites pass in ~69s)*

### 4. Execute Module 5 Unified Evaluation & Benchmarking Pipeline
To run the complete evaluation, performance profiling, robustness checks, and generate all high-res plots:
```bash
python Module_5_Evaluation_Metrics/run_evaluation_pipeline.py
```

### 5. Interactive Notebook Demonstration
Launch Jupyter and open the demonstration notebooks:
```bash
# Module 4 Engine Demonstration
jupyter notebook Module_4_Recommendation_Engine/01_Recommendation_Engine_Demonstration.ipynb

# Module 5 Evaluation Analysis
jupyter notebook Module_5_Evaluation_Metrics/notebooks/05_Final_Evaluation_Analysis.ipynb
```

---

##  Repository Architecture

```
Ted_Talk_Recommendation/
├── .gitignore                                 # Git ignore rules for virtualenvs, caches, artifacts
├── pytest.ini                                 # Pytest testpath and pythonpath configuration
├── README.md                                  # Project overview and reproduction guide
├── requirements.txt                           # Verified Python dependencies
├── config/
│   └── recommender_config.json                # Centralized hyperparameters & weights
├── docs/
│   ├── MILESTONE_2_TECHNICAL_DOCUMENTATION.md # 33-section comprehensive technical report
│   └── MODULE_5_EVALUATION_REPORT.md          # 19-section Module 5 evaluation report
├── src/
│   └── ted_recommender/                       # Production Python Package
│       ├── __init__.py
│       ├── config.py                          # Dynamic root resolution & config dataclasses
│       ├── data.py                            # Dataset loaders, schema validation, text cleaners
│       ├── features.py                        # TF-IDF pipeline & Sentence-BERT embedding generator
│       ├── recommender.py                     # Content-based, Semantic, Hybrid, MMR, & Explanations
│       ├── collaborative.py                   # Feasibility audit & future interface contract
│       └── evaluation.py                      # Ranking metrics (P@K, R@K, NDCG@K, MAP@K, ILD)
├── tests/                                     # Core Automated Test Suite (20/20 Passed)
│   ├── test_data.py                           # Schema, row alignment, fingerprint, and cleaning tests
│   ├── test_vectorization.py                  # Matrix shapes, sparsity, and transformation tests
│   ├── test_recommender.py                    # Top-k, self-exclusion, MMR, and edge case tests
│   └── test_evaluation.py                     # Metric mathematical correctness & audit checks
├── Module_1_Data_Collection/                  # Module 1 Scripts & Data
│   └── Scraped/Data/                          # Raw, Processed, and External CSVs (2550 talks)
├── Module_2_Data_Exploration_Preprocessing/   # Module 2 EDA Notebooks & Visualizations
├── Module_3_Feature_Engineering_Vectorization/# Module 3 Vectorization Notebooks & Feature Matrices
├── Module_4_Recommendation_Engine/            # Module 4 Implementation & Benchmark
│   ├── 01_Recommendation_Engine_Demonstration.ipynb
│   ├── train_and_benchmark.py
│   └── outputs/milestone_2_benchmark_report.json
└── Module_5_Evaluation_Metrics/               # Module 5 Complete Evaluation Framework
    ├── configs/evaluation_config.json         # Evaluation parameters, K-cutoffs, and paths
    ├── data/processed/                        # 2,544 sanitized evaluation ground-truth queries
    ├── notebooks/                             # 5 standard v4 Jupyter evaluation notebooks
    ├── results/
    │   ├── benchmarks/                        # Multi-model comparison scorecards & query logs
    │   ├── performance/                       # Offline load vs online latency profiles
    │   ├── plots/                             # 7 publication-quality evaluation comparison plots
    │   └── reports/                           # Data quality, robustness, & human evaluation reports
    ├── src/evaluation/                        # Modular evaluation package
    │   ├── ranking_metrics.py                 # Precision, Recall, F1, NDCG, MAP, HitRate, MRR
    │   ├── diversity_metrics.py               # ILD, Topic Entropy, Gini-Simpson, Integrity
    │   ├── coverage_metrics.py                # Catalogue coverage, Novelty bits, Gini index
    │   ├── evaluation_dataset.py              # Parsing, membership validation, & sampling
    │   ├── benchmark.py                       # Unified fair benchmarking engine
    │   ├── performance.py                     # Cold-start vs latency profiling
    │   ├── robustness.py                      # 12-check boundary stress suite (12/12 PASS)
    │   └── human_evaluation.py               # Graded relevance framework & template
    └── tests/                                 # Module 5 Test Suite (19/19 Passed)
```

---

## Measured Benchmark Performance (Module 5 Unified Scorecard)

Evaluated under strict identical conditions across 100 stratified ground-truth queries ($K=10$):

| Recommender Model | NDCG@10 | MAP@10 | Recall@10 | HitRate@10 | ILD@10 (Diversity) | Catalog Coverage | Novelty (bits) | Mean Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF Content-Based** | 0.0063 | 0.0026 | 0.0077 | 0.0400 | 0.6851 | 51.88% (1,323 talks) | 11.8556 | 26.59 ms |
| **Sentence-BERT Semantic** | 0.0053 | 0.0018 | 0.0073 | 0.0400 | 0.5597 | 48.98% (1,249 talks) | 11.8498 | 19.49 ms |
| **Hybrid Multi-Signal (No MMR)** | **0.0076** | **0.0035** | 0.0073 | 0.0400 | 0.5763 | 50.59% (1,290 talks) | 11.8411 | 40.05 ms |
| **Hybrid + MMR ($\lambda=0.7$)** | 0.0074 | 0.0033 | 0.0073 | 0.0400 | **0.6953 (+20.6%)** | **54.16% (1,381 talks)** | 11.8487 | 54.38 ms |

*Measurements taken across the full 2,550-talk catalogue running locally on Intel CPU.*

---

## Collaborative Filtering Decision
An empirical audit of the dataset confirmed that the TED archive supplies only aggregate talk statistics (`views`, `comments`, `ratings` emotion counts) and **contains zero individual user identifiers or user-item interaction logs**.

In adherence to scientific integrity:
- **No fake users or synthetic interactions were fabricated.**
- Collaborative filtering is formally reported as **UNSUPPORTED BY AVAILABLE EVIDENCE**.
- An extensible `CollaborativeSignalProvider` interface is implemented in `src/ted_recommender/collaborative.py` to seamlessly blend user signals when telemetry logs become available.

---

## Seven-Module Status Summary

| Module | Status | Verification Evidence |
|:---|:---:|:---|
| **Module 1: Data Collection** | **COMPLETE (PASS)** | 2,550 records loaded from verified TED public archive |
| **Module 2: Data Exploration & Preprocessing** | **COMPLETE (PASS)** | Cleaned text, transcription cue removal, missing transcript fallback |
| **Module 3: Feature Engineering & Vectorization** | **COMPLETE (PASS)** | 4 TF-IDF vectorizers fitted; 2550×44,613 matrix; 2550×384 embeddings |
| **Module 4: Develop Recommendation Engine** | **COMPLETE (PASS)** | TF-IDF, Semantic, Hybrid, MMR, Explanations, Personalisation; 20 tests passing |
| **Module 5: Evaluation Metrics** | **COMPLETE (PASS)** | 8 ranking metrics, 4 diversity/coverage metrics, ground truth (2,544 queries), 4-model unified benchmark, performance profiling, 12/12 stress checks, human eval framework, 5 notebooks, 7 plots; 19 tests passing (39/39 total suite) |
| **Module 6: Streamlit Deployment** | **PLANNED** | Scheduled for next milestone |
| **Module 7: Presentation & Documentation** | **COMPLETE (PASS)** | Comprehensive technical reports in `docs/` (`MILESTONE_2_TECHNICAL_DOCUMENTATION.md` & `MODULE_5_EVALUATION_REPORT.md`) |

---

## Documentation
For detailed architectural designs, mathematical formulations, data dictionaries, and benchmark reports, refer to:
- **[`docs/MODULE_5_EVALUATION_REPORT.md`](docs/MODULE_5_EVALUATION_REPORT.md)** — Comprehensive Module 5 Evaluation Report (Ranking, Diversity, Coverage, Latency, Robustness, Human Eval)
- **[`docs/MILESTONE_2_TECHNICAL_DOCUMENTATION.md`](docs/MILESTONE_2_TECHNICAL_DOCUMENTATION.md)** — Architectural Design and Milestone 2 Engineering Documentation

---
**Author:** S Koushick  
**Internship:** Infosys Springboard Internship 7.0  
**Deliverable:** Milestone 2 & Milestone 3 (Module 5) Completion

