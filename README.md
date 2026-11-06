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
pytest tests/ -v
```
*(All 16 unit and integration test suites pass in ~54s)*

### 4. Interactive Notebook Demonstration
Launch Jupyter and open the Module 4 demonstration notebook:
```bash
jupyter notebook Module_4_Recommendation_Engine/01_Recommendation_Engine_Demonstration.ipynb
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
│   └── MILESTONE_2_TECHNICAL_DOCUMENTATION.md # 33-section comprehensive technical report
├── src/
│   └── ted_recommender/                       # Production Python Package
│       ├── __init__.py
│       ├── config.py                          # Dynamic root resolution & config dataclasses
│       ├── data.py                            # Dataset loaders, schema validation, text cleaners
│       ├── features.py                        # TF-IDF pipeline & Sentence-BERT embedding generator
│       ├── recommender.py                     # Content-based, Semantic, Hybrid, MMR, & Explanations
│       ├── collaborative.py                   # Feasibility audit & future interface contract
│       └── evaluation.py                      # Ranking metrics (P@K, R@K, NDCG@K, MAP@K, ILD)
├── tests/                                     # Automated Test Suite (16/16 Passed)
│   ├── test_data.py                           # Schema, row alignment, and text cleaning tests
│   ├── test_vectorization.py                  # Matrix shapes, sparsity, and transformation tests
│   ├── test_recommender.py                    # Top-k, self-exclusion, MMR, and edge case tests
│   └── test_evaluation.py                     # Metric mathematical correctness & audit checks
├── Module_1_Data_Collection/                  # Module 1 Scripts & Data
│   └── Scraped/Data/                          # Raw, Processed, and External CSVs (2550 talks)
├── Module_2_Data_Exploration_Preprocessing/   # Module 2 EDA Notebooks & Visualizations
├── Module_3_Feature_Engineering_Vectorization/# Module 3 Vectorization Notebooks & Feature Matrices
└── Module_4_Recommendation_Engine/            # Module 4 Implementation & Benchmark
    ├── 01_Recommendation_Engine_Demonstration.ipynb
    ├── train_and_benchmark.py
    └── outputs/milestone_2_benchmark_report.json
```

---

##  Measured Benchmark Performance

| Recommender Model | Average Latency | Feature Dimension | Primary Strength |
|:---|:---:|:---:|:---|
| **TF-IDF Content-Based** | **30.07 ms** | 44,613 sparse | Exact keyword, speaker, and terminology matching |
| **Semantic Transformer** | **15.66 ms** | 384 dense | Conceptual alignment, synonymy, latent topical discovery |
| **Hybrid (TF-IDF + Semantic + MMR)** | **67.40 ms** | Multi-signal | Optimal balance of precision, semantic recall, and diversity |

*Measurements taken across 2,550 talks running locally on CPU.*

---

##  Collaborative Filtering Decision
An empirical audit of the dataset confirmed that the TED archive supplies only aggregate talk statistics (`views`, `comments`, `ratings` emotion counts) and **contains zero individual user identifiers or user-item interaction logs**.

In adherence to scientific integrity:
- **No fake users or synthetic interactions were fabricated.**
- Collaborative filtering is formally reported as **UNSUPPORTED BY AVAILABLE EVIDENCE**.
- An extensible `CollaborativeSignalProvider` interface is implemented in `src/ted_recommender/collaborative.py` to seamlessly blend user signals when telemetry logs become available.

---

##  Seven-Module Status Summary

| Module | Status | Verification Evidence |
|:---|:---:|:---|
| **Module 1: Data Collection** | **COMPLETE (PASS)** | 2,550 records loaded from verified TED public archive |
| **Module 2: Data Exploration & Preprocessing** | **COMPLETE (PASS)** | Cleaned text, transcription cue removal, missing transcript fallback |
| **Module 3: Feature Engineering & Vectorization** | **COMPLETE (PASS)** | 4 TF-IDF vectorizers fitted; 2550×44,613 matrix; 2550×384 embeddings |
| **Module 4: Develop Recommendation Engine** | **COMPLETE (PASS)** | TF-IDF, Semantic, Hybrid, MMR, Explanations, Personalisation; 16 tests passing |
| **Module 5: Evaluation Metrics** | **EVALUATION-READY** | Metric formulas implemented; baseline benchmark run; Milestone 3 scope |
| **Module 6: Streamlit Deployment** | **PLANNED** | Scheduled for Milestone 3 |
| **Module 7: Presentation & Documentation** | **COMPLETE (PASS)** | Comprehensive technical documentation in `docs/` |

---

##  Documentation
For detailed architectural designs, mathematical formulations, data dictionaries, and ablation discussions, refer to:
 **[`docs/MILESTONE_2_TECHNICAL_DOCUMENTATION.md`](docs/MILESTONE_2_TECHNICAL_DOCUMENTATION.md)**

---
**Author:** S Koushick  
**Internship:** Infosys Springboard Internship 7.0  
**Deliverable:** Milestone 2 Completion

