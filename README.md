# TEDMind — AI-Powered Personalized TED Talk Discovery Platform
### Infosys Springboard Internship 7.0 — Master Engineering Product

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/django-5.2-green.svg)](https://www.djangoproject.com/)
[![Testing](https://img.shields.io/badge/tests-110%20passed-success.svg)](TESTING.md)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Sentence--Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-blueviolet.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## 1. Project Overview

**TEDMind** is a production-minded AI discovery and personalized recommendation platform designed to help users uncover ideas worth spreading. Built on top of 2,550 curated presentations from the verified TED catalog, TEDMind combines natural language processing (multi-field TF-IDF), dense contextual transformers (`all-MiniLM-L6-v2`), multi-signal hybrid fusion, and Maximal Marginal Relevance (MMR) diversity reranking.

---

## 2. Key Capabilities & Architecture

- **Multi-Signal Hybrid Engine:** Configurable score fusion between lexical keyword TF-IDF ($44,613$ features) and dense contextual Sentence-BERT ($384$ dimensions).
- **Maximal Marginal Relevance (MMR):** Algorithmic re-ranking ($\lambda=0.70$) that improves topic diversity by **+20.6%** without degrading relevance.
- **Explainable AI:** Evidence-backed rationales explaining *why* talks are recommended (shared topic badges, speaker affinity, cosine scores).
- **User Accounts & Personalization:** Secure user registration, authentication, topic preference onboarding, and seamless migration of guest bookmarks.
- **Custom Collections:** User-curated playlists and study paths with personal learning notes.
- **Negative Feedback Loop:** Dynamic exclusion of disliked talks (`dislike`, `not_interested`) to eliminate unwanted topics.
- **RESTful API Layer:** Programmatic endpoints for talks, recommendations, user interactions, and system health checks (`/api/`).
- **Deployment Readiness:** Environment-variable isolation (`.env`), HTTP security headers (HSTS, X-Frame-Options, CSRF, No-Sniff), and Docker Compose containerization configuration.

---

## 3. Quickstart & Launch Guide

### Local Development (Windows PowerShell)

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Set Python paths
$env:PYTHONPATH = "src;Module_6_Django_Application;Module_6_Django_Application/apps"

# 3. Apply database migrations
python Module_6_Django_Application/manage.py migrate

# 4. Start local development server
python Module_6_Django_Application/manage.py runserver 127.0.0.1:8000
```

Open your browser at: 👉 **`http://127.0.0.1:8000/`**

### Containerized Deployment (Docker Compose)

```bash
docker-compose up -d --build
docker-compose exec web python Module_6_Django_Application/manage.py migrate
docker-compose exec web python Module_6_Django_Application/manage.py collectstatic --noinput
```

---

## 4. Automated Testing Suite (110/110 Passing)

TEDMind enforces rigorous quality assurance across all modules and Phase 2 learning pathways:

| Test Suite | File Location | Tests | Status |
| :--- | :--- | :---: | :---: |
| **Web Application Views & UI** | `Module_6_Django_Application/tests/test_views.py` | 9 | **PASS** |
| **Accounts & Personalization** | `Module_6_Django_Application/tests/test_accounts.py` | 7 | **PASS** |
| **REST API Endpoints** | `Module_6_Django_Application/tests/test_api.py` | 7 | **PASS** |
| **Data Models & Constraints** | `Module_6_Django_Application/tests/test_models.py` | 1 | **PASS** |
| **Learning Path & Heartbeat Services** | `Module_6_Django_Application/tests/test_learning_services.py` | 18 | **PASS** |
| **Learning Admin Safeguards & Merging** | `Module_6_Django_Application/tests/test_learning_admin.py` | 7 | **PASS** |
| **Core ML & Pedagogical Engine** | `tests/test_ted_recommender.py`, `tests/test_learning.py` | 29 | **PASS** |
| **Module 5 Evaluation Metrics** | `Module_5_Evaluation_Metrics/tests/test_*.py` | 32 | **PASS** |
| **Total Test Suite** | — | **110 Tests** | **100% PASS** |

To run the complete web suite:
```powershell
cd Module_6_Django_Application
..\venv\Scripts\python.exe manage.py test tests
```

---

## 5. Measured Benchmark Scorecard (Module 5: Evaluation Metrics)

Evaluated across 100 deterministic evaluation queries under identical conditions (inspected from `Module_5_Evaluation_Metrics/results/benchmarks/benchmark_summary_table.csv`):

| Model | NDCG@10 | MAP@10 | Precision@10 | Recall@10 | ILD@10 (Diversity) | Coverage (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Content-Based** | 0.0063 | 0.0026 | 0.0040 | 0.0077 | 0.6851 | 51.88% | **28.96 ms** |
| **Sentence-BERT Semantic** | 0.0053 | 0.0018 | 0.0040 | 0.0073 | 0.5597 | 48.98% | **23.64 ms** |
| **Hybrid Multi-Signal (No MMR)** | **0.0076** | **0.0035** | 0.0040 | 0.0073 | 0.5763 | 50.59% | 34.83 ms |
| **Hybrid Multi-Signal + MMR** | 0.0074 | 0.0033 | 0.0040 | 0.0073 | **0.6953 (+20.6%)** | **54.16%** | 51.20 ms |

*Note: Benchmarks were evaluated against the editorial companion talk proxy ground truth (`related_talks`). Qualitative human relevance evaluation is pending (0/200 human grades assigned).*

---

## 6. Project Documentation Index

- [docs/MODULE_7_FINAL_TECHNICAL_DOCUMENTATION.md](docs/MODULE_7_FINAL_TECHNICAL_DOCUMENTATION.md) — Comprehensive technical documentation, architecture, formulas, evaluation audit, and Phase 2 specifications.
- [docs/presentation/TEDMind_Final_Presentation.pdf](docs/presentation/TEDMind_Final_Presentation.pdf) — Master 10-slide final project presentation deck (Infosys Springboard 7.0).
- [docs/presentation/TEDMind_5_Minute_Speaker_Script.md](docs/presentation/TEDMind_5_Minute_Speaker_Script.md) — 5-minute timed speaker script with slide transition notes.
- [ARCHITECTURE.md](ARCHITECTURE.md) — System components, multi-signal scoring formulas, and singleton lifecycle.
- [SECURITY.md](SECURITY.md) — Authentication safeguards, HTTP headers, CSRF enforcement, and vulnerability policy.
- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) — Entity-relationship models, table definitions, and constraints.
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) — RESTful API endpoints, request/response JSON schemas, and status codes.
- [TESTING.md](TESTING.md) — Automated testing matrix, commands, and validation procedures.
- [DEPLOYMENT.md](DEPLOYMENT.md) — Docker Compose, Gunicorn, PostgreSQL, and backup/restore guidelines.
- [CHANGELOG.md](CHANGELOG.md) — Chronological log of versions, migrations, and new features.

---
**Author:** S Koushick  
**Internship:** Infosys Springboard Internship 7.0  
**Project:** Personalized TED Talks Recommendation System Using Machine Learning
