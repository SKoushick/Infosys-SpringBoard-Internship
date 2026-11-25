# TEDMind Testing & Quality Assurance Guide

This document details the automated test suites, testing matrix, and verification commands for the TEDMind platform.

---

## 1. Test Matrix Overview

| Test Module | Test File | Test Count | Scope |
| :--- | :--- | :--- | :--- |
| **Django View Tests** | `Module_6_Django_Application/tests/test_views.py` | 9 | Home, explore, search, talk details, recommendation forms, seed/query results, guest bookmarks, about page. |
| **Accounts & Personalization** | `Module_6_Django_Application/tests/test_accounts.py` | 7 | User registration, password validation, guest session migration, onboarding interests, collections, feedback. |
| **REST API Tests** | `Module_6_Django_Application/tests/test_api.py` | 7 | Health checks, talk list, search, details, companion talks, recommendation generation, saved talk API. |
| **Data Models & Constraints** | `Module_6_Django_Application/tests/test_models.py` | 1 | Basic model constraints and integrity. |
| **Learning Path Services** | `Module_6_Django_Application/tests/test_learning_services.py` | 18 | Transactional deletion, progress merging, idempotent heartbeats, concurrency races, input sanitization. |
| **Learning Admin Safeguards** | `Module_6_Django_Application/tests/test_learning_admin.py` | 7 | Admin single/bulk deletion routing, inlines, read-only audit logs, rollback protection. |
| **Core ML & Pedagogical Engine** | `tests/test_ted_recommender.py`, `tests/test_learning.py` | 29 | TF-IDF, SBERT embeddings, MMR diversity, hybrid fusion, pedagogical contracts, complexity scoring. |
| **Evaluation Metrics** | `Module_5_Evaluation_Metrics/tests/` | 32 | NDCG@K, MAP@K, ILD diversity, catalogue coverage, fair query verification, robustness stress tests. |
| **Total Automated Tests** | — | **110 Tests** | **100% Passing** |

---

## 2. Test Execution Commands

### Run All Web Application Tests (49 Tests)
```powershell
cd Module_6_Django_Application
..\venv\Scripts\python.exe manage.py test tests
```

### Run Django System Check
```powershell
cd Module_6_Django_Application
..\venv\Scripts\python.exe manage.py check
```

### Run Core ML & Pedagogical Engine Tests (29 Tests)
```powershell
$env:PYTHONPATH = "src;Module_4_Recommendation_Engine"
.\venv\Scripts\python.exe -m pytest tests -q
```

### Run Module 5 Evaluation Benchmark Tests (32 Tests)
```powershell
$env:PYTHONPATH = "src;Module_4_Recommendation_Engine;Module_5_Evaluation_Metrics/src"
.\venv\Scripts\python.exe -m pytest Module_5_Evaluation_Metrics/tests -q
```
