# TEDMind System Architecture Document

## 1. Executive Architecture Summary

TEDMind is an AI-powered personalized TED Talk discovery and recommendation platform built with Python, Django 5.2, Scikit-Learn, and PyTorch (Sentence-BERT). It processes 2,550 curated TED presentations using a multi-signal architecture combining lexical features (multi-field TF-IDF) and dense semantic representations (`sentence-transformers/all-MiniLM-L6-v2`) reranked via Maximal Marginal Relevance (MMR) for diversity.

```
                          ┌────────────────────────┐
                          │   Client Browser /     │
                          │   Mobile / REST Client │
                          └───────────┬────────────┘
                                      │ HTTP / HTTPS
                                      ▼
                          ┌────────────────────────┐
                          │   Django 5.2 Gateway   │
                          │   CSRF / Security      │
                          └─────┬────────────┬─────┘
                                │            │
           ┌────────────────────┴───┐    ┌───┴────────────────────┐
           │ Presentation Layer     │    │ REST API Gateway       │
           │ (Django Templates,     │    │ (/api/talks, /api/rec) │
           │  Bootstrap 5, Vanilla) │    └───────────┬────────────┘
           └────────────┬───────────┘                │
                        │                            │
                        ▼                            ▼
          ┌─────────────────────────────────────────────────────────┐
          │         RecommenderService Layer (Singleton)            │
          │   - In-memory talk dictionary lookup (2,550 talks)     │
          │   - TF-IDF Sparse Matrix (Cosine Similarity)            │
          │   - SBERT Dense 384-d Matrix (Dot Product)              │
          │   - Multi-Signal Hybrid Scoring (w_tfidf + w_semantic)  │
          │   - Maximal Marginal Relevance (MMR Diversity)          │
          │   - Transparent Evidence-Based Explanations             │
          └───────────────────────────┬─────────────────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
      ┌─────────────────────────┐           ┌─────────────────────────┐
      │   Relational Database   │           │ Precomputed ML Assets   │
      │   (SQLite / PostgreSQL) │           │ (NumPy .npy, Joblib)    │
      │   - Users & Profiles    │           │ - tfidf_vectorizer.pkl  │
      │   - SavedTalks          │           │ - combined_tfidf.npz    │
      │   - Collections & Items │           │ - sbert_embeddings.npy  │
      │   - ViewingHistory      │           └─────────────────────────┘
      │   - UserFeedback        │
      │   - RecommendationEvents│
      └─────────────────────────┘
```

---

## 2. Recommender Engine Design

### 2.1 Multi-Signal Scoring Pipeline
Given a user query vector $q$, candidate talks $d_i$ are scored using a linear combination of lexical TF-IDF similarity and dense Sentence-BERT cosine similarity:

$$S(q, d_i) = w_{\text{tfidf}} \cdot \text{Sim}_{\text{lexical}}(q_{\text{tfidf}}, d_{i,\text{tfidf}}) + w_{\text{semantic}} \cdot \text{Sim}_{\text{semantic}}(q_{\text{sbert}}, d_{i,\text{sbert}})$$

Where:
- $w_{\text{tfidf}} = 0.5$ and $w_{\text{semantic}} = 0.5$ for default hybrid mode.
- Pure modes set weights dynamically to $(1.0, 0.0)$ or $(0.0, 1.0)$.

### 2.2 Maximal Marginal Relevance (MMR)
To combat recommendation redundancy and "filter bubbles", candidate talks are reranked using MMR:

$$\text{MMR}(q, D, R) = \operatorname*{argmax}_{d_i \in D \setminus R} \left[ \lambda \cdot \text{Sim}(q, d_i) - (1 - \lambda) \max_{d_j \in R} \text{Sim}(d_i, d_j) \right]$$

Where:
- $\lambda = 0.7$ provides the empirically optimal balance between relevance accuracy and category diversity.
- Matrix operations remain sparse in memory, avoiding full catalog densification.

### 2.3 Resilient Thread-Safe Singleton
`RecommenderService.get_instance()` initializes precomputed matrices once per worker process:
- Unit-normalized SBERT embeddings (2550 x 384 float32) load in under 15ms.
- Multi-field TF-IDF sparse matrices load in under 20ms.
- Online query inference completes in **30–80 milliseconds**.
- Offline Hugging Face mode (`HF_HUB_OFFLINE=1`) prevents network latency stalls.

---

## 3. Database & User Persistence Layer

The database schema decouples talks metadata from user-specific state:
- **`UserProfile`**: Stores bio, onboarding status, and preferred topics array.
- **`SavedTalk`**: Supports both anonymous guest sessions (`session_key`) and registered accounts (`user`).
- **`Collection` & `CollectionItem`**: Supports custom user playlists and study paths.
- **`ViewingHistory`**: Provides audit trails of viewed talks for real-time personalization.
- **`UserFeedback`**: Captures explicit user signals (`like`, `dislike`, `not_interested`).
- **`RecommendationEvent`**: Records query latency, method, and recommendation candidate IDs.
