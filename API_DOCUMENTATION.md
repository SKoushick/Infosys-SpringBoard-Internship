# TEDMind REST API Specification

Base URL: `/api/`  
Format: JSON (`application/json`)  
Authentication: Session / Cookie / Optional Token

---

## 1. System Health

### `GET /api/health/`
Checks engine and model readiness.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "TEDMind Recommender Engine",
  "catalog_size": 2550,
  "models_loaded": {
    "tfidf_recommender": true,
    "semantic_recommender": true,
    "hybrid_recommender": true
  },
  "timestamp": 1731862500.12
}
```

---

## 2. Talk Catalog & Exploration

### `GET /api/talks/`
Returns a paginated list of TED talks with optional search filters.

**Parameters:**
- `q`: Substring search query in title and description (optional)
- `tag`: Filter by specific topic tag (optional)
- `speaker`: Filter by speaker name (optional)
- `page`: Page index (default: 1)
- `per_page`: Items per page (default: 12, max: 50)

**Response (200 OK):**
```json
{
  "total_count": 2550,
  "page": 1,
  "per_page": 12,
  "total_pages": 213,
  "results": [
    {
      "talk_id": 1,
      "title": "Do schools kill creativity?",
      "main_speaker": "Ken Robinson",
      "tags": ["children", "creativity", "education"],
      "views_formatted": "47,227,110",
      "duration_minutes": 19.4,
      "url": "https://www.ted.com/talks/ken_robinson_says_schools_kill_creativity"
    }
  ]
}
```

### `GET /api/talks/{id}/`
Returns metadata for a specific presentation.

### `GET /api/talks/{id}/similar/`
Returns the top companion recommendations for a talk.

---

## 3. Recommendation Generation

### `POST /api/recommendations/generate/`
Generates ranked recommendations from multiple input signals.

**Request Body:**
```json
{
  "seed_talk_id": 1,
  "query_text": "creativity and arts education",
  "interests": ["education", "creativity"],
  "method": "hybrid",
  "top_k": 5,
  "apply_mmr": true
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "method": "hybrid",
  "top_k": 5,
  "apply_mmr": true,
  "latency_ms": 42.1,
  "count": 5,
  "recommendations": [
    {
      "talk_id": 66,
      "title": "Bring on the learning revolution!",
      "main_speaker": "Ken Robinson",
      "score": 0.8842,
      "match_pct": 88,
      "explanation": {
        "rationale": "High semantic overlap (0.89) and shared topics: education, creativity",
        "shared_tags": ["education", "creativity"],
        "speaker_match": true,
        "combined_score": 0.8842,
        "tfidf_score": 0.8751,
        "semantic_score": 0.8933
      }
    }
  ]
}
```

---

## 4. User Profile & Saved Talks

### `GET /api/users/me/`
Returns the active user's profile and learning preferences.

### `GET /api/users/me/saved/`
Returns the list of saved talks for the current user or guest session.

### `POST /api/users/me/saved/`
**Request Body:**
```json
{
  "talk_id": 1
}
```

### `DELETE /api/users/me/saved/{talk_id}/`
Removes the specified talk from saved bookmarks.
