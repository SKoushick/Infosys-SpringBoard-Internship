# Module 6: TEDMind — Personalized TED Talks Recommendation Web Application

**Infosys Springboard Internship 7.0 — Milestone 6 Demonstration Application**  
*Academic Year: 2026 | Project: Personalized TED Talks Recommendation System Using Machine Learning*

---

## 1. Executive Summary

**TEDMind** is the interactive, production-grade Django web application delivering the machine learning models engineered across Modules 1 through 5. It serves real-time, personalized TED Talk recommendations across 2,550 curated presentations using multi-field TF-IDF, Sentence-BERT (`all-MiniLM-L6-v2`) semantic embeddings, multi-signal hybrid fusion, and Maximal Marginal Relevance (MMR) diversity reranking.

---

## 2. Architecture & Design Principles

```
Module_6_Django_Application/
├── manage.py                   # Django management CLI with auto-configured PYTHONPATH
├── requirements.txt            # Python dependencies (Django 5.2+, PyTorch, Scikit-Learn)
├── .env.example                # Environment variables template
├── config/                     # Core Django project configuration
│   ├── settings.py             # App registration, SQLite database, static/template configs
│   ├── urls.py                 # Root URL router
│   ├── wsgi.py / asgi.py       # WSGI/ASGI application gateways
├── apps/
│   ├── talks/                  # Catalog browsing, search, talk details, session bookmarking
│   │   ├── models.py           # SavedTalk (session-backed SQLite bookmarking)
│   │   ├── views.py            # Home, Explore, Detail, Bookmarks, About views
│   │   ├── urls.py             # Talk routing
│   │   └── context_processors.py # Injects saved talks count & top tags globally
│   └── recommendations/        # Recommendation inference & explanation engine
│       ├── services.py         # RecommenderService singleton (zero-retraining, in-memory cache)
│       ├── views.py            # Form input view, ranked results view, title autocomplete API
│       └── urls.py             # Recommendation routing
├── templates/                  # Bootstrap 5 responsive UI templates
│   ├── base.html               # Shared layout, navbar, search drawer, footer
│   ├── talks/                  # home.html, explore.html, talk_detail.html, saved_talks.html, about.html
│   └── recommendations/        # recommend_form.html, results.html
├── static/                     # Assets
│   ├── css/tedmind.css         # TED red (#e62b1e) design system, cards, badges
│   └── js/tedmind.js           # AJAX bookmarking, live search autocomplete, topic pill chips
└── tests/
    └── test_views.py           # 9 automated Django integration & view test cases
```

### Key Technical Characteristics
1. **Thread-Safe Singleton Service (`RecommenderService`)**:
   Loads the 2,550-talk dataset, multi-field TF-IDF pipeline, and unit-normalized Sentence-BERT embeddings once at application startup. Subsequent web requests execute inference in under 50 milliseconds with zero redundant disk reads.
2. **Transparent ML Explanations**:
   Every recommendation is paired with an empirical match score, rationale explanation, shared topic tags, and model breakdown (TF-IDF vs. Semantic contributions).
3. **Session-Backed Personalization**:
   Users can bookmark talks in their active browsing session. The recommender automatically leverages their saved bookmarks to synthesize a real-time preference profile without requiring user registration or fake collaborative filtering.
4. **Resilient Local Execution**:
   Enforces `HF_HUB_OFFLINE=1` to guarantee instant local inference without external Hugging Face network dependencies or connection timeout stalls.

---

## 3. The 6 Application Interfaces

| Page | URL Path | Key Capabilities |
| :--- | :--- | :--- |
| **1. Home Dashboard** | `/` | Hero banner, global search bar, popular topic pills, featured high-impact talks, live metrics, tech stack breakdown. |
| **2. Explore Catalog** | `/explore/` | Full-catalog browsing (2,550 talks), multi-parameter search (keyword, topic tag, speaker), pagination, one-click bookmarking. |
| **3. Recommendation Hub** | `/recommend/` | Triple input modalities: (A) Seed talk autocomplete, (B) Multi-topic tag pills, (C) Free-text intent prompt; model selector (Hybrid, Semantic, TF-IDF), Top-K selector, MMR diversity toggle. |
| **4. Recommendation Results** | `/recommend/results/` | Ranked candidate cards, match percentage badges, rationale explanations, shared tags, external links to TED.com, and quick-tuning parameters bar. |
| **5. Talk Details** | `/talks/<talk_id>/` | Comprehensive presentation metadata, duration, view count, clickable topic pills, direct TED link, and 3 instantly computed companion talks. |
| **6. Project & Evaluation** | `/about/` | Problem statement, mathematical pipeline documentation, verified Module 5 evaluation scorecard, and research integrity disclosure. |

---

## 4. Setup and Launch Instructions

### Prerequisites
- Python 3.10+ (Verified on Python 3.11.7)
- Virtual environment (`venv`) with dependencies installed.

### Step-by-Step Launch (Windows PowerShell)

```powershell
# 1. Navigate to the repository root
cd c:\Users\DELL\Desktop\INTERNSHIPS\Infosys_SpringBoard\Ted_Talk_Recommendation

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Set Python path for root package imports
$env:PYTHONPATH = "src;Module_6_Django_Application;Module_6_Django_Application/apps"

# 4. Run database migrations (SQLite)
.\venv\Scripts\python.exe Module_6_Django_Application/manage.py migrate

# 5. Start the Django development server
.\venv\Scripts\python.exe Module_6_Django_Application/manage.py runserver 127.0.0.1:8000
```

### Step-by-Step Launch (macOS / Linux Bash)

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Set PYTHONPATH
export PYTHONPATH="src:Module_6_Django_Application:Module_6_Django_Application/apps"

# 3. Run database migrations
python Module_6_Django_Application/manage.py migrate

# 4. Start the server
python Module_6_Django_Application/manage.py runserver 127.0.0.1:8000
```

Once running, open your web browser at:  
👉 **`http://127.0.0.1:8000/`**

---

## 5. Running Automated Tests

The application includes 9 automated Django view and integration test cases covering routing, search filtering, recommendation inference, and session bookmarking.

```powershell
cd Module_6_Django_Application
..\venv\Scripts\python.exe manage.py test tests
```

**Expected Output:**
```
Creating test database for alias 'default'...
.........
----------------------------------------------------------------------
Ran 9 tests in 24.853s

OK
Destroying test database for alias 'default'...
```

---

## 6. Mentor Demonstration Script (6-Step Walkthrough)

When presenting this demo to your mentor:

1. **Step 1: Open the Dashboard (`/`)**
   - Point out the TED-branded styling (`#e62b1e`), hero search, and the 6 featured presentations loaded dynamically from the 2,550-talk dataset.
   - Highlight the "Platform Intelligence" scorecard displaying the real-time dataset size and vectorizer specs.
2. **Step 2: Demonstrate Catalog Search (`/explore/`)**
   - Click "Explore Talks" or type a query like `"creativity"` or `"artificial intelligence"`.
   - Show how talks are filtered instantly with speaker attribution, tags, and pagination.
   - Click the bookmark icon on any talk to demonstrate active session bookmarking.
3. **Step 3: Deep-Dive into Talk Details (`/talks/1/`)**
   - Open Sir Ken Robinson's talk (*Do schools kill creativity?*).
   - Show the total views (47M+), tags, external watch link on TED.com, and the 3 automatically suggested companion talks at the bottom.
   - Click "Recommend Similar Talks" to pass the talk into the recommendation engine.
4. **Step 4: Execute Multi-Signal Hybrid Recommendation (`/recommend/`)**
   - Show the three input methods:
     - Method A: Seed talk (e.g., Talk #1).
     - Method B: Select 2–3 topic tags (e.g., `technology`, `education`, `creativity`).
     - Method C: Type a natural query (e.g., `"how can technology transform classrooms and human thinking"`).
   - Select **Multi-Signal Hybrid (TF-IDF + Semantic)**.
   - Keep **MMR Diversity Reranking** toggled ON to demonstrate redundancy reduction.
   - Click **Generate Recommendations**.
5. **Step 5: Inspect Recommendations & Explainability (`/recommend/results/`)**
   - Observe the sub-100ms inference time displayed in the results badge.
   - Point out the Match Percentage (e.g., 88% Match), shared topic badges, and the **Recommendation Rationale** box explaining *why* the talk was selected.
   - Use the **Quick Tuning Bar** at the top to switch between Hybrid, Semantic, and TF-IDF models on the fly to show how ranking positions adapt.
6. **Step 6: Review Technical Integrity (`/about/`)**
   - Navigate to the About page.
   - Show the mathematical formulations of TF-IDF Cosine Similarity, Sentence-BERT Cosine Similarity, Hybrid Fusion, and MMR.
   - Present the empirical Module 5 benchmark table showing exact NDCG@10, MAP@10, and Intra-List Diversity scores.
   - Highlight the **Research Integrity Statement** confirming no synthetic ratings or fabricated collaborative filtering matrices were used.
