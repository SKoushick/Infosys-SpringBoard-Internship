# TEDMind Product Changelog

All notable changes to the TEDMind platform are documented here.

---

## [2.0.0] - 2026-11-17 — Production Transformation

### Added
- **User Authentication & Accounts Architecture (`apps.accounts`)**:
  - Implemented `UserProfile` model linked 1:1 with `User` storing `preferred_topics`, `bio`, and `recommendations_enabled`.
  - Added user registration, login, logout, profile settings, and guided onboarding interest selection.
  - Implemented automatic guest-session bookmark migration to permanent user accounts upon login (`_migrate_session_bookmarks`).
- **Relational Data Persistence (`apps.talks`)**:
  - Enhanced `SavedTalk` to support both authenticated users and guest sessions.
  - Added `Collection` and `CollectionItem` models for user-curated playlists and study paths.
  - Added `ViewingHistory` model to track talk visits for real-time personalization.
  - Added `UserFeedback` model for explicit user signals (`like`, `dislike`, `not_interested`).
  - Added `RecommendationEvent` for query latency and candidate auditing.
- **RESTful API Layer (`apps.api`)**:
  - Implemented `GET /api/health/` system health check.
  - Implemented `GET /api/talks/`, `GET /api/talks/<id>/`, and `GET /api/talks/<id>/similar/`.
  - Implemented `POST /api/recommendations/generate/` for programmatic multi-signal recommendations.
  - Implemented `GET/POST/DELETE /api/users/me/saved/` endpoints.
- **Security Hardening**:
  - Added environment-variable configuration via `python-dotenv`.
  - Enforced `X_FRAME_OPTIONS = "DENY"`, `SECURE_CONTENT_TYPE_NOSNIFF`, `SECURE_BROWSER_XSS_FILTER`.
  - Added production HTTPS security headers (`HSTS`, `SECURE_SSL_REDIRECT`, `SESSION_COOKIE_SECURE`).
  - Created `SECURITY.md` vulnerability reporting and security controls policy.
- **DevOps & Production Readiness**:
  - Created `Dockerfile` and `docker-compose.yml` with PostgreSQL and Gunicorn support.
  - Created `DATABASE_SCHEMA.md`, `API_DOCUMENTATION.md`, `TESTING.md`, and `DEPLOYMENT.md`.
- **Test Automation**:
  - Added `test_accounts.py` (7 tests).
  - Added `test_api.py` (7 tests).
  - Maintained all 9 baseline `test_views.py` tests (totaling 23 web tests passing).

### Changed
- Refactored `Module_6_Django_Application/config/settings.py` for environment-driven `DEBUG`, `SECRET_KEY`, and `ALLOWED_HOSTS`.
- Upgraded `static/css/tedmind.css` with a refined editorial technology design system (deep navy `#0f172a`, subtle red accents, elevated card shadows).
- Updated navbar with dynamic user profile dropdown and authentication controls.

---

## [1.0.0] - 2026-11-17 — Initial Demonstration Baseline
- Initial Module 6 Django Frontend with 6 pages.
- Fast `RecommenderService` singleton loading TF-IDF, Sentence-BERT, and Hybrid models.
- Guest session bookmarking and passing 9 view tests.
