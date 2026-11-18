# TEDMind Application Security Policy

## 1. Overview
TEDMind is designed with security as an architectural priority. This document outlines the security controls, authentication safeguards, data protection measures, and vulnerability reporting procedures implemented across the platform.

---

## 2. Core Security Controls

### 2.1 Environment Isolation & Secret Management
- **`DEBUG=False` in Production:** Debug mode exposes internal stack traces, system paths, and configurations. It must never be enabled in public environments.
- **Environment-based `DJANGO_SECRET_KEY`:** Required when `DEBUG=False`. If absent, application startup terminates immediately.
- **Restricted `ALLOWED_HOSTS`:** Configurable per environment to prevent HTTP Host Header poisoning attacks.
- **Credential Protection:** Database credentials, secret keys, and tokens are stored in environment variables (`.env`) excluded from version control via `.gitignore`.

### 2.2 Transport & Cookie Security
- **HTTP Strict Transport Security (HSTS):** `SECURE_HSTS_SECONDS = 31536000` (1 year), `SECURE_HSTS_INCLUDE_SUBDOMAINS = True`, and `SECURE_HSTS_PRELOAD = True` in production.
- **SSL Redirection:** `SECURE_SSL_REDIRECT = True` enforces HTTPS for all incoming requests.
- **Cookie Flags:**
  - `SESSION_COOKIE_SECURE = True` (HTTPS only in production)
  - `SESSION_COOKIE_HTTPONLY = True` (mitigates client-side XSS cookie extraction)
  - `CSRF_COOKIE_SECURE = True` (HTTPS only in production)
  - `CSRF_COOKIE_HTTPONLY = False` (allows single-page AJAX components to read CSRF tokens securely from cookies)

### 2.3 Protection Against Web Vulnerabilities
- **Cross-Site Request Forgery (CSRF):** Enforced via Django's `CsrfViewMiddleware`. All state-changing mutations (saving talks, updating preferences, submitting queries) require a valid CSRF token.
- **Cross-Site Scripting (XSS):**
  - Django template auto-escaping is active across all templates.
  - `SECURE_BROWSER_XSS_FILTER = True`
  - `SECURE_CONTENT_TYPE_NOSNIFF = True` prevents MIME-type sniffing.
- **Clickjacking Protection:** `X_FRAME_OPTIONS = "DENY"` blocks framing of TEDMind in foreign `<frame>`, `<iframe>`, or `<embed>` tags.
- **SQL Injection Prevention:** All database operations utilize Django ORM parameterized queries with `select_related` and `prefetch_related`. Raw SQL string concatenation is strictly prohibited.

### 2.4 Authentication & Access Control
- **Password Hashing:** Django's default PBKDF2 with SHA-256 password hasher with work factor salt rounds.
- **Object-Level Authorization:** Users can only query, modify, or delete their own private bookmarks, profile settings, and viewing history.
- **Session Protection:** Anonymous sessions are rotated and migrated seamlessly upon user authentication.

---

## 3. Reporting a Security Vulnerability

If you identify a security issue or vulnerability in TEDMind:
1. Do **not** post details in public GitHub issues or discussions.
2. Report the vulnerability privately with steps to reproduce.
3. We will acknowledge receipt within 24 hours and provide an estimated remediation timeline.
