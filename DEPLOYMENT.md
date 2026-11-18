# TEDMind Production Deployment Guide

This document outlines deployment procedures for production environments using Docker, Gunicorn, Nginx, and PostgreSQL.

---

## 1. Prerequisites
- Docker & Docker Compose
- Python 3.11+
- PostgreSQL 15+ (or Docker managed)

---

## 2. Production Environment Configuration (`.env`)

Create `.env` in `Module_6_Django_Application/` (or set environment variables on your cloud host):

```ini
DJANGO_DEBUG=False
DJANGO_SECRET_KEY=generate_strong_50_character_random_string_here
DJANGO_ALLOWED_HOSTS=tedmind.yourdomain.com,127.0.0.1
DATABASE_URL=postgresql://tedmind_user:secure_password@db:5432/tedmind_db
DJANGO_SECURE_SSL_REDIRECT=True
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

---

## 3. Deployment with Docker Compose

Run the entire stack (PostgreSQL + TEDMind Django Gunicorn):

```bash
# Build and run in detached mode
docker-compose up -d --build

# Run database migrations
docker-compose exec web python Module_6_Django_Application/manage.py migrate

# Collect static assets
docker-compose exec web python Module_6_Django_Application/manage.py collectstatic --noinput
```

---

## 4. Manual Deployment with Gunicorn & Nginx

```bash
# 1. Install dependencies
pip install -r Module_6_Django_Application/requirements.txt
pip install gunicorn psycopg2-binary

# 2. Run migrations and collectstatic
python Module_6_Django_Application/manage.py migrate
python Module_6_Django_Application/manage.py collectstatic --noinput

# 3. Start Gunicorn
gunicorn --chdir Module_6_Django_Application config.wsgi:application \
  --bind 127.0.0.1:8000 \
  --workers 3 \
  --timeout 120
```

---

## 5. Backup & Restore Procedures

### Database Backup
```bash
# PostgreSQL backup
pg_dump -U tedmind_user -d tedmind_db > backups/tedmind_backup_$(date +%F).sql

# SQLite backup
cp Module_6_Django_Application/db.sqlite3 backups/db_$(date +%F).sqlite3
```

### Database Restore
```bash
# PostgreSQL restore
psql -U tedmind_user -d tedmind_db < backups/tedmind_backup_2026-11-17.sql

# SQLite restore
cp backups/db_2026-11-17.sqlite3 Module_6_Django_Application/db.sqlite3
```
