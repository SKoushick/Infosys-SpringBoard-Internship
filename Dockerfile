# Multi-stage Dockerfile for TEDMind Production Web Application
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="src;Module_6_Django_Application;Module_6_Django_Application/apps" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY Module_6_Django_Application/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy project code and ML artifacts
COPY src/ src/
COPY Module_3_Feature_Engineering_Vectorization/ Module_3_Feature_Engineering_Vectorization/
COPY Module_4_Recommendation_Engine/ Module_4_Recommendation_Engine/
COPY Module_6_Django_Application/ Module_6_Django_Application/

EXPOSE 8000

CMD ["gunicorn", "--chdir", "Module_6_Django_Application", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "120"]
