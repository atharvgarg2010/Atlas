# ─────────────────────────────────────────────────────────────────────────────
# Project Atlas — Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build: docker build -f deployment/Dockerfile -t project-atlas .
# Run:   docker-compose -f deployment/docker-compose.yml up
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

# ── System Dependencies ───────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python Dependencies ───────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching.
# Re-run pip install only when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application Source ────────────────────────────────────────────────────────
COPY . .

# ── Logs Directory ────────────────────────────────────────────────────────────
RUN mkdir -p /app/logs

# ── Non-root User ─────────────────────────────────────────────────────────────
# Run as non-root for security hardening
RUN groupadd --system atlas && useradd --system --gid atlas atlas
RUN chown -R atlas:atlas /app
USER atlas

# ── Health Check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from database.connection import get_db; get_db().ping()" \
    || exit 1

# ── Default Command ───────────────────────────────────────────────────────────
CMD ["python", "main.py"]
