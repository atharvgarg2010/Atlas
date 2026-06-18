#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deployment/scripts/healthcheck.sh
# Project Atlas — System Health Check
#
# Verifies:
#   1. PostgreSQL container is reachable and accepting connections
#   2. Python can import core modules without errors
#   3. Database ping succeeds via SQLAlchemy
#
# Usage:
#   bash deployment/scripts/healthcheck.sh
#
# Exit codes:
#   0 — all checks passed
#   1 — one or more checks failed
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PASS="✓"
FAIL="✗"
EXIT_CODE=0

echo "──────────────────────────────────────────────"
echo "  Project Atlas — Health Check"
echo "──────────────────────────────────────────────"

# ── Check 1: Python version ───────────────────────────────────────────────────
PYTHON_VERSION=$(python --version 2>&1)
if python -c "import sys; assert sys.version_info >= (3, 12)" 2>/dev/null; then
    echo "  $PASS  Python      : $PYTHON_VERSION"
else
    echo "  $FAIL  Python      : $PYTHON_VERSION (requires 3.12+)"
    EXIT_CODE=1
fi

# ── Check 2: Core imports ─────────────────────────────────────────────────────
cd "$PROJECT_ROOT"
if python -c "from config import get_settings; from core.logging import get_logger" 2>/dev/null; then
    echo "  $PASS  Imports     : OK"
else
    echo "  $FAIL  Imports     : Failed — run: pip install -r requirements.txt"
    EXIT_CODE=1
fi

# ── Check 3: .env file ────────────────────────────────────────────────────────
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "  $PASS  .env file   : Found"
else
    echo "  $FAIL  .env file   : Missing — copy .env.example to .env and fill values"
    EXIT_CODE=1
fi

# ── Check 4: PostgreSQL container ─────────────────────────────────────────────
if docker ps --filter "name=atlas_db" --filter "status=running" --format "{{.Names}}" \
   | grep -q "atlas_db" 2>/dev/null; then
    echo "  $PASS  DB container: Running"
else
    echo "  $FAIL  DB container: Not running — run: docker-compose -f deployment/docker-compose.yml up -d db"
    EXIT_CODE=1
fi

# ── Check 5: Database connectivity ───────────────────────────────────────────
if python -c "
from config import get_settings
from database.connection import init_db
s = get_settings()
db = init_db(s.database_url)
db.ping()
" 2>/dev/null; then
    echo "  $PASS  Database    : Connection OK"
else
    echo "  $FAIL  Database    : Cannot connect — check DATABASE_URL in .env"
    EXIT_CODE=1
fi

echo "──────────────────────────────────────────────"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  RESULT: ALL CHECKS PASSED"
else
    echo "  RESULT: SOME CHECKS FAILED (see above)"
fi
echo "──────────────────────────────────────────────"
exit $EXIT_CODE
