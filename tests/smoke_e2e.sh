#!/usr/bin/env bash
# Usage: bash tests/smoke_e2e.sh
# Requires: API running at localhost:8000

set -euo pipefail
BASE="http://localhost:8000"

echo "--- health ---"
curl -sf "$BASE/health/live" | python3 -m json.tool

echo "--- library ---"
curl -sf "$BASE/library" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} songs')"

echo "--- jobs ---"
curl -sf "$BASE/jobs" | python3 -m json.tool

echo "--- SSE (3 frames) ---"
if command -v timeout >/dev/null 2>&1; then
  timeout 5 curl -sN "$BASE/events/stream" | head -6 || true
else
  curl --max-time 5 -sN "$BASE/events/stream" | head -6 || true
fi

echo "✓ smoke passed"
