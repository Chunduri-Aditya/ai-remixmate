#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  run_pipeline.sh — Fire-and-forget Demucs pipeline                  ║
# ║                                                                      ║
# ║  Kicks off the full pipeline (stems → FLAC → index) on your         ║
# ║  entire library, logs everything to a file, and runs until done.    ║
# ║                                                                      ║
# ║  Usage:                                                              ║
# ║    ./run_pipeline.sh                    # default: htdemucs          ║
# ║    ./run_pipeline.sh --model htdemucs_ft   # higher quality model   ║
# ║    ./run_pipeline.sh --no-compress      # skip FLAC compression     ║
# ║    ./run_pipeline.sh --no-enhance       # skip audio enhancement    ║
# ║                                                                      ║
# ║  Check progress anytime:                                             ║
# ║    tail -f pipeline.log                                              ║
# ║                                                                      ║
# ║  When you wake up:                                                   ║
# ║    cat pipeline.log | tail -20                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API="http://localhost:8000"
LOG_FILE="$(cd "$(dirname "$0")" && pwd)/pipeline.log"
POLL_INTERVAL=15    # seconds between progress checks
MODEL="htdemucs"
ENHANCE=true
COMPRESS=true
INDEX=true
DELETE_WAV=true

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
for arg in "$@"; do
  case "$arg" in
    --model)         shift; MODEL="${1:-htdemucs}" ;;
    --model=*)       MODEL="${arg#*=}" ;;
    --no-compress)   COMPRESS=false ;;
    --no-enhance)    ENHANCE=false ;;
    --no-index)      INDEX=false ;;
    --keep-wav)      DELETE_WAV=false ;;
    --help|-h)
      echo "Usage: $0 [--model htdemucs|htdemucs_ft|mdx_extra] [--no-compress] [--no-enhance] [--no-index] [--keep-wav]"
      exit 0
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
  echo "$msg" | tee -a "$LOG_FILE"
}

api_call() {
  curl -s --max-time 10 "$@" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "" > "$LOG_FILE"

log "╔══════════════════════════════════════════════╗"
log "║   AI RemixMate — Demucs Pipeline             ║"
log "╚══════════════════════════════════════════════╝"
log ""
log "Model:       $MODEL"
log "Enhance:     $ENHANCE"
log "Compress:    $COMPRESS"
log "Index:       $INDEX"
log "Delete WAV:  $DELETE_WAV"
log "Log file:    $LOG_FILE"
log ""

# Check API is running
HEALTH=$(api_call "$API/health")
if [ -z "$HEALTH" ]; then
  log "❌ API is not running at $API. Start it first: ./start.sh api"
  log "   Then re-run this script."
  exit 1
fi

SONG_COUNT=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('library_songs', '?'))" 2>/dev/null || echo "?")
log "✅ API online — $SONG_COUNT songs in library"
log ""

# ---------------------------------------------------------------------------
# Kick off the pipeline
# ---------------------------------------------------------------------------
log "🚀 Starting pipeline..."

RESPONSE=$(api_call -X POST "$API/library/initialize" \
  -H "Content-Type: application/json" \
  -d "{
    \"enhance\": $ENHANCE,
    \"model\": \"$MODEL\",
    \"delete_wav\": $DELETE_WAV,
    \"run_compress\": $COMPRESS,
    \"run_index\": $INDEX
  }")

JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id', ''))" 2>/dev/null || echo "")

if [ -z "$JOB_ID" ]; then
  log "❌ Failed to start pipeline. API response:"
  log "   $RESPONSE"
  exit 1
fi

log "✅ Job started: $JOB_ID"
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "   Go to sleep. Check progress anytime:"
log "   tail -f pipeline.log"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log ""

# ---------------------------------------------------------------------------
# Poll until done
# ---------------------------------------------------------------------------
LAST_MSG=""
START_TIME=$(date +%s)

while true; do
  JOB=$(api_call "$API/jobs/$JOB_ID")

  if [ -z "$JOB" ]; then
    log "⚠  Could not reach API — retrying in ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
    continue
  fi

  STATUS=$(echo "$JOB" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
  PROGRESS=$(echo "$JOB" | python3 -c "import sys,json; print(round(json.load(sys.stdin).get('progress', 0) * 100, 1))" 2>/dev/null || echo "0")
  MESSAGE=$(echo "$JOB" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message', ''))" 2>/dev/null || echo "")
  ETA=$(echo "$JOB" | python3 -c "import sys,json; e=json.load(sys.stdin).get('eta_sec'); print(f'{int(e//60)}m {int(e%60)}s' if e else '—')" 2>/dev/null || echo "—")

  # Only log if message changed (avoid spam)
  if [ "$MESSAGE" != "$LAST_MSG" ]; then
    ELAPSED=$(( $(date +%s) - START_TIME ))
    ELAPSED_FMT="$(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m"
    log "[$PROGRESS%] $MESSAGE  (ETA: $ETA · elapsed: $ELAPSED_FMT)"
    LAST_MSG="$MESSAGE"
  fi

  # Check terminal states
  if [ "$STATUS" = "done" ]; then
    ELAPSED=$(( $(date +%s) - START_TIME ))
    ELAPSED_FMT="$(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "✅ PIPELINE COMPLETE"
    log "   Total time: $ELAPSED_FMT"
    log ""

    # Parse results
    echo "$JOB" | python3 -c "
import sys, json
r = json.load(sys.stdin).get('result', {})
print(f\"   Stems split:    {r.get('split_done', '?')}\")
print(f\"   Split failed:   {r.get('split_failed', '?')}\")
print(f\"   FLAC compressed: {r.get('compress_converted', '?')}\")
print(f\"   Songs indexed:  {r.get('total_indexed', '?')}\")
" 2>/dev/null | tee -a "$LOG_FILE"

    log ""
    log "   Good morning! Your library is ready."
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
  fi

  if [ "$STATUS" = "failed" ]; then
    ERROR=$(echo "$JOB" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error', 'unknown error'))" 2>/dev/null || echo "unknown")
    log ""
    log "❌ PIPELINE FAILED"
    log "   Error: $ERROR"
    log "   Job ID: $JOB_ID"
    exit 1
  fi

  sleep "$POLL_INTERVAL"
done
