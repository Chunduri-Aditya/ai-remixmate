#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  run_overnight.sh — Sleep-proof pipeline runner                      ║
# ║                                                                      ║
# ║  Wraps the pipeline with caffeinate (macOS) to prevent your Mac     ║
# ║  from sleeping mid-Demucs. Close the lid, walk away, sleep well.    ║
# ║                                                                      ║
# ║  Usage:                                                              ║
# ║    ./run_overnight.sh                                                ║
# ║    ./run_overnight.sh --model htdemucs_ft                           ║
# ║                                                                      ║
# ║  This script:                                                        ║
# ║    1. Starts the API if it's not already running                     ║
# ║    2. Prevents system sleep (caffeinate on macOS)                    ║
# ║    3. Runs the full Demucs → FLAC → Index pipeline                  ║
# ║    4. Logs everything to pipeline.log                                ║
# ║    5. Plays a sound when done (macOS)                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   AI RemixMate — Overnight Pipeline 🌙       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Step 1: Make sure API is running ──────────────────────────────────────
API_CHECK=$(curl -s --max-time 3 http://localhost:8000/health 2>/dev/null || echo "")
if [ -z "$API_CHECK" ]; then
  echo "📡 API not running — starting it..."
  bash start.sh api &
  sleep 4
  echo ""
fi

# ── Step 2: Prevent sleep ─────────────────────────────────────────────────
if command -v caffeinate &>/dev/null; then
  echo "☕ caffeinate enabled — your Mac won't sleep during processing."
  echo ""
  # -i = prevent idle sleep, -s = prevent system sleep (even with lid closed on AC power)
  caffeinate -is bash "$DIR/run_pipeline.sh" "$@"
  EXIT_CODE=$?
else
  echo "⚠  caffeinate not available (not macOS?) — running without sleep prevention."
  echo "   Make sure your computer is set to never sleep."
  echo ""
  bash "$DIR/run_pipeline.sh" "$@"
  EXIT_CODE=$?
fi

# ── Step 3: Done — notify ────────────────────────────────────────────────
if [ $EXIT_CODE -eq 0 ]; then
  # Play completion sound on macOS
  if command -v afplay &>/dev/null; then
    afplay /System/Library/Sounds/Glass.aiff 2>/dev/null &
  fi
  # macOS notification
  if command -v osascript &>/dev/null; then
    osascript -e 'display notification "All stems split, compressed, and indexed!" with title "AI RemixMate" subtitle "Pipeline Complete ✅" sound name "Glass"' 2>/dev/null || true
  fi
fi

exit $EXIT_CODE
