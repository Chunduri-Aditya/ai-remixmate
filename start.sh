#!/usr/bin/env bash
# start.sh — Install/update dependencies and start AI RemixMate.
#
# Usage:
#   ./start.sh              # setup + React UI + API (default)
#   ./start.sh frontend     # same as default
#   ./start.sh setup        # install/update packages only
#   ./start.sh api          # API only
#   ./start.sh stop         # kill everything
#   ./start.sh --skip-setup # start without reinstalling packages
#   ./start.sh --no-open    # don't auto-open a browser tab (default: opens one)
#   ./start.sh --host ADDR  # API bind address (default: 127.0.0.1)
#   ./start.sh --dev        # run the API with --reload
#   (ui and both were the Streamlit UI modes; they now exit 2)

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
API_PORT=8000
FRONTEND_PORT=5173
CERT_DIR="$ROOT/certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

API_HOST="127.0.0.1"

HTTPS_MODE=false
SKIP_SETUP=false
OPEN_BROWSER=true
DEV_MODE=false
PY="$ROOT/.venv/bin/python"
VENV_DIR="$ROOT/.venv"
PY_CHECKED=false

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Resolve the Python interpreter RemixMate should use.
# Homebrew/macOS Python 3.12+ is PEP-668 "externally managed" — pip install
# only works inside a venv. We always use $ROOT/.venv and create it when needed.
ensure_python_env() {
  if $PY_CHECKED; then
    return
  fi
  PY_CHECKED=true

  # 1. Project venv from a previous run
  if [ -x "$PY" ]; then
    echo "  Using project venv: $PY ($($PY --version 2>&1))"
    return
  fi

  # 2. Create .venv/ — Python 3.12 (pyproject requires >=3.12; torch pins rule out 3.14)
  local BASE=""
  for candidate in python3.12 python3.13 python3; do
    if command -v "$candidate" &>/dev/null; then
      BASE="$(command -v "$candidate")"
      break
    fi
  done
  if [ -z "$BASE" ]; then
    echo "❌  Python 3 not found. Install Python 3.12 and retry."
    echo "    macOS:  brew install python@3.12"
    echo "    See PREREQUISITES.md"
    exit 1
  fi

  echo "  Creating virtual environment .venv/ (PEP 668 safe)…"
  echo "  Base interpreter: $BASE ($($BASE --version 2>&1))"
  "$BASE" -m venv "$VENV_DIR"
  echo "  ✅  Created $PY ($($PY --version 2>&1))"
}

# Wait until $1 (a URL) responds, up to $2 seconds (default 25). Returns
# non-zero on timeout instead of hanging forever if the server never comes up.
wait_for_http() {
  local url=$1
  local timeout=${2:-25}
  if ! command -v curl &>/dev/null; then
    echo "  ⚠️   curl not found — cannot poll $url for readiness."
    return 1
  fi
  local waited=0
  while ! curl -sk -o /dev/null --max-time 1 "$url" 2>/dev/null; do
    sleep 0.5
    waited=$((waited + 1))
    if [ "$waited" -ge $((timeout * 2)) ]; then
      return 1
    fi
  done
  return 0
}

# Auto-open $1 in the default browser — macOS `open`, Linux `xdg-open`/`wslview`.
# Respects --no-open. Never fails the script if no opener is found.
open_browser() {
  local url=$1
  if ! $OPEN_BROWSER; then
    return 0
  fi
  if command -v open &>/dev/null; then
    open "$url" &>/dev/null &
  elif command -v xdg-open &>/dev/null; then
    xdg-open "$url" &>/dev/null &
  elif command -v wslview &>/dev/null; then
    wslview "$url" &>/dev/null &
  else
    echo "  ℹ️   Couldn't auto-detect a browser opener — open $url manually."
    return 1
  fi
  echo "  🌐  Opened $url in your browser."
}

kill_port() {
  local port=$1
  local pids
  pids=$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "  Stopping listener(s) on port $port: $pids"
    kill $pids 2>/dev/null || true
    local waited=0
    while [ -n "$pids" ] && [ "$waited" -lt 10 ]; do
      sleep 0.5
      waited=$((waited + 1))
      pids=$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)
    done
    if [ -n "$pids" ]; then
      echo "  Port $port still busy after 5s, sending KILL: $pids"
      kill -9 $pids 2>/dev/null || true
      sleep 0.5
    fi
  else
    echo "  Port $port is free."
  fi
}

banner() {
  echo ""
  echo "╔══════════════════════════════════════╗"
  if $HTTPS_MODE; then
    echo "║   AI RemixMate — HTTPS Mode 🔒       ║"
  else
    echo "║      AI RemixMate — Starting up      ║"
  fi
  echo "╚══════════════════════════════════════╝"
  echo ""
}

ensure_certs() {
  if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    echo "🔐  No certificates found — generating..."
    bash "$CERT_DIR/generate.sh"
    echo ""
  else
    echo "🔒  Using existing certificates:"
    echo "    cert: $CERT_FILE"
    echo "    key:  $KEY_FILE"
    echo ""
  fi
}

setup_deps() {
  echo "📦  Installing / updating dependencies..."
  echo ""

  ensure_python_env
  echo ""

  # Warn if Python is very new (torch/demucs may lack wheels yet)
  local PY_MAJOR PY_MINOR
  read -r PY_MAJOR PY_MINOR < <($PY -c 'import sys; print(sys.version_info.major, sys.version_info.minor)')
  if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 13 ]; then
    echo "  ⚠️   Python $PY_MAJOR.$PY_MINOR detected — this project requires 3.12."
    echo "      If pip fails, run:  brew install python@3.12 && rm -rf .venv && ./start.sh"
    echo ""
  fi

  # ── System tools ──────────────────────────────────────────────────────
  echo "🔧  System tools"
  if ! command -v ffmpeg &>/dev/null; then
    echo "❌  ffmpeg not found — required for audio processing."
    echo "    macOS:  brew install ffmpeg"
    echo "    Linux:  sudo apt install ffmpeg"
    exit 1
  fi
  echo "  ✅  ffmpeg"

  if ! command -v node &>/dev/null || ! command -v npm &>/dev/null; then
    echo "❌  Node.js / npm not found — required for the React frontend."
    echo "    macOS:  brew install node"
    echo "    Linux:  sudo apt install nodejs npm"
    exit 1
  fi
  echo "  ✅  node $(node --version) / npm $(npm --version)"
  echo ""

  # ── Python packages ───────────────────────────────────────────────────
  echo "🐍  Python packages"
  cd "$ROOT"
  $PY -m pip install --upgrade pip setuptools wheel -q
  $PY -m pip install -r requirements.txt -q
  $PY -m pip install -e ".[dev]" -q
  echo "  ✅  requirements.txt + editable install"
  echo ""

  # ── Frontend packages ─────────────────────────────────────────────────
  echo "⚛️   Frontend packages"
  cd "$ROOT/frontend"
  if [ -f package-lock.json ]; then
    npm ci --silent 2>/dev/null || npm install --silent
  else
    npm install --silent
  fi
  echo "  ✅  npm packages installed"
  echo ""

  cd "$ROOT"
  echo "✅  Dependency setup complete."
  echo ""
}

run_check() {
  : # bin/check.sh was archived to archive/bin; start_api waits on /health/live instead
}

stop_all() {
  echo "⏹  Stopping all RemixMate processes..."
  kill_port $API_PORT
  kill_port $FRONTEND_PORT
  echo "Done."
}

start_frontend() {
  echo "⚛️   Clearing port $FRONTEND_PORT..."
  kill_port $FRONTEND_PORT
  LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')

  echo "🚀  Starting React frontend at http://localhost:$FRONTEND_PORT"
  [ -n "$LAN_IP" ] && echo "    LAN URL: http://$LAN_IP:$FRONTEND_PORT"
  (cd "$ROOT/frontend" && exec npm run dev) &
  FRONTEND_PID=$!
  echo "    PID: $FRONTEND_PID"
}

start_api() {
  echo "🔧  Clearing port $API_PORT..."
  kill_port $API_PORT

  ensure_python_env

  # --reload with no --reload-dir watches the ENTIRE project root by
  # default — including library/, outputs/, and data/, which a download or
  # remix job writes to continuously (new WAV/FLAC stems, rendered mixes,
  # jobs.db, music_index.json). Every one of those writes looked like a
  # source-code change to watchfiles, so uvicorn restarted the whole server
  # mid-job — killing the ThreadPoolExecutor worker that was rendering it.
  # That's what a "stuck at 90% forever" / "557s elapsed" job actually was:
  # not a slow render, but the server pulling the rug out from under it.
  # Restricting the watch to scripts/ (the only place app code lives) fixes
  # it: code edits still hot-reload, audio/job-store writes no longer do.
  # Reload is opt in via --dev.
  RELOAD_ARGS=()
  if $DEV_MODE; then
    RELOAD_ARGS=(--reload --reload-dir "$ROOT/scripts")
  fi

  local scheme=http
  if $HTTPS_MODE; then
    scheme=https
    echo "🚀  Starting API at https://$API_HOST:$API_PORT (HTTPS)"
    echo "    Docs: https://$API_HOST:$API_PORT/docs"
    (cd "$ROOT" && exec $PY -m uvicorn scripts.api.main:app "${RELOAD_ARGS[@]}" --timeout-graceful-shutdown 30 --no-proxy-headers --host "$API_HOST" --port $API_PORT \
      --ssl-certfile "$CERT_FILE" --ssl-keyfile "$KEY_FILE") &
  else
    echo "🚀  Starting API at http://$API_HOST:$API_PORT"
    echo "    Docs: http://$API_HOST:$API_PORT/docs"
    (cd "$ROOT" && exec $PY -m uvicorn scripts.api.main:app "${RELOAD_ARGS[@]}" --timeout-graceful-shutdown 30 --no-proxy-headers --host "$API_HOST" --port $API_PORT) &
  fi
  API_PID=$!
  echo "    PID: $API_PID"

  if ! wait_for_http "$scheme://$API_HOST:$API_PORT/health/live" 25; then
    echo "  ⚠️   API didn't answer /health/live within 25s."
  fi
}

print_frontend_urls() {
  LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
  echo ""
  echo "✅  RemixMate running (React frontend):"
  echo ""
  echo "  ⚛️   React UI             →  http://localhost:$FRONTEND_PORT"
  echo "  🎧  DJ Widget (float)    →  http://localhost:$FRONTEND_PORT/widget"
  echo "  🛠️   Operations           →  http://localhost:$FRONTEND_PORT/operations"
  [ -n "$LAN_IP" ] && echo "  📱  Phone / LAN          →  http://$LAN_IP:$FRONTEND_PORT"
  echo "  📖  API Docs             →  http://localhost:$API_PORT/docs"
  echo "  📡  SSE stream           →  http://localhost:$API_PORT/events/stream"
  echo ""
  echo "Press Ctrl+C to stop."
}

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------

MODE="frontend"   # default: React UI + API
while [ $# -gt 0 ]; do
  arg="$1"
  case "$arg" in
    --https|-s|--secure)
      HTTPS_MODE=true
      ;;
    --no-https|--http)
      HTTPS_MODE=false
      ;;
    --skip-setup)
      SKIP_SETUP=true
      ;;
    --no-open|--no-browser)
      OPEN_BROWSER=false
      ;;
    --dev)
      DEV_MODE=true
      ;;
    --host)
      if [ -z "${2:-}" ]; then
        echo "--host needs an address"
        exit 1
      fi
      API_HOST="$2"
      shift
      ;;
    --host=*)
      API_HOST="${arg#--host=}"
      ;;
    stop|api|ui|both|frontend|setup)
      MODE="$arg"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [setup|frontend|api|stop] [--https|--skip-setup|--no-open|--dev|--host ADDR]"
      exit 1
      ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

case "$MODE" in
  ui|both)
    echo "Streamlit UI was removed. Run ./start.sh for the React UI + API."
    exit 2
    ;;
esac

banner

case "$MODE" in
  stop)
    stop_all
    exit 0
    ;;
  setup)
    ensure_python_env
    setup_deps
    run_check
    exit 0
    ;;
esac

# Always resolve Python (creates .venv/ on first run)
ensure_python_env

if ! $SKIP_SETUP; then
  setup_deps
  run_check
fi

if $HTTPS_MODE; then
  ensure_certs
fi

trap 'kill $API_PID $FRONTEND_PID 2>/dev/null' EXIT INT TERM

case "$MODE" in
  api)
    start_api
    echo ""
    echo "✅  API running. Press Ctrl+C to stop."
    wait $API_PID
    ;;
  frontend|"")
    start_api
    start_frontend
    print_frontend_urls
    FRONTEND_URL="http://localhost:$FRONTEND_PORT"
    if wait_for_http "$FRONTEND_URL" 25; then
      open_browser "$FRONTEND_URL"
    else
      echo "  ⚠️   Frontend didn't respond within 25s — open $FRONTEND_URL manually once it's up."
    fi
    wait
    ;;
  *)
    echo "Usage: $0 [setup|frontend|api|stop] [--https|--skip-setup|--no-open|--dev|--host ADDR]"
    exit 1
    ;;
esac
