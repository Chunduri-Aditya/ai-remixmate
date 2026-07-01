#!/usr/bin/env bash
# start.sh — Install/update dependencies and start AI RemixMate.
#
# Usage:
#   ./start.sh              # setup + React UI + API (default)
#   ./start.sh frontend     # same as default
#   ./start.sh setup        # install/update packages only
#   ./start.sh api          # API only
#   ./start.sh ui           # legacy Streamlit UI + API
#   ./start.sh stop         # kill everything
#   ./start.sh --skip-setup # start without reinstalling packages
#   ./start.sh --no-open    # don't auto-open a browser tab (default: opens one)

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
API_PORT=8000
UI_PORT=8501
FRONTEND_PORT=5173
CERT_DIR="$ROOT/certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

HTTPS_MODE=false
SKIP_SETUP=false
OPEN_BROWSER=true
PY=""   # absolute path — set by ensure_python_env()
VENV_DIR="$ROOT/remix-env"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Resolve the Python interpreter RemixMate should use.
# Homebrew/macOS Python 3.12+ is PEP-668 "externally managed" — pip install
# only works inside a venv. We auto-create remix-env/ when needed.
ensure_python_env() {
  if [ -n "$PY" ] && [ -x "$PY" ]; then
    return
  fi

  # 1. Already inside an activated venv
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY="$VIRTUAL_ENV/bin/python"
    echo "  Using active venv: $PY ($($PY --version 2>&1))"
    return
  fi

  # 2. Project venv from a previous run
  if [ -x "$VENV_DIR/bin/python" ]; then
    PY="$VENV_DIR/bin/python"
    echo "  Using project venv: $PY ($($PY --version 2>&1))"
    return
  fi

  # 3. Create remix-env/ — prefer 3.11/3.12 over Homebrew 3.14 (torch compat)
  local BASE=""
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
      BASE="$(command -v "$candidate")"
      break
    fi
  done
  if [ -z "$BASE" ]; then
    echo "❌  Python 3 not found. Install Python 3.10–3.12 and retry."
    echo "    macOS:  brew install python@3.12"
    echo "    See PREREQUISITES.md"
    exit 1
  fi

  echo "  Creating virtual environment remix-env/ (PEP 668 safe)…"
  echo "  Base interpreter: $BASE ($($BASE --version 2>&1))"
  "$BASE" -m venv "$VENV_DIR"
  PY="$VENV_DIR/bin/python"
  echo "  ✅  Created $PY ($($PY --version 2>&1))"
  echo ""
  echo "  Tip: next time run  source remix-env/bin/activate  before ./start.sh"
}

# Wait until $1 (a URL) responds, up to $2 seconds (default 25). Returns
# non-zero on timeout instead of hanging forever if the server never comes up.
wait_for_http() {
  local url=$1
  local timeout=${2:-25}
  if ! command -v curl &>/dev/null; then
    sleep 2   # best effort — no curl to poll with
    return 0
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
  pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "  Killing process(es) on port $port: $pids"
    echo "$pids" | xargs kill -9 2>/dev/null || true
    sleep 0.5
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
    echo "  ⚠️   Python $PY_MAJOR.$PY_MINOR detected — some ML packages need 3.10–3.12."
    echo "      If pip fails, run:  brew install python@3.12 && rm -rf remix-env && ./start.sh"
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
  $PY -m pip install -U yt-dlp -q
  echo "  ✅  requirements.txt + editable install + yt-dlp"
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
  if [ -f "$ROOT/bin/check.sh" ]; then
    echo "🧪  Running readiness check..."
    bash "$ROOT/bin/check.sh" || true
    echo ""
  fi
}

stop_all() {
  echo "⏹  Stopping all RemixMate processes..."
  kill_port $API_PORT
  kill_port $UI_PORT
  kill_port $FRONTEND_PORT
  echo "Done."
}

start_frontend() {
  echo "⚛️   Clearing port $FRONTEND_PORT..."
  kill_port $FRONTEND_PORT
  LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')

  echo "🚀  Starting React frontend at http://localhost:$FRONTEND_PORT"
  [ -n "$LAN_IP" ] && echo "    LAN URL: http://$LAN_IP:$FRONTEND_PORT"
  (cd "$ROOT/frontend" && npm run dev) &
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
  RELOAD_ARGS=(--reload --reload-dir "$ROOT/scripts")

  if $HTTPS_MODE; then
    echo "🚀  Starting API at https://0.0.0.0:$API_PORT (HTTPS)"
    echo "    Docs: https://localhost:$API_PORT/docs"
    $PY -m uvicorn scripts.api.main:app "${RELOAD_ARGS[@]}" --host 0.0.0.0 --port $API_PORT \
      --ssl-certfile "$CERT_FILE" --ssl-keyfile "$KEY_FILE" &
  else
    echo "🚀  Starting API at http://0.0.0.0:$API_PORT"
    echo "    Docs: http://localhost:$API_PORT/docs"
    (cd "$ROOT" && $PY -m uvicorn scripts.api.main:app "${RELOAD_ARGS[@]}" --host 0.0.0.0 --port $API_PORT) &
  fi
  API_PID=$!
  echo "    PID: $API_PID"
}

start_ui() {
  echo "🎨  Clearing port $UI_PORT..."
  kill_port $UI_PORT
  LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')

  if $HTTPS_MODE; then
    echo "🚀  Starting UI at https://localhost:$UI_PORT (HTTPS)"
    [ -n "$LAN_IP" ] && echo "    Phone URL: https://$LAN_IP:$UI_PORT"
    streamlit run scripts/ui/app.py \
      --server.port $UI_PORT \
      --server.address 0.0.0.0 \
      --server.headless true \
      --server.enableCORS false \
      --server.enableXsrfProtection true \
      --server.maxUploadSize 200 \
      --server.sslCertFile "$CERT_FILE" \
      --server.sslKeyFile "$KEY_FILE" &
  else
    echo "🚀  Starting UI at http://localhost:$UI_PORT"
    [ -n "$LAN_IP" ] && echo "    Phone URL: http://$LAN_IP:$UI_PORT"
    (cd "$ROOT" && streamlit run scripts/ui/app.py \
      --server.port $UI_PORT \
      --server.address 0.0.0.0 \
      --server.headless true \
      --server.enableCORS false \
      --server.enableXsrfProtection false \
      --server.maxUploadSize 200) &
  fi
  UI_PID=$!
  echo "    PID: $UI_PID"
}

print_frontend_urls() {
  LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
  echo ""
  echo "✅  RemixMate running (React frontend):"
  echo ""
  echo "  ⚛️   React UI             →  http://localhost:$FRONTEND_PORT"
  echo "  🎧  DJ Widget (float)    →  http://localhost:$FRONTEND_PORT/widget"
  echo "  ⬇️   Downloads            →  http://localhost:$FRONTEND_PORT/operations"
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
for arg in "$@"; do
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
    stop|api|ui|both|frontend|setup)
      MODE="$arg"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [setup|frontend|api|ui|both|stop] [--https|--skip-setup|--no-open]"
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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

# Always resolve Python (creates remix-env/ on first run)
ensure_python_env

if ! $SKIP_SETUP; then
  setup_deps
  run_check
fi

if $HTTPS_MODE; then
  ensure_certs
fi

case "$MODE" in
  api)
    start_api
    echo ""
    echo "✅  API running. Press Ctrl+C to stop."
    wait $API_PID
    ;;
  ui)
    start_api
    sleep 2
    start_ui
    LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
    echo ""
    echo "✅  RemixMate running (Streamlit legacy UI):"
    echo ""
    if $HTTPS_MODE; then
      echo "  🔒  Streamlit UI         →  https://localhost:$UI_PORT"
      [ -n "$LAN_IP" ] && echo "  📱  Phone / LAN          →  https://$LAN_IP:$UI_PORT"
      echo "  📖  API Docs             →  https://localhost:$API_PORT/docs"
    else
      echo "  🎵  Streamlit UI         →  http://localhost:$UI_PORT"
      [ -n "$LAN_IP" ] && echo "  📱  Phone / LAN          →  http://$LAN_IP:$UI_PORT"
      echo "  📖  API Docs             →  http://localhost:$API_PORT/docs"
    fi
    echo ""
    echo "Tip: run './start.sh' for the new React UI + DJ widget."
    echo "Press Ctrl+C to stop."
    UI_URL="http://localhost:$UI_PORT"
    $HTTPS_MODE && UI_URL="https://localhost:$UI_PORT"
    if wait_for_http "$UI_URL" 25; then
      open_browser "$UI_URL"
    else
      echo "  ⚠️   UI didn't respond within 25s — open $UI_URL manually once it's up."
    fi
    wait
    ;;
  frontend|"")
    start_api
    sleep 2
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
  both)
    start_api
    sleep 2
    start_ui
    LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
    echo ""
    echo "✅  All running:"
    echo ""
    if $HTTPS_MODE; then
      echo "  🔒  Streamlit UI         →  https://localhost:$UI_PORT"
      [ -n "$LAN_IP" ] && echo "  📱  Phone / LAN          →  https://$LAN_IP:$UI_PORT"
      echo "  📖  API Docs             →  https://localhost:$API_PORT/docs"
    else
      echo "  🎵  Streamlit UI (legacy)→  http://localhost:$UI_PORT"
      [ -n "$LAN_IP" ] && echo "  📱  Phone / LAN          →  http://$LAN_IP:$UI_PORT"
      echo "  📖  API Docs             →  http://localhost:$API_PORT/docs"
    fi
    echo ""
    echo "Tip: run './start.sh' for the new React UI + DJ widget."
    echo "Press Ctrl+C to stop."
    UI_URL="http://localhost:$UI_PORT"
    $HTTPS_MODE && UI_URL="https://localhost:$UI_PORT"
    if wait_for_http "$UI_URL" 25; then
      open_browser "$UI_URL"
    else
      echo "  ⚠️   UI didn't respond within 25s — open $UI_URL manually once it's up."
    fi
    wait
    ;;
  *)
    echo "Usage: $0 [setup|frontend|api|ui|both|stop] [--https|--skip-setup|--no-open]"
    exit 1
    ;;
esac
