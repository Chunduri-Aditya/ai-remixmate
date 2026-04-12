#!/usr/bin/env bash
# start.sh — Start AI RemixMate (API + UI), killing any stale processes first.
#
# Usage:
#   ./start.sh              # start both API and Streamlit UI (HTTP, legacy)
#   ./start.sh frontend     # start API + React/Vite frontend (new primary UI)
#   ./start.sh --https      # start both with HTTPS (generates certs if needed)
#   ./start.sh api          # start API only
#   ./start.sh ui           # start Streamlit UI only
#   ./start.sh stop         # kill everything

set -e

API_PORT=8000
UI_PORT=8501
FRONTEND_PORT=5173          # Vite dev server (React frontend)
CERT_DIR="$(cd "$(dirname "$0")" && pwd)/certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

# HTTP by default — use ./start.sh --https to enable HTTPS
HTTPS_MODE=false

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

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

  FRONTEND_DIR="$(cd "$(dirname "$0")" && pwd)/frontend"
  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "📦  Installing frontend dependencies (first run)..."
    (cd "$FRONTEND_DIR" && npm install)
    echo ""
  fi

  echo "🚀  Starting React frontend at http://localhost:$FRONTEND_PORT"
  [ -n "$LAN_IP" ] && echo "    LAN URL: http://$LAN_IP:$FRONTEND_PORT"
  (cd "$FRONTEND_DIR" && npm run dev) &
  FRONTEND_PID=$!
  echo "    PID: $FRONTEND_PID"
}

start_api() {
  echo "🔧  Clearing port $API_PORT..."
  kill_port $API_PORT

  if $HTTPS_MODE; then
    echo "🚀  Starting API at https://0.0.0.0:$API_PORT (HTTPS)"
    echo "    Docs: https://localhost:$API_PORT/docs"
    uvicorn scripts.api.main:app --reload --host 0.0.0.0 --port $API_PORT \
      --ssl-certfile "$CERT_FILE" --ssl-keyfile "$KEY_FILE" &
  else
    echo "🚀  Starting API at http://0.0.0.0:$API_PORT"
    echo "    Docs: http://localhost:$API_PORT/docs"
    uvicorn scripts.api.main:app --reload --host 0.0.0.0 --port $API_PORT &
  fi
  API_PID=$!
  echo "    PID: $API_PID"
}

start_ui() {
  echo "🎨  Clearing port $UI_PORT..."
  kill_port $UI_PORT
  # Detect LAN IP for the network link hint
  LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')

  if $HTTPS_MODE; then
    local PROTO="https"
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
    streamlit run scripts/ui/app.py \
      --server.port $UI_PORT \
      --server.address 0.0.0.0 \
      --server.headless true \
      --server.enableCORS false \
      --server.enableXsrfProtection false \
      --server.maxUploadSize 200 &
  fi
  UI_PID=$!
  echo "    PID: $UI_PID"
}

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------

MODE="both"
for arg in "$@"; do
  case "$arg" in
    --https|-s|--secure)
      HTTPS_MODE=true
      ;;
    --no-https|--http)
      HTTPS_MODE=false
      ;;
    stop|api|ui|both|frontend)
      MODE="$arg"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [api|ui|frontend|stop|both] [--https|--no-https]"
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

banner

# Generate certs if HTTPS mode
if $HTTPS_MODE; then
  ensure_certs
fi

PROTO="http"
$HTTPS_MODE && PROTO="https"

case "$MODE" in
  stop)
    stop_all
    exit 0
    ;;
  api)
    start_api
    echo ""
    echo "✅  API running. Press Ctrl+C to stop."
    wait $API_PID
    ;;
  ui)
    start_ui
    echo ""
    echo "✅  UI running. Press Ctrl+C to stop."
    wait $UI_PID
    ;;
  frontend)
    # New primary UI: React + Vite on :5173 alongside FastAPI on :8000
    start_api
    sleep 2
    start_frontend
    LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
    echo ""
    echo "✅  RemixMate running (React frontend mode):"
    echo ""
    echo "  ⚛️   React UI (primary)     →  http://localhost:$FRONTEND_PORT"
    [ -n "$LAN_IP" ] && echo "  📱  Phone / LAN          →  http://$LAN_IP:$FRONTEND_PORT"
    echo "  📖  API Docs             →  http://localhost:$API_PORT/docs"
    echo "  📡  SSE stream           →  http://localhost:$API_PORT/events/stream"
    echo ""
    echo "Press Ctrl+C to stop."
    wait
    ;;
  both|"")
    start_api
    sleep 2   # give API a moment to bind before UI tries to connect
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
    echo "Tip: run './start.sh frontend' to use the new React UI."
    echo "Press Ctrl+C to stop."
    # Wait for either process to exit
    wait
    ;;
  *)
    echo "Usage: $0 [api|ui|frontend|stop|both] [--https]"
    exit 1
    ;;
esac
