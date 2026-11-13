#!/usr/bin/env bash
# start.sh — Install/update dependencies and start AI RemixMate.
#
# Usage:
#   ./start.sh              # React UI + API (default); installs packages first
#                           # when they are missing or the dependency files changed
#   ./start.sh frontend     # same as default
#   ./start.sh setup        # install/update packages only (always reinstalls)
#   ./start.sh api          # API only
#   ./start.sh stop         # stop this app's listeners on the configured ports
#   ./start.sh scan DIR [--stems] [--analyze]
#                           # import the audio files in DIR into library/
#                           # (needs .venv from ./start.sh setup; opens no port)
#   ./start.sh --skip-setup # start without checking or installing packages
#   ./start.sh --no-open    # don't auto-open a browser tab (default: opens one)
#   ./start.sh --host ADDR  # API bind address (default: 127.0.0.1)
#   ./start.sh --dev        # run the API with --reload
#   (ui and both were the Streamlit UI modes; they now exit 2)
#
# Ports (environment variables):
#   REMIXMATE_API_PORT  API port (default 8000)
#   REMIXMATE_UI_PORT   React dev server port (default 5173)
#   Example:  REMIXMATE_API_PORT=8765 REMIXMATE_UI_PORT=5199 ./start.sh
# start.sh stops only listeners that belong to this checkout. When another
# program holds a port it prints who and exits non zero.

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
ROOT_REAL="$(cd "$ROOT" && pwd -P)"
API_PORT="${REMIXMATE_API_PORT:-8000}"
FRONTEND_PORT="${REMIXMATE_UI_PORT:-5173}"
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
STAMP_FILE="$VENV_DIR/.remixmate-setup-stamp"
PY_CHECKED=false

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Print "major minor" for interpreter $1, or nothing when it does not run.
py_version_of() {
  "$1" -c 'import sys; print(sys.version_info[0], sys.version_info[1])' 2>/dev/null
}

# Resolve the Python interpreter RemixMate should use.
# Homebrew/macOS Python 3.12+ is PEP-668 "externally managed" — pip install
# only works inside a venv. We always use $ROOT/.venv and create it when needed.
# The project needs exactly Python 3.12 (torch/demucs pins), so every
# interpreter is probed and only one that reports 3.12 is accepted.
ensure_python_env() {
  if $PY_CHECKED; then
    return
  fi
  PY_CHECKED=true

  # 1. Project venv from a previous run
  if [ -x "$PY" ]; then
    local venv_ver
    venv_ver="$(py_version_of "$PY" || true)"
    if [ "$venv_ver" != "3 12" ]; then
      local shown="${venv_ver/ /.}"
      echo "❌  .venv runs Python ${shown:-of unknown version}, but this project needs Python 3.12."
      echo "    Delete the .venv folder, then run ./start.sh setup (needs python3.12 on PATH)."
      echo "    macOS:  brew install python@3.12    See PREREQUISITES.md"
      exit 1
    fi
    echo "  Using project venv: $PY ($($PY --version 2>&1))"
    return
  fi

  # 2. Create .venv/ from the first base interpreter that reports 3.12
  local BASE="" found="" candidate path ver
  for candidate in python3.12 python3 python; do
    path="$(command -v "$candidate" 2>/dev/null || true)"
    [ -n "$path" ] || continue
    ver="$(py_version_of "$path" || true)"
    found="$found $candidate=${ver/ /.}"
    [ -n "$ver" ] || found="${found}does-not-run"
    if [ "$ver" = "3 12" ]; then
      BASE="$path"
      break
    fi
  done
  if [ -z "$BASE" ]; then
    echo "❌  Python 3.12 not found. This project needs exactly Python 3.12."
    echo "    Found:${found:- no python3.12, python3 or python on PATH}"
    echo "    macOS:  brew install python@3.12"
    echo "    See PREREQUISITES.md"
    exit 1
  fi

  echo "  Creating virtual environment .venv/ (PEP 668 safe)…"
  echo "  Base interpreter: $BASE ($($BASE --version 2>&1))"
  if ! "$BASE" -m venv "$VENV_DIR"; then
    echo "❌  Could not create .venv with $BASE."
    echo "    Debian/Ubuntu:  sudo apt install python3.12-venv, delete .venv, then run ./start.sh setup"
    exit 1
  fi
  echo "  ✅  Created $PY ($($PY --version 2>&1))"
}

# 0 when $1 is a TCP port number (digits only, 1 to 65535).
valid_port() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
  esac
  [ "${#1}" -le 5 ] && [ "$1" -ge 1 ] && [ "$1" -le 65535 ]
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

# Print the working directory of pid $1, or nothing when lsof cannot read it.
listener_cwd() {
  lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1 || true
}

# 0 only when pid $1 belongs to this checkout: its cwd is $ROOT_REAL or below
# it, and its command is the API (uvicorn scripts.api.main, or the
# multiprocessing child that --reload spawns) or Vite. An unknown cwd is foreign.
listener_is_ours() {
  local pid=$1 cwd cmd
  cwd="$(listener_cwd "$pid")"
  [ -n "$cwd" ] || return 1
  case "$cwd" in
    "$ROOT_REAL"|"$ROOT_REAL"/*) ;;
    *) return 1 ;;
  esac
  cmd="$(ps -o command= -p "$pid" 2>/dev/null || true)"
  case "$cmd" in
    *scripts.api.main*|*multiprocessing*|*vite*) return 0 ;;
  esac
  return 1
}

# Free port $1 when only this app listens on it. When another program holds
# it, print who and how to pick another port via env var $2, signal nothing
# and return 1.
kill_port() {
  local port=$1
  local envvar=$2
  local pids pid ours foreign=false
  pids=$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [ -z "$pids" ]; then
    echo "  Port $port is free."
    return 0
  fi
  for pid in $pids; do
    if ! listener_is_ours "$pid"; then
      foreign=true
      echo "❌  Port $port is held by another program (not this checkout of RemixMate):"
      echo "    pid:     $pid"
      echo "    command: $(ps -o command= -p "$pid" 2>/dev/null || true)"
      echo "    cwd:     $(listener_cwd "$pid")"
    fi
  done
  if $foreign; then
    echo "    Left it running. Stop that program, or pick another port:"
    echo "      $envvar=<other port> ./start.sh"
    return 1
  fi
  echo "  Stopping this app's listener(s) on port $port:" $pids
  kill $pids 2>/dev/null || true
  local waited=0
  while [ -n "$pids" ] && [ "$waited" -lt 10 ]; do
    sleep 0.5
    waited=$((waited + 1))
    pids=$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)
  done
  if [ -n "$pids" ]; then
    ours=""
    for pid in $pids; do
      if listener_is_ours "$pid"; then
        ours="$ours $pid"
      fi
    done
    if [ -n "$ours" ]; then
      echo "  Port $port still busy after 5s, sending KILL:$ours"
      kill -9 $ours 2>/dev/null || true
      sleep 0.5
    fi
  fi
  return 0
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

  if ! $PY -m pip --version &>/dev/null; then
    echo "❌  .venv has no pip."
    echo "    Debian/Ubuntu:  sudo apt install python3.12-venv, delete .venv, then run ./start.sh setup"
    exit 1
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
  $PY -m pip install -r requirements-dev.txt -q
  $PY -m pip install -e ".[dev]" -q
  echo "  ✅  requirements-dev.txt (includes requirements.txt) + editable install"
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
  # Last step: a setup that failed above leaves no stamp, so the next run installs again.
  deps_fingerprint > "$STAMP_FILE"
}

# Checksums of the dependency files; a change to any of them means reinstall.
deps_fingerprint() {
  local f
  for f in requirements.txt requirements-dev.txt pyproject.toml frontend/package.json frontend/package-lock.json; do
    if [ -f "$ROOT/$f" ]; then
      printf '%s ' "$f"
      cksum < "$ROOT/$f"
    fi
  done
}

# 0 when the last complete setup matches the dependency files and node_modules exists.
deps_current() {
  [ -f "$STAMP_FILE" ] || return 1
  [ -d "$ROOT/frontend/node_modules" ] || return 1
  [ "$(cat "$STAMP_FILE")" = "$(deps_fingerprint)" ]
}

# Scan mode: import a folder with the project venv. It never builds a venv,
# prints no banner and never touches a port. A folder argument becomes an
# absolute path before the cd to $ROOT, so a relative path keeps its meaning.
run_scan() {
  [ -x "$PY" ] || { echo "No .venv yet. Run ./start.sh setup first."; exit 1; }
  local a args=()
  for a in "$@"; do
    if [ -d "$a" ]; then
      a="$(CDPATH='' cd -- "$a" && pwd)"
    fi
    args+=("$a")
  done
  cd "$ROOT" && exec "$PY" -m scripts.scan "${args[@]}"
}

run_check() {
  : # bin/check.sh was archived to archive/bin; start_api waits on /health/live instead
}

stop_all() {
  echo "⏹  Stopping RemixMate listeners on API port $API_PORT and UI port $FRONTEND_PORT..."
  echo "    (set REMIXMATE_API_PORT / REMIXMATE_UI_PORT to the ports you started with)"
  kill_port "$API_PORT" REMIXMATE_API_PORT || true
  kill_port "$FRONTEND_PORT" REMIXMATE_UI_PORT || true
  echo "Done."
}

start_frontend() {
  LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')

  echo "🚀  Starting React frontend at http://localhost:$FRONTEND_PORT"
  [ -n "$LAN_IP" ] && echo "    LAN URL: http://$LAN_IP:$FRONTEND_PORT"
  (cd "$ROOT/frontend" && REMIXMATE_API_PORT="$API_PORT" REMIXMATE_UI_PORT="$FRONTEND_PORT" exec npm run dev -- --strictPort) &
  FRONTEND_PID=$!
  echo "    PID: $FRONTEND_PID"
}

start_api() {
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

  # The API builds its dev CORS origins from RMX_UI_PORT. The name sits outside
  # REMIXMATE_ because scripts/core/config.py parses every REMIXMATE_ name as a key.
  local scheme=http
  if $HTTPS_MODE; then
    scheme=https
    echo "🚀  Starting API at https://$API_HOST:$API_PORT (HTTPS)"
    echo "    Docs: https://$API_HOST:$API_PORT/docs"
    (cd "$ROOT" && unset REMIXMATE_UI_PORT && export RMX_UI_PORT="$FRONTEND_PORT" && exec $PY -m uvicorn scripts.api.main:app "${RELOAD_ARGS[@]}" --timeout-graceful-shutdown 30 --no-proxy-headers --host "$API_HOST" --port $API_PORT \
      --ssl-certfile "$CERT_FILE" --ssl-keyfile "$KEY_FILE") &
  else
    echo "🚀  Starting API at http://$API_HOST:$API_PORT"
    echo "    Docs: http://$API_HOST:$API_PORT/docs"
    (cd "$ROOT" && unset REMIXMATE_UI_PORT && export RMX_UI_PORT="$FRONTEND_PORT" && exec $PY -m uvicorn scripts.api.main:app "${RELOAD_ARGS[@]}" --timeout-graceful-shutdown 30 --no-proxy-headers --host "$API_HOST" --port $API_PORT) &
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
    scan)
      shift
      run_scan "$@"
      ;;
    stop|api|ui|both|frontend|setup)
      MODE="$arg"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [setup|frontend|api|stop] [--https|--skip-setup|--no-open|--dev|--host ADDR] | $0 scan DIR [--stems] [--analyze]"
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

for port_spec in "REMIXMATE_API_PORT=$API_PORT" "REMIXMATE_UI_PORT=$FRONTEND_PORT"; do
  if ! valid_port "${port_spec#*=}"; then
    echo "❌  ${port_spec%%=*} must be a port number from 1 to 65535, got '${port_spec#*=}'."
    exit 1
  fi
done
API_PORT=$((10#$API_PORT))
FRONTEND_PORT=$((10#$FRONTEND_PORT))
if [ "$API_PORT" -eq "$FRONTEND_PORT" ]; then
  echo "❌  REMIXMATE_API_PORT and REMIXMATE_UI_PORT are both $API_PORT; pick two different ports."
  exit 1
fi

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

# Port preflight before any install: a port held by another program fails
# in seconds, not after the install.
case "$MODE" in
  api)
    kill_port "$API_PORT" REMIXMATE_API_PORT || exit 1
    ;;
  frontend|"")
    kill_port "$API_PORT" REMIXMATE_API_PORT || exit 1
    kill_port "$FRONTEND_PORT" REMIXMATE_UI_PORT || exit 1
    ;;
esac

# Always resolve Python (creates .venv/ on first run)
ensure_python_env

if $SKIP_SETUP; then
  :
elif deps_current; then
  echo "📦  Dependencies up to date. Run ./start.sh setup to reinstall."
else
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
    echo "Usage: $0 [setup|frontend|api|stop] [--https|--skip-setup|--no-open|--dev|--host ADDR] | $0 scan DIR [--stems] [--analyze]"
    exit 1
    ;;
esac
