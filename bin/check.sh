#!/usr/bin/env bash
# check.sh — Validate AI RemixMate is ready to run.
#
# Usage:  bash bin/check.sh

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0
FAIL=0
WARN=0

# Prefer project venv (same as start.sh) over system Python
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY="$VIRTUAL_ENV/bin/python"
elif [ -x "$ROOT/remix-env/bin/python" ]; then
    PY="$ROOT/remix-env/bin/python"
elif command -v python3 &>/dev/null; then
    PY=python3
else
    PY=""
fi

pass() { echo "  ✅  $1"; ((PASS++)); }
fail() { echo "  ❌  $1"; ((FAIL++)); }
warn() { echo "  ⚠️   $1"; ((WARN++)); }

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   AI RemixMate — Readiness Check     ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── 1. Python version ──────────────────────────────────────────────
echo "📦  Python"
if [ -n "$PY" ]; then
    PY_VER=$($PY --version 2>&1)
    pass "$PY_VER"
else
    fail "Python 3 not found"
fi

# ── 2. Core Python packages ────────────────────────────────────────
echo ""
echo "📦  Core Dependencies"
for pkg in streamlit fastapi uvicorn librosa soundfile numpy scipy torch; do
    if $PY -c "import $pkg" 2>/dev/null; then
        pass "$pkg"
    else
        fail "$pkg not installed"
    fi
done

# ── 3. Optional packages ──────────────────────────────────────────
echo ""
echo "📦  Optional Dependencies"
for pkg in demucs yt_dlp ytmusicapi; do
    if $PY -c "import $pkg" 2>/dev/null; then
        pass "$pkg"
    else
        warn "$pkg not installed (some features may be unavailable)"
    fi
done

# ── 4. Node / frontend ───────────────────────────────────────────
echo ""
echo "⚛️   Frontend"
if command -v node &>/dev/null; then
    pass "node $(node --version)"
else
    fail "node not found (required for React UI)"
fi
if command -v npm &>/dev/null; then
    pass "npm $(npm --version)"
else
    fail "npm not found"
fi
if [ -d "$ROOT/frontend/node_modules" ]; then
    pass "frontend/node_modules"
else
    warn "frontend/node_modules missing — run ./start.sh setup"
fi

# ── 5. System tools ───────────────────────────────────────────────
echo ""
echo "🔧  System Tools"
for tool in ffmpeg ffprobe; do
    if command -v $tool &>/dev/null; then
        pass "$tool"
    else
        fail "$tool not found (required for audio processing)"
    fi
done

# ── 6. GPU detection ─────────────────────────────────────────────
echo ""
echo "🖥️   GPU Support"
GPU=$($PY -c "
import torch
if torch.cuda.is_available():
    print('CUDA: ' + torch.cuda.get_device_name(0))
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Apple MPS (Metal)')
else:
    print('CPU only')
" 2>/dev/null || echo "unknown")
pass "$GPU"

# ── 7. Project structure ─────────────────────────────────────────
echo ""
echo "📁  Project Structure"
for f in scripts/api/main.py scripts/api/routes.py scripts/ui/app.py scripts/core/gpu.py \
         scripts/core/dj_engine.py scripts/core/stems.py scripts/download.py \
         frontend/src/pages/Widget.tsx frontend/src/pages/Operations.tsx \
         start.sh bin/check.sh PREREQUISITES.md; do
    if [ -f "$ROOT/$f" ]; then
        pass "$f"
    else
        fail "$f missing"
    fi
done

# ── 8. Config ────────────────────────────────────────────────────
echo ""
echo "⚙️   Configuration"
if [ -f "$ROOT/.streamlit/config.toml" ]; then
    pass ".streamlit/config.toml"
else
    warn ".streamlit/config.toml missing (only needed for Streamlit UI)"
fi
if [ -f "$ROOT/config.yaml" ]; then
    pass "config.yaml"
else
    fail "config.yaml missing"
fi

# ── 9. Library ────────────────────────────────────────────────────
echo ""
echo "🎵  Library"
SONG_COUNT=$(find "$ROOT/library" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
STEM_COUNT=$(find "$ROOT/library" \( -name "vocals.flac" -o -name "vocals.wav" \) 2>/dev/null | wc -l | tr -d ' ')
pass "$SONG_COUNT songs in library"
pass "$STEM_COUNT songs with stems"

# ── 10. Ports ─────────────────────────────────────────────────────
echo ""
echo "🌐  Ports"
for port in 8000 5173 8501; do
    if lsof -ti tcp:$port &>/dev/null; then
        warn "Port $port is in use (stop existing processes: ./start.sh stop)"
    else
        pass "Port $port is free"
    fi
done

# ── 11. Syntax check ────────────────────────────────────────────
echo ""
echo "🧪  Syntax Validation"
for f in scripts/api/routes.py scripts/ui/app.py scripts/api/tasks.py; do
    if $PY -m py_compile "$ROOT/$f" 2>/dev/null; then
        pass "$(basename $f)"
    else
        fail "$f has syntax errors"
    fi
done

# ── Summary ─────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ $PASS passed   ❌ $FAIL failed   ⚠️  $WARN warnings"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "  🚀  Ready to launch!  Run:  ./start.sh"
    echo ""
else
    echo ""
    echo "  ⛔  Fix the failures above, or run:  ./start.sh setup"
    echo ""
    exit 1
fi
