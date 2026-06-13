#!/usr/bin/env bash
# check.sh — Validate AI RemixMate is ready to run.
#
# Usage:  bash check.sh

set -e

PASS=0
FAIL=0
WARN=0

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
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1)
    pass "$PY_VER"
else
    fail "Python 3 not found"
fi

# ── 2. Core Python packages ────────────────────────────────────────
echo ""
echo "📦  Core Dependencies"
for pkg in streamlit fastapi uvicorn librosa soundfile numpy scipy torch; do
    if python3 -c "import $pkg" 2>/dev/null; then
        pass "$pkg"
    else
        fail "$pkg not installed"
    fi
done

# ── 3. Optional packages ──────────────────────────────────────────
echo ""
echo "📦  Optional Dependencies"
for pkg in demucs yt_dlp ytmusicapi sentence_transformers; do
    if python3 -c "import $pkg" 2>/dev/null; then
        pass "$pkg"
    else
        warn "$pkg not installed (some features may be unavailable)"
    fi
done

# ── 4. System tools ───────────────────────────────────────────────
echo ""
echo "🔧  System Tools"
for tool in ffmpeg ffprobe; do
    if command -v $tool &>/dev/null; then
        pass "$tool"
    else
        fail "$tool not found (required for audio processing)"
    fi
done

# ── 5. GPU detection ─────────────────────────────────────────────
echo ""
echo "🖥️   GPU Support"
GPU=$(python3 -c "
import torch
if torch.cuda.is_available():
    print('CUDA: ' + torch.cuda.get_device_name(0))
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Apple MPS (Metal)')
else:
    print('CPU only')
" 2>/dev/null || echo "unknown")
pass "$GPU"

# ── 6. Project structure ─────────────────────────────────────────
echo ""
echo "📁  Project Structure"
ROOT="$(cd "$(dirname "$0")" && pwd)"

for f in scripts/api/main.py scripts/api/routes.py scripts/ui/app.py scripts/core/gpu.py \
         scripts/core/dj_engine.py scripts/core/stems.py scripts/download.py start.sh; do
    if [ -f "$ROOT/$f" ]; then
        pass "$f"
    else
        fail "$f missing"
    fi
done

# ── 7. Config ────────────────────────────────────────────────────
echo ""
echo "⚙️   Configuration"
if [ -f "$ROOT/.streamlit/config.toml" ]; then
    pass ".streamlit/config.toml"
else
    fail ".streamlit/config.toml missing"
fi

# ── 8. Library ────────────────────────────────────────────────────
echo ""
echo "🎵  Library"
SONG_COUNT=$(find "$ROOT/library" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
STEM_COUNT=$(find "$ROOT/library" -name "vocals.flac" -o -name "vocals.wav" 2>/dev/null | wc -l | tr -d ' ')
pass "$SONG_COUNT songs in library"
pass "$STEM_COUNT songs with stems"

# ── 9. Ports ─────────────────────────────────────────────────────
echo ""
echo "🌐  Ports"
for port in 8000 8501; do
    if lsof -ti tcp:$port &>/dev/null; then
        warn "Port $port is in use (stop existing processes first: ./start.sh stop)"
    else
        pass "Port $port is free"
    fi
done

# ── 10. Syntax check ────────────────────────────────────────────
echo ""
echo "🧪  Syntax Validation"
if python3 -m py_compile "$ROOT/scripts/api/routes.py" 2>/dev/null; then
    pass "routes.py"
else
    fail "routes.py has syntax errors"
fi
if python3 -m py_compile "$ROOT/scripts/ui/app.py" 2>/dev/null; then
    pass "app.py"
else
    fail "app.py has syntax errors"
fi
if python3 -m py_compile "$ROOT/scripts/api/tasks.py" 2>/dev/null; then
    pass "tasks.py"
else
    fail "tasks.py has syntax errors"
fi

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
    echo "  ⛔  Fix the failures above before running."
    echo ""
    exit 1
fi
