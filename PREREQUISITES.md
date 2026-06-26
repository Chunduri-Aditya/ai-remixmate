# AI RemixMate — Prerequisites

Everything you need before running the app. The `./start.sh` script can install most of this for you automatically.

## System requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| **OS** | macOS 12+ or Linux | Windows via WSL2 works with the same steps |
| **Python** | 3.10+ | 3.11 recommended |
| **Node.js** | 18+ | For the React frontend (`frontend/`) |
| **npm** | 9+ | Bundled with Node |
| **ffmpeg** | any recent | Required for audio download/conversion |
| **Disk** | ~2 GB free | Plus space for your song library (grows with downloads) |
| **RAM** | 8 GB+ | 16 GB+ recommended for Demucs stem separation |

## Install system tools

### macOS

```bash
brew install ffmpeg node python@3.11
```

### Debian / Ubuntu

```bash
sudo apt update
sudo apt install -y ffmpeg python3 python3-pip python3-venv nodejs npm
```

## Python environment

**`./start.sh` creates and uses `remix-env/` automatically** (required on macOS/Homebrew Python, which blocks system-wide `pip` installs per PEP 668).

Optional manual setup:

```bash
cd ai-remixmate
python3.12 -m venv remix-env    # prefer 3.11–3.12 over Homebrew 3.14
source remix-env/bin/activate   # Windows: remix-env\Scripts\activate
```

If you already use **Conda/Anaconda**, activate your env first — `start.sh` will use the active venv instead of creating `remix-env/`.

## One-command setup + launch

From the project root:

```bash
./start.sh
```

This will:

1. Upgrade/install Python packages (`requirements.txt`, editable install, latest `yt-dlp`)
2. Install/update frontend npm packages
3. Run a quick readiness check
4. Start the **API** (port 8000) and **React UI** (port 5173)

### Other start modes

```bash
./start.sh              # setup + React UI + API (default)
./start.sh frontend     # same as above
./start.sh setup        # install/update packages only, don't start servers
./start.sh api          # API only (for GitHub Pages widget + local backend)
./start.sh ui           # legacy Streamlit UI + API
./start.sh stop         # kill all RemixMate processes
./start.sh --skip-setup # start without reinstalling packages (faster restarts)
```

## Manual install (if you prefer)

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
pip install -U yt-dlp
cd frontend && npm install && cd ..
```

## Validate before launch

```bash
bash bin/check.sh
```

## URLs after startup

| Service | URL |
|---------|-----|
| React UI (Mission Control) | http://localhost:5173/mission-control |
| Downloads | http://localhost:5173/operations |
| **DJ Widget (floating)** | http://localhost:5173/widget |
| API docs | http://localhost:8000/docs |
| Legacy Streamlit UI | http://localhost:8501 (only with `./start.sh ui`) |

## Optional: GitHub Pages widget

The static frontend can be hosted on GitHub Pages while the API runs locally:

1. Enable **Settings → Pages → Source: GitHub Actions**
2. Run `./start.sh api`
3. Open https://chunduri-aditya.github.io/ai-remixmate/widget

## GPU acceleration

Demucs stem separation uses:

- **Apple Silicon** → MPS (Metal), with automatic CPU fallback
- **NVIDIA** → CUDA
- **Otherwise** → CPU (slower but works)

`bin/check.sh` reports which backend is available.
