# AI RemixMate prerequisites

What a machine needs before `./start.sh` runs. The script creates the Python environment and installs the Python and npm packages itself; the system tools below you install once.

## System tools

| Tool | Version | Where the version comes from |
|------|---------|------------------------------|
| Python | 3.12 | `requires-python = ">=3.12"` in `pyproject.toml`; CI and the `Dockerfile` (`python:3.12-slim`) use 3.12 |
| Node.js and npm | Node 22 | CI runs the frontend job on `node-version: "22"` |
| ffmpeg | any current release | `start.sh` exits if `ffmpeg` is not on `PATH` |

Use 3.12 exactly: `start.sh` exits with a message when no 3.12 interpreter exists or when an existing `.venv` runs another version, and its comments note that the `torch` pins in `requirements.txt` rule out 3.14.

## Install on macOS (Homebrew)

```bash
brew install python@3.12 node@22 ffmpeg
```

`node@22` is keg only, so Homebrew does not link it onto `PATH`. Add it before running `start.sh`:

```bash
export PATH="$(brew --prefix node@22)/bin:$PATH"
node --version   # expect v22.x
```

## Install on Linux

Install Python 3.12, Node 22 and ffmpeg with your distribution's package manager or an upstream installer. `start.sh` needs `python3.12` (or a `python3` that is 3.12), `node`, `npm` and `ffmpeg` on `PATH`.

## Python environment

`start.sh` always uses the project venv at `.venv/`. On the first run it creates `.venv/` from the first interpreter it finds among `python3.12`, then `python3` or `python`, and only one that reports 3.12 is accepted. An existing `.venv` on another version makes it exit: delete `.venv` and run `./start.sh setup`. A venv is required on macOS because Homebrew Python blocks system wide `pip` installs (PEP 668).

To create it by hand:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e ".[dev]"
cd frontend && npm ci && cd ..
```

On the maintainer's machine on 2026-09-30, `.venv/` took 1.5G and `frontend/node_modules` took 199M (`du -sh`), before any songs, stems or rendered mixes.

## start.sh modes

```bash
./start.sh              # install packages only when missing or changed, then API (port 8000) and React UI (port 5173)
./start.sh frontend     # same as the default
./start.sh api          # install packages, then the API only
./start.sh setup        # install or update packages only, start nothing
./start.sh stop         # stop this app's listeners on the configured ports (8000 and 5173 by default)
```

Flags, combinable with `frontend` and `api`:

| Flag | Effect |
|------|--------|
| `--skip-setup` | never check or install packages. Without it, a run installs only when the stamp in `.venv/` differs from the dependency files or `frontend/node_modules` is missing; `./start.sh setup` always installs |
| `--no-open` | do not open a browser tab |
| `--dev` | run the API with `--reload`, watching `scripts/` only |
| `--https` | serve the API over TLS, generating `certs/cert.pem` and `certs/key.pem` with `certs/generate.sh` (mkcert, or openssl as fallback) when missing |
| `--host ADDR` | API bind address; the default is `127.0.0.1` |

Those five are the only modes. The retired `ui` and `both` modes print a message and exit with status 2.

After the API starts, `start.sh` polls `GET /health/live` for up to 25 seconds and warns if it does not answer.

## URLs after startup

The ports below are the defaults. Set `REMIXMATE_API_PORT` and `REMIXMATE_UI_PORT` to change them, and set the same variables for `./start.sh stop`. `start.sh` exits when another program holds a port.

| Service | URL |
|---------|-----|
| React UI | http://localhost:5173 (opens Mission Control) |
| DJ widget | http://localhost:5173/widget |
| API docs | http://127.0.0.1:8000/docs |
| Liveness probe | http://127.0.0.1:8000/health/live |

## Running the checks

The same steps `.github/workflows/ci.yml` runs, from the project root:

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest -m "not dj_analysis" --ignore=tests/e2e_test_suite.py -q
.venv/bin/python -m pytest -m dj_analysis -q
cd frontend && npx tsc --noEmit && npx vitest run && npx eslint . --max-warnings 0 && npx vite build && cd ..
```

CI installs the pinned `requirements-dev.txt` into a fresh venv; a long lived local `.venv` can drift above those pins, so a local green run is not proof CI is green.

## Stem separation device

Demucs runs on the device `scripts/core/gpu.py` detects: MPS on Apple Silicon, CUDA on an NVIDIA GPU, CPU otherwise. Force one with the `REMIXMATE_DEVICE` environment variable (`mps`, `cuda` or `cpu`).
