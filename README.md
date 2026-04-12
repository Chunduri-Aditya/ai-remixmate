# AI RemixMate

A real-time DJ engine that takes two songs and renders a beat-locked transition between them. BPM-matched, key-aware, stem-separated, mastered to broadcast LUFS standards. Built end-to-end in Python — analysis, mixing, the audio math, the API, the web UI, all of it.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![FastAPI](https://img.shields.io/badge/api-FastAPI-009688) ![PyTorch](https://img.shields.io/badge/ml-PyTorch-EE4C2C) ![License](https://img.shields.io/badge/license-MIT-green)

> Personal portfolio project. Active development. The library on my machine has 681 tracks and the system handles them.

---

## What it does

Give it two tracks. It analyzes both for tempo, key, energy, and bar structure. Picks the right exit and entry cue points at musical phrase boundaries. Time-stretches Song B to match Song A's BPM through librosa's phase vocoder. Phase-locks Song B's downbeat onto Song A's bar grid at the sample level. Renders a stem-aware crossfade where drums, bass, and vocals fade independently — the way a real DJ would do it. Optionally synthesizes a bridge beat from scratch with numpy. Masters the output to ITU-R BS.1770-4 (-14 LUFS, true-peak limited).

That's the engine. Wrapped around it: a FastAPI backend with an async job queue, SQLite persistence, structured JSON logging, an immutable audit trail, and a Streamlit web UI.

---

## Quick start

```bash
git clone https://github.com/Chunduri-Aditya/ai-remixmate.git
cd ai-remixmate
python -m venv remix-env && source remix-env/bin/activate
pip install -e ".[dev]"
./start.sh
```

You'll need `ffmpeg` (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian/Ubuntu).

Then open:
- **Web UI** → <http://localhost:8501>
- **API docs** → <http://localhost:8000/docs>

Docker also works:
```bash
docker compose up
```

---

## The parts that are actually interesting

Most "AI DJ" projects do a crossfade and call it done. The thing I cared about was making mixes that don't sound automated — and that meant getting four problems right.

**Beat-grid lock at the sample level.** A naive crossfade lines up two waveforms and hopes for the best. Real DJs land Song B's downbeat exactly on Song A's grid. The engine computes both bar-grid phases at the cue points and applies a sample-level correction (clamped to ±half a bar so it can't overshoot). After time-stretching, the entry sample index has to compensate for the stretch ratio:

```python
entry_sample_b = int(entry_time_b * sr / stretch_ratio)
```

Easy to get wrong. I got it wrong about six times before it locked.

**Stem-aware crossfading.** When Demucs stems are available (vocals, drums, bass, other), the engine fades each one on its own envelope. Drums and bass from Song B come in earlier in the window. Vocals are delayed. This is the move that makes the output sound like a person did it.

**Dynamic EQ fade.** Two tracks playing simultaneously stack low frequencies and the result sounds like mud. A low-shelf filter pulls Song A's bass down as Song B's bass rises. Solves the bass-clash problem that breaks most automated mixes.

**Procedural bridge beats.** When the two songs need a connector, the engine synthesizes a drum loop from scratch in numpy — sine-envelope kick, filtered-noise snare, highpass hi-hats — across six genre presets (techno, house, hip-hop, trap, DnB, ambient). No sample files. The whole bridge volume follows a bell curve so it rises into the transition and falls out the other side. You can also export the pattern as Strudel code and tweak it live in the browser.

The mastering chain runs ITU-R BS.1770-4 LUFS measurement with K-weighting and a true-peak look-ahead limiter. -14 LUFS, broadcast standard. The output won't blow up your speakers if you crank it.

---

## Architecture

```
scripts/
├── api/                    # FastAPI — 11 routers, 7 task modules, async job store
│   ├── main.py             # Lifespan, CORS, request-ID middleware
│   ├── jobs.py             # SQLite job persistence, ETA, cancel/retry
│   ├── routers/            # /library, /download, /stems, /analyze,
│   │                       #  /dj-remix, /spotify, /jobs, /crates, ...
│   └── task_modules/       # Long-running task functions
│
├── core/                   # Audio engine — 38 modules
│   ├── dj_engine.py        # Transition renderer (the heart)
│   ├── dj_analysis.py      # Beat / Section / SongStructure / TransitionPlan
│   ├── stems.py            # Demucs separation + stem-aware mixer
│   ├── beat_synth.py       # Procedural drum synthesis + Strudel export
│   ├── mastering.py        # ITU-R BS.1770-4 LUFS + true-peak limiter
│   ├── gpu.py              # MPS / CUDA / CPU auto-detection
│   ├── music_index.py      # 35-dim numpy vector index for semantic search
│   ├── key_detection.py    # Krumhansl-Schmuckler key profiling
│   └── ...                 # genre, recommend, audit, paths, audio_enhance, ...
│
└── ui/
    └── app.py              # Streamlit dashboard (primary UI)

frontend/                    # React + Vite + TypeScript (next-gen UI, in progress)
tests/                       # pytest + librosa probe guard + e2e suite
docs/                        # Architecture notes, DJ theory, tokenization roadmap
archive/                     # Frozen first-generation scripts
```

The runtime layout (gitignored): `library/` for downloaded songs, `outputs/` for rendered mixes, `data/` for SQLite stores and the audit log, `models/` for Demucs weights.

---

## Tech stack

| Layer | Stack |
|---|---|
| Audio analysis | librosa, numpy, scipy |
| Stem separation | Demucs (Meta AI), PyTorch (MPS / CUDA / CPU) |
| Mastering | ITU-R BS.1770-4 LUFS, true-peak limiter |
| Semantic search | 35-dim numpy vector index, weighted cosine similarity, JSON-persisted |
| Backend | FastAPI, Uvicorn, Pydantic v2 |
| Persistence | SQLite (jobs, crates, tags, favorites) + JSONL audit log |
| Download | yt-dlp, ytmusicapi, Spotify OAuth |
| UI | Streamlit (current), React + Vite (next) |
| Testing | pytest, pytest-asyncio |
| Packaging | pyproject.toml (PEP 517), Docker, docker-compose |

Python 3.10+. Runs on Apple Silicon, NVIDIA, or CPU. The GPU layer auto-detects.

---

## Try it

The minimum interesting thing to do with this is download two tracks and render a transition.

```bash
# Download a song (Demucs stems optional)
curl -X POST http://localhost:8000/download \
  -H "Content-Type: application/json" \
  -d '{"query": "Anyma Voices In My Head", "separate": true}'

# Render the transition
curl -X POST http://localhost:8000/dj-remix \
  -H "Content-Type: application/json" \
  -d '{
    "song_a": "Anyma - Voices In My Head",
    "song_b": "Dom Dolla - Define",
    "transition_bars": 16,
    "use_stems": true,
    "output_format": "flac"
  }'

# Poll for the result
curl http://localhost:8000/jobs/{job_id}
```

Want to audition the crossfade window without committing to a full render? Use `/dj-remix/preview` instead — it renders only the transition and returns BPM data, Camelot positions, harmonic score, and a stream URL.

For everything else (compatibility scoring, smart search, crates and tags, Spotify import, the whole thing), the interactive Swagger UI lives at <http://localhost:8000/docs>.

---

## Why I built this

Two reasons.

The first is simple. I wanted to learn audio DSP and ML deployment at the same time, and music is the domain I care about most. Reading papers about phase vocoders is one thing. Implementing one and then debugging it for a week because the bass-clash is killing your transitions is a different kind of learning.

The second is the part I think about more. There's a gap between "the model works in a notebook" and "the system works in a real product, on a real library, with a real interface." Almost everything I learned on this project happened in that gap. The audio quality work, the job queue, the lifespan refactor, the GPU detection, the stem-aware crossfade math — none of it is hard in isolation. All of it is hard when it has to coexist.

That's the version of this project that matters to me. Not the algorithms in isolation. The system that holds them together.

---

## Status

Active. Currently pushing on:
- Splitting the Streamlit UI into modular pages
- Building out the React frontend (`frontend/`) as the long-term UI
- Tightening test coverage on the core engine
- Better documentation of the DJ theory in [`docs/DJ_THEORY.md`](docs/DJ_THEORY.md)

The backend is stable. The mastering output passes the thresholds in `tests/test_report.json` (LUFS within ±1 dB, beat alignment under 40 ms). 681 tracks in my local library, no issues.

Full history in [CHANGELOG.md](CHANGELOG.md).

---

## Tests

```bash
pytest tests/ -v
pytest tests/ -x              # stop on first failure
pytest -m "not dj_analysis"   # skip librosa-dependent tests
```

`conftest.py` includes a librosa probe guard that auto-skips `dj_analysis`-marked tests when librosa or numba can't initialize, so the suite never crashes on a broken environment.

---

## Configuration

`config.yaml` exposes the tuneable parameters. Override locally with `config.local.yaml` (gitignored).

```yaml
audio:
  sample_rate: 22050
  transition_bars: 16
  output_format: wav        # "flac" for ~60% smaller files

bridge_beat:
  default_intensity: 0.38
  default_genre: auto

api:
  max_active_jobs: 3        # rate-limit cap
  worker_threads: 2

stems:
  enabled: false            # off by default; Demucs is slow
  model: htdemucs
```

---

## Caveats

This is a personal portfolio project, not a product. The download integrations (yt-dlp, Spotify) are for personal use on tracks you have rights to. Don't use this to redistribute copyrighted material. All music rights remain with the original creators.

---

## License

MIT — see [LICENSE](LICENSE).

## Author

**Aditya Chunduri** · [github.com/Chunduri-Aditya](https://github.com/Chunduri-Aditya) · chunduri@usc.edu · chunduriaditya2@gmail.com
