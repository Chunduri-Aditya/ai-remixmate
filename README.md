# AI RemixMate

A DJ engine that takes two songs and renders a beat locked transition between them: tempo matched, key aware, stem separated, and mastered to a loudness target. The audio engine and API are Python (FastAPI); the web UI is React, Vite and TypeScript.

![Python](https://img.shields.io/badge/python-3.12-blue) ![FastAPI](https://img.shields.io/badge/api-FastAPI-009688) ![PyTorch](https://img.shields.io/badge/ml-PyTorch-EE4C2C) ![License](https://img.shields.io/badge/license-MIT-green)

> Personal portfolio project. It runs as a single user local app out of the box; the hosted multi user mode lives on the `hosted-app` branch and is documented under [`app/`](app/).

---

## Quick start

Requirements: Python 3.12, Node 22, ffmpeg. Full details in [PREREQUISITES.md](PREREQUISITES.md).

```bash
git clone https://github.com/Chunduri-Aditya/ai-remixmate.git && cd ai-remixmate
./start.sh setup
./start.sh
```

`./start.sh setup` creates `.venv`, installs the Python packages (runtime and test) and the npm packages. `./start.sh` then starts the API and the React UI. It installs again only when packages are missing or the dependency files changed. Run the tests with:

```bash
.venv/bin/python -m pytest -m "not dj_analysis" --ignore=tests/e2e_test_suite.py -q
```

Ports: `REMIXMATE_API_PORT` (default 8000) and `REMIXMATE_UI_PORT` (default 5173) move the two servers, for example `REMIXMATE_API_PORT=8765 REMIXMATE_UI_PORT=5199 ./start.sh`. Use the same variables with `./start.sh stop`. `start.sh` stops only its own listeners and exits when another program holds a port.

Other modes (from `start.sh`):

```bash
./start.sh api                # API only, on http://127.0.0.1:8000
./start.sh --skip-setup       # start without checking packages
./start.sh --dev              # API with --reload
./start.sh stop               # stop this app's API and UI listeners
```

The API binds to `127.0.0.1` by default. Then open:

- React UI: <http://localhost:5173>
- Floating DJ widget: <http://localhost:5173/widget>
- API reference (Swagger UI): <http://127.0.0.1:8000/docs>, with ReDoc at `/redoc`

Docker also works and publishes the API on `127.0.0.1:8000`:

```bash
docker compose up
```

### Adding songs

Songs enter the library through the catalogue ingest tool, which pulls Creative Commons or public domain tracks from ccMixter or the Internet Archive, classifies the licence, separates stems and runs analysis:

```bash
.venv/bin/python -m scripts.download --source ccmixter "glitch hop" --search-only
REMIXMATE_LICENSE_ENFORCE=1 .venv/bin/python -m scripts.download --source ccmixter "glitch hop" --pick 2
```

With `REMIXMATE_LICENSE_ENFORCE=1` the licence gate rejects tracks that are not CC or public domain. `--help` lists every option.

---

## What it does

Give it two tracks from the library. It analyzes both for tempo, key, energy and bar structure, picks exit and entry cue points at phrase boundaries, time stretches Song B to Song A's tempo with librosa's phase vocoder, and corrects Song B's downbeat onto Song A's bar grid. It then renders a stem aware crossfade where drums, bass and vocals move on their own envelopes, optionally adds a synthesized bridge beat, and masters the result to the loudness target in `config.yaml`.

Around the transition engine:

- **DJ chains.** Render 2 to 8 songs as one continuous mix with phrase locked transitions (`POST /dj-chain`).
- **Previews.** `POST /dj-remix/preview` renders only the transition window so you can audition it before a full render.
- **Setlist planning.** `/setlist/optimize` orders a playlist by harmonic compatibility, tempo continuity and an energy arc (`ramp_up`, `mountain`, `wave`, `ramp_down`). `/setlist/import-csv` takes an Exportify CSV export.
- **Camelot modulation.** `/setlist/camelot/modulation` classifies a key change into a named type (`energy_boost`, `diagonal`, `relative_shift`, `add_four_protocol` and others) with a cost and a blend recommendation.
- **Wordplay.** `/setlist/wordplay` scores lyric phrase overlap between the end of one song and the start of the next, using the Genius API.
- **Feedback learning.** A thumbs up or down on a render (`POST /dj-remix/feedback`) updates the spectral cue matching weights.
- **Library tools.** Crates, tags, favorites, search, recommendations, cue export, and storage reporting and cleanup for rendered mixes.
- **Jobs.** Long renders run as queued jobs with progress over server sent events (`/events/stream`) and polling (`/jobs/{job_id}`).
- **Spotify metadata.** Read only search and track lookups (`/spotify/search`, `/spotify/track/{track_id}`) when a client id and secret are configured.
- **Operator gated routes.** Stem splitting (`/stems/split`) answers 503 unless `REMIXMATE_STEM_SPLIT_ENABLED` is truthy, and the generative AI Lab routes answer 503 unless `REMIXMATE_GENERATIVE_ENABLED` is truthy.

The UI pages: Mission Control, Library Atlas, Mix Deck, Set Builder, Signal Search, AI Lab, Mix Vault, Operations (processing status and the job queue), Account, the legal pages, and the floating DJ widget. The widget can pop out into an always on top window through Document Picture in Picture in Chromium browsers and suggests harmonically and tempo matched next tracks.

---

## The parts that are actually interesting

Most "AI DJ" projects do a crossfade and call it done. The goal here was mixes that do not sound automated, which meant getting a handful of problems right.

**Beat grid lock at the sample level.** A naive crossfade lines up two waveforms and hopes. The engine computes both bar grid phases at the cue points and applies a sample level correction to Song B, clamped to half a bar either way so it cannot overshoot into the wrong bar.

**Stem aware crossfading.** When Demucs stems exist for both songs (vocals, drums, bass, other), each stem fades on its own envelope, shaped by how similar that stem pair is: near identical drums get a long overlapping blend, clashing ones get a fast handoff. Bass gets a cosine taper swap rather than a long overlap, because two basslines on top of each other is what makes automated mixes sound muddy. Below `dj.transition_confidence_threshold` the engine falls back to a plain crossfade.

The curves are equal power (`cos` and `sin`), not equal gain. Two different songs are uncorrelated, so their powers add rather than their amplitudes; amplitude complementary curves dip in loudness at the midpoint of every transition.

**Beat quantized fades.** The `stepped` transition effect divides the stem crossfade into one step per beat instead of a smooth sweep. The steps are equal in power, so the staircase stays perceptually even.

**Masking aware EQ and spectral sync.** During stem transitions a multiband EQ ducks the bands where one track masks the other, and a time varying EQ eases the mix's spectrum from Song A's outro toward Song B's profile, leaving the bass band alone. Both switch off in `config.yaml` for A/B comparison.

**Procedural bridge beats.** When two songs need a connector, the engine synthesizes a drum loop in numpy from genre presets (`techno`, `house`, `hiphop`, `trap`, `dnb`, `ambient`). No sample files. The bridge volume follows a sine arch so it rises into the transition and falls out the other side, and the pattern can be exported as Strudel code.

**Setlist intelligence.** `setlist_planner.py` runs a weighted greedy optimizer over a playlist (default weights 0.50 harmonic, 0.30 tempo, 0.20 energy arc: `w_harmonic`, `w_bpm`, `w_energy` at `scripts/core/setlist_planner.py:189`), with optional Markov chain blending for sequencing learned from historical setlists. Energy is computed locally from spectral flux and RMS rather than a streaming service's audio features API.

**Mastering.** The chain measures integrated loudness per `ITU-R BS.1770-4` with K weighting and finishes with a true peak limiter with look ahead. The target comes from `audio.target_lufs`.

---

## Architecture

```
scripts/
├── api/                    # FastAPI app
│   ├── main.py             # lifespan, CORS, request id and auth middleware, /docs
│   ├── auth.py             # per request user scoping + Supabase JWT (off unless enabled)
│   ├── jobs.py             # job store, ETA, cancel and retry
│   ├── routers/            # system, legal, library, analysis, remix, setlist, stems,
│   │                       #  crates, jobs, events, feedback, account, admin,
│   │                       #  spotify, generative
│   └── task_modules/       # long running task functions
├── core/                   # audio engine and stores
│   ├── dj_engine.py        # transition renderer
│   ├── dj_analysis.py      # beats, sections, song structure, transition plans
│   ├── stems.py            # Demucs separation + stem aware mixer
│   ├── spectral_mixing.py  # masking aware multiband EQ
│   ├── spectral_sync.py    # in transition spectral sync
│   ├── beat_synth.py       # procedural drum synthesis + Strudel export
│   ├── mastering.py        # BS.1770-4 loudness + true peak limiter
│   ├── key_detection.py    # key detection + camelot_modulation()
│   ├── setlist_planner.py  # greedy optimizer, Markov model, Exportify CSV parser
│   ├── wordplay.py         # Genius lyrics + n-gram overlap
│   ├── license.py          # licence classification and the publish gate
│   ├── db.py               # SQLite per store, or Postgres when DATABASE_URL is set
│   └── ...
└── download.py             # operator catalogue ingest (CC and public domain sources)

frontend/                   # React + Vite + TypeScript UI
app/                        # hosted mode plan, status ledger and deploy files
tests/                      # pytest suite
docs/                       # DJ theory notes and the docs index
archive/                    # frozen earlier code and legacy docs
```

Runtime directories (gitignored): `library/` for ingested songs and stems, `outputs/` for rendered mixes, `data/` for SQLite stores and the audit log, `models/` for model weights.

---

## Tech stack

| Layer | Stack |
|---|---|
| Audio analysis | librosa, numpy, scipy |
| Stem separation | Demucs, PyTorch (MPS, CUDA or CPU, auto detected) |
| Mastering | `BS.1770-4` loudness, true peak limiter |
| Setlist planning | greedy + Markov optimizer, Camelot wheel, spectral flux energy |
| Lyric matching | lyricsgenius, NLTK, `scikit-learn` `TF-IDF` |
| Backend | FastAPI, Uvicorn, Pydantic v2 |
| Persistence | SQLite locally, Postgres in hosted mode (SQLAlchemy Core), JSONL audit log |
| UI | React, Vite, TypeScript |
| Testing | pytest, vitest |
| Packaging | pyproject.toml, Docker, docker compose |

---

## Try it

With the API running locally (auth off, so no token is needed):

```bash
# Render a transition between two songs already in the library
curl -X POST http://127.0.0.1:8000/dj-remix \
  -H "Content-Type: application/json" \
  -d '{"song_a": "Artist A - Title A", "song_b": "Artist B - Title B", "transition_bars": 16}'

# Poll the job
curl http://127.0.0.1:8000/jobs/{job_id}

# Optimize an Exportify playlist export
curl -X POST "http://127.0.0.1:8000/setlist/import-csv?arc=mountain" \
  -F "file=@my_playlist.csv"

# Classify a Camelot key change
curl "http://127.0.0.1:8000/setlist/camelot/modulation?from_key=8A&to_key=9B"
```

Every route, its request schema and its response shape are in the interactive docs at <http://127.0.0.1:8000/docs>.

---

## Configuration

`config.yaml` holds the tunable parameters. Override values in a local `config.local.yaml` (gitignored), or with environment variables named `REMIXMATE_<SECTION>_<KEY>` (for example `REMIXMATE_AUDIO_SAMPLE_RATE`). Environment variables win over both files.

A selection of keys, with their values as they appear in `config.yaml`:

```yaml
audio:
  sample_rate: 44100
  target_lufs: -14.0

remix:
  max_song_seconds: 180

dj:
  mixing_engine: current        # "current" or "classic" (the vendored engine, mastered at -8 LUFS)
  default_transition_bars: 16
  masking_eq_enabled: true
  spectral_matching: true
  spectral_sync: true
  transition_confidence_threshold: 0.30

library:
  max_size_gb: 50.0
  library_dir: ""               # "" = <project root>/library
  outputs_dir: ""               # "" = <project root>/outputs

retention:
  mix_max_age_days: 30
  mix_max_count: 50
  sweep_after_render: true
  sweep_interval_hours: 6

api:
  port: 8000

spotify:
  client_id: ""
  client_secret: ""
```

Comments in the file describe each key it holds. The `separation:` section is read by no code: `separate_song_stems()` defaults to `htdemucs` (`scripts/core/stems.py:130`) and resolves the device through `scripts/core/gpu.py`, so set `REMIXMATE_DEVICE=cpu|mps|cuda` to force one. `analysis.beat_backend` and `models.clap_model` exist only as dataclass fields in `scripts/core/config.py`, with no section in `config.yaml`.

---

## Tests

```bash
.venv/bin/python -m pytest tests -q                                          # full suite
.venv/bin/python -m pytest -m "not dj_analysis" --ignore=tests/e2e_test_suite.py -q   # the CI subset
.venv/bin/ruff check .
cd frontend && npx vitest run
```

`tests/conftest.py` probes librosa and numba and skips `dj_analysis` marked tests when they cannot initialize. It also points the outputs and data directories at a throwaway directory before any project module is imported, so a test run does not write into the real `outputs/` or `data/`.

---

## Hosted mode

The `hosted-app` branch turns the local tool into a multi user service: Supabase JWT auth (`REMIXMATE_AUTH_ENABLED`, `SUPABASE_JWKS_URL`), Postgres through `DATABASE_URL`, per user jobs and outputs, an operator curated CC catalogue behind `REMIXMATE_LICENSE_ENFORCE`, and legal pages with DMCA intake. With auth off, every request runs as the single `local` user and no token is required.

Start at [`app/README.md`](app/README.md). The phased plan is [`app/DEPLOYMENT_PLAN.md`](app/DEPLOYMENT_PLAN.md), current state and open gaps are in [`app/STATUS.md`](app/STATUS.md), and the server runbook is [`app/deploy/PROVISION.md`](app/deploy/PROVISION.md).

---

## Why I built this

Two reasons.

I wanted to learn audio DSP and ML deployment at the same time, and music is the domain I care about most. Reading papers about phase vocoders is one thing. Implementing one and then debugging it for a week because a bass clash is ruining your transitions is a different kind of learning.

The second is the gap between "the model works in a notebook" and "the system works in a real product, on a real library, with a real interface." Almost everything I learned on this project happened in that gap. The audio quality work, the job queue, the GPU detection, the stem aware crossfade math: none of it is hard in isolation. All of it is hard when it has to coexist.

Full history in [CHANGELOG.md](CHANGELOG.md).

---

## Legal

The library is built from Creative Commons and public domain sources through the ingest tool, and each song keeps a `license.json` beside it. Rendered mixes carry a `CREDITS.txt` with attribution. Lyrics used by the wordplay layer are fetched from the Genius API for local phrase matching and never redistributed.

Nothing copyrighted is committed to this repository. `library/`, `outputs/`, `data/` and `models/` are gitignored.

---

## License

MIT, see [LICENSE](LICENSE).

## Author

**Aditya Chunduri** · [github.com/Chunduri-Aditya](https://github.com/Chunduri-Aditya)
