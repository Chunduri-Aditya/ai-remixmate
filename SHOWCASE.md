# AI RemixMate — showcase

Short version and a 2-minute demo. For the full capability map, measured line counts, endpoint list, and the five hard audio problems, read [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) — every number there is command-derived.

## What it is

A real-time DJ engine, built end to end in Python and React, that takes two tracks and renders a beat-locked, key-aware, stem-separated transition mastered to −14 LUFS (ITU-R BS.1770-4). Around the transition engine sits a setlist planner (harmonic scoring against the Camelot wheel, BPM continuity, energy-arc shaping, a Markov model over historical setlists), a lyric-phrase "wordplay" layer over the Genius API, and a FastAPI backend with an async job queue, SQLite/Postgres persistence, structured logging, and an immutable audit trail.

The parts worth pointing at:

- **Sample-level beat-grid lock**, with the time-stretch ratio compensated into the entry index — the error the ear catches but a waveform view does not.
- **Equal-power crossfading** with stem-independent fade curves (drums, bass, vocals fade separately), not equal-gain.
- **Research-grade MIR**, not just `librosa.beat_track`: the roadmap and `ai_remixmate_feature_lab/` track swapping in Beat This! (ISMIR 2024), TIV harmonic scoring alongside the Camelot wheel, and CLAP semantic embeddings over the handcrafted feature set — each behind a flag with the previous engine kept selectable.

## Branches

| Branch | What it is | State |
|---|---|---|
| `main` | Single-user local app — FastAPI single process, `ThreadPoolExecutor`, SQLite, a local music library on the filesystem | Stable. This is what `./start.sh` runs. 46 commits, validated against a 583-track / 69 GB library |
| `hosted-app` | Migration to a multi-user hosted product: catalog cut to Creative-Commons audio, per-user tenancy, Supabase auth, Postgres, deploy on a small VPS | Phases 0–3 verified, Phase 4 (auth) code-done, **Phase 5 (SQLite → Postgres) verified against a real Postgres 16 container**. Phases 6–8 (retention, deploy, legal) not started. Canonical plan: [`app/DEPLOYMENT_PLAN.md`](app/DEPLOYMENT_PLAN.md) |

`hosted-app` is 24 commits ahead of `main`.

## 2-minute demo (local, no cloud)

```bash
cd ~/Desktop/Projects/ai-remixmate

# backend + frontend test gates
.venv/bin/python -m pytest -m "not dj_analysis and not integration" \
  --ignore=tests/e2e_test_suite.py --ignore=tests/test_audio_source.py -q
cd frontend && npx vite build && cd ..

# run it
./start.sh                    # API on :8000, React UI on :5173, DJ widget at :5173/widget
```

Last run 2026-09-08 on `hosted-app`: **763 passed, 90 deselected** (safe subset); **`vite build` ✓ 1.6 s**.

`dj_analysis` and `integration` tests are deselected here — they need real audio files and a longer run; see `PREREQUISITES.md`.

## Done

- Transition engine: beat-grid lock, equal-power stem crossfade, phase-vocoder time-stretch, optional numpy bridge-beat synthesis, LUFS master
- Setlist planner: Camelot harmonic scoring, energy-arc shaping, Markov ordering, Exportify CSV import
- FastAPI backend, async job queue, SQLite persistence, structured JSON logging, immutable audit trail
- React 18 / TypeScript UI — 9 pages plus a floating picture-in-picture DJ widget
- `hosted-app` Phases 0–5: test-isolation fixture, CC-only catalog gate, generative-feature flag, per-user contextvar tenancy across mixes/jobs/crates/feedback, RS256 JWKS auth gate, SQLAlchemy Core stores on SQLite or Postgres with multi-process restart safety

## Known gaps

- **No live hosted deployment.** `hosted-app` Phases 6–8 are provisioning a VPS, a managed Postgres, Cloudflare Pages, a curated ~150-track CC catalog with Demucs run at ingest, and legal pages — operator work with a monthly cost, not a code change. The `DEPLOYMENT_PLAN.md` (with its rejected-Jamendo licensing analysis and mutation tests) is the artifact, not a running URL.
- **Phase 4 auth** is code-complete but has not been run against a live Supabase project.
- `main` still has `yt-dlp` pulling YouTube audio as a first-class feature and no auth on any endpoint — it is explicitly "not appropriate as-is for a production service". `hosted-app` is the answer to that; it is not merged back.
- Repo-wide `ruff check` reports ~1583 pre-existing findings (mostly `W293` blank-line whitespace); the migration gates on per-phase scoped ruff instead. A repo-wide `ruff check --fix` pass is worth doing before a Phase 7 CI job.
- Minor doc drift: `README.md` says 681 tracks / Python 3.10+, `PROJECT_OVERVIEW.md` (command-derived, newer) says 583 tracks / Python 3.12.

## Layout

`~/Desktop/Projects/ai-remixmate` · default branch `main`, active branch `hosted-app` · <https://github.com/Chunduri-Aditya/ai-remixmate>
