# AI RemixMate: Project Overview

> Portfolio summary with numbers pinned to `main`, not a map of this branch. Agents start at `CLAUDE.md` and `ARCHITECTURE.md`.

A real time DJ engine built end to end in Python and React. It takes two songs, understands them
musically, and renders a transition between them that sounds like a person made it.

Every number in this document was measured on 2026-08-28 with a command, not estimated. It describes
the single user app on `main`; `app/STATUS.md` records what the `hosted-app` branch changed.

---

## At a glance

| | |
|---|---|
| **Live application code** | 41,529 lines Python across 92 files |
| **Test code** | 10,784 lines across 43 files, 774 tests collected |
| **Frontend** | 14,863 lines TypeScript, React and CSS |
| **REST API** | 86 endpoints across 13 routers |
| **Audio engine** | 51 modules under `scripts/core/` |
| **UI** | 9 pages plus a floating picture in picture DJ widget |
| **Development span** | 2025-07-23 to 2026-08-28, 46 commits on `main` |
| **Validated against** | a 583 song / 69 GB personal library |

Stack: Python 3.12, FastAPI, librosa, PyTorch, Demucs, numpy/scipy, SQLite · React 18, TypeScript,
Vite, Zustand, TanStack Query, WaveSurfer.js, Web Audio API.

---

## The five hard problems

Most automated DJ tools crossfade two files and stop. These are the parts that actually decide
whether a mix sounds human, and what this project does about each.

### 1. Beat grid lock at the sample level

A naive crossfade aligns two waveforms and hopes. A real DJ lands Song B's downbeat exactly on Song
A's grid. The engine computes both bar grid phases at the cue points and applies a sample level
correction, clamped to half a bar so it cannot overshoot. After time stretching, the entry index has
to compensate for the stretch ratio:

```python
entry_sample_b = int(entry_time_b * sr / stretch_ratio)
```

Getting the stretch compensation wrong shifts the whole incoming track off the grid, and the error
is subtle enough to pass a visual waveform check while being obvious to the ear.

### 2. Equal power crossfading, not equal gain

Two different songs are uncorrelated signals, so their **powers** add during an overlap, not their
amplitudes. Curves that sum to 1.0 in amplitude, the intuitive choice, dip about 3 dB at the midpoint
of every single transition. The engine uses `cos`/`sin` quarter cycle curves so
`fade_out² + fade_in² = 1` at every sample.

This is guarded by a behavioural test that crossfades two noise beds and asserts the midpoint stays
within 1 dB of the window edges, so the bug cannot silently return.

### 3. Stem aware transitions

When Demucs stems exist (vocals, drums, bass, other), each stem pair gets its own envelope shaped by
how acoustically similar that pair is: near identical drums get a long overlapping blend, clashing
ones get a fast handoff.

Bass is deliberately different. Two basslines overlapping is the single thing that makes automated
mixes sound muddy, so bass gets a dedicated cosine taper swap at one point rather than a crossfade,
and it stays the sole authority over the low end.

### 4. Masking aware multiband EQ

An offline variant of Hafezi and Reiss (2015). Where one track's mid or high content masks something
the other track needs, that band is ducked, bidirectionally, across the transition window. Gain
curves taper to 0 dB at the window edges so the processed region splices back in without a seam.

### 5. Broadcast standard mastering

Output is normalised to ITU-R BS.1770-4 loudness with a true peak limiter, plus per stem type LUFS
normalisation derived from the user's own corpus rather than fixed constants.

---

## Capability map

**Transition engine**
- Tempo matching by phase vocoder time stretch
- Phrase boundary and bar grid cue point selection
- Beat grid lock with sample level correction
- Stem aware crossfade with similarity driven curves
- Dedicated bass swap ramp, dynamic low shelf EQ
- Masking aware multiband EQ, time varying spectral sync
- Beat quantized stepped crossfade, one equal power step per beat
- Procedural bridge beat synthesis in numpy across six genre presets, exportable as Strudel code

**Music understanding**
- BPM and beat tracking (librosa, with an optional Beat This! ISMIR 2024 backend)
- Key detection via Krumhansl-Schmuckler, mapped to the Camelot wheel
- Tonal Interval Vector harmonic scoring (Bernardes et al. 2016)
- Song structure and phrase segmentation
- Energy profiling with arousal and valence, computed locally

**Set intelligence**
- Greedy setlist optimiser weighting harmonic compatibility, BPM continuity and energy arc
- Markov chain blending trained on historical setlist patterns
- Nine named Camelot modulation types, each with a psychoacoustic cost
- Lyric n gram matching between adjacent tracks through the Genius API

**Search and library**
- 35 dimensional numpy embedding index for similarity search, no FAISS dependency
- CLAP 512 dimensional semantic text to audio search
- Demucs stem separation with GPU auto detection across MPS, CUDA and CPU
- Self healing library index, storage accounting and eviction

**Platform**
- FastAPI backend, async job queue, write through SQLite persistence
- Server sent events for live progress, with polling fallback
- React SPA: 9 pages, global player with a 3 band EQ and visualiser, drag and drop set builder
- rekordbox XML and Serato GEOB cue export
- Online learning loop: thumbs up and down verdicts retrain per band spectral matching weights

---

## Architecture

```
scripts/
├── api/          FastAPI app, 13 routers, 86 endpoints, job store, async task modules
└── core/         51 modules, the audio engine
    ├── dj_engine.py        renderer: beat grid lock, stem crossfade, dynamic EQ
    ├── dj_analysis.py      structure detection, transition planning
    ├── classic_engine/     frozen reference renderer, kept byte faithful
    ├── mastering.py        ITU-R BS.1770-4 loudness and true peak limiting
    ├── setlist_planner.py  Camelot + BPM + energy optimiser with Markov blending
    └── ...
frontend/src/    React 18 + TypeScript, 9 pages, Zustand stores, Web Audio engine
tests/           43 files, 774 tests
```

**Two engines by design.** A frozen vendored snapshot of the reference mixing path runs alongside
the live engine, selected by config. It exists so current work can be compared against renders that
were already judged good, rather than trusting that a change was an improvement.

---

## Engineering practices worth pointing at

These are the parts that separate a working prototype from a system.

**Tests assert what you hear.** `tests/test_behavioral.py` is a regression guard for audible bugs.
It does not check array shapes; it crossfades two noise beds and asserts the midpoint does not dip,
renders a transition and asserts Song B is silent in the first half, measures per stem onset timing
with distinct test tones.

**Guards are proven to fail.** New tests are verified by reintroducing the bug and watching them go
red, then restoring. A test that has never failed is not evidence of anything. Several bugs in this
codebase were found precisely because a guard was mutated and unexpectedly stayed green.

**Documented claims get audited against the code.** An adversarial audit measured per stem entry
timing and disproved the project's own documentation, which claimed drums and bass entered ahead of
vocals. The docs were corrected rather than the claim quietly retained.

**Traps are written down where they bite.** `DEVELOPING.md` records 18 gotchas, each of which cost
real debugging time once: FastAPI route registration order shadowing a static path, two enum
serialisations coexisting in one SQLite table and silently undercounting any raw query, two logging
setups where only one is wired, config keys that vanish without error if the dataclass field is
missing.

**Security work is real, not decorative.** Outbound metadata HTTP is hardened with a host allowlist,
API key redaction and SSRF fixes, with tests that verify refusal happens before the request is made
rather than merely observing that nothing came back.

---

## Honest limits

- Demucs stem separation runs 2 to 5 times real time on CPU. Practical on Apple Silicon with `mps`,
  slow otherwise.
- The stem crossfade is driven by similarity, not stem identity. There is no logic that specifically
  delays vocals relative to drums; under equal similarity they enter together.
- Track duration currently reports a placeholder value from the analysis pipeline rather than the
  measured length.
- The UI is desktop only by design. The embedded widget covers the small screen case.
- On `main`, downloading audio through yt-dlp may breach YouTube's Terms of Service, so that branch
  is scoped to personal, educational and portfolio use; see the Legal section in `README.md` on
  `main`. The `hosted-app` branch removed every user download path in `8c4f204` and serves an
  operator curated Creative Commons catalog only; see `app/STATUS.md`.

---

## Where to read more

| Document | Contents |
|---|---|
| `README.md` | Product overview, setup, configuration, legal position |
| `DEVELOPING.md` | Design decisions, common tasks, and the 18 gotchas |
| `archive/docs/legacy/ARCHITECTURE.md` | Deep system map, request traces, data flow (written for `main`, archived) |
| `CHANGELOG.md` | What changed, when, and why |
