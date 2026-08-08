# DJ Engine Improvement Plan — Code-Verified (July 2026)

Supersedes §5 of `COMMERCIAL_DJ_LANDSCAPE.md`. That backlog was written against `CLAUDE.md`'s
architecture *description*. This one is written against the actual source, read directly and
cross-checked by three independent passes (two subagents + my own file reads) — every claim below
has a file:line citation. Two of the original six items turned out to be mis-scoped, one premise was
wrong, and one finding is bigger than any single item: **all four open correctness bugs in
`REMIX_QUALITY_INSIGHTS.md` are already fixed.** That changes what's safe to build next.

---

## 0. Read this first: the correctness baseline is clean

`REMIX_QUALITY_INSIGHTS.md` (dated this session, before this verification pass) lists four open
findings. I re-checked each directly against current code. All four are resolved:

| Finding | REMIX_QUALITY_INSIGHTS.md said | Verified current state |
|---|---|---|
| #1 — compatibility formula didn't match documented SetFlow weights, no genre/timbre/vocal-clash term | Open — "did not close the underlying gap" | **Fixed.** `track_metadata.py:790-939` implements the full documented formula (0.35 harmonic + 0.25 bpm + 0.15 energy + 0.15 genre_proximity + 0.10 timbral_similarity − vocal_clash_penalty) plus a 4-condition veto (key/bpm/vocal-clash). Docstring at line 803 explicitly references finding #1 in the past tense. |
| #2 — BPM octave errors (e.g. 63.8 vs 129.2) feed directly into render math | Open — "not yet root-caused" | **Fixed.** `beat_tracker.py:115-215` — `resolve_bpm_octave()`, exactly the "dedicated classifier branch" `DJ_THEORY.md` §7 called for. Autocorrelation-scored octave candidates + a log-normal prior centered at 122 BPM. Wired into both the primary and fallback tempo paths (lines 242, 308). |
| #3 — stretch-ratio clamp silently absorbed octave errors instead of surfacing them | Open — "nothing upstream flags *why*" | **Fixed.** `dj_engine.py:289-305` now logs a `WARNING` distinguishing a routine clamp from one that "hugs the safety boundary," naming octave-detection error as the likely cause and citing findings #2/#3 directly in the log message. |
| #4 — limiter's attack/release shared one time constant, causing hard-clip instead of smooth limiting | Open — "fix shape (not yet applied)" | **Fixed.** `mastering.py:184-238` (`_smooth_gain_envelope`) now runs a genuine two-coefficient envelope follower — separate `alpha_attack`/`alpha_release`, numba-JIT'd with a pure-Python fallback. Docstring explicitly names finding #4. |

**Why this matters beyond bookkeeping:** `SELF_IMPROVING_DJ_RESEARCH.md` (the project's own,
already-written personalization plan) states its own precondition at the top: *"Depends on:
REMIX_QUALITY_INSIGHTS.md fixes (#1–#4) being live."* That precondition is now met. Its Stage A
(behavioral telemetry) is unblocked — see Phase 4 below. `REMIX_QUALITY_INSIGHTS.md` itself should
get a status update so the next person reading it doesn't re-litigate fixed bugs.

---

## 1. Corrections to the original 6-item backlog

| # | Original framing | What the code actually shows | Correction |
|---|---|---|---|
| 1 | Add a confidence gate to the render pipeline | Confirmed real gap — `use_stem_blend`/`have_stems` only check file *existence*, never compatibility (`task_modules/remix.py:186-194`, `dj_engine.py:1315-1327`) | Scope unchanged, but now with exact insertion point + which signals to reuse (§2.1) |
| 2 | Build segment-level key confidence | **`detect_key_segments()` already exists** (`key_detection.py:375-416`, windowed, confidence-scored) — it has **zero callers**. `analyze_structure()` only ever calls the whole-track `detect_key()` (`dj_analysis.py:372-374`) | This is "wire up dead code," not "build new code" — much smaller task (§1.3) |
| 3 | Fast preview-tier separation | Confirmed real gap — `stems.py` always runs full Demucs, no preview/fast param anywhere | Scope unchanged (§2.2) |
| 4 | Learned differentiable transition controller (DJTransGAN-style) | `spectral_matching.py`'s existing mechanism is 24 per-mel-band **similarity-scoring weights** (`data/spectral_match_model.json`: `{version, n_bands:24, weights:[...], n_updates}`), used only in `match_score()`/`choose_entry_bar()` — it re-ranks candidates, it never touches audio | Framing corrected: this is a candidate *generator* to compare against a scoring mechanism, not a drop-in replacement (§5.1) |
| 5 | Extend the 👍/👎 loop from transition-point to next-track selection | **Wrong premise.** I assumed the REMIX_QUALITY_INSIGHTS.md bug blocking this was in `setlist_planner.py` — it isn't; `setlist_planner.py`'s cost function (`transition_cost()`, lines 185-242, 0.50/0.30/0.20 harmonic/bpm/energy) matches its own docstring exactly, no bug there. The actual bug was in `track_metadata.py` (§0, finding #1) and it's fixed. More importantly: `SELF_IMPROVING_DJ_RESEARCH.md` already designed this — in far more depth than my one-paragraph version, including the finding that **zero behavioral telemetry exists today** (no skip/replay/reorder tracking) | Retracted as a standalone item. Redirect to the existing doc's Stage A (§4) |
| 6 | Publish a versioned analysis API | Confirmed real gap — current `analysis.py` routes (`POST /compatibility`, `GET /compatibility/matches/{name}`, `POST /analyze`, `GET /recommend/{name}`, `GET /library/similar/{name}`, `GET /index/stats`, `POST /index/rebuild`, `GET /library/crate-search`, `POST /library/build-clap-index`) expose nothing resembling bars/beats/segments | Scope unchanged, now additive not a rework (§3.1) |

**One new item, not in the original six, that outranks most of them:** `vocal_analyzer.py` is fully
built and tested (CREPE F0, voiced-fraction, phrase timing — `tests/test_vocal_analyzer.py` exists)
and has **zero integration anywhere**. The `vocal_clash_penalty` that just got wired into
`compatibility_score()` (§0, finding #1) uses a cheap proxy — `vocal_density`, a Demucs-stem
energy ratio (`music_intelligence.py:111,366`) — not this richer signal. `FURTHER_STUDIES.md`
already flags this as *"the highest-value next step in the compatibility-scoring area"* — I'm not
discovering this, I'm surfacing an existing, correct call that hadn't made it into the commercial-
landscape comparison. See §1.2.

---

## 2. The plan, phased by dependency and risk

**Phase 1 — zero new infrastructure, do first**
1.1 Run the LUFS calibration that already exists but has never been executed
1.2 Wire `vocal_analyzer.py` into `vocal_clash_penalty`
1.3 Wire the dead `detect_key_segments()` into `SongStructure`

**Phase 2 — engine robustness, no new dependencies**
2.1 Confidence-gated fallback in the render path
2.2 Fast preview-tier stem separation

**Phase 3 — public surface**
3.1 Versioned bars/beats/segments analysis endpoint

**Phase 4 — personalization foundation (execute the existing plan, don't re-design it)**
4.1 `SELF_IMPROVING_DJ_RESEARCH.md` Stage A — now unblocked per §0

**Phase 5 — research track, lowest priority, feature-lab only**
5.1 DJTransGAN-style learned candidate generator, benchmarked against the existing scoring weights

---

### 1.1 — Activate the LUFS calibration you already built

Not a code task. `data/stem_lufs_targets.json` does not exist on disk — `normalize_stems_to_corpus_targets()`
(`mastering.py:687-751`) is fully functional but has silently been running on `_STEM_FALLBACK_TARGETS`
this whole time because `POST /library/calibrate-lufs` (which calls `analyze_library_stem_targets()` +
`save_stem_targets()`, `mastering.py:565-684`) has never been run against the real library. Run it
once. This is the fastest, lowest-risk win in this whole document — it activates a feature that
already exists.

### 1.2 — Wire `vocal_analyzer.py` into `vocal_clash_penalty`

```text
Goal: Replace the vocal_density energy-ratio proxy currently driving vocal_clash_penalty
(track_metadata.py:888-892) with vocal_analyzer.py's richer signal (voiced_fraction, mean_pitch_hz,
phrase timing) so the penalty reflects actual pitch-register/rhythm overlap between two tracks'
vocals, not just "both tracks happen to have vocals."

Constraints:
- vocal_analyzer.py has zero API exposure today (FURTHER_STUDIES.md 1A, confirmed no route in
  scripts/api/routers/ or schema in schemas.py). Decide: standalone POST /vocal-analysis, or fold
  into POST /analyze as an optional field when vocal stems exist. Either way, add a schema.
- Keep vocal_density as the graceful-degradation fallback when Demucs vocal stems aren't available
  — this codebase already has that pattern everywhere else (e.g. BeatThisTracker → LibrosaBeatTracker).
- Don't touch tests/test_vocal_analyzer.py's existing coverage — extend it.

Files: scripts/core/vocal_analyzer.py (read the existing output shape, don't rewrite the analysis),
scripts/core/track_metadata.py (~880-892), scripts/core/music_intelligence.py (vocal_density,
~111 and ~366), scripts/api/schemas.py, scripts/api/routers/analysis.py (if exposing standalone),
tests/test_vocal_analyzer.py.

Definition of done: two tracks that both have strong vocals but in non-overlapping pitch
register/timing score a meaningfully LOWER vocal_clash_penalty than two tracks whose vocals
actually overlap in register and rhythm — a distinction the current energy-ratio proxy cannot
make. Existing vocal_analyzer tests still pass.
```

### 1.3 — Wire the existing `detect_key_segments()` into `SongStructure`

```text
Goal: Replace the single whole-track detect_key() call in analyze_structure()
(dj_analysis.py:372-374) with the already-built, currently-unused detect_key_segments()
(key_detection.py:375-416 — 30s windows, 50% overlap, per-segment KeyResult with confidence), so
SongStructure/Section carry per-phrase key + confidence instead of one global guess that's wrong
for any track that modulates key mid-song.

Constraints:
- CAMELOT and NOTE_NAMES (key_detection.py:38,41-48) are the single source of truth — don't touch.
- Section (dj_analysis.py:68-90) currently has no key/confidence fields. Grep for existing
  Section(...) construction sites before changing its shape — this is a data-model change, not
  purely additive.
- Keep the track-level key_confidence field on SongStructure for backward compatibility (derive it
  as an aggregate of segment confidences, or keep the original whole-track call as a fallback —
  don't just delete it).

Files: scripts/core/dj_analysis.py (SongStructure ~93-168, Section ~68-90, analyze_structure
~372-374), scripts/core/key_detection.py (detect_key_segments, already built), scripts/core/tiv_scoring.py
(gate/weight harmonic scoring by the new per-phrase confidence instead of trusting one global key).

Definition of done: a track with a synthetic mid-song key change (pitch-shift-spliced test fixture)
produces two different phrase-level key calls with a visible confidence drop at the boundary,
instead of one averaged/wrong whole-track key. Nothing downstream that reads SongStructure.key_name
breaks silently.
```

### 2.1 — Confidence-gated fallback in the render path

```text
Goal: Add a compatibility gate in scripts/api/task_modules/remix.py, between apply_spectral_matching()
(line 146) and the use_stem_blend/render dispatch (lines 194-221), that falls back to a shorter,
simple equal-power crossfade instead of the full stem-aware pipeline when compatibility is low —
using signals ALREADY computed at that point (plan.harmonic_score, the TIV-based consonance set in
dj_analysis.py:816-819; and spectral_info["score"] returned by apply_spectral_matching, already in
[0,1]) rather than inventing a new metric.

Constraints:
- current-engine only — never touch scripts/core/classic_engine/, it's a frozen vendored snapshot.
- New config.yaml keys need a matching dataclass field on DJConfig (config.py:79-102) — all 12
  existing dj.* keys currently match 1:1; keep that invariant or tests/test_config_drift.py will
  catch the drift (it already has 4 tests guarding exactly this).
- Don't do async I/O inside the task function — job_store.submit_job() runs it in a
  ThreadPoolExecutor.

Files: scripts/api/task_modules/remix.py (~140-221), scripts/core/dj_engine.py (fallback render
path — reuse the existing plain render() path, which already exists for the no-stems case, rather
than building a third one), config.yaml (dj: section), scripts/core/config.py (DJConfig).

Definition of done: a deliberately low-compatibility pair (low harmonic_score AND low spectral
match score) renders via the fallback path; job metadata records which path was used; a new case
in tests/test_behavioral.py (which already has 24 passing tests in this area) covers it; nothing
else in that file regresses.
```

### 2.2 — Fast preview-tier stem separation

```text
Goal: Add a lower-quality/faster separation tier to scripts/core/stems.py — confirmed today that
separate_song_stems() always runs full Demucs on the whole file with no preview/duration-limited/
lower-res param anywhere — for interactive contexts (live preview, PiP widget), leaving the default
full-quality path completely untouched.

Constraints: benchmark on real hardware before picking an approach — device auto-detection
(_detect_device(), stems.py:111-124, with automatic CPU retry-on-GPU-failure at 246-254) is already
solid; reuse it, don't rebuild it. Don't change the output format of the full-quality path — final
renders must stay byte-comparable to today's.

Files: scripts/core/stems.py, config.yaml (separation: section — note there's currently no
mastering: section in config.yaml, so don't assume one exists when adding new keys elsewhere),
frontend call sites that currently block on full separation for a live preview.

Definition of done: a benchmark (throwaway script, not committed) showing wall-clock time for full
vs. fast-path separation on 3 real library songs, wired into whichever preview flow currently waits
on full separation.
```

### 3.1 — Versioned bars/beats/segments analysis endpoint

```text
Goal: Add a versioned GET endpoint under scripts/api/routers/analysis.py exposing the bars/beats/
segments/per-phrase-key-confidence structure that dj_analysis.py computes internally (richer after
1.3 lands). Confirmed current routes — POST /compatibility, GET /compatibility/matches/{name},
POST /analyze, GET /recommend/{name}, GET /library/similar/{name}, GET /index/stats,
POST /index/rebuild, GET /library/crate-search, POST /library/build-clap-index — none of these
expose structure directly, so this is additive.

Constraints: version it (e.g. /analysis/v1/{name}/structure) so later internal changes to
dj_analysis.py's data model don't silently break external consumers. Router registration order
matters (analysis is deliberately registered before library in routes.py) — add a case to
tests/test_route_shadowing.py (which already has 8 tests guarding exactly this) proving the new
static route isn't shadowed and doesn't shadow anything else.

Files: scripts/api/routers/analysis.py, scripts/api/schemas.py, frontend/src/lib/api.ts,
tests/test_route_shadowing.py.

Definition of done: hitting the endpoint for a library track returns a documented, versioned JSON
shape with bars/beats/segments/key+confidence; a short note in ARCHITECTURE.md.
```

### 4.1 — Execute `SELF_IMPROVING_DJ_RESEARCH.md` Stage A (don't redesign it)

Not a new prompt — this document already has one, in more depth than anything I'd write fresh:
`SELF_IMPROVING_DJ_RESEARCH.md` §3, which specifies `scripts/core/feedback_store.py`, a new
`data/feedback.db` (mirroring the existing `jobs.db` write-through pattern), a `feedback_events`
table (event_type, song_a, song_b, job_id, position_sec, context, created_at), and a new router
(`POST /feedback/event`, `GET /feedback/summary`). Its own stated gate — *"Depends on:
REMIX_QUALITY_INSIGHTS.md fixes (#1–#4) being live"* — is satisfied per §0 of this document. This
is the real version of my original, thinner "extend the feedback loop to setlist selection" idea;
go implement that doc's Stage A as written rather than a shallower substitute. Stage A is explicitly
scoped to *collect data, learn nothing yet* — 2-3 weeks of telemetry before any model work starts.

### 5.1 — DJTransGAN-style learned candidate generator (research track)

```text
Goal: Prototype a small differentiable EQ+fader controller (in the spirit of DJTransGAN,
arXiv:2110.06525) in ai_remixmate_feature_lab/ (isolated by design, not wired to the UI), as an
alternative candidate generator to compare against — not replace — spectral_matching.py's existing
mechanism, which is confirmed to be 24 per-mel-band SIMILARITY-SCORING weights (data/spectral_match_model.json)
used only to re-rank entry-bar candidates in choose_entry_bar(); it never outputs audio-affecting
parameters. A learned controller that actually competes with it would need to output something
render-affecting (fade curve, EQ), which is a bigger step than what exists today — scope this
explicitly as a research prototype, not a production swap-in.

Constraints: bootstrap dataset from data/spectral_match_history.json (session_id-keyed, already
has sims/entry_bar/score/ts) before reaching for external DJ-mix scraping.

Files: ai_remixmate_feature_lab/ (new module), read-only reference to data/spectral_match_model.json
and data/spectral_match_history.json.

Definition of done: a notebook/script reporting whether the learned controller's predicted
parameters correlate with 👍 verdicts better than the current hand-tuned similarity weights — a
research result, not a merged feature.
```

---

## 3. What I'm explicitly not re-proposing, and why

- **"Fix the compatibility-score bug before building on it."** Already fixed — verified by reading
  `track_metadata.py:790-939` directly, not inferring it from a doc.
- **A from-scratch setlist-level feedback loop.** `SELF_IMPROVING_DJ_RESEARCH.md` already designed
  this in more depth (staged A→D, an explicit signal inventory, a bounded-drift model, an offline
  counterfactual evaluation step) than anything worth writing fresh. Redoing it thinner would be a
  regression, not a contribution.
- **Segment-level key detection as new engineering.** It's not new engineering — `detect_key_segments()`
  already exists. The actual task is integration (§1.3).
