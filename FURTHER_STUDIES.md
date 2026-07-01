# AI RemixMate — Further Studies

> Everything genuinely NOT yet built, as of June 30, 2026. Read `CONTEXT.md` first for current
> state — this file is its complement: nothing here should already exist in the codebase. Every
> item below was checked against actual files (`ls`/`grep`), not just against other docs, because
> several other planning docs in this repo (`IMPROVEMENTS_V2.md`, `AGENT_PROMPT.md`) had drifted
> and listed already-built things as still-open.
>
> Confidence tags: **[Verified]** = confirmed absent/present via direct file/code check this
> session. **[Likely]** = strong inference, not directly re-verified this session. **[Unverified]**
> = carried over from another doc, not independently checked — verify before trusting.

---

## 1. Stage 4 — analysis/quality gaps (from IMPROVEMENTS_V2.md)

### 1A. Wire `vocal_analyzer.py` into the product — [Verified] module exists, zero integration

`scripts/core/vocal_analyzer.py` is fully built and tested (`tests/test_vocal_analyzer.py`, CREPE
F0 + autocorrelation fallback, vibrato detection, phrase counting, energy dynamics). It has:

- **No API route.** Nothing in `scripts/api/routers/` or `scripts/api/schemas.py` exposes
  `VocalReport`/`F0Curve`. A fresh agent could reasonably add `POST /vocal-analysis` or fold it
  into `POST /analyze` as an optional field when vocal stems exist.
- **No wiring into `compatibility_score()`'s `vocal_clash_penalty`.** The June 30 fix to
  `track_metadata.py` added `vocal_clash_penalty` using a cheap proxy (`TrackMetadata.vocal_density`
  — a Demucs-stem energy ratio or spectral-band heuristic, see `CONTEXT.md` Section 5). The richer
  signal `vocal_analyzer.py` produces (`voiced_fraction`, `mean_pitch_hz`, phrase timing) isn't used
  there at all. Two real tracks don't necessarily clash just because both have vocals — overlapping
  *pitch register and rhythm* is what actually clashes. `vocal_analyzer.py`'s output is the right
  input for a much better clash heuristic than the current energy-ratio proxy. This is the highest-
  value next step in the compatibility-scoring area.
- **No wiring into transition planning.** `dj_analysis.py:plan_transition()` doesn't know where
  vocal phrases start/end, so cue points can land mid-phrase on a vocal-heavy track. Phrase
  boundaries from `VocalReport` could feed `_detect_phrase_boundaries()`'s existing downbeat-snap
  logic.

### 1B. CUE-DETR (ML cue-point detection) — [Verified] not started

No `scripts/ml/` directory exists at all. `IMPROVEMENTS_V2.md`'s "Module: Vocal Analyzer... new
module (Stage 4)" entry is now done (see 1A); the CUE-DETR entry in the same table is still
accurate — nothing has been built. Current cue-point detection is SSM-novelty boundary detection
snapped to the bar grid (`dj_analysis.py:_detect_phrase_boundaries()` + `_snap_to_bar_grid()`), not
a learned detector. Lower priority than 1A — the existing heuristic is reasonable; this would be a
quality refinement, not a gap with a known symptom.

### 1C. B-Roll matching (CLIP + CLAP multimodal) — [Verified] not started

`scripts/core/broll_matcher.py` doesn't exist. No video/visual capability exists anywhere in the
codebase. This is the least-started, most speculative item on this list — there's no existing
partial implementation to extend, and no clear user-facing demand signal recorded anywhere (it
appears in `IMPROVEMENTS_V2.md`'s gap table as a research-report recommendation, not as something
the maintainer asked for independently). Deprioritize until 1A/1B are done.

### 1D. Global energy-arc optimizer for `setlist_planner.py` — [Verified] still greedy

`setlist_planner.py`'s own docstring says "greedy approach runs in O(n²)" — confirmed still true.
The planner does pairwise transition-cost optimization only; it can't see two moves ahead, so
locally-fine transitions can still produce a globally-incoherent set. `docs/DJ_THEORY.md` section 7
frames this as a constrained-optimization (TSP-adjacent) problem. This is the biggest lift on this
list and the least likely to be "why one specific remix sounds bad" — it's a full-set coherence
problem, not a per-transition one. Lowest priority of the Stage 4 items unless the actual complaint
becomes "the whole set doesn't flow," not "this one transition is off."

---

## 2. Schema drift & CI gaps

- **No `openapi-typescript` generation in CI.** `frontend/src/types/index.ts` mirrors Pydantic
  schemas by hand. This has caused at least two real bugs (`favoritesApi.list` shape mismatch,
  `MixDeck` needing a separate per-song fetch because the bulk endpoint silently omits analysis
  fields) — both fixed reactively, not prevented. `.github/workflows/` only has `codeql.yml`
  (security scanning) and `pages.yml` (GH Pages deploy) — **[Verified]** no test/build/typecheck
  workflow exists at all. Adding even a bare `pytest` + `tsc --noEmit` GitHub Action on PR would
  catch a meaningful fraction of the bug classes that have been fixed reactively across the last
  several sessions.
- **`scripts/api/schemas.py` ↔ `frontend/src/types/index.ts` sync is manual and undocumented as a
  process.** No pre-commit hook, no codegen step, no test that fails on drift. Same root cause as
  above; the long-term fix (`openapi-typescript`) and the cheap interim fix (a test that diffs field
  names) are both still open.

## 3. Test coverage gaps

- **[Unverified, carried from `AUDIT.md`/`SESSION_HANDOFF.md`]** `tests/test_router_smoke.py` exists
  (confirmed present), but its breadth against the Stage 2/3 endpoints (`/library/crate-search`,
  `/library/build-clap-index`, `/library/{name}/export-cues`, `/library/calibrate-lufs`,
  `/setlist/*`) hasn't been audited this session — verify which routes it actually exercises before
  assuming full coverage.
- **One known-broken test, unrelated to anything fixed this session:**
  `tests/test_behavioral.py::TestClashPathBassSwap::test_bass_swap_bar_at_midpoint_after_clash_shortening`
  fails with `AttributeError: can't set attribute 'total_bars'` — `SongStructure.total_bars` is a
  read-only `@property` and the test tries to set it directly on a constructed instance. Either give
  the test a proper way to construct a `SongStructure` with an explicit `total_bars`, or refactor
  `total_bars` to a settable field with a computed default. **[Verified]** this is real and
  reproduces; not touched, not yet root-caused beyond the immediate symptom.
- **`task_modules/` + most routers still have shallow coverage relative to their surface area** —
  `AUDIT.md`'s original Dimension 3 findings here are mostly still accurate; `jobs.py`,
  `setlist_planner.py`, and `mastering.py` have since gotten real test coverage, but
  `task_modules/{download,stems,generative,lab}.py` likely still don't. **[Unverified]** — re-audit
  before trusting either way.

## 4. Carried over from `SESSION_HANDOFF.md` — needs a live browser session to verify

These were fixed in code but never re-verified against the running app (the session that fixed them
ran out before getting back to verification, and no session since has had a live app + browser to
check):

- **Storage/index phantom-entry fix.** `LibraryManager.unregister()` was added and wired into the
  delete route; `list_songs()` was made self-healing for stale entries. Not re-verified live.
- **Naming fix.** Unified song-naming policy (`scripts/core/paths.py:sanitize_song_name()`) was
  meant to stop a specific repeated-400 bug on a song with an apostrophe/parenthesis in its name.
  Not re-tested against that specific song post-fix.
- **Job-card click/confirm feature.** `RightInspector.tsx` job cards should navigate on click with a
  `window.confirm()` guard for RUNNING/PENDING jobs. `tsc --noEmit` was clean; never clicked in an
  actual browser.
- **Aggressive-test sweep was incomplete** the session that wrote this list: Set Builder chain-remix
  render, Signal Search re-scoring after the `has_analysis()` fix, and full AI Lab job execution
  were flagged as untested and never run since.

## 5. Self-improving DJ (personalization from user behavior)

Full proposal in `SELF_IMPROVING_DJ_RESEARCH.md` — not duplicated here. Short version: the
prerequisite (behavioral telemetry — skips, replays, manual overrides) doesn't exist in the
codebase at all yet (**[Verified]** — no `feedback_store.py`, no skip/play tracking anywhere, only
binary `favorites`/`crates`). That doc's Stage A (instrumentation) is the actual next step if this
direction is pursued; the doc is explicit that Stages C-E (the actual learning) are gated on Stage
B showing real signal in the logged data, and may legitimately not be worth building if that gate
fails. Don't start coding a recommendation model before reading that doc's Section 0.

## 6. Doc hygiene — cleanup recommended, needs your decision (can't delete workspace files without confirmation)

| File(s) | Recommendation | Why |
|---|---|---|
| `AUDIT.md` + old `CONTEXT.md` content (now overwritten) | Archive or delete | All D1-critical findings fixed June 28; this file's only remaining value is historical record |
| `IMPROVEMENTS.md` | Archive or delete | Stages 1-3 fully done; nothing actionable remains |
| `IMPROVEMENTS_V2.md` | Archive or delete | Stages 1-3 done; Stage 4 content is now more accurately captured in Section 1 above |
| `REMIX_QUALITY_INSIGHTS.md` | Archive or delete | Findings #1-4 fixed June 30; finding #5 was a "confirm you're on current main" check (moot); finding #6 (vocal analyzer, energy-arc optimizer) is now Section 1A/1D above |
| `SESSION_HANDOFF.md` | Archive or delete | Fixed items now in `CONTEXT.md`; open items now in Section 4 above |
| `claude-code-improvements.md` through `-7.md` (7 files) | Archive or delete | Confirmed implemented (frontend components + render-path hardening both verified present in code); zero remaining actionable content |
| `docs/COMPASS_ARTIFACT.md` | **Needs a decision, not just archival** — this file is a near-duplicate of `docs/TOKENIZATION_ROADMAP.md`'s content despite `docs/README.md`'s index describing it as a "vision, scope, what's in/out" doc. Either (a) delete it and fix the index entry, or (b) actually write the vision/scope doc it's supposed to be and keep the name |
| `docs/GUIDE.md`, `docs/QUICK_REFERENCE.md`, `docs/COMPREHENSIVE_DOCUMENTATION.md`, `docs/DOCUMENTATION_INDEX.md` | Needs a decision — these describe the Streamlit-era architecture in real depth (not just a passing mention). Either rewrite for the React-era architecture or clearly mark as historical/archived. Don't delete blind — some DSP/algorithm explanations in `COMPREHENSIVE_DOCUMENTATION.md` may still be accurate even though the surrounding architecture description isn't; worth a skim before deciding file-by-file |
| Root `README.md` | **Fix, don't archive** — only the "Streamlit primary / React next" framing (Quick Start section + Status section + tech-stack table) is stale. The rest (what it does, why it was built, tech stack list minus that one row) is accurate and should stay |
| `docs/TOKENIZATION_ROADMAP.md` | Keep as-is — legitimate reference material, not a status doc |
| `docs/DJ_THEORY.md` | Keep as-is — evergreen musicological reference |
| `docs/AI_RemixMate_Launch_Plan.md`, `docs/PORTFOLIO_INTEGRATION.md` | Keep as-is — business/marketing docs, not implementation status, out of scope for this technical cleanup |
| `CHANGELOG.md` | Keep as-is, but it stopped getting entries after April 2026 — if it's meant to stay current, the June sessions' work (Stage 1-3 improvements, the June 28 audit fixes, the June 30 DSP fixes) should get entries |

I did not delete any files — workspace files require your explicit confirmation to remove. If you
want me to actually archive (move to an `archive/docs/` folder) or delete the superseded files
listed above, say so and I'll do it.
