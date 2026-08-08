# AI RemixMate — Full-Surface QA Prompts

Four independent Claude Code prompts, one per discipline. Run them as **separate
sessions**, not one combined pass — a single session context-switching between
DSP math, security adversarial review, TS/Pydantic diffing, and running pytest
produces shallower work in all four than four focused passes do. Fresh context
per session also matters for the security pass specifically: zero-implementer-
bias is what caught the `force=True` bug in the last review, and that only
works if the reviewer hasn't seen how the code got written.

Run order if sequential: Phase 4 first (cheap, tells you if you're starting
from a broken baseline), then 1–3 in any order.

---

## Phase 1 — Behavioral / Audio Correctness
**Model: Opus.** This needs domain reasoning (why equal-power vs equal-gain,
why the classic engine is frozen, what a correct beat-grid lock actually
looks like at the sample level) combined with writing and interpreting real
audio-analysis code. Getting this wrong silently ships a DJ engine that
compiles and passes shape-only tests but sounds broken — that's the whole
reason `tests/test_behavioral.py` exists as a separate category from the rest
of the suite.

```
You are doing an adversarial behavioral-correctness review of AI RemixMate's
audio engine — not "does it run," but "does it actually sound like the
docs/comments claim." Assume nothing has been verified. You have full repo
access; work from scripts/core/dj_engine.py, dj_analysis.py, mastering.py,
spectral_mixing.py, spectral_matching.py, spectral_sync.py, and beat_synth.py,
plus tests/test_behavioral.py as your reference for what "correct" means here.

For each of these claimed properties, write a synthetic-audio test (numpy
sine/noise tracks with known BPM/key — you don't need real music) that
actually measures the property, run it, and report pass/fail with numbers:

1. Beat-grid lock: entry_sample_b = int(entry_time_b * sr / stretch_ratio).
   Verify downbeats land within the claimed ±half-bar clamp after time-stretch,
   across at least 3 different stretch ratios (0.9x, 1.0x, 1.15x) — not just
   the identity case.
2. Stem-aware crossfade: drums/bass from Song B enter early, vocals delayed.
   Measure onset timing per stem in the rendered output, don't just check the
   envelope code exists.
3. Standard crossfade is claimed equal-power (cos/sin), not equal-gain — verify
   the summed output actually dips ~3 dB at the midpoint for two uncorrelated
   inputs. Verify seam blends (_seam_blend(), ~30ms) are equal-gain, not
   equal-power, as claimed — correlated material at a seam behaves differently
   than uncorrelated material mid-transition.
4. _stem_bass_ramp(): true cosine-taper mute at the swap point, no IIR
   ringing/bleed. Confirm by frequency-domain inspection of the bass band
   across the swap boundary.
5. Dynamic EQ (low-shelf) + masking-aware multiband EQ (spectral_mixing.py):
   measure band energy before/after — Song A's bass should measurably drop as
   Song B's rises. Confirm masking-EQ gain curves taper to 0 dB at window
   edges (listen for / measure a click at the splice boundary if they don't).
   Confirm the bass band is untouched by masking EQ regardless of config.
6. Mastering: final output actually hits -14 LUFS via ITU-R BS.1770-4, and the
   true-peak limiter never allows output above its ceiling — test with a
   deliberately hot input, not just a normal-level one.
7. Two-engine split: classic_engine/ is supposed to be a byte-faithful frozen
   vendored copy. Diff it against its original source snapshot if you can find
   one (check git history/CHANGELOG for when it was vendored); at minimum,
   confirm no `current`-engine bugfix in this session's git history touched
   any file under scripts/core/classic_engine/.

Report format: numbered findings, file:line, the specific claim violated, how
you measured it (include the test code), severity (critical/high/medium/nit).
If everything you tested actually holds, say so explicitly per item — don't
just report failures, report what you verified.
```

---

## Phase 2 — Security / Adversarial Audit, Full Surface
**Model: Opus.** Same reasoning as the prior review that found the
`force=True` data-loss bug — that class of bug (a "safety" flag that silently
bypasses the check it's named after) has now shown up **three times** in one
file (`library.py`'s `prune_raw`/`prune_enhanced`/`evict_lru`). Assume it
exists elsewhere until proven otherwise.

```
Adversarial security/correctness review of AI RemixMate's API layer. Zero
implementer context — review as if you've never seen this code and don't trust
any comment claiming something is "safe." Scope: all 12 routers in
scripts/api/routers/, scripts/api/task_modules/, scripts/api/jobs.py,
scripts/core/config.py. (library.py/stems.py were already reviewed this
session — skip those unless you find something the prior review missed.)

Specific patterns to hunt, each already proven to exist at least once in this
codebase:

1. Bypass flags: grep the whole repo for `force=True` / `force=` and for any
   other boolean parameter that skips a named safety check. For each hit,
   trace every call site — is it ever reachable with attacker- or
   race-controlled inputs, or from an automatic/scheduled path (like
   evict_lru's auto_evict_on_download) with no human in the loop?
2. Path traversal: confirm every endpoint that takes a song/file name as input
   routes through the same confinement pattern as _require_song() in
   scripts/api/routers/_helpers.py (regex allowlist + Path().name + resolve()
   + prefix check). Check routers/downloads.py, crates.py, setlist.py,
   spotify.py, generative.py specifically — anywhere a name reaches the
   filesystem or a subprocess call.
3. Stale module-level bindings: _helpers.py's _require_song() binds its own
   copy of LIBRARY_DIR at import time, independent of scripts.core.paths and
   any router's own import — a real bug class already found in this session's
   own test-writing (had to patch three separate copies for one test to work).
   Find every other module that does `from scripts.core.paths import
   LIBRARY_DIR` (or similar) at module scope and flag it as the same fragility,
   even if it's not a security bug per se — it's already caused false test
   failures once.
4. Unlocked read-modify-write: library.py's index race was just fixed with an
   RLock. Check job_store.py's SQLite writes, feedback_store.py, and
   music_index.py's JSON persistence for the same pattern — concurrent
   ThreadPoolExecutor workers reading, mutating, and saving shared state with
   no lock.
5. Fail-open config reads: grep for `getattr(cfg` / `getattr(getattr(cfg`
   patterns. For each, if the getattr fails or config is malformed, does the
   default direction favor safety (fail closed) or convenience (fail open)?
   _prune_after_split() was already fixed to fail closed on config-read
   exceptions — check whether that's the exception or genuinely the norm.
6. Auth/authorization: is there any access control at all on destructive
   endpoints (DELETE /library/{name}, /library/storage/prune, batch stem
   split)? If this is intentionally a single-user local app, say so and skip —
   don't manufacture findings for a non-threat model, but state the assumption
   explicitly so it's a documented decision, not a silent gap.

Report format: numbered findings, file:line, severity (critical/high/medium/
low/nit), reproduction steps, suggested fix. Confirm at least the critical/high
findings by actually triggering them (test or direct reproduction), not just
by reading the code and inferring.
```

---

## Phase 3 — Frontend / Integration / Contract Drift
**Model: Sonnet.** This is systematic diffing and pattern verification, not
deep reasoning — Opus is overkill here and slower for no accuracy gain.

```
Audit contract consistency between AI RemixMate's FastAPI backend and React
frontend. This is mechanical diffing, not creative review — be exhaustive
and literal, not exploratory.

1. Diff every Pydantic response/request model in scripts/api/schemas.py
   against its corresponding TypeScript interface in
   frontend/src/types/index.ts. Flag every field-name mismatch, type mismatch
   (e.g. backend Optional[str] vs. a frontend field typed as required string),
   and any field present on one side but absent on the other. These types are
   manually kept in sync (no codegen) — assume drift exists until you've
   checked every model.
2. CLAUDE.md documents that job progress is 0–100 over SSE/normalized Job
   type, but 0–1 over the raw REST GET /jobs endpoints, and that
   normalizeJob() in api.ts is the single place that should reconcile this.
   Find every component/hook that reads job.progress (or any raw job field)
   and confirm it went through normalizeJob() first — flag any place that
   reads a raw REST job response directly.
3. Cross-check every SSE event type actually emitted by
   scripts/api/routers/events.py (heartbeat, job_created, job_updated,
   job_completed, job_failed, library_changed) against handlers in
   frontend/src/hooks/useSSE.ts, and against the polling fallback path — an
   event type handled over SSE but missing from the polling fallback means
   frontend state silently stops updating whenever SSE disconnects.
4. CLAUDE.md flags that router registration order in scripts/api/routes.py
   matters (analysis before library, or library's dynamic GET /library/{name}
   shadows analysis's static routes) and that tests/test_route_shadowing.py
   guards this. Confirm that test still covers every static route registered
   under a dynamic-prefix router today, including any added after that test
   was last updated — list any static route NOT covered.
5. Spot-check 3–5 API client functions in frontend/src/lib/api.ts against
   their actual router handlers for silently mismatched assumptions (wrong
   HTTP method, wrong path param name, response shape assumed vs. actual).

Report format: numbered findings, file:line on both sides of each mismatch,
severity, and for each one state concretely what breaks at runtime (not just
"types don't match" — what does the user see).
```

---

## Phase 4 — Full Regression Baseline
**Model: Haiku** (or skip Claude Code entirely and just run this yourself —
it's mechanical enough that an agent adds little). Don't spend Opus/Sonnet
budget on running a test suite and reporting the output verbatim.

```
Run AI RemixMate's full test suite and report results with zero
summarization of failures — every failing test's full output, not a count.

1. source .venv/bin/activate
2. pytest -m "not dj_analysis and not integration" \
   --ignore=tests/e2e_test_suite.py --ignore=tests/test_audio_source.py -q
3. pytest -m "dj_analysis" -q
   — IMPORTANT: this subset (librosa/numba-dependent tests) was never run in
   the sandbox session that did the recent library.py/stems.py work, because
   librosa/numba can fail to initialize there. This is the first real run of
   this subset against those changes — treat any failure here as new
   information, not a known/pre-existing issue.
4. pytest tests/test_audio_source.py tests/e2e_test_suite.py -q
   (these were excluded from the "safe" subset above — run them separately
   and report whether they need a live API server or other setup to pass)
5. ruff check . && black --check . && mypy scripts/
6. cd frontend && npm run typecheck && npm run lint -- --max-warnings 0 && npm test

Report: pass/fail/skip counts per command, and the complete output of every
failure (not truncated, not paraphrased). If a command can't run at all
(missing dependency, no live server, etc.), say exactly what's missing rather
than skipping it silently.
```
