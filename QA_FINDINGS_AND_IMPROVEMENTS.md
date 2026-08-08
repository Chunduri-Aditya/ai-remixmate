# AI RemixMate — Full-Surface QA Findings & Improvements

Consolidated results from the four-phase QA pass (`QA_TESTING_PROMPTS.md`). All four phases have now
run: Phase 1 (behavioral/audio correctness), Phase 2 (security/adversarial audit), Phase 3
(frontend/integration contract drift), Phase 4 (regression baseline — 677/677 backend + 32/32
frontend tests pass; see bottom for the pre-existing lint/type debt Phase 4 also surfaced, which is
unrelated to this QA pass and not touched here).

**Status legend:** ✅ Fixed & tested · 📝 Docs corrected (not a code bug) · ⚠️ Open, needs a fix ·
ℹ️ Investigated, no action needed

---

## Priority queue (everything still open, ranked)

| # | Finding | Severity | Phase | Status |
|---|---|---|---|---|
| 1 | Unauthenticated arbitrary-directory file write via `POST /download` `name` | **HIGH** | Security | ⚠️ Open |
| 2 | `JobResponse` missing `meta` field — permanent job-store corruption | **HIGH** | Frontend | ⚠️ Open |
| 3 | Arbitrary local-file read / likely SSRF via `bridge_beat_path` | MEDIUM | Security | ⚠️ Open |
| 4 | `CompatibilityResult` phantom `transition_plan` field (frontend) | MEDIUM | Frontend | ⚠️ Open |
| 5 | `SongInfo` — 6 backend fields missing frontend types, 1 phantom field | MEDIUM | Frontend | ⚠️ Open |
| 6 | `DJPreviewRequest` — preview never sends `transition_effect` | MEDIUM | Frontend | ⚠️ Open |
| 7 | `library_changed` SSE event documented/handled but never emitted | MEDIUM | Frontend | ⚠️ Open |
| 8 | Polling fallback doesn't refresh heartbeat/uptime/completion-log state | MEDIUM | Frontend | ⚠️ Open |
| 9 | `/compatibility` skips `_require_song` confinement | LOW/MED | Security | ⚠️ Open |
| 10 | `DJRemixRequest` — 3 dead frontend fields, `bridge_beat_path` unreachable from UI | LOW | Frontend | ⚠️ Open |
| 11 | Several backend schemas have zero frontend type mirror | LOW | Frontend | ⚠️ Open |
| 12 | `tagsApi.list/songTags` missing response unwrap (dormant, will break on use) | LOW | Frontend | ⚠️ Open |
| 13 | `/spotify/import` writes `meta.json` via unsanitized `query` as path | LOW | Security | ⚠️ Open |
| 14 | `auto_evict_on_download` fails open (`True`) on config-read exception | LOW | Security | ⚠️ Open |
| 15 | `test_route_shadowing.py` blind to dynamic-vs-dynamic route collisions | LOW | Frontend | ⚠️ Open |
| 16 | Unbounded PENDING job queue (TOCTOU on the running-job cap) | LOW (DoS) | Security | ⚠️ Open |
| 17 | `/beat/synthesize` raw `genre` interpolated into filename | NIT | Security | ⚠️ Open (defensive only — confirmed not exploitable) |
| 18 | `crates.py` `{song_name:path}` params unvalidated | NIT | Security | ⚠️ Open (hygiene only) |

**Already resolved this session**, for reference: true-peak limiter (Phase 1 #2, fixed+tested),
stem-crossfade doc correction (Phase 1 #1, docs fixed), masking_eq_strength doc clarification
(Phase 1 #4), plus the earlier library.py/stems.py/download.py batch (index race, force=True bugs,
zero-byte stem validation, etc. — see git log, not repeated here).

---

## Phase 1 — Behavioral / Audio Correctness

Method: synthetic test audio (sine/noise/click trains, known BPM/key), measured end-to-end, not
asserted from docs.

### Findings

**1. Stem entry timing is similarity-driven, not stem-type-driven** — 📝 **docs corrected**
- Claim tested: "Drums/bass from Song B come in early; vocals are delayed."
- `_stem_crossfade_curves(trans_samples, similarity)` takes no stem-identity parameter. Measured
  per-stem B-onset with distinct test tones: drums, vocals, and other all entered at the *identical*
  timestamp (54.7% of window) under equal similarity. Only `bass` differs, via its own dedicated
  swap-ramp path — that part of the claim is real.
- Independently re-verified against source in this session (`dj_engine.py:1531-1533`): confirmed the
  `else` branch for non-bass stems calls the identical `_stem_crossfade_curves(trans_samples, sim)`
  with no stem-name argument.
- **Severity note:** called MEDIUM in the original review; this session's assessment is it's closer
  to HIGH — it's not an edge case, it's the default behavior of a documented headline feature.
- **Resolution:** user chose "fix the docs" over "implement the feature" (2026-07). `CLAUDE.md`,
  `ARCHITECTURE.md`, and `AGENTS.md` corrected to describe actual (similarity-driven) behavior.
  Implementing genuine stem-type-aware timing (drums/bass leading vocals by design, not by
  similarity coincidence) remains a real, unclaimed feature opportunity — not on the roadmap here.

**2. "True-peak limiter" was sample-peak only** — ✅ **fixed & tested**
- `apply_limiter()` detected/clamped `np.abs(audio)` — sample-domain values only — despite being
  documented as an ITU-R BS.1770-4 true-peak limiter. Measured up to +2.02 dB of inter-sample
  overshoot above ceiling on hot transient material via 4x-oversampled reconstruction, even though
  every discrete sample individually reported within ceiling. Systematic, not just an edge case
  (0.38 dB overshoot even on a "gentle" sine bed).
- **Fix** (`scripts/core/mastering.py::apply_limiter`): detect peaks, apply gain reduction, and
  hard-clip all at a 4x-oversampled rate (ITU-R's own recommended factor), decimate back to native
  rate, then re-measure the actual output's true peak and apply one corrective linear gain if still
  over ceiling. Took three iterations to actually close the gap — worth recording:
  1. Oversample gain envelope only, apply to native audio → left 0.98 dB overshoot on adversarial
     transients.
  2. Also clip at oversampled rate before decimating → improved to 0.76 dB overshoot (decimation's
     reconstruction filter rings on the clip's sharp corner — classic Gibbs behavior).
  3. Added a verify-and-correct pass (re-measure true peak on the actual output, correct if needed)
     → 0.00 dB overshoot, by measurement rather than by assumption.
- New `oversample: int = 4` parameter; `oversample=1` restores the old sample-peak-only behavior as
  an explicit escape hatch.
- Tests: `tests/test_mastering.py::TestLimiterTruePeak` (5 new tests — hot transients, gentle bed,
  oversample=1 escape hatch, length preservation, no NaN/inf). 36/36 mastering tests pass, 598/598 in
  the full safe-subset suite.

### Nits

**3. Mastering's ±20 dB gain clamp under-corrects on pathological input** — ℹ️ **no action needed**
- Only reachable with an absurd +13.8 LUFS input (needs -27.8 dB, clamp allows -20 dB max). Across
  the entire realistic range (-24.8 to +1.2 LUFS) the target was hit exactly (0.00 LU error). The
  original review already did this calibration work; agreed, not worth touching.

**4. `masking_eq_strength=0.0` isn't "off"** — 📝 **docs corrected, formula untouched**
- `gain_db = -(2.0 ** strength) * m` means strength=0.0 still applies a full 1x masking gain, not
  zero — a config field named "strength" naturally reads as a linear 0=disabled scale.
- This is **not a bug**: `2**strength` is Hafezi & Reiss's own Eq.(2), cited directly in the
  docstring — the shipped default is paper-faithful. Changing the formula would deviate from the
  reference being implemented, a bigger call than the original "nit" framing implied.
- **Resolution:** left the math alone. Added a docstring clarification in
  `scripts/core/spectral_mixing.py::masking_gain_curve()` and a config comment in
  `scripts/core/config.py` (`masking_eq_strength` field) stating explicitly that
  `masking_eq_enabled` is the real kill switch, not `strength`.

### Verified — claims that hold exactly as documented

- **Beat-grid lock** — `entry_sample_b = int(entry_time_b*sr/ratio)`, ±half-bar clamp, verified
  across stretch ratios 0.9/1.0/1.15 (residual ≤ 12.7 ms against a 1000 ms clamp).
- **Equal-power crossfade / equal-gain seam** — standard crossfade: `fo²+fi² = 1.0000` everywhere,
  mid-dip +0.01 dB (flat) vs. an equal-gain reference's -2.72 dB dip. `_seam_blend()` of identical
  signals: error 5.96e-08 (correctly equal-gain).
- **`_stem_bass_ramp` cosine taper** — bit-exact zero on both sides of the swap, zero bleed/ringing,
  shape error 0.000 vs. ideal cos².
- **Dynamic EQ + masking EQ edges/protect** — Song A's bass measured -32.7 dB across the swap while
  B's rises in; masking EQ edge taper error < 0.1 (splice-transparent); protected bass band delta
  0.000 dB regardless of strength.
- **`classic_engine/` frozen** — single creation commit, clean working tree, none of this session's
  current-engine fixes touched it. (Byte-diff against the original pre-vendoring snapshot isn't
  possible — that snapshot isn't in this repo's history.)

---

## Phase 2 — Security / Adversarial Audit

**Threat model, stated explicitly (this changes every severity call below):** this is not a safe
"single-user localhost" app by default. `ApiConfig.host` defaults to `"0.0.0.0"`
(`config.py:145`, `main.py:200`, `start.sh`'s uvicorn invocation). Zero authentication or
authorization on any endpoint, including destructive ones. CORS is an allowlist, but CORS only
constrains browsers — non-browser clients ignore it entirely. Net: everything below is reachable by
anyone on the same LAN.

### Findings (most severe first, all ⚠️ open)

**1. HIGH — Unauthenticated arbitrary-directory file write via `POST /download` `name`** — CONFIRMED
by direct reproduction (not just code reading)
- `DownloadRequest.name` (`schemas.py:164`) has no validation. Flow: `download_song` →
  `task_download(name=req.name)` → `TrackSpec(name=…)` → `download_track` → `_resolve_url`
  (`download.py:333`), where `name = spec.name or _sanitize(query)` — a provided `name` is truthy,
  so `_sanitize()` is skipped entirely. Becomes `song_dir(name) = LIBRARY_DIR / name`, then
  `dest_dir.mkdir(parents=True, exist_ok=True)` (`download.py:395`), and yt-dlp writes `full.wav` +
  `full.info.json` (+ Demucs stems, `separate=True` forced) into it. Every other name-taking
  endpoint routes through `_require_song()`'s three-layer confinement; this path bypasses it
  completely. `pathlib` join semantics make it trivial — an absolute `name` resets the join
  (`Path("/lib") / "/abs/path" == "/abs/path"`); relative `../` also climbs out.
- Reproduced live: an attacker-controlled `name` created a directory outside `library/` even with a
  bogus/failing download URL (the `mkdir` fires before the network call resolves). With a real
  audio URL, the attacker also plants/overwrites `full.wav`/`full.info.json` at the chosen path.
- **Fix direction:** run `spec.name` through `sanitize_song_name()` in `_resolve_url`
  (`name = _sanitize(spec.name) if spec.name else _sanitize(query)`, reject empties). Better: also
  validate `DownloadRequest.name` at the schema boundary against the same regex `_require_song`
  uses, so the contract matches. `task_playlist_download` is safe today (sanitizes via
  `_sanitize(entry.get("title"))`) but the central `_resolve_url` fix covers it too.

**2. MEDIUM — Arbitrary local-file read (and likely SSRF) via `bridge_beat_path`** — confirmed by
code
- `DJRemixRequest.bridge_beat_path` / `DJChainRequest.bridge_beat_path`
  (`task_modules/remix.py:218`, also `:452` in `task_dj_chain`) go straight to
  `librosa.load(bridge_beat_path, ...)` with `bridge_beat_mode="file"`, no confinement check —
  unlike `song_a`/`song_b`, which are `_require_song`-guarded. Loaded audio gets mixed into the
  render, downloadable via `/outputs/{id}/…` — any file `ffmpeg`/`audioread` can decode gets read
  back to the attacker. librosa's ffmpeg backend can open `http(s)://` URLs too, which is an SSRF
  primitive against internal endpoints (e.g. cloud metadata services). Files that fail to decode
  raise and get swallowed, so it's not a blind oracle, but it's still an unauthenticated read/SSRF
  primitive.
- **Fix direction:** require `bridge_beat_path` to resolve under `OUTPUTS_DIR` (the only legitimate
  source — `/beat/upload` and `/beat/synthesize` both write there) using the same
  `resolve()` + prefix-check pattern `stream_output` already uses. Reject URLs outright.

**3. LOW/MEDIUM — `POST /compatibility` skips `_require_song` confinement**
- `analysis.py:70,106` — `check_compatibility` passes `req.song_a`/`req.song_b` into `_meta_for(name,
  ...)` unguarded. `_meta_for` does `d = LIBRARY_DIR / name`; if it exists, reads `d/"meta.json"`,
  else calls `run_song_analysis(name)`. A traversal/absolute `name` reads `meta.json` from an
  arbitrary existing directory (a parse-success oracle) and feeds an arbitrary name into
  `run_song_analysis`. Lower impact than #1/#2 since the read target must already exist and parse.
- **Fix direction:** add `_require_song(req.song_a)` / `_require_song(req.song_b)` at the top of the
  endpoint — same one-line pattern already used everywhere else.

**4. LOW — Traversal `meta.json` write via `POST /spotify/import` `query`**
- `spotify.py:200` — the fire-and-forget `_write_meta` thread does
  `song_dir_path = LIBRARY_DIR / query` with the raw, unsanitized search-query string, gates on
  `.exists()`, then writes `meta.json` there. A traversal `query` pointing at an existing directory
  plants/overwrites a `meta.json`. Bonus correctness bug found in the same spot: `query` is the
  search string, not the sanitized on-disk name the download actually produces, so in normal use
  this branch usually silently no-ops (the intended Spotify BPM/Camelot pre-population rarely
  lands).
- **Fix direction:** derive the path from `sanitize_song_name(query)` at minimum, confine to
  `LIBRARY_DIR`. Fixing the correctness bug (deriving the actual on-disk name) is a separate,
  related follow-up.

**5. LOW — `auto_evict_on_download` fails open toward silent deletion**
- `download.py:617-623` — any config-read exception sets `_AUTO_EVICT = True` and
  `_PRUNE_ON_DL = True` in the `except` branch. `_AUTO_EVICT` gates `evict_lru()`, which deletes
  other songs' `full.wav` and then whole song directories over cap — a fully automatic, no-human
  path. A malformed/unreadable config currently fails open toward silent data deletion, the inverse
  of the fail-closed fix already applied to `_prune_after_split()` this session.
- **Fix direction:** default `_AUTO_EVICT = False` in the `except` — deletion should require an
  affirmatively-readable opt-in, not survive a config parse failure.

**Minor — Unbounded PENDING job queue (resource exhaustion, TOCTOU)**
- `_check_job_cap` only counts `status=="running"` (cap 4) against a 2-worker executor, so `PENDING`
  jobs pile up without limit (`/spotify/import-playlist` alone can submit up to 50 in one call).
  Also a TOCTOU: the cap check and `submit_job` aren't atomic, so concurrent requests can both pass.
  Low severity, but a real, reachable DoS vector.
- **Fix direction:** cap on total non-terminal (PENDING + RUNNING) jobs, not just RUNNING; make the
  check-and-submit atomic.

### Nits

**6. `/beat/synthesize`'s raw `genre` interpolated into output filename** — attempted, did NOT
escape (confirmed by direct reproduction, not inference)
- `remix.py:191` builds `f"bridge_{genre}_{int(bpm)}bpm_{bars}bars.wav"` from the raw request param
  even though `render_beat()` normalizes `genre` internally. Tried to trigger a traversal write: the
  fixed `bridge_` prefix glues to the first path segment (`bridge_../` becomes the literal directory
  name `bridge_..`), and libsndfile's `fopen` neither creates intermediate directories nor lets `..`
  resolve through a nonexistent `bridge_<…>/` — the write just fails. Not exploitable. Still worth a
  defensive fix (strip/reuse the normalized genre value in the filename) since raw user input in a
  path is a bad habit even when today's specific construction happens to block the escape.

**7. `crates.py`'s `{song_name:path}` params are unvalidated but not a traversal vector**
- `crates.py:102,130,136,148,168,175` use `{song_name:path}` (slashes allowed), skip
  `_require_song`, no existence check — arbitrary/nonexistent names can be tagged/favorited. But
  every sink is parameterized SQLite (`?` placeholders), never the filesystem — data-integrity
  looseness (orphan rows, junk keys), not SQL injection or path traversal. Light validation would be
  good hygiene, not urgent.

### Patterns verified clean (checked, not assumed)

- **Unlocked read-modify-write:** `jobs.py`, `feedback_store.py`, `music_index.py` all correctly
  serialize mutation + persist under a lock (`jobs.py::_lock`, `feedback_store.py::_lock`,
  `music_index.py::self._lock`); SQLite writes are inside those locks, so `check_same_thread=False`
  is genuinely safe. One caveat: `crates.py`'s `_conn()` has no Python-level lock, relying solely on
  SQLite's own file locking — safe from corruption but can raise `OperationalError: database is
  locked` under concurrent writes (low severity).
- **Command injection:** none found. yt-dlp driven through its Python API with list args, never a
  shell; `ffmpeg`/Demucs `subprocess.run` calls use list argv, not `shell=True`.
- **`_require_song()` itself:** the three-layer confinement (regex allowlist → `Path(name).name` →
  `resolve()` + prefix check) is sound; `stream_output`/`stream_stem` replicate it correctly.
- **Spotify IDs:** validated with `re.fullmatch` before any HTTP call.
- **`force=` bypass pattern:** `storage_prune` and `evict_lru`'s internal calls all use
  `force=False` now (this session's earlier fixes); no reachable `force=True` from an
  attacker-controlled path remains.
- **Stale module-scope `LIBRARY_DIR` bindings** (fragility, not a security bug per se, flagged
  because it already caused test failures this session): `_helpers.py:26` binds its own copy;
  ~15 other modules import it at module scope the same way — `routers/{library,remix,system,
  analysis}.py`, `jobs.py`, `core/{crate_digger,style_transfer,inpainting,codec_tokens,music_index,
  feedback_store,audio_source,analysis_pipeline,mastering}.py`, `download.py`. Fix direction:
  reference `paths.LIBRARY_DIR` at call time instead of importing the value, or add a single
  accessor function.

**Suggested priority (per the review):** fix #1 and #2 first — the two unauthenticated, LAN-reachable
primitives. #3/#4 are one-line `_require_song`/sanitize additions. Given #1's reachability, also
consider defaulting the bind host to `127.0.0.1` unless LAN access is a deliberate requirement —
that's a product/deployment decision, not made here.

---

## Phase 3 — Frontend / Integration Contract Drift

### Part 1 — Pydantic ↔ TypeScript schema diff

**1. HIGH — `JobResponse` has no `meta` field — every REST-polled job silently loses its metadata**
- Backend: `schemas.py:146-157` (`JobResponse`, no `meta` field), `jobs.py:512-524`
  (`job_to_response()` never sets it). Frontend: `Job.meta?: Record<string, unknown>`
  (`types/index.ts:42`), consumed in `MixVault.tsx`, `Operations.tsx`, `RightInspector.tsx`.
- `meta` genuinely is stored server-side at creation (`jobs.py:272`, populated by
  `remix.py:261` with `song_a`/`song_b`/`preview`) and IS included in the SSE payload
  (`_job_to_response_dict()`, `jobs.py:200-211`). Only the REST-facing `JobResponse` model omits it
  — FastAPI's `response_model` strips it even if a handler tried to include it.
- **Concrete break:** `MissionControl.tsx:252` and `Operations.tsx:436` call `jobsApi.list()` once
  on mount; `setJobs`/`upsertJob` do a full-replace with no field merge (`appStore.ts:90-110`). So
  mounting Operations or reloading MissionControl overwrites every job in the shared store with
  `meta: undefined` — including jobs SSE had already populated correctly. Once a job is COMPLETED no
  more SSE events ever arrive for it, so this corruption is **permanent**: MixVault's "hide
  previews" toggle breaks (`!j.meta?.preview` reads true when meta is gone), and song-pair labels on
  job cards across MixVault/Operations/RightInspector go blank.
- **Fix direction:** add `meta: Optional[Dict[str, Any]] = None` to `JobResponse` and populate it in
  `job_to_response()`.

**2. MEDIUM — `CompatibilityResult` — frontend has a phantom `transition_plan` field**
- Backend never sends it (`schemas.py:115-131`, `analysis.py:142-159`); frontend type declares
  `transition_plan?: TransitionPlan` (`types/index.ts:208`).
- **Concrete break:** `MixDeck.tsx:441-444` derives cue-preview markers
  (`cueStartA`/`cueEndA`/`cueStartB`/`cueEndB`) from this field — permanently undefined. Dead
  feature, no error, just silently blank markers.

**3. MEDIUM — `SongInfo` — 6 backend fields missing from frontend type, 1 phantom frontend field**
- Backend always sends `size_mb`, `has_full_wav`, `license_type`, `source`, `last_accessed`, `mode`
  (`schemas.py:51-74`); frontend type (`types/index.ts:47-60`) is missing all six, and declares
  `embedding?: number[]` with no backend counterpart.
- **Concrete break:** any component destructuring these fields gets silent `undefined` with no
  compile error — license/source badges, "last played" sorting can't be built without an unsafe
  cast.

**4. MEDIUM — `DJPreviewRequest` — preview never sends the one field that actually varies the
render**
- Backend fields: `song_a`, `song_b`, `transition_bars`, `transition_effect`
  (`schemas.py:291-299`). Frontend type declares `transition_duration?`/`preset?` (neither exists on
  backend) and omits `transition_effect` entirely.
- **Concrete break:** `MixDeck.tsx:464-470` sends `transition_duration`/`preset` (silently dropped —
  Pydantic's default `extra='ignore'`, no override) and never sends `transition_effect`. Preview
  always renders with server-default `transition_effect="auto"`; the actual Launch Remix afterward
  correctly sends the user's chosen effect (`MixDeck.tsx:491`) — **the preview a user listens to can
  audibly differ from the full render they then commit to.**

**5. LOW — `DJRemixRequest` — 3 dead frontend fields, 1 backend field unreachable from the UI**
- Frontend declares `transition_duration?`, `effects?`, `target_bpm?` — none map to a backend field
  (confirmed dead at `MixDeck.tsx:487-488`, sent and silently ignored). No `bridge_beat_path` field
  in the frontend type at all — grep confirms it's never set anywhere, so the documented
  `bridge_beat_mode: 'file'` (user-uploaded WAV bridge) option is unreachable through the typed
  request, even setting aside Security Finding #2 above.

**6. LOW — Several backend schemas have zero frontend type mirror**
- `GenreResult`, `MusicVectorResponse`, `DJChainRequest`, `QualityReportResponse`,
  `TransitionScoreResponse`, `RemixFeedbackRequest`/`RemixFeedbackResponse` have no counterpart in
  `types/index.ts`. Call sites use `Record<string, unknown>` opts bags (e.g.
  `remixApi.chain(songs, opts?: Record<string, unknown>)`, `api.ts:296`) — any field-name typo in a
  caller compiles cleanly and fails only at runtime.

**7. LOW — `tagsApi.list()`/`songTags()` missing response unwrap (dormant)**
- Backend returns `{"tags": [...]}` / `{"song_name": ..., "tags": [...]}`
  (`crates.py:117-120,130-133`); frontend calls `get<string[]>(...)` with no `.then(r => r.tags)`
  unwrap, unlike `favoritesApi.list` right below it which does unwrap. Same class of bug
  CLAUDE.md documents as already fixed for `favoritesApi`. Currently dormant (zero component usages
  of `tagsApi.list`/`songTags` found) but will break the instant a Tags UI is wired up.

### Part 2 — `job.progress` / `normalizeJob` discipline — ℹ️ confirmed clean

Every REST job read goes through `normalizeJob()`: `jobsApi.list/get`, `downloadApi.single/
batch/playlist`, and all of `useSSE.ts`'s handlers. `useJobTimer.ts` only reads the
already-normalized `Job.progress`. No component reads a raw REST/SSE job field directly. The one
real defect in this area is Finding #1 above — not a `normalizeJob` bug, but the schema gap leaking
through a discipline that otherwise works correctly.

### Part 3 — SSE event types vs. handlers vs. polling fallback

**8. MEDIUM — `library_changed` documented and handled but never emitted**
- `emit_library_changed()` is defined (`events.py:176-178`) but nothing in `scripts/` ever calls it
  — not downloads, library delete, stem-split, or crate mutation. CLAUDE.md and `events.py`'s own
  docstring both list it as emitted; it isn't. `useSSE.ts:120-123` handles it correctly, but the
  handler will never fire. (Backend actually emits 8 types including `job_cancelled` and
  `system_status`, both correctly handled — frontend is more complete than the documented list, it's
  just that one of the documented six is fictional.)
- **Fix direction:** either wire real callers (library delete, download-complete, stem-split-complete,
  crate mutations) or remove the dead documentation/handler.

**9. MEDIUM — Polling fallback only refreshes the job list; heartbeat-only state freezes on SSE
disconnect**
- `useJobPoller.ts:21-29` calls `jobsApi.list()` → `setJobs()` every 4s while disconnected. Nothing
  else. `uptimeSeconds`/`machineProfile` are set exclusively by the `heartbeat` SSE handler
  (`useSSE.ts:72-78`) — during an SSE outage these freeze at their last value with no visible
  indication. `completionLog` (MissionControl's sparkline) is only bumped inside `upsertJob`'s
  `justCompleted` branch (`appStore.ts:90-102`) — `setJobs` (what the poller uses) bypasses it
  entirely, so throughput gets understated during an outage. Activity-log toasts are SSE-only; the
  poller never calls `pushActivity`, so the activity feed goes silent for the outage duration even
  though job state underneath is still updating.
- **Fix direction:** give the poller its own lightweight heartbeat-equivalent refresh, and route
  poller-observed job completions through the same `justCompleted`/activity-log path SSE uses.

### Part 4 — Route shadowing test coverage

**10. LOW — Route-shadowing test has a real, verified blind spot for dynamic-vs-dynamic collisions**
- `test_route_shadowing.py:72-86` filters to `"{" not in r.path` before checking anything, so it
  structurally can't catch a dynamic route shadowing another dynamic route. Verified concretely
  (direct `path_regex` matching against the live route table, not theoretical):
  `GET /library/similar/audio` and `GET /library/similar/export-cues` both dispatch to
  `GET /library/similar/{name}` instead of `GET /library/{name}/audio` /
  `GET /library/{name}/export-cues`, because `analysis.router` (which owns `/library/similar/{name}`)
  registers before `library.router`.
- **Concrete break:** a library song literally named `similar` can never have its audio streamed or
  DJ cues exported through these URLs — narrow (needs that exact name) but real.
- **Fix direction:** extend the test to also check dynamic-prefix collisions, or rename the
  `/library/similar/{name}` route to avoid the ambiguity (e.g. `/library/{name}/similar`).

### Part 5 — `api.ts` spot-checks vs. actual handlers

Rolled into the findings above: `analysisApi.compatibility` has a dead `{job_id: string}` union
branch (`response_model=CompatibilityResult` makes it unreachable; `MixDeck.tsx:452`'s type guard is
always true) plus the `transition_plan` phantom field (#2); `remixApi.preview`/`create` map to #4/#5;
`tagsApi.list/songTags` is #7; `jobsApi.list/get` feeds #1, the most consequential finding in this
phase.

---

## Phase 4 — Full Regression Baseline

**Run on real hardware.** Results:

| Check | Result |
|---|---|
| Safe-subset pytest (no librosa/e2e) | ✅ 602 passed |
| `dj_analysis` (librosa/numba) pytest | ✅ 66 passed — **closes the gap flagged above**: the Phase 1 true-peak limiter fix and the library.py locking fix are now validated against the real librosa-backed pipeline, not just synthetic-audio sandbox tests |
| `test_audio_source.py` + `e2e_test_suite.py` | ✅ 9 passed |
| Frontend typecheck (`tsc`) | ✅ pass |
| Frontend tests (vitest) | ✅ 32 passed |
| Ruff | ❌ pre-existing debt, unrelated to this QA pass |
| Black | ❌ 92+ files pre-existing formatting debt |
| Mypy | ❌ 166 pre-existing type errors |
| Frontend ESLint | ❌ no `eslint.config.js` (ESLint v9 flat-config migration never done) |

**677/677 backend tests pass, 32/32 frontend tests pass — zero test failures anywhere.** That's the
number that matters for "did this QA pass's fixes actually work." The ruff/black/mypy/ESLint
failures are pre-existing repo-wide debt with no connection to anything fixed in this pass — verified
by spot-checking the flagged lines (e.g. mypy's `Path` import errors at `mastering.py:641/718/750`
are the exact same pre-existing gap this session's own ruff pass already found and correctly left
alone as out-of-scope; `classic_engine/dj_engine.py`'s `Path` errors are in the frozen vendored
engine, which must never be touched). One exception: a genuine syntax error (an unterminated string
literal, a raw newline inside a string instead of `\n`) in
`ai_remixmate_feature_lab/python/ai_remixmate_features/io/json_export.py` was blocking ruff/black
from parsing that whole isolated directory at all — fixed inline as a one-line, verified-correct,
non-functional change, since it was a parse blocker for the tooling rather than a code-quality
finding.

**Not done, and out of scope without being asked:** mass-reformatting 92 files (black), fixing 166
mypy errors across `spotify.py`/`wordplay.py`/`license.py`/`setlist_planner.py`/etc., or cleaning up
`archive/legacy-v1`'s 200+ whitespace warnings. None of this is a QA finding — it's general repo
health debt that predates this pass, much of it in code explicitly marked legacy/archived/frozen.
Setting up ESLint v9's flat config (`eslint.config.js`) is a separate, bounded, real task — say the
word if you want it done.

---

## Recommended order if tackling the open items

1. **Security #1** (download `name` traversal) — HIGH, confirmed by reproduction, trivial fix
   (`sanitize_song_name()` in `_resolve_url` + schema validation).
2. **Frontend #1** (`JobResponse` missing `meta`) — HIGH, permanent data corruption once triggered,
   trivial fix (one field + one assignment).
3. **Security #2** (`bridge_beat_path` unconfined) — MEDIUM, same confinement pattern as #1.
4. **Security #3** (`/compatibility`) — one-line `_require_song()` addition.
5. Frontend MEDIUMs #2-4, #8-9 — mostly schema/type additions and one missing SSE emitter call.
6. Remaining LOWs and NITs as time allows — none of them are urgent on their own, but several
   (Security #4/#5, Frontend #7/#10) are cheap enough to knock out in the same pass as their
   neighbors above.
7. **Phase 4** — run the full regression baseline (including `dj_analysis`) on real hardware before
   calling any of this session's DSP-adjacent fixes (true-peak limiter especially) fully verified.
