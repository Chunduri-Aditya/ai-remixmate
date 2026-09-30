# PHASES

_Milestones for the `hosted-app` branch, one at a time. Committed. No dates: a box is ticked only when its evidence exists, and every tick below points at the `app/STATUS.md` section or commit that records it. `app/STATUS.md` stays the ledger for the hosted migration; this file indexes it and adds the fix passes that follow the repo map. Numbers live in their sources, next to the command that produced them._

## Hosted migration (from `app/STATUS.md`, "Phase ledger")

### Phase 0: Guardrails (test isolation)
- [x] Every pytest run isolates storage unless a test opts out. Evidence: `304da7e`, `52cd9ad`, `a4e8b50`; `tests/test_storage_isolation.py`.

### Phase 1: Catalog teardown and enforced CC gate
- [x] Code: teardown, licence gate, ingest guards, mix credits. Evidence: `3d06bf2` `2f2833f` `f6c0c6c` `8c4f204` `46aa166` `dbe2db1` `cc9de1a`, hardened in `42dfd79` and `664a18a`.
- [ ] Live catalog fill to the floor. Blocked: ccMixter refuses content downloads (`app/STATUS.md`, "Honest gaps", catalog item). See "Now".

### Phase 2: Guard generative routes and slim the image
- [x] Evidence: `f27aecb`; `tests/test_generative_gate.py`, `tests/test_root_route.py`, `tests/test_generative_urls.py`.

### Phase 3: User contextvar and per user data
- [x] Evidence: `653633a` (3a), `4a40867` (3b); `tests/test_user_scope.py`, `tests/test_crate_keys.py`.

### Phase 4: Auth and tenancy (Supabase)
- [x] Code: backend `d8b510b`, frontend `9068435`, local JWKS tool `32bceb4`; `tests/test_auth.py`, `tests/test_tenancy_jobs.py`, `tests/test_local_jwks.py`.
- [ ] Live Supabase project and a two account isolation run through the real login UI (`app/STATUS.md`, "Honest gaps", Supabase item).

### Phase 5: SQLite to Postgres (jobs, crates, feedback)
- [x] Code: `e4e894d` (jobs), `cd0ccc8` (crates, feedback); `scripts/core/db.py`; the container run is recorded in the Phase 5 verify block of `app/DEPLOYMENT_PLAN.md`.
- [ ] Managed Postgres run (`app/STATUS.md`, "Honest gaps", managed Postgres item).

### Phase 6: Per user mix retention
- [x] Evidence: `b8317f2`, `a0d3a30`; `tests/test_mix_retention.py`, `tests/test_lifespan.py`.

### Phase 7: Deploy and infra
- [x] Code: `8dbfe81`, `fb1a5f9`, `9c3fcae`, `a4e8b50`, `13a6d0f`, systemd proxy header flag in `074090b`; `tests/test_deploy_scripts.py`, `tests/test_live_verify.py`.
- [ ] Live box, volume, DNS and Cloudflare Pages; `/health/ready` from the public internet, a restart drain, a `pg_dump` restore, CI green on a PR. Runbook: `app/deploy/PROVISION.md` (`app/STATUS.md`, "Honest gaps").

### Phase 8: Legal and product surface
- [x] Code: `2919ec2`, `a0d3a30`, `0f4b1f3`; `tests/test_legal_phase8.py`, `tests/test_account_lifecycle.py`, `tests/test_admin_takedown.py`.
- [ ] SMTP delivery of DMCA email; until `REMIXMATE_SMTP_*` is set the send is a logged stub (`app/STATUS.md`, "Honest gaps", SMTP item).

### Phase 9: Hardening run
- [x] Commits `52cd9ad` to `2cc961d`; the table in `app/STATUS.md` "Phase 9: hardening" names the guarding test per commit and marks the CI, image, `pre-commit`, dependency bump, proxy flag, docs and ruff commits as guarded only by a gate or by nothing. Last recorded gate results: `app/STATUS.md` "Green bar right now" (measured at `bd1c446`, not rerun for this file).

## Now
Open decisions recorded in `app/STATUS.md` ("Honest gaps") and `SHOWCASE.md` ("Branches", "Known gaps"). None is a code change the agent can make alone.

- [ ] Catalog source: ccMixter refuses every content download, so growing the catalog needs another fetch path or source.
- [ ] Provisioning: Supabase, the VPS, DNS, Cloudflare Pages, SMTP and managed Postgres are operator work with a monthly cost.
- [ ] Push `hosted-app`, run CI on a PR, merge. `main` still carries the old download path with no auth and is not merged with `hosted-app`.
- [ ] Licence file: `LICENSE` is MIT in a private repo meant for sale; a proprietary notice is undecided.
- [ ] Frontend advisory: the `npm audit` findings in `react-router` are fixed only by the `react-router-dom` 7 major.
- [ ] Torch upgrade: `requirements.txt` pins `torch` and `torchaudio` together, coupled to Demucs.
- [ ] Library full mixes: a prune call during the hosted checks removed most `full_enhanced.wav` files with no backup; renders, recommend and the index read stems. Route guard: OPS-02, UI-07.
- [ ] Tier 3 engine items, not started: stem blend effects (MX-13), preview parity (MX-17), chain entry never before the previous exit (MX-24), fade on a truncated last chain song (MX-39), stem similarity on the transition windows (MX-58); "bridge intensity applied once" has no backlog id.

## Next: UI fix pass
High and medium UI ids from `docs/agents/BACKLOG.md`, in its rank order. Rules for every item: the fix starts with a test that fails on the current code and passes after it; an item marked unproven needs a repro first; the fixing commit deletes the item from the backlog and its id from the maps. Frontend commands: `DEVELOPING.md`, "Commands".

- [ ] UI-01. Check: extend `frontend/src/components/__tests__/CamelotWheel.test.ts` so the wheel lights the backend Camelot code for a minor key and a major key (A minor at 8A, C major at 8B).
- [ ] UI-02. Check: `frontend/src/__tests__/crates.test.tsx::installFetch` answers with the real `{crate_id, songs, count}` body and a member crate renders checked.
- [ ] UI-07. Check: an Operations test asserts no prune request is sent before an in page confirmation. Pairs with OPS-02 on the route.
- [ ] UI-03 (unproven). Check: a hosted mode repro of the waveform 401 first, then `frontend/src/components/__tests__/WaveformDeck.test.tsx` asserts `WaveSurfer.create` gets fetch params that include credentials, with the same case for `frontend/src/pages/MixVault.tsx::WaveformPlayer`.
- [ ] UI-04 (unproven). Check: a hosted mode repro first, then a new test asserts `frontend/src/lib/playerEngine.ts::PlayerEngine` requests media with credentials.
- [ ] UI-05. Check: feed a heartbeat frame without `uptime_seconds` and assert Mission Control never renders NaN.
- [ ] UI-06. Check: in `frontend/src/shell/__tests__/AppShell.test.tsx`, close the inspector with Escape and assert a control reopens it.
- [ ] UI-08. Check: a Set Builder test with one track more than the `scripts/api/schemas.py::DJChainRequest` cap asserts no chain request is sent.
- [ ] UI-09. Check: in `frontend/src/__tests__/contracts.test.tsx`, the preview body carries `transition_effect` and no `preset`.
- [ ] UI-10. Check: a pytest asserts `GET /jobs` rows carry `meta` without `user_id`, and the retry case in `frontend/src/__tests__/jobsux.test.tsx` runs on a REST shaped job.
- [ ] UI-11. Check: hydrate the store with already finished jobs and assert no toast and no completion log entry.
- [ ] UI-12. Check: TransitionTimeline labels the bars the page chose, or the compatibility response carries a transition plan and a test reads it.
- [ ] UI-13. Check: in `frontend/src/__tests__/features.test.tsx`, with the operator flag off, Library Atlas hides the operator actions, and a 403 surfaces instead of being swallowed.
- [ ] UI-14. Check: mount Mission Control over a store that holds an SSE job with `meta` and assert the job and its `meta` survive.
- [ ] UI-15. Check: render the result card with `quality_passed` false and assert a warning state.
- [ ] UI-16. Check: change the RemixControls transition bars and assert the remix body sends that value, or assert the duplicate control is gone.
- [ ] UI-17. Check: in track mode, Signal Search never requests more than the cap on `scripts/api/routers/analysis.py::similar_songs`.
- [ ] UI-18. Check: one case per `apiHealth` and `sseConnected` pair asserts the ConnectionDot label and colour class.
- [ ] UI-19 (unproven). Check: with `GET /features` reporting auth on and frontend auth unconfigured, the shell shows a configuration error.

## Next: mixing fix pass
High and medium MX ids from `docs/agents/BACKLOG.md`, in its rank order. Same rules as the UI pass. The classic engine under `scripts/core/classic_engine/` is kept frozen (the MX-11 and MX-21 fix sketches), so those two wait until classic work is in scope. Theory for the checks: `docs/DJ_THEORY.md` (drift listed under DOC-02).

- [ ] MX-01. Check: analyse a fixture longer than the analysis window and assert the cached duration equals the file duration.
- [ ] MX-02. Check: a render with a tempo ratio inside the no stretch band and a late entry asserts the entry sample equals entry time times sample rate.
- [ ] MX-03. Check: a click track pair with different downbeat offsets asserts song B's bar start lands on song A's grid.
- [ ] MX-04. Check: a relative key case (8A against 8B) in `tests/test_compatibility_score.py` scores a nonzero key term and passes as compatible; update the pinned test that encodes the old score.
- [ ] MX-05. Check: an analysed adjacent key candidate outranks an unanalysed candidate at equal BPM in `scripts/core/recommend.py::_composite_score`.
- [ ] MX-06. Check: in `tests/test_feedback_router.py`, a real remix job shape backfills effect and tempo ratio into the logged context.
- [ ] MX-07. Check: posting an unknown `transition_effect` or bridge mode returns 422.
- [ ] MX-08. Check: mirror `tests/test_render_e2e.py::test_chain_stops_between_songs_once_cancelled` for the two song remix and the preview; a cancelled job writes no output folder.
- [ ] MX-09. Decide first whether chains use stems (the backlog item says so). Check: a chain over songs with stems takes the stem path, or the dead branch and its docstring are removed. Traps: `scripts/core/classic_engine/dj_engine.py::DJEngine.render_chain` takes no `stems_dirs` and `tests/test_render_e2e.py::test_chain_stops_between_songs_once_cancelled` runs chains on classic, so pass the argument only on the current engine; the branch gate in `scripts/core/dj_engine.py::DJEngine.render_chain` is `is_dir()` on the song dir, so pass None for a song without stems; the branch gets no transition effect, bridge beat or stem blend confidence gate.
- [ ] MX-10. Check: in `tests/test_cue_export.py`, exported cues sit on phrase boundaries, including past the analysis window.
- [ ] MX-11 (classic, frozen). Check: a clash shortened classic plan puts the bass swap at the midpoint of the shortened transition.
- [ ] MX-12. Check: an unknown `key_profile` returns 422 on `POST /analyze` and on the structure route.
- [ ] MX-13. Check: an echo request changes the stem path transition window.
- [ ] MX-14. Check: two close genre scores give a confidence below 1.0.
- [ ] MX-32. Check: a `GET /library/similar/{name}` and a `GET /recommend/{name}` over a scratch library leave every song dir byte identical.
- [ ] MX-35. Check: after a cancel, a progress update from the task neither changes the stored message nor emits another `job_cancelled` frame.
- [ ] MX-39. Check: a chain whose last song is longer than `remix.max_song_seconds` ends in the same fade the two song remix applies.
- [ ] MX-67. Check: a changed `dj.default_transition_bars` reaches the planned transition of a `POST /dj-remix` that sends no `transition_bars`.
- [ ] MX-15 (unproven). Check: a second structure GET for the same song does not rerun analysis, or the route returns a job.
- [ ] MX-16 (unproven). Check: a setlist request over a new track cap returns 422.
- [ ] MX-17 (unproven). Check: a parity test on plan fields between the preview and the full render for the same pair.
- [ ] MX-18 (unproven). Check: a song whose only source is `full_enhanced.wav` analyses without `FileNotFoundError`.
- [ ] MX-19 (unproven). Check: when meta.json cannot be written, `POST /analyze` ends failed, not done.
- [ ] MX-20 (unproven). Check: after an octave correction the bar length agrees with the corrected BPM.
- [ ] MX-21 (unproven, classic, frozen). Check: a signal limited to the ceiling is not flagged as clipping.
- [ ] MX-22 (unproven). Check: a low confidence phrase key falls back to the whole track key.
- [ ] MX-23 (unproven). Check: onset alignment across the overlap for a tempo gap inside the no stretch band.
- [ ] MX-24 (unproven). Check: a chain with a late entry and an early next exit replays no audio.
- [ ] MX-25 (unproven). Check: short term loudness across both seams of the stem window stays within a stated bound.
- [ ] MX-26 (unproven). Check: an unknown Instrument Lab mode returns 422 and mode `all` is capped.
- [ ] MX-27 (unproven). Check: an unpatched clash (C major against F# major) shortens the transition.
- [ ] MX-28 (unproven). Check: a key detection failure stores empty key fields, not a real key.
- [ ] MX-29 (unproven). Check: in `tests/test_spectral_sync.py`, energy below the bass crossover is bit identical in the window.
- [ ] MX-30 (unproven). Check: a source rebuilt from stems keeps the original mix balance, or the per stem gain is stored.
- [ ] MX-31 (unproven). Check: style transfer with `full.wav` pruned loads its source through `scripts/core/audio_source.py`.
- [ ] MX-62 (unproven). Check: 8A to 9B does not score as adjacent in `scripts/core/music_intelligence.py::_camelot_move_analysis` or `scripts/core/recommend.py::_camelot_score`.

## Later
- Every OPS, DOC and low severity item, and anything above not yet picked up: `docs/agents/BACKLOG.md`.
