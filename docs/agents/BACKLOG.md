# BACKLOG

_Ranked problems found while mapping the repo at `72e2368`, for the fix passes that follow. Generated from the mapping run, then edited by hand; edit it by hand from now on._

- Ids are stable and never reused. The commit that fixes an item deletes it here and removes its id from the maps in the same commit.
- Line numbers and quoted lines are as of `72e2368`. Reopen the file before trusting a line.
- Evidence: "confirmed by reading" means the code alone proves the impact; "unproven" means a repro is owed before the fix, and the fix starts with a failing test either way.
- Rank: severity (high: wrong output, data loss, security or a broken flow; medium: degraded or misleading; low: dead code, drift, cosmetic), then confirmed before unproven.

## Index

| id | severity | evidence | title |
|---|---|---|---|
| UI-01 | high | confirmed | Camelot wheel shows the wrong keys: Open Key numbering and no mode |
| UI-02 | high | confirmed | Crate membership never loads; crate toggles break |
| UI-07 | high | confirmed | Storage prune runs on one click with no confirmation |
| MX-01 | high | confirmed | Cached duration is capped at 120 s for every song |
| MX-02 | high | confirmed | Entry index divides by the stretch ratio even when the 2 percent band skipped the stretch |
| MX-03 | high | confirmed | Beat grid lock moves song B off its own bar start |
| MX-04 | high | confirmed | POST /compatibility scores relative major/minor (and every ring change) as key_score 0 and vetoes compatible |
| OPS-01 | high | confirmed | DELETE /account reports clean when the job purge fails in the DB |
| OPS-02 | high | confirmed | POST /library/storage/prune has no dry run or preview |
| UI-03 | high | unproven | WaveSurfer fetches audio without credentials |
| UI-04 | high | unproven | Global player audio element is crossOrigin anonymous |
| OPS-03 | high | unproven | Local mode catalog mutations reachable by a cross site simple POST |
| UI-05 | medium | confirmed | Mission Control shows "uptime NaNm": the SSE heartbeat carries no uptime |
| UI-06 | medium | confirmed | Right inspector cannot be reopened once closed |
| UI-08 | medium | confirmed | Set Builder allows chains over the 8 song API limit |
| UI-09 | medium | confirmed | Mix Deck preview drops the chosen effect and preset |
| UI-10 | medium | confirmed | REST JobResponse carries no meta, so Inspector retry, thumbs and Mix Vault titles fail for REST loaded jobs |
| UI-11 | medium | confirmed | Historic jobs toast and count as fresh completions when the store hydrates |
| UI-12 | medium | confirmed | Transition timeline and cue regions depend on a field the API never sends |
| UI-13 | medium | confirmed | Library Atlas shows operator only actions to everyone and swallows the 403 |
| UI-14 | medium | confirmed | Mission Control replaces the whole job store on mount |
| UI-15 | medium | confirmed | Mix Ready card ignores quality_passed |
| UI-17 | medium | confirmed | Signal Search offers k=50 but the similar route caps k at 20 |
| UI-18 | medium | confirmed | ConnectionDot labels are inverted relative to how apiHealth is set |
| MX-05 | medium | confirmed | /recommend ranks unanalysed songs above analysed ones and treats real 0.5 scores as missing |
| MX-06 | medium | confirmed | Feedback context backfill reads keys the job never stores |
| MX-07 | medium | confirmed | Unvalidated transition_effect and bridge options silently degrade renders |
| MX-08 | medium | confirmed | Running remix and preview ignore cancel and still write output |
| MX-09 | medium | confirmed | Chain task never passes stems_dirs, so chains never use the stem blend |
| MX-10 | medium | confirmed | Cue export uses energy section starts, not SSM phrase boundaries, and only within the first 120 s |
| MX-11 | medium | confirmed | Classic plan puts the bass swap at the end of a clash shortened transition |
| MX-12 | medium | confirmed | Unknown key_profile is accepted and silently replaced by a different key detector |
| MX-13 | medium | confirmed | Stem path ignores echo, filter, reverb and the auto exit effect |
| MX-14 | medium | confirmed | Genre confidence is always 1.0 |
| MX-32 | medium | confirmed | GET /library/similar and GET /recommend write meta.json into the shared library |
| MX-35 | medium | confirmed | Cancelled job keeps emitting job_cancelled frames and changing message |
| MX-39 | medium | confirmed | Chain hard cuts a last song truncated at the load cap |
| MX-67 | medium | confirmed | `dj.default_transition_bars` is dead for API renders; the real default is the request model literal 16 |
| OPS-04 | medium | confirmed | evict_lru dry run always returns an empty list |
| OPS-05 | medium | confirmed | Operator supplied ingest name is not sanitized |
| UI-19 | medium | unproven | Frontend and backend auth modes can disagree silently |
| MX-15 | medium | unproven | Structure endpoint runs full track analysis synchronously per GET |
| MX-16 | medium | unproven | Setlist optimize has no track count cap |
| MX-17 | medium | unproven | Preview analyses uncapped audio while the full render analyses the first max_song_seconds |
| MX-18 | medium | unproven | Analysis loader ignores full_enhanced.wav |
| MX-19 | medium | unproven | meta.json write failure is swallowed while the job reports success |
| MX-20 | medium | unproven | Octave corrected BPM disagrees with the beat grid |
| MX-21 | medium | unproven | Classic master_mix flags limited samples as clipping |
| MX-22 | medium | unproven | Section key confidence gate at 0.35 can never reject a successful detection |
| MX-23 | medium | unproven | No stretch band leaves B at its own tempo across the window |
| MX-24 | medium | unproven | Middle song body can be skipped and audio replayed in chains |
| MX-25 | medium | unproven | LUFS normalised stem window butts against raw full mix audio |
| MX-26 | medium | unproven | Instrument Lab is ungated and mode 'all' renders songs^4 full length combos |
| MX-27 | medium | unproven | psychoacoustic_consonance never goes below about 0.89, so every clash branch keyed to 0.35 is dead in the current engine |
| MX-28 | medium | unproven | Key detection failure is cached as a real C major 8B |
| MX-29 | medium | unproven | Spectral sync bass guard protects no band at the configured 150 Hz |
| MX-30 | medium | unproven | Rebuilt source from stems has a different balance than the original mix |
| MX-31 | medium | unproven | Generative sources default to full.wav, which prune deletes |
| MX-62 | medium | unproven | Camelot move scorers ignore the ring letter for adjacency |
| OPS-06 | medium | unproven | backup.sh leaves dumps world readable and puts the DB password on the command line |
| OPS-07 | medium | unproven | `docker-compose` and the systemd unit mount the catalog read only while routes write into it |
| OPS-08 | medium | unproven | Importing scripts.api.jobs from a second process fails the live API's jobs |
| OPS-09 | medium | unproven | python -m scripts.api.main binds 0.0.0.0 in local mode without --no-proxy-headers |
| OPS-10 | medium | unproven | conftest does not unset DATABASE_URL |
| DOC-01 | medium | unproven | POST /analyze is not operator gated despite SECURITY.md |
| UI-20 | low | confirmed | A dj_chain job type is queried and branched on but never exists |
| UI-21 | low | confirmed | Dead frontend exports, store fields and Vite proxies |
| UI-22 | low | confirmed | Error toasts show the raw ApiError string including the response body |
| UI-24 | low | confirmed | Shortcut help and number slots ignore the generative flag |
| UI-25 | low | confirmed | Player volume and EQ set before first play are not applied to the graph |
| UI-26 | low | confirmed | AI Lab model selector is ignored by the API |
| UI-27 | low | confirmed | Open in Vault is a raw anchor |
| UI-28 | low | confirmed | Operations hides read only storage stats from non operators |
| UI-29 | low | confirmed | Empty library CTAs point at pages that cannot help |
| UI-30 | low | confirmed | Widget pop out resets state |
| UI-31 | low | confirmed | ToastProvider mounted twice; both toast stacks overlap at the same corner |
| UI-32 | low | confirmed | Inspector Retry swallows request failures |
| UI-33 | low | confirmed | Request types carry fields the backend ignores |
| UI-34 | low | confirmed | library_changed and system_status events are never emitted |
| UI-35 | low | confirmed | CREDITS.txt served as audio/wav |
| MX-33 | low | confirmed | preset request field is accepted and ignored |
| MX-34 | low | confirmed | Dead mixing code: TIV branch, unused helpers, constants and branches |
| MX-36 | low | confirmed | Wordplay bonus is applied as an energy bump, not a pair preference |
| MX-37 | low | confirmed | `analyze-missing` upserts each song into the index twice |
| MX-38 | low | confirmed | Remix fade applied after mastering report |
| MX-40 | low | confirmed | CLAP index reads an energy field analysis.json never contains |
| MX-41 | low | confirmed | Phrase snap preference for 16 and 32 bars is subsumed by 8 |
| MX-42 | low | confirmed | B high pass ramp only processes the half where B is multiplied by zero |
| MX-43 | low | confirmed | Instrument Lab labels assign letters from set order per combo |
| MX-44 | low | confirmed | Current master_mix clipping check is dead for chain renders |
| MX-46 | low | confirmed | Legacy key detectors mislabel keys |
| MX-47 | low | confirmed | PythonBridge.dj_remix mixes the render with itself |
| MX-48 | low | confirmed | arc_target_energy on each row belongs to the next track |
| MX-49 | low | confirmed | Stem split reports success with fewer than four stems |
| OPS-11 | low | confirmed | remix.max_pitch_shift_semitones is read by nothing |
| OPS-12 | low | confirmed | library.auto_evict_on_download is read by no code path |
| OPS-13 | low | confirmed | Demucs model name is unvalidated free text |
| OPS-14 | low | confirmed | separation config block is read by no stem code |
| OPS-15 | low | confirmed | _venv_python docstring order differs from code |
| DOC-02 | low | confirmed | docs/DJ_THEORY.md contradicts the code on bass swap timing, stretch limits, clash length and A/B pairs |
| DOC-03 | low | confirmed | DEVELOPING.md drifts from the code and CI in four places |
| DOC-04 | low | confirmed | Comments and docstrings that contradict the code |
| DOC-05 | low | confirmed | _resolve_engine docstring calls classic the default |
| DOC-06 | low | confirmed | Classic engine documented as byte faithful after three edits |
| UI-36 | low | unproven | No `color-scheme` declaration on a dark only UI |
| UI-37 | low | unproven | Library list silently truncates at 5000 and is expensive |
| UI-38 | low | unproven | Account fetch bypasses the 401 handler and timeout |
| UI-39 | low | unproven | Processing status polls every second in background tabs |
| UI-40 | low | unproven | Rail Widget item strands the user in a shellless page |
| MX-50 | low | unproven | /ai/tokenize writes into the shared library without an operator check |
| MX-51 | low | unproven | stem_blend result field reports the gate decision, not the path taken |
| MX-52 | low | unproven | audio_enhance true peak limiter is whole file peak scaling |
| MX-53 | low | unproven | Tokenizer reads WAV stems only |
| MX-54 | low | unproven | Beat grid ignores model predicted downbeats |
| MX-55 | low | unproven | Short sections are dropped, leaving gaps in bar coverage |
| MX-56 | low | unproven | render() dynamic fade leaves B's high band at about 0.85 then steps to unity |
| MX-57 | low | unproven | render_chain uses a different gain curve from render() |
| MX-58 | low | unproven | Stem similarity compares processed A stems with raw unstretched B stems |
| MX-59 | low | unproven | GPU time stretch always raises on MPS and silently falls back |
| MX-60 | low | unproven | Instrument Lab tempo fallback reads full.wav then assumes 120 |
| MX-61 | low | unproven | Similarity vector key term is a tonic one hot, not Camelot aware; drop dimension is constant for analysed songs |
| MX-63 | low | unproven | Spectral sync target ramp spans the whole window while render() B is silent in the first half |
| MX-64 | low | unproven | Second split writes WAV stems while stale FLAC stems win |
| MX-65 | low | unproven | key_to_camelot has no C# minor entry |
| MX-66 | low | unproven | Guard test plan fixture uses the inverted tempo ratio |
| OPS-16 | low | unproven | pages.yml is still dispatchable but its origin is blocked by CORS |
| OPS-17 | low | unproven | eslint config imports an undeclared package |
| OPS-18 | low | unproven | Retention sweep treats every top level outputs dir as a user |
| DOC-07 | low | unproven | app/README.md is stale about the source of truth and status |
| DOC-08 | low | unproven | config.yaml cors_origins '*' comment contradicts cors.py |

## UI

### UI-01 Camelot wheel shows the wrong keys: Open Key numbering and no mode

- Severity: high. Evidence: confirmed by reading.
- Where: `frontend/src/components/CamelotWheel.tsx::CAMELOT` (line 9): `'Am':  { pos: 1,  type: 'A' }, 'C':   { pos: 1,  type: 'B' },`
- Where: `frontend/src/pages/MixDeck.tsx::MixDeck` (line 606): `<CamelotWheel keyA={songInfoA?.key} keyB={songInfoB?.key} size={220} />`
- Symptom: The CAMELOT map puts C major at 1B and A minor at 1A, while the backend (scripts/core/key_detection.py::CAMELOT) puts C major at 8B and A minor at 8A, so every lit segment is off by 7 from the Camelot chip shown beside it.
- Also: CAMELOT uses Open Key numbering (Am=1A, C=1B) where backend Camelot is Am=8A; flat spellings (Db, Eb, Ab, Bb) never match backend sharp roots (C#, D#, G#, A#); callers pass SongInfo.key, a root only, and the frontend SongInfo type has no mode, so minor keys render as major
- Also: Callers pass SongInfo.key, which holds a root note only (for example 'G' with mode 'minor' in a separate field), so minor songs light the major ring. Sharp roots (C#, D#, G#, A#) match no map entry because the map uses flats.
- Impact: Mix Deck and the DJ Widget highlight the wrong Camelot codes, contradicting the backend camelot values shown on the same page.
- Fix sketch: Key the wheel on Camelot codes (for example '8B') and have callers pass SongInfo.camelot; derive positions from the code instead of a note name map.

### UI-02 Crate membership never loads; crate toggles break

- Severity: high. Evidence: confirmed by reading.
- Where: `frontend/src/lib/api.ts::cratesApi` (line 326):

  ```
  songs:   (id: string)                 => get<string[]>(`/crates/${id}/songs`),
  ```

- Where: `frontend/src/__tests__/crates.test.tsx::installFetch` (line 80): `if (/^\/crates\/[^/]+\/songs$/.test(path)) return jsonResponse(method === 'GET' ? [] : {})`
- Symptom: cratesApi.songs is typed string[] but GET /crates/{id}/songs returns {crate_id, songs, count}; CratesSection calls names.includes on the object and throws, so every crate checkbox renders unchecked, the query error is toasted, and ticking a crate the song is already in POSTs again and gets an uncaught 409.
- Also: cratesApi.songs is typed string[] but GET /crates/{id}/songs returns {crate_id, songs, count}; CratesSection calls names.includes, which throws, so every crate checkbox stays unchecked, checking a crate the song is already in 409s, and removing a song from a crate is impossible from the UI
- Impact: The Library Atlas drawer cannot show or remove crate membership. scripts/api/routers/crates.py::get_crate_songs returns an object, and LibraryAtlas.tsx::CratesSection line 220 calls names.includes on it. frontend/src/__tests__/crates.test.tsx mocks the route with a bare array, which hides the bug.
- Fix sketch: Change cratesApi.songs to get<{songs: string[]}>(...).then(r => r.songs ?? []); fix the crates.test.tsx mock to return the real shape; wrap toggleCrate in try/catch.

### UI-07 Storage prune runs on one click with no confirmation

- Severity: high. Evidence: confirmed by reading.
- Where: `frontend/src/pages/Operations.tsx::StoragePanel` (line 116): `const r = await storageApi.prune()`
- Where: `frontend/src/pages/Operations.tsx::StoragePanel` (line 213): `<button className="ops-job__btn" disabled={busy !== null} onClick={runPrune}>`
- Symptom: runPrune calls POST /library/storage/prune directly; the route has no dry run and deletes full.wav for every song with stems, while evict is behind window.confirm
- Also: The Prune button calls POST /library/storage/prune directly; evict and reanalyze both ask for confirmation first, but prune does not.
- Impact: An operator misclick removes every source WAV that has stems; the incident notes record a prior mass deletion via this route
- Fix sketch: Add an in page confirmation (the viewer blocks confirm in some contexts) and a dry_run preview on the route

### UI-03 WaveSurfer fetches audio without credentials

- Severity: high. Evidence: unproven.
- Where: `frontend/src/components/WaveformDeck.tsx::WaveformDeck` (line 48): `const ws = WaveSurfer.create({`
- Symptom: WaveSurfer.create in WaveformDeck and MixVault WaveformPlayer passes no fetchParams, so the fetch uses same origin credentials and a cross origin API gets no cookie
- Also: WaveSurfer.create gets no fetchParams, so load() uses default fetch credentials (same origin) and no Bearer header; MixVault's WaveformPlayer (line 77) does the same.
- Impact: Hosted Mix Deck and Mix Vault waveforms are expected to 401 when auth is on
- Fix sketch: Pass fetchParams: { credentials: 'include' } in both WaveSurfer.create calls (frontend/src/pages/MixVault.tsx::WaveformPlayer too)

### UI-04 Global player audio element is crossOrigin anonymous

- Severity: high. Evidence: unproven.
- Where: `frontend/src/lib/playerEngine.ts::PlayerEngine` (line 29): `this.audio.crossOrigin = 'anonymous'`
- Symptom: playerEngine sets crossOrigin='anonymous', so a cross origin API (hosted api.<domain>) receives no `sb-access-token` cookie on /library/{name}/audio and /outputs requests
- Also: PlayerEngine sets crossOrigin = 'anonymous', which omits cookies on cross origin media requests.
- Impact: Hosted playback through the PlayerBar is expected to 401 when auth is on; local same origin mode unaffected
- Fix sketch: Use crossOrigin='use credentials' (the API already sets allow_credentials with an explicit origin list) so the analyser graph keeps working

### UI-05 Mission Control shows "uptime NaNm": the SSE heartbeat carries no uptime

- Severity: medium. Evidence: confirmed by reading.
- Where: `frontend/src/pages/MissionControl.tsx::HeartbeatWidget` (line 204): `if (s === 0) return 'unknown'`
- Where: `frontend/src/hooks/useSSE.ts::useSSE` (line 77): `setUptimeSeconds(d.uptime_seconds)`
- Where: `scripts/api/routers/events.py::SSEBroadcaster.send_heartbeat` (line 143): `"active_jobs": active.get(uid, 0),`
- Symptom: The SSE heartbeat carries only active_jobs and clients; useSSE stores d.uptime_seconds (undefined) and formatUptime guards only 0, so it renders NaN.
- Also: SSEBroadcaster.send_heartbeat sends only active_jobs and clients; useSSE stores d.uptime_seconds (undefined) and MissionControl HeartbeatWidget formatUptime(undefined) renders NaN; machine_profile is never emitted so Inspector host rows never render
- Also: The SSE heartbeat carries only active_jobs and clients; _machine_profile is filled at startup and read nowhere.
- Impact: Whenever SSE is connected, the heartbeat widget reads 'uptime NaNm'. The heartbeat frame comes from scripts/api/routers/events.py::SSEBroadcaster.send_heartbeat.
- Fix sketch: Send uptime_seconds in the heartbeat frame, or guard with !Number.isFinite(s) in formatUptime and skip setUptimeSeconds when the field is absent.

### UI-06 Right inspector cannot be reopened once closed

- Severity: medium. Evidence: confirmed by reading.
- Where: `frontend/src/hooks/useKeyboardShortcuts.ts::useKeyboardShortcuts` (line 99): `if (inspectorOpen) toggleInspector()`
- Where: `frontend/src/shell/AppShell.tsx::AppShell` (line 131): `{inspectorOpen && <RightInspector />}`
- Symptom: Esc (any time focus is outside an input, including when closing the shortcuts modal) or the X button sets inspectorOpen false; no control anywhere toggles it back open, so Jobs, Activity and System are gone until a full reload.
- Also: AppShell unmounts RightInspector when inspectorOpen is false; the only toggle callers are the Inspector close button and Escape (which only closes)
- Impact: grep for toggleInspector finds only the close button (RightInspector.tsx) and this Esc branch; job cancel, retry and progress UI becomes unreachable after one Esc press.
- Fix sketch: Add an inspector toggle button to LeftRail (or a shortcut such as `i`), and make Esc close only the modal when the modal is open.

### UI-08 Set Builder allows chains over the 8 song API limit

- Severity: medium. Evidence: confirmed by reading.
- Where: `frontend/src/pages/SetBuilder.tsx::SetBuilder` (line 257): `if (set.length < 2 || submitting) return`
- Where: `frontend/src/pages/SetBuilder.tsx::SetBuilder` (line 261): `const res = await remixApi.chain(set.map((e) => e.name), {`
- Symptom: The chain button is enabled for any set of 2 or more, but DJChainRequest.songs has max_length=8.
- Also: addToSet has no cap and launchChain posts every entry; scripts/api/schemas.py::DJChainRequest.songs has max_length=8, so a set of 9 or more returns 422
- Impact: A set of 9 or more tracks always fails with a 422 shown as raw text.
- Fix sketch: Disable the button above 8 tracks with a hint, or split the set into chained jobs.

### UI-09 Mix Deck preview drops the chosen effect and preset

- Severity: medium. Evidence: confirmed by reading.
- Where: `frontend/src/pages/MixDeck.tsx::MixDeck` (line 466): `const res = await remixApi.preview({`
- Where: `frontend/src/pages/MixDeck.tsx::MixDeck` (line 471): `preset: remixOpts.preset,`
- Symptom: launchPreview sends preset and transition_duration (not in scripts/api/schemas.py::DJPreviewRequest, silently dropped) and omits transition_effect, so the backend default 'auto' is always used
- Also: launchPreview sends preset, which DJPreviewRequest does not declare (Pydantic ignores it), and never sends transition_effect, so the preview always renders with the default effect.
- Impact: A preview auditioned with 'none', 'echo' or 'stepped' selected differs from the full render the user then launches
- Fix sketch: Send transition_effect: remixOpts.transition_effect; drop preset and transition_duration from DJPreviewRequest type

### UI-10 REST JobResponse carries no meta, so Inspector retry, thumbs and Mix Vault titles fail for REST loaded jobs

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/api/schemas.py::JobResponse` (line 179): `class JobResponse(BaseModel):`
- Where: `frontend/src/shell/RightInspector.tsx::JobCard` (line 126): `const meta = (job.meta as Record<string, unknown>) ?? {}`
- Where: `frontend/src/pages/MixVault.tsx::MixVault` (line 342): `(!j.meta?.preview || showPreviews),`
- Symptom: JobResponse and job_to_response omit meta while SSE frames carry it; Mix Vault loads only via REST so titles fall back to the date, the previews toggle never hides previews and previewCount is always 0; Inspector retry and thumbs are inert for any job whose store copy came from REST
- Also: GET /jobs returns no meta; Mix Vault cannot tell previews from full mixes for jobs loaded over REST (the Show previews toggle and preview count do nothing), and the Inspector Retry for a REST loaded job finds no meta.request and returns without retrying.
- Also: Retry on a FAILED dj_remix card and thumbs on a COMPLETED card do nothing, with no message, for any job whose store entry was last written from GET /jobs (MissionControl setJobs on mount, useJobPoller merge). REST JobResponse has no meta field; only SSE frames (_job_to_response_dict) carry meta.
- Also: JobResponse has no meta field, so jobs loaded by the mount fetch have meta undefined. The preview filter passes, the previews toggle count is 0, and mixTitle falls back to the date. upsertJob replaces any SSE copy that did carry meta.
- Impact: Mix Vault shows 'Mix, date' titles and mixes previews in with full mixes; retry silently does nothing for REST hydrated jobs
- Fix sketch: Add meta: Optional[Dict[str, Any]] to JobResponse, filled in scripts/api/jobs.py::job_to_response with user_id stripped

### UI-11 Historic jobs toast and count as fresh completions when the store hydrates

- Severity: medium. Evidence: confirmed by reading.
- Where: `frontend/src/hooks/useJobToasts.ts::useJobToasts` (line 14): `if (prev === job.status) return`
- Where: `frontend/src/stores/appStore.ts::useAppStore` (line 93): `const justCompleted = job.status === 'COMPLETED' && prev?.status !== 'COMPLETED'`
- Where: `frontend/src/pages/MixVault.tsx::MixVault` (line 328): `.then(([remixes, chains]) => [...remixes, ...chains].forEach(upsertJob))`
- Symptom: useJobToasts starts with an empty prevRef and no session or created_at filter, so when MissionControl hydrates the store from /jobs every already COMPLETED or FAILED job produces a success or error toast.
- Also: useJobToasts starts with an empty prevRef, so every COMPLETED or FAILED job loaded from REST triggers a toast (5 visible at once)
- Also: upsertJob appends to completionLog whenever a job is first seen as COMPLETED; Mix Vault upserts up to 200 old completed jobs from REST and the poller and Mission Control load old ones too
- Also: The mount load calls upsertJob for up to two pages of done jobs; upsertJob records each as just completed (Date.now() into completionLog), and useJobToasts pushes a success toast for each status it has not seen.
- Impact: On load, users see a burst of 'complete'/'failed' toasts (newest five kept) for jobs that finished in earlier sessions, while the Inspector itself hides those same jobs via sessionStartedAt.
- Fix sketch: Seed prevRef with the statuses present on first run without toasting, or skip terminal jobs whose created_at predates useAppStore.sessionStartedAt.

### UI-12 Transition timeline and cue regions depend on a field the API never sends

- Severity: medium. Evidence: confirmed by reading.
- Where: `frontend/src/components/TransitionTimeline.tsx::TransitionTimeline` (line 21): `const transBars = plan?.transition_bars ?? 16`
- Where: `frontend/src/types/index.ts::CompatibilityResult` (line 212): `transition_plan?: TransitionPlan`
- Symptom: CompatibilityResult (scripts/api/schemas.py) has no transition_plan, so TransitionTimeline always labels the crossfade with the default 16 bars and WaveformDeck never draws cue regions.
- Also: CompatibilityResult.transition_plan is read for cue regions and TransitionTimeline widths; scripts/api/routers/analysis.py::check_compatibility returns the backend CompatibilityResult model, which has no such field
- Impact: The timeline shows '16 bars' whatever the user picked, and the waveform cue overlay is dead code.
- Fix sketch: Pass the page's chosen bars into TransitionTimeline, or add transition_plan to the compatibility response.

### UI-13 Library Atlas shows operator only actions to everyone and swallows the 403

- Severity: medium. Evidence: confirmed by reading.
- Where: `frontend/src/pages/LibraryAtlas.tsx::SongRow` (line 183): `onClick={() => onDelete(song.name)}`
- Symptom: Delete, delete batch, scan broken, analyze missing and split stems are rendered without checking features.operator, and handleDelete, handleAnalyzeMissing and handleScanBroken swallow errors with catch {}.
- Also: Delete, Delete selected, Analyze missing and Scan broken render without checking features.operator; the routes depend on require_operator (scripts/api/routers/library.py delete_song, delete_songs_batch, analyze_missing, scan_broken)
- Impact: When hosted with auth on, a non operator clicks these buttons and nothing happens. The server requires require_operator on DELETE /library/{name}, /library/delete-batch, /library/scan-broken, /library/analyze-missing and /stems/split. Only Operations reads features.operator.
- Fix sketch: Read useFeatures().data?.operator in LibraryAtlas and hide or disable these controls; surface failures instead of swallowing them.

### UI-14 Mission Control replaces the whole job store on mount

- Severity: medium. Evidence: confirmed by reading.
- Where: `frontend/src/pages/MissionControl.tsx::MissionControl` (line 264): `if (jobs) setJobs(jobs)`
- Symptom: setJobs(jobs) with the newest 20 REST jobs replaces the map, dropping older entries and every SSE supplied meta
- Also: setJobs(jobs) replaces the jobs map with the newest REST page (default limit 20), dropping jobs loaded by Mix Vault and any SSE meta.
- Impact: Visiting Mission Control removes older jobs from the Inspector and strips meta, disabling retry and Mix Vault titles for those jobs until SSE resends them
- Fix sketch: Merge by id as useJobPoller does, and keep an existing meta when the incoming job has none

### UI-15 Mix Ready card ignores quality_passed

- Severity: medium. Evidence: confirmed by reading.
- Where: `frontend/src/pages/MixDeck.tsx::RemixResultCard` (line 303): `<span className="font-display">Mix Ready</span>`
- Symptom: RemixResultCard always shows 'Mix Ready' and never reads the quality_passed flag that the remix task puts in the job result.
- Impact: A render that failed the quality report looks the same as one that passed. scripts/api/task_modules/remix.py sets quality_passed in the result.
- Fix sketch: Read r.quality_passed and show a warning state when it is false.

### UI-17 Signal Search offers k=50 but the similar route caps k at 20

- Severity: medium. Evidence: confirmed by reading.
- Where: `frontend/src/pages/SignalSearch.tsx::SignalSearch` (line 288): `{[5, 10, 20, 50].map((n) => (`
- Symptom: The count buttons include 50; in track mode that calls GET /library/similar/{name}?k=50, and scripts/api/routers/analysis.py::similar_songs declares k le=20, so the API answers 422 and the raw error text is shown.
- Also: Track mode calls analysisApi.similar(refSong, k) with k from [5,10,20,50]; scripts/api/routers/analysis.py::similar_songs declares k le=20, so k=50 returns 422 and the raw error text is shown
- Impact: Choosing 50 and searching by track always fails with a 422. Text mode accepts up to 50.
- Fix sketch: Offer 50 only in text mode, or clamp k to 20 when mode is 'track'.

### UI-18 ConnectionDot labels are inverted relative to how apiHealth is set

- Severity: medium. Evidence: confirmed by reading.
- Where: `frontend/src/shell/LeftRail.tsx::ConnectionDot` (line 123): `? 'Degraded — API reachable, no live stream'`
- Symptom: When SSE is down but the API answers, useJobPoller sets apiHealth 'ok' and the dot is crimson 'Disconnected'. When polling fails (API unreachable) useJobPoller sets 'degraded' and the dot reads amber 'Degraded, API reachable, no live stream'.
- Impact: Status indicator tells the user the API is reachable exactly when it is not, and disconnected when it is reachable; in hosted mode without VITE_COOKIE_DOMAIN the dot stays crimson permanently while the app works via polling.
- Fix sketch: Derive the dot from (apiHealth, sseConnected): ok+sse green, ok+no sse amber 'API reachable, no live stream', degraded/down crimson; or make useJobPoller set 'down' on failure and a distinct state on success without SSE.

### UI-19 Frontend and backend auth modes can disagree silently

- Severity: medium. Evidence: unproven.
- Where: `frontend/src/App.tsx::isSilentError` (line 21): `return error instanceof ApiError && (error.status === 401 || error.status === 503)`
- Symptom: Frontend auth is on only with VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY; backend auth is REMIXMATE_AUTH_ENABLED. /features returns `auth` but no UI reads it. With backend on and frontend off, every gated call 401s, no unauthorized handler is registered, and isSilentError suppresses the toast.
- Impact: A misconfigured deploy shows empty pages with no login screen and no error; the reverse mismatch lets users sign in while the backend treats all of them as the shared 'local' tenant.
- Fix sketch: Compare features.auth with authConfigured once in AppShell and render a blocking configuration error when they differ; or toast 401 when no unauthorized handler is registered.

### UI-20 A dj_chain job type is queried and branched on but never exists

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/pages/MixVault.tsx::MixVault` (line 326): `jobsApi.list({ jobType: 'dj_chain', status: 'done', limit: 100 }),`
- Where: `frontend/src/shell/RightInspector.tsx::JobCard` (line 143): `} else if (job.type === 'dj_chain' && Array.isArray(meta.songs)) {`
- Symptom: Mix Vault requests job_type=dj_chain, but JobType has no dj_chain; chains are created as dj_remix. The second list is always empty and isChain is never true.
- Also: scripts/api/routers/remix.py::dj_chain creates JobType.DJ_REMIX; Mix Vault queries job_type=dj_chain (always empty) and Set Builder upserts type 'dj_chain', which flips to dj_remix on the first SSE frame
- Also: The second /jobs?job_type=dj_chain call always returns an empty list; chains arrive only through the dj_remix call.
- Also: scripts/api/schemas.py::JobType has no dj_chain member and chains are tagged dj_remix, so `job.type === 'dj_chain'` never matches (retry branch, retry button and thumbs conditions).
- Impact: Dead request and dead branch; chains render with the remix icon.
- Fix sketch: Drop the dj_chain request and detect chains via meta.songs once meta is served over REST.

### UI-21 Dead frontend exports, store fields and Vite proxies

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/lib/api.ts::tagsApi` (line 340): `export const tagsApi = {}`
- Where: `frontend/src/stores/appStore.ts::useAppStore` (line 69): `activeNav: 'mission-control',`
- Where: `frontend/vite.config.ts` (line 31): `'/events': {`
- Symptom: tagsApi is an empty object, LibraryStats type does not match backend LibraryStats and has no consumer, FeedbackSummaryResponse and StructureV1Response have no consumer, appStore activeNav and logCompletion are never read or called, cratesApi.rename has no UI caller
- Also: LeftRail and useKeyboardShortcuts call setActiveNav; no component reads activeNav (active state comes from the URL).
- Also: EVENTS_URL defaults to `${BASE}/events/stream` and output URLs are built on BASE ('/api'), so every request goes through the /api proxy; DEVELOPING.md Gotchas still lists /events and /outputs as proxied paths.
- Also: All client URLs go through BASE (/api) and EVENTS_URL defaults to /api/events/stream, so the /events and /outputs proxy entries are never hit
- Impact: Drift and noise for agents reading the client
- Fix sketch: Delete unused exports (update contracts.test.tsx which asserts tagsApi keys are absent)

### UI-22 Error toasts show the raw ApiError string including the response body

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/App.tsx::QueryLayer` (line 58): `const message = error instanceof Error && error.message ? error.message : 'Request failed'`
- Symptom: The global onError uses error.message, which ApiError formats as `[status] path: <body text>`, so users see JSON detail blobs or proxy HTML.
- Impact: Toasts read like `[422] /library/similar/x?k=50: {"detail":[...]}`; long bodies are clipped by the 340px toast.
- Fix sketch: Parse ApiError body JSON for `detail` and map status to a short sentence; keep the raw text in console.

### UI-24 Shortcut help and number slots ignore the generative flag

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/components/ShortcutsModal.tsx::SHORTCUTS` (line 12): `{ keys: 'g → a', desc: 'AI Lab' },`
- Symptom: SHORTCUTS always lists `g → a` AI Lab; Cmd/Ctrl 1 to 8 use fixed NUMBER_NAV indexes, so with AI Lab hidden Cmd 6 does nothing and Cmd 7 opens Mix Vault, the sixth rail item.
- Impact: Help modal advertises a shortcut that does nothing when generative is off; number keys stop matching rail positions.
- Fix sketch: Filter SHORTCUTS by useFeatures().generative; optionally build NUMBER_NAV from the filtered rail list.

### UI-25 Player volume and EQ set before first play are not applied to the graph

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/lib/playerEngine.ts::PlayerEngine` (line 57): `this.gainNode.gain.value = 0.8`
- Symptom: setVolume before the graph exists sets audio.volume; ensureGraph then sets gain 0.8 so effective level is v times 0.8; setEq before first play is dropped because filters are null, while the store shows the value
- Impact: Displayed volume and EQ can differ from what plays until the user touches them again
- Fix sketch: Pass the store's volume and eq into ensureGraph or reapply them after the graph is built

### UI-26 AI Lab model selector is ignored by the API

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/pages/AILab.tsx::AILab` (line 167): `if (model) opts.model = model`
- Symptom: launch adds opts.model, but none of StyleTransferRequest, InpaintRequest or TokenizeRequest declares a model field, so Pydantic drops it.
- Impact: The Model dropdown has no effect on the job.
- Fix sketch: Remove the selector, or add and honour a model field in the generative request schemas.

### UI-27 Open in Vault is a raw anchor

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/pages/MixDeck.tsx::RemixResultCard` (line 319): `<a className="md-result-card__vault-link" href="/mix-vault">`
- Symptom: RemixResultCard uses <a href="/mix-vault"> instead of a router Link, which forces a full reload and ignores the BrowserRouter basename (BASE_URL).
- Impact: Clicking it reloads the SPA, which drops the in memory job store and SSE connection, and it breaks under a non root base path.
- Fix sketch: Use <Link to="/mix-vault"> or navigate('/mix-vault').

### UI-28 Operations hides read only storage stats from non operators

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/pages/Operations.tsx::Operations` (line 326): `{operator && <StoragePanel />}`
- Symptom: StoragePanel, which also shows GET /library/storage stats, is rendered only when features.operator is true, although that GET is not operator gated.
- Impact: Non operators see no storage information.
- Fix sketch: Render the stats for everyone and gate only the prune and evict buttons on operator.

### UI-29 Empty library CTAs point at pages that cannot help

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/pages/SetBuilder.tsx::SetBuilder` (line 348): `{songs.length === 0 ? 'Add songs in Operations' : 'Browse Library Atlas'} <ArrowRight size={12} />`
- Symptom: Set Builder's empty state says 'Add songs in Operations' (Operations has no ingest), and Signal Search's CLAP hint sends users to an Operations Maintenance panel that only operators see.
- Impact: Misleading guidance. The matching Signal Search line is SignalSearch.tsx line 180.
- Fix sketch: Point to the operator ingest note, and show the CLAP hint only when features.operator is true.

### UI-30 Widget pop out resets state

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/pages/Widget.tsx::Widget` (line 321): `<WidgetPanel inPip onPopOut={() => {}} canPip={false} />,`
- Symptom: The inline and portaled WidgetPanel are different element positions, so popping out mounts a fresh panel and the current pick and set list are lost.
- Impact: Users lose their now playing pick and queue when they float the widget.
- Fix sketch: Lift current and setList state into Widget and pass them to both panels.

### UI-31 ToastProvider mounted twice; both toast stacks overlap at the same corner

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/shell/AppShell.tsx::AppShell` (line 72): `<ToastProvider>`
- Symptom: App mounts one ToastProvider (receives QueryLayer errors) and AppShell mounts another (receives useJobToasts). Both render `.toast-area` fixed at the same bottom right position with `z-index` var(--z-toast), above the PlayerBar (`z-index` 60).
- Also: App.tsx and AppShell.tsx both mount ToastProvider; query errors go to the outer stack, job toasts to the inner, both fixed in the same position
- Impact: A query error toast and a job toast shown together draw on top of each other; both cover the right end of the PlayerBar.
- Fix sketch: Remove the AppShell ToastProvider so all toasts share the App level one, and offset `.toast-area` bottom by var(--player-bar-height).

### UI-32 Inspector Retry swallows request failures

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/shell/RightInspector.tsx::JobCard` (line 154): `// silently ignore — user sees the existing error card`
- Symptom: handleRetry catches every error and does nothing, so a 422, 429 job cap or 5xx on retry leaves no trace in toasts or the activity log (the call is outside React Query, so the global onError never sees it).
- Impact: User clicks Retry, nothing happens, no reason given.
- Fix sketch: pushActivity({level:'error', ...}) in the catch, matching handleCancel.

### UI-33 Request types carry fields the backend ignores

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/types/index.ts::DJRemixRequest` (line 253): `transition_duration?: number`
- Symptom: DJRemixRequest has transition_duration, effects, target_bpm and DJPreviewRequest has transition_duration and preset; none exist on the pydantic models, which ignore extras
- Impact: Controls that set these values have no effect on renders
- Fix sketch: Remove the fields or add them to scripts/api/schemas.py with real handling

### UI-34 library_changed and system_status events are never emitted

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/api/routers/events.py::emit_library_changed` (line 221): `async def emit_library_changed() -> None:`
- Symptom: emit_library_changed and emit_system_status have no callers; useSSE handlers for them only ever see the connect ack
- Impact: Library views do not refresh after analysis or deletion by other tabs; dead handler code
- Fix sketch: Emit library_changed when analyze, split or delete jobs finish and invalidate library queries in useSSE

### UI-35 CREDITS.txt served as audio/wav

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/api/routers/library.py::stream_output` (line 499): `return FileResponse(str(path), media_type="audio/wav", filename=safe_filename)`
- Symptom: stream_output hardcodes media_type audio/wav for every file under outputs, including the CREDITS.txt Mix Vault links to
- Impact: Credits download carries the wrong content type; opening it inline would not render as text
- Fix sketch: Let FileResponse guess the type from the suffix, or map .txt and .json explicitly

### UI-36 No `color-scheme` declaration on a dark only UI

- Severity: low. Evidence: unproven.
- Where: `frontend/index.html::head` (line 7): `<meta name="theme-color" content="#09090b" />`
- Symptom: Neither index.html nor tokens.css/global.css declares `color-scheme: dark`, so native controls (select popups, date pickers, form autofill) may render in the light system style.
- Impact: Possible light native widgets on the dark UI; cosmetic.
- Fix sketch: Add `color-scheme: dark` to :root in tokens.css or `<meta name="color-scheme" content="dark">`.

### UI-37 Library list silently truncates at 5000 and is expensive

- Severity: low. Evidence: unproven.
- Where: `frontend/src/lib/api.ts::libraryApi` (line 160): `get<{ songs: SongInfo[] }>('/library?per_page=5000').then((r) => r.songs ?? []),`
- Symptom: libraryApi.list requests per_page=5000 and ignores stats.total_songs; list_library walks every file with rglob on each call and Mission Control refetches every 30 s
- Impact: Beyond 5000 songs pages show a partial library; each call costs a full library walk
- Fix sketch: Page or use /library/names for pickers; compute size stats only in /library/storage

### UI-38 Account fetch bypasses the 401 handler and timeout

- Severity: low. Evidence: unproven.
- Where: `frontend/src/pages/AccountPage.tsx::authedFetch` (line 21):

  ```
  const res = await fetch(`${BASE}${path}`, { method, headers, credentials: 'include' })
  ```

- Symptom: authedFetch does its own fetch, so an expired session on export or delete does not sign the user out and has no timeout
- Impact: Account errors surface as raw text instead of returning to login
- Fix sketch: Export a raw variant of request from lib/api.ts that returns the Response and reuse it

### UI-39 Processing status polls every second in background tabs

- Severity: low. Evidence: unproven.
- Where: `frontend/src/pages/Operations.tsx::ProcessingQueuePanel` (line 70): `refetchIntervalInBackground: true,`
- Symptom: ProcessingQueuePanel sets refetchInterval 1000 with refetchIntervalInBackground true.
- Also: ProcessingQueuePanel sets refetchInterval 1000 with refetchIntervalInBackground true
- Impact: An Operations tab left open scans the library on the server every second.
- Fix sketch: Drop refetchIntervalInBackground or lengthen the interval.

### UI-40 Rail Widget item strands the user in a shellless page

- Severity: low. Evidence: unproven.
- Where: `frontend/src/shell/LeftRail.tsx::NAV_ITEMS` (line 97): `path: '/widget',`
- Symptom: The 'DJ Widget (floating)' rail item navigates the same tab to /widget, which renders outside AppShell with no rail and no link back (Widget.tsx has no navigate or anchor).
- Impact: User must use browser Back to return; the PlayerBar and SSE unmount on the way.
- Fix sketch: Open /widget in a new window (window.open) as the label implies, or add a back link in Widget.

## Mixing

### MX-01 Cached duration is capped at 120 s for every song

- Severity: high. Evidence: confirmed by reading.
- Where: `scripts/core/analysis_pipeline.py::run_song_analysis` (line 206): `"duration":           round(struct.duration, 1),`
- Symptom: Library Atlas duration column, its duration sort and Mix Deck duration read analysis.json duration, which is the length of the first 120 s clip
- Impact: _load_song_audio(song, sr=sr, duration=120.0) at line 141 caps the audio, struct.duration = len(audio)/sr, and scripts/api/routers/_helpers.py reads analysis.json duration into SongInfo.duration, rendered by LibraryAtlas.tsx and MixDeck.tsx. Any song longer than 2:00 shows 2:00.
- Fix sketch: Write the real file duration (soundfile.info on the resolved source or stems) instead of the clip length; keep the 120 s window only for feature extraction

### MX-02 Entry index divides by the stretch ratio even when the 2 percent band skipped the stretch

- Severity: high. Evidence: confirmed by reading.
- Where: `scripts/core/dj_engine.py::DJEngine.render` (line 954): `entry_sample_b = int(plan.entry_time_b * sr / safe_ratio)`
- Symptom: _time_stretch returns B unchanged when the clamped ratio is within 0.02 of 1, but entry_sample_b is still computed as entry_time_b * sr / safe_ratio, so the entry lands early by entry_time_b * (1, 1/ratio) seconds
- Impact: For BPM ratios between 0.98 and 1.02, B enters off its planned phrase start; at an entry around 100 s and a ratio near 1.02 the offset is around 2 s, more than a bar at house tempos. Same pattern in DJEngine.render_stem_blend and in render_chain's _orig_to_stretched.
- Fix sketch: Return the effective ratio from _time_stretch (1.0 inside the band) and use it for index mapping in all three renderers; add a test with ratio 1.015 and a late entry.

### MX-03 Beat grid lock moves song B off its own bar start

- Severity: high. Evidence: confirmed by reading.
- Where: `scripts/core/dj_engine.py::DJEngine.render` (line 982): `phase_a = exit_sample_a % bar_samples`
- Symptom: phase_a and phase_b are taken modulo bar length from sample 0 of each file, but both cues are already bar starts on beat tracker grids with arbitrary offsets, so any nonzero correction shifts B's window start by up to half a bar away from its downbeat relative to A
- Impact: Two song remixes, previews and chains can start B's window up to half a bar off A's bar grid (flam or trainwreck). The same block is copied in DJEngine.render_chain and DJEngine.render_stem_blend.
- Fix sketch: Drop the modulo correction (both cues are bar starts), or align using detected beat or downbeat times near each cue (A's nearest beat vs B's nearest beat in stretched time). Add a click track test with known different downbeat offsets that asserts output alignment.

### MX-04 POST /compatibility scores relative major/minor (and every ring change) as key_score 0 and vetoes compatible

- Severity: high. Evidence: confirmed by reading.
- Where: `scripts/core/track_metadata.py::MetadataClient.compatibility_score` (line 894): `if dist == 999:      # different modes`
- Symptom: 8A vs 8B (A minor vs C major, identical notes) returns key_score 0.0 and compatible=false; strict matches at the default min_score 85 can never include a cross ring partner because overall tops out at 0.65 without the key term
- Impact: camelot_distance returns 999 for any letter change (line 208), key_score becomes 0.0, the compatible veto requires key_score > 0; top_compatibility_matches multiplies overall by 100 so max without key is 65 < 85. Mix Deck compatibility and strict matches hide the safest DJ move. Contradicts pitch_shift_for_camelot, camelot_modulation (relative_shift) and recommend._camelot_score (0.85) on the same pair. tests/test_core_modules.py pins camelot_distance('8B','3A') == 999.
- Fix sketch: Replace track_metadata.camelot_distance/key term with key_detection.camelot_distance or the DJ_THEORY move table (relative pair and same ring neighbours compatible); update the pinned test and add a relative key case to tests/test_compatibility_score.py

### MX-05 /recommend ranks unanalysed songs above analysed ones and treats real 0.5 scores as missing

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/core/recommend.py::_composite_score` (line 283): `has_harmonic = camelot_score != 0.5  # 0.5 = neutral sentinel`
- Where: `scripts/core/recommend.py::_composite_score` (line 300): `return bpm_score  # cold cache fallback`
- Symptom: An unanalysed candidate at equal BPM scores 1.0 (pure bpm) and outranks an analysed adjacent key candidate; a distance 3 Camelot pair (real score 0.50) or an energy gap of 0.25 is treated as no data
- Also: A candidate at Camelot distance 3 (or an energy gap of 0.25) is scored as if its key (or energy) were unknown and reported with harmonic_score null.
- Also: Tracks with no Camelot key rank near the top of GET /recommend/{name}.
- Impact: Cold cache fallback `return bpm_score` (line 300) gives up to 1.0 while the full formula caps below that for any imperfect key; _camelot_score returns exactly 0.50 for dist 3 (line 242) which collides with the sentinel. Matches the w5b report finding that /recommend ranks unanalysed tracks first.
- Fix sketch: Use None for missing dimensions instead of the 0.5 sentinel, and penalise or separate candidates without harmonic data instead of scoring them on BPM alone

### MX-06 Feedback context backfill reads keys the job never stores

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/api/routers/feedback.py::_context_from_job` (line 48): `"transition_bars", "preset", "effect",`
- Symptom: POST /feedback/event with a job_id never backfills preset, effect or tempo_shift_ratio into the logged context.
- Impact: Remix routes store request params under meta.request (key transition_effect, not effect) and results name the ratio tempo_ratio; only transition_bars, harmonic_score and transition_confidence ever match, so Stage B training data lacks the render settings the docstring promises.
- Fix sketch: Read meta.get('request', {}) and map transition_effect to effect and tempo_ratio to tempo_shift_ratio; add a test with a real remix job shape.

### MX-07 Unvalidated transition_effect and bridge options silently degrade renders

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/api/schemas.py::DJRemixRequest` (line 283): `transition_effect: str = Field(`
- Symptom: A typo such as transition_effect='Echo' or 'flanger' returns 202 and renders with no exit effect; bridge_beat_mode values other than auto or file silently mean none.
- Impact: transition_effect, bridge_beat_mode and bridge_beat_genre are plain str in DJRemixRequest, DJChainRequest and DJPreviewRequest; DJEngine.render only branches on auto, stepped, echo, filter, reverb, so anything else falls to no processing.
- Fix sketch: Type the fields as Literal[...] so FastAPI returns 422, and add a test posting an unknown effect.

### MX-08 Running remix and preview ignore cancel and still write output

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/api/task_modules/remix.py::task_dj_remix` (line 445): `sf.write(str(out_path), mix_mastered, sr, subtype="PCM_24")`
- Symptom: DELETE /jobs/{id} on a RUNNING remix shows CANCELLED, but the task keeps rendering, writes outputs/<uid>/dj_<sid>/, records spectral history and runs the retention sweep; the wrapper then discards the result so the folder is orphaned.
- Impact: Only task_dj_chain polls is_cancelled (grep over scripts/); cancel costs a full render of CPU on a 2 worker pool and leaves an unreferenced output dir that still counts toward retention.mix_max_count.
- Fix sketch: Poll is_cancelled between the load, analyse, plan, render and master steps in task_dj_remix and task_remix_preview and return {'cancelled': True} before writing; add a test mirroring test_chain_stops_between_songs_once_cancelled.

### MX-09 Chain task never passes stems_dirs, so chains never use the stem blend

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/api/task_modules/remix.py::task_dj_chain` (line 618): `mix = engine.render_chain(`
- Symptom: task_dj_chain calls render_chain without stems_dirs, making the documented Gap 1C stem branch unreachable
- Also: Every /dj-chain transition uses the plain render_chain crossfade even when all songs have stems.
- Impact: Chains always use the plain cos/sin crossfade with no stem bass swap, per stem LUFS or pitch shift, even when every song has stems. If wired as is, the stem window (pitch shifted, own grid lock) would not match the unshifted chain body segments.
- Fix sketch: Decide whether chains should use stems; if yes pass song_dir per song and make the stem window reuse the chain's b_start and apply pitch shift to the body too; if no, delete the branch and the docstring claim.

### MX-10 Cue export uses energy section starts, not SSM phrase boundaries, and only within the first 120 s

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/core/analysis_pipeline.py::run_song_analysis` (line 201): `round(s.start_time, 2) for s in struct.sections`
- Symptom: rekordbox and Serato cues land on heuristic section label changes (arbitrary bars) and never past two minutes
- Impact: analysis.json phrase_boundaries are built from struct.sections; struct.phrase_boundaries (SSM novelty snapped to 8 bar downbeats) is never persisted; routers/library.py::export_cues uses the list as cue points.
- Fix sketch: Persist struct.phrase_boundaries under phrase_boundaries (and sections under their own key), and analyse the full track or record the window

### MX-11 Classic plan puts the bass swap at the end of a clash shortened transition

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/core/classic_engine/dj_analysis.py::plan_transition` (line 525): `bass_swap_bar = transition_bars // 2`
- Symptom: When classic is selected and harmonic_score < 0.35 shortens a 16 bar transition to 8, EQPlan.bass_swap_bar stays at 8 (computed from 16 bars), so the swap happens at the last bar instead of the midpoint; the current planner builds the EQ plan after shortening.
- Impact: Wrong bass handover timing on classic renders of clashing pairs; reachable only with dj.mixing_engine: classic (config.yaml sets current, but a missing key or bad value falls back to classic).
- Fix sketch: Leave classic frozen unless classic is the task; if fixed, compute EQPlan after the harmonic shortening block as scripts/core/dj_analysis.py::plan_transition does.

### MX-12 Unknown key_profile is accepted and silently replaced by a different key detector

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/core/dj_analysis.py::analyze_structure` (line 415): `log.warning("Key detection (profile=%s) failed: %s", key_profile, exc)`
- Symptom: POST /analyze?key_profile=bogus returns 202 and the job succeeds with the KS only music_intelligence.detect_key result; GET /analysis/v1/{name}/structure does the same
- Impact: key_detection.detect_key raises ValueError for an invalid profile, the broad except at line 409 swallows it, _key_tuple stays None and compute_track_vector re detects with a different algorithm; the routers type key_profile as a free str.
- Fix sketch: Type key_profile as Literal['ks','edma','edmm','auto'] on both routes (422) and let ValueError propagate from analyze_structure

### MX-13 Stem path ignores echo, filter, reverb and the auto exit effect

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/core/dj_engine.py::DJEngine.render_stem_blend` (line 1511): `stepped = str(transition_effect).lower() == "stepped"`
- Symptom: render_stem_blend reads transition_effect only to detect 'stepped'; no exit effect helper is called on this path, while the request schema says 'stepped' keeps the auto exit effect
- Impact: A user choosing echo, filter or reverb for a pair with stems gets no effect in the full render, while the preview (always render()) does apply it.
- Fix sketch: Apply the resolved exit effect to A's stem windows (or the A contribution sum) before summing, mirroring render(); add a test that an echo request changes the stem path window.

### MX-14 Genre confidence is always 1.0

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/core/genre.py::detect_genre` (line 466): `confidence = best_score / max_possible if max_possible > 0 else 0.0`
- Symptom: POST /analyze result confidence is 1.0 for every successful detection
- Impact: max_possible = max(scores.values()) at line 465 is the same value as best_score, so the ratio is 1.0; the exception path returns pop with 0.0.
- Fix sketch: Use a margin (best minus runner up) or a softmax over scores as confidence

### MX-32 GET /library/similar and GET /recommend write meta.json into the shared library

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/api/routers/analysis.py::similar_songs` (line 388): `idx.upsert_song(name)`
- Where: `scripts/core/recommend.py::_save_cache` (line 91): `(song_dir / _CACHE_FILE).write_text(json.dumps(data))`
- Symptom: A GET for an unindexed song without bpm in meta.json runs _quick_features and writes library/<song>/meta.json with raw RMS energy_mean
- Also: similar_songs calls MusicIndex.upsert_song on an index miss. That can write library/<song>/meta.json and data/music_index.json (_save), so a GET has side effects, and the SameSite=Lax cookie authenticates cross site GETs.
- Also: A GET request estimates BPM for up to 120 uncached songs and writes library/<song>/meta.json; on a read only catalog mount the write fails silently and the estimate is redone every call.
- Impact: Side effect on GET (cookie auth accepted on GET per REPORT); _quick_features energy_mean is raw RMS while run_song_analysis stores the normalised curve mean in the same key.
- Fix sketch: Do not persist quick features from a GET, or store them under separate keys

### MX-35 Cancelled job keeps emitting job_cancelled frames and changing message

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/api/jobs.py::update_job` (line 676): `sse_type = status_to_event.get(current_status, "job_updated")`
- Symptom: After a cancel, each progress update from the still running task overwrites the job message and emits another job_cancelled SSE frame.
- Impact: update_job does not check _cancelled; status stays cancelled so the event map yields job_cancelled on every call, and the persisted message becomes e.g. 'Saving output…' on a CANCELLED row. Toast or card logic keyed on job_cancelled can fire repeatedly.
- Fix sketch: In update_job, return early (or drop message/progress changes and skip _emit) when the stored status is cancelled.

### MX-39 Chain hard cuts a last song truncated at the load cap

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/api/task_modules/remix.py::task_dj_chain` (line 566): `audio, _ = load_source_audio(name, sr=sr, mono=True, duration=max_s)`
- Symptom: A chain whose last song is longer than remix.max_song_seconds ends with an abrupt cut instead of the 3 s fade the two song remix applies.
- Impact: task_dj_chain loads with the cap but has no b_truncated check and never calls _fade_out_tail.
- Fix sketch: Detect truncation of songs[-1] like task_dj_remix and apply _fade_out_tail after master_mix.

### MX-67 `dj.default_transition_bars` is dead for API renders; the real default is the request model literal 16

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/core/dj_analysis.py::plan_transition` (line 829): `if transition_bars is None:`
- Where: `scripts/api/schemas.py::DJRemixRequest` (line 261): `transition_bars: int = Field(16, ge=8, le=64, description="8 | 16 | 32")`
- Symptom: plan_transition reads the config default only when transition_bars is None, and every route passes req.transition_bars, whose default is the literal 16 in DJRemixRequest, DJChainRequest and DJPreviewRequest. Inside the unreachable branch the 8 bar rule for a BPM ratio off by more than 0.08 outranks the config value.
- Impact: Changing `dj.default_transition_bars` in config.yaml changes nothing on any API render; an agent or operator tuning it sees no effect. Mix Deck starts its slider at its own value, so three defaults exist.
- Fix sketch: Make the request field Optional with default None and let plan_transition apply the config default (or read the config value into the Field default), then add a test that a config change reaches a planned transition.

### MX-15 Structure endpoint runs full track analysis synchronously per GET

- Severity: medium. Evidence: unproven.
- Where: `scripts/api/routers/analysis.py::song_structure_v1` (line 273): `audio = _load_song_audio(name, sr=sr, duration=None)`
- Symptom: Each GET /analysis/v1/{name}/structure decodes the whole track and runs beat tracking, HPSS key detection per 30 s window, SSM and the full MusicVector in the request thread with no job cap
- Impact: No cache, no _check_job_cap, available to every authenticated user; long tracks can tie up API workers.
- Fix sketch: Cache the structure per song and analysis version, or move it behind a job with the per user cap

### MX-16 Setlist optimize has no track count cap

- Severity: medium. Evidence: unproven.
- Where: `scripts/api/routers/setlist.py::OptimizeRequest` (line 41): `tracks: List[TrackIn] = Field(..., min_length=2)`
- Symptom: A large tracks list or a 5 MB CSV runs an O(n^2) greedy search synchronously inside the request.
- Impact: No max_length; each step calls transition_cost for every remaining track, so thousands of tracks tie up a server thread for a long time on a hosted box.
- Fix sketch: Add max_length (and a row cap after parse_exportify_csv) and a 422 above it.

### MX-17 Preview analyses uncapped audio while the full render analyses the first max_song_seconds

- Severity: medium. Evidence: unproven.
- Where: `scripts/api/task_modules/remix.py::task_remix_preview` (line 722): `audio_a, sr = load_source_audio(song_a, sr=None, mono=True)`
- Symptom: task_remix_preview loads songs with no duration and at A's native rate; task_dj_remix loads the first remix.max_song_seconds at 44100 Hz, so exit threshold (85 percent of duration) and max_exit differ for long songs; the preview also never takes the stem path or pitch shift
- Also: For a song A longer than remix.max_song_seconds the preview can audition a different exit bar than the full mix renders.
- Impact: For songs longer than the cap the preview can audition a different exit point and a different render path than the full mix, contradicting the code comment that what you audition is what the full remix will do.
- Fix sketch: Load with duration=_max_song_seconds() and sr=44100 in the preview, and route through the same stem gate (render_stem_blend with full_output=False); add a parity test on plan fields.

### MX-18 Analysis loader ignores full_enhanced.wav

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/analysis_pipeline.py::_load_song_audio` (line 53): `full_wav = d / "full.wav"`
- Symptom: Songs whose only source is full_enhanced.wav pass `analyze-missing` target selection (has_source_audio) then fail with FileNotFoundError every batch; songs with both stems and full_enhanced.wav are analysed from a stem sum while renders use full_enhanced.wav
- Impact: audio_source._SOURCE_FILES includes full_enhanced.wav; this private loader duplicates it without that file. Whether any library song has full_enhanced.wav and no stems was not checked.
- Fix sketch: Replace _load_song_audio with scripts.core.audio_source.load_source_audio

### MX-19 meta.json write failure is swallowed while the job reports success

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/analysis_pipeline.py::run_song_analysis` (line 195): `pass`
- Symptom: On a read only library mount (docker compose, systemd ProtectSystem) /analyze completes DONE but the song stays unanalysed forever
- Impact: Both meta.json writers (write_meta_cache and the merge block ending at line 194 to 195) use except Exception: pass; has_analysis then stays false; REPORT w1 review flagged the ro mount.
- Fix sketch: Log at WARNING and raise (or return an error flag) when meta.json or analysis.json cannot be written

### MX-20 Octave corrected BPM disagrees with the beat grid

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/beat_tracker.py::LibrosaBeatTracker.track` (line 289): `bpm = resolve_bpm_octave(refined_bpm, audio, sr, hop_length=hop_length)`
- Symptom: When resolve_bpm_octave doubles or halves the tempo, bars stay four tracked beats long while transition_seconds uses the corrected BPM
- Impact: resolve_bpm_octave only changes the scalar (its docstring says so); analyze_structure builds bars from beat_times at the tracker's density; plan_transition mixes bar_start_time (grid) with bars * 4 * 60 / bpm_a. No test checks grid against corrected BPM.
- Fix sketch: When the correction changes the octave, resample the beat grid (interpolate or decimate beats) or skip the correction for the grid and store both values

### MX-21 Classic master_mix flags limited samples as clipping

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/classic_engine/mastering.py::master_mix` (line 292): `clipping, count, peak  = detect_clipping(mastered, threshold_db=ceiling_db)`
- Symptom: Classic checks clipping on the limited output with >= ceiling; its limiter hard clips to exactly the ceiling after a lagging one time constant gain, so pinned samples count and QualityReport.passed goes false.
- Impact: Classic renders the limiter touched (likely most at -8 LUFS) report has_clipping true and quality_passed false, a misleading quality verdict.
- Fix sketch: Judge clipping on the pre limiter input as current mastering.py::master_mix does, or use a threshold just above the ceiling; classic is frozen so only if classic work is in scope.

### MX-22 Section key confidence gate at 0.35 can never reject a successful detection

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/dj_analysis.py::plan_transition` (line 899): `_MIN_SECTION_KEY_CONF = 0.35`
- Symptom: plan_transition always prefers the local phrase key over the whole track key when per phrase data exists
- Impact: key_detection.detect_key maps confidence as (best_r + 1) / 2; rotations of a profile sum to a constant so the best of 24 correlations is at least 0, giving confidence >= 0.5 on every success. Only the 0.0 failure fallback is filtered. DJ_THEORY 'Key Detection Pipeline' says confidence is the normalised correlation.
- Fix sketch: Define confidence from the gap between best and second best correlation (or best_r itself) and recalibrate the gate against labelled tracks

### MX-23 No stretch band leaves B at its own tempo across the window

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/dj_engine.py::_time_stretch` (line 314): `if abs(ratio - 1.0) < 0.02:`
- Symptom: Inside the 2 percent band B plays at bpm_b while the window, bar length and grid lock assume bpm_a, so beats drift apart across the overlap
- Impact: Audible beat drift at the end of 16 bar transitions between songs up to 2 percent apart (for example 125.6 vs 128 BPM drifts roughly half a second over 16 bars, if BPM estimates are right).
- Fix sketch: Lower the band (for example 0.2 percent) or always stretch the window region; add a test measuring onset alignment for a 1.5 percent tempo gap.

### MX-24 Middle song body can be skipped and audio replayed in chains

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/dj_engine.py::DJEngine.render_chain` (line 1335): `body_end    = min(next_exit_b, len(audio_b))`
- Symptom: Plan i+1's exit for the middle song is chosen independently of plan i's entry (spectral matching can move entry up to bar 56); when next_exit_b falls before b_start + trans_s the body is skipped and the next window restarts earlier in the same song
- Impact: A chain can repeat a stretch of the middle song or jump backwards at the second transition.
- Fix sketch: Constrain plan i+1's exit to be at or after plan i's entry plus transition, or clamp next_exit_b to body_start in render_chain; add a chain test with a late entry and an early next exit.

### MX-25 LUFS normalised stem window butts against raw full mix audio

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/dj_engine.py::DJEngine.render_stem_blend` (line 1490): `normed_a = normalize_stems_to_corpus_targets(stems_a_input, targets=_corpus_targets)`
- Symptom: Stems are normalised to corpus LUFS targets and summed for the window while pre_a and post_b are the raw full mixes; only a 30 ms seam blend covers the splice
- Impact: A level or timbre step at the window start and end when the stem sum loudness differs from the source mix.
- Fix sketch: Match the window's loudness at each edge to the adjoining full mix (gain ramp over a bar), or normalise per stem relative to the song's own mix; add a test comparing short term loudness across both seams.

### MX-26 Instrument Lab is ungated and mode 'all' renders songs^4 full length combos

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/instrument_lab.py::render_all_combos` (line 378): `combos = enumerate_combos(songs, include_pure=include_pure)`
- Symptom: POST /instrument-lab needs only require_user; InstrumentLabRequest.mode is a free str and any value other than 'targeted' goes to enumerate_combos, which with 4 songs yields 4^4 combos, each full length when target_duration is None.
- Impact: One signed in user can occupy a worker with hundreds of full length stretch and write passes and fill outputs/<uid>/lab_<id>/.
- Fix sketch: Make mode a Literal['targeted','all'], cap combos or require target_duration for 'all', or gate the route like the other heavy features.

### MX-27 psychoacoustic_consonance never goes below about 0.89, so every clash branch keyed to 0.35 is dead in the current engine

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/key_detection.py::psychoacoustic_consonance` (line 128): `max_roughness = n_partials * n_partials  # conservative upper bound`
- Symptom: plan_transition never shortens a clashing transition, render never picks the filter sweep exit, and the stem blend confidence gate passes whenever spectral_score is above about -0.29
- Impact: A standalone replica of lines 81 to 130 (no project import) gives min 0.892 (C vs C#) over all 144 tonic pairs at 72e2368; the function also ignores mode and places both tonics in octave 4. dj_analysis.py:911 `harmonic_score < 0.35`, dj_engine.py:1016 and :1248, and remix.py::_should_use_stem_blend threshold 0.30 are unreachable or always true. tests/test_behavioral.py only exercises shortening by patching the scorer to 0.10.
- Fix sketch: Drive plan_transition.harmonic_score from a Camelot or TIV scorer with a real low end (or rescale consonance to pitch class intervals with a normalizer that reaches 0), then add an unpatched clash test (C major vs F# major must shorten)

### MX-28 Key detection failure is cached as a real C major 8B

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/key_detection.py::detect_key` (line 368): `camelot='8B',`
- Symptom: A song whose key detection threw shows 8B in meta.json, Library Atlas and every scorer, with no marker
- Impact: The except branch returns C major 8B with confidence 0.0; run_song_analysis persists key, mode and camelot but not key_confidence, so downstream cannot tell the fallback from a real 8B. music_intelligence.detect_key and MusicVector defaults do the same.
- Fix sketch: Return empty key fields (or persist key_confidence) on failure and have scorers treat them as unknown

### MX-29 Spectral sync bass guard protects no band at the configured 150 Hz

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/spectral_sync.py::sync_transition_spectrum` (line 115): `band_gain[band_centers <= bass_crossover_hz, :] = 1.0`
- Symptom: The guard sets unity gain only for mel bands whose centre is at or below bass_crossover_hz; with 24 bands and n_fft 2048 band 0 is centred at 160.1 Hz (44100 Hz) or 163.4 Hz (48000 Hz), measured with librosa 0.11.0 in .venv
- Impact: The 20 to 300 Hz band gets up to plus or minus dj.spectral_sync_max_db inside the transition, contradicting the module docstring, config comment and DEVELOPING.md claim that bass stays untouched.
- Fix sketch: Protect bins below the crossover directly (bin_gain[freqs <= crossover] = 1) or protect bands whose lower edge is below it; add a test that low frequency energy is bit identical in the window.

### MX-30 Rebuilt source from stems has a different balance than the original mix

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/stems.py::separate_song_stems` (line 385): `normalised = _enhance_stems(stem_arrays, sr)`
- Symptom: separate_song_stems normalizes each stem to its own LUFS target (vocals -14, drums -16, bass -18, other -18) via enhance_stems and rewrites the files; after prune, load_source_audio sums these stems as the song's source.
- Impact: Analysis and plain render() input for pruned songs is a rebalanced mix (vocals pushed up relative to bass), not the ingested track.
- Fix sketch: Store stems at Demucs gain (or store the per stem gain) and apply corpus normalization only inside render_stem_blend, which already does it.

### MX-31 Generative sources default to full.wav, which prune deletes

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/style_transfer.py::_load_source_audio` (line 155): `candidates = [d / "full.wav", d / "full.flac"]`
- Symptom: style_transfer._load_source_audio and inpainting._load_stem look only for full.wav/full.flac when source_stem is 'full' (the default); after a clean stem split _prune_after_split deletes full.wav and full_enhanced.wav, so the request fails with FileNotFoundError. They do not use audio_source.load_source_audio.
- Impact: With REMIXMATE_GENERATIVE_ENABLED on, style transfer and inpainting fail for every split and pruned song; same pattern at scripts/core/inpainting.py::_load_stem.
- Fix sketch: Route both loaders through scripts/core/audio_source.py (resolve_source_file then stem sum) for 'full', and stem_files for named stems.

### MX-62 Camelot move scorers ignore the ring letter for adjacency

- Severity: medium. Evidence: unproven.
- Where: `scripts/core/music_intelligence.py::_camelot_move_analysis` (line 591): `if dist == 1:`
- Symptom: 8A to 9B (diagonal) scores 0.90 'adjacent' in compute_transition_score and /recommend, while camelot_compatible and DJ_THEORY's table (dist == 1, same letter) treat it as not adjacent
- Impact: Same rule copied in scripts/core/recommend.py::_camelot_score line 233; inflates harmonic scores for cross ring neighbours and disagrees with pitch_shift_for_camelot.
- Fix sketch: Require let_a == let_b for the adjacent branch and give diagonal its own score; share one scorer between music_intelligence and recommend

### MX-33 preset request field is accepted and ignored

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/api/schemas.py::DJRemixRequest` (line 262): `preset: str = Field("auto", description="Genre preset or 'auto'")`
- Where: `scripts/api/task_modules/remix.py::task_dj_remix` (line 293): `preset: str,`
- Symptom: Changing preset on /dj-remix or /dj-chain changes nothing in the render.
- Also: preset reaches task_dj_remix and task_dj_chain only as log_audit metadata; no planning or render step reads it
- Impact: task_dj_remix and task_dj_chain only pass preset to log_audit.
- Fix sketch: Either wire preset into planning or remove it from the request models and the frontend request types.

### MX-34 Dead mixing code: TIV branch, unused helpers, constants and branches

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/dj_analysis.py::plan_transition` (line 958): `chroma_a = getattr(song_a, "mean_chroma", None)`
- Where: `scripts/api/jobs.py::retry_job` (line 786): `def retry_job(job_id: str, fn: Callable, **kwargs) -> Optional[str]:`
- Where: `scripts/core/dj_engine.py::_HP_START_HZ` (line 88): `_HP_START_HZ: float     = getattr(getattr(_cfg, "dj", None), "hp_filter_start_hz", 400.0)`
- Where: `scripts/core/dj_engine.py::DJEngine.render_from_stems` (line 1360): `def render_from_stems(`
- Where: `scripts/core/genre.py::_extract_detection_features` (line 335): `feats["chroma_variance"] = float(np.var(chroma))`
- Where: `scripts/core/key_detection.py::camelot_modulation` (line 621): `if min_delta == 7:`
- Symptom: plan_transition reads mean_chroma via getattr but no code sets that attribute on SongStructure, so tiv_compatibility is always None
- Also: Module docstring advertises retry_job, but no route or task calls it.
- Also: _HP_START_HZ, _HP_END_HZ and _BASS_CROSSOVER are read from config in dj_engine.py but never used; the effective values are the copies in dj_analysis.py
- Also: No caller in scripts or tests for DJEngine.render_from_stems or _fade_envelope
- Also: chroma_variance (a full chroma_cqt) and low_energy_ratio are computed per analysis but _score_genre reads neither
- Also: +7 wheel moves (one semitone up) are classified non_standard instead of modulated_boost
- Impact: TIV scoring never contributes; remix_recipe always falls back to harmonic_score.
- Fix sketch: Populate mean_chroma in analyze_structure or delete the branch and the field.

### MX-36 Wordplay bonus is applied as an energy bump, not a pair preference

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/api/routers/setlist.py::optimize_setlist` (line 262): `tgt.energy = min(1.0, tgt.energy + bonus * 0.05)`
- Where: `scripts/api/routers/setlist.py::optimize_setlist` (line 264): `planner = SetlistPlanner()`
- Symptom: wordplay_weight nudges a track's energy upward regardless of which track precedes it, so lyric links do not steer adjacency.
- Also: POST /setlist/optimize with any markov_weight returns the same order.
- Impact: The src id of each pair is ignored; the bump changes the arc fit and the reported energy of the target track instead of lowering the src to tgt transition cost. Same code in import_exportify_csv.
- Fix sketch: Pass a pair bonus map into SetlistPlanner.optimize and subtract it from transition cost for that pair.

### MX-37 `analyze-missing` upserts each song into the index twice

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/api/task_modules/analysis.py::task_analyze_missing` (line 102): `_index_upsert(name)`
- Symptom: Two MusicIndex.upsert_song calls (each rewriting data/music_index.json) per analysed song
- Impact: run_song_analysis already upserts at analysis_pipeline.py lines 215 to 219; extra full index JSON write per song in batch runs.
- Fix sketch: Drop the _index_upsert call in task_analyze_missing

### MX-38 Remix fade applied after mastering report

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/api/task_modules/remix.py::task_dj_remix` (line 436): `mix_mastered = _fade_out_tail(mix_mastered, sr)`
- Symptom: _fade_out_tail runs after master_mix, so quality_report LUFS and peak describe the audio before the 3 s fade.
- Also: Result lufs and peak_dbfs for a remix whose song B hit the cap do not match the saved file.
- Impact: Reported LUFS is slightly off for truncated renders; chain renders never fade a truncated last song.
- Fix sketch: Fade before master_mix or recompute the report after the fade.

### MX-40 CLAP index reads an energy field analysis.json never contains

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/crate_digger.py::CrateDigger.index_library` (line 305): `meta_entry["energy"]  = float(d.get("energy", 0.5))`
- Symptom: Every `crate-search` result reports energy 0.5
- Impact: analysis.json keys are bpm, phrase_boundaries, duration, key, camelot; energy_mean lives in meta.json.
- Fix sketch: Read energy_mean from meta.json

### MX-41 Phrase snap preference for 16 and 32 bars is subsumed by 8

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/dj_analysis.py::_snap_to_bar_grid` (line 686): `on_preferred = any(bar_index % n == 0 for n in preferred_lengths_bars)`
- Symptom: _snap_to_bar_grid docstring promises a preference for 16 and 32 bar multiples; any multiple of 8 scores the same
- Impact: Every multiple of 16 or 32 is a multiple of 8, so the extra entries change nothing.
- Fix sketch: Score by the largest matching length, or document 8 bar snapping only

### MX-42 B high pass ramp only processes the half where B is multiplied by zero

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/dj_engine.py::DJEngine.render` (line 1031): `b_hp_section = b_trans[:hp_ramp_samples]`
- Symptom: render and render_chain high pass B only over the first half of the window, which the dynamic fade (render) or b_env (render_chain) zeroes; the audible half is unfiltered
- Impact: dj.hp_filter_start_hz and dj.hp_filter_end_hz have no audible effect in the current engine; the class docstring's 'introduce without bass clash' step does nothing.
- Fix sketch: Either move the ramp onto the audible half (before the swap) or delete it and the config keys; document that bass separation relies on A's high pass after the swap.

### MX-43 Instrument Lab labels assign letters from set order per combo

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/instrument_lab.py::_combo_filename` (line 441): `songs = list(set(combo.mapping.values()))`
- Symptom: _combo_filename and InstrumentCombo.short_label map songs to letters via list(set(...)) of that combo's own sources, so letter A differs across combos and runs.
- Impact: File names and job result labels do not identify which song supplied a stem; only the mapping field does.
- Fix sketch: Build the letter map once from the request's songs order and pass it in.

### MX-44 Current master_mix clipping check is dead for chain renders

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/mastering.py::master_mix` (line 505): `clipping, count, _     = detect_clipping(audio)`
- Symptom: Clipping is judged on the input to master_mix; render_chain peak normalizes its output to 0.891 before mastering, below the -0.3 dBFS threshold, so has_clipping can never be true for chains.
- Impact: QualityReport.has_clipping carries no information for chain mixes (and for render() output unless spectral sync boosted peaks).
- Fix sketch: Decide what the flag should mean (source clipping before engine normalization, or output overshoot) and measure at that point.

### MX-46 Legacy key detectors mislabel keys

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/musical_analysis.py::MusicalAnalyzer._detect_key_chroma` (line 126): `key_names = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']`
- Where: `scripts/core/metrics.py::AudioMetrics.detect_key` (line 133): `key_names = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']`
- Symptom: MusicalAnalyzer and AudioMetrics name the chromatic correlation index with a circle of fifths list, so a C# major detection is reported as G; minor Camelot numbers are also wrong (Am=5)
- Impact: np.roll(MAJOR_PROFILE, i) is the key i semitones above C, but key_names[i] follows fifths; reached by the remixmate CLI (python_bridge.py, real_optimizer.py) and tests/e2e_test_suite.py, not by the API.
- Fix sketch: Delete the legacy detectors in favour of key_detection.detect_key, or index NOTE_NAMES chromatically and use key_detection.CAMELOT

### MX-47 PythonBridge.dj_remix mixes the render with itself

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/python_bridge.py::PythonBridge.dj_remix` (line 449): `mix_result  = self.mixer.mix_audio(mixed, mixed, mix_params)`
- Symptom: dj_remix passes the same rendered array as both inputs to mix_audio; the method has no callers.
- Impact: Dead code; would double the signal if ever wired up.
- Fix sketch: Delete dj_remix or pass the mastering step a real second input.

### MX-48 arc_target_energy on each row belongs to the next track

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/setlist_planner.py::SetlistPlanner.optimize` (line 534): `"arc_target_energy": round(_arc_target_energy(position, arc), 3),`
- Symptom: The setlist shows each track's arc target shifted by one position.
- Impact: position is len(ordered) / n, the candidate's position, but the value is stored on current's row; the last row is hardcoded to 1.0.
- Fix sketch: Record _arc_target_energy(current_position / max(n, 1, 1), arc) on current's row.

### MX-49 Stem split reports success with fewer than four stems

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/stems.py::separate_song_stems` (line 367): `if not produced:`
- Symptom: separate_song_stems returns success when any stem was produced; task_stem_split returns success True with n_stems < 4 and prune then skips silently.
- Impact: Operator sees a successful job for a half split song; the remix gate (any one stem) then treats it as stemmed.
- Fix sketch: Fail or mark partial when fewer than four stems are produced.

### MX-50 /ai/tokenize writes into the shared library without an operator check

- Severity: low. Evidence: unproven.
- Where: `scripts/api/routers/generative.py::tokenize_stems` (line 146): `def tokenize_stems(req: TokenizeRequest, background_tasks: BackgroundTasks = None):`
- Symptom: The generative router only has the env flag gate; tokenize_stems writes library/<song>/tokens/<stem>.npz, while every other library writing route uses require_operator.
- Impact: With the generative flag on in hosted mode, any signed in user writes files into the shared catalog.
- Fix sketch: Add Depends(require_operator) to /ai/tokenize or write tokens under user_outputs_dir.

### MX-51 stem_blend result field reports the gate decision, not the path taken

- Severity: low. Evidence: unproven.
- Where: `scripts/api/task_modules/remix.py::task_dj_remix` (line 512): `"stem_blend":        use_stem_blend,`
- Symptom: render_stem_blend falls back to render() when no stem file loads, but the result still says stem_blend true
- Impact: Misleading job metadata when stem files exist but fail to read.
- Fix sketch: Have render_stem_blend report whether it used stems (return info or set an attribute) and surface that.

### MX-52 audio_enhance true peak limiter is whole file peak scaling

- Severity: low. Evidence: unproven.
- Where: `scripts/core/audio_enhance.py::apply_true_peak_limiter` (line 278): `audio = audio * (ceil / peak)`
- Symptom: apply_true_peak_limiter multiplies the whole array by ceiling/peak when the sample peak exceeds the ceiling; it runs after LUFS normalization in enhance_audio and enhance_stems.
- Impact: One transient can drop the Demucs input or a stem well below its LUFS target; the name and module docstring promise a limiter.
- Fix sketch: Use scripts/core/mastering.py::apply_limiter or rename to peak_normalize.

### MX-53 Tokenizer reads WAV stems only

- Severity: low. Evidence: unproven.
- Where: `scripts/core/codec_tokens.py::tokenize_stems` (line 511): `stem_paths = {name: song_path / f"{name}.wav" for name in stem_names}`
- Symptom: tokenize_stems builds paths with .wav only, so after FLAC compression it returns 'No stems found'.
- Impact: POST /ai/tokenize fails for compressed songs when the generative flag is on.
- Fix sketch: Use scripts/core/audio_source.py::stem_files.

### MX-54 Beat grid ignores model predicted downbeats

- Severity: low. Evidence: unproven.
- Where: `scripts/core/dj_analysis.py::analyze_structure` (line 313): `bar_index = i // 4`
- Symptom: With beat_this installed, bars start every fourth tracked beat from the first beat while phrase snapping uses the model's downbeats
- Impact: plan_transition maps snapped boundaries to the nearest bar start of a grid that may be phase shifted; beat_this is not installed in the repo .venv, so default local runs are unaffected.
- Fix sketch: Build bars from BeatResult.downbeat_times when the backend is beat_this

### MX-55 Short sections are dropped, leaving gaps in bar coverage

- Severity: low. Evidence: unproven.
- Where: `scripts/core/dj_analysis.py::_label_sections` (line 583): `if length >= min_section_bars:`
- Symptom: Runs under 4 bars vanish from struct.sections, so a time can fall in no section
- Impact: _local_key in plan_transition falls back to the whole track key inside gaps; analysis.json cue list skips them.
- Fix sketch: Merge short runs into the previous section instead of dropping them

### MX-56 render() dynamic fade leaves B's high band at about 0.85 then steps to unity

- Severity: low. Evidence: unproven.
- Where: `scripts/core/dj_engine.py::DJEngine._apply_dynamic_eq_fade` (line 839): `fade_high = np.sin(np.clip(t * 0.65, 0, np.pi / 2))  # Slow rise`
- Symptom: fade_high for direction 'in' is sin(0.65 t), which ends at sin(0.65 * pi / 2), about 0.85; post_b then continues at full level; the band sum is also spliced against unfiltered audio at both edges
- Impact: About a 1.4 dB step in B's highs at the end of every render() transition, plus possible clicks where the band sum meets unfiltered audio.
- Fix sketch: Scale the slow curves so they reach 1 (or 0) at the window end, e.g. sin(clip(t / 0.65 ...)) style mapping, and crossfade the band sum into the original over a few ms.

### MX-57 render_chain uses a different gain curve from render()

- Severity: low. Evidence: unproven.
- Where: `scripts/core/dj_engine.py::DJEngine.render_chain` (line 1310): `a_env[mid:] = np.cos(np.linspace(0.0, np.pi / 2.0, tail, dtype=np.float32))`
- Symptom: Chain windows use a plain cos/sin second half fade with no 3 band dynamic fade, so fixes to render() never reach chains
- Impact: Chain transitions sound different from two song remixes of the same pair.
- Fix sketch: Factor one window renderer used by render and render_chain.

### MX-58 Stem similarity compares processed A stems with raw unstretched B stems

- Severity: low. Evidence: unproven.
- Where: `scripts/core/dj_engine.py::DJEngine.render_stem_blend` (line 1612): `similarities[s] = _stem_similarity(stems_a.get(s), stems_b.get(s), sr)`
- Symptom: similarities uses stems_b (original B) instead of stretched_b, while stems_a was already LUFS normalised and masking EQ processed; the rms feature is level dependent
- Impact: Curve choice (extended, equal power, sharp) can be driven by level and tempo differences instead of timbre.
- Fix sketch: Compute similarity on matching processed windows (A exit window vs stretched, normalised B entry window).

### MX-59 GPU time stretch always raises on MPS and silently falls back

- Severity: low. Evidence: unproven.
- Where: `scripts/core/gpu.py::gpu_time_stretch` (line 269): `stretched = torchaudio.functional.phase_vocoder(stft, rate, torch.tensor([hop]))`
- Symptom: phase_advance is passed as a CPU tensor torch.tensor([hop]) while the STFT lives on the device; probe with torch 2.7.1 on MPS raised RuntimeError 'Expected all tensors to be on the same device'; the except logs at debug and uses librosa
- Impact: The GPU stretch never runs; every stretch pays a device STFT then the librosa CPU path. Output quality is unaffected.
- Fix sketch: Build phase_advance on the device as torch.linspace(0, math.pi * hop, n_freq, device=dev)[..., None] and add a device test (skipped without a GPU).

### MX-60 Instrument Lab tempo fallback reads full.wav then assumes 120

- Severity: low. Evidence: unproven.
- Where: `scripts/core/instrument_lab.py::_get_song_bpm` (line 207): `wav = song_dir(song_name) / "full.wav"`
- Symptom: _get_song_bpm falls back to a librosa estimate on full.wav only, which prune deletes, then returns 120.0.
- Impact: Songs without cached bpm in metadata get stretched as if 120 BPM, producing off tempo hybrids.
- Fix sketch: Use load_source_audio or refuse songs without analysed BPM.

### MX-61 Similarity vector key term is a tonic one hot, not Camelot aware; drop dimension is constant for analysed songs

- Severity: low. Evidence: unproven.
- Where: `scripts/core/music_index.py::_build_vector` (line 124): `vec[1 + idx] = 1.0`
- Symptom: /library/similar treats A minor vs C major and C vs G as unrelated keys; dim 16 is always 0.6 because meta.json has no duration or drop_position
- Impact: Module docstring and route docstring describe the key block as harmonic compatibility (Camelot wheel); line 136 falls back to 0.6 when duration or drop is missing, which run_song_analysis never persists to meta.json.
- Fix sketch: Encode key as Camelot position on a circle (sin, cos per ring) and persist duration and drop_position to meta.json

### MX-63 Spectral sync target ramp spans the whole window while render() B is silent in the first half

- Severity: low. Evidence: unproven.
- Where: `scripts/core/spectral_sync.py::sync_transition_spectrum` (line 105): `r = (1.0 - np.cos(np.linspace(0.0, np.pi, frames))) / 2.0`
- Symptom: The target morphs from A's profile to B's over the full window (0.5 at mid), so in render() A alone is equalised halfway toward B before B is audible
- Impact: Colouration of A's solo half on the plain render path.
- Fix sketch: Pass the audible overlap span from the renderer (second half for render()) to apply_spectral_sync.

### MX-64 Second split writes WAV stems while stale FLAC stems win

- Severity: low. Evidence: unproven.
- Where: `scripts/core/stems.py::separate_song_stems` (line 356): `dst = write_dir / f"{stem_name}.wav"`
- Symptom: separate_song_stems always writes <stem>.wav and never removes an existing <stem>.flac; stem_files and _load_stem prefer .flac, and task_compress_stems skips a stem whose .flac exists, leaving both.
- Impact: When a source file survives (keep_raw_after_separation true, or split before compression) a new split is silently ignored by every renderer in favour of the old FLAC.
- Fix sketch: Unlink <stem>.flac when writing a new <stem>.wav into the permanent song dir.

### MX-65 key_to_camelot has no C# minor entry

- Severity: low. Evidence: unproven.
- Where: `scripts/core/track_metadata.py::_CAMELOT` (line 175): `("E",  "major"): "12B", ("Db", "minor"): "12A",`
- Symptom: key_to_camelot('C#', 'minor') returns None although every other sharp minor maps
- Impact: LocalAnalysisProvider emits sharp names; external metadata with C# minor would get camelot None and a neutral key score.
- Fix sketch: Add ("C#", "minor"): "12A" or derive this table from key_detection.CAMELOT with enharmonic aliases

### MX-66 Guard test plan fixture uses the inverted tempo ratio

- Severity: low. Evidence: unproven.
- Where: `tests/test_dj_engine_guards.py::_plan` (line 62): `tempo_shift_ratio=bpm_b / bpm_a if bpm_a > 0 else 1.0,`
- Symptom: The shared plan fixture builds tempo_shift_ratio as bpm_b / bpm_a, the opposite of the TransitionPlan contract
- Impact: Any future assertion on stretch direction built on this fixture would pin the wrong behaviour.
- Fix sketch: Use bpm_a / bpm_b.

## Operations and security

### OPS-01 DELETE /account reports clean when the job purge fails in the DB

- Severity: high. Evidence: confirmed by reading.
- Where: `scripts/api/jobs.py::purge_user_jobs` (line 486): `deleted = max(deleted, len(ids))`
- Symptom: purge_user_jobs pops the in memory jobs first, catches any DB exception, and returns a positive count. delete_account's _step never sees the error, and _residual checks list_jobs with no filter, which reads only the in memory dict, so the result is clean=true with the job rows still in the database.
- Impact: An account deletion can claim complete while the user's job rows (meta, queries, results) remain in jobs.db or Postgres. That is a privacy and compliance failure.
- Fix sketch: Re raise the DB exception from purge_user_jobs (or return a failure flag) so _step records it, and have _residual query the DB directly (SELECT count WHERE meta user_id = uid) instead of list_jobs.

### OPS-02 POST /library/storage/prune has no dry run or preview

- Severity: high. Evidence: confirmed by reading.
- Where: `scripts/api/routers/library.py::storage_prune` (line 235): `def storage_prune():`
- Symptom: A single body less POST deletes full.wav and full_enhanced.wav for every song with four healthy stems, synchronously, with no preview or confirmation. With auth off, require_operator lets any local caller in. This is the call that removed the library's full mixes during the hosted checks.
- Impact: Irreversible loss of full length source audio across the whole library in one request. STATUS.md records the loss and says there is no backup.
- Fix sketch: Add dry_run: bool = Query(True) like storage_evict, returning the would prune list and freed bytes without calling prune_raw or prune_enhanced; have the UI require a dry run response before sending dry_run=false.

### OPS-03 Local mode catalog mutations reachable by a cross site simple POST

- Severity: high. Evidence: unproven.
- Where: `scripts/api/auth.py::require_operator` (line 334): `if not auth_enabled():`
- Symptom: With auth off, require_operator returns 'local' for any caller. Body less POST routes (storage/prune, storage/evict?dry_run=false, stems/compress-batch, mixes/sweep?dry_run=false) are CORS simple requests, so any web page open in the operator's browser can send them to 127.0.0.1:8000 without a preflight. CORS only hides the response.
- Impact: A malicious page could delete library audio on a dev box running the API. Whether the browser blocks public to loopback requests (Private or Local Network Access) depends on the browser and version, so this is unverified.
- Fix sketch: In local mode require a custom header (e.g. X-RemixMate-Local: 1, which forces a preflight) or a JSON body on every operator and destructive route, or reject requests whose Origin is not in the CORS allow list.

### OPS-04 evict_lru dry run always returns an empty list

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/core/library.py::LibraryManager.evict_lru` (line 766): `if dry_run:`
- Symptom: In dry run, pass 1 only logs and continues, and pass 2 only logs. Neither appends to evicted, so POST /library/storage/evict (dry_run defaults to true) reports that nothing would be evicted, even when dry_run=false would delete whole song directories.
- Impact: The safety preview is blind. An operator who sees an empty list and runs dry_run=false can lose full mixes and whole song dirs, stems included.
- Fix sketch: In both dry run branches append entry.name to evicted and decrement a simulated current by the file or dir size, so the preview matches what the real run would delete.

### OPS-05 Operator supplied ingest name is not sanitized

- Severity: medium. Evidence: confirmed by reading.
- Where: `scripts/download.py::ingest_track` (line 319): `dest_dir = song_dir(name)`
- Symptom: ingest_track only checks that sanitize_song_name(name) is non empty, then uses the raw name for dest_dir. A --name such as 'Artist: Title' lands on disk but fails SONG_NAME_RE in _require_song, so every API call for that song returns 400.
- Impact: An ingested catalog track can be unreachable through the API (audio, detail, analyze, delete) until it is renamed by hand.
- Fix sketch: Set name = sanitize_song_name(name) before song_dir(), or reject names where sanitize_song_name(name) != name with a clear error.

### OPS-06 backup.sh leaves dumps world readable and puts the DB password on the command line

- Severity: medium. Evidence: unproven.
- Where: `app/deploy/backup.sh::backup.sh` (line 54): `pg_dump "$PG_URL" | gzip > "$BACKUP_FILE"`
- Symptom: There is no umask or chmod before writing the pg_dump and tar outputs, and pg_dump receives the full URL including the password as an argv element, which is visible in ps.
- Impact: Other local users on the VPS can read user data backups, and the DB password is exposed in the process list while the backup runs.
- Fix sketch: umask 077 at the top; pass credentials through PGPASSWORD or a .pgpass file and give pg_dump the URL without the password.

### OPS-07 `docker-compose` and the systemd unit mount the catalog read only while routes write into it

- Severity: medium. Evidence: unproven.
- Where: `docker-compose.yml::services.api.volumes` (line 23): `- ./library:/app/library:ro`
- Symptom: Compose mounts ./library read only, and remixmate.service's ReadWritePaths omits /mnt/catalog. /analyze, /library/similar, stem split and LibraryManager index saves write meta.json, analysis.json, stems and .index.json into LIBRARY_DIR.
- Impact: Analysis and index writes fail, possibly silently where callers swallow exceptions, so recommendations and similarity may run on stale or missing metadata in deployed setups.
- Fix sketch: Move per song derived data (meta.json, analysis.json, .index.json) under DATA_DIR, or document and allow a writable catalog path; add a readiness check that probes writability.

### OPS-08 Importing scripts.api.jobs from a second process fails the live API's jobs

- Severity: medium. Evidence: unproven.
- Where: `scripts/api/jobs.py::_load_from_db` (line 342): `_load_from_db()`
- Symptom: _load_from_db runs at import and rewrites every RUNNING or PENDING row it loads to FAILED. Any CLI, probe or script that imports scripts.api.jobs (directly or through scripts.api.main) while the API is running marks the API's jobs that are still running as FAILED in the DB.
- Impact: Live jobs show as failed in the DB and in later loads. REPORT seeds record this happening to data/jobs.db during an import check.
- Fix sketch: Move the rollback into the lifespan startup (once per server) instead of at import, or stamp rows with a process id or boot id and only roll back rows owned by a dead process.

### OPS-09 python -m scripts.api.main binds 0.0.0.0 in local mode without --no-proxy-headers

- Severity: medium. Evidence: unproven.
- Where: `scripts/api/main.py` (line 248): `host = getattr(getattr(cfg, "api", None), "host", "0.0.0.0")`
- Symptom: The __main__ runner takes host from cfg.api.host, which config.yaml sets to 0.0.0.0, and calls uvicorn.run without proxy_headers=False. main.py's docstring advertises this entry point. The Dockerfile uses it too, which is fine inside a container.
- Impact: A developer running the documented module entry exposes an unauthenticated API with open operator and DELETE /account routes to the LAN.
- Fix sketch: Default the __main__ host to 127.0.0.1 unless auth is enabled or an explicit env override is set, and pass proxy_headers=False to match start.sh and the systemd unit.

### OPS-10 conftest does not unset DATABASE_URL

- Severity: medium. Evidence: unproven.
- Where: `tests/conftest.py` (line 75): `if os.environ.get("REMIXMATE_TESTS_USE_REAL_DATA") != "1":`
- Symptom: The storage isolation shim redirects the data and outputs env vars but leaves DATABASE_URL. db.get_engine uses that URL for every store whenever it is set.
- Impact: A pytest run from a shell with DATABASE_URL exported (e.g. after sourcing the hosted env) writes test jobs, crates and feedback into that Postgres.
- Fix sketch: os.environ.pop("DATABASE_URL", None) inside the isolation block, and add a case to tests/test_storage_isolation.py.

### OPS-11 remix.max_pitch_shift_semitones is read by nothing

- Severity: low. Evidence: confirmed by reading.
- Where: `config.yaml::remix.max_pitch_shift_semitones` (line 26): `max_pitch_shift_semitones: 2  # maximum allowed pitch correction`
- Symptom: config.yaml advertises a 2 semitone cap; pitch_shift_for_camelot hardcodes 3
- Impact: grep finds no reader outside config.py; editing the key has no effect.
- Fix sketch: Pass cfg.remix.max_pitch_shift_semitones into pitch_shift_for_camelot or delete the key

### OPS-12 library.auto_evict_on_download is read by no code path

- Severity: low. Evidence: confirmed by reading.
- Where: `config.yaml::library.auto_evict_on_download` (line 92): `auto_evict_on_download: true  # Run LRU eviction after every download — can delete`
- Symptom: config.yaml and the LibraryConfig comment say eviction runs after every download, but no code calls evict_lru outside POST /library/storage/evict. The key is only echoed by GET /library/storage.
- Impact: /library/storage reports a behaviour that never happens; operators may expect the cap to be enforced.
- Fix sketch: Remove the key and its schema field, or wire it into ingest_track after register.

### OPS-13 Demucs model name is unvalidated free text

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/api/schemas.py::StemSplitRequest` (line 245): `model: str = Field("htdemucs", description="Demucs model name (htdemucs | htdemucs_ft | mdx_extra)")`
- Symptom: StemSplitRequest.model is a plain str; an unknown value runs the enhance step (writing full_enhanced.wav), fails Demucs, retries on CPU, then fails.
- Impact: Operator typo wastes a worker slot for two Demucs attempts; no 422.
- Fix sketch: Literal['htdemucs','htdemucs_ft','mdx_extra'] on StemSplitRequest and BatchStemSplitRequest.

### OPS-14 separation config block is read by no stem code

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/config.py::SeparationConfig` (line 61): `model: str  = "htdemucs"`
- Symptom: SeparationConfig.model/device/preview_seconds (and env REMIXMATE_SEPARATION_MODEL shown as an example) never reach Demucs; the model comes from the request body default 'htdemucs' and device from gpu.get_device/REMIXMATE_DEVICE.
- Impact: Changing separation.model in config.yaml has no effect; DEVELOPING.md says it does.
- Fix sketch: Default StemSplitRequest.model and separate_song_stems from cfg.separation.model/device, or delete the block and the doc line.

### OPS-15 _venv_python docstring order differs from code

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/stems.py::_venv_python` (line 75): `candidates.insert(0, cand)`
- Symptom: Docstring says `remix-env` first, then the demucs script's interpreter; code inserts the demucs shebang interpreter at index 0, so it wins when present.
- Impact: Operators reasoning about which interpreter runs Demucs (`remix-env` vs .venv) get the order wrong.
- Fix sketch: Fix the docstring, and consider preferring sys.executable when it has demucs so `remix-env` can be retired.

### OPS-16 pages.yml is still dispatchable but its origin is blocked by CORS

- Severity: low. Evidence: unproven.
- Where: `.github/workflows/pages.yml::on.workflow_dispatch` (line 25): `workflow_dispatch:`
- Symptom: The workflow builds a GitHub Pages frontend that talks to a local API, but cors.py's _LEGACY_BLOCK strips chunduri-aditya.github.io from the allowed origins.
- Impact: A manual dispatch deploys a frontend that cannot call the API; the path is dead.
- Fix sketch: Delete pages.yml or move it to archive/.

### OPS-17 eslint config imports an undeclared package

- Severity: low. Evidence: unproven.
- Where: `frontend/eslint.config.js` (line 3): `import js from '@eslint/js'`
- Symptom: eslint.config.js imports @eslint/js, which is absent from package.json and present only as a transitive dependency of eslint
- Impact: Lint breaks if the hoisting changes
- Fix sketch: Add @eslint/js to devDependencies

### OPS-18 Retention sweep treats every top level outputs dir as a user

- Severity: low. Evidence: unproven.
- Where: `scripts/core/mix_retention.py::sweep_all_users` (line 184): `for user_root in sorted(p for p in root.iterdir() if p.is_dir()):`
- Symptom: sweep_all_users iterates every directory under OUTPUTS_DIR as outputs/<uid>/. Legacy session dirs from before tenancy (outputs/dj_* holding one file) are never swept and are unreachable, since stream_output anchors on outputs/<uid>/.
- Impact: Orphaned renders accumulate; a legacy session dir that contained subdirectories would be swept as if it were a user.
- Fix sketch: Migrate legacy dirs with scripts/migrate_uids.py, or skip names matching the session id pattern (dj_, chain_) in sweep_all_users.

## Docs and comments

### DOC-01 POST /analyze is not operator gated despite SECURITY.md

- Severity: medium. Evidence: unproven.
- Where: `scripts/api/routers/analysis.py::analyze_song` (line 327): `@router.post("/analyze", response_model=JobResponse, status_code=202, tags=["analysis"])`
- Symptom: The Operator row in SECURITY.md lists analysis jobs over the shared library as require_operator, but analyze_song (and generative tokenize_stems) have no require_operator dependency. Both write into the shared song folder (meta.json, tokens/).
- Also: Any signed in user can re key a catalog song for everyone (for example key_profile=ks) while /library/analyze-missing requires an operator
- Also: analyze_song is only user gated, but task_analyze runs run_song_analysis, which writes library/<song>/meta.json and analysis.json that every user shares. SECURITY.md's operator row lists analysis jobs over the shared library.
- Impact: Any signed in user can rewrite shared analysis for catalog songs from the Library Atlas Analyze button, while `analyze-missing` on the same page is operator only. Either the doc or the gate is wrong.
- Fix sketch: Decide the intended gate: add Depends(require_operator) to /analyze and /ai/tokenize, or correct SECURITY.md.

### DOC-02 docs/DJ_THEORY.md contradicts the code on bass swap timing, stretch limits, clash length and A/B pairs

- Severity: low. Evidence: confirmed by reading.
- Where: `docs/DJ_THEORY.md::Bass Swap Timing` (line 356): `- Cut A's bass EQ: starts at bar T/2, reaches -∞ dB at bar T*0.75`
- Where: `docs/DJ_THEORY.md::Time-Stretching Limits` (line 157): `Phase vocoder time-stretching introduces artifacts that become musically unacceptable beyond ±3–5 BPM of original tempo.`
- Where: `docs/DJ_THEORY.md::Camelot Move Selection Algorithm` (line 350): `- score < 0.4 → 4-8 bars (quick cut, minimize harmonic exposure)`
- Where: `docs/DJ_THEORY.md::The Seven Camelot Moves (Scored)` (line 86):

  ```
  | **Parallel mode** | `num_b == num_a`, `let_b != let_a` | 0.85 | Same root, different mode — emotional shift without clash |
  ```

- Symptom: Doc puts the swap at bar T*0.75; plan_transition sets bass_swap_bar = transition_bars // 2
- Also: Doc says phase vocoder stretches beyond 3 to 5 BPM are unacceptable; code clamps only to ratio 0.5 to 2.0
- Also: Doc suggests 4 to 8 bars for low scores; API floor is 8 and plan_transition rounds up to multiples of 8
- Also: 'The System' calls same number A/B relative keys sharing notes; 'The Seven Camelot Moves' table calls them 'Parallel mode, same root'
- Impact: Doc drift.
- Fix sketch: State the implemented midpoint swap or mark the section as theory.

### DOC-03 DEVELOPING.md drifts from the code and CI in four places

- Severity: low. Evidence: confirmed by reading.
- Where: `DEVELOPING.md::Key design decisions` (line 128): `**Dynamic EQ.** A low shelf filter pulls Song A's bass down as Song B's rises, preventing bass`
- Where: `DEVELOPING.md::Commands` (line 78): `npm run lint          # eslint, --max-warnings 0`
- Where: `DEVELOPING.md::Commands` (line 80): `npm test              # vitest`
- Where: `DEVELOPING.md::Harmonic compatibility` (line 156): `**Harmonic compatibility.** Two independent systems exist and serve different purposes: the`
- Where: `DEVELOPING.md::## Stem processing (Demucs)` (line 258):

  ```
  - Bridge integration: `scripts/remixmate_cli.py` `generate_stems_and_params(base_song,
  ```

- Symptom: 'Dynamic EQ' says a low shelf pulls A's bass down; dj_engine.py has no low shelf (render uses a 3 band fade and a high pass after the swap)
- Also: npm run lint runs 'eslint src' while CI runs 'npx eslint .'; npm test starts vitest in watch mode.
- Also: npm test runs vitest in watch mode (never exits in an agent shell) and npm run lint lints src only while CI runs npx eslint . --max-warnings 0
- Also: Agents reading DEVELOPING.md miss psychoacoustic_consonance (drives the current plan), _camelot_move_analysis, recommend._camelot_score, compatibility_score's own table, and duplicate Camelot tables in music_intelligence, track_metadata, spotify, musical_analysis, metrics and CamelotWheel.tsx
- Also: The Stem processing section says stem paths and mix params come from remixmate_cli.generate_stems_and_params (no callers, .wav only) and that separation respects generative.max_vram_gb (stems.py never reads it); it also names separation.model/device as the Demucs config.
- Impact: Misleads agents looking for the bass handling on the plain render path.
- Fix sketch: Describe _apply_dynamic_eq_fade plus the post swap high pass.

### DOC-04 Comments and docstrings that contradict the code

- Severity: low. Evidence: confirmed by reading.
- Where: `frontend/src/lib/api.ts::normalizeJob` (line 103):

  ```
  // (lowercase status, `job_type`, progress 0–1, epoch timestamps) while SSE
  ```

- Where: `scripts/api/auth.py::auth_context_middleware` (line 222):

  ```
  as ``"local"``. Today the only open routes are ``/health*`` (queries jobs
  ```

- Where: `scripts/api/routers/library.py::calibrate_stem_lufs` (line 520): `these corpus-derived targets instead of the flat -20 LUFS fallback.`
- Where: `scripts/api/routers/remix.py::dj_remix_preview` (line 159): `- tempo_ratio      — bpm_b / bpm_a (1.0 = perfect match)`
- Where: `scripts/core/config.py::DJConfig` (line 120): `# both songs' stems are on disk. Reuses the same 0.35 "harmonic clash"`
- Where: `scripts/core/dj_analysis.py::plan_transition` (line 820): `section lengths.  Always rounded to the nearest phrase_size.`
- Where: `scripts/core/instrument_lab.py` (line 9): `3. Applies per-stem gain normalisation + mastering`
- Where: `scripts/core/setlist_planner.py::SetlistPlanner` (line 427): `For larger libraries, a beam search fallback is available.`
- Where: `scripts/core/spectral_matching.py::trajectory_similarity` (line 148): `Frames are compared pairwise (both windows span the same musical time —`
- Where: `scripts/remixmate_cli.py::build_parser` (line 243): `bridge_parser.add_argument('--online', action='store_true', help='Fetch missing songs via YouTube Music')`
- Where: `tests/conftest.py` (line 67): `# "data/song_embeddings.json"), consumed by track_metadata.py / database.py.`
- Symptom: job_to_response now uses _norm_status (uppercase, COMPLETED); the comment and the useJobPoller test fixture still describe lowercase REST status
- Also: It says the only open routes are /health* and root, but main.py also mounts the legal router open, and system.router includes /features.
- Also: calibrate_stem_lufs docstring says remixes otherwise use a flat -20 LUFS fallback; load_stem_targets falls back to per stem _STEM_FALLBACK_TARGETS.
- Also: Docs say tempo_ratio is bpm_b / bpm_a.
- Also: Misleading comment on dj.transition_confidence_threshold
- Also: Comment and value disagree
- Also: DJConfig comment says the gate reuses the 0.35 harmonic clash cutoff; the default and config.yaml value is 0.30.
- Also: Docstring says transition_bars is rounded to the nearest phrase_size; code rounds up (12 becomes 16)
- Also: Module docstring lists per stem gain normalisation plus mastering and optional key aware pitch shift; render_combo only sums and peak normalizes to 0.891.
- Also: Agents may look for or rely on a beam search path.
- Also: Docstring says both windows span the same musical time because the engines tempo match them; matching runs on unstretched B before any render, comparing equal seconds, not equal bars
- Also: remixmate bridge --online says it fetches missing songs via YouTube Music; PythonBridge.create_remix discards the flag.
- Also: cfg.database.path and embeddings_path are read by no code. scripts/core/database.py::RemixMateDatabase defaults to a working directory relative models/remixmate.db.
- Impact: Misleads anyone changing the job contract
- Fix sketch: Update the comment; add a direct normalizeJob test with the real REST and SSE shapes

### DOC-05 _resolve_engine docstring calls classic the default

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/api/task_modules/remix.py::_resolve_engine` (line 55): `"classic" (default) — the vendored AI_REMIXMATE_BEST_MIXING_ALGO path`
- Symptom: Docstring says "classic" (default) while config.yaml sets dj.mixing_engine: current; classic is only the dataclass and error fallback.
- Also: _resolve_engine docstring calls classic the default and config.py says the threshold reuses 0.35, while config.yaml runs current and the threshold is 0.30.
- Impact: Misleads readers about which engine renders by default; a config load failure or typo silently switches to classic.
- Fix sketch: State that config.yaml selects current and classic is the fallback; consider falling back to current and logging a warning on unknown values.

### DOC-06 Classic engine documented as byte faithful after three edits

- Severity: low. Evidence: confirmed by reading.
- Where: `scripts/core/classic_engine/__init__.py` (line 5): `(analysis → transition planning → rendering → mastering), kept byte-for-byte`
- Symptom: __init__ docstring and DEVELOPING.md say the package is a byte for byte frozen copy; git log shows 530ba73 and e7a053b (lint edits) and e251f22 (tempo ratio and render_chain pre stretch flipped).
- Impact: Agents and users believe classic reproduces the reference renders; it does not since e251f22.
- Fix sketch: Reword to name the commits that diverged, or restore the snapshot and document the tempo defect instead.

### DOC-07 app/README.md is stale about the source of truth and status

- Severity: low. Evidence: unproven.
- Where: `app/README.md::Status` (line 35): `Phase 0 (test storage isolation) is done and verified.`
- Symptom: It says DEPLOYMENT_PLAN.md is the source of truth and that Phase 0 is done, while app/STATUS.md is the live ledger through Phase 9.
- Impact: Misleads readers about hosted progress and which doc to trust.
- Fix sketch: Point Status at app/STATUS.md and drop the phase sentence.

### DOC-08 config.yaml cors_origins '*' comment contradicts cors.py

- Severity: low. Evidence: unproven.
- Where: `config.yaml::api.cors_origins` (line 121): `- "*"                      # tighten this in production`
- Symptom: config.yaml lists '*' with 'tighten this in production', but resolve_cors_origins drops '*' and falls back to the Vite localhost origins; production uses REMIXMATE_FRONTEND_URL or REMIXMATE_CORS_ORIGINS.
- Impact: Misleading; an operator may edit config.yaml expecting it to control production CORS.
- Fix sketch: Replace with an empty list and a comment pointing at REMIXMATE_FRONTEND_URL and REMIXMATE_CORS_ORIGINS.
