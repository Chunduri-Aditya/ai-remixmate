# UI map

*Where every page, hook, store and client call of the React frontend lives, and which backend handler each call reaches. As of 72e2368. [tests/test_agent_docs.py](../../tests/test_agent_docs.py) enforces that every cited path and `path::symbol` resolves; whether a sentence is still true is a review job.*

## How to use this map

- Find your page in the route table under Routing, then read its subsection under Pages.
- Follow the page to its endpoint in "Page to endpoint to router", then open the named handler in `scripts/api/routers/`.
- Check the backlog ids named next to what you are about to change before changing it; problems live in [BACKLOG.md](BACKLOG.md), not here.

Commands, recipes and design decisions live in `DEVELOPING.md`; the hosted ledger in `app/STATUS.md`; music theory in `docs/DJ_THEORY.md`. This file links to them instead of restating them.

## Routing

Provider stack, outermost first (`frontend/src/main.tsx`, then `frontend/src/App.tsx::App`):

`StrictMode` > `BrowserRouter` (basename `import.meta.env.BASE_URL`) > `RootErrorBoundary` > `ToastProvider` (app level) > `QueryLayer` (owns the `QueryClient`) > `AuthLayer` > `AppRoutes`. Inside the signed in branch `frontend/src/shell/AppShell.tsx::AppShell` mounts a second `ToastProvider` (see UI-31). `ReactQueryDevtools` renders only when `import.meta.env.DEV`.

Order matters: `frontend/src/shell/AuthProvider.tsx::AuthProvider` calls `useQueryClient()` to clear the cache on sign out, so it must stay inside `QueryLayer`; `QueryLayer` calls `useToast()`, so it must stay inside the app level `ToastProvider`.

Routes live in two files, with nested `<Routes>`:

- `frontend/src/App.tsx::AppRoutes`: the public legal routes, then a catch all wrapped in `frontend/src/shell/RequireAuth.tsx::RequireAuth`, which holds an inner `<Routes>` with `/widget` (no shell) and everything else going to `AppShell`.
- `frontend/src/shell/AppShell.tsx::AppShell`: the canvas `<Routes>` with relative paths (`mission-control`, not `/mission-control`).

Every page is `React.lazy` over a default export, so a new page file must `export default function`.

| Path | Component | Gate | Purpose |
|---|---|---|---|
| `/legal`, `/legal/:page` | `frontend/src/pages/LegalPage.tsx::LegalPage` | public, outside `RequireAuth` | Terms, Privacy, DMCA with the report form, Acceptable Use |
| login screen | `frontend/src/shell/LoginPage.tsx::LoginPage` | rendered by `RequireAuth` when auth is configured and there is no session | sign in |
| `/widget` | `frontend/src/pages/Widget.tsx::Widget` | `RequireAuth`, no shell | compact now playing and recommendation panel |
| `/` | redirect to `/mission-control` | `RequireAuth` | |
| `/mission-control` | `frontend/src/pages/MissionControl.tsx::MissionControl` | `RequireAuth` | health, library stats, jobs, quick actions; hydrates the job store |
| `/library-atlas` | `frontend/src/pages/LibraryAtlas.tsx::LibraryAtlas` | `RequireAuth` | song table and per song actions |
| `/mix-deck` | `frontend/src/pages/MixDeck.tsx::MixDeck` | `RequireAuth` | two decks, compatibility, preview and full render |
| `/set-builder` | `frontend/src/pages/SetBuilder.tsx::SetBuilder` | `RequireAuth` | sequence a set, chain render |
| `/signal-search` | `frontend/src/pages/SignalSearch.tsx::SignalSearch` | `RequireAuth` | similarity and text search |
| `/ai-lab` | `frontend/src/pages/AILab.tsx::AILab` | `RequireAuth` and `frontend/src/shell/AppShell.tsx::RequireGenerative` | generative tools; redirects to `/mission-control` unless `features.generative` is true |
| `/mix-vault` | `frontend/src/pages/MixVault.tsx::MixVault` | `RequireAuth` | finished mixes, downloads, credits |
| `/operations` | `frontend/src/pages/Operations.tsx::Operations` | `RequireAuth`; operator panels read `features.operator` | processing status, storage, maintenance |
| `/account` | `frontend/src/pages/AccountPage.tsx::AccountPage` | `RequireAuth` | export and delete account |
| any other signed in path | redirect to `/mission-control` | `RequireAuth` | |

Every canvas route except the redirects is wrapped as `PageErrorBoundary` (with `pageName`) > `Suspense` with `PageFallback` > page. The legal and widget routes use a null `Suspense` fallback and no page boundary.

Base path: `frontend/vite.config.ts` sets `base` from `VITE_BASE` (default root); `frontend/src/main.tsx` passes `import.meta.env.BASE_URL` as the router basename; `frontend/src/shell/LoginPage.tsx::LEGAL_BASE` reuses it for plain anchor legal links. They are plain anchors because `RequireAuth` is also rendered without a Router in `frontend/src/__tests__/auth.test.tsx`.

## Shell layout

`AppShell` is a CSS grid (`frontend/src/shell/AppShell.css`, class `.app-shell`): columns `var(--shell-rail-width) 1fr`, plus `var(--shell-inspector-width)` when `.app-shell--inspector-open`. Its height subtracts `var(--player-bar-height)` so the fixed player bar does not cover the canvas. The widths live in `frontend/src/styles/tokens.css`.

Mounted once for every shell page, so they survive navigation: `useSSE()`, `useJobPoller(!sseConnected)`, `useKeyboardShortcuts(...)`, and `frontend/src/shell/AppShell.tsx::JobToastWatcher` (runs `useJobToasts`). None of them run on `/widget` or `/legal`.

| Zone | Symbol | Notes |
|---|---|---|
| Left rail | `frontend/src/shell/LeftRail.tsx::LeftRail` | Items from `frontend/src/shell/LeftRail.tsx::NAV_ITEMS`; the AI Lab item is filtered out unless `features.generative` is true. Active state comes from `location.pathname`. Footer: legal buttons (leave the shell), Account, Sign out (only when auth is configured and a session exists), `frontend/src/shell/LeftRail.tsx::ConnectionDot` (labels inverted, see UI-18). The Widget item opens `/widget` in the same tab, see UI-40. |
| Canvas | `main` element, class `app-shell__canvas` | the nested `<Routes>` |
| Right inspector | `frontend/src/shell/RightInspector.tsx::RightInspector` | Rendered only while `inspectorOpen`; once closed it cannot be reopened, see UI-06. Tabs `jobs`, `activity`, `system`. The jobs tab lists `frontend/src/stores/appStore.ts::selectRecentJobs`. Buttons: Clear (drops terminal jobs), Restart session (moves `sessionStartedAt` to now). |
| Player bar | `frontend/src/components/PlayerBar.tsx::PlayerBar` | fixed to the bottom; state in `frontend/src/stores/playerStore.ts::usePlayerStore`; audio through `frontend/src/lib/playerEngine.ts::playerEngine` (Web Audio EQ). Shell only. |
| Modal | `frontend/src/components/ShortcutsModal.tsx::ShortcutsModal` | opened by `?`, closed by backdrop click, Close, or Esc |

`frontend/src/shell/RightInspector.tsx::JobCard` behaviour to preserve:

- A card click routes by `frontend/src/shell/RightInspector.tsx::jobDestination` on a substring of the job type: remix or chain to `/mix-vault`, download or playlist to `/operations`, style, inpaint or tokeniz to `/ai-lab`, anything else to `/library-atlas`. For RUNNING or PENDING jobs it asks `window.confirm` first. The style, inpaint and tokeniz branches match only the optimistic label AI Lab upserts (`type: mode`); the server files every generative and stem job as `analyze` with the kind in `meta.type`, so after the first SSE frame those cards route to `/library-atlas`. Branch on `meta.type` (SSE only, UI-10) if a card must return to AI Lab.
- Cancel calls `jobsApi.cancel` and removes the card only after the DELETE succeeds; a failure goes to the activity log.
- Retry (FAILED `dj_remix`) and thumbs (COMPLETED `dj_remix`) read `job.meta`, which only SSE frames carry (UI-10); Retry swallows request failures (UI-32).
- `SystemTab` shows uptime only when `uptimeSeconds` is positive and Host, GPU and Tier only when `machineProfile` is set; the heartbeat sends neither (UI-05, UI-34).

## Auth in the browser

Two modes, chosen at build time by `frontend/src/lib/supabase.ts::authConfigured` (true only when both `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` are non empty after trim). The backend mode is chosen separately at runtime by `REMIXMATE_AUTH_ENABLED` (`scripts/api/auth.py::auth_enabled`). Nothing in the UI compares the two; a mismatch fails silently, see UI-19.

| | Local mode (`authConfigured` false) | Hosted mode (`authConfigured` true) |
|---|---|---|
| Supabase client | `supabase` is `null` | `createClient` with `persistSession`, `autoRefreshToken` and `detectSessionInUrl` on; session in localStorage |
| `AuthLayer` | renders children with `frontend/src/shell/authContext.ts::NOOP_AUTH` (or whatever a test harness provides) | mounts `AuthProvider` |
| `RequireAuth` | passes through | splash while loading, `LoginPage` with no session, children otherwise |
| Request headers | no `Authorization` | `frontend/src/lib/api.ts::request` awaits `frontend/src/lib/supabase.ts::getAccessToken` per call and sends a Bearer header |
| Cookie | none | `frontend/src/lib/supabase.ts::syncSseCookie` writes `sb-access-token` |
| Sign out on 401 | no handler registered | handler calls `supabase.auth.signOut()` |

`AuthProvider` effect: `getSession()` then `setSession` and `syncSseCookie`; `onAuthStateChange` does the same on every event (token refresh included) and calls `queryClient.clear()` on `SIGNED_OUT`; it registers the sign out handler with `frontend/src/lib/api.ts::setUnauthorizedHandler`; cleanup unsubscribes and unregisters.

The cookie: name `sb-access-token` (must match `scripts/api/auth.py::_DEFAULT_COOKIE_NAME`, overridable server side by `SUPABASE_COOKIE_NAME`), the raw access token, `Path=/`, `SameSite=Lax`, `Secure` only on https, `Domain` set from `VITE_COOKIE_DOMAIN` when that var is set (on both write and clear), no `Max-Age`, not httpOnly. It exists for requests that cannot carry a header: the `EventSource` (opened with credentials) and media elements.

What the backend accepts (`scripts/api/auth.py::_extract_token`): the Bearer header first; the cookie only for methods in `scripts/api/auth.py::_COOKIE_METHODS` (GET, HEAD). `scripts/api/auth.py::auth_context_middleware` never returns 401; `scripts/api/auth.py::require_user` returns 401 on gated routes when auth is on and no valid token arrives; `scripts/api/auth.py::require_operator` equals `scripts/api/auth.py::require_admin` when auth is on and passes everyone as `local` when auth is off.

401 path: `request` calls the registered handler on any 401 before throwing `frontend/src/lib/api.ts::ApiError`, and `frontend/src/App.tsx::isSilentError` keeps 401 out of toasts. Only calls through `request` trigger it; the SSE stream, media elements and pages that call `fetch` directly do not.

Hosted deployment wiring (Cloudflare Pages, the cookie domain) is in `DEVELOPING.md` section "Gotchas" and `app/deploy/PROVISION.md`.

## Feature flags

`GET /features` is `scripts/api/routers/system.py::features`, on the system router that `scripts/api/main.py` mounts without `require_user`. It reads every flag per request. Client: `frontend/src/lib/api.ts::featuresApi`, typed `frontend/src/types/index.ts::Features`, cached by `frontend/src/hooks/useFeatures.ts::useFeatures` (query key `features`, `staleTime: Infinity`; refetched only after `queryClient.clear()` on sign out or a reload). While pending, every caller treats a flag as off.

| Key | Server source | UI readers |
|---|---|---|
| `auth` | `scripts/api/auth.py::auth_enabled` | none |
| `stem_split` | `REMIXMATE_STEM_SPLIT_ENABLED` via `scripts/api/routers/stems.py::_stem_split_enabled` | Library Atlas Split buttons |
| `generative` | `REMIXMATE_GENERATIVE_ENABLED` via `scripts/api/routers/generative.py::_generative_enabled` | LeftRail item, `RequireGenerative`, `useKeyboardShortcuts`, Mission Control quick action |
| `license_enforce` | `scripts/core/license.py::license_enforce_enabled` | none |
| `spectral_learning` | `scripts/core/spectral_matching.py::_learning_enabled` | none |
| `operator` | `scripts/api/routers/system.py::_is_operator` (would `require_operator` admit this caller) | Operations operator panels |

`operator` is per caller, so it depends on the token the features request carried; the infinite cache is safe only because sign out clears the whole cache. Flags hide controls; the server gates (503 for a disabled feature, 401 or 403 for operator routes) remain the enforcement.

## Errors and toasts

| Layer | Symbol | Behaviour |
|---|---|---|
| Whole app | `frontend/src/components/RootErrorBoundary.tsx::RootErrorBoundary` | any render error replaces the tree with a Reload button |
| Per page | `frontend/src/components/PageErrorBoundary.tsx::PageErrorBoundary` | "failed to load" with the error message; Try again resets and rerenders |
| Query and mutation errors | `frontend/src/App.tsx::createQueryClient` | toast with `error.message` unless `isSilentError` (ApiError 401 or 503); the text is the raw ApiError string, see UI-22 |
| Retry | `createQueryClient` | queries: no retry for ApiError 4xx; anything else (5xx, timeout status 0, network error, JSON parse error) retries twice. Mutations do not retry. `staleTime` 15 s, no refetch on window focus. |
| Request wrapper | `request`, `ApiError` | timeout from `frontend/src/lib/api.ts::DEFAULT_TIMEOUT_MS` becomes `ApiError` status 0; a non 2xx becomes `ApiError` with status, path and body text; a network failure or a JSON parse error is rethrown unchanged (a plain `Error`, not `ApiError`), which is why `createQueryClient` retries it |
| Job toasts | `frontend/src/hooks/useJobToasts.ts::useJobToasts` | toast when a job changes to COMPLETED or FAILED, including historic jobs on first sight (UI-11) |

`frontend/src/components/Toast.tsx::ToastProvider` is mounted twice (UI-31): `useToast()` returns the nearest one, so the app level provider receives only `QueryLayer` errors and the AppShell provider receives only job toasts. No page calls `useToast`.

Errors thrown from event handlers or direct `api.*` calls outside React Query reach neither the boundaries nor the toasts; each caller handles its own. A 503 is always silent, so a page that calls a disabled feature must render its own unavailable state.

## Keyboard shortcuts

`frontend/src/hooks/useKeyboardShortcuts.ts::useKeyboardShortcuts` is one window keydown listener mounted by AppShell. It ignores events whose target is an input, textarea, select or contenteditable element.

| Keys | Action |
|---|---|
| Cmd or Ctrl K | go to `/library-atlas` and focus the element marked `data-library-search-input` |
| Cmd or Ctrl plus a digit | index into `frontend/src/hooks/useKeyboardShortcuts.ts::NUMBER_NAV`; the AI Lab slot is ignored when `generative` is off, later slots keep their numbers |
| `g` then a letter | `frontend/src/hooks/useKeyboardShortcuts.ts::NAV_MAP` (case sensitive); the AI Lab letter checks the flag |
| Space | only on `/mix-deck`: dispatches the event named by `MIX_DECK_TOGGLE_EVENT`, duplicated as a const in `frontend/src/components/WaveformDeck.tsx` |
| `?` | open `ShortcutsModal` |
| Esc | close the modal and, if open, the inspector |

The help list is a separate static array, `frontend/src/components/ShortcutsModal.tsx::SHORTCUTS`, which ignores the generative flag (UI-24). Account and Widget have no shortcut. The digit handler accepts only keys matching `/^[1-8]$/` before it indexes `NUMBER_NAV`, and every slot in that range is taken, so a ninth page needs both a new array entry and the regex widened; an entry alone is a dead key with no error.

## Data layer

### API client

Entry point `frontend/src/lib/api.ts`. Every typed call goes through `frontend/src/lib/api.ts::request` via the `get`, `post`, `del` and `patch` helpers, grouped into objects: `libraryApi`, `storageApi`, `audioApi`, `stemsApi`, `analysisApi`, `remixApi`, `jobsApi`, `cratesApi`, `favoritesApi`, `setlistApi`, `feedbackApi`, `aiApi`, `healthApi`, `featuresApi`.

| Concern | Behaviour | Where |
|---|---|---|
| Base URL | `VITE_API_BASE`, else `/api`. In dev the Vite proxy strips `/api` and forwards to the API on port 8000 | `frontend/src/lib/api.ts::BASE` (not exported), `frontend/vite.config.ts` |
| SSE URL | `VITE_EVENTS_URL`, else the events stream under `BASE` | `frontend/src/lib/api.ts::EVENTS_URL` |
| Auth | Bearer header from `getAccessToken` on every call; `credentials: 'include'` always | `request` |
| Timeout | AbortController at `DEFAULT_TIMEOUT_MS` (30 s), surfaced as `ApiError` status 0 | `request` |
| Errors | a non 2xx throws `ApiError`; a 401 first calls the unauthorized handler | `ApiError` |
| Success | always parses JSON; every DELETE route the UI calls returns a JSON body today, so a new route returning 204 would throw here | `request` |
| Job shape | REST and SSE job payloads both pass through `frontend/src/lib/api.ts::normalizeJob` (its comment is stale, see DOC-04) | see Realtime |

Paths that bypass `request` (no timeout, no 401 handler, no `ApiError`):

- `frontend/src/pages/AccountPage.tsx::authedFetch` for `GET /account/export` and `DELETE /account` (UI-38).
- `frontend/src/pages/LegalPage.tsx` uses a bare `fetch` for the public legal routes, without credentials.
- URL builders the browser fetches itself: `audioApi.streamUrl`, `audioApi.exportCuesUrl`, `stemsApi.stemUrl`, and the output stream URLs built on `BASE` in Mix Deck, Mix Vault and AI Lab. These carry no Bearer header; hosted, they authenticate only through the cookie, which the backend accepts on GET and HEAD. Plain audio elements and anchors send the cookie cross origin (same site, Lax). WaveSurfer (`frontend/src/components/WaveformDeck.tsx::WaveformDeck`, `frontend/src/pages/MixVault.tsx::WaveformPlayer`) and the global player (`frontend/src/lib/playerEngine.ts::PlayerEngine`) send no cookie cross origin, see UI-03 and UI-04; both are unproven against a live hosted build.

`libraryApi.list` asks for `per_page=5000` and unwraps `.songs`; see UI-37 for truncation and cost. `scripts/api/routers/library.py::list_library` serves it.

### Realtime

- `frontend/src/hooks/useSSE.ts::useSSE`: an `EventSource` on `EVENTS_URL` with credentials, reconnect backoff doubling from 1 s up to 30 s; `onopen` sets `sseConnected` and `apiHealth`. Every frame is `{type, data, ts}` built by `scripts/api/routers/events.py::_frame`.
- `frontend/src/hooks/useJobPoller.ts::useJobPoller`: `GET /jobs` at the backend default `limit` (20 in `scripts/api/routers/jobs.py::list_jobs`) every 4 s while SSE is down; merges by `job_id` instead of replacing the store.
- `frontend/src/hooks/useJobTimer.ts::useJobTimer`: elapsed time from `created_at` (queue time counts) and a linear ETA once progress passes 2; the backend `eta_sec` is dropped by `normalizeJob`.
- `useJobToasts` and `useFeatures`: above.

Backend side: job events go through `scripts/api/jobs.py::_emit`, serialized by `scripts/api/jobs.py::_job_to_response_dict`; `scripts/api/routers/events.py::SSEBroadcaster.broadcast` delivers job frames only to the owner's queues (`scripts/api/routers/events.py::_payload_owner` reads `meta.user_id`). REST uses `scripts/api/jobs.py::job_to_response` and `scripts/api/schemas.py::JobResponse`.

| Field | SSE | REST | After `normalizeJob` |
|---|---|---|---|
| status | uppercase (`scripts/api/jobs.py::_norm_status`) | uppercase (same helper) | uppercase |
| type | `type`, lowercase | `job_type` | `type` |
| progress | 0 to 100 | 0 to 1 | 0 to 100; scaled only when the raw object has `job_type` |
| timestamps | ISO strings | epoch seconds | ISO |
| meta | full dict (`user_id`, `song_a`, `song_b`, `songs`, `preview`, `request`) | absent | present only when fed by SSE |
| eta_sec | absent | present | dropped |

Invariant: anything that reads `job.meta` (Inspector retry and thumbs, Mix Vault titles and the preview filter) works only for jobs that arrived over SSE and were not later overwritten by a REST copy (UI-10). Three paths overwrite: the poller merge, Mix Vault upserting its REST query, and Mission Control's `setJobs` (UI-14).

Events the frontend handles but the backend never sends: `library_changed` and `system_status` beyond the connect ack in `scripts/api/routers/events.py::_sse_generator` (UI-34). The heartbeat from `scripts/api/routers/events.py::SSEBroadcaster.send_heartbeat` does not carry what `frontend/src/types/index.ts::HeartbeatData` expects (UI-05). Nothing invalidates React Query library caches when a job finishes; pages refetch on their own timers or on explicit `invalidateQueries`.

### Stores

`frontend/src/stores/appStore.ts::useAppStore` (Zustand) holds client state; server data lives in React Query.

- `jobs`, keyed by `job_id`. Writers: `upsertJob` (SSE, Mix Vault, pages that launch jobs; it replaces the whole job object), the poller's direct `setState` merge in `frontend/src/hooks/useJobPoller.ts::useJobPoller` (bypasses `upsertJob`, so no `completionLog` entry), `setJobs` (replaces the whole map; only Mission Control calls it, see UI-14; also bypasses `upsertJob`), `removeJob` (Inspector cancel), `clearJobs` and `restartSession` (keep PENDING and RUNNING).
- `completionLog`: only `upsertJob` appends a timestamp, when it first sees a job as COMPLETED, so Mix Vault's REST load counts historic mixes as fresh completions (UI-11); the poller merge and Mission Control's `setJobs` do not touch it, though both still trigger `useJobToasts`.
- `sessionStartedAt`: hides terminal jobs created before page load in `selectRecentJobs`.
- `activityLog` (capped), `apiHealth`, `sseConnected`, `uptimeSeconds`, `machineProfile`.
- `inspectorOpen`, `inspectorTab`: see UI-06.
- `activeNav` is written but never read, and `logCompletion` has no caller (UI-21).

`frontend/src/stores/playerStore.ts::usePlayerStore` drives the singleton `playerEngine`: queue, index, shuffle, repeat, volume, mute, EQ (clamped). It logs playback telemetry (`play_start`, `skip`, `complete`, `replay`) through `feedbackApi.logEvent` with a swallowed catch, only when the track has `songA` or `songName`. The engine builds its Web Audio graph lazily on first play, so volume and EQ set earlier are lost (UI-25).

## Pages

Page list and one line purposes: `DEVELOPING.md` section "Frontend pages". Each subsection adds the symbol, calls, states and backlog ids.

### Mission Control (`/mission-control`)

- Component `MissionControl`; helpers `frontend/src/pages/MissionControl.tsx::HeartbeatWidget`, `StatCard`, `JobRow`, `QuickAction`, `Sparkline`.
- User can: see API health, library size, active job count, a completion sparkline, recent jobs; jump to Library Atlas, Mix Deck, Set Builder, Operations, and AI Lab when `generative` is on.
- Calls: `GET /health/live` every 10 s, `GET /library` (query key `library-stats`) every 30 s, `GET /jobs` once (query key `jobs-init`).
- States: skeletons while loading; "No jobs yet"; no error state of its own (errors reach the global toast).
- Backlog: uptime reads NaN (UI-05); the mount replaces the job store (UI-14); the sparkline counts historic jobs only after Mix Vault has loaded them this session (UI-11).

### Library Atlas (`/library-atlas`)

- Component `LibraryAtlas`; children `frontend/src/pages/LibraryAtlas.tsx::SongRow`, `StemExpansionRow` (renders `frontend/src/components/StemsPlayerRow.tsx::StemsPlayerRow`), `SongDrawer`, `CratesSection`, `BrokenSongsPanel`.
- User can: search, filter (favorites, stems, analysed, high energy), sort, favorite, play (queues the filtered list into the global player), analyse, split stems, export rekordbox cues, delete, multi select for bulk analyse or split, scan for broken songs and bulk delete them, open a drawer with recommendations, similar tracks and crate membership, load a song into Mix Deck.
- Calls: `GET /library` (query key `library-atlas`, shared with Mix Deck, Set Builder, Signal Search and AI Lab), favorites routes, `POST /analyze`, `POST /stems/split`, `DELETE /library/{name}`, `POST /library/analyze-missing`, `POST /library/scan-broken`, `POST /library/delete-batch`; in the drawer `GET /recommend/{name}` and `GET /library/similar/{name}`; crate routes; URL builders for audio, stems and cue export.
- States: loading spinner; an empty library message naming the ingest tool; "No tracks match your filters" with reset. Rows are virtualized with `useVirtualizer` except while a stems row is expanded, which renders every row.
- The broken scan result is read from the SSE fed job store (`job.result.broken`), not polled.
- Backlog: crate membership never loads, and the test mock hides it (UI-02); operator actions shown to everyone with failures swallowed (UI-13); the drawer's similar call writes on a GET (MX-32).

### Mix Deck (`/mix-deck`)

- Component `MixDeck`; children `DeckCard` (renders `WaveformDeck`), `CompatPanel`, `JobStatusBadge`, `frontend/src/pages/MixDeck.tsx::RemixResultCard`, plus `frontend/src/components/CamelotWheel.tsx::CamelotWheel`, `frontend/src/components/TransitionTimeline.tsx::TransitionTimeline`, `frontend/src/components/RemixControls.tsx::RemixControls`.
- Entry: `song_a` and `song_b` query params seed the decks once, then are cleared.
- User can: pick two songs, check compatibility, see strict matches for Deck A and load one into Deck B, choose preset, effect and bridge beat, set crossfade bars, render a preview or a full remix, rate the result, play it in the global player.
- Calls: `GET /library/{name}` per deck, `GET /library/similar/{name}`, `GET /compatibility/matches/{name}` (any error becomes null, which hides the panel), `POST /compatibility`, `POST /dj-remix/preview`, `POST /dj-remix`, `POST /dj-remix/feedback`.
- Request bodies: `transition_bars` comes from the "Crossfade bars" slider only, clamped to 8 to 64. Four defaults disagree: the slider state starts at 32 (`transitionDuration` in `MixDeck`), `frontend/src/components/remixOptions.ts::REMIX_DEFAULTS` gives 16 (what Set Builder sends), `TransitionTimeline` falls back to 16, and the request models default to 16 while `dj.default_transition_bars` never reaches an API render (MX-67). No route exposes config values, so showing the server default in the UI needs a new response field or a hardcode kept in sync across all four. Pydantic ignores unknown fields, so extra fields in the frontend request types are dropped silently (UI-33, MX-33).
- States: splash with no deck loaded; per field spinners; a `compatError` box for compatibility, preview and remix failures; job badge from the store.
- "Check compatibility" on an unanalysed song makes the server analyse inside the request, and the client aborts at the request timeout.
- Backlog: wheel mislabels keys (UI-01); preview drops effect and preset (UI-09); the RemixControls bars control is ignored here (UI-16); cue regions and timeline depend on a field the API never sends (UI-12); "Mix Ready" ignores `quality_passed` (UI-15); "Open in Vault" is a raw anchor (UI-27).

### Set Builder (`/set-builder`)

- Component `SetBuilder`; children `EnergyArc`, `SetRow`, `RemixControls` under Settings.
- User can: add songs from a pool, drag to reorder (dnd kit), remove, clear, export a numbered set.txt, auto sequence, render a chained mix.
- Calls: `GET /library`, `POST /setlist/optimize` (tracks with title, artist, bpm, energy, camelot; reads the returned names), `POST /dj-chain` (songs plus the `RemixControls` values).
- States: pool and set empty states; inline error; chain status line from the store. The set lives only in component state and is lost on navigation.
- The optimistic job is typed `dj_chain`; the server creates `dj_remix`, so the first SSE frame relabels it (UI-20).
- Backlog: chains longer than the API limit (UI-08); the empty library CTA points at Operations (UI-29).

### Signal Search (`/signal-search`)

- Component `SignalSearch`; children `ResultCard`, `ScoreRing`.
- User can: search by track (vector index) or by text (CLAP), choose the result count, filter results client side by BPM, key or Camelot text and minimum similarity (the state is named `energyMin` but filters on `score`), send a result to Mix Deck.
- Calls: by track `GET /library/similar/{name}`; by text `GET /library/crate-search`. The page never sends the camelot, BPM range or name filters that `analysisApi.crateSearch` supports.
- States: splash, loading, inline error, "No tracks match your filters".
- The empty CLAP index hint sends users to Operations, whose Maintenance panel only operators see. The send to Deck B button navigates with only `song_b`, which empties Deck A.
- Backlog: the count buttons exceed the similar route cap (UI-17).

### AI Lab (`/ai-lab`)

- Component `AILab`; children `JobResultBanner`, `MiniPlayer`.
- Visible only when `features.generative` is true: the rail filters it, both shortcut paths check the flag, and `RequireGenerative` redirects the route.
- Calls: `GET /ai/models`, `POST /ai/style-transfer`, `POST /ai/inpaint`, `POST /ai/tokenize`.
- The server files every generative job as `analyze` with `meta.type`, so the optimistic label is replaced on the first SSE frame.
- Backlog: the Model dropdown is ignored by the API (UI-26); tokenize writes into the shared library behind only the flag (MX-50).

### Mix Vault (`/mix-vault`)

- Component `MixVault`; children `MixCard`, `WaveformPlayer` (inline WaveSurfer).
- User can: play in the global player with feedback context, download the mix and CREDITS.txt, expand details and credits, toggle previews.
- Calls: `GET /jobs` filtered by job type and `done` status (one query for `dj_remix`, one for `dj_chain`), output URLs, `POST /feedback/event` (`play_start`, `complete`).
- States: "Vault is empty" with a CTA to Mix Deck; fetch errors go only to `console.warn`.
- Expired renders: the retention sweep's `scripts/api/jobs.py::mark_expired` sets `result.expired = True` on jobs whose output was removed, and the card list drops those jobs (`j.result?.expired !== true`). Any page that lists past mixes must apply the same filter or it shows dead download links.
- Backlog: the `dj_chain` query is always empty (UI-20); REST jobs lack meta, so titles fall back and the previews toggle hides (UI-10); the mount load toasts every historic mix (UI-11); CREDITS.txt is served as audio (UI-35).

### Operations (`/operations`)

- Component `Operations`; children `ProcessingQueuePanel`, `ProcessingBucket`, `frontend/src/pages/Operations.tsx::StoragePanel`, `MaintenancePanel`.
- Calls: `GET /library/processing-status` every 1 s (UI-39); operators only: `GET /library/storage` every 5 s, `POST /library/storage/prune`, `POST /library/storage/evict` (dry run, then real), `POST /library/build-clap-index`, `POST /library/calibrate-lufs`, `POST /library/analyze-missing` with force.
- States: the processing panel returns null on error or with zero songs; the storage panel returns null on error or before data. A non operator on an empty library sees only the intro note.
- Backlog: prune runs on one click (UI-07) and has no dry run on the server (OPS-02); read `scripts/api/routers/library.py::storage_prune` before touching it and never exercise it against the real `library/`; `tests/test_storage_prune_safety.py` already builds a scratch `LibraryManager(library_dir=...)` and patches `get_library_manager`, so start there. Non operators lose the read only storage stats too (UI-28).

### Account (`/account`)

- Component `AccountPage`.
- Calls: `GET /account/export` (saved as a zip), `DELETE /account`, then sign out.
- States: inline error; Delete stays disabled until `DELETE` is typed and is always disabled when auth is not configured.
- Backlog: the fetch bypasses the client (UI-38).

### Legal (`/legal`, `/legal/:page`)

- Component `LegalPage`; child `DmcaForm`.
- Calls: `GET /legal/{page}` (plain text), `POST /legal/dmca` with the `DmcaRequest` fields (no auth header, no credentials).
- States: loading, error, text; the DMCA form shows the returned report id.

### DJ Widget (`/widget`)

- Component `Widget`; children `WidgetPanel`, `RecRow`, `CamelotWheel`.
- Runs outside `AppShell`: no SSE, no job store, no player bar.
- Calls, direct and not through React Query: `GET /library`, `GET /library/{name}`, `GET /library/similar/{name}`, `POST /feedback/event` (`reorder` when a suggestion other than the top one is queued).
- Backlog: the same wheel defect as Mix Deck (UI-01); the pop out resets state (UI-30); the rail item strands the user (UI-40).

### Shared components

| Component | Used by | Note |
|---|---|---|
| `WaveformDeck` | Mix Deck | WaveSurfer load with no fetch params (UI-03); styles live in `frontend/src/pages/MixDeck.css` |
| `frontend/src/pages/MixVault.tsx::WaveformPlayer` | Mix Vault | same fetch behaviour |
| `StemsPlayerRow` | Library Atlas | a plain audio element per stem |
| `CamelotWheel` | Mix Deck, Widget | UI-01 |
| `RemixControls` with `frontend/src/components/remixOptions.ts::REMIX_DEFAULTS` | Mix Deck, Set Builder | bars control is dead in Mix Deck (UI-16) |
| `TransitionTimeline` | Mix Deck | UI-12 |

### Operator gating

The server gates catalog mutating routes with `require_operator`. The UI reads `features.operator` only in Operations; Library Atlas shows every operator action to everyone (UI-13).

| Action (page) | Endpoint | Server gate | UI hides for non operators |
|---|---|---|---|
| Delete song, delete broken batch, scan broken, analyse missing (Library Atlas) | `DELETE /library/{name}`, `POST /library/delete-batch`, `POST /library/scan-broken`, `POST /library/analyze-missing` | `require_operator` | no |
| Split stems, row or bulk (Library Atlas) | `POST /stems/split` | `require_operator` and `stem_split` | only by the flag |
| Analyse one song (Library Atlas) | `POST /analyze` | `require_user` only (DOC-01) | not needed |
| Storage prune, evict preview, evict (Operations) | `POST /library/storage/prune`, `POST /library/storage/evict` | `require_operator` | yes, whole panel |
| CLAP index, LUFS calibration, reanalyse all (Operations) | `POST /library/build-clap-index`, `POST /library/calibrate-lufs`, `POST /library/analyze-missing` | `require_operator` | yes, whole panel |

## Page to endpoint to router

Every router is mounted under `require_user` through `scripts/api/routes.py::router` except `system` and `legal`. The analysis router is registered before the library router on purpose: `GET /library/crate-search` would otherwise match `GET /library/{name}`. Handlers without `response_model` return an untyped dict.

| Page or caller | Method and path | Handler | Response | Frontend type notes |
|---|---|---|---|---|
| Shell | `GET /features` | `scripts/api/routers/system.py::features` | dict | `Features`, matches |
| Mission Control | `GET /health/live` | `scripts/api/routers/system.py::liveness` | dict | `HealthStatus`; uptime never sent |
| Most pages | `GET /library` | `scripts/api/routers/library.py::list_library` | `LibraryListResponse` | `SongInfo` lacks `mode`, `size_mb`, `has_full_wav`, `source`; `key` is a root note only, with mode separate |
| Mix Deck, Widget | `GET /library/{name}` | `scripts/api/routers/library.py::get_song` | `SongInfo` | same as above |
| Library Atlas | `DELETE /library/{name}`, `POST /library/delete-batch`, `POST /library/scan-broken`, `POST /library/analyze-missing` | `delete_song`, `delete_songs_batch`, `scan_broken`, `analyze_missing` in `scripts/api/routers/library.py` | dict, dict, `JobResponse`, `JobResponse` | all operator gated, UI-13 |
| Library Atlas | `POST /stems/split` | `scripts/api/routers/stems.py::stem_split` | `JobResponse` | flag and operator gated |
| Library Atlas, Mix Deck, Widget, Signal Search | `GET /library/similar/{name}` | `scripts/api/routers/analysis.py::similar_songs` | dict | `k` cap, UI-17; writes on GET, MX-32 |
| Library Atlas | `GET /recommend/{name}` | `scripts/api/routers/analysis.py::recommend_songs` | dict | its docstring says a cold cache can take 20 to 30 s, near the client timeout |
| Library Atlas | `POST /analyze` | `scripts/api/routers/analysis.py::analyze_song` | `JobResponse` | matches |
| Library Atlas | crate and favorite routes | `list_crates`, `create_crate`, `delete_crate`, `get_crate_songs`, `add_to_crate`, `remove_from_crate`, `list_favorites`, `add_favorite`, `remove_favorite` in `scripts/api/routers/crates.py` | dicts | `cratesApi.songs` shape, UI-02; `create` returns only id and name; `cratesApi.rename` has no caller |
| Mix Deck | `POST /compatibility` | `scripts/api/routers/analysis.py::check_compatibility` | `CompatibilityResult` | no `transition_plan`, UI-12; the key score, MX-04 |
| Mix Deck | `GET /compatibility/matches/{name}` | `scripts/api/routers/analysis.py::compatibility_matches` | dict | 409 when the source is unanalysed, caught to null |
| Mix Deck, Inspector retry (`meta.preview`) | `POST /dj-remix/preview` | `scripts/api/routers/remix.py::dj_remix_preview` | `JobResponse` | UI-09 |
| Mix Deck, Inspector retry (replays `meta.request` verbatim) | `POST /dj-remix` | `scripts/api/routers/remix.py::dj_remix` | `JobResponse` | ignored fields, UI-33; option validation, MX-07 |
| Mix Deck | `POST /dj-remix/feedback` | `scripts/api/routers/remix.py::dj_remix_feedback` | `RemixFeedbackResponse` | 404 when no spectral record |
| Set Builder, Inspector retry | `POST /dj-chain` | `scripts/api/routers/remix.py::dj_chain` | `JobResponse` | song cap, UI-08; job type is `dj_remix`, UI-20 |
| Set Builder | `POST /setlist/optimize` | `scripts/api/routers/setlist.py::optimize_setlist` | `OptimizeResponse` | client maps `name` to `title`; matches |
| Signal Search | `GET /library/crate-search` | `scripts/api/routers/analysis.py::crate_search` | dict | matches |
| Mix Vault, poller, Mission Control | `GET /jobs` | `scripts/api/routers/jobs.py::list_jobs` | list of `JobResponse` | no meta, UI-10; filters take backend values (`done`, `dj_remix`) and apply before `limit`; `limit` defaults to 20 and is capped at 1 to 100, so a full history needs paging or a server change |
| Inspector | `DELETE /jobs/{id}` | `scripts/api/routers/jobs.py::cancel_job` | dict | 409 when terminal |
| Mix Vault, Mix Deck, AI Lab | `GET /outputs/{session_id}/{filename}` | `scripts/api/routers/library.py::stream_output` | `FileResponse` | UI-35 |
| Library Atlas, player | audio, stem and cue export routes under `GET /library/{name}` | `stream_audio`, `stream_stem`, `export_cues` in `scripts/api/routers/library.py` | files | URL builders, see the bypass list |
| Operations | `GET /library/processing-status`, `GET /library/storage` | `processing_status`, `storage_status` in `scripts/api/routers/library.py` | `ProcessingStatusResponse`, `StorageStatusResponse` | match |
| Operations (operator) | `POST /library/storage/prune`, `POST /library/storage/evict` | `storage_prune`, `storage_evict` in `scripts/api/routers/library.py` | `StoragePruneResponse`, `StorageEvictResponse` | OPS-02, UI-07 |
| Operations (operator) | `POST /library/build-clap-index`, `POST /library/calibrate-lufs`, `POST /library/analyze-missing` | `scripts/api/routers/analysis.py::build_clap_index`, `scripts/api/routers/library.py::calibrate_stem_lufs`, `analyze_missing` | `JobResponse` | match |
| AI Lab | `GET /ai/models`, `POST /ai/style-transfer`, `POST /ai/inpaint`, `POST /ai/tokenize` | `model_status`, `style_transfer`, `inpaint_transition`, `tokenize_stems` in `scripts/api/routers/generative.py` | `ModelStatusResponse`, `JobResponse` | `frontend/src/lib/api.ts::AiModelStatus` matches |
| Widget, player, Inspector, Mix Vault | `POST /feedback/event` | `scripts/api/routers/feedback.py::log_feedback_event` | `FeedbackEventResponse` | matches |
| Shell | `GET /events/stream` | `scripts/api/routers/events.py::sse_stream` | SSE | see Realtime |
| Account | `GET /account/export`, `DELETE /account` | `export_account`, `delete_account` in `scripts/api/routers/account.py` | stream, dict | bypasses the client |
| Legal (public) | `GET /legal/{page}`, `POST /legal/dmca` | `legal_page`, `submit_dmca` in `scripts/api/routers/legal.py` | text, dict | bypasses the client |

## Backend routes with no UI

- All of `scripts/api/routers/admin.py` and `scripts/api/routers/spotify.py`.
- `scripts/api/routers/library.py`: `list_library_names`, `mix_storage_status`, `mix_storage_sweep`, `initialize_library`.
- `scripts/api/routers/analysis.py`: `song_structure_v1` (MX-15), `index_stats`, `rebuild_index`.
- `scripts/api/routers/stems.py`: `stem_split_batch`, `compress_stems_single`, `compress_stems_batch`.
- `scripts/api/routers/remix.py`: `synthesize_beat`, `upload_beat` (so the `file` bridge beat mode is unreachable; `frontend/src/components/remixOptions.ts::RemixOptions` offers only `none` and `auto`), `instrument_lab`, `instrument_lab_songs`.
- `scripts/api/routers/setlist.py`: `import_exportify_csv`, `camelot_modulation`, `find_wordplay`.
- `scripts/api/routers/crates.py`: every tag route and `rename_crate`. `frontend/src/lib/api.ts::tagsApi` is an empty object kept for a contract test (UI-21).
- `scripts/api/routers/feedback.py::feedback_summary`, `scripts/api/routers/jobs.py::get_job`, `scripts/api/routers/events.py::sse_health`, `scripts/api/routers/system.py::health` and `scripts/api/routers/system.py::readiness`.
- Frontend types with no consumer: `FeedbackSummaryResponse`, `StructureV1Response` and its parts, `LibraryStats` (UI-21). `TransitionPlan` is read but no route fills it (UI-12).

## Styling

The design direction is in `DESIGN.md`; token values are in `frontend/src/styles/tokens.css` and summarized in `DEVELOPING.md` section "Design system". `frontend/src/styles/global.css` imports the tokens and holds the reset and utility classes; `frontend/src/main.tsx` imports `global.css`. `frontend/index.html` sets the theme colour meta tag and loads Space Grotesk, Inter and JetBrains Mono from Google Fonts. Dark only, no theme switch, no `prefers-color-scheme` handling (UI-36).

One CSS file per page or component, imported by that TSX file, with three exceptions: `WaveformDeck` has no CSS file and takes its `md-waveform*` classes from `frontend/src/pages/MixDeck.css`, so it renders unstyled outside Mix Deck; `CamelotWheel` is inline SVG only; `RootErrorBoundary` reuses `PageErrorBoundary.css`. No Tailwind and no CSS modules, so class names are global and the prefix is the only namespace.

| Prefix | Owner |
|---|---|
| `mc-`, `la-`, `md-`, `sb-`, `ss-`, `al-`, `mv-`, `ops-`, `wgt-`, `acct-`, `legal-` | pages: Mission Control, Library Atlas, Mix Deck, Set Builder, Signal Search, AI Lab, Mix Vault, Operations, Widget, Account, Legal |
| `page-base` | `frontend/src/pages/PageBase.css`, the shared page frame |
| `app-shell`, `left-rail`, `right-inspector`, `inspector-`, `auth-` | shell files |
| `pb-`, `player-`, `rc-`, `tt-`, `spr-`, `peb-`, `shortcuts-`, `toast-` | components |

Canvas and WaveSurfer cannot read CSS variables, so these hold literal copies of token colours and must be changed by hand when a token changes: `frontend/src/pages/MixDeck.tsx::DECK_HEX`, the WaveSurfer colours in `WaveformPlayer`, the canvas fills in `frontend/src/components/PlayerBar.tsx::MiniViz`, the `hsl()` colours in `CamelotWheel` and `frontend/src/pages/Widget.tsx::camelotHue`, the white label fills (`#fff`, `rgba(255,255,255,0.75)`, `rgba(255,255,255,0.3)`) in `CamelotWheel`, and the `rgba(0,0,0,0.2)` badge background in `frontend/src/pages/MixVault.tsx::MixCard`. CSS files with literal hex outside the tokens: LeftRail.css, PageErrorBoundary.css, PlayerBar.css, LibraryAtlas.css, LegalPage.css.

## Frontend tests

Commands and CI: `DEVELOPING.md` section "Commands" (it drifts from CI, see DOC-03) and `.github/workflows/ci.yml`. Run the suite once with `npx vitest run` from `frontend/`; `npm test` starts vitest in watch mode and never exits. CI lints the whole folder, config files included, and `frontend/eslint.config.js` imports a package not declared in `frontend/package.json` (OPS-17). Environment: vitest with jsdom and `frontend/src/test-setup.ts`.

| Test file | Covers |
|---|---|
| `frontend/src/lib/__tests__/api.test.ts` | `request`: happy path, `ApiError`, timeout |
| `frontend/src/__tests__/api-contract.test.ts`, `frontend/src/__tests__/contracts.test.tsx` | crates wire shapes, removed client methods, AI Lab and Mix Deck request bodies |
| `frontend/src/__tests__/auth.test.tsx`, `frontend/src/__tests__/supabase.test.ts`, `frontend/src/shell/__tests__/AuthProvider.test.tsx` | RequireAuth, Bearer wiring, cookie mirror and domain |
| `frontend/src/__tests__/features.test.tsx`, `frontend/src/shell/__tests__/AppShell.test.tsx` | flag gating in LeftRail, shortcuts, Mission Control, Library Atlas, Operations, the AI Lab route guard |
| `frontend/src/__tests__/jobsux.test.tsx` | Inspector cancel and retry, the poller merge |
| `frontend/src/__tests__/crates.test.tsx` | Library Atlas crate add wire format (its mock hides UI-02) |
| `frontend/src/__tests__/vault_account_legal.test.tsx`, `frontend/src/__tests__/errors.test.tsx`, `frontend/src/__tests__/smoke.test.ts` | Mix Vault credits, Account, Legal and DMCA, error boundaries and toasts, route smoke render |
| `frontend/src/stores/__tests__/playerStore.test.ts`, `frontend/src/hooks/__tests__/useJobTimer.test.ts`, `frontend/src/components/__tests__/` | player store with a mocked engine, timer, RemixControls, WaveformDeck abort, key normalization |

Mocked away: `useSSE`, `useJobPoller` and `useJobToasts` in the smoke, AppShell and vault tests; the whole API module in `smoke.test.ts`; `playerEngine` in the player test; WaveSurfer and the virtualizer where pages render; `fetch` stubbed per file.

Not tested: `useSSE` and its event handlers, `normalizeJob` directly, `appStore` (`setJobs`, `completionLog`, inspector toggle), `useJobToasts`, `playerEngine`, the Camelot map in `CamelotWheel` (its test covers only `normalizeKey`), the Signal Search count, the Set Builder chain size, Widget, PlayerBar, WaveSurfer credentials. Stale fixtures: the poller test feeds lowercase REST statuses, the retry tests feed `meta.request` that REST never serves, and the crates mock returns a bare array. There is no browser or end to end suite for the frontend.

## Seeing a UI change locally

Start the API and Vite as `DEVELOPING.md` section "Running locally" describes. Locally, auth is off on both sides: with no `VITE_SUPABASE_*` vars the frontend skips sign in, and with `REMIXMATE_AUTH_ENABLED` unset the backend runs every request as `local`, which also passes every operator gate. So operator only controls appear and work locally, which hides UI-13 and UI-28; test hosted behaviour against `app/STATUS.md` and `app/deploy/PROVISION.md` instead.

Tracks enter only through the operator ingest tool `scripts/download.py`; there is no in app acquisition. Pages with no library or no analysed songs:

| Page | Empty `library/` | Songs present, none analysed |
|---|---|---|
| Mission Control | shows zero songs; works | works |
| Library Atlas | empty message naming the ingest tool | BPM, key and energy show dashes; drawer recommendations and similar may be empty |
| Mix Deck | deck selects hold only the placeholder | strict matches hidden (409 swallowed); compatibility analyses inside the request and can hit the client timeout |
| Set Builder | empty pool, CTA to Operations (UI-29) | auto sequence sends null BPM and Camelot and default energy |
| Signal Search | nothing to pick in track mode; text mode returns the empty CLAP index 404 | similar depends on the server upserting unanalysed songs into the index |
| AI Lab | nothing to pick | depends on the generative backend |
| Mix Vault | unaffected (reads jobs) | unaffected |
| Operations | processing panel hidden | buckets show everything as unprocessed or stems only |
| Widget | search finds nothing | wheel hidden (no key); suggestions depend on the similar route |

## Adding a page

`DEVELOPING.md` section "Common tasks", item "Add a new frontend page", gives the base recipe (page and CSS files on the `PageBase.css` pattern, a `NavItem`, a `NavDestination` member, a `<Route>`, public pages in `App.tsx` outside `RequireAuth`). What it misses:

1. Add the `NavDestination` member in `frontend/src/types/index.ts` before the `NavItem`, or `tsc` fails on `NAV_ITEMS`.
2. Make the page a default export and add a lazy import const in `frontend/src/shell/AppShell.tsx`.
3. Wrap the route element exactly like its siblings, `PageErrorBoundary` then `Suspense` with `PageFallback`, and use a relative path.
4. A feature gated page is gated in three places, as AI Lab is: a route wrapper like `RequireGenerative` (treat pending as off), the `NAV_ITEMS` filter in `LeftRail`, and the shortcut handlers. The server route still gates on its own.
5. Shortcuts: add a `NAV_MAP` letter and a `SHORTCUTS` row by hand; for a number, append to `NUMBER_NAV` and widen the `[1-8]` regex.
6. If the page starts jobs of a new type, check `jobDestination`, which otherwise sends the card to `/library-atlas`. It branches on `job.type`, and any job the server files as `analyze` (every generative and stem job) reads `analyze` after the first SSE frame whatever the optimistic label said; branch on `meta.type` instead (SSE only, UI-10).
7. Add the path to `frontend/src/__tests__/smoke.test.ts::ROUTES`, and extend that file's mock of the API module with any API object the page calls on mount; the mock replaces the whole module, so a missing object is undefined in the test.
8. A page that must render without the shell but with sign in, like `/widget`, goes in the inner `<Routes>` of `AppRoutes`, above the route that falls through to `AppShell`.

## Traps

- Two auth switches, one per side, never compared (UI-19). Hosted with the API on a subdomain needs `VITE_COOKIE_DOMAIN`, or the stream and media GETs arrive without a token, SSE never connects and the poller carries jobs.
- Job `meta` exists only on SSE frames; every REST write erases it (UI-10, UI-14).
- Toasts fire only for React Query errors and job transitions; direct `api.*` calls in handlers must handle their own errors, and 401 and 503 are silent.
- Operator only actions render for every user in Library Atlas and pass for everyone locally (UI-13); verify gating against a hosted build.
- `GET /library/similar/{name}` and `GET /recommend/{name}` write into the shared library (MX-32); do not assume a GET is read only in these routers.
- Mix Deck song keys are root notes with a separate `mode`; pass `camelot` to anything that draws the wheel (UI-01).
- The Vite `/events` and `/outputs` proxy entries are never hit; every request goes through `/api` (UI-21).
