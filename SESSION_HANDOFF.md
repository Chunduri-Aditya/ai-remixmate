# Session Handoff — June 30, 2026

Read this first in the new chat. It's the state of the live-testing/bug-fix session that
just ended, written so a fresh session doesn't have to re-derive any of it.

## What this session actually was

Not a planning session — a live-testing session. The app was open in the browser
(Claude in Chrome MCP) and every fix below was triggered by something that actually broke
on screen or in a terminal log, not by reading code in the abstract.

## Fixed and verified live this session

| Bug | Root cause | Fix | Verified live? |
|---|---|---|---|
| Library Atlas crash: `Cannot read properties of null (reading 'toFixed')` | `!== undefined` checks don't catch `null`; backend returns explicit `null` for unanalyzed fields | `LibraryAtlas.tsx`, `MixDeck.tsx`: `!== undefined` → `!= null` | Yes |
| "Missing analysis" undercounted (69 vs true 529) | `has_analysis()` only checked for a `bpm` key in `meta.json`, but `music_index.py`'s lightweight indexing pass *also* writes a partial `meta.json` with `bpm` — without ever running real analysis | `analysis_pipeline.py:has_analysis()` now also requires `analysis.json` to exist | Yes — count jumped 69→529 live |
| Repeated 400 floods on "Hoang - Don't Say (feat. Nevve)" | Three independent song-naming policies disagreed on punctuation (download sanitizer vs API validator vs ytmusic sanitizer); compounded by React Query retrying non-retryable 4xx errors | Unified naming in `scripts/core/paths.py:sanitize_song_name()` / `SONG_NAME_RE`; `main.tsx` query client no longer retries 4xx | Naming logic verified standalone; **not re-tested against this specific song post-fix** |
| Storage page showing 748 vs real 588 songs | `LibraryManager`'s separate `.index.json` never got updated by `DELETE /library/{name}` (which only did `shutil.rmtree`) | Added `LibraryManager.unregister()`, wired into delete route; made `list_songs()` self-healing for already-stale entries | **Not re-verified live** — attention moved to the naming bug right after |
| DJ compatibility score marking objectively bad pairs `compatible: true` | `track_metadata.py:compatibility_score()` averaged key/bpm/energy into `overall` with a flat `≥0.55` threshold — a total key clash (`key_score=0`) or 2.5x BPM gap (`bpm_score=0`) could still average above threshold | Added a veto: `compatible = overall >= 0.55 and key_score > 0.0 and bpm_score > 0.0` | Yes — verified via direct `POST /compatibility` calls before/after |
| Job cards in the Inspector panel weren't clickable / no warning before navigating away from a running job | n/a (new feature, user-requested) | `RightInspector.tsx`: click navigates to a job-type-appropriate route; shows `window.confirm()` first if job is RUNNING/PENDING; added `.inspector-job-card--clickable` CSS | `tsc --noEmit` clean; **not yet verified live in browser** |

## Explicitly investigated and ruled NOT a bug

- **-9.8 LUFS on a DJ remix output** — looked alarming against CLAUDE.md's "-14 LUFS broadcast standard" line, but `task_modules/remix.py` intentionally targets `-8.0 LUFS` for `/dj-remix` outputs specifically (vs -14 for streaming-style outputs). 1.8 LU off target is inside the mastering module's own tolerance. CLAUDE.md's top-line doc comment is just imprecise, not a code bug.

## Open / not yet fixed (carried into the new chat)

1. **BPM source-of-truth inconsistency.** "I Remember": cached library BPM = 63.8, but the live render engine independently recomputed 129.2 BPM for the same song (~2x — classic beat-tracker octave error). Not root-caused yet. Likely affects more than one song. **This is now understood to be load-bearing for the "remixing is bad" complaint — see `REMIX_QUALITY_INSIGHTS.md`.**
2. **Limiter clipping under extreme tempo-stretch.** An aggressive-test render produced `quality_passed: false` with 227,029 samples over the -1.0 dBFS ceiling. Root cause identified this session (see insights doc) but the fix itself was not applied.
3. **Storage/index phantom-entry fix** — not re-verified live post-fix.
4. **Naming fix** — not re-tested against the specific song that originally triggered the 400 flood.
5. **Job-card click/confirm feature** — not verified live (only typechecked).
6. **Aggressive-test sweep was incomplete.** Set Builder chain-remix render, Signal Search re-scoring (post `has_analysis()` fix — results may no longer cluster at 99.7-99.99% similarity), and full AI Lab job execution were identified as untested but never run.

## Reliability note for whoever picks this up

Every edit to a `scripts/`-tree file restarts `uvicorn --reload`, which costs 10-90+ seconds
(librosa/demucs/torch imports) and kills any in-progress background job. This happened
repeatedly this session and is a structural annoyance, not something fixable purely in code —
flagging so the next session doesn't waste time being surprised by it.

## Where the deeper material lives

- `AUDIT.md` / `CONTEXT.md` — a prior deep audit (June 27) of `dj_engine.py`, `dj_analysis.py`,
  `key_detection.py`, `mastering.py`, `api/jobs.py`, schema drift. Most of its **critical/major**
  findings were fixed in the June 28 "Audit Bug Fixes" pass (see `CLAUDE.md` changelog) —
  cross-check before assuming something there is still broken.
- `IMPROVEMENTS.md` / `IMPROVEMENTS_V2.md` — staged gap-closure plan vs Spotify/Apple Music,
  Stages 1-3 complete as of this session, Stage 4 (vocal analyzer, CUE-DETR, global energy-arc
  optimizer) not started.
- `docs/DJ_THEORY.md` — the project's own research-grade reference for what "correct" DJ mixing
  requires (Camelot theory, compatibility weights, phrase alignment, bass-swap timing, limiter/EQ
  practice). Used as the ground truth for `REMIX_QUALITY_INSIGHTS.md`.
- `REMIX_QUALITY_INSIGHTS.md` (new, written this session) — the actual answer to "why is the
  remixing bad," synthesized from the above plus this session's live findings.
