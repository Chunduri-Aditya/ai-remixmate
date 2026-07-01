# AI RemixMate — Self-Improving DJ Research Track
> Drafted: June 30, 2026 | Status: proposal, not started | Depends on: REMIX_QUALITY_INSIGHTS.md fixes (#1–#4) being live

## 0. Read this before building anything

[Certain] This codebase has zero behavioral telemetry today. No skip events, no play-through
tracking, no "user rejected this transition" signal anywhere — only `favorites` (explicit,
binary, sparse) and `crates` (curation, not feedback). "Self-improving from user behavior"
presupposes a stream of behavior to learn from. That stream doesn't exist yet. Stage A below is
not optional scaffolding — it's the actual prerequisite, and it alone is worth 2-3 weeks before
any learning algorithm is justified.

[Likely] The honest framing problem: this is a single-user (or very-low-N) hobby project. Classic
"self-improving" framing implies enough interaction volume for statistical learning to outperform
a fixed prior. At N=1 user generating maybe 10-50 mixes a week, a contextual bandit or RL agent
will either (a) overfit to noise and chase idiosyncratic one-off reactions, or (b) need so much
regularization toward the existing SetFlow prior that "self-improving" is really "online
preference re-ranking with the original DJ_THEORY.md formula doing 90% of the work." That's still
a legitimate research contribution — but it should be called what it is. Section 4 below is
designed around this constraint rather than against it.

[Guessing, flag for explicit decision] Don't start this until the Stage 1-4 correctness fixes in
REMIX_QUALITY_INSIGHTS.md are confirmed live. Learning a preference model on top of a compatibility
score that still has BPM octave errors or vocal-clash blind spots means the model will partly learn
to compensate for bugs instead of taste — contaminating the signal you're trying to study, and
making any later bug fix invalidate part of the learned model.

---

## 1. The actual research question

Not "can an AI DJ learn user taste" (yes, trivially, with enough data — not interesting on its
own). The question worth a paper or a strong portfolio writeup is narrower:

**Can a small, interpretable online correction term — learned from implicit behavioral signals,
applied on top of a fixed musicological prior (SetFlow / DJ_THEORY.md) — measurably reduce skip
rate and increase set-completion rate, without the system drifting away from the musicological
constraints that make a mix sound competent in the first place?**

This framing matters for three reasons:
- It's falsifiable with the data volume this project will actually generate.
- It's a real, underexplored gap: most "AI DJ" research (the SetFlow paper itself, Spotify's
  internal systems) treats compatibility scoring as either fully hand-tuned or fully learned from
  huge corpora — not as a small online correction to a hand-tuned prior at personalization scale.
- It connects directly to the bugs already found: REMIX_QUALITY_INSIGHTS.md finding #1 showed the
  hand-tuned formula doesn't even match its own documented weights. A learned correction term
  needs that fixed first, or it's correcting noise.

---

## 2. What "user behavior" actually means here (signal inventory)

| Signal | Exists today? | Strength | What it tells you |
|---|---|---|---|
| Favorite / unfavorite a song | [Certain] Yes — `crates.py` | Strong, explicit | Long-term taste, not transition quality |
| Crate membership / ordering | [Certain] Yes — `crates.py` CRUD | Medium, explicit | Curation intent, weak transition signal |
| Job cancelled mid-render | [Certain] Yes — `DELETE /jobs/{id}`, `_cancelled` set in `api/jobs.py` | Weak-medium, implicit | "Didn't want this specific remix" — could be impatience, not quality |
| Skip during playback | [Guessing] No tracking — frontend has an audio player (Mix Vault) but no skip/seek event is sent to the backend | Strong, implicit | The single most valuable signal and the one that doesn't exist |
| Replay / re-download same remix | Not tracked | Strong, implicit | Positive signal — proxy for "this transition worked" |
| Manual re-roll ("try a different transition_effect/preset") | Not tracked as a distinct event — each is just a new `/dj-remix` job | Medium, implicit | Rejection of the auto-selected preset/effect |
| Set Builder manual reorder after auto-arrange | Not tracked | Strong, implicit | Direct disagreement with `setlist_planner.py`'s ordering |
| Compatibility check requested but remix never rendered | Not tracked | Weak, implicit | Possible rejection signal, high false-positive rate (could just be browsing) |
| Explicit thumbs up/down on a rendered mix | Doesn't exist | Strongest if added | Needs new UI — cheapest high-quality signal to add |

**Conclusion:** two new things need to exist before anything else: (1) a lightweight playback
event log (play started / skipped-at-Xs / completed / replayed) wired to the existing audio
player in Mix Vault and the DJ Widget, and (2) an explicit thumbs up/down affordance on a
completed render, surfaced via the Inspector panel job card (which `RightInspector.tsx` already
makes clickable per the June 30 session work — a thumbs up/down is a small addition to that same
surface, not a new screen).

---

## 3. Data architecture (Stage A — build this first, learn nothing yet)

Mirrors the existing `jobs.db` write-through SQLite pattern (`scripts/api/jobs.py`) rather than
introducing a new storage paradigm.

```
scripts/core/feedback_store.py     — new, parallels jobs.py structurally
data/feedback.db                   — new SQLite file, write-through + in-memory cache

Table: feedback_events
  id            TEXT PRIMARY KEY (uuid)
  event_type    TEXT   -- 'play_start' | 'skip' | 'complete' | 'replay' |
                        -- 'reorder' | 'rerender' | 'thumbs_up' | 'thumbs_down'
  song_a        TEXT
  song_b        TEXT   -- nullable, single-song events (e.g. favorite) leave this null
  job_id        TEXT   -- nullable FK to jobs.db, when the event is tied to a render
  position_sec  REAL   -- nullable, where in playback a skip happened
  context       TEXT   -- JSON blob: transition_bars, preset, effect, compatibility scores
                        -- AT THE TIME of the render -- critical for later offline analysis,
                        -- since compatibility_score() itself will keep changing (see Section 0)
  created_at    REAL
```

API surface (new router, `scripts/api/routers/feedback.py`):

```
POST /feedback/event        { event_type, song_a, song_b?, job_id?, position_sec? }
GET  /feedback/summary       -- aggregate counts per song / song-pair, for Stage B analysis
```

Frontend wiring: `useSSE.ts` already centralizes job-state side effects — playback events don't
need SSE (they're user-initiated, not server-pushed), so this is a direct `feedbackApi.logEvent()`
call from the audio player component (Mix Vault) and the DJ Widget's play/skip/seek handlers,
fire-and-forget (don't block playback on the network call).

**Stage A is complete when:** `feedback_events` has been collecting for at least 2-3 weeks of
normal usage, with a `GET /feedback/summary` that returns sane aggregates. No model work starts
before this — the entire point is to find out empirically whether skip rate / replay rate even
vary enough across transitions to be learnable from, before investing in Stage B/C.

---

## 4. The correction-term model (Stage C — only after Stage B confirms signal exists)

Given the N=1-ish data regime (Section 0), the model design choice is deliberately conservative:

**Not** a replacement for `compatibility_score()`. **Not** a neural ranker. A small set of
per-user scalar weight adjustments layered on top of the existing SetFlow formula:

```
compatibility_learned = compatibility_score(meta_a, meta_b)        # fixed prior, unchanged
                       + Δ · f(meta_a, meta_b; θ)
```

Where `θ` is a low-dimensional weight vector (start with 5 parameters — one soft multiplier per
SetFlow term: harmonic_match, beat_alignment, energy_smoothness, genre_proximity,
timbral_similarity) updated via online logistic regression against the binary "did this
transition get skipped / thumbs-down'd" outcome. This is deliberately the simplest model that
could work:

- **Interpretable** — after a month, you can print "this user's θ up-weights genre_proximity
  2.3x relative to the SetFlow default and down-weights timbral_similarity" as an actual finding,
  not a black box.
- **Cheap to fit** — logistic regression on a few hundred to low-thousands of labeled events runs
  in milliseconds, no GPU, no training infrastructure beyond what `scripts/core/` already has.
- **Bounded drift** — clip `θ` to a sane range (e.g. each multiplier ∈ [0.5, 2.0]) so the learned
  correction can re-weight the prior's dimensions but can't invert them (a vocal_clash_penalty
  can get down-weighted by a user who doesn't care about vocal overlap, but it can't become a
  *bonus* — guards against the "self-improving" system learning to recommend objectively bad
  pairings because of a feedback-loop artifact).
- **A real ablation exists**: compare skip rate with `Δ=0` (pure SetFlow) vs the learned θ over a
  held-out window. This is the actual evaluation, not a vibes check.

[Guessing] A contextual bandit (Thompson sampling over θ, rather than plain online logistic
regression) is the natural Stage D upgrade once Stage C's offline ablation shows the correction
term helps — bandits handle the explore/exploit tradeoff properly (the system needs to
occasionally serve a transition it's unsure about to keep learning), which plain online
regression doesn't. Don't start here; start with the simpler thing and only add bandit machinery
if the ablation justifies the complexity.

### Where it plugs into the existing codebase

- `scripts/core/preference_model.py` (new) — `PreferenceModel.score_adjustment(meta_a, meta_b, user_theta)`
- `track_metadata.py:compatibility_score()` gets an optional `preference_model=None` kwarg; when
  provided, adds the Δ term to `overall` post-hoc. Default `None` preserves today's deterministic
  behavior — this must never become a hard dependency of the core scoring path.
- `setlist_planner.py`'s `transition_cost()` is the other consumer — set ordering should also
  benefit from the same learned correction, since Section 2 found manual reordering is one of the
  strongest implicit signals available.
- θ persisted in `data/preference_weights.json`, versioned with a `schema_version` field — when
  the underlying SetFlow formula changes (as it just did, finding #1), old θ values may no longer
  be meaningful and the version bump is the trigger to reset or carefully migrate them.

---

## 5. Evaluation methodology (Stage B, run before Stage C is greenlit)

Before writing a single line of model code:

1. **Does skip/replay rate vary enough to be learnable?** Pull 2-3 weeks of Stage A data, bucket
   transitions by `compatibility_score()`'s `overall` value, and check whether skip rate actually
   correlates with score. If a 0.9-scored transition and a 0.6-scored one get skipped at the same
   rate, the existing score isn't predictive of behavior yet (could mean the score is still wrong
   somewhere, or that skip behavior is dominated by something the score doesn't capture, e.g. song
   selection/mood rather than transition quality) — and a learned correction on top of it inherits
   that same blind spot.
2. **Inter-session consistency check.** Does the same song pair get consistently skipped or
   consistently played through across multiple sessions? If behavior is highly inconsistent
   session-to-session, that's evidence of noise dominating signal at this data volume — a strong
   argument for delaying Stage C regardless of what step 1 shows.
3. **Offline counterfactual estimate** (only if 1 and 2 look promising) — before deploying a
   learned θ live, use inverse propensity scoring on the logged data to estimate what skip rate
   *would have been* under a candidate θ, the same way contextual bandit literature evaluates
   policies offline before A/B testing them. This avoids burning weeks of live A/B testing on a θ
   that offline evaluation would have already ruled out.

**Stage C is only greenlit if steps 1 and 2 show real signal.** If they don't, the honest
conclusion is "this project doesn't generate enough behavioral data for online personalization to
beat a well-tuned fixed prior" — which is itself a publishable negative result given how often
"self-improving" is asserted without this check ever being run.

---

## 6. Staged roadmap

| Stage | Goal | Effort | Gate to proceed |
|---|---|---|---|
| **A — Instrumentation** | `feedback_store.py`, `/feedback/event`, frontend event wiring, thumbs up/down UI | ~1 week | Events flowing for ≥2 weeks of real usage |
| **B — Offline analysis** | Notebook/script: skip-rate vs score correlation, inter-session consistency, IPS estimate | ~3-4 days | Section 5 steps 1-2 show real signal |
| **C — Online correction term** | `preference_model.py`, wiring into `compatibility_score()` + `setlist_planner.py`, bounded θ, ablation report | ~1-2 weeks | Ablation (Δ=0 vs learned θ) shows measurable skip-rate improvement |
| **D — Contextual bandit upgrade** | Thompson sampling over θ for explore/exploit | ~1-2 weeks | Stage C ablation positive AND enough data volume that exploration cost is acceptable |
| **E — Writeup** | Methodology + ablation results as a short paper/portfolio piece — the "online correction to a fixed musicological prior at single-user scale" framing from Section 1 is the actual contribution | ~1 week | Stage C or D complete with real numbers, not projections |

Total realistic timeline if every gate passes: 6-8 weeks. **Likely outcome given the N=1 data
regime:** Stage A-B alone may show insufficient signal for C to be worth building — that's a
valid place to stop, and arguably a more honest research contribution than forcing Stage C-E to
"prove out" the original pitch.

---

## 7. Relationship to existing roadmap docs

- Sits *after* `IMPROVEMENTS_V2.md` Stage 4 (Vocal Analyzer, CUE-DETR, energy-arc optimizer) in
  priority — those fix deterministic correctness/quality gaps with knowable ROI; this is
  exploratory and could legitimately return "not enough data, don't build C-E" as its result.
- Directly depends on `REMIX_QUALITY_INSIGHTS.md` findings #1-#4 being fixed and stable, per
  Section 0 — a moving compatibility-score target makes learned θ values non-comparable across
  time.
- `docs/DJ_THEORY.md` section 3 (SetFlow formula) is the fixed prior this entire track is built
  around correcting, not replacing — if a future revision changes those weights again, Section 4's
  θ schema-versioning is what prevents silently-stale personalization data.
