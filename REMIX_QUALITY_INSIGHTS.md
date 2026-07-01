# Why the remixing is bad — synthesized findings

Compiled from `docs/DJ_THEORY.md` (the project's own ground-truth reference for correct DJ
mixing), `AUDIT.md` + `CONTEXT.md` (June 27 deep audit), `IMPROVEMENTS.md` / `IMPROVEMENTS_V2.md`
(gap-closure plans), `claude-code-improvements-7.md` (render-path hardening spec), and live
findings from this session's aggressive testing against the real library. Ordered by how
directly each one explains an audibly bad mix, not by file location.

[Certain] = confirmed by reading the actual current code. [Likely] = strong inference from
adjacent evidence, not directly observed. [Guessing] = plausible but unverified.

---

## 1. The compatibility score doesn't match the project's own research

[Certain] `docs/DJ_THEORY.md` section 3 specifies a research-backed compatibility formula
(SetFlow algorithm, validated against 1,557 mixes / 24,202 tracks):

```
compatibility = 0.35 × harmonic_match
              + 0.25 × beat_alignment
              + 0.15 × energy_smoothness
              + 0.15 × genre_proximity
              + 0.10 × timbral_similarity
              − vocal_clash_penalty
```

What's actually implemented in `scripts/core/track_metadata.py:MetadataClient.compatibility_score()`
is a *different* formula: `key_score × 0.45 + bpm_score × 0.40 + energy_score × 0.15`. No
`genre_proximity`, no `timbral_similarity`, no `vocal_clash_penalty` term at all.

This matters because the document that defines what "good" means for this project — the
thing every Camelot move, transition length, and bass-swap timing rule traces back to — was
never wired into the actual scoring function the engine uses to pick or rate pairings. The
engine is over-weighting BPM (40% vs the documented 25% for beat_alignment, which itself
isn't the same thing as raw BPM match) and has zero mechanism for catching the single most
audible failure mode in DJ mixing: **two vocals playing over each other.** `vocal_clash_penalty`
exists in the theory doc and nowhere in the code.

**This session's fix** (the veto: `compatible = overall >= 0.55 and key_score > 0 and bpm_score > 0`)
closed the worst symptom — pairs with a total key clash or a 2.5x BPM gap no longer get
marked `compatible: true`. It did not close the underlying gap: genre and timbre still don't
factor into the score, and there's still no vocal-overlap detection anywhere in the
compatibility path despite stem separation (which would make vocal-presence detection cheap)
already existing in the codebase via Demucs.

---

## 2. BPM detection has at least one confirmed octave error, and it feeds directly into render math

[Certain] During this session's aggressive testing, the cached library BPM for "I Remember"
was 63.8, but the live `/dj-remix` render engine independently recomputed 129.2 BPM for the
same song — almost exactly 2x. `docs/DJ_THEORY.md` section 7 names this exact failure mode
explicitly: *"Octave errors (70 vs 140) require dedicated classifier branch"* — i.e. the
project's own reference material already knew this class of bug was likely and the codebase
doesn't have the dedicated handling it calls for.

[Likely] This is not cosmetic. `tempo_shift_ratio` (the time-stretch factor applied to Song B)
is derived from the BPM pairing. A 2x BPM error doesn't produce a "slightly wrong" stretch —
it produces a stretch ratio off by a factor of ~2, which is exactly the boundary of the clamp
described in finding #3 below. Two songs that are actually close in tempo can get computed as
needing a near-2x stretch (or vice versa) purely because one of the two BPM readings picked
the wrong octave.

**Not yet root-caused**: whether this is isolated to "I Remember" or systemic. The two BPM
values come from different code paths (cached analysis pipeline vs. whatever `dj_engine.py`
recomputes at render time) — those two paths should agree and currently don't, which is itself
worth fixing independent of which one is "more correct."

---

## 3. The stretch-ratio safety clamp silently absorbs the BPM error instead of surfacing it

[Certain] `scripts/core/dj_engine.py:239-253` (added in the Batch 7 hardening pass,
`claude-code-improvements-7.md`):

```python
_MIN_STRETCH = 0.5   # B is at most 2× faster than A
_MAX_STRETCH = 2.0   # B is at most 2× slower than A

def _safe_ratio(ratio: float) -> float:
    if ratio is None or not math.isfinite(ratio) or ratio <= 0.0:
        return 1.0
    return float(min(max(ratio, _MIN_STRETCH), _MAX_STRETCH))
```

This is good defensive engineering for *crash* prevention — it stops `NaN`/`inf`/garbage
ratios from reaching librosa. But it is silent on the *musical* failure mode: when a BPM
octave error (finding #2) produces a ratio at or near the 2.0 boundary, the clamp doesn't
reject it or flag it — it lets a maximal, audibly extreme time-stretch through as if it were
a legitimate creative choice. `docs/DJ_THEORY.md` section 4 is explicit that phase-vocoder
time-stretching "becomes musically unacceptable beyond ±3-5 BPM of original tempo." A ratio
anywhere near 2.0 is an order of magnitude past that threshold — this is the gap between
"doesn't crash" and "sounds correct," and right now the codebase only guarantees the former.

[Guessing] No log line currently distinguishes "ratio was 1.4, totally fine" from "ratio hit
the 2.0 ceiling because of a bad BPM read" — `_safe_ratio` does log when it clamps
(`dj_engine.py` around the `_time_stretch` call), but nothing upstream flags *why* a ratio
landed near the boundary in the first place. Worth checking job logs from a bad-sounding
render for this clamp-log line as the fastest way to confirm/deny this theory on a specific
song.

---

## 4. The limiter's attack and release share one time constant — that's why it clips instead of limiting

[Certain] `scripts/core/mastering.py:apply_limiter()` (lines 184-236). The algorithm:

1. Computes a look-ahead peak envelope via `maximum_filter1d` (3ms window) — this part is fine,
   it does anticipate upcoming peaks.
2. Computes a per-sample target gain reduction from that envelope.
3. Smooths the gain curve with a **single one-pole IIR filter using the `release_ms=50ms` time
   constant for both directions** — there is no separate, fast attack constant. The same
   `alpha = exp(-1/rel_samp)` governs how fast gain reduction engages *and* how fast it
   releases.
4. Hard-clips whatever's left as a "safety net": `np.clip(limited, -ceiling, ceiling)`.

[Likely] Real brick-wall limiters use a fast attack (near-instant, often samples) and a slow
release (tens of ms, to avoid audible pumping) — the asymmetry is the entire point of a
look-ahead limiter design. Here, a sudden transient that needs 100% gain reduction takes
~50ms to ramp down to that level even though the look-ahead window already "knew" the peak
was coming. During that ~50ms lag, the unattenuated peak passes through and gets caught by
step 4's hard clip instead of being smoothly limited. Step 4 is documented as a "safety net"
(i.e. should fire rarely) — but under dense transient material it becomes the primary limiting
mechanism, and a hard clip is audible distortion, not loudness control.

[Likely] This directly explains the 227,029-sample clipping defect (`quality_passed: false`)
found this session on an extreme-tempo-stretch render: extreme time-stretching (see findings
#2-#3) produces denser, sharper transients than the limiter's symmetric 50ms smoothing can
track, so the hard-clip fallback fires constantly instead of occasionally.

**Fix shape** (not yet applied): split into a fast attack coefficient (e.g. ~1-5ms) and the
existing slow release coefficient, computed independently — `alpha_attack` when reduction is
decreasing (gain needs to drop), `alpha_release` when reduction is increasing back toward 1.0.
This is a standard two-coefficient limiter design, not a novel one.

Secondary, lower-severity issue in the same function (flagged in `AUDIT.md`, not yet fixed):
it's labeled "true-peak" but only measures sample peaks — no oversampling — so inter-sample
peaks can exceed the ceiling even when the limiter "succeeds." Worth fixing alongside the
attack/release split since both touch the same function.

---

## 5. Historical correctness bugs — mostly fixed, worth a sanity check before assuming any are live

`AUDIT.md` (June 27) found several critical/major DSP bugs. Per `CLAUDE.md`'s June 28 "Audit
Bug Fixes" changelog entry, these were addressed:

- `render()` not silencing Song B during its "solo" first half (was the single worst bug in
  the audit — `dj_engine.py:_apply_dynamic_eq_fade`) — **fixed**.
- Clash-path bass swap disabled exactly when keys clash (`dj_analysis.py:plan_transition`) —
  **fixed**.
- Camelot semitone table wrong on A-ring and part of B-ring (`key_detection.py:pitch_shift_for_camelot`)
  — **fixed**, now derives from `CAMELOT`/`NOTE_NAMES`.
- SSE broadcasts silently dropped from worker threads (`api/main.py`) — **fixed**.
- Dead low-pass coefficients computed but never applied in the bass swap (`dj_engine.py`) —
  **removed**.

[Likely] These were genuinely the worst correctness bugs in the audit and fixing them should
have measurably improved mix quality already. If "the remixing is bad" predates these fixes,
worth confirming the user is testing against current `main`, not a stale build — that's a
five-minute check before chasing new bugs.

Still open from the same audit (lower severity, not yet addressed): zero behavioral test
coverage on `setlist_planner.py` (the flagship set-sequencing feature has *no tests at all*),
and `dj_analysis.py` recomputing `beat_track`/`chroma_cqt` redundantly across multiple
functions (performance, not correctness, but worth knowing).

---

## 6. Structural gaps that cap quality regardless of bug fixes

From `IMPROVEMENTS.md` / `IMPROVEMENTS_V2.md`, Stage 4 items never started:

- **No global energy-arc optimizer.** `setlist_planner.py` does greedy pairwise transition
  selection only — it cannot see two moves ahead. `docs/DJ_THEORY.md` section 7 frames set
  sequencing as a constrained-optimization problem (TSP-adjacent) specifically *because*
  greedy selection produces locally-fine, globally-incoherent sets. This is plausibly a real
  contributor to "the remix doesn't feel intentional as a whole" even when individual
  transitions are technically correct.
- **No vocal analyzer.** Ties back to finding #1 — `vocal_clash_penalty` is in the documented
  formula and nowhere in the code. Demucs stems already exist in the pipeline; a vocal-presence
  energy ratio per stem (mentioned as a feasible proxy in `docs/DJ_THEORY.md` section 7) is a
  relatively small addition that would close a real, named gap.
- **CUE-DETR / better cue-point detection** — not started, lower priority than the above two.

---

## Recommended order of attack

1. Confirm finding #5 isn't a non-issue (check the user is on current `main`).
2. Root-cause the BPM octave error (#2) — it's upstream of both the stretch-ratio and limiter
   findings, so fixing it may shrink or eliminate #3 and #4 as visible symptoms even before
   either is directly patched.
3. Fix the limiter attack/release split (#4) — self-contained, doesn't depend on anything else,
   directly explains a reproduced clipping defect.
4. Wire `vocal_clash_penalty` and `genre_proximity`/`timbral_similarity` into
   `compatibility_score()` to match `docs/DJ_THEORY.md` (#1) — this changes which pairs get
   recommended in the first place, which is upstream of how any individual transition sounds.
5. Global energy-arc optimizer (#6) is the biggest lift and the least likely to be "why a
   single remix sounds bad" — it's a set-level coherence problem, not a per-transition one.
   Lowest priority of the five unless the complaint is specifically about full-set flow.
