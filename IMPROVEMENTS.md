# RemixMate — Gap Closure Improvements Plan

> **Source:** Gap analysis vs. Spotify/Apple Music mixing pipelines (June 2026)  
> **Principle:** Stage 1 = wire existing code. Stage 2 = new implementations. Stage 3 = research/publication.

---

## Status Key

| Symbol | Meaning |
|--------|---------|
| ✅ Built | Logic exists and is wired |
| 🔌 Unwired | Logic exists, not connected to the render path |
| 🏗 Partial | Infrastructure exists, needs extension |
| ❌ Missing | Not implemented |

---

## Gap → Codebase Mapping

| Gap (from analysis) | Status | Location | Notes |
|---|---|---|---|
| Per-stem LUFS normalization | 🔌 Unwired | `mastering.py:normalize_stems_to_target()` | Exists. Not called in `dj_engine.py:render_stem_blend()` |
| Stem-aware bass-swap transitions | 🔌 Unwired | `dj_engine.py:render()` has it; `render_stem_blend()` does NOT | Bass swap logic lives in `render()` but was omitted from the stem path |
| Real-time key transposition | 🏗 Partial | `key_detection.py:pitch_shift_for_camelot()` computes semitones; no audio pitch-shift function exists | Phase-vocoder available via librosa but not wrapped for pitch shifting |
| Psychoacoustic harmonic compatibility | ❌ Missing | Camelot gating only in `key_detection.py:camelot_modulation()` | Rule-based cost, not Gebhardt sinusoidal-partial consonance |
| Phrase-boundary-aligned cue selection | 🏗 Partial | `dj_analysis.py:plan_transition()` picks sections by type label | Sections exist but cue point picks on label ("outro"), not SSM-novelty bar alignment |
| Global energy-arc planning | 🏗 Partial | `setlist_planner.py` does greedy pairwise | No TSP/global optimizer across full set |
| EDM-tuned key profiles (Essentia EDMA/EDMM) | ❌ Missing | Only Krumhansl-Schmuckler in `key_detection.py` | EDM key detection significantly worse with K-S profiles |
| Per-stem crossfade in chain renders | ❌ Missing | `render_chain()` uses flat cosine; no stem path | Chain render doesn't call `render_stem_blend()` |

---

## Stage 1 — Wire Existing Code (Low risk, high impact)

These are **connection tasks**, not implementation tasks. All the DSP is built.

---

### 1A. Wire per-stem LUFS normalization into `render_stem_blend()`

**File:** `scripts/core/dj_engine.py`  
**Function:** `DJEngine.render_stem_blend()` (line ~1080)  
**Import needed:** `from scripts.core.mastering import normalize_stems_to_target`

**Problem:** `render_stem_blend()` loads stems and stretches them, then sums them without level alignment. A loud bass stem from Song B will crush Song A's softer stems.

**Fix:** After the `stretched_b` dict is built (after time-stretch at line ~1150), normalize all stems to −20 LUFS before entering the crossfade loop.

```python
# Add after stretched_b is populated (~line 1155):
from scripts.core.mastering import normalize_stems_to_target, analyze_stem_lufs

# Build input dicts for normalize function
stems_a_input = {s: (stems_a[s], sr) for s in _STEM_NAMES if stems_a.get(s) is not None}
stems_b_input = {s: (stretched_b[s], sr) for s in _STEM_NAMES if stretched_b.get(s) is not None}

# Normalize each stem to -20 LUFS before mixing (leaves 6 dB headroom for summing 4 stems)
stems_a_norm = normalize_stems_to_target(stems_a_input, target_lufs=-20.0)
stems_b_norm = normalize_stems_to_target(stems_b_input, target_lufs=-20.0)

# Then replace stems_a[s] / stretched_b[s] lookups in the loop with stems_a_norm[s] etc.
```

**Test:** Write a test that creates stems with 10 dB level difference, runs `render_stem_blend()`, and asserts the output doesn't clip and the transition doesn't have a level jump. Add to `tests/test_core_modules.py`.

**Effort:** ~1 hour (wiring + test)

---

### 1B. Wire bass-swap into `render_stem_blend()`

**File:** `scripts/core/dj_engine.py`  
**Function:** `DJEngine.render_stem_blend()` — the per-stem crossfade loop (line ~1184)

**Problem:** `render_stem_blend()` applies the same `_stem_crossfade_curves()` to the bass stem as to vocals/other. The bass stem from Song A should exit early (at the swap point) and Song B's bass should enter clean — identical to what `render()` does with `_butter_highpass` at line ~793.

**Fix:** In the per-stem loop, treat the `"bass"` stem as a special case:

```python
for s in _STEM_NAMES:
    fade_out, fade_in = _stem_crossfade_curves(trans_samples, similarities[s])
    
    # Bass stem: force early exit for A, clean entry for B (bass swap)
    if s == "bass":
        swap_sample = int(plan.eq.bass_swap_bar * (trans_samples / max(plan.transition_bars, 1)))
        swap_sample = min(swap_sample, trans_samples)
        # A's bass fades to zero by swap_sample
        fade_out = np.concatenate([
            np.linspace(1.0, 0.0, swap_sample, dtype=np.float32),
            np.zeros(trans_samples - swap_sample, dtype=np.float32)
        ])
        # B's bass enters clean from zero at swap_sample
        fade_in = np.concatenate([
            np.zeros(swap_sample, dtype=np.float32),
            np.linspace(0.0, 1.0, trans_samples - swap_sample, dtype=np.float32)
        ])
```

**Why this matters:** This is the core "confirmed gap" from the analysis. Spotify/Apple apply whole-track level normalization; RemixMate will be the only pipeline doing per-stem normalization AND per-stem time-controlled bass handoff.

**Effort:** ~1.5 hours (logic + test)

---

### 1C. Wire `render_stem_blend()` into `render_chain()`

**File:** `scripts/core/dj_engine.py`  
**Function:** `DJEngine.render_chain()` (line ~860)

**Problem:** `render_chain()` uses flat cosine fades without stem awareness. It never calls `render_stem_blend()`.

**Fix:** Add optional `stems_dirs` parameter to `render_chain()`. When provided, call `render_stem_blend()` instead of the inline crossfade per transition pair.

```python
def render_chain(
    self,
    tracks: list,
    plans: list,
    bridge_beats: Optional[list] = None,
    bridge_gain: float = 0.38,
    transition_effect: str = "auto",
    stems_dirs: Optional[list] = None,   # NEW: list of N stem directories
) -> np.ndarray:
```

**Effort:** ~2 hours (refactor + test chain with stems)

---

## Stage 2 — New Implementations (Medium complexity)

---

### 2A. Real-time key transposition for harmonic match

**New function to add:** `scripts/core/key_detection.py` — `pitch_shift_audio(audio, sr, semitones)`  
**Uses:** `librosa.effects.pitch_shift()` or the GPU path in `scripts/core/gpu.py`

**What already exists:** `pitch_shift_for_camelot(source, target)` at line 529 already computes the semitone delta. The audio pipeline just never applies it.

**Where to wire it:** `scripts/api/task_modules/remix.py` — in the DJ remix task, after loading track_b audio, check `plan.eq` for key compatibility. If harmonic_score < 0.35 AND semitone shift is ≤ 2, apply pitch shift before passing to `engine.render()`.

```python
# In scripts/core/key_detection.py — add this function:
def pitch_shift_audio(audio: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    """
    Pitch-shift audio by semitones without changing tempo.
    Uses librosa phase-vocoder (or GPU path if available).
    Clamped to ±6 semitones to prevent quality degradation.
    """
    if semitones == 0:
        return audio
    semitones = int(np.clip(semitones, -6, 6))
    try:
        from scripts.core.gpu import gpu_pitch_shift
        return gpu_pitch_shift(audio, sr=sr, n_steps=semitones)
    except (ImportError, Exception):
        import librosa
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(semitones))
```

**Wire it in `plan_transition()`** (`dj_analysis.py`): add a `suggested_pitch_shift` field to `TransitionPlan` using `pitch_shift_for_camelot()`. Then the remix task can decide whether to apply it.

**New `TransitionPlan` field:**
```python
@dataclass
class TransitionPlan:
    ...
    suggested_pitch_shift: int = 0   # semitones to shift B for harmonic match
```

**Logic for when to apply:** Only if Camelot distance > 1 AND shift ≤ 2 semitones. Larger shifts degrade quality more than a harmonic mismatch would.

**Effort:** ~3 hours (function + wiring + quality test)

---

### 2B. SSM-novelty phrase-boundary cue selection

**File:** `scripts/core/dj_analysis.py`  
**Function:** `plan_transition()` — cue point selection logic

**Problem:** Current code picks exit/entry cue points by section type label (`"outro"`, `"break"`). This doesn't guarantee an 8/16-bar phrase boundary — it just picks a structurally labeled region. Apple's AutoMix does the same thing (fires at track end), which is why user reports say "it mixes at the wrong time."

**What to add:** Use `librosa.segment.recurrence_matrix()` and novelty-curve peak detection to find true structural boundaries that align to the 8/16-bar grid.

```python
# Add to dj_analysis.py — new helper:
def _detect_phrase_boundaries(audio: np.ndarray, sr: int, bpm: float) -> List[float]:
    """
    Detect structural boundaries aligned to 8-bar grid using SSM novelty curves.
    Returns list of boundary times in seconds.
    
    Method: Vande Veire & De Bie 2018 — MFCC+RMS SSM, geometric mean,
    novelty curve peak-picking, snapped to nearest downbeat.
    """
    import librosa
    
    # Compute MFCC feature matrix
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=audio)
    features = np.vstack([mfcc, rms])
    
    # Self-similarity matrix
    R = librosa.segment.recurrence_matrix(features, mode='affinity', sym=True)
    
    # Novelty curve
    novelty = librosa.segment.timelag_filter(R)
    novelty_curve = np.mean(novelty, axis=0)
    
    # Find peaks in novelty curve
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(novelty_curve, distance=int(sr * 8 * 60.0 / bpm / 512))
    
    # Convert frame indices to seconds
    times = librosa.frames_to_time(peaks, sr=sr)
    
    # Snap each boundary to nearest 8-bar downbeat
    bar_sec = 4 * 60.0 / bpm
    grid_8bar = bar_sec * 8
    snapped = [round(t / grid_8bar) * grid_8bar for t in times]
    
    return sorted(set(snapped))
```

**Wire into `plan_transition()`:** Replace the section-label exit point selection with the nearest phrase boundary from `_detect_phrase_boundaries()` that falls after 50% of the track duration.

**Effort:** ~4 hours (implementation + integration + listening test on sample tracks)

---

### 2C. Essentia EDMA/EDMM key profiles

**File:** `scripts/core/key_detection.py`

**Problem:** Krumhansl-Schmuckler profiles (`_MAJOR_PROF`, `_MINOR_PROF`) were designed for classical music tonal perception. They underperform on EDM because electronic music has a different pitch-class distribution (more 1-5-8 power chords, fewer leading tones).

**What to add:** Essentia's EDM-tuned profiles (EDMA = average EDM profile, EDMM = combined). These are published and available in the Essentia source.

```python
# Add to key_detection.py alongside existing K-S profiles:

# Essentia EDMA profile (Faraldo et al., ECIR 2016 — tuned on EDM)
_EDMA_MAJOR_PROF = np.array([
    0.1294, 0.0428, 0.0778, 0.0313, 0.1003, 0.0783,
    0.0340, 0.1208, 0.0322, 0.0775, 0.0290, 0.0463
], dtype=np.float64)

_EDMA_MINOR_PROF = np.array([
    0.1273, 0.0308, 0.0686, 0.0991, 0.0283, 0.0786,
    0.0319, 0.1122, 0.0638, 0.0267, 0.0595, 0.0737
], dtype=np.float64)

# Normalize
_EDMA_MAJOR_PROF /= _EDMA_MAJOR_PROF.sum()
_EDMA_MINOR_PROF /= _EDMA_MINOR_PROF.sum()
```

**Modify `detect_key()`** to accept a `profile` parameter:

```python
def detect_key(audio, sr, method='cqt', profile='auto') -> KeyResult:
    """
    profile: 'ks' = Krumhansl-Schmuckler (classical/pop)
             'edma' = Essentia EDMA (EDM)
             'auto' = use EDMA if spectral centroid > 4kHz (EDM heuristic), else K-S
    """
```

**Effort:** ~2 hours (profiles + auto-selection heuristic + benchmark test on sample tracks)

---

### 2D. Psychoacoustic harmonic compatibility score

**File:** `scripts/core/key_detection.py` — new function alongside `camelot_modulation()`

**What this replaces/supplements:** The current `harmonic_score` in `TransitionPlan` is computed in `plan_transition()` and is just a Camelot distance rescaled to [0,1]. This is a binary gate, not a continuous perceptual measure.

**What to add:** Gebhardt/Davies/Seeber 2016 consonance score — compute the spectral partials of each track's most prominent pitches, then measure roughness between them using the Sethares roughness model.

```python
# Add to key_detection.py:
def psychoacoustic_consonance(chroma_a: np.ndarray, chroma_b: np.ndarray) -> float:
    """
    Compute psychoacoustic consonance between two chroma vectors.
    
    Based on Gebhardt, Davies & Seeber (Applied Sciences, 2016):
    computes pairwise roughness between sinusoidal partials derived
    from the top-N chroma peaks of each track.
    
    Returns consonance score [0.0, 1.0]:
      1.0 = maximally consonant (unison/octave)
      0.0 = maximally dissonant (tritone/minor second)
    """
    # Top 3 pitches from each chroma vector
    top_a = np.argsort(chroma_a)[-3:]
    top_b = np.argsort(chroma_b)[-3:]
    
    # Sethares roughness between all pairs
    total_roughness = 0.0
    n_pairs = 0
    for i in top_a:
        for j in top_b:
            semitone_diff = abs(int(i) - int(j)) % 12
            # Roughness lookup (Plomp & Levelt 1965 approximation)
            # 0 semitones = consonant (0.0 roughness), 6 = tritone (1.0)
            roughness_table = [0.0, 1.0, 0.8, 0.4, 0.3, 0.2, 1.0,
                               0.2, 0.3, 0.4, 0.8, 1.0]
            total_roughness += roughness_table[semitone_diff]
            n_pairs += 1
    
    if n_pairs == 0:
        return 0.5
    
    avg_roughness = total_roughness / n_pairs
    return float(1.0 - avg_roughness)
```

**Wire into `plan_transition()`:** Use `psychoacoustic_consonance(struct_a.chroma_vector, struct_b.chroma_vector)` to compute `harmonic_score` instead of the current Camelot distance formula. `SongStructure` needs a `chroma_vector` field (add it to the dataclass and populate it in `analyze_structure()`).

**Effort:** ~3 hours (score + wiring + compare against Camelot on test tracks)

---

## Stage 3 — Research / Publication Targets

---

### 3A. Benchmark Spotify/Apple key detection vs. GiantSteps ground truth

**What:** Run RemixMate's `detect_key()` AND (using tags scraped from the public catalog) Spotify's legacy echo nest tags against the GiantSteps EDM dataset (604 Beatport tracks, hand-annotated). Compare weighted MIREX scores.

**Why it's publishable:** No peer-reviewed paper exists that directly benchmarks Spotify or Apple key tags against GiantSteps. The gap analysis confirms this. This is a 4-page ISMIR/ICASSP paper.

**How:**
```
scripts/
└── benchmarks/
    ├── giantsteps_eval.py      # Download GiantSteps, run detect_key(), score
    ├── spotify_tag_compare.py  # Compare against Spotify key integers (if accessible)
    └── results/
        └── key_accuracy_2026.json
```

**Target metric:** Beat the ~74.3% weighted MIREX score of Korzeniowski & Widmer (2017) on GiantSteps by combining EDMA profiles (Stage 2C) with HPSS preprocessing (already in `detect_key()`).

---

### 3B. Paper framing: stem-aware mixing pipeline

**Title candidate:** "Beyond Beat-Matching: Per-Stem LUFS-Normalized, Phrase-Aligned DJ Transitions"

**Novel contributions from this improvement plan:**
1. Per-stem LUFS normalization before crossfade (Stage 1A) — confirmed absent from Spotify/Apple
2. Per-stem bass handoff at harmonic boundary (Stage 1B) — confirmed absent
3. Phrase-boundary-aligned cue selection via SSM novelty (Stage 2B) — Apple detects phrases but fires at track end; RemixMate fires at musical boundary
4. Psychoacoustic consonance score as continuous harmonic compatibility metric (Stage 2D) — replaces Camelot binary gate

**Contrast with Apple Music Understanding framework (WWDC 2026):** Apple exposes instrument activity and phrases as *analysis outputs* for video editors. RemixMate uses equivalent data as *control signals for mixing*. That distinction is the contribution.

---

## Implementation Order

```
Week 1: Stage 1 (wiring)
  Day 1-2:  1A — per-stem LUFS normalization in render_stem_blend()
  Day 3:    1B — bass stem swap logic in render_stem_blend()
  Day 4-5:  1C — stem path in render_chain() + integration test

Week 2: Stage 2 (new code)
  Day 1-2:  2A — pitch_shift_audio() + TransitionPlan.suggested_pitch_shift
  Day 3-4:  2B — SSM-novelty phrase boundaries in plan_transition()
  Day 5:    2C — EDMA key profiles + auto-profile selection

Week 3: Stage 2 continued + Stage 3 setup
  Day 1-2:  2D — psychoacoustic_consonance() + wire into harmonic_score
  Day 3-5:  3A — GiantSteps benchmark script + run evaluation
```

---

## Quick Wins Checklist

- [ ] **1A** — `normalize_stems_to_target()` called in `render_stem_blend()` before crossfade loop
- [ ] **1B** — bass stem uses swap-point logic instead of `_stem_crossfade_curves()`
- [ ] **1C** — `stems_dirs` param added to `render_chain()`
- [ ] **2A** — `pitch_shift_audio()` added to `key_detection.py`; `suggested_pitch_shift` field on `TransitionPlan`
- [ ] **2B** — `_detect_phrase_boundaries()` replaces section-label cue selection in `plan_transition()`
- [ ] **2C** — EDMA profiles in `key_detection.py`; `detect_key(profile='auto')`
- [ ] **2D** — `psychoacoustic_consonance()` replaces Camelot-distance `harmonic_score`
- [ ] **3A** — GiantSteps benchmark script at `scripts/benchmarks/giantsteps_eval.py`

---

## Key Reference Files

| Module | Relevant Functions | Gap being closed |
|---|---|---|
| `scripts/core/dj_engine.py` | `render_stem_blend()`, `render_chain()` | 1A, 1B, 1C |
| `scripts/core/mastering.py` | `normalize_stems_to_target()`, `analyze_stem_lufs()` | 1A (already built) |
| `scripts/core/key_detection.py` | `detect_key()`, `pitch_shift_for_camelot()`, `camelot_modulation()` | 2A, 2C, 2D |
| `scripts/core/dj_analysis.py` | `plan_transition()`, `analyze_structure()`, `SongStructure` | 2B, 2D |
| `scripts/core/setlist_planner.py` | `SetlistPlanner.optimize()` | (Stage 2 extension) |
| `scripts/api/task_modules/remix.py` | DJ remix task | 2A wiring point |

---

*Last updated: June 26, 2026 — Initial gap-to-code mapping*
