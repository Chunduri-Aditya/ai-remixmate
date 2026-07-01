# AI RemixMate — Improvements V2 Roadmap
> Generated: June 29, 2026 | Based on research report + June 28 codebase audit

## Codebase Delta Audit (What the Report Says vs What's Actually Built)

Before staging any work, here's the honest gap between the research report's recommendations
and the current `main` branch state. This prevents re-implementing things that already exist.

| Gap / Module | Report Says | Codebase Reality (June 28) | True Delta |
|---|---|---|---|
| Gap 1: Per-stem LUFS | pyloudnorm + corpus-derived per-stem-type targets (FxNorm scheme) | `normalize_stems_to_target()` exists but uses **one fixed -20 LUFS for all stem types** | Upgrade to per-stem-type targets derived from library corpus |
| Gap 2: Stem bass-swap | True stem-level muting at phrase boundary | EQ IIR shelving filter approximation in `_apply_dynamic_eq_fade` | Replace with Demucs bass-stem duck/zero at phrase boundary |
| Gap 3: Key transposition | pyrubberband pitch shift | `pitch_shift_audio()` wired into `render_stem_blend()` | **DONE — skip** |
| Gap 4: TIV harmonic scoring | TIVlib continuous Tonal Interval Space scoring | Custom Plomp-Levelt roughness formula in `psychoacoustic_consonance()` | Vendor TIVlib, add continuous TIS score alongside existing formula |
| Gap 5: ML cue selection | Beat This! downbeats + 8/16-bar snapping (route i) | librosa beat_track + SSM-novelty boundaries (phrase not bar-grid aligned) | (a) Beat This! integration; (b) snap phrase boundaries to exact bar grid |
| Module: Crate Digger | CLAP 512-D + FAISS | `music_index.py` with 35-D handcrafted embeddings | New module: CLAP embeddings replace the 35-D index |
| Module: Energy Planner | Essentia arousal regression + arc cost term | Greedy Markov setlist optimizer with static energy field | Add Essentia feature extraction + arc deviation cost |
| Module: Vocal Analyzer | CREPE F0 + vibrato/accuracy metrics | Nothing | New module (Stage 4) |
| Module: B-Roll Engine | CLIP + CLAP multimodal alignment | Nothing | New module (Stage 4) |

**Conclusion:** Gaps 1, 2, 4, 5 need real upgrade work. Gap 3 is done. The three Stage 1–3
modules are net-new. The research report's staging advice is correct.

---

## Strategic Context (Why This Order Matters for PhD + Portfolio)

PhD applications in audio ML / AI safety care about:
1. **Depth of MIR foundations** — TIVlib scoring + Beat This! + CUE-DETR show you know the
   literature and can implement it, not just call librosa.
2. **Evaluation rigour** — formal listening test (even 10 mixes, 15 listeners) is what separates
   an ISMIR submission from a GitHub repo.
3. **Novelty combination** — TIV harmonic + stem bass-swap + ML cue snapping as one integrated
   system is the paper angle (Stages 1–2 combined).

GitHub stars / DJ adoption care about:
1. **CLAP-powered search** — DJs actually use this (find me something that sounds like X).
2. **Energy arc planner** — visible, demo-able, immediately useful.
3. **rekordbox/Serato cue export** — removes the gap between RemixMate output and a real gig.

Ship Stages 1–2 first to solidify the research claim; Stage 3 in parallel for portfolio surface area.

---

## Stage 1 — Analysis Foundation (Week 1–2)
**Goal:** Raise per-stem mix quality and harmonic intelligence. No new deps beyond pyloudnorm + TIVlib.
**Threshold to continue:** per-stem normalization passes A/B on ≥10 test mixes; TIV score correlates
with Camelot adjacency on 24 known-compatible pairs (correlation > 0.8).

### 1A — Per-Stem-Type LUFS Normalization (FxNorm Scheme)

**What's wrong now:** `normalize_stems_to_target(stems, target_lufs=-20.0)` applies the same
-20 LUFS to drums, bass, vocals, and other — but these have wildly different frequency content
and perceived loudness. FxNorm-Automix (Sony, ISMIR 2022) derives a **per-stem-type** target
from a reference corpus (mean LUFS across all tracks' drums stems, all bass stems, etc.).

**What to build:**

```
scripts/core/mastering.py
  - analyze_library_stem_targets(library_dir: Path) -> dict[str, float]
      Iterate all song dirs, measure LUFS for each stem type found,
      return {"drums": -18.2, "bass": -22.1, "vocals": -19.5, "other": -21.3}.
      Cache result to data/stem_lufs_targets.json.

  - normalize_stems_to_corpus_targets(
        stems: dict[str, np.ndarray],
        sr: int,
        targets: dict[str, float],        # from analyze_library_stem_targets()
        fallback_lufs: float = -20.0,
    ) -> dict[str, np.ndarray]
      Per-stem normalization using corpus-derived targets.
      Falls back to fallback_lufs for stem types not in targets.
      Applies true-peak limiting at -1 dBTP per stem before returning.

scripts/core/dj_engine.py
  - In render_stem_blend(): replace normalize_stems_to_target() call with
    normalize_stems_to_corpus_targets(), loading targets from data/stem_lufs_targets.json
    or falling back to analyze_library_stem_targets() if cache missing.

scripts/api/routers/library.py (or system.py)
  - POST /library/calibrate-lufs — async job that runs analyze_library_stem_targets()
    and writes data/stem_lufs_targets.json. Returns job_id.
```

**Dependency:** `pyloudnorm` (already in mastering.py? verify) — install: `pip install pyloudnorm`

**Acceptance tests (`tests/test_mastering.py`):**
- `test_stem_type_targets_differ()` — drums target ≠ bass target after analyze_library_stem_targets
- `test_normalization_hits_target_lufs()` — each stem within ±1 LU of its corpus target
- `test_silent_stem_no_crash()` — silent stem returns silent (no NaN/inf)
- `test_true_peak_applied()` — output peak ≤ -1 dBTP per stem

**Effort:** 1.5 days | **Value:** Portfolio H / Novelty M / Adoption M

---

### 1B — TIVlib Tonal Interval Space Harmonic Scoring

**What's wrong now:** `psychoacoustic_consonance()` uses a Plomp-Levelt roughness model
(custom formula over partial frequencies). This is a reasonable proxy but is NOT the
Tonal Interval Space / TIVlib approach cited in the research literature and not directly
comparable to published benchmarks.

**What to build:**

```
third_party/TIVlib/                      # git submodule or vendored clone
  (clone from https://github.com/GramAx/TIVlib)

scripts/core/tiv_scoring.py              # new module
  - tiv_from_chroma(chroma: np.ndarray) -> Any
      Wrap tiv.TIV.from_pcp(chroma) — converts 12-bin HPCP to TIV vector.

  - tiv_harmonic_score(
        chroma_a: np.ndarray,     # shape (12,) — mean chroma over segment
        chroma_b: np.ndarray,
        weights: dict | None = None,    # default: dissonance 0.402, hierarchical 0.246,
                                        # tonal distance 0.202, voice leading 0.193
    ) -> float
      Returns 0.0 (incompatible) → 1.0 (perfectly compatible).
      Uses TIS angular distance: score = 1 - (angle / π).

  - compare_tiv_vs_camelot(
        chroma_a, camelot_a, chroma_b, camelot_b
    ) -> dict
      Returns {"tiv_score": float, "camelot_adjacent": bool, "camelot_distance": int}
      for validation.

scripts/core/dj_analysis.py
  - In plan_transition(): add tiv_score to TransitionPlan output when chroma vectors
    are available from analyze_structure(). Blend with psychoacoustic_consonance()
    as a weighted average (default: 0.5/0.5) until TIV is validated.
  - TransitionPlan dataclass: add tiv_compatibility: float | None = None

scripts/core/key_detection.py
  - Existing psychoacoustic_consonance() stays as fallback when chroma unavailable.
    Add NOTE in docstring pointing to tiv_scoring.py as the preferred path.
```

**TIVlib install note:** NOT on PyPI. Clone and add to PYTHONPATH or vendor into `third_party/`.
Add to `start.sh`: `export PYTHONPATH="$PYTHONPATH:$(pwd)/third_party/TIVlib"`.

**Acceptance tests (`tests/test_tiv_scoring.py`):**
- `test_same_key_max_score()` — C major vs C major = 1.0
- `test_adjacent_camelot_high_score()` — known adjacent pairs (8B→9B) score > 0.7
- `test_tritone_low_score()` — tritone substitution (8B→2B) score < 0.3
- `test_correlation_with_camelot_distance()` — pearson r > 0.8 on all 24×24 pairs
- `test_output_in_unit_range()` — all pairs return [0, 1]
- `test_fallback_when_tivlib_missing()` — graceful import error → returns None

**Effort:** 2 days | **Value:** Portfolio H / Novelty H / Adoption M | **Publishable:** Yes (DAFx/ISMIR)

---

## Stage 2 — The Publishable Core (Week 3–6)
**Goal:** Formal evaluation-ready system: better downbeats, bar-aligned cues, true stem bass-swap.
**Threshold to escalate:** Run the formal listening test (design below). If statistically significant
listener preference vs current crossfade, submit to DAFx 2027 or ISMIR 2027.

### 2A — Beat This! Integration (Better Downbeats)

**What's wrong now:** librosa beat_track gives beat times but downbeat detection is weak —
it doesn't reliably identify which beat is beat 1 of each bar. SSM phrase boundaries therefore
can't snap to the bar grid correctly.

**What to build:**

```
scripts/core/beat_tracker.py             # new module — swappable backend
  class BeatTracker(Protocol):
      def track(self, audio, sr) -> BeatResult:
          ...  # returns beats, downbeats, bpm

  @dataclass
  class BeatResult:
      beat_times: np.ndarray
      beat_frames: np.ndarray
      downbeat_times: np.ndarray        # new: beat 1 of each bar
      bpm: float

  class LibrosaBeatTracker:             # existing backend, wrapped
      def track(self, audio, sr) -> BeatResult: ...

  class BeatThisTracker:               # NEW — Beat This! (ISMIR 2024)
      # pip install beat-this
      # or: from beat_this.inference import File2Beats
      def track(self, audio, sr) -> BeatResult: ...

  def get_tracker(backend: str = "librosa") -> BeatTracker:
      # "librosa" | "beat_this" — reads from config.yaml: analysis.beat_backend

scripts/core/dj_analysis.py
  - _analyze_impl(): replace bare librosa.beat.beat_track() with get_tracker(cfg).track()
  - _detect_phrase_boundaries(): use BeatResult.downbeat_times for 8/16-bar snapping (see 2B)
  - SongStructure: add downbeat_times: list[float] = field(default_factory=list)
```

**Install:** `pip install beat-this` (ISMIR 2024, ~50 MB model, ~1s inference on CPU for 3min track)

**Acceptance tests (`tests/test_beat_tracker.py`):**
- `test_librosa_backend_returns_beat_result()` — correct dtype, non-empty
- `test_both_backends_same_bpm_approx()` — on synthetic 120 BPM metronome, both within 1 BPM
- `test_beat_this_downbeats_on_4_4()` — downbeat count ≈ total_beats / 4 within 10%
- `@pytest.mark.dj_analysis` guard on beat-this tests (model dependency)

**Effort:** 2 days | **Value:** Portfolio M / Novelty M / Adoption M

---

### 2B — Bar-Grid Cue Point Snapping

**What's wrong now:** `_detect_phrase_boundaries()` uses SSM novelty — it finds structurally
interesting moments, but they're not snapped to bar boundaries. A DJ never drops a track
mid-bar; cues must land on downbeat 1 of bars 8, 16, 32, etc.

**What to build:**

```
scripts/core/dj_analysis.py
  - _snap_to_bar_grid(
        boundary_times: list[float],
        downbeat_times: np.ndarray,
        preferred_lengths_bars: list[int] = [8, 16, 32],
    ) -> list[float]
      For each boundary time, find the nearest downbeat that is also a multiple
      of preferred_lengths_bars from the previous boundary.
      Returns bar-aligned boundary times.

  - _detect_phrase_boundaries(): after peak-picking novelty curve, call
    _snap_to_bar_grid(raw_boundaries, downbeat_times) before returning.
    Pass downbeat_times from BeatResult (requires 2A).

  - plan_transition(): use snapped phrase boundaries for exit_bar_a / entry_bar_b selection.
    Currently uses raw bars[]; upgrade to prefer boundaries in struct.phrase_boundaries
    that align to 8/16-bar grid.
```

**Acceptance tests (`tests/test_dj_analysis.py`):**
- `test_snap_returns_downbeat_aligned_times()` — all snapped times are within 50ms of a downbeat
- `test_snap_prefers_8_bar_multiples()` — on synthetic 4-bar downbeats, snapped cue at bar 8
- `test_snap_empty_boundaries_returns_empty()` — no crash on empty input

**Effort:** 1.5 days | **Value:** Portfolio H / Novelty H / Adoption H (this is the biggest audible quality improvement)

---

### 2C — True Stem-Level Bass Muting at Phrase Boundary

**What's wrong now:** The bass-swap in `_apply_dynamic_eq_fade()` applies an IIR shelving filter
to the full mixed audio — it's an approximation that bleeds mid frequencies and doesn't achieve
true bass isolation. When Demucs stems are available, the correct approach is to duck/zero
the actual `bass` stem of the outgoing track at the swap point.

**What to build:**

```
scripts/core/dj_engine.py
  - render_stem_blend(): at bass_swap_sample, instead of relying on EQ fade:
      1. For samples < bass_swap_sample: Song A bass stem at full level, Song B bass = 0
      2. For samples ≥ bass_swap_sample: Song B bass stem at full level, Song A bass = 0
      Apply a 4-bar linear ramp (not hard cut) centered on bass_swap_sample for click-free swap.

  - _stem_bass_ramp(
        stem_a_bass: np.ndarray,
        stem_b_bass: np.ndarray,
        swap_sample: int,
        ramp_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]
      Returns (a_bass_faded, b_bass_faded) with smooth crossover.
      Replaces the low-shelf IIR approach in the bass stem path only.

  - Keep _apply_dynamic_eq_fade() for the fallback path (no stems available).
    Document: "stems-available path uses _stem_bass_ramp(); stems-unavailable path
    uses EQ approximation."
```

**Acceptance tests (`tests/test_behavioral.py` — add to existing file):**
- `test_bass_stem_muted_before_swap()` — in stems render, bass energy in first half < 0.01 RMS
- `test_bass_stem_active_after_swap()` — bass energy in second half > 0.1 RMS
- `test_no_click_at_swap()` — max amplitude delta at swap_sample < 0.05 (smooth transition)
- `test_fallback_eq_path_unchanged()` — no-stems path still passes existing B-silence test

**Effort:** 2 days | **Value:** Portfolio H / Novelty H / Adoption H

---

### 2D — Formal Listening Test Infrastructure

**What's build:** Not a module — a reproducible evaluation harness for ISMIR/DAFx submission.

```
scripts/benchmarks/listening_test.py
  - generate_ab_pairs(
        song_pairs: list[tuple[str, str]],
        output_dir: Path,
        conditions: list[str] = ["baseline", "stem_aware", "tiv_harmonic"],
    ) -> dict
      For each song pair, render a mix under each condition.
      Output: output_dir/{pair_id}/{condition}.wav (normalized to -14 LUFS for fair comparison).
      Returns a JSON manifest for survey assembly.

  - render_baseline(track_a, track_b, plan) -> np.ndarray
      Current crossfade (no stems, Camelot-only).

  - render_stem_aware(track_a, track_b, stems_a, stems_b, plan) -> np.ndarray
      Full stem blend + bass muting + TIV-informed plan.

  Listening test protocol (document in README):
    - 10 track pairs (5 same-genre, 5 cross-genre)
    - 15 listeners (can use Prolific for paid panel or music students)
    - MUSHRA-style or A/B preference: "Which mix sounds more professional?"
    - Metrics: % preference, 95% CI (binomial), Cohen's κ for inter-rater agreement
    - Ground truth: compare cue placement error vs manually annotated cue points

```

**Effort:** 1 day (scaffolding) + listening study time (~1 week async)
**Threshold:** ≥65% preference for stem-aware condition → paper submission.

---

## Stage 3 — Portfolio & Adoption Drivers (Parallel with Stage 2)

### 3A — Crate Digger Assistant (CLAP + FAISS)

**The problem with the current index:** `music_index.py` uses 35 handcrafted dimensions
(BPM, key, energy, spectral features). CLAP (LAION Contrastive Language-Audio Pretraining)
embeds audio and text into a shared 512-D space, enabling queries like "find me something
dark and hypnotic like this" — 71.9% agreement with human listeners on similarity.

**What to build:**

```
scripts/core/crate_digger.py            # new module
  class CrateDigger:
      CLAP_MODEL = "laion/clap-htsat-unfused"   # ~300 MB

      def __init__(self, library_dir: Path, index_path: Path):
          # lazy-load CLAP model (laion_clap or transformers)
          ...

      def index_library(self, progress_cb=None) -> None
          """Compute CLAP embeddings for all songs; persist to index_path (.npz)."""
          # 10s segment at the midpoint of each track; 50% overlap; mean-pool
          # Save: {name: embedding_512d} as .npz

      def find_similar(
          self,
          query_name: str | None = None,    # by song name (look up its embedding)
          query_audio: np.ndarray | None = None,  # or raw audio
          query_text: str | None = None,    # or text description
          k: int = 10,
          camelot_filter: str | None = None,  # optional harmonic filter
          bpm_range: tuple[float, float] | None = None,
      ) -> list[dict]
          """Return k most similar tracks with similarity score + metadata."""

      def _clap_embed_audio(self, audio: np.ndarray, sr: int) -> np.ndarray
      def _clap_embed_text(self, text: str) -> np.ndarray

scripts/api/routers/library.py
  - GET /library/crate-search?query_name=X&query_text=Y&k=10&camelot=8B
      Returns CrateDigger.find_similar() results.
  - POST /library/build-clap-index
      Async job: CrateDigger.index_library() with progress updates via SSE.

frontend/src/pages/LibraryAtlas.tsx
  - Add a "Crate Digger" search mode: text query OR click-to-query from existing track.
  - Show CLAP similarity score alongside the 35-D cosine score.
```

**Install:** `pip install laion_clap` or via `transformers` (`AutoModel.from_pretrained("laion/clap-htsat-unfused")`)

**Acceptance tests (`tests/test_crate_digger.py`):**
- `test_find_similar_returns_k_results()` — k=5 returns exactly 5 entries
- `test_same_song_highest_similarity()` — query by song returns itself as rank 1 with score ≈ 1.0
- `test_text_query_no_crash()` — text="dark techno bass heavy" returns valid results
- `test_camelot_filter_applied()` — all returned tracks match camelot=8B when filter set
- `test_index_persistence()` — save and reload index; same top-5 results

**Effort:** 3–4 days | **Value:** Portfolio H / Novelty M / **Adoption H** | Demo-able in 30s

---

### 3B — DJ Set Energy Arc Upgrade (Essentia Features)

**What's wrong now:** `SetlistPlanner` uses `track.energy` — a float from Spotify/Exportify.
The research says to compute arousal/valence from actual audio (Essentia features beat handcrafted
for valence) and optimize against a formal energy-arc cost term.

**What to build:**

```
scripts/core/energy_profiler.py         # new module
  def extract_essentia_features(
      audio: np.ndarray, sr: int
  ) -> dict[str, float]:
      """
      Compute arousal-predictive features via essentia.standard:
        - RMS energy, dynamic range
        - Spectral centroid, rolloff, flux (brightness proxy)
        - Onset rate (tempo percept)
        - Loudness (ITU, EBU R128 integrated)
      Returns flat dict of float features.
      Falls back to np-only proxy if essentia not installed.
      """

  def predict_arousal(features: dict) -> float:
      """
      Linear regression over essentia features → arousal [0, 1].
      Coefficients from DEAM (MediaEval Dynamic Annotations of Music) literature.
      Returns a score in [0, 1] where 1 = maximum perceived energy.
      """

scripts/core/setlist_planner.py
  - TrackNode: add arousal_predicted: float | None = None
  - transition_cost(): replace raw energy_delta with:
        if track.arousal_predicted is not None:
            energy_cost = |b.arousal_predicted - ideal_energy| * 0.6 + |b.arousal_predicted - a.arousal_predicted| * 0.4
  - SetlistPlanner.optimize(): if essentia available, compute arousal for tracks without it

scripts/api/task_modules/analysis.py
  - After Demucs separation, extract essentia features and store in song metadata.
```

**Effort:** 2 days | **Value:** Portfolio H / Novelty M / Adoption H

---

### 3C — rekordbox / Serato Cue Point Export

**What's wrong now:** RemixMate generates optimized cue points and phrase boundaries
internally but there's no way to import them into actual DJ software.

**What to build:**

```
scripts/core/cue_export.py              # new module
  def export_rekordbox_xml(
      song_name: str,
      cue_points: list[float],          # seconds
      phrase_boundaries: list[float],   # seconds
      bpm: float,
      output_path: Path,
  ) -> None:
      """
      Writes a rekordbox-compatible XML file with HOT CUE markers.
      Format: <TEMPO Bpm="128.0" ...><POSITION_MARK Type="0" Start="X.XXX" .../>
      Compatible with: rekordbox 6+, DJay Pro (via import), VirtualDJ
      """

  def export_serato_markers(
      audio_path: Path,
      cue_points: list[float],
      output_path: Path | None = None,  # None = write ID3 tags into audio_path
  ) -> None:
      """
      Write Serato cue markers as ID3 GEOB tags using mutagen.
      Works with .mp3 files only; .wav requires AIFF workaround.
      """

scripts/api/routers/analysis.py
  - GET /library/{name}/export-cues?format=rekordbox|serato
    Returns the XML or points to the tagged file.
```

**Install:** `pip install mutagen` (for Serato ID3 writing)

**Effort:** 1.5 days | **Value:** Portfolio M / Novelty L / **Adoption H** (DJs immediately use this)

---

## Stage 4 — Research Depth for PhD Applications (Weeks 8–12)

> Only invest here after Stages 1–2 confirm the MIR foundations are solid and a listening test
> is being prepared. These are the items that differentiate RemixMate from every other portfolio
> project in audio ML.

### 4A — Vocal Performance Analyzer (CREPE + F0 Analysis)

```
scripts/core/vocal_analyzer.py
  Inputs: vocal stem (np.ndarray from Demucs)
  Outputs: VocalReport {
    mean_pitch_hz: float,
    pitch_std_cents: float,       # < 50 cents → accurate intonation
    vibrato_rate_hz: float | None,   # typical 5–7 Hz
    vibrato_extent_cents: float | None,
    phrase_count: int,
    avg_phrase_duration_s: float,
    energy_dynamics_db: float,    # p95 - p5 loudness range
  }

Tools: CREPE (pip install crepe) for F0, scipy for vibrato detection (FFT of F0 trajectory)
Research: Nakano et al. Interspeech 2006; mir_eval for pitch comparison
```

**Why it matters for PhD:** Applies MIR signal processing (F0 tracking, vibrato analysis) to
commercial music stems — underexplored angle. Publishable at an ISMIR workshop.

**Effort:** 3 days | **Value:** Portfolio H / **Novelty H** / Adoption L

---

### 4B — CUE-DETR Reimplementation

```
scripts/ml/cue_detector.py
  Fine-tune facebook/detr-resnet-50 on EDM-CUE dataset (arXiv:2407.06823)
  Input: 128-band Mel spectrogram (22050 Hz, 2048-sample window) as "image"
  Output: bounding boxes in time → cue point timestamps

Requires:
  - EDM-CUE dataset download (~380 hours, 4,710 tracks)
  - GPU for fine-tuning (~8h on A100 / ~48h on M2 Max)
  - transformers, datasets, torch

Evaluation metric: F-measure @ 0.5s tolerance vs annotated cue points
Target: beat reported CUE-DETR numbers to justify the implementation
```

**Why it matters for PhD:** End-to-end model training + quantitative evaluation = the exact
signal PhD committees want to see. Frame it as "transfer learning for DJ cue estimation."

**Effort:** 2–3 weeks (dataset + training + eval) | **Value:** Portfolio H / **Novelty H** / Adoption M

---

### 4C — B-Roll Matching Engine (CLIP + CLAP)

```
scripts/core/broll_matcher.py
  - Extract CLAP embeddings from audio segments (already have from 3A)
  - Extract CLIP embeddings from video keyframes (torchvision)
  - Learn a lightweight cross-modal alignment head (linear or MLP)
    trained on a small set of labelled audio-video pairs
  - find_broll(audio_segment, video_library) -> list[VideoMatch]

Research: Video2Music (arXiv:2311.00968); rhythm+optical-flow pretraining (arXiv:2309.09421)
```

**Effort:** 1 week | **Value:** Portfolio H / Novelty M / Adoption M

---

## Implementation Sequence (Gantt-style)

```
Week 1:   [1A] FxNorm per-stem LUFS targets
Week 2:   [1B] TIVlib vendor + TIS harmonic score
Week 3:   [2A] Beat This! integration + BeatTracker interface
Week 4:   [2B] Bar-grid cue snapping  |  [3C] rekordbox/Serato export (parallel)
Week 5:   [2C] Stem-level bass muting
Week 6:   [2D] Listening test generation + first 10 mixes  |  [3B] Essentia energy (parallel)
Week 7:   [3A] Crate Digger (CLAP + FAISS) — demo-ready
Week 8:   Listening study runs (async, not blocking)
Weeks 8-12: [4A] Vocal Analyzer  →  [4B] CUE-DETR  →  [4C] B-Roll
```

---

## Claude Code Prompt Templates (for implementation sessions)

**For Gap 1A (per-stem LUFS):**
```
Add analyze_library_stem_targets(library_dir) → dict[str, float] and
normalize_stems_to_corpus_targets(stems, sr, targets, fallback_lufs=-20.0)
to scripts/core/mastering.py.

Rules:
- Use pyloudnorm: import pyloudnorm as pyln; meter = pyln.Meter(sr); lufs = meter.integrated_loudness(audio)
- Cache corpus targets to data/stem_lufs_targets.json
- Existing normalize_stems_to_target() stays unchanged (backward compat)
- True-peak limit each normalized stem to -1 dBTP using existing apply_limiter()
- Add POST /library/calibrate-lufs router endpoint (async job, returns job_id)

Acceptance tests (add to tests/test_mastering.py):
- drums target ≠ bass target (not all the same)
- normalized stem within ±1 LU of corpus target
- silent stem → silent output, no NaN/inf
- output peak ≤ -1 dBTP

Plan first. Only touch mastering.py, routers/library.py, tests/test_mastering.py.
Do not modify dj_engine.py or the existing LUFS track-level chain.
```

**For Gap 1B (TIVlib):**
```
Vendor TIVlib into third_party/TIVlib/ (clone from https://github.com/GramAx/TIVlib).
Create scripts/core/tiv_scoring.py with:
  tiv_from_chroma(chroma: np.ndarray) → Any
  tiv_harmonic_score(chroma_a, chroma_b, weights=None) → float  # returns [0, 1]
  compare_tiv_vs_camelot(chroma_a, camelot_a, chroma_b, camelot_b) → dict

Add tiv_compatibility: float | None = None to TransitionPlan in dj_analysis.py.
Do NOT remove psychoacoustic_consonance() — keep as fallback.

Tests (new file tests/test_tiv_scoring.py):
- C major vs C major → 1.0
- Adjacent Camelot pair (8B→9B) → > 0.7
- Tritone (8B→2B) → < 0.3
- All 24×24 pairs in [0, 1]
- pearson(tiv_score, camelot_adjacency) > 0.8

PYTHONPATH update: add third_party/TIVlib to start.sh.
Plan first. Only touch: third_party/, scripts/core/tiv_scoring.py, scripts/core/dj_analysis.py.
```

**For Stage 3A (Crate Digger):**
```
Create scripts/core/crate_digger.py with class CrateDigger using laion_clap
(LAION-CLAP model: laion/clap-htsat-unfused).

Public API:
  index_library(progress_cb=None) → None    # writes data/clap_index.npz
  find_similar(query_name, query_audio, query_text, k, camelot_filter, bpm_range) → list[dict]

Router endpoint: GET /library/crate-search?query_name=X&query_text=Y&k=10
                 POST /library/build-clap-index → {job_id}

Fallback: if laion_clap not installed, route to existing music_index.py similarity.

Tests (tests/test_crate_digger.py):
- k=5 → exactly 5 results
- self-query → rank 1 with score ≈ 1.0
- text query → valid list (model may vary)
- camelot filter → all results match filter

Plan first. New files only: scripts/core/crate_digger.py, scripts/api/routers/search.py
or add to library.py, tests/test_crate_digger.py.
```

---

## Open Questions / Risks

| Risk | Severity | Mitigation |
|---|---|---|
| TIVlib not on PyPI — vendoring adds maintenance burden | Medium | Pin to a specific commit hash in requirements or a script; document in CLAUDE.md |
| Beat This! model (~300 MB) makes first-run slow | Low | Lazy-load; cache after first download; add to Dockerfile |
| CLAP model (~300 MB) second large model | Low | Same lazy-load pattern; share ModelManager with MusicGen |
| CUE-DETR training requires GPU + 380h dataset | High (for Stage 4B) | Gate on Stage 1–2 completion; use a university compute cluster |
| Formal listening test: finding 15 participants | Medium | Use CS/music department classmates or Prolific (£2-3/participant for 15min) |
| Serato marker format is undocumented (reverse-engineered) | Medium | Use `mutagen` + Serato community docs; test with Serato trial version |

---

## CLAUDE.md Additions (append after this plan is validated)

Add to **Gotchas** section:
```
**TIVlib not on PyPI** — vendored in third_party/TIVlib/. Import requires
PYTHONPATH to include that directory. Set in start.sh; see tiv_scoring.py.

**CLAP model download** — first call to CrateDigger.index_library() downloads
~300 MB to ~/.cache/huggingface/. Requires internet. Subsequent calls use cache.

**Beat This! downbeat model** — requires `pip install beat-this` (~50 MB model).
Falls back to librosa via BeatTracker interface if not installed.

**Per-stem LUFS targets** — data/stem_lufs_targets.json must be generated via
POST /library/calibrate-lufs before normalize_stems_to_corpus_targets() returns
meaningful results. Without it, falls back to flat -20 LUFS.
```
