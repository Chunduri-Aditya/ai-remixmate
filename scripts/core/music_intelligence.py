"""
scripts/core/music_intelligence.py — Music Intelligence Engine

The central brain of AI RemixMate.  Integrates and extends the existing
musical_analysis.py with a complete per-track feature vector and an AI
transition scorer.

Features computed per track (MusicVector):
  • BPM + tempo stability
  • Key + mode (major/minor) via Krumhansl-Schmuckler on CQT chroma
  • Camelot notation  (harmonic mixing)
  • Energy curve      (normalised RMS over time — for drop/build detection)
  • Drop position     (time-of-loudest-energy spike)
  • Spectral brightness (centroid + rolloff + contrast)
  • Danceability      (beat regularity × beat strength × bass ratio)
  • Vocal density     (uses Demucs stem if available, else spectral proxy)
  • Chord progression (simplified chroma → chord template matching)

Transition scoring (TransitionScore):
  overall = 0.35 × beat_alignment
          + 0.35 × harmonic_match    (Camelot wheel)
          + 0.20 × energy_smoothness
          + 0.10 × (1 − vocal_clash)
          − vocal_clash_penalty

All functions fail gracefully — if librosa is unavailable or audio is
pathological, sensible defaults are returned so the DJ engine never crashes.

Usage:
    from scripts.core.music_intelligence import compute_track_vector, compute_transition_score
    vec = compute_track_vector(audio, sr, bpm=128.0, stems_dir=Path("library/My Song"))
    score = compute_transition_score(vec_a, vec_b)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Musical constants ─────────────────────────────────────────────────────────

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Camelot Wheel — full notation (outer ring B = major, inner ring A = minor)
CAMELOT: Dict[str, str] = {
    'C major':  '8B',  'G major':  '9B',  'D major': '10B', 'A major': '11B',
    'E major': '12B',  'B major':  '1B',  'F# major': '2B', 'C# major': '3B',
    'G# major': '4B',  'D# major': '5B',  'A# major': '6B', 'F major':  '7B',
    'A minor':  '8A',  'E minor':  '9A',  'B minor': '10A', 'F# minor': '11A',
    'C# minor': '12A', 'G# minor': '1A',  'D# minor': '2A', 'A# minor': '3A',
    'F minor':  '4A',  'C minor':  '5A',  'G minor':  '6A', 'D minor':  '7A',
}

# Krumhansl-Schmuckler tonal hierarchy profiles (normalised)
_MAJOR_PROF = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                         2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
_MINOR_PROF = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                         2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)
_MAJOR_PROF /= _MAJOR_PROF.sum()
_MINOR_PROF /= _MINOR_PROF.sum()

# Chord templates: [root, third/minor-third, fifth]
_CHORD_TMPLS: Dict[str, np.ndarray] = {
    'maj':  np.array([1,0,0,0,1,0,0,1,0,0,0,0], dtype=float),
    'min':  np.array([1,0,0,1,0,0,0,1,0,0,0,0], dtype=float),
    'dom7': np.array([1,0,0,0,1,0,0,1,0,0,1,0], dtype=float),
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MusicVector:
    """
    Complete per-track music feature vector for AI-assisted DJ mixing.

    Populated by compute_track_vector().  All fields have safe defaults so
    the object is always valid even if some analysis steps fail.
    """
    # ─── Core identity ────────────────────────────────────────────────────────
    bpm:              float = 0.0
    key_name:         str   = 'C'      # root note, e.g. 'A'
    mode:             str   = 'major'  # 'major' or 'minor'
    camelot:          str   = '8B'     # Camelot notation, e.g. '11A'
    key_confidence:   float = 0.0      # 0–1

    # ─── Energy & dynamics ───────────────────────────────────────────────────
    energy_curve:     List[float] = field(default_factory=list)  # 0–1, per 0.5 s
    energy_mean:      float = 0.5
    energy_std:       float = 0.2
    drop_position:    Optional[float] = None  # seconds, or None

    # ─── Timbre ──────────────────────────────────────────────────────────────
    spectral_centroid:    float = 2000.0   # Hz — brightness centre-of-mass
    spectral_rolloff:     float = 8000.0   # Hz — 85 % energy below this
    spectral_contrast:    float = 0.0      # mean contrast
    zero_crossing_rate:   float = 0.05

    # ─── Rhythm ──────────────────────────────────────────────────────────────
    danceability:     float = 0.5
    beat_strength:    float = 0.5
    tempo_stability:  float = 0.5

    # ─── Vocals ──────────────────────────────────────────────────────────────
    vocal_density:    float = 0.5   # 0 = instrumental, 1 = heavily vocal

    # ─── Harmony ─────────────────────────────────────────────────────────────
    chord_sequence:   List[str]   = field(default_factory=list)
    chroma_vector:    List[float] = field(default_factory=list)  # 12-dim

    def to_dict(self) -> Dict:
        """JSON-serialisable dict (numpy arrays → Python lists)."""
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, np.floating):
                out[k] = float(v)
            elif isinstance(v, np.integer):
                out[k] = int(v)
            else:
                out[k] = v
        return out


@dataclass
class TransitionScore:
    """AI score for a proposed A→B track transition."""
    overall:              float = 0.0   # weighted composite 0–1

    beat_alignment:       float = 0.0   # BPM closeness
    harmonic_match:       float = 0.0   # Camelot compatibility
    energy_smoothness:    float = 0.0   # A-end vs B-start energy
    timbral_similarity:   float = 0.0   # spectral character match
    vocal_clash:          float = 0.0   # 0–1 (1 = both vocal-heavy)
    vocal_clash_penalty:  float = 0.0   # subtracted from overall

    camelot_a: str = ''
    camelot_b: str = ''
    key_a:     str = ''
    key_b:     str = ''

    # Camelot move semantics (from DJ_THEORY.md §2)
    camelot_move:      str = ''   # e.g. "adjacent", "energy_boost", "same_key"
    camelot_move_desc: str = ''   # human-readable description of the move

    recommended_transition_bars: int = 16
    notes: List[str] = field(default_factory=list)  # human-readable advice

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# ── Key detection ─────────────────────────────────────────────────────────────

def detect_key(audio: np.ndarray, sr: int) -> Tuple[str, str, str, float]:
    """
    Detect musical key and mode via Krumhansl-Schmuckler on CQT chroma.

    Returns:
        (key_name, mode, camelot, confidence)
        e.g.  ('A', 'minor', '8A', 0.82)
    """
    try:
        import librosa
        # CQT chroma is more accurate for tonal content than STFT chroma
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=24)
        c = chroma.mean(axis=1)
        c = c / (c.sum() + 1e-10)

        best_val, best_key, best_mode = -np.inf, 0, 'major'
        for k in range(12):
            maj = float(np.corrcoef(c, np.roll(_MAJOR_PROF, k))[0, 1])
            mnn = float(np.corrcoef(c, np.roll(_MINOR_PROF, k))[0, 1])
            if maj > best_val:
                best_val, best_key, best_mode = maj, k, 'major'
            if mnn > best_val:
                best_val, best_key, best_mode = mnn, k, 'minor'

        key_name   = NOTE_NAMES[best_key]
        camelot    = CAMELOT.get(f'{key_name} {best_mode}', '8B')
        confidence = float(np.clip((best_val + 1) / 2, 0.0, 1.0))
        return key_name, best_mode, camelot, confidence

    except Exception as exc:
        log.debug('Key detection failed: %s', exc)
        return 'C', 'major', '8B', 0.0


# ── Energy curve ──────────────────────────────────────────────────────────────

def compute_energy_curve(audio: np.ndarray, sr: int,
                          hop_seconds: float = 0.5) -> np.ndarray:
    """
    Normalised RMS energy curve (one value per `hop_seconds`).
    Returns float32 array in [0, 1].
    """
    try:
        import librosa
        hop   = max(1, int(sr * hop_seconds))
        frame = hop * 4
        rms   = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop)[0]
        lo, hi = rms.min(), rms.max()
        if hi > lo:
            return ((rms - lo) / (hi - lo)).astype(np.float32)
        return np.zeros(len(rms), dtype=np.float32)
    except Exception:
        return np.array([0.5], dtype=np.float32)


def detect_drop(energy_curve: np.ndarray,
                hop_seconds: float = 0.5) -> Optional[float]:
    """
    Locate the most prominent 'drop' — the highest-energy frame that follows
    a build-up in the latter two-thirds of the track.
    Returns position in seconds, or None if no clear drop.
    """
    if len(energy_curve) < 8:
        return None
    start    = len(energy_curve) // 3
    sub      = energy_curve[start:]
    peak_rel = int(np.argmax(sub))
    peak_val = float(sub[peak_rel])
    if peak_val < 0.75:          # not energetic enough to be a drop
        return None
    return float((start + peak_rel) * hop_seconds)


# ── Danceability ──────────────────────────────────────────────────────────────

def compute_danceability(audio: np.ndarray,
                          sr: int) -> Tuple[float, float, float]:
    """
    Spotify-inspired danceability (no ML model, fully derived from signal).

    Returns:
        (danceability, beat_strength, tempo_stability)  all in [0, 1]

    Algorithm:
      • beat_strength   = mean onset strength at beat positions / max onset
      • tempo_stability = 1 − std(inter-beat-intervals) / mean(ibi)
      • bass_ratio      = bass RMS / mid RMS  (kicks drive dancing)
      • danceability    = 0.40×stability + 0.35×beat_strength + 0.25×bass_ratio
    """
    try:
        import librosa
        onset_env       = librosa.onset.onset_strength(y=audio, sr=sr)
        _, beats        = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        if len(beats) < 4:
            return 0.3, 0.3, 0.3

        # Beat strength
        beat_str = float(np.mean(onset_env[beats]) / (np.max(onset_env) + 1e-8))
        beat_str = float(np.clip(beat_str, 0, 1))

        # Tempo stability (inter-beat interval regularity)
        ibi       = np.diff(beats.astype(float))
        stability = float(np.clip(
            1.0 - np.std(ibi) / (np.mean(ibi) + 1e-8), 0.0, 1.0
        ))

        # Bass-to-mid ratio (measure kick/bass presence)
        clip  = audio[:sr * 20] if len(audio) > sr * 20 else audio
        try:
            from scripts.core.gpu import gpu_stft
            S = np.abs(gpu_stft(clip))
        except (ImportError, Exception):
            S = np.abs(librosa.stft(clip))
        freqs = librosa.fft_frequencies(sr=sr)
        bi    = freqs <= 250
        mi    = (freqs > 250) & (freqs <= 2000)
        bass  = float(S[bi].mean()) if bi.any() else 1e-8
        mid   = float(S[mi].mean()) if mi.any() else 1e-8
        bass_r = float(np.clip(bass / (mid + 1e-8) / 3.0, 0.0, 1.0))

        dance = float(np.clip(
            0.40 * stability + 0.35 * beat_str + 0.25 * bass_r, 0.0, 1.0
        ))
        return dance, beat_str, stability

    except Exception as exc:
        log.debug('Danceability failed: %s', exc)
        return 0.5, 0.5, 0.5


# ── Spectral brightness ───────────────────────────────────────────────────────

def compute_spectral_brightness(audio: np.ndarray,
                                  sr: int) -> Tuple[float, float, float]:
    """
    Return (centroid_hz, rolloff_hz, contrast_mean).

    spectral_centroid  — average frequency weighted by energy ('brightness')
    spectral_rolloff   — frequency below which 85 % of energy lives
    spectral_contrast  — mean spectral contrast across bands
    """
    try:
        import librosa
        try:
            from scripts.core.gpu import gpu_stft
            S = np.abs(gpu_stft(audio))
        except (ImportError, Exception):
            S = np.abs(librosa.stft(audio))
        cent    = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
        roll    = float(np.mean(librosa.feature.spectral_rolloff(
            S=S, sr=sr, roll_percent=0.85)))
        cont    = float(np.mean(librosa.feature.spectral_contrast(S=S, sr=sr)))
        return cent, roll, cont
    except Exception as exc:
        log.debug('Spectral brightness failed: %s', exc)
        return 2000.0, 8000.0, 0.0


# ── Chord progression ─────────────────────────────────────────────────────────

def compute_chord_progression(audio: np.ndarray, sr: int,
                                max_chords: int = 16) -> List[str]:
    """
    Simplified per-bar chord labelling via chroma template matching.
    Analyses 2-second frames (≈1 bar at 120 BPM).
    Returns a de-duplicated list of chord symbols e.g. ['Am', 'F', 'C', 'G'].
    """
    try:
        import librosa
        hop_len = sr * 2  # 2-second frames
        chroma  = librosa.feature.chroma_cqt(
            y=audio, sr=sr, bins_per_octave=24, hop_length=hop_len)

        chords: List[str] = []
        for i in range(min(chroma.shape[1], max_chords)):
            c = chroma[:, i]
            c = c / (c.sum() + 1e-10)
            best_name, best_sc = 'C', -np.inf
            for root in range(12):
                for qtype, tmpl in _CHORD_TMPLS.items():
                    t  = np.roll(tmpl, root)
                    t  = t / (t.sum() + 1e-10)
                    sc = float(np.corrcoef(c, t)[0, 1]) if not np.all(t == 0) else 0.0
                    if sc > best_sc:
                        best_sc   = sc
                        suffix    = 'm' if qtype == 'min' else ('7' if qtype == 'dom7' else '')
                        best_name = NOTE_NAMES[root] + suffix
            chords.append(best_name)

        # De-duplicate consecutive repeats
        if not chords:
            return []
        deduped = [chords[0]]
        for i in range(1, len(chords)):
            if chords[i] != chords[i - 1]:
                deduped.append(chords[i])
        return deduped

    except Exception as exc:
        log.debug('Chord progression failed: %s', exc)
        return []


# ── Vocal density ─────────────────────────────────────────────────────────────

def detect_vocal_density(audio: np.ndarray, sr: int,
                           vocals_path: Optional[Path] = None) -> float:
    """
    Estimate the fraction of the track with significant vocal content.

    Strategy:
      1. If a Demucs vocals.wav stem exists, use its RMS directly (accurate).
      2. Else fall back to spectral proxy: vocal formant band (300–3400 Hz)
         vs bass band (< 200 Hz) energy ratio.

    Returns float in [0, 1].  1.0 = vocal throughout.
    """
    # ── Preferred: Demucs stem ─────────────────────────────────────────────
    if vocals_path and Path(vocals_path).exists():
        try:
            import librosa
            voc, _ = librosa.load(str(vocals_path), sr=sr, mono=True, duration=120.0)
            hop    = sr // 10
            rms_v  = librosa.feature.rms(
                y=voc, frame_length=hop * 2, hop_length=hop)[0]
            thresh = 0.10 * rms_v.max()
            return float((rms_v > thresh).mean())
        except Exception as exc:
            log.debug('Stem vocal density failed: %s', exc)

    # ── Fallback: spectral proxy ───────────────────────────────────────────
    try:
        import librosa
        try:
            from scripts.core.gpu import gpu_stft
            S = np.abs(gpu_stft(audio))
        except (ImportError, Exception):
            S = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        vi    = (freqs >= 300) & (freqs <= 3400)
        bi    = freqs <= 200
        ve    = S[vi].mean(axis=0) if vi.any() else np.ones(S.shape[1])
        be    = S[bi].mean(axis=0) if bi.any() else np.ones(S.shape[1]) * 1e-8
        return float(((ve / (be + 1e-8)) > 3.0).mean())
    except Exception:
        return 0.5


# ── Master: compute_track_vector ──────────────────────────────────────────────

def compute_track_vector(
    audio:      np.ndarray,
    sr:         int,
    bpm:        float = 0.0,
    stems_dir:  Optional[Path] = None,
) -> MusicVector:
    """
    Compute a complete MusicVector for a track.

    Args:
        audio:     Mono float32 audio array
        sr:        Sample rate
        bpm:       Pre-detected BPM (0.0 = auto-detect inside this function)
        stems_dir: Directory containing Demucs stems (vocals.wav etc.)
                   — used for more accurate vocal density estimation.

    Returns:
        MusicVector with all features populated (graceful fallbacks on error).
    """
    vec = MusicVector()

    # ── BPM ───────────────────────────────────────────────────────────────────
    if bpm > 0:
        vec.bpm = float(bpm)
    else:
        try:
            import librosa
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            vec.bpm  = float(tempo)
        except Exception:
            vec.bpm = 120.0

    # ── Key + Camelot ─────────────────────────────────────────────────────────
    vec.key_name, vec.mode, vec.camelot, vec.key_confidence = detect_key(audio, sr)

    # ── Energy curve + drop ───────────────────────────────────────────────────
    hop_s = 0.5
    ec              = compute_energy_curve(audio, sr, hop_seconds=hop_s)
    vec.energy_curve  = ec.tolist()
    vec.energy_mean   = float(ec.mean())
    vec.energy_std    = float(ec.std())
    vec.drop_position = detect_drop(ec, hop_seconds=hop_s)

    # ── Spectral brightness ───────────────────────────────────────────────────
    cent, roll, cont = compute_spectral_brightness(audio, sr)
    vec.spectral_centroid = cent
    vec.spectral_rolloff  = roll
    vec.spectral_contrast = cont
    try:
        import librosa
        vec.zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    except Exception:
        pass

    # ── Danceability ──────────────────────────────────────────────────────────
    vec.danceability, vec.beat_strength, vec.tempo_stability = compute_danceability(audio, sr)

    # ── Vocal density ─────────────────────────────────────────────────────────
    vocals_path: Optional[Path] = None
    if stems_dir:
        for fname in ('vocals.wav', 'vocal.wav', 'Vocals.wav'):
            p = Path(stems_dir) / fname
            if p.exists():
                vocals_path = p
                break
    vec.vocal_density = detect_vocal_density(audio, sr, vocals_path=vocals_path)

    # ── Chord progression ─────────────────────────────────────────────────────
    vec.chord_sequence = compute_chord_progression(audio, sr)

    # ── Mean chroma vector ────────────────────────────────────────────────────
    try:
        import librosa
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        vec.chroma_vector = chroma.mean(axis=1).tolist()
    except Exception:
        vec.chroma_vector = [0.0] * 12

    log.info(
        'MusicVector  BPM=%.1f  Key=%s %s (%s, conf=%.2f) '
        'Dance=%.2f  Vocal=%.2f  Drop=%s',
        vec.bpm, vec.key_name, vec.mode, vec.camelot, vec.key_confidence,
        vec.danceability, vec.vocal_density,
        f'{vec.drop_position:.1f}s' if vec.drop_position else 'none',
    )
    return vec


# ── Harmonic compatibility ────────────────────────────────────────────────────

def _camelot_move_analysis(
    camelot_a: str, camelot_b: str
) -> Tuple[float, str, str]:
    """
    Analyse a Camelot Wheel transition and return (score, move_name, move_description).

    Implements the seven canonical moves documented in DJ_THEORY.md §2, grounded in:
      - Mixed In Key best-practice rules (Yakov Vorobyev, 2006)
      - Armin Van Buuren's documented +7 "energy boost" technique (Ultra 2017)
      - Psychoacoustics of critical-band beating for dissonance scoring

    The +7 clockwise detection is the critical addition over naive distance scoring.
    A naive |dist| calculation would score the +7 move as distance=5 (very poor),
    when it's actually a deliberate, musically valid technique producing a
    1-semitone upward pitch shift that the audience perceives as energising.
    """
    def _parse(c: str) -> Tuple[int, str]:
        try:
            return int(c[:-1]), c[-1].upper()
        except (ValueError, IndexError):
            return 8, 'B'

    num_a, let_a = _parse(camelot_a)
    num_b, let_b = _parse(camelot_b)

    dist_cw  = (num_b - num_a) % 12
    dist_ccw = (num_a - num_b) % 12
    dist     = min(dist_cw, dist_ccw)

    # Same key — identical notes, zero harmonic change
    if dist == 0 and let_a == let_b:
        return (1.00, "same_key",
                f"Same key ({camelot_a}) — identical notes, zero harmonic change")

    # Parallel mode — same root note, major ↔ minor mood shift
    if dist == 0:
        return (0.85, "parallel_mode",
                f"Parallel mode ({camelot_a} → {camelot_b}) — same root, "
                "emotional colour shift without harmonic clash")

    # +7 clockwise energy boost — Armin Van Buuren's signature move
    # Mathematically equivalent to his "−5 move". Produces 1-semitone upward
    # pitch shift that registers as increased energy. Use sparingly + quick overlaps.
    if dist_cw == 7 and let_a == let_b:
        return (0.80, "energy_boost",
                f"+7 energy boost ({camelot_a} → {camelot_b}) — 1-semitone upward "
                "pitch shift, euphoric 'key change' effect. Use sparingly; "
                "keep overlap short to avoid audible clash.")

    # ±1 adjacent — safest key change (only one scale note differs)
    if dist == 1:
        direction = "clockwise (+1)" if dist_cw == 1 else "counter-clockwise (−1)"
        return (0.90, "adjacent",
                f"Adjacent key {direction} ({camelot_a} → {camelot_b}) — "
                "one note changes, subtle energy lift or drop")

    # ±2 double step — slight tension, use shorter overlap
    if dist == 2:
        return (0.65, "double_step",
                f"±2 double step ({camelot_a} → {camelot_b}) — slight tension, "
                "acceptable with short overlap or effects bridge")

    # ±3–5 distant — significant harmonic clash
    if dist <= 5:
        score = round(0.50 - (dist - 3) * 0.10, 2)
        return (score, "distant",
                f"Distant keys, {dist} steps apart ({camelot_a} → {camelot_b}) — "
                "significant clash; use echo-out/filter transition or full reset")

    # ≥6 — maximum dissonance (opposite sides of the wheel)
    return (0.05, "clash",
            f"Harmonic clash ({camelot_a} → {camelot_b}, {dist} steps) — "
            "opposite sides of Camelot Wheel; avoid simultaneous playback")


def camelot_harmonic_score(camelot_a: str, camelot_b: str) -> float:
    """
    Camelot wheel harmonic compatibility score (0–1).

    Delegates to _camelot_move_analysis which implements the seven canonical
    moves from DJ_THEORY.md §2. The key addition over the previous version
    is explicit detection of the +7 clockwise 'energy boost' move (Armin Van
    Buuren's signature) — previously mis-scored as distance=5 (0.20) when
    it is actually a legitimate DJ technique deserving a score of 0.80.

    Returns just the score for backwards compatibility.
    Use _camelot_move_analysis() directly to get the move name + description.
    """
    if not camelot_a or not camelot_b or camelot_a == '?' or camelot_b == '?':
        return 0.5   # unknown → neutral
    score, _, _ = _camelot_move_analysis(camelot_a, camelot_b)
    return score


def timbral_similarity(vec_a: "MusicVector", vec_b: "MusicVector") -> float:
    """
    Score timbral similarity between two tracks in [0, 1].

    Combines three spectral features that collectively characterise a track's
    sonic texture (DJ_THEORY.md §7 — timbral fingerprint):
      spectral_centroid  — brightness / where the energy lives frequency-wise
      zero_crossing_rate — noisiness / percussiveness proxy
      spectral_contrast  — separation between peaks and valleys in spectrum

    A score of 1.0 means the two tracks sound spectrally identical.
    A score < 0.4 suggests a major timbral contrast (e.g. ambient → hard techno).
    Such contrast can be intentional white-space strategy or an unpleasant mismatch
    depending on set context — the notes field in TransitionScore flags it.
    """
    centroid_diff = abs(vec_a.spectral_centroid - vec_b.spectral_centroid) / 8000.0
    zcr_diff      = abs(vec_a.zero_crossing_rate - vec_b.zero_crossing_rate) * 10.0
    contrast_diff = abs(vec_a.spectral_contrast  - vec_b.spectral_contrast)  / 30.0
    raw = (centroid_diff + zcr_diff + contrast_diff) / 3.0
    return round(float(np.clip(1.0 - raw, 0.0, 1.0)), 4)


# ── AI transition scorer ──────────────────────────────────────────────────────

def compute_transition_score(vec_a: MusicVector, vec_b: MusicVector) -> TransitionScore:
    """
    Score the proposed A→B transition and return an annotated TransitionScore.

    Formula (research-backed weights from SetFlow algorithm + Kim et al. ISMIR 2020
    analysis of 1,557 real DJ mixes — full rationale in docs/DJ_THEORY.md §3):

      overall = 0.35 × harmonic_match
              + 0.25 × beat_alignment
              + 0.15 × energy_smoothness
              + 0.10 × timbral_similarity
              + 0.10 × (1 − vocal_clash)
              + 0.05 × danceability_compat
              − vocal_clash_penalty

    Change from previous version:
      • beat_alignment weight reduced 0.35 → 0.25 (research shows harmonic is
        the dominant constraint; BPM is adjustable, key clash is not)
      • timbral_similarity added as 0.10 dimension (new)
      • danceability_compat added as 0.05 dimension (new)
      • Camelot move named and described in all notes
      • Energy direction (climbing/descending/maintaining) surfaced in notes
      • Genre-aware recommended_transition_bars based on BPM + compatibility
    """
    ts = TransitionScore()
    ts.camelot_a = vec_a.camelot
    ts.camelot_b = vec_b.camelot
    ts.key_a     = f'{vec_a.key_name} {vec_a.mode}'
    ts.key_b     = f'{vec_b.key_name} {vec_b.mode}'

    # ── Beat alignment ─────────────────────────────────────────────────────────
    # Kim et al. (ISMIR 2020): 86.1% of real DJ transitions adjust < 5% tempo.
    # Deviation > 5% is increasingly noticeable; > 30% requires half/double-time.
    if vec_a.bpm > 0 and vec_b.bpm > 0:
        ratio     = vec_b.bpm / vec_a.bpm
        # Also check half/double-time compatibility
        best_ratio = min(
            abs(ratio - 1.0),
            abs(ratio * 2 - 1.0),
            abs(ratio / 2 - 1.0),
        )
        ts.beat_alignment = float(np.clip(1.0 - best_ratio * 4.0, 0.0, 1.0))
        bpm_delta = vec_b.bpm - vec_a.bpm
        if abs(best_ratio) > 0.30:
            ts.notes.append(
                f'⚠️  Large BPM gap: {vec_a.bpm:.0f} → {vec_b.bpm:.0f} — '
                f'consider half/double-time mixing or BPM creep across 3+ tracks'
            )
        elif 0.05 < abs(best_ratio) <= 0.30:
            ts.notes.append(
                f'ℹ️  BPM shift: {vec_a.bpm:.0f} → {vec_b.bpm:.0f} '
                f'({bpm_delta:+.1f} BPM, {abs(best_ratio) * 100:.0f}% stretch)'
            )
    else:
        ts.beat_alignment = 0.5

    # ── Harmonic match + Camelot move analysis ─────────────────────────────────
    if vec_a.camelot and vec_b.camelot and vec_a.camelot != '?' and vec_b.camelot != '?':
        h_score, move_name, move_desc = _camelot_move_analysis(vec_a.camelot, vec_b.camelot)
        ts.harmonic_match    = h_score
        ts.camelot_move      = move_name
        ts.camelot_move_desc = move_desc

        if move_name == "same_key":
            ts.notes.append(f'✅  {move_desc}')
        elif move_name == "adjacent":
            ts.notes.append(f'✅  {move_desc}')
        elif move_name == "parallel_mode":
            ts.notes.append(f'✅  {move_desc}')
        elif move_name == "energy_boost":
            ts.notes.append(f'⚡  {move_desc}')
        elif move_name == "double_step":
            ts.notes.append(f'ℹ️  {move_desc}')
        elif h_score < 0.4:
            ts.notes.append(f'⚠️  {move_desc}')
        else:
            ts.notes.append(f'ℹ️  {move_desc}')
    else:
        ts.harmonic_match = camelot_harmonic_score(vec_a.camelot, vec_b.camelot)
        ts.camelot_move   = "unknown"

    # ── Energy smoothness + direction ──────────────────────────────────────────
    # Analyse last 20% of A vs first 20% of B.
    # Also surface energy direction (climbing / descending / maintaining) —
    # a key element of set arc management (DJ_THEORY.md §1).
    if vec_a.energy_curve and vec_b.energy_curve:
        ec_a   = np.array(vec_a.energy_curve)
        ec_b   = np.array(vec_b.energy_curve)
        tail_a = ec_a[max(0, len(ec_a) - len(ec_a) // 5):]
        head_b = ec_b[:max(1, len(ec_b) // 5)]
        e_end  = float(tail_a.mean())
        e_start = float(head_b.mean())
        gap    = abs(e_end - e_start)
        ts.energy_smoothness = float(np.clip(1.0 - gap * 2.0, 0.0, 1.0))

        # Energy direction note
        delta = e_start - e_end
        if delta > 0.15:
            ts.notes.append(f'📈  Energy climbing: A exits at {e_end:.2f}, B enters at {e_start:.2f}')
        elif delta < -0.15:
            ts.notes.append(f'📉  Energy dropping: A exits at {e_end:.2f}, B enters at {e_start:.2f}')
        elif gap > 0.4:
            ts.notes.append(
                f'⚠️  Energy mismatch: A ends {e_end:.2f}, B starts {e_start:.2f} '
                '— consider a filter sweep or echo-out to bridge the gap'
            )
    else:
        ts.energy_smoothness = 0.5

    # ── Timbral similarity ─────────────────────────────────────────────────────
    # Spectral fingerprint match — prevents jarring sonic texture jumps.
    # See DJ_THEORY.md §7 and timbral_similarity() docstring.
    ts.timbral_similarity = timbral_similarity(vec_a, vec_b)
    if ts.timbral_similarity < 0.35:
        ts.notes.append(
            f'ℹ️  High timbral contrast (score {ts.timbral_similarity:.2f}) — '
            'different sonic character; can work as deliberate white-space or surprise, '
            'but use effects bridge if unintentional'
        )

    # ── Danceability compatibility ─────────────────────────────────────────────
    dance_gap = abs(vec_a.danceability - vec_b.danceability)
    dance_compat = float(np.clip(1.0 - dance_gap * 2.5, 0.0, 1.0))

    # ── Vocal clash ───────────────────────────────────────────────────────────
    ts.vocal_clash = float(vec_a.vocal_density * vec_b.vocal_density)
    if vec_a.vocal_density > 0.6 and vec_b.vocal_density > 0.6:
        ts.vocal_clash_penalty = ts.vocal_clash * 0.40
        ts.notes.append(
            f'⚠️  Both tracks are vocal-heavy '
            f'(A: {vec_a.vocal_density:.0%}, B: {vec_b.vocal_density:.0%}) '
            '— use instrumental/drums stems at the transition, '
            'or delay B vocals until after the bass swap'
        )
    else:
        ts.vocal_clash_penalty = 0.0

    # ── Genre-aware recommended transition length ──────────────────────────────
    # Based on DJ_THEORY.md §5 genre-specific duration table.
    # Logic: BPM range determines genre bracket; compatibility score determines
    # whether to use the long or short end of that bracket.
    avg_bpm = (vec_a.bpm + vec_b.bpm) / 2.0 if vec_a.bpm > 0 and vec_b.bpm > 0 else 128.0
    high_compat = ts.harmonic_match >= 0.85 and ts.beat_alignment >= 0.80
    low_compat  = ts.harmonic_match < 0.50 or ts.beat_alignment < 0.55

    if avg_bpm >= 165:          # Drum & bass / jungle
        ts.recommended_transition_bars = 8 if low_compat else 16
    elif avg_bpm >= 140:        # Hard techno / trance
        ts.recommended_transition_bars = 16 if low_compat else (64 if high_compat else 32)
    elif avg_bpm >= 125:        # Techno / house
        ts.recommended_transition_bars = 8 if low_compat else (64 if high_compat else 32)
    elif avg_bpm >= 100:        # Deep house
        ts.recommended_transition_bars = 8 if low_compat else (32 if high_compat else 16)
    else:                       # Hip-hop / trap / downtempo
        ts.recommended_transition_bars = 4 if low_compat else (16 if high_compat else 8)

    # ── Overall (research-backed weighting) ────────────────────────────────────
    raw = (
        0.35 * ts.harmonic_match
        + 0.25 * ts.beat_alignment
        + 0.15 * ts.energy_smoothness
        + 0.10 * ts.timbral_similarity
        + 0.10 * (1.0 - ts.vocal_clash)
        + 0.05 * dance_compat
    )
    ts.overall = float(np.clip(raw - ts.vocal_clash_penalty, 0.0, 1.0))

    return ts
