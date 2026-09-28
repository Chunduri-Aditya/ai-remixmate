"""
Main remix orchestrator: handles audio conversion, stem separation, mixing, and arrangement.

This module provides the core remixing functionality, including:
- Audio format conversion (multi-format support)
- Stem separation (Demucs-based)
- Classic mixing (stem-based)
- Arrangement mixing (structure-aware, timeline-based)
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path

# Third-party
import numpy as np
import librosa
import soundfile as sf

# Optional Demucs import (for stem separation)
try:
    import demucs.separate
    DEMUCS_AVAILABLE = True
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Demucs not available: {e}. Stem separation will be limited.")
    DEMUCS_AVAILABLE = False

# Local imports
from .config import (
    CONVERTED_DIR,
    AUDIO_OUTPUT_DIR,
    SEPARATED_DIR,
    SAMPLE_RATE,
    MAX_DURATION_SEC
)
from .dj_mixing import (
    balance_stem_volumes,
    detect_bpm,
    estimate_key,
    time_stretch_audio,
    pitch_shift_audio,
    are_keys_compatible,
    create_crossfade_curve,
    apply_eq_filter,
    apply_reverb,
    apply_delay,
    ensure_stereo
)
from .structure_detection import detect_sections, detect_beat_grid
from .timeline_planner import TimelinePlanner
from .timeline_renderer import TimelineRenderer
from .recommendations import analyze_track_characteristics
from .ml_audio_features import get_strategy_engine

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)

STEM_NAMES = ("vocals", "drums", "bass", "other")

# ============================================================================
# AUDIO CONVERSION
# ============================================================================

def convert_audio_to_wav(input_path, output_path=None):
    """
    Convert any audio format to WAV.

    Supports: MP3, M4A, FLAC, OGG, AAC, WMA, AIFF, etc.
    Uses librosa (primary) and pydub (fallback).

    Args:
        input_path: Path to input audio file
        output_path: Optional output path (defaults to CONVERTED_DIR)

    Returns:
        Path to converted WAV file

    Raises:
        Exception: If conversion fails with both methods
    """
    try:
        # Try librosa first (faster, better quality)
        y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=False)
        y = ensure_stereo(y)

        if output_path is None:
            os.makedirs(CONVERTED_DIR, exist_ok=True)
            filename = Path(input_path).stem + ".wav"
            output_path = os.path.join(CONVERTED_DIR, filename)

        sf.write(output_path, y, SAMPLE_RATE)
        logger.info(f"Converted {input_path} to {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"Librosa conversion failed: {e}, trying pydub")
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(2)

            if output_path is None:
                os.makedirs(CONVERTED_DIR, exist_ok=True)
                filename = Path(input_path).stem + ".wav"
                output_path = os.path.join(CONVERTED_DIR, filename)

            audio.export(output_path, format="wav")
            logger.info(f"Converted {input_path} to {output_path} using pydub")
            return output_path

        except Exception as e2:
            logger.error(f"Audio conversion failed: {e2}")
            raise

# ============================================================================
# STEM SEPARATION
# ============================================================================

def _generated_at():
    """Return an ISO timestamp for metadata files."""
    return datetime.now(timezone.utc).isoformat()


def _fallback_stem_paths(audio_path):
    """Return full-track fallback paths for each expected stem."""
    return {stem_name: audio_path for stem_name in STEM_NAMES}


def _build_stem_status(
    source_audio_path,
    method,
    success,
    stem_paths,
    warning=None,
    error=None,
    stem_dir=None,
):
    """Build machine-readable stem separation status metadata."""
    return {
        "source_audio_path": str(source_audio_path),
        "method": method,
        "success": bool(success),
        "stem_paths": {
            stem_name: str(stem_paths.get(stem_name, ""))
            for stem_name in STEM_NAMES
        },
        "warning": warning,
        "error": error,
        "stem_dir": str(stem_dir) if stem_dir else None,
        "generated_at": _generated_at(),
    }


def separate_stems_with_status(audio_path, output_dir=None):
    """
    Separate audio into stems and return paths plus status metadata.

    Existing callers should continue using separate_stems(), which preserves
    the historical dict-only return value.
    """
    if output_dir is None:
        output_dir = SEPARATED_DIR

    os.makedirs(output_dir, exist_ok=True)

    if not DEMUCS_AVAILABLE:
        warning = (
            "Demucs is not installed. Using the full track as each stem."
        )
        logger.warning(warning)
        stem_paths = _fallback_stem_paths(audio_path)
        return _build_stem_status(
            audio_path,
            method="fallback",
            success=False,
            stem_paths=stem_paths,
            warning=warning,
        )

    track_name = Path(audio_path).stem
    track_output_dir = os.path.join(output_dir, "htdemucs", track_name)

    try:
        # Use Demucs to separate.
        import demucs.separate
        demucs.separate.main([
            "--mp3", "--mp3-bitrate", "320",
            audio_path, "-o", output_dir
        ])

        stems = {}
        missing = []
        for stem_name in STEM_NAMES:
            stem_path = os.path.join(track_output_dir, f"{stem_name}.wav")
            if os.path.exists(stem_path):
                stems[stem_name] = stem_path
                continue

            stem_path = os.path.join(track_output_dir, f"{stem_name}.mp3")
            if os.path.exists(stem_path):
                stems[stem_name] = stem_path
            else:
                missing.append(stem_name)

        if missing:
            warning = (
                "Demucs did not produce all required stems "
                f"({', '.join(missing)} missing). Using the full track as each stem."
            )
            logger.warning(warning)
            stem_paths = _fallback_stem_paths(audio_path)
            return _build_stem_status(
                audio_path,
                method="fallback",
                success=False,
                stem_paths=stem_paths,
                warning=warning,
                stem_dir=track_output_dir,
            )

        return _build_stem_status(
            audio_path,
            method="demucs",
            success=True,
            stem_paths=stems,
            stem_dir=track_output_dir,
        )

    except Exception as e:
        error = str(e)
        warning = (
            "Demucs failed during stem separation. "
            "Using the full track as each stem."
        )
        logger.error(f"Stem separation failed: {e}")
        stem_paths = _fallback_stem_paths(audio_path)
        return _build_stem_status(
            audio_path,
            method="fallback",
            success=False,
            stem_paths=stem_paths,
            warning=warning,
            error=error,
            stem_dir=track_output_dir,
        )


def separate_stems(audio_path, output_dir=None):
    """
    Separate audio into stems using Demucs.

    Separates audio into: vocals, drums, bass, other.
    Falls back to original audio if Demucs unavailable.

    Args:
        audio_path: Path to audio file
        output_dir: Optional output directory (defaults to SEPARATED_DIR)

    Returns:
        Dict with keys 'vocals', 'drums', 'bass', 'other' mapping to file paths
    """
    return separate_stems_with_status(audio_path, output_dir)["stem_paths"]


def metadata_path_for_output(output_path):
    """Return the JSON metadata path for a generated remix file."""
    path = Path(output_path)
    if path.name == "remix.wav":
        metadata_name = "remix_metadata.json"
    elif path.name == "remix_arrangement.wav":
        metadata_name = "remix_arrangement_metadata.json"
    else:
        metadata_name = f"{path.stem}_metadata.json"
    return str(path.with_name(metadata_name))


def save_remix_metadata(output_path, metadata):
    """Save remix metadata beside the generated output file."""
    metadata_path = metadata_path_for_output(output_path)
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def load_remix_metadata(output_path_or_metadata_path):
    """Load remix metadata from an output path or explicit metadata JSON path."""
    path = Path(output_path_or_metadata_path)
    metadata_path = path if path.suffix.lower() == ".json" else Path(
        metadata_path_for_output(path)
    )
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_stem_status(metadata):
    """Format stem separation metadata for beginner-facing UI status text."""
    if not metadata:
        return "Stem status is unavailable."

    separation = metadata.get("stem_separation", metadata)
    track_a = separation.get("track_a", {})
    track_b = separation.get("track_b", {})
    tracks = [track_a, track_b]

    if all(track.get("method") == "demucs" and track.get("success") for track in tracks):
        return "True Demucs stems used for both tracks."

    fallback_tracks = [
        (label, track)
        for label, track in (("Track A", track_a), ("Track B", track_b))
        if track.get("method") == "fallback" or not track.get("success")
    ]

    if fallback_tracks:
        reasons = []
        for label, track in fallback_tracks:
            reason = track.get("warning") or track.get("error")
            if reason:
                reasons.append(f"{label}: {reason}")

        message = (
            "Fallback mode used: Demucs not installed or failed. "
            "Output used full-track audio as each stem."
        )
        if reasons:
            message = f"{message} Reason: {' '.join(reasons)}"
        return message

    return "Stem status is unavailable."


def _build_remix_metadata(
    output_path,
    mode,
    use_arrangement_mixing,
    stem_status_a,
    stem_status_b,
    parameters=None,
):
    """Build machine-readable metadata for a rendered remix."""
    return {
        "generated_at": _generated_at(),
        "output_path": str(output_path),
        "mode": mode,
        "use_arrangement_mixing": bool(use_arrangement_mixing),
        "parameters": parameters or {},
        "stem_separation": {
            "track_a": stem_status_a,
            "track_b": stem_status_b,
        },
    }

def load_stems(stem_paths):
    """
    Load stem audio files into memory.

    Args:
        stem_paths: Dict mapping stem names to file paths

    Returns:
        Dict mapping stem names to audio arrays (numpy arrays)
    """
    stems = {}
    for name, path in stem_paths.items():
        if path and os.path.exists(path):
            try:
                y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=False)
                stems[name] = ensure_stereo(y)
            except Exception as e:
                logger.error(f"Failed to load {name} stem: {e}")
                stems[name] = None
        else:
            stems[name] = None
    return stems

# ============================================================================
# MAIN REMIX FUNCTION
# ============================================================================

def remix_two_files(
    file1_path,
    file2_path,
    mode="mashup",
    use_arrangement_mixing=False,
    use_intelligent_mixing=False,
    crossfade_length=8.0,
    mixing_technique="crossfade",
    apply_beatmatching=True,
    apply_harmonic_mixing=False,
    bass_swap=False,
    eq_low_cut=0.0,
    eq_high_cut=1.0,
    reverb_decay=0.0,
    delay_ms=0.0,
    aggressiveness=0.5,
    energy_shape="chill_to_peak",
    output_path=None
):
    """
    Main remix function - orchestrates the entire remixing pipeline.

    Args:
        file1_path: Path to first audio file
        file2_path: Path to second audio file
        mode: Remix mode ("mashup", "base_vocals_match_instr", "base_instr_match_vocals")
        use_arrangement_mixing: Enable arrangement-level mixing (structure-aware)
        use_intelligent_mixing: Enable intelligent mixing (adaptive strategy)
        crossfade_length: Crossfade duration in seconds
        mixing_technique: Mixing technique ("crossfade", "bass_swap", "quick_cut")
        apply_beatmatching: Whether to apply beatmatching
        apply_harmonic_mixing: Whether to apply harmonic mixing
        bass_swap: Enable bass swap technique
        eq_low_cut: Low frequency cut (0.0-1.0)
        eq_high_cut: High frequency cut (0.0-1.0)
        reverb_decay: Reverb decay time in seconds
        delay_ms: Delay time in milliseconds
        aggressiveness: Arrangement aggressiveness (0.0-1.0)
        energy_shape: Target energy curve shape
        output_path: Optional output file path

    Returns:
        Path to output remix file

    Raises:
        Exception: If remix fails
    """
    try:
        # Convert to WAV if needed
        file1_wav = (
            convert_audio_to_wav(file1_path)
            if not file1_path.lower().endswith('.wav')
            else file1_path
        )
        file2_wav = (
            convert_audio_to_wav(file2_path)
            if not file2_path.lower().endswith('.wav')
            else file2_path
        )

        # Separate stems
        logger.info("Separating stems for track 1...")
        stems1_status = separate_stems_with_status(file1_wav)
        stems1_paths = stems1_status["stem_paths"]
        logger.info("Separating stems for track 2...")
        stems2_status = separate_stems_with_status(file2_wav)
        stems2_paths = stems2_status["stem_paths"]

        # Load stems
        stems1 = load_stems(stems1_paths)
        stems2 = load_stems(stems2_paths)

        track1_features = None
        track2_features = None
        if use_intelligent_mixing or apply_beatmatching or apply_harmonic_mixing:
            track1_features = analyze_track_characteristics(file1_wav)
            track2_features = analyze_track_characteristics(file2_wav)

        if use_intelligent_mixing and track1_features and track2_features:
            strategy = get_strategy_engine().predict_best_mixing_strategy(
                track1_features,
                track2_features,
                track1_features.get("key"),
                track2_features.get("key")
            )
            mixing_technique = strategy.get("technique", mixing_technique)
            crossfade_length = strategy.get("crossfade_length", crossfade_length)
            apply_beatmatching = strategy.get("apply_beatmatching", apply_beatmatching)
            apply_harmonic_mixing = strategy.get("apply_harmonic_mixing", apply_harmonic_mixing)
            bass_swap = mixing_technique == "bass_swap"

        bpm1 = track1_features.get("bpm") if track1_features else None
        bpm2 = track2_features.get("bpm") if track2_features else None
        key1 = track1_features.get("key") if track1_features else None
        key2 = track2_features.get("key") if track2_features else None

        metadata_parameters = {
            "crossfade_length": crossfade_length,
            "mixing_technique": mixing_technique,
            "apply_beatmatching": bool(apply_beatmatching),
            "apply_harmonic_mixing": bool(apply_harmonic_mixing),
            "bass_swap": bool(bass_swap),
            "eq_low_cut": eq_low_cut,
            "eq_high_cut": eq_high_cut,
            "reverb_decay": reverb_decay,
            "delay_ms": delay_ms,
            "aggressiveness": aggressiveness,
            "energy_shape": energy_shape,
        }

        # Arrangement-level mixing
        if use_arrangement_mixing:
            output = _render_arrangement_mix(
                stems1, stems2, file1_wav, file2_wav,
                mode, aggressiveness, energy_shape, apply_harmonic_mixing,
                key1, key2, output_path
            )
            metadata = _build_remix_metadata(
                output,
                mode,
                use_arrangement_mixing=True,
                stem_status_a=stems1_status,
                stem_status_b=stems2_status,
                parameters=metadata_parameters,
            )
            save_remix_metadata(output, metadata)
            return output

        # Classic mixing
        output = _render_classic_mix(
            stems1, stems2, mode, crossfade_length, mixing_technique,
            apply_beatmatching, apply_harmonic_mixing, bass_swap,
            eq_low_cut, eq_high_cut, reverb_decay, delay_ms,
            bpm1, bpm2, key1, key2, output_path
        )
        metadata = _build_remix_metadata(
            output,
            mode,
            use_arrangement_mixing=False,
            stem_status_a=stems1_status,
            stem_status_b=stems2_status,
            parameters=metadata_parameters,
        )
        save_remix_metadata(output, metadata)
        return output

    except Exception as e:
        logger.error(f"Remix failed: {e}")
        raise

# ============================================================================
# ARRANGEMENT MIXING
# ============================================================================

def _render_arrangement_mix(
    stems1, stems2, file1_path, file2_path,
    mode, aggressiveness, energy_shape, apply_harmonic_mixing,
    key1, key2, output_path
):
    """
    Render arrangement-level mix (structure-aware, timeline-based).

    Args:
        stems1: Dict of stems from track 1
        stems2: Dict of stems from track 2
        file1_path: Path to track 1
        file2_path: Path to track 2
        mode: Remix mode
        aggressiveness: Arrangement aggressiveness (0.0-1.0)
        energy_shape: Target energy curve shape
        output_path: Optional output path

    Returns:
        Path to output file
    """
    try:
        # Load full audio for structure detection
        audio1_raw, sr1 = librosa.load(file1_path, sr=SAMPLE_RATE, mono=False)
        audio2_raw, sr2 = librosa.load(file2_path, sr=SAMPLE_RATE, mono=False)
        audio1 = ensure_stereo(audio1_raw)
        audio2 = ensure_stereo(audio2_raw)

        # Detect structure
        logger.info("Detecting structure for track 1...")
        sections1 = detect_sections(audio1, sr1, "A", stems1.get("vocals"))
        logger.info("Detecting structure for track 2...")
        sections2 = detect_sections(audio2, sr2, "B", stems2.get("vocals"))

        if apply_harmonic_mixing:
            key1 = key1 or estimate_key(audio1, sr1)
            key2 = key2 or estimate_key(audio2, sr2)
            stems2 = _apply_harmonic_mixing_to_stems(stems2, key1, key2)

        # Get beat grids
        beats1, bars1, phrases1 = detect_beat_grid(audio1, sr1)
        beats2, bars2, phrases2 = detect_beat_grid(audio2, sr2)

        # Analyze genre
        track1_features = analyze_track_characteristics(file1_path)
        genre = track1_features.get("genre", "edm")

        # Plan arrangement
        planner = TimelinePlanner()
        plan = planner.plan_arrangement(
            sections1, sections2, beats1, beats2,
            genre=genre, mode=mode,
            aggressiveness=aggressiveness, energy_shape=energy_shape
        )

        logger.info(f"Arrangement plan: {plan.explanation}")

        # Detect BPMs
        bpm1 = detect_bpm(audio1, sr1)
        bpm2 = detect_bpm(audio2, sr2)
        target_bpm = bpm1

        # Render timeline
        renderer = TimelineRenderer(sr=SAMPLE_RATE)
        final_mix = renderer.render_timeline(
            plan, stems1, stems2, bpm1, bpm2, target_bpm
        )

        # Save output
        if output_path is None:
            os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(AUDIO_OUTPUT_DIR, "remix_arrangement.wav")

        sf.write(output_path, final_mix, SAMPLE_RATE)
        logger.info(f"Arrangement mix saved to {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Arrangement mixing failed: {e}")
        raise

# ============================================================================
# CLASSIC MIXING
# ============================================================================

def _render_classic_mix(
    stems1, stems2, mode, crossfade_length, mixing_technique,
    apply_beatmatching, apply_harmonic_mixing, bass_swap,
    eq_low_cut, eq_high_cut, reverb_decay, delay_ms,
    bpm1, bpm2, key1, key2, output_path
):
    """
    Render classic stem-based mix.

    Args:
        stems1: Dict of stems from track 1
        stems2: Dict of stems from track 2
        mode: Remix mode
        crossfade_length: Crossfade duration
        mixing_technique: Mixing technique
        apply_beatmatching: Whether to apply beatmatching
        bass_swap: Enable bass swap
        eq_low_cut: Low frequency cut
        eq_high_cut: High frequency cut
        reverb_decay: Reverb decay
        delay_ms: Delay time
        output_path: Optional output path

    Returns:
        Path to output file
    """
    try:
        if apply_beatmatching:
            stems2 = _apply_beatmatching_to_stems(stems2, bpm1, bpm2)

        if apply_harmonic_mixing:
            stems2 = _apply_harmonic_mixing_to_stems(stems2, key1, key2)

        # Select stems based on mode
        if mode == "mashup":
            vocals = _mix_stems(stems1.get("vocals"), stems2.get("vocals"))
            drums = _mix_stems(stems1.get("drums"), stems2.get("drums"))
            bass = _mix_stems(stems1.get("bass"), stems2.get("bass"))
            other = _mix_stems(stems1.get("other"), stems2.get("other"))
        elif mode == "base_vocals_match_instr":
            vocals = stems1.get("vocals")
            drums = stems2.get("drums")
            bass = stems2.get("bass")
            other = stems2.get("other")
        else:  # base_instr_match_vocals
            vocals = stems2.get("vocals")
            drums = stems1.get("drums")
            bass = stems1.get("bass")
            other = stems1.get("other")

        # Balance volumes
        all_stems = {
            "vocals": vocals,
            "drums": drums,
            "bass": bass,
            "other": other
        }
        balanced = balance_stem_volumes(all_stems)

        # Apply mixing technique
        if mixing_technique == "crossfade":
            final = _apply_crossfade(balanced, crossfade_length)
        elif mixing_technique == "bass_swap":
            final = _apply_bass_swap(balanced, crossfade_length, bass_swap)
        else:  # quick_cut
            final = _apply_quick_cut(balanced)

        # Apply effects
        if eq_low_cut > 0.0 or eq_high_cut < 1.0:
            final = apply_eq_filter(final, eq_low_cut, eq_high_cut, SAMPLE_RATE)

        if reverb_decay > 0.0:
            final = apply_reverb(final, reverb_decay, SAMPLE_RATE)

        if delay_ms > 0.0:
            final = apply_delay(final, delay_ms, 0.3, SAMPLE_RATE)

        # Normalize
        max_val = np.max(np.abs(final))
        if max_val > 1.0:
            final = final / max_val

        # Save output
        if output_path is None:
            os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(AUDIO_OUTPUT_DIR, "remix.wav")

        sf.write(output_path, final, SAMPLE_RATE)
        logger.info(f"Classic mix saved to {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Classic mixing failed: {e}")
        raise

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _apply_beatmatching_to_stems(stems, target_bpm, source_bpm, max_stretch=1.15):
    """Time-stretch a track's stems so source_bpm moves toward target_bpm."""
    if not target_bpm or not source_bpm or source_bpm <= 0:
        return stems

    stretch_rate = float(target_bpm) / float(source_bpm)
    stretch_rate = float(np.clip(stretch_rate, 1.0 / max_stretch, max_stretch))

    if abs(stretch_rate - 1.0) < 0.01:
        return stems

    logger.info(
        "Beatmatching stems: source %.1f BPM -> target %.1f BPM (rate %.3f)",
        source_bpm, target_bpm, stretch_rate
    )

    return {
        name: (
            time_stretch_audio(stem, stretch_rate, SAMPLE_RATE)
            if stem is not None else None
        )
        for name, stem in stems.items()
    }

def _harmonic_shift_semitones(target_key, source_key, max_semitones=2):
    """
    Return a small pitch shift that makes source_key Camelot-compatible with target_key.

    The project key detector uses simplified Camelot numbers, so this keeps the
    correction conservative and only shifts within max_semitones.
    """
    if not target_key or not source_key:
        return 0

    if are_keys_compatible(target_key, source_key):
        return 0

    try:
        source_num = int(source_key[:-1])
        source_letter = source_key[-1]
    except (TypeError, ValueError, IndexError):
        return 0

    candidates = []
    for semitones in range(-max_semitones, max_semitones + 1):
        if semitones == 0:
            continue
        shifted_num = ((source_num - 1 + semitones) % 12) + 1
        shifted_key = f"{shifted_num}{source_letter}"
        if are_keys_compatible(target_key, shifted_key):
            candidates.append((abs(semitones), semitones))

    if not candidates:
        return 0

    return min(candidates)[1]

def _apply_harmonic_mixing_to_stems(stems, target_key, source_key):
    """Pitch-shift a track's stems by a conservative Camelot-compatible amount."""
    semitones = _harmonic_shift_semitones(target_key, source_key)
    if semitones == 0:
        return stems

    logger.info(
        "Harmonic mixing stems: source key %s -> target key %s (shift %+d semitones)",
        source_key, target_key, semitones
    )

    return {
        name: (
            pitch_shift_audio(stem, semitones, SAMPLE_RATE)
            if stem is not None else None
        )
        for name, stem in stems.items()
    }

def _mix_stems(stem1, stem2):
    """
    Mix two stems together.

    Args:
        stem1: First stem (numpy array or None)
        stem2: Second stem (numpy array or None)

    Returns:
        Mixed stem (numpy array) or None
    """
    if stem1 is None and stem2 is None:
        return None
    if stem1 is None:
        return ensure_stereo(stem2)
    if stem2 is None:
        return ensure_stereo(stem1)

    stem1 = ensure_stereo(stem1)
    stem2 = ensure_stereo(stem2)

    # Align to shortest
    min_len = min(len(stem1), len(stem2))
    return (stem1[:min_len] + stem2[:min_len]) / 2.0

def _align_stems_to_same_length(stems):
    """
    Align all stems to the same length (minimum length).

    Handles both mono and stereo audio. Replaces None stems with zeros.

    Args:
        stems: Dict with keys "vocals", "drums", "bass", "other"

    Returns:
        Dict with aligned stems (all same length)
    """
    # Find minimum length (first dimension)
    lengths = []
    for key, stem in stems.items():
        if stem is not None:
            stereo_stem = ensure_stereo(stem)
            stems[key] = stereo_stem
            lengths.append(stereo_stem.shape[0])

    if not lengths:
        logger.warning("All stems are None in _align_stems_to_same_length")
        return {
            key: np.zeros((0, 2), dtype=np.float32)
            for key in stems.keys()
        }

    min_len = min(lengths)

    # Get reference shape from first non-None stem
    reference_stem = None
    for stem in stems.values():
        if stem is not None:
            reference_stem = stem
            break

    if reference_stem is None:
        logger.warning("No reference stem found")
        return stems

    # Align all stems to min_len
    aligned = {}
    for key, stem in stems.items():
        if stem is not None:
            # Trim to minimum length
            aligned[key] = stem[:min_len]
        else:
            # Create zeros with same shape as reference
            aligned[key] = np.zeros((min_len, 2), dtype=reference_stem.dtype)

    return aligned

def _apply_crossfade(stems, fade_length):
    """
    Apply crossfade mixing.

    Args:
        stems: Dict of stems
        fade_length: Fade length in seconds

    Returns:
        Mixed audio array
    """
    # Align all stems to same length
    aligned_stems = _align_stems_to_same_length(stems)

    # Combine all stems
    combined = (
        aligned_stems["vocals"] +
        aligned_stems["drums"] +
        aligned_stems["bass"] +
        aligned_stems["other"]
    )

    # Simple crossfade (fade out at end)
    fade_samples = int(fade_length * SAMPLE_RATE)
    fade_curve = create_crossfade_curve(fade_samples, "s-curve")

    if fade_samples > 0 and len(combined) >= fade_samples:
        combined[-fade_samples:] *= (1.0 - fade_curve)[:, np.newaxis]

    return combined

def _apply_bass_swap(stems, fade_length, enable_bass_swap):
    """
    Apply bass swap technique.

    Args:
        stems: Dict of stems
        fade_length: Fade length in seconds
        enable_bass_swap: Whether to enable bass swap

    Returns:
        Mixed audio array
    """
    # Align all stems to same length
    aligned_stems = _align_stems_to_same_length(stems)

    # Combine all stems
    combined = (
        aligned_stems["vocals"] +
        aligned_stems["drums"] +
        aligned_stems["bass"] +
        aligned_stems["other"]
    )
    return combined  # Simplified implementation

def _apply_quick_cut(stems):
    """
    Apply quick cut (instant switch).

    Args:
        stems: Dict of stems

    Returns:
        Mixed audio array
    """
    # Align all stems to same length
    aligned_stems = _align_stems_to_same_length(stems)

    # Combine all stems
    combined = (
        aligned_stems["vocals"] +
        aligned_stems["drums"] +
        aligned_stems["bass"] +
        aligned_stems["other"]
    )
    return combined  # Simplified implementation
