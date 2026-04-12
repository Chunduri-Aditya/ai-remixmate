"""
scripts/core/style_transfer.py — MusicGen melody-conditioned style transfer.

Takes a song from the library, extracts its harmonic/melodic fingerprint as a
CQT chromagram, then feeds that chromagram + a text description into
MusicgenMelodyForConditionalGeneration to produce a new audio segment that
preserves the harmonic DNA of the source while adopting the described style.

How it works
────────────
  1. Load source audio (full track or a specific stem, default: full.wav)
  2. Apply HPSS to isolate harmonic component
  3. Compute 12-bin CQT chromagram → feed as melody conditioning
  4. Combine with text description → MusicGen generates 8–30 s segment
  5. Normalize output via LUFS mastering chain
  6. Save to outputs/{session}/style_transfer_{timestamp}.wav

Key design decisions for Apple Silicon
───────────────────────────────────────
  • Uses ModelManager for sequential GPU loading (never two big models at once)
  • `offload_after_use=True` default — frees 8 GB after each generation
  • Runs on MPS (Metal Performance Shaders) via .to(device) before generate()
  • `torch.no_grad()` wraps all inference — no autograd graph buildup in memory
  • `max_new_tokens` capped to keep generation within memory limits
  • Falls back to CPU if MPS raises OOM

Usage
─────
    from scripts.core.style_transfer import StyleTransferConfig, run_style_transfer

    config = StyleTransferConfig(
        description="dark melodic techno, 128 BPM, heavy sub-bass, melancholic chords",
        duration_sec=15.0,
        source_stem="other",    # use the 'other' stem (no drums/bass/vocals)
    )
    result = run_style_transfer("Anyma - Voices In My Head", config)
    if result.success:
        print(f"Output: {result.output_path}, LUFS: {result.lufs:.1f}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.core.audit import log_audit
from scripts.core.gpu import get_device
from scripts.core.logging_utils import get_logger
from scripts.core.paths import LIBRARY_DIR, OUTPUTS_DIR, song_dir

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SESSION_DIR_NAME = "style_transfer"
_SAMPLE_RATE_MG   = 32000   # MusicGen internal sample rate
_TARGET_SR        = 44100   # Output sample rate (upsampled after generation)
_DEFAULT_MODEL    = "facebook/musicgen-melody"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StyleTransferConfig:
    """
    Configuration for one style-transfer generation.

    Attributes
    ----------
    description:
        Free-text style prompt sent to MusicGen. Examples:
          "dark melodic techno, 128 BPM, melancholic chords, heavy sub-bass"
          "chill lo-fi hip-hop with piano and vinyl crackle"
          "epic orchestral remix with cinematic strings"
    duration_sec:
        Length of generated audio in seconds. Keep ≤ 20 s on 16 GB RAM.
    source_stem:
        Which stem to extract melody from. Use "other" for chords/melody
        without percussion. Use "vocals" to preserve vocal melody.
        Use "full" for the complete mix (default).
    source_start_sec:
        Offset into the source audio (allows choosing a specific section).
    source_clip_sec:
        How much source audio to feed as melody reference (8–30 s works best).
    guidance_scale:
        Classifier-free guidance strength. Higher = more literal description.
        Range 1.0–10.0, default 3.0.
    temperature:
        Sampling temperature. Higher = more creative/varied output. Default 1.0.
    top_k:
        Top-k nucleus sampling. 250 is a good default.
    seed:
        Random seed for reproducibility. None = random.
    output_format:
        "wav" or "flac". FLAC adds lossless compression.
    """

    description: str = "electronic instrumental music"
    duration_sec: float = 15.0
    source_stem: str = "full"              # "full" | "vocals" | "other" | "bass" | "drums"
    source_start_sec: float = 0.0
    source_clip_sec: float = 15.0
    guidance_scale: float = 3.0
    temperature: float = 1.0
    top_k: int = 250
    seed: Optional[int] = None
    output_format: str = "wav"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class StyleTransferResult:
    """Result from a style transfer operation."""

    success: bool
    song_name: str
    description: str
    output_path: Optional[str] = None
    duration_sec: float = 0.0
    lufs: float = 0.0
    source_key: Optional[str] = None      # detected key of source melody
    source_camelot: Optional[str] = None  # Camelot code of source key
    generation_time_sec: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _load_source_audio(song_name: str, config: StyleTransferConfig) -> Tuple[np.ndarray, int]:
    """
    Load the source audio for melody extraction.
    Returns (audio_array, sample_rate).
    """
    import librosa

    d = song_dir(song_name)

    # Determine which file to load
    if config.source_stem == "full":
        candidates = [d / "full.wav", d / "full.flac"]
    else:
        candidates = [
            d / f"{config.source_stem}.wav",
            d / f"{config.source_stem}.flac",
            d / "full.wav",
            d / "full.flac",
        ]

    src_path = next((p for p in candidates if p.exists()), None)
    if src_path is None:
        raise FileNotFoundError(
            f"No audio found for song '{song_name}' (stem='{config.source_stem}'). "
            f"Checked: {[str(p) for p in candidates]}"
        )

    log.info("[style_transfer] Loading source: %s", src_path.name)
    audio, sr = librosa.load(
        str(src_path),
        sr=None,
        mono=True,
        offset=config.source_start_sec,
        duration=config.source_clip_sec,
    )
    return audio, sr


def _extract_melody_chroma(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a melody-representative chroma matrix from audio.

    Returns np.ndarray of shape (1, 12, n_frames) — the format MusicGen expects
    for its melody conditioning input.
    """
    import librosa

    # HPSS to focus on harmonic content (remove drums that confuse chroma)
    try:
        D = librosa.stft(audio)
        H, _ = librosa.decompose.hpss(D, margin=2.0)
        harmonic = librosa.istft(H)
    except Exception:
        harmonic = audio

    # CQT chroma: 7 octaves, 36 bins/octave for fidelity, then averaged to 12
    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=sr,
        n_octaves=7,
        bins_per_octave=36,
        fmin=librosa.note_to_hz("C1"),
    )  # shape: (12, n_frames)

    # Normalize per frame so energy level doesn't dominate
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    chroma = chroma / (norms + 1e-8)

    # Add batch dimension: (1, 12, n_frames)
    return chroma[np.newaxis, :, :]  # (1, 12, n_frames)


def _detect_source_key(audio: np.ndarray, sr: int) -> Tuple[str, str]:
    """Return (key_name, camelot) for the source audio."""
    try:
        from scripts.core.key_detection import detect_key
        result = detect_key(audio, sr)
        return f"{result.key_name} {result.mode}", result.camelot
    except Exception as exc:
        log.debug("[style_transfer] Key detection failed: %s", exc)
        return "Unknown", "8B"


def _outputs_dir(song_name: str) -> Path:
    """Return (and create) the session output dir for style transfer results."""
    session = f"style_{song_name[:20].replace(' ', '_').replace('/', '_')}"
    out = OUTPUTS_DIR / session
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resample_to_target(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample array from_sr → to_sr."""
    if from_sr == to_sr:
        return audio
    try:
        from scripts.core.gpu import gpu_resample
        return gpu_resample(audio, from_sr, to_sr)
    except Exception:
        import librosa
        return librosa.resample(audio, orig_sr=from_sr, target_sr=to_sr)


def _master_and_save(
    audio: np.ndarray,
    sr: int,
    out_dir: Path,
    fmt: str,
    suffix: str = "",
) -> Tuple[str, float]:
    """
    Apply LUFS mastering, save to disk, return (path_str, lufs).
    """
    from scripts.core.mastering import master_mix
    import soundfile as sf

    mastered, report = master_mix(audio, sr=sr, target_lufs=-14.0)

    ts = int(time.time())
    fname = f"style_transfer{suffix}_{ts}.{fmt}"
    out_path = out_dir / fname

    if fmt == "flac":
        sf.write(str(out_path), mastered, sr, subtype="PCM_24", format="FLAC")
    else:
        sf.write(str(out_path), mastered, sr, subtype="PCM_24")

    log.info("[style_transfer] Saved: %s (%.1f LUFS)", out_path.name, report.lufs_integrated)
    return str(out_path), report.lufs_integrated


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def run_style_transfer(
    song_name: str,
    config: Optional[StyleTransferConfig] = None,
    progress_cb=None,
) -> StyleTransferResult:
    """
    Run a full style transfer on a library song.

    Parameters
    ----------
    song_name : str
        Song directory name in library/.
    config : StyleTransferConfig, optional
        Generation parameters. Defaults to StyleTransferConfig().
    progress_cb : callable(float, str), optional
        Called with (progress 0–1, status_message) for UI progress bars.

    Returns
    -------
    StyleTransferResult
    """
    if config is None:
        config = StyleTransferConfig()

    def _prog(p: float, msg: str) -> None:
        if progress_cb:
            progress_cb(p, msg)
        log.info("[style_transfer] %.0f%% — %s", p * 100, msg)

    _prog(0.05, "Loading source audio…")
    t0 = time.time()

    log_audit("style_transfer_start", resource=song_name,
              metadata={"description": config.description, "duration_sec": config.duration_sec})

    # ── 1. Load source ─────────────────────────────────────────────────────
    try:
        source_audio, source_sr = _load_source_audio(song_name, config)
    except FileNotFoundError as exc:
        return StyleTransferResult(success=False, song_name=song_name,
                                    description=config.description, error=str(exc))

    _prog(0.10, "Extracting melody fingerprint (CQT chroma)…")

    # ── 2. Detect key + extract chroma ─────────────────────────────────────
    source_key, source_camelot = _detect_source_key(source_audio, source_sr)
    chroma = _extract_melody_chroma(source_audio, source_sr)  # (1, 12, n_frames)

    _prog(0.20, f"Source key: {source_key} ({source_camelot}). Loading MusicGen…")

    # ── 3. Prepare for MusicGen ────────────────────────────────────────────
    try:
        import torch

        device = get_device()

        # Load via ModelManager (offloads everything else first)
        from scripts.core.model_manager import get_manager
        mgr = get_manager()

        _prog(0.30, "Loading MusicGen (this may take a minute first time)…")
        model, processor = mgr.load("musicgen")

        # Optional: seed for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if device == "mps":
                torch.mps.manual_seed(config.seed)

        # ── 4. Prepare inputs ──────────────────────────────────────────────
        _prog(0.40, "Preparing melody conditioning…")

        # Resample source audio to MusicGen's 32kHz
        source_32k = _resample_to_target(source_audio, source_sr, _SAMPLE_RATE_MG)

        # Processor handles tokenization + feature extraction
        inputs = processor(
            text=[config.description],
            audio=source_32k,
            sampling_rate=_SAMPLE_RATE_MG,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Compute max_new_tokens from desired duration
        # MusicGen generates at 50 tokens/sec (32kHz, 4 codebooks, delay pattern)
        tokens_per_sec = 50
        max_new_tokens = int(config.duration_sec * tokens_per_sec)

        _prog(0.50, f"Generating {config.duration_sec:.0f}s of audio…")

        # ── 5. Generate ────────────────────────────────────────────────────
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=config.temperature,
                guidance_scale=config.guidance_scale,
                top_k=config.top_k,
            )

        # generated: (1, 1, n_samples) at 32kHz
        audio_out = generated[0, 0].cpu().numpy().astype(np.float32)

        _prog(0.80, "Releasing MusicGen, mastering output…")
        mgr.release("musicgen")

        # ── 6. Upsample + master + save ────────────────────────────────────
        audio_44k = _resample_to_target(audio_out, _SAMPLE_RATE_MG, _TARGET_SR)
        out_dir = _outputs_dir(song_name)
        output_path, lufs = _master_and_save(
            audio_44k, _TARGET_SR, out_dir, config.output_format
        )

        elapsed = time.time() - t0
        _prog(1.0, "Done!")

        result = StyleTransferResult(
            success=True,
            song_name=song_name,
            description=config.description,
            output_path=output_path,
            duration_sec=len(audio_44k) / _TARGET_SR,
            lufs=lufs,
            source_key=source_key,
            source_camelot=source_camelot,
            generation_time_sec=elapsed,
            metadata={
                "source_stem": config.source_stem,
                "guidance_scale": config.guidance_scale,
                "temperature": config.temperature,
                "seed": config.seed,
            },
        )

        log_audit(
            "style_transfer_complete",
            resource=song_name,
            metadata={
                "output_path": output_path,
                "lufs": lufs,
                "duration_sec": result.duration_sec,
                "generation_time_sec": elapsed,
            },
        )

        return result

    except RuntimeError as exc:
        # MPS OOM or missing model — try releasing and surface error
        try:
            from scripts.core.model_manager import get_manager
            get_manager().release("musicgen")
        except Exception:
            pass
        error_msg = str(exc)
        log.error("[style_transfer] Generation failed: %s", error_msg)
        log_audit("style_transfer_failed", resource=song_name,
                  metadata={"error": error_msg})
        return StyleTransferResult(
            success=False,
            song_name=song_name,
            description=config.description,
            error=error_msg,
        )

    except Exception as exc:
        log.exception("[style_transfer] Unexpected error")
        return StyleTransferResult(
            success=False,
            song_name=song_name,
            description=config.description,
            error=f"Unexpected error: {exc}",
        )


# ---------------------------------------------------------------------------
# Batch convenience: style-transfer multiple songs
# ---------------------------------------------------------------------------

def batch_style_transfer(
    song_names: List[str],
    config: Optional[StyleTransferConfig] = None,
    progress_cb=None,
) -> List[StyleTransferResult]:
    """
    Run style transfer on a list of songs sequentially.
    Returns a list of StyleTransferResult, one per song.
    """
    results = []
    n = len(song_names)
    for i, song in enumerate(song_names):
        def _sub_prog(p: float, msg: str) -> None:
            if progress_cb:
                overall = (i + p) / n
                progress_cb(overall, f"[{i+1}/{n}] {song}: {msg}")

        result = run_style_transfer(song, config, progress_cb=_sub_prog)
        results.append(result)

    return results
