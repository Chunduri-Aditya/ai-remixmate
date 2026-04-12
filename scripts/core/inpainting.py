"""
scripts/core/inpainting.py — VampNet masked acoustic token inpainting.

VampNet is a bidirectional transformer that operates directly on DAC tokens.
It fills masked regions of a token sequence conditioned on the surrounding
context — this is the ideal primitive for:

  • Creative transition fills between two songs
  • Beat-locked variation generation (vamping over a groove)
  • Harmonic interpolation between sections

How the pipeline works
──────────────────────
  1. Load audio for song_a (or a stem) and song_b (or a stem)
  2. Tokenize both with DAC 44.1kHz (9 codebooks, ~86 Hz)
  3. Create a mask over the gap region (end of A, start of B)
  4. Run VampNet to fill the masked tokens via multi-pass sampling
  5. Decode tokens back to audio via DAC decoder
  6. Crossfade with the DJ engine for a smooth handoff
  7. Master + save

Mask types
──────────
  periodic      — keep every P-th timestep (rhythmic anchor, freest generation)
  prefix_suffix — keep N bars from A's end + N bars from B's start, mask middle
  beat_driven   — keep beat-aligned tokens, mask off-beats (groove-preserving)
  compression   — keep only first K codebooks (preserve coarse structure, vary timbre)

VampNet is OPTIONAL — if not installed, falls back to a simpler cross-synthesis
using EnCodec tokens + cosine-interpolation in the continuous latent space,
which is less powerful but always available.

References
──────────
  VampNet: Music Generation via Masked Acoustic Token Modeling
  Garcia et al., ISMIR 2023 — https://arxiv.org/abs/2307.04686
  GitHub: https://github.com/hugofloresgarcia/vampnet
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

_DAC_SR       = 44100
_DAC_TOKEN_HZ = 86        # ~86 tokens per second at 44.1kHz with stride 512
_SESSION_DIR  = "inpaint"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InpaintConfig:
    """
    Configuration for one VampNet inpainting operation.

    Attributes
    ----------
    mask_type:
        How to construct the mask over the token sequence.
        "periodic"      — keep every `periodic_prompt`-th token.
        "prefix_suffix" — keep song_a prefix + song_b suffix, inpaint middle.
        "beat_driven"   — keep beat-position tokens, mask off-beats.
        "compression"   — only keep first `n_codebooks_to_keep` codebooks.
    prefix_bars:
        How many bars from the end of song_a to include as prefix context.
    suffix_bars:
        How many bars from the start of song_b to include as suffix context.
    periodic_prompt:
        For mask_type="periodic": keep every P-th token (P=4 ≈ 4-th note anchoring).
    n_codebooks_to_keep:
        For mask_type="compression": keep this many codebooks (coarse structure).
    sampling_steps:
        Number of VampNet masked prediction passes. More = better quality.
    rand_mask_intensity:
        Fraction of non-anchored tokens to randomly mask (0.0–1.0).
    source_stem:
        Which stem to use for melody/harmonic source. "full" = complete mix.
    bpm:
        BPM for beat-aligned mask types. None = auto-detect.
    crossfade_bars:
        How many bars of the DJ crossfade to use around the inpainted region.
    output_format:
        "wav" or "flac".
    """

    mask_type: str = "prefix_suffix"
    prefix_bars: int = 4
    suffix_bars: int = 4
    periodic_prompt: int = 4
    n_codebooks_to_keep: int = 3
    sampling_steps: int = 36
    rand_mask_intensity: float = 0.7
    source_stem: str = "full"
    bpm: Optional[float] = None
    crossfade_bars: int = 4
    output_format: str = "wav"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class InpaintResult:
    """Result from an inpainting operation."""

    success: bool
    song_a: str
    song_b: str
    output_path: Optional[str] = None
    method: str = "vampnet"           # "vampnet" | "cosine_fallback"
    duration_sec: float = 0.0
    lufs: float = 0.0
    generation_time_sec: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audio loading helpers
# ---------------------------------------------------------------------------

def _load_stem(song_name: str, stem: str, duration_sec: float = None,
               offset_sec: float = 0.0) -> Tuple[np.ndarray, int]:
    """Load a stem (or full audio) from the library. Returns (audio, sr)."""
    import librosa
    d = song_dir(song_name)

    if stem == "full":
        candidates = [d / "full.wav", d / "full.flac"]
    else:
        candidates = [
            d / f"{stem}.wav",
            d / f"{stem}.flac",
            d / "full.wav",
            d / "full.flac",
        ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            f"No audio for '{song_name}' (stem='{stem}'). "
            f"Checked: {[str(p) for p in candidates]}"
        )

    audio, sr = librosa.load(
        str(src), sr=None, mono=True,
        offset=offset_sec,
        duration=duration_sec,
    )
    return audio, sr


def _detect_bpm(audio: np.ndarray, sr: int) -> float:
    """Auto-detect BPM with librosa beat tracker."""
    try:
        import librosa
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, hop_length=512)
        return float(np.atleast_1d(tempo)[0])
    except Exception:
        return 120.0


def _bars_to_tokens(n_bars: int, bpm: float, token_hz: float = _DAC_TOKEN_HZ) -> int:
    """Convert bar count to token count given BPM and token rate."""
    bar_duration_sec = (60.0 / bpm) * 4  # 4/4 time
    return int(n_bars * bar_duration_sec * token_hz)


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------

def build_periodic_mask(
    n_tokens: int,
    period: int = 4,
    rand_intensity: float = 0.7,
) -> np.ndarray:
    """
    Keep every period-th token (rhythmic anchor), randomly mask rest.

    Returns bool mask of shape (n_tokens,). True = KEEP, False = MASK (inpaint).
    """
    mask = np.zeros(n_tokens, dtype=bool)
    mask[::period] = True               # anchor every period-th token

    # Randomly keep additional tokens at (1 - rand_intensity) fraction
    random_keep = np.random.rand(n_tokens) > rand_intensity
    mask |= random_keep

    return mask


def build_prefix_suffix_mask(
    n_tokens: int,
    prefix_tokens: int,
    suffix_tokens: int,
) -> np.ndarray:
    """
    Keep prefix_tokens from start and suffix_tokens from end.
    Mask everything in between (the transition gap).

    Returns bool mask. True = KEEP.
    """
    mask = np.zeros(n_tokens, dtype=bool)
    if prefix_tokens > 0:
        mask[:prefix_tokens] = True
    if suffix_tokens > 0:
        mask[-suffix_tokens:] = True
    return mask


def build_beat_driven_mask(
    n_tokens: int,
    bpm: float,
    token_hz: float = _DAC_TOKEN_HZ,
    rand_intensity: float = 0.5,
) -> np.ndarray:
    """
    Keep tokens that align with beat positions, randomly mask off-beats.

    Returns bool mask. True = KEEP.
    """
    beat_interval_tokens = int(round(60.0 / bpm * token_hz))  # tokens per beat
    mask = np.zeros(n_tokens, dtype=bool)

    for i in range(0, n_tokens, beat_interval_tokens):
        mask[i] = True                  # downbeat anchor

    # Random off-beat keep
    random_keep = np.random.rand(n_tokens) > rand_intensity
    mask |= random_keep

    return mask


def build_compression_mask(
    n_codebooks: int,
    n_tokens: int,
    keep_codebooks: int = 3,
) -> np.ndarray:
    """
    Keep first `keep_codebooks` codebooks entirely, mask the rest.

    Returns 2D bool mask of shape (n_codebooks, n_tokens).
    True = KEEP.
    """
    mask = np.zeros((n_codebooks, n_tokens), dtype=bool)
    mask[:keep_codebooks, :] = True
    return mask


# ---------------------------------------------------------------------------
# DAC tokenization helpers
# ---------------------------------------------------------------------------

def _tokenize_audio_dac(
    audio: np.ndarray,
    sr: int,
    device: str,
) -> Tuple[np.ndarray, Any]:
    """
    Tokenize audio with DAC. Returns (codes, dac_model).
    codes shape: (n_codebooks, n_frames).
    """
    import torch
    import torchaudio

    from scripts.core.model_manager import get_manager
    dac_model = get_manager().load("dac")

    # Resample to DAC's 44.1kHz if needed
    if sr != _DAC_SR:
        resampler = torchaudio.transforms.Resample(sr, _DAC_SR).to(device)
        t = torch.from_numpy(audio).unsqueeze(0).to(device)
        audio_44k = resampler(t).squeeze(0).cpu().numpy()
    else:
        audio_44k = audio

    # Preprocess and encode
    t = torch.from_numpy(audio_44k.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    t = dac_model.preprocess(t, _DAC_SR)

    with torch.no_grad():
        z, codes, latents, _, _ = dac_model.encode(t)

    # codes: (1, n_codebooks, n_frames)
    return codes[0].cpu().numpy(), dac_model


def _decode_dac(codes: np.ndarray, dac_model, device: str) -> np.ndarray:
    """Decode DAC codes back to waveform. Returns 1D float32 audio at 44.1kHz."""
    import torch

    codes_t = torch.from_numpy(codes).unsqueeze(0).long().to(device)  # (1, n_cb, T)

    with torch.no_grad():
        # Get quantized latent from codes
        z = dac_model.quantizer.from_codes(codes_t)[0]
        audio_t = dac_model.decode(z)

    audio_np = audio_t[0, 0].cpu().numpy().astype(np.float32)
    return np.clip(audio_np, -1.0, 1.0)


# ---------------------------------------------------------------------------
# VampNet inpainting
# ---------------------------------------------------------------------------

def _inpaint_vampnet(
    combined_codes: np.ndarray,
    mask_1d: np.ndarray,
    config: InpaintConfig,
    device: str,
) -> np.ndarray:
    """
    Run VampNet over combined_codes using mask_1d.

    combined_codes: (n_codebooks, n_frames)
    mask_1d:        (n_frames,) bool — True = keep
    Returns filled_codes: (n_codebooks, n_frames)
    """
    import vampnet
    import torch

    interface = vampnet.interface.Interface.default()

    # VampNet expects AudioSignal objects; we feed codes directly via the
    # lower-level API which accepts pre-tokenized inputs
    codes_t = torch.from_numpy(combined_codes).unsqueeze(0).long()  # (1, n_cb, T)
    mask_t  = torch.from_numpy(~mask_1d).unsqueeze(0).unsqueeze(0)  # (1, 1, T) — True=MASK

    log.info(
        "[inpainting] VampNet: %d tokens, %.1f%% masked, %d steps",
        combined_codes.shape[1],
        (~mask_1d).mean() * 100,
        config.sampling_steps,
    )

    with torch.no_grad():
        filled_codes = interface.vamp(
            codes_t,
            mask=mask_t,
            sampling_steps=config.sampling_steps,
            mask_temperature=10.5,
        )

    return filled_codes[0].cpu().numpy()  # (n_codebooks, n_frames)


# ---------------------------------------------------------------------------
# Cosine interpolation fallback (no VampNet required)
# ---------------------------------------------------------------------------

def _inpaint_cosine_fallback(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    n_interp_tokens: int,
    device: str,
) -> np.ndarray:
    """
    Fallback when VampNet is not available.
    Interpolates between the last frames of A and the first frames of B using
    nearest-neighbor token blending — not as creative as VampNet but always works.

    codes_a, codes_b: (n_codebooks, n_frames)
    Returns combined codes: (n_codebooks, n_a + n_interp + n_b)
    """
    n_cb = codes_a.shape[0]
    alpha_vals = np.linspace(0, 1, n_interp_tokens)

    # Simple: for each interp step, pick from a or b with probability alpha
    interp = np.zeros((n_cb, n_interp_tokens), dtype=np.int64)
    for t, alpha in enumerate(alpha_vals):
        if alpha < 0.5:
            # Weighted toward A
            interp[:, t] = codes_a[:, -1]  # last frame of A
        else:
            # Weighted toward B
            interp[:, t] = codes_b[:, 0]   # first frame of B

    return np.concatenate([codes_a, interp, codes_b], axis=1)


# ---------------------------------------------------------------------------
# Main inpainting function
# ---------------------------------------------------------------------------

def run_inpainting(
    song_a: str,
    song_b: str,
    config: Optional[InpaintConfig] = None,
    progress_cb=None,
) -> InpaintResult:
    """
    Generate a creative transition between song_a and song_b using VampNet.

    Parameters
    ----------
    song_a : str
        Library song name (the outgoing track).
    song_b : str
        Library song name (the incoming track).
    config : InpaintConfig, optional
        Inpainting configuration.
    progress_cb : callable(float, str), optional
        Progress callback for UI.

    Returns
    -------
    InpaintResult
    """
    if config is None:
        config = InpaintConfig()

    def _prog(p: float, msg: str) -> None:
        if progress_cb:
            progress_cb(p, msg)
        log.info("[inpainting] %.0f%% — %s", p * 100, msg)

    t0 = time.time()
    device = get_device()

    log_audit("inpainting_start", resource=f"{song_a} → {song_b}",
              metadata={"mask_type": config.mask_type})

    # ── 1. Detect BPM ─────────────────────────────────────────────────────
    _prog(0.05, f"Loading {song_a}…")
    try:
        audio_a, sr_a = _load_stem(song_a, config.source_stem)
        audio_b, sr_b = _load_stem(song_b, config.source_stem)
    except FileNotFoundError as exc:
        return InpaintResult(success=False, song_a=song_a, song_b=song_b, error=str(exc))

    bpm = config.bpm
    if bpm is None:
        _prog(0.10, "Auto-detecting BPM…")
        bpm = _detect_bpm(audio_a, sr_a)
        log.info("[inpainting] Auto-detected BPM: %.1f", bpm)

    # ── 2. Extract tail of A and head of B ────────────────────────────────
    _prog(0.15, "Extracting transition segments…")
    bar_sec = (60.0 / bpm) * 4

    # Last N bars of A and first N bars of B
    tail_a_sec  = config.prefix_bars * bar_sec
    head_b_sec  = config.suffix_bars * bar_sec

    tail_a = audio_a[max(0, len(audio_a) - int(tail_a_sec * sr_a)):]
    head_b = audio_b[:int(head_b_sec * sr_b)]

    # ── 3. Tokenize with DAC ──────────────────────────────────────────────
    _prog(0.25, "Tokenizing with DAC (loading codec)…")
    try:
        codes_a, dac_model = _tokenize_audio_dac(tail_a, sr_a, device)
        codes_b, _         = _tokenize_audio_dac(head_b, sr_b, device)
    except Exception as exc:
        return InpaintResult(success=False, song_a=song_a, song_b=song_b,
                              error=f"DAC tokenization failed: {exc}")

    n_cb   = codes_a.shape[0]
    n_a    = codes_a.shape[1]
    n_b    = codes_b.shape[1]
    n_total = n_a + n_b

    log.info("[inpainting] codes_a: %s, codes_b: %s", codes_a.shape, codes_b.shape)

    # Combined token sequence
    combined = np.concatenate([codes_a, codes_b], axis=1)  # (n_cb, n_a + n_b)

    # ── 4. Build mask ──────────────────────────────────────────────────────
    _prog(0.35, f"Building {config.mask_type} mask…")

    if config.mask_type == "prefix_suffix":
        mask_1d = build_prefix_suffix_mask(n_total, n_a, n_b)

    elif config.mask_type == "periodic":
        mask_1d = build_periodic_mask(
            n_total, config.periodic_prompt, config.rand_mask_intensity
        )

    elif config.mask_type == "beat_driven":
        mask_1d = build_beat_driven_mask(
            n_total, bpm, rand_intensity=config.rand_mask_intensity
        )

    elif config.mask_type == "compression":
        # 2D mask (n_codebooks, n_tokens) — project to 1D for logging
        mask_2d = build_compression_mask(n_cb, n_total, config.n_codebooks_to_keep)
        mask_1d = mask_2d.all(axis=0)  # 1D view for stats
        # Use mask_2d in VampNet if supported; fall back to broadcast 1D
    else:
        # Default: prefix/suffix
        mask_1d = build_prefix_suffix_mask(n_total, n_a, n_b)

    pct_masked = (~mask_1d).mean() * 100
    log.info("[inpainting] Mask: %.1f%% tokens will be inpainted", pct_masked)

    # ── 5. Inpaint ─────────────────────────────────────────────────────────
    _prog(0.45, "Running VampNet inpainting…")
    method = "vampnet"

    try:
        filled_codes = _inpaint_vampnet(combined, mask_1d, config, device)
        log.info("[inpainting] VampNet succeeded")

    except (ImportError, RuntimeError, AttributeError) as exc:
        # Graceful fallback: cosine interpolation
        log.warning(
            "[inpainting] VampNet unavailable (%s). Using cosine fallback.", exc
        )
        method = "cosine_fallback"
        # Inpaint gap = 4 bars worth of tokens
        n_gap = _bars_to_tokens(config.crossfade_bars, bpm)
        filled_codes = _inpaint_cosine_fallback(codes_a, codes_b, n_gap, device)

    # ── 6. Decode back to audio ────────────────────────────────────────────
    _prog(0.75, "Decoding tokens back to audio…")
    try:
        audio_out = _decode_dac(filled_codes, dac_model, device)
    except Exception as exc:
        return InpaintResult(success=False, song_a=song_a, song_b=song_b,
                              error=f"DAC decode failed: {exc}")

    # Release DAC after decode
    from scripts.core.model_manager import get_manager
    get_manager().release("dac")

    # ── 7. Master + save ───────────────────────────────────────────────────
    _prog(0.88, "Mastering and saving…")
    try:
        from scripts.core.mastering import master_mix
        import soundfile as sf

        mastered, report = master_mix(audio_out, sr=_DAC_SR, target_lufs=-14.0)

        out_dir = OUTPUTS_DIR / f"inpaint_{song_a[:10].replace(' ', '_')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        fname = f"inpaint_{ts}.{config.output_format}"
        out_path = out_dir / fname

        if config.output_format == "flac":
            sf.write(str(out_path), mastered, _DAC_SR, subtype="PCM_24", format="FLAC")
        else:
            sf.write(str(out_path), mastered, _DAC_SR, subtype="PCM_24")

        elapsed = time.time() - t0
        _prog(1.0, "Done!")

        log_audit(
            "inpainting_complete",
            resource=f"{song_a} → {song_b}",
            metadata={"output_path": str(out_path), "method": method, "bpm": bpm},
        )

        return InpaintResult(
            success=True,
            song_a=song_a,
            song_b=song_b,
            output_path=str(out_path),
            method=method,
            duration_sec=len(mastered) / _DAC_SR,
            lufs=report.lufs_integrated,
            generation_time_sec=elapsed,
            metadata={
                "bpm": bpm,
                "mask_type": config.mask_type,
                "pct_masked": round(pct_masked, 1),
                "n_codebooks": n_cb,
                "sampling_steps": config.sampling_steps if method == "vampnet" else 0,
            },
        )

    except Exception as exc:
        log.exception("[inpainting] Save/master failed")
        return InpaintResult(
            success=False,
            song_a=song_a,
            song_b=song_b,
            error=f"Save failed: {exc}",
        )
