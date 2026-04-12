"""
scripts/core/codec_tokens.py — Neural codec tokenization module for AI RemixMate.

Tokenizes Demucs-separated stems into discrete codec tokens using EnCodec or DAC.
These tokens serve as a compressed, learnable representation of audio suitable for
neural remix and transformation pipelines.

Key behaviour
─────────────
• Supports EnCodec (24kHz, bandwidth-variable) and DAC (44.1kHz, high-fidelity)
• Lazy-loads codec models from HuggingFace (transformers) to avoid startup overhead
• Tokenizes each stem independently and saves compressed .npz files
• Provides symmetrical encode/decode for lossy audio recovery
• Integrates with the library filesystem (library/{song}/tokens/{stem}.npz)
• Audit-logs all tokenization operations

Usage
─────
    from scripts.core.codec_tokens import (
        CodecConfig, TokenResult, tokenize_stems, load_stem_tokens, get_token_stats,
        encode_audio, decode_tokens
    )

    # Main workflow
    config = CodecConfig(codec="encodec", bandwidth=6.0)
    result = tokenize_stems("Anyma - Abyss", config=config)
    if result.success:
        print(f"Tokenized {result.total_tokens} tokens across {len(result.tokens)} stems")
        print(f"Token rate: {result.token_rate_hz} Hz, Codebook size: {result.codebook_size}")

    # Load pre-tokenized stem
    tokens = load_stem_tokens("Anyma - Abyss", "vocals")
    if tokens:
        reconstructed_audio = decode_tokens(tokens, config=config)

    # Statistics
    stats = get_token_stats("Anyma - Abyss")
    print(f"Available stems: {stats['stems']}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.core.audit import log_audit
from scripts.core.gpu import get_device
from scripts.core.paths import LIBRARY_DIR, song_dir

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded codec models (module-level caching)
# ---------------------------------------------------------------------------

_encodec_model = None
_encodec_processor = None
_dac_model = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CodecConfig:
    """Configuration for neural codec tokenization."""

    codec: str = "encodec"          # "encodec" or "dac"
    sample_rate: int = 24000        # 24000 for encodec, 44100 for dac
    bandwidth: float = 6.0          # kbps, used for EnCodec (1.5, 3.0, 6.0, 12.0, 24.0)
    device: str = "auto"            # "auto", "mps", "cuda", "cpu"

    def __post_init__(self):
        """Validate codec config and resolve device."""
        if self.codec not in ("encodec", "dac"):
            raise ValueError(f"codec must be 'encodec' or 'dac', got {self.codec}")

        if self.codec == "encodec" and self.sample_rate != 24000:
            log.warning(
                "[codec] EnCodec expects 24000 Hz, got %d. Will resample.",
                self.sample_rate
            )

        if self.codec == "dac" and self.sample_rate != 44100:
            log.warning(
                "[codec] DAC expects 44100 Hz, got %d. Will resample.",
                self.sample_rate
            )

        if self.device == "auto":
            self.device = _resolve_device()


@dataclass
class TokenResult:
    """Result of a tokenization operation."""

    success: bool                           # Did tokenization succeed?
    song_name: str                          # Song name
    codec: str                              # Codec used ("encodec" or "dac")
    tokens: Dict[str, Any] = field(default_factory=dict)  # stem_name → token data dict
    token_rate_hz: float = 0.0              # Token generation rate (Hz)
    num_codebooks: int = 0                  # Number of codebook codes
    codebook_size: int = 0                  # Vocab size per codebook (e.g., 1024)
    total_tokens: int = 0                   # Sum of all frame tokens across all stems
    error: Optional[str] = None             # Error message if not success


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def _resolve_device() -> str:
    """Resolve the device using scripts.core.gpu.get_device()."""
    return get_device()


# ---------------------------------------------------------------------------
# Lazy codec loading
# ---------------------------------------------------------------------------


def _load_encodec(device: str) -> Tuple[Any, Any]:
    """
    Lazy-load EnCodec 24kHz model from HuggingFace.

    Returns (model, processor) tuple.
    Caches the model in a module-level variable.
    """
    global _encodec_model, _encodec_processor

    if _encodec_model is not None and _encodec_processor is not None:
        log.debug("[codec] EnCodec already loaded, returning cached model")
        return _encodec_model, _encodec_processor

    try:
        from transformers import EncodecModel, AutoProcessor

        log.info("[codec] Loading EnCodec 24kHz from HuggingFace...")
        model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        model.to(device)
        model.eval()

        # Load processor for correct input formatting
        processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

        _encodec_model = model
        _encodec_processor = processor
        log.info("[codec] EnCodec loaded successfully")
        return model, processor

    except ImportError as e:
        log.error("[codec] transformers not installed or EnCodec not available: %s", e)
        raise RuntimeError(
            "EnCodec requires 'transformers' library. "
            "Install with: pip install transformers"
        ) from e
    except Exception as e:
        log.error("[codec] Failed to load EnCodec: %s", e)
        raise RuntimeError(f"Failed to load EnCodec: {e}") from e


def _load_dac(device: str) -> Any:
    """
    Lazy-load DAC 44.1kHz model.

    Returns the DAC model instance.
    Caches the model in a module-level variable.
    """
    global _dac_model

    if _dac_model is not None:
        log.debug("[codec] DAC already loaded, returning cached model")
        return _dac_model

    try:
        import dac

        log.info("[codec] Loading DAC 44.1kHz model...")
        model = dac.DAC.from_pretrained("dac")
        model.to(device)
        model.eval()

        _dac_model = model
        log.info("[codec] DAC loaded successfully")
        return model

    except ImportError as e:
        log.error("[codec] dac library not installed: %s", e)
        raise RuntimeError(
            "DAC requires the 'dac' library. "
            "Install with: pip install dac-audio"
        ) from e
    except Exception as e:
        log.error("[codec] Failed to load DAC: %s", e)
        raise RuntimeError(f"Failed to load DAC: {e}") from e


# ---------------------------------------------------------------------------
# Encoding and decoding
# ---------------------------------------------------------------------------


def encode_audio(
    audio: np.ndarray,
    sr: int,
    config: CodecConfig = None,
) -> Dict[str, Any]:
    """
    Encode a single audio array into codec tokens.

    Parameters
    ----------
    audio : np.ndarray
        1-D audio waveform (float32, range [-1, 1])
    sr : int
        Sample rate of the audio
    config : CodecConfig, optional
        Codec configuration. Defaults to EnCodec 24kHz.

    Returns
    -------
    Dict[str, Any]
        Contains:
        - "codes": np.ndarray of shape (n_codebooks, n_frames)
        - "scales": np.ndarray for EnCodec quantization scales (or None for DAC)
        - "bandwidth": bandwidth used (for EnCodec)
        - "token_rate_hz": Generation rate in Hz
        - "codebook_size": Vocab size per codebook
        - "sample_rate": Target sample rate used
    """
    if config is None:
        config = CodecConfig()

    # Convert audio to target sample rate if needed
    if sr != config.sample_rate:
        log.debug(
            "[codec] Resampling audio from %d to %d Hz",
            sr, config.sample_rate
        )
        from scripts.core.gpu import gpu_resample
        audio = gpu_resample(audio, sr, config.sample_rate, device=config.device)

    # Ensure audio is float32 and normalized
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

    try:
        if config.codec == "encodec":
            return _encode_encodec(audio, config)
        elif config.codec == "dac":
            return _encode_dac(audio, config)
        else:
            raise ValueError(f"Unknown codec: {config.codec}")
    except Exception as e:
        log.error("[codec] Encoding failed: %s", e)
        raise


def _encode_encodec(audio: np.ndarray, config: CodecConfig) -> Dict[str, Any]:
    """Encode audio with EnCodec."""
    import torch
    from scripts.core.gpu import to_tensor, to_numpy

    model, processor = _load_encodec(config.device)

    log.debug("[codec] EnCodec encoding with bandwidth %.1f kbps", config.bandwidth)

    with torch.no_grad():
        # Use processor to prepare properly formatted inputs
        if processor is not None:
            inputs = processor(
                raw_audio=audio,
                sampling_rate=config.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(config.device)
            padding_mask = inputs.get("padding_mask", None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(config.device)
            enc_out = model.encode(input_values, padding_mask, bandwidth=config.bandwidth)
        else:
            # Raw tensor path (fallback)
            audio_tensor = to_tensor(audio.reshape(1, 1, -1), device=config.device)
            enc_out = model.encode(audio_tensor, bandwidth=config.bandwidth)

        # enc_out.audio_codes shape: (batch, n_codebooks, n_frames)
        codes_np = enc_out.audio_codes[0].cpu().numpy()  # Remove batch dim
        scale_np = (enc_out.audio_scales[0].cpu().numpy()
                    if enc_out.audio_scales is not None else None)

    # Calculate token rate
    n_frames = codes_np.shape[1]
    duration_sec = audio.shape[0] / config.sample_rate
    token_rate_hz = n_frames / duration_sec if duration_sec > 0 else 0.0

    log.debug(
        "[codec] EnCodec: %d frames, %.1f Hz token rate, %d codebooks",
        n_frames, token_rate_hz, codes_np.shape[0]
    )

    return {
        "codes": codes_np,
        "scales": scale_np,
        "bandwidth": config.bandwidth,
        "token_rate_hz": token_rate_hz,
        "codebook_size": 1024,  # EnCodec default
        "sample_rate": config.sample_rate,
    }


def _encode_dac(audio: np.ndarray, config: CodecConfig) -> Dict[str, Any]:
    """Encode audio with DAC."""
    import torch
    from scripts.core.gpu import to_tensor, to_numpy

    model = _load_dac(config.device)

    # Convert to tensor and add channel dim
    audio_tensor = to_tensor(audio.reshape(1, 1, -1), device=config.device)

    log.debug("[codec] DAC encoding")

    with torch.no_grad():
        # DAC encode
        codes, _, _ = model.encode(audio_tensor)

    # codes shape: (batch, num_codebooks, n_frames)
    codes_np = to_numpy(codes[0])  # Remove batch dim

    # Calculate token rate
    n_frames = codes_np.shape[1]
    duration_sec = audio.shape[0] / config.sample_rate
    token_rate_hz = n_frames / duration_sec if duration_sec > 0 else 0.0

    log.debug(
        "[codec] DAC: %d frames, %.1f Hz token rate, %d codebooks",
        n_frames, token_rate_hz, codes_np.shape[0]
    )

    return {
        "codes": codes_np,
        "scales": None,
        "bandwidth": None,
        "token_rate_hz": token_rate_hz,
        "codebook_size": 1024,  # DAC default
        "sample_rate": config.sample_rate,
    }


def decode_tokens(
    token_data: Dict[str, Any],
    config: CodecConfig = None,
) -> np.ndarray:
    """
    Decode tokens back to audio waveform.

    Parameters
    ----------
    token_data : Dict[str, Any]
        Token data dict (output of encode_audio or loaded from disk)
    config : CodecConfig, optional
        Codec configuration. Uses codec from token_data if available.

    Returns
    -------
    np.ndarray
        Reconstructed audio waveform (float32)
    """
    if config is None:
        config = CodecConfig()

    codes = token_data.get("codes")
    if codes is None:
        raise ValueError("token_data missing 'codes'")

    try:
        if config.codec == "encodec":
            return _decode_encodec(codes, token_data, config)
        elif config.codec == "dac":
            return _decode_dac(codes, token_data, config)
        else:
            raise ValueError(f"Unknown codec: {config.codec}")
    except Exception as e:
        log.error("[codec] Decoding failed: %s", e)
        raise


def _decode_encodec(
    codes: np.ndarray,
    token_data: Dict[str, Any],
    config: CodecConfig,
) -> np.ndarray:
    """Decode EnCodec tokens."""
    import torch
    from scripts.core.gpu import to_tensor, to_numpy

    model, _ = _load_encodec(config.device)

    # Convert codes to tensor (add batch dim): (1, n_codebooks, n_frames)
    codes_tensor = torch.from_numpy(codes[np.newaxis]).long().to(config.device)

    scale = token_data.get("scales")
    if scale is not None:
        scale_tensor = torch.from_numpy(
            scale[np.newaxis] if scale.ndim == len(codes.shape) else scale.reshape(1, 1)
        ).float().to(config.device)
    else:
        scale_tensor = None

    log.debug("[codec] EnCodec decoding")

    with torch.no_grad():
        # transformers EncodecModel.decode expects (audio_codes, audio_scales)
        audio = model.decode(codes_tensor, scale_tensor)

    # audio_values shape: (batch, 1, n_samples)
    decoded = audio.audio_values if hasattr(audio, "audio_values") else audio
    audio_np = to_numpy(decoded[0, 0])

    # Clip to [-1, 1]
    audio_np = np.clip(audio_np, -1.0, 1.0)

    log.debug("[codec] EnCodec decoded %d samples", len(audio_np))
    return audio_np


def _decode_dac(
    codes: np.ndarray,
    token_data: Dict[str, Any],
    config: CodecConfig,
) -> np.ndarray:
    """Decode DAC tokens."""
    import torch
    from scripts.core.gpu import to_tensor, to_numpy

    model = _load_dac(config.device)

    # Convert codes to tensor (add batch dim)
    codes_tensor = to_tensor(codes.reshape(1, *codes.shape), device=config.device).long()

    log.debug("[codec] DAC decoding")

    with torch.no_grad():
        audio = model.decode(codes_tensor)

    # audio shape: (batch, 1, n_samples)
    audio_np = to_numpy(audio[0, 0])

    # Clip to [-1, 1]
    audio_np = np.clip(audio_np, -1.0, 1.0)

    log.debug("[codec] DAC decoded %d samples", len(audio_np))
    return audio_np


# ---------------------------------------------------------------------------
# Main tokenization workflow
# ---------------------------------------------------------------------------


def tokenize_stems(
    song_name: str,
    config: CodecConfig = None,
) -> TokenResult:
    """
    Main function: Tokenize all stems for a song.

    Finds stems in library/{song_name}/, encodes each (vocals, drums, bass, other),
    saves tokens to library/{song_name}/tokens/{stem}.npz, and returns TokenResult.

    Parameters
    ----------
    song_name : str
        Name of the song (directory in library/)
    config : CodecConfig, optional
        Codec configuration. Defaults to EnCodec 24kHz.

    Returns
    -------
    TokenResult
        Detailed result with success flag, tokenized stems, and metadata.
    """
    if config is None:
        config = CodecConfig()

    log.info("[codec] Starting tokenization for song: %s", song_name)
    log_audit("tokenization_start", resource=song_name, metadata={"codec": config.codec})

    song_path = song_dir(song_name)
    if not song_path.exists():
        error = f"Song directory not found: {song_path}"
        log.error("[codec] %s", error)
        log_audit("tokenization_error", resource=song_name, metadata={"error": error})
        return TokenResult(success=False, song_name=song_name, codec=config.codec, error=error)

    # Standard Demucs stems
    stem_names = ["vocals", "drums", "bass", "other"]
    stem_paths = {name: song_path / f"{name}.wav" for name in stem_names}

    # Filter to stems that exist
    existing_stems = {name: path for name, path in stem_paths.items() if path.exists()}

    if not existing_stems:
        error = f"No stems found in {song_path}"
        log.error("[codec] %s", error)
        log_audit("tokenization_error", resource=song_name, metadata={"error": error})
        return TokenResult(success=False, song_name=song_name, codec=config.codec, error=error)

    log.info("[codec] Found %d stems: %s", len(existing_stems), list(existing_stems.keys()))

    # Ensure tokens directory exists
    tokens_dir = song_path / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    result = TokenResult(
        success=True,
        song_name=song_name,
        codec=config.codec,
    )

    total_frames = 0
    sample_rates = set()

    # Encode each stem
    for stem_name, stem_path in existing_stems.items():
        try:
            log.debug("[codec] Loading stem: %s", stem_name)
            import librosa
            audio, sr = librosa.load(str(stem_path), sr=None, mono=True)
            sample_rates.add(sr)

            log.debug("[codec] Encoding stem: %s (sr=%d, len=%d)", stem_name, sr, len(audio))
            token_data = encode_audio(audio, sr, config)

            # Save to npz
            token_file = tokens_dir / f"{stem_name}.npz"
            np.savez_compressed(
                token_file,
                codes=token_data["codes"],
                scales=token_data.get("scales"),
                token_rate_hz=token_data["token_rate_hz"],
                codebook_size=token_data["codebook_size"],
                bandwidth=token_data.get("bandwidth"),
                sample_rate=token_data["sample_rate"],
            )

            log.info("[codec] Saved tokens: %s → %s", stem_name, token_file)

            # Store metadata
            result.tokens[stem_name] = {
                "file": str(token_file),
                "token_rate_hz": token_data["token_rate_hz"],
                "num_frames": token_data["codes"].shape[1],
                "num_codebooks": token_data["codes"].shape[0],
                "codebook_size": token_data["codebook_size"],
            }

            # Accumulate stats
            total_frames += token_data["codes"].shape[1]
            if result.token_rate_hz == 0.0:
                result.token_rate_hz = token_data["token_rate_hz"]
            if result.num_codebooks == 0:
                result.num_codebooks = token_data["codes"].shape[0]
            if result.codebook_size == 0:
                result.codebook_size = token_data["codebook_size"]

        except Exception as e:
            error = f"Failed to encode {stem_name}: {e}"
            log.error("[codec] %s", error)
            result.tokens[stem_name] = {"error": error}

    result.total_tokens = total_frames * result.num_codebooks if result.num_codebooks > 0 else 0

    log.info(
        "[codec] Tokenization complete: %d stems, %d total tokens, %.1f Hz",
        len(result.tokens), result.total_tokens, result.token_rate_hz
    )
    log_audit(
        "tokenization_complete",
        resource=song_name,
        metadata={
            "codec": config.codec,
            "stems": len(result.tokens),
            "total_tokens": result.total_tokens,
            "token_rate_hz": result.token_rate_hz,
        }
    )

    return result


# ---------------------------------------------------------------------------
# Loading and statistics
# ---------------------------------------------------------------------------


def load_stem_tokens(song_name: str, stem: str) -> Optional[Dict[str, Any]]:
    """
    Load saved tokens from disk.

    Parameters
    ----------
    song_name : str
        Name of the song
    stem : str
        Stem name (vocals, drums, bass, other)

    Returns
    -------
    Dict[str, Any] or None
        Token data dict if found, None otherwise.
    """
    token_file = song_dir(song_name) / "tokens" / f"{stem}.npz"

    if not token_file.exists():
        log.warning("[codec] Token file not found: %s", token_file)
        return None

    try:
        data = np.load(token_file, allow_pickle=True)
        result = {
            "codes": data["codes"],
            "scales": data.get("scales"),
            "token_rate_hz": float(data["token_rate_hz"]),
            "codebook_size": int(data["codebook_size"]),
            "sample_rate": int(data["sample_rate"]),
        }

        # Optional fields
        if "bandwidth" in data:
            result["bandwidth"] = float(data["bandwidth"])

        log.debug("[codec] Loaded tokens: %s / %s", song_name, stem)
        return result

    except Exception as e:
        log.error("[codec] Failed to load tokens: %s", e)
        return None


def get_token_stats(song_name: str) -> Dict[str, Any]:
    """
    Return statistics about stored tokens.

    Parameters
    ----------
    song_name : str
        Name of the song

    Returns
    -------
    Dict[str, Any]
        Contains:
        - "stems": List of stem names with tokens
        - "total_tokens": Sum of all tokens
        - "token_rate_hz": Generation rate (if consistent)
        - "codebook_size": Codebook size (if consistent)
        - "token_files": Dict of stem → file path
    """
    tokens_dir = song_dir(song_name) / "tokens"

    if not tokens_dir.exists():
        return {
            "stems": [],
            "total_tokens": 0,
            "token_files": {},
        }

    stats = {
        "stems": [],
        "total_tokens": 0,
        "token_rate_hz": None,
        "codebook_size": None,
        "token_files": {},
    }

    for token_file in sorted(tokens_dir.glob("*.npz")):
        stem_name = token_file.stem

        try:
            data = np.load(token_file, allow_pickle=True)
            codes = data["codes"]
            n_frames = codes.shape[1]
            n_codebooks = codes.shape[0]

            num_tokens = n_frames * n_codebooks
            stats["stems"].append(stem_name)
            stats["total_tokens"] += num_tokens
            stats["token_files"][stem_name] = str(token_file)

            # Track metadata (should be consistent across stems)
            if stats["token_rate_hz"] is None and "token_rate_hz" in data:
                stats["token_rate_hz"] = float(data["token_rate_hz"])
            if stats["codebook_size"] is None and "codebook_size" in data:
                stats["codebook_size"] = int(data["codebook_size"])

        except Exception as e:
            log.warning("[codec] Failed to read token file %s: %s", token_file, e)

    return stats
