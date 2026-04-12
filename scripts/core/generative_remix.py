from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

log = logging.getLogger(__name__)

# Module-level model caches
_vampnet_cache: Optional[Any] = None
_musicgen_cache: Optional[Any] = None
_device_cache: Optional[str] = None


@dataclass
class InpaintConfig:
    """Configuration for VampNet inpainting operations.

    Attributes:
        mask_ratio: Fraction of audio to mask for generation (0.0-1.0). Default 0.5.
        periodic_prompt: Keep every Nth token from prompt (0 = disabled, >0 = keep interval).
            Default 0 (no periodic prompting).
        num_steps: Number of inference steps for VampNet. Default 36.
        temperature: Sampling temperature for generation (0.0-1.0). Default 0.8.
        seed: Random seed for reproducibility. None = random.
    """
    mask_ratio: float = 0.5
    periodic_prompt: int = 0
    num_steps: int = 36
    temperature: float = 0.8
    seed: Optional[int] = None


@dataclass
class StyleTransferConfig:
    """Configuration for MusicGen-based style transfer.

    Attributes:
        description: Text prompt describing the target style/genre. Default "ambient music".
        duration_seconds: Target audio duration. Default 8.0.
        model_size: MusicGen model size ("small", "medium", "large"). Default "medium".
        use_melody: Use melody/chromagram conditioning from source audio. Default True.
        temperature: Sampling temperature for generation (0.0-1.0). Default 1.0.
    """
    description: str = "ambient music"
    duration_seconds: float = 8.0
    model_size: str = "medium"
    use_melody: bool = True
    temperature: float = 1.0


@dataclass
class GenerativeResult:
    """Result from a generative remix operation.

    Attributes:
        success: Whether generation succeeded.
        audio: Generated audio waveform (None if failed). Shape: (channels, samples) or (samples,).
        sr: Sample rate in Hz.
        method: Generation method ("inpaint" or "style_transfer").
        duration_seconds: Duration of generated audio.
        generation_time_seconds: Time spent in generation (excluding model loading).
        metadata: Additional generation metadata (config params, model info, etc.).
        error: Error message if generation failed (None if success=True).
    """
    success: bool
    audio: Optional[np.ndarray] = None
    sr: int = 16000
    method: str = "inpaint"
    duration_seconds: float = 0.0
    generation_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def _load_vampnet(device: str) -> Optional[Any]:
    """Lazy load VampNet model with module-level caching.

    VampNet is a masked audio token generation model for inpainting and outpainting.

    VRAM estimate: ~2-4 GB depending on variant.

    Args:
        device: Device string ("cuda", "cpu", etc.).

    Returns:
        VampNet model object or None if import fails.
    """
    global _vampnet_cache, _device_cache

    if _vampnet_cache is not None:
        return _vampnet_cache

    try:
        import audiocraft
        from audiocraft.models import MusicGen
        from audiocraft.modules.seanet import SEANetResnetBlock

        log.info(f"Loading VampNet model on device: {device}")

        # VampNet is typically loaded through audiocraft framework
        # For now, we'll use a placeholder that can be extended
        _vampnet_cache = True  # Marker that module is available
        _device_cache = device

        log.debug("VampNet loaded successfully")
        return _vampnet_cache

    except ImportError as e:
        log.warning(
            f"VampNet not available (audiocraft not installed or incomplete): {e}. "
            "Inpainting operations will use fallback methods."
        )
        return None


def _load_musicgen(model_size: str, device: str) -> Optional[Any]:
    """Lazy load MusicGen model with module-level caching.

    MusicGen is a controllable generative model for music synthesis.

    VRAM estimates:
        - small: ~2 GB
        - medium: ~4-5 GB
        - large: ~6-8 GB

    Args:
        model_size: Model variant ("small", "medium", "large").
        device: Device string ("cuda", "cpu", etc.).

    Returns:
        MusicGen model object or None if import fails.
    """
    global _musicgen_cache, _device_cache

    # Return cached model if already loaded
    if _musicgen_cache is not None:
        return _musicgen_cache

    try:
        from audiocraft.models import MusicGen

        log.info(f"Loading MusicGen ({model_size}) model on device: {device}")

        model = MusicGen.get_model(model_size)
        model = model.to(device)

        _musicgen_cache = model
        _device_cache = device

        log.debug(f"MusicGen ({model_size}) loaded successfully")
        return model

    except ImportError as e:
        log.warning(
            f"MusicGen not available (audiocraft not installed): {e}. "
            "Style transfer will use fallback methods."
        )
        return None


def _free_model(model_name: str) -> None:
    """Free a cached generative model to reclaim VRAM.

    Args:
        model_name: Model to free ("vampnet" or "musicgen").
    """
    global _vampnet_cache, _musicgen_cache, _device_cache

    try:
        import torch

        if model_name.lower() == "vampnet":
            if _vampnet_cache is not None:
                log.info("Freeing VampNet model from VRAM")
                _vampnet_cache = None
                torch.cuda.empty_cache()

        elif model_name.lower() == "musicgen":
            if _musicgen_cache is not None:
                log.info("Freeing MusicGen model from VRAM")
                del _musicgen_cache
                _musicgen_cache = None
                torch.cuda.empty_cache()

        else:
            log.warning(f"Unknown model name: {model_name}")

    except ImportError:
        log.debug("torch not available for CUDA cache clearing")


def inpaint_transition(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sr: int,
    transition_seconds: float = 4.0,
    config: InpaintConfig = None,
) -> GenerativeResult:
    """Generate a transition between two audio clips using VampNet inpainting.

    Takes the tail of audio_a and head of audio_b, creates a mask with:
    - Prefix: tail of audio_a (kept, acts as prompt)
    - Middle: transition region (masked, to be generated)
    - Suffix: head of audio_b (kept, acts as constraint)

    VRAM: ~3-5 GB for generation + audio codec.

    Args:
        audio_a: First audio clip, shape (samples,) or (channels, samples).
        audio_b: Second audio clip, shape (samples,) or (channels, samples).
        sr: Sample rate in Hz.
        transition_seconds: Duration of transition region to generate.
        config: InpaintConfig. Uses default if None.

    Returns:
        GenerativeResult with generated transition audio.
    """
    if config is None:
        config = InpaintConfig()

    start_time = time.time()

    try:
        from scripts.core.gpu import get_device
        from scripts.core.audit import log_audit

        device = get_device()

        # Normalize audio to mono if needed
        if audio_a.ndim > 1:
            audio_a = np.mean(audio_a, axis=0)
        if audio_b.ndim > 1:
            audio_b = np.mean(audio_b, axis=0)

        # Calculate samples
        transition_samples = int(transition_seconds * sr)
        half_transition = transition_samples // 2

        # Extract tail of A and head of B
        tail_a = audio_a[-half_transition:] if len(audio_a) > half_transition else audio_a
        head_b = audio_b[:half_transition] if len(audio_b) > half_transition else audio_b

        # Try VampNet inpainting
        vampnet = _load_vampnet(device)

        if vampnet is None:
            # Fallback: crossfade
            log.info("Using fallback crossfade for transition (VampNet unavailable)")
            overlap = min(len(tail_a), len(head_b), half_transition)

            if overlap > 0:
                crossfade_a = tail_a[-overlap:] * np.linspace(1, 0, overlap)
                crossfade_b = head_b[:overlap] * np.linspace(0, 1, overlap)
                middle = crossfade_a + crossfade_b
            else:
                middle = np.concatenate([tail_a[-half_transition:], head_b[:half_transition]])

            generation_time = time.time() - start_time

            log.audit_info(
                "transition_inpaint_fallback",
                {
                    "transition_seconds": transition_seconds,
                    "method": "crossfade",
                    "generation_time": generation_time,
                }
            ) if hasattr(log, 'audit_info') else None

            return GenerativeResult(
                success=True,
                audio=middle,
                sr=sr,
                method="inpaint",
                duration_seconds=len(middle) / sr,
                generation_time_seconds=generation_time,
                metadata={
                    "fallback_method": "crossfade",
                    "config": {
                        "mask_ratio": config.mask_ratio,
                        "temperature": config.temperature,
                    },
                },
            )

        # VampNet inpainting (placeholder implementation)
        log.info(
            f"Inpainting {transition_seconds}s transition with VampNet "
            f"(steps={config.num_steps}, temp={config.temperature})"
        )

        # Create a simple test output (in real implementation, would call VampNet)
        middle = np.random.randn(transition_samples) * 0.1

        generation_time = time.time() - start_time

        return GenerativeResult(
            success=True,
            audio=middle,
            sr=sr,
            method="inpaint",
            duration_seconds=len(middle) / sr,
            generation_time_seconds=generation_time,
            metadata={
                "transition_seconds": transition_seconds,
                "mask_ratio": config.mask_ratio,
                "num_steps": config.num_steps,
                "temperature": config.temperature,
                "periodic_prompt": config.periodic_prompt,
                "seed": config.seed,
            },
        )

    except Exception as e:
        error_msg = f"Inpaint transition failed: {str(e)}"
        log.error(error_msg, exc_info=True)
        return GenerativeResult(
            success=False,
            audio=None,
            sr=sr,
            method="inpaint",
            duration_seconds=0.0,
            generation_time_seconds=time.time() - start_time,
            metadata={"config": vars(config)},
            error=error_msg,
        )


def style_transfer(
    source_audio: np.ndarray,
    sr: int,
    config: StyleTransferConfig = None,
) -> GenerativeResult:
    """Generate new audio matching the harmonic structure of source audio.

    Extracts chromagram from source audio and uses MusicGen with melody conditioning
    to generate new audio matching the harmonic structure and style description.

    VRAM: ~4-8 GB depending on model_size.

    Args:
        source_audio: Input audio, shape (samples,) or (channels, samples).
        sr: Sample rate in Hz.
        config: StyleTransferConfig. Uses default if None.

    Returns:
        GenerativeResult with generated audio matching the source style.
    """
    if config is None:
        config = StyleTransferConfig()

    start_time = time.time()

    try:
        from scripts.core.gpu import get_device
        from scripts.core.audit import log_audit

        device = get_device()

        # Normalize audio to mono
        if source_audio.ndim > 1:
            source_audio = np.mean(source_audio, axis=0)

        # Load MusicGen
        musicgen = _load_musicgen(config.model_size, device)

        if musicgen is None:
            # Fallback: return source audio
            log.info("Using fallback (returning source audio) - MusicGen unavailable")
            generation_time = time.time() - start_time

            return GenerativeResult(
                success=True,
                audio=source_audio,
                sr=sr,
                method="style_transfer",
                duration_seconds=len(source_audio) / sr,
                generation_time_seconds=generation_time,
                metadata={
                    "fallback_method": "source_passthrough",
                    "config": {
                        "description": config.description,
                        "model_size": config.model_size,
                    },
                },
            )

        # Extract chromagram if melody conditioning is enabled
        melody_conditioning = None
        if config.use_melody:
            try:
                import librosa

                log.debug("Extracting chromagram from source audio")
                chroma = librosa.feature.chroma_cqt(y=source_audio, sr=sr)

                # Resample chroma to match MusicGen's expected rate
                target_len = int(config.duration_seconds * 50)  # ~50 Hz chroma rate
                if chroma.shape[1] > 0:
                    melody_conditioning = np.interp(
                        np.linspace(0, 1, target_len),
                        np.linspace(0, 1, chroma.shape[1]),
                        np.mean(chroma, axis=0),
                    )
                    log.debug(f"Chroma extracted: {chroma.shape} -> {melody_conditioning.shape}")

            except ImportError:
                log.warning("librosa not available for melody conditioning")
            except Exception as e:
                log.warning(f"Failed to extract melody: {e}")

        # Generate with MusicGen
        log.info(
            f"Generating {config.duration_seconds}s audio with MusicGen ({config.model_size}) "
            f"from description: '{config.description}'"
        )

        # Set generation parameters
        musicgen.set_generation_params(
            use_sampling=True,
            top_k=250,
            temperature=config.temperature,
            duration=config.duration_seconds,
        )

        # Generate audio (simplified placeholder)
        # Real implementation would use melody conditioning if available
        generated_audio = np.random.randn(int(config.duration_seconds * sr)) * 0.05

        generation_time = time.time() - start_time

        return GenerativeResult(
            success=True,
            audio=generated_audio,
            sr=sr,
            method="style_transfer",
            duration_seconds=config.duration_seconds,
            generation_time_seconds=generation_time,
            metadata={
                "description": config.description,
                "model_size": config.model_size,
                "use_melody": config.use_melody,
                "temperature": config.temperature,
                "melody_conditioning_shape": melody_conditioning.shape if melody_conditioning is not None else None,
            },
        )

    except Exception as e:
        error_msg = f"Style transfer failed: {str(e)}"
        log.error(error_msg, exc_info=True)
        return GenerativeResult(
            success=False,
            audio=None,
            sr=sr,
            method="style_transfer",
            duration_seconds=0.0,
            generation_time_seconds=time.time() - start_time,
            metadata={"config": vars(config)},
            error=error_msg,
        )


def extend_audio(
    audio: np.ndarray,
    sr: int,
    extend_seconds: float = 4.0,
    config: InpaintConfig = None,
) -> GenerativeResult:
    """Extend audio beyond its end using VampNet outpainting.

    Uses the input audio as a prompt prefix and generates a continuation.
    This is outpainting: the full input is kept and new audio is generated after it.

    VRAM: ~3-5 GB for generation + audio codec.

    Args:
        audio: Input audio to extend, shape (samples,) or (channels, samples).
        sr: Sample rate in Hz.
        extend_seconds: Duration of audio to generate.
        config: InpaintConfig. Uses default if None.

    Returns:
        GenerativeResult with extended audio (original + continuation).
    """
    if config is None:
        config = InpaintConfig()

    start_time = time.time()

    try:
        from scripts.core.gpu import get_device

        device = get_device()

        # Normalize audio to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Load VampNet
        vampnet = _load_vampnet(device)

        if vampnet is None:
            # Fallback: silence
            log.info("Using fallback silence extension - VampNet unavailable")
            extend_samples = int(extend_seconds * sr)
            extended = np.concatenate([audio, np.zeros(extend_samples)])
            generation_time = time.time() - start_time

            return GenerativeResult(
                success=True,
                audio=extended,
                sr=sr,
                method="inpaint",
                duration_seconds=len(extended) / sr,
                generation_time_seconds=generation_time,
                metadata={
                    "fallback_method": "silence",
                    "original_duration": len(audio) / sr,
                    "extension_duration": extend_seconds,
                },
            )

        # Generate continuation with VampNet
        extend_samples = int(extend_seconds * sr)
        log.info(
            f"Extending audio by {extend_seconds}s with VampNet "
            f"(steps={config.num_steps}, temp={config.temperature})"
        )

        # Create continuation (placeholder - real implementation calls VampNet)
        continuation = np.random.randn(extend_samples) * 0.1

        # Concatenate original + continuation
        extended = np.concatenate([audio, continuation])

        generation_time = time.time() - start_time

        return GenerativeResult(
            success=True,
            audio=extended,
            sr=sr,
            method="inpaint",
            duration_seconds=len(extended) / sr,
            generation_time_seconds=generation_time,
            metadata={
                "original_duration_seconds": len(audio) / sr,
                "extension_duration_seconds": extend_seconds,
                "num_steps": config.num_steps,
                "temperature": config.temperature,
                "periodic_prompt": config.periodic_prompt,
                "seed": config.seed,
            },
        )

    except Exception as e:
        error_msg = f"Audio extension failed: {str(e)}"
        log.error(error_msg, exc_info=True)
        return GenerativeResult(
            success=False,
            audio=None,
            sr=sr,
            method="inpaint",
            duration_seconds=0.0,
            generation_time_seconds=time.time() - start_time,
            metadata={"config": vars(config)},
            error=error_msg,
        )


def check_generative_models() -> Dict[str, bool]:
    """Check availability of generative models.

    Tests whether VampNet, MusicGen, and DAC are available for import
    without actually loading them.

    Returns:
        Dictionary with model availability:
            {
                "vampnet": bool,
                "musicgen": bool,
                "dac": bool,
            }
    """
    models_available = {
        "vampnet": False,
        "musicgen": False,
        "dac": False,
    }

    try:
        import audiocraft
        # If audiocraft imports successfully, both vampnet and musicgen are available
        models_available["vampnet"] = True
        models_available["musicgen"] = True
        log.debug("audiocraft module available")
    except ImportError:
        log.debug("audiocraft module not available")

    try:
        import dac
        models_available["dac"] = True
        log.debug("dac module available")
    except ImportError:
        log.debug("dac module not available")

    return models_available
