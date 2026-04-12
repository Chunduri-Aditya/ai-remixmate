"""
scripts/api/task_modules/generative.py — AI generative task functions.

task_style_transfer        — MusicGen melody-conditioned style transfer
task_inpaint_transition    — VampNet masked acoustic token inpainting
task_tokenize_stems        — Tokenize stems with EnCodec or DAC
"""

from pathlib import Path
from typing import Any, Dict, Optional

from scripts.api.jobs import update_job
from scripts.core.audit import log_audit
from scripts.core.paths import OUTPUTS_DIR


def task_style_transfer(
    job_id: str,
    song_name: str,
    description: str,
    duration_sec: float = 15.0,
    source_stem: str = "full",
    source_start_sec: float = 0.0,
    source_clip_sec: float = 15.0,
    guidance_scale: float = 3.0,
    temperature: float = 1.0,
    top_k: int = 250,
    seed: int = None,
    output_format: str = "wav",
) -> dict:
    """
    Run MusicGen melody-conditioned style transfer on a library song.

    Preserves the harmonic/melodic fingerprint of the source while generating
    new audio in the described style. Runs on MPS (Apple Silicon) via ModelManager.
    """
    from scripts.core.style_transfer import StyleTransferConfig, run_style_transfer

    log_audit("style_transfer_start", resource=song_name, job_id=job_id,
              metadata={"description": description, "duration_sec": duration_sec})

    config = StyleTransferConfig(
        description=description,
        duration_sec=duration_sec,
        source_stem=source_stem,
        source_start_sec=source_start_sec,
        source_clip_sec=source_clip_sec,
        guidance_scale=guidance_scale,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        output_format=output_format,
    )

    def _progress(p: float, msg: str) -> None:
        update_job(job_id, progress=round(p, 3), message=msg)

    result = run_style_transfer(song_name, config, progress_cb=_progress)

    if not result.success:
        raise RuntimeError(result.error or "Style transfer failed")

    # Build audio URL for frontend
    output_path = Path(result.output_path) if result.output_path else None
    audio_url = None
    if output_path and output_path.exists():
        # Relative to OUTPUTS_DIR for the /files/ endpoint
        try:
            rel = output_path.relative_to(OUTPUTS_DIR)
            audio_url = f"/files/{rel}"
        except ValueError:
            audio_url = None

    log_audit("style_transfer_complete", resource=song_name, job_id=job_id,
              metadata={"output_path": result.output_path, "lufs": result.lufs,
                        "generation_time_sec": result.generation_time_sec})

    return {
        "song_name": song_name,
        "description": description,
        "output_path": result.output_path,
        "audio_url": audio_url,
        "duration_sec": result.duration_sec,
        "lufs": result.lufs,
        "source_key": result.source_key,
        "source_camelot": result.source_camelot,
        "generation_time_sec": result.generation_time_sec,
        "method": "musicgen-melody",
        "success": True,
    }


def task_inpaint_transition(
    job_id: str,
    song_a: str,
    song_b: str,
    mask_type: str = "prefix_suffix",
    prefix_bars: int = 4,
    suffix_bars: int = 4,
    periodic_prompt: int = 4,
    n_codebooks_to_keep: int = 3,
    sampling_steps: int = 36,
    rand_mask_intensity: float = 0.7,
    source_stem: str = "full",
    bpm: float = None,
    output_format: str = "wav",
) -> dict:
    """
    Generate a creative transition bridge between two songs using VampNet.

    Falls back to cosine token interpolation if VampNet is not installed.
    """
    from scripts.core.inpainting import InpaintConfig, run_inpainting

    log_audit("inpainting_start", resource=f"{song_a} → {song_b}", job_id=job_id,
              metadata={"mask_type": mask_type})

    config = InpaintConfig(
        mask_type=mask_type,
        prefix_bars=prefix_bars,
        suffix_bars=suffix_bars,
        periodic_prompt=periodic_prompt,
        n_codebooks_to_keep=n_codebooks_to_keep,
        sampling_steps=sampling_steps,
        rand_mask_intensity=rand_mask_intensity,
        source_stem=source_stem,
        bpm=bpm,
        output_format=output_format,
    )

    def _progress(p: float, msg: str) -> None:
        update_job(job_id, progress=round(p, 3), message=msg)

    result = run_inpainting(song_a, song_b, config, progress_cb=_progress)

    if not result.success:
        raise RuntimeError(result.error or "Inpainting failed")

    output_path = Path(result.output_path) if result.output_path else None
    audio_url = None
    if output_path and output_path.exists():
        try:
            rel = output_path.relative_to(OUTPUTS_DIR)
            audio_url = f"/files/{rel}"
        except ValueError:
            audio_url = None

    log_audit("inpainting_complete", resource=f"{song_a} → {song_b}", job_id=job_id,
              metadata={"output_path": result.output_path, "method": result.method})

    return {
        "song_a": song_a,
        "song_b": song_b,
        "output_path": result.output_path,
        "audio_url": audio_url,
        "method": result.method,
        "duration_sec": result.duration_sec,
        "lufs": result.lufs,
        "generation_time_sec": result.generation_time_sec,
        "metadata": result.metadata,
        "success": True,
    }


def task_tokenize_stems(
    job_id: str,
    song_name: str,
    codec: str = "encodec",
    bandwidth: float = 6.0,
) -> dict:
    """Tokenize all Demucs stems for a library song using EnCodec or DAC."""
    from scripts.core.codec_tokens import CodecConfig, tokenize_stems

    update_job(job_id, progress=0.05, message=f"Tokenizing '{song_name}' with {codec}…")

    config = CodecConfig(codec=codec, bandwidth=bandwidth)
    result = tokenize_stems(song_name, config=config)

    if not result.success:
        raise RuntimeError(result.error or "Tokenization failed")

    update_job(job_id, progress=1.0, message="Tokenization complete")

    return {
        "song_name": song_name,
        "codec": result.codec,
        "stems": list(result.tokens.keys()),
        "total_tokens": result.total_tokens,
        "token_rate_hz": result.token_rate_hz,
        "num_codebooks": result.num_codebooks,
        "codebook_size": result.codebook_size,
        "success": True,
    }
