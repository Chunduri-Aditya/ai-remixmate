"""
scripts/api/routers/generative.py — AI Studio generative endpoints.

POST /ai/style-transfer        — MusicGen melody-conditioned style transfer
POST /ai/inpaint               — VampNet masked acoustic token inpainting
POST /ai/tokenize              — Tokenize stems with EnCodec or DAC
GET  /ai/models                — AI model registry status
"""

from fastapi import APIRouter, BackgroundTasks

from scripts.api import jobs as job_store
from scripts.api.routers._helpers import _require_song, _check_job_cap
from scripts.api.schemas import (
    InpaintRequest,
    JobResponse,
    JobType,
    ModelInfo,
    ModelStatusResponse,
    StyleTransferRequest,
    TokenizeRequest,
)
from scripts.api.tasks import (
    task_inpaint_transition,
    task_style_transfer,
    task_tokenize_stems,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Style Transfer  — POST /ai/style-transfer
# ---------------------------------------------------------------------------

@router.post(
    "/ai/style-transfer",
    response_model=JobResponse,
    status_code=202,
    tags=["ai-studio"],
    summary="MusicGen melody-conditioned style transfer",
    description=(
        "Generate new audio that preserves the harmonic/melodic fingerprint of "
        "the source song while adopting the described style. Uses MusicGen Melody "
        "(facebook/musicgen-melody) via the centralized ModelManager. "
        "Returns a job_id — poll /jobs/{job_id} for progress and result."
    ),
)
def style_transfer(req: StyleTransferRequest, background_tasks: BackgroundTasks = None):
    _check_job_cap()
    _require_song(req.song_name)
    job_id = job_store.create_job(JobType.ANALYZE, {"type": "style_transfer"})
    job_store.submit_job(
        job_id,
        task_style_transfer,
        song_name=req.song_name,
        description=req.description,
        duration_sec=req.duration_sec,
        source_stem=req.source_stem,
        source_start_sec=req.source_start_sec,
        source_clip_sec=req.source_clip_sec,
        guidance_scale=req.guidance_scale,
        temperature=req.temperature,
        top_k=req.top_k,
        seed=req.seed,
        output_format=req.output_format,
    )
    return job_store.job_to_response(job_store.get_job(job_id))


# ---------------------------------------------------------------------------
# Inpainting — POST /ai/inpaint
# ---------------------------------------------------------------------------

@router.post(
    "/ai/inpaint",
    response_model=JobResponse,
    status_code=202,
    tags=["ai-studio"],
    summary="VampNet masked acoustic token inpainting",
    description=(
        "Generate a creative transition bridge between two songs. "
        "Uses VampNet bidirectional transformer over DAC tokens. "
        "Falls back to cosine token interpolation if VampNet is not installed. "
        "Returns a job_id — poll /jobs/{job_id} for result."
    ),
)
def inpaint_transition(req: InpaintRequest, background_tasks: BackgroundTasks = None):
    _check_job_cap()
    _require_song(req.song_a)
    _require_song(req.song_b)
    job_id = job_store.create_job(JobType.ANALYZE, {"type": "inpaint_transition"})
    job_store.submit_job(
        job_id,
        task_inpaint_transition,
        song_a=req.song_a,
        song_b=req.song_b,
        mask_type=req.mask_type,
        prefix_bars=req.prefix_bars,
        suffix_bars=req.suffix_bars,
        periodic_prompt=req.periodic_prompt,
        n_codebooks_to_keep=req.n_codebooks_to_keep,
        sampling_steps=req.sampling_steps,
        rand_mask_intensity=req.rand_mask_intensity,
        source_stem=req.source_stem,
        bpm=req.bpm,
        output_format=req.output_format,
    )
    return job_store.job_to_response(job_store.get_job(job_id))


# ---------------------------------------------------------------------------
# Tokenize — POST /ai/tokenize
# ---------------------------------------------------------------------------

@router.post(
    "/ai/tokenize",
    response_model=JobResponse,
    status_code=202,
    tags=["ai-studio"],
    summary="Tokenize Demucs stems with EnCodec or DAC",
    description=(
        "Encodes all stems for a library song into discrete codec tokens "
        "(EnCodec 24kHz or DAC 44.1kHz) and saves .npz files to "
        "library/{song}/tokens/. Required before VampNet inpainting with DAC."
    ),
)
def tokenize_stems(req: TokenizeRequest, background_tasks: BackgroundTasks = None):
    _check_job_cap()
    _require_song(req.song_name)
    job_id = job_store.create_job(JobType.ANALYZE, {"type": "tokenize_stems"})
    job_store.submit_job(
        job_id,
        task_tokenize_stems,
        song_name=req.song_name,
        codec=req.codec,
        bandwidth=req.bandwidth,
    )
    return job_store.job_to_response(job_store.get_job(job_id))


# ---------------------------------------------------------------------------
# Model status — GET /ai/models
# ---------------------------------------------------------------------------

@router.get(
    "/ai/models",
    response_model=ModelStatusResponse,
    tags=["ai-studio"],
    summary="AI model registry status",
    description=(
        "Returns which generative models are currently loaded in memory, "
        "estimated VRAM usage, and device information."
    ),
)
def model_status():
    try:
        from scripts.core.model_manager import get_manager
        status = get_manager().status()
        models = [
            ModelInfo(
                name=name,
                loaded=info["loaded"],
                vram_estimate_gb=info["vram_estimate_gb"],
                description=info["description"],
                last_used=info["last_used"],
            )
            for name, info in status["models"].items()
        ]
        return ModelStatusResponse(
            device=status["device"],
            mps_allocated_gb=status.get("mps_allocated_gb", 0.0),
            max_vram_gb=status["max_vram_gb"],
            models=models,
        )
    except Exception as exc:
        return ModelStatusResponse(device="unknown", models=[])
