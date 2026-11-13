"""
scripts/api/main.py — FastAPI application entry point.

Start the server:
    uvicorn scripts.api.main:app --reload --port 8000

Or via the remixmate CLI:
    python -m scripts.api.main

Interactive docs:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from scripts.api.auth import auth_context_middleware, require_user
from scripts.api.cors import resolve_cors_origins
from scripts.api.routers import legal, system
from scripts.api.routes import router
from scripts.core.config import cfg, configure_logging
from scripts.core.logging_utils import get_logger, set_request_id
from scripts.core.paths import ensure_directories

logger = get_logger(__name__)

# First retention sweep of every outputs/<uid>/ runs this long after startup;
# later sweeps follow cfg.retention.sweep_interval_hours.
SWEEP_START_DELAY_SEC = 60

# ---------------------------------------------------------------------------
# Lifespan — replaces deprecated @app.on_event("startup"/"shutdown")
# ---------------------------------------------------------------------------


async def _retention_sweep_loop() -> None:
    """Sweep every user's outputs after the start delay, then on the interval.
    The filesystem work runs in a thread so the event loop stays free."""
    import asyncio as _asyncio  # noqa: PLC0415

    from scripts.core.mix_retention import sweep_all_users  # noqa: PLC0415

    await _asyncio.sleep(SWEEP_START_DELAY_SEC)
    while True:
        results = await _asyncio.to_thread(sweep_all_users)
        removed = sum(len(r.removed) for r in results.values())
        if removed:
            logger.info(f"mix retention: background sweep removed {removed} session(s)")
        await _asyncio.sleep(float(cfg.retention.sweep_interval_hours) * 3600)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """
    FastAPI lifespan context manager.
    Runs startup logic before yield, shutdown logic after.
    """
    # ── startup ──────────────────────────────────────────────────────────────
    configure_logging()
    ensure_directories()

    from scripts.core import legal_store  # noqa: PLC0415

    legal_store.init_schema()

    # Wire SSE broadcaster into job store + start heartbeat background task
    import asyncio as _asyncio  # noqa: PLC0415

    from scripts.api.jobs import register_sse_hook  # noqa: PLC0415
    from scripts.api.routers.events import (  # noqa: PLC0415
        broadcaster,
        closing,
        start_heartbeat,
    )

    # Capture the running event loop here (async context) so worker threads
    # can schedule coroutines safely via call_soon_threadsafe.
    # asyncio.get_event_loop() raises RuntimeError on Python 3.12 when called
    # from a ThreadPoolExecutor worker that has no current event loop.
    _event_loop = _asyncio.get_running_loop()

    def _sync_emit(event_type: str, job_dict: dict) -> None:
        """Thread-safe shim: schedule async broadcast from sync job-store callbacks."""
        try:
            _event_loop.call_soon_threadsafe(
                _event_loop.create_task,
                broadcaster.broadcast(event_type, job_dict),
            )
        except Exception:  # noqa: BLE001
            pass

    register_sse_hook(_sync_emit)
    start_heartbeat()
    sweep_task = _asyncio.create_task(_retention_sweep_loop())
    logger.info("AI RemixMate API started — SSE backbone active")

    yield  # application runs here

    # ── shutdown ─────────────────────────────────────────────────────────────
    from scripts.api.jobs import _executor  # noqa: PLC0415

    # End open SSE streams first, so they do not hold the server open while
    # the executor drains.
    closing.set()
    sweep_task.cancel()
    (sweep_outcome,) = await _asyncio.gather(sweep_task, return_exceptions=True)
    if isinstance(sweep_outcome, Exception):
        logger.warning(f"mix retention: background sweep had failed: {sweep_outcome}")

    logger.info("Shutdown signal received — waiting for in-flight jobs…")
    _executor.shutdown(wait=True, cancel_futures=False)
    logger.info("All jobs finished — shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI RemixMate API",
    description=(
        "Metadata-first DJ remix engine. "
        "Scan a local music folder, check harmonic compatibility via the Camelot Wheel, "
        "and render phrase-aligned DJ transitions and sets."
    ),
    version="0.2.0",
    contact={"name": "Aditya Chunduri", "url": "https://github.com/Chunduri-Aditya/ai-remixmate"},
    license_info={"name": "MIT", "url": "https://github.com/Chunduri-Aditya/ai-remixmate/blob/main/LICENSE"},
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Validation errors — same 422 shape as FastAPI's default, minus the ``input``
# echo, so no response body reflects a raw client value (e.g. a path string).
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_error_without_echo(request: Request, exc: RequestValidationError):
    detail = [{k: v for k, v in err.items() if k != "input"} for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(detail)})


# ---------------------------------------------------------------------------
# Request ID middleware — adds unique tracing ID to each request
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    Middleware that assigns a unique request ID to each request.
    Uses X-Request-ID header if provided, otherwise generates a UUID.
    The request ID is stored in a context variable for all downstream handlers.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.debug("Request completed", extra={"method": request.method, "path": request.url.path})
    return response


# ---------------------------------------------------------------------------
# CORS — Vite localhost in dev; prod from REMIXMATE_FRONTEND_URL / CORS_ORIGINS
# ---------------------------------------------------------------------------

_cfg_origins = getattr(getattr(cfg, "api", None), "cors_origins", None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=resolve_cors_origins(_cfg_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth — resolves the acting user per request and pins current_user (Phase 4).
# No-op ("local") unless REMIXMATE_AUTH_ENABLED is truthy. Registered as HTTP
# middleware, not a dependency, so sync handlers + the job store see the
# contextvar (a dependency's .set() would not reach a sibling sync endpoint).
# ---------------------------------------------------------------------------

app.middleware("http")(auth_context_middleware)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# Health probes mount first and OPEN — no require_user — so liveness/readiness
# stay reachable when auth is on.
app.include_router(system.router)

# Public legal pages + DMCA intake (Phase 8) — open; must not read per-user data.
app.include_router(legal.router)

# Everything else requires an authenticated user (401 when auth is on and the
# request carries no valid Supabase token; "local" when auth is off).
app.include_router(router, dependencies=[Depends(require_user)])

# ---------------------------------------------------------------------------
# Root — the SPA is a separate deploy (Cloudflare Pages). This API serves no UI.
# ---------------------------------------------------------------------------


def _root_payload():
    """302 to the frontend when REMIXMATE_FRONTEND_URL is set, else a small
    pointer at the API's own entry points (keeps local dev sane)."""
    frontend = os.environ.get("REMIXMATE_FRONTEND_URL", "").strip()
    if frontend:
        return RedirectResponse(url=frontend, status_code=302)
    return JSONResponse({
        "service": "ai-remixmate API",
        "docs": "/docs",
        "health": "/health/ready",
        "frontend": "set REMIXMATE_FRONTEND_URL to redirect here",
        "author": "Aditya Chunduri",
        "repository": "https://github.com/Chunduri-Aditya/ai-remixmate",
    })


@app.get("/", include_in_schema=False)
async def root():
    return _root_payload()


@app.get("/app", include_in_schema=False)
async def app_root():
    return _root_payload()


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = getattr(getattr(cfg, "api", None), "host", "0.0.0.0")
    port = getattr(getattr(cfg, "api", None), "port", 8000)
    reload = getattr(getattr(cfg, "api", None), "reload", True)

    uvicorn.run(
        "scripts.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        # uvicorn waits for open connections before lifespan shutdown, and an
        # SSE stream never closes on its own; this bounds that wait so
        # closing.set() and the executor drain actually run.
        timeout_graceful_shutdown=5,
    )
