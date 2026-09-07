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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from scripts.api.auth import auth_context_middleware, require_user
from scripts.api.routers import system
from scripts.api.routes import router
from scripts.core.config import cfg, configure_logging
from scripts.core.logging_utils import get_logger, set_request_id
from scripts.core.paths import ensure_directories

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — replaces deprecated @app.on_event("startup"/"shutdown")
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """
    FastAPI lifespan context manager.
    Runs startup logic before yield, shutdown logic after.
    """
    # ── startup ──────────────────────────────────────────────────────────────
    configure_logging()
    ensure_directories()

    # Wire SSE broadcaster into job store + start heartbeat background task
    from scripts.api.routers.events import start_heartbeat, broadcaster  # noqa: PLC0415
    from scripts.api.jobs import register_sse_hook                       # noqa: PLC0415
    import asyncio as _asyncio                                           # noqa: PLC0415

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
    logger.info("AI RemixMate API started — SSE backbone active")

    yield  # application runs here

    # ── shutdown ─────────────────────────────────────────────────────────────
    from scripts.api.jobs import _executor  # noqa: PLC0415

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
        "Download songs, check harmonic compatibility via the Camelot Wheel, "
        "detect genre, and render phrase-aligned DJ transitions."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

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
# CORS — allow all origins in dev; tighten for production
# ---------------------------------------------------------------------------

_default_origins = [
    "http://localhost:8501",    # Streamlit (legacy primary UI)
    "https://localhost:8501",
    "http://127.0.0.1:8501",
    "https://127.0.0.1:8501",
    "http://localhost:5173",    # Vite React dev server (new primary UI)
    "https://localhost:5173",
    "http://127.0.0.1:5173",
    "https://127.0.0.1:5173",

    "https://chunduri-aditya.github.io",   # GitHub Pages build (static frontend → local API)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(cfg, "api", None) and cfg.api.cors_origins or _default_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _allow_private_network(request: Request, call_next):
    """
    Chrome Private Network Access: an HTTPS page (GitHub Pages) calling a
    local server sends a preflight with Access-Control-Request-Private-Network.
    Approve it so the static frontend can reach the local API.
    """
    response = await call_next(request)
    if (
        request.method == "OPTIONS"
        and request.headers.get("access-control-request-private-network", "").lower() == "true"
    ):
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


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
    )
