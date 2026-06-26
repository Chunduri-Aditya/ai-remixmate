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

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from scripts.api.routes import router
from scripts.core.config import cfg, configure_logging
from scripts.core.logging_utils import get_logger, set_request_id
from scripts.core.paths import ensure_directories

logger = get_logger(__name__)

_UI_DIR = Path(__file__).parents[1] / "ui" / "static"
_UI_DIR.mkdir(parents=True, exist_ok=True)

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

    def _sync_emit(event_type: str, job_dict: dict) -> None:
        """Thread-safe shim: schedule async broadcast from sync job-store callbacks."""
        try:
            loop = _asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(
                    loop.create_task,
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
# Routes
# ---------------------------------------------------------------------------

app.include_router(router)

# ---------------------------------------------------------------------------
# Serve static HTML UI at root (INTERNAL/EXPERIMENTAL — not the primary UI)
# Primary UI: Streamlit on port 8501 — see docs/GUIDE.md
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the static HTML frontend (experimental). Primary UI is Streamlit on :8501."""
    html = (_UI_DIR / "index.html").read_text()
    return HTMLResponse(html)


@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
async def serve_app():
    html = (_UI_DIR / "index.html").read_text()
    return HTMLResponse(html)


# Static assets (JS/CSS files if added later)
if (_UI_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_UI_DIR / "assets")), name="assets")


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
