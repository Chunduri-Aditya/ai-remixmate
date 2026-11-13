"""
scripts/api/cors.py — Resolve CORS allow-origins for prod vs local (Phase 7).

Prod: ``REMIXMATE_CORS_ORIGINS`` (comma-separated) or ``REMIXMATE_FRONTEND_URL``.
Dev: Vite localhost origins on the UI port only — no Streamlit ``:8501``, no
GitHub Pages. ``start.sh`` passes the UI port as ``RMX_UI_PORT``; without it
``REMIXMATE_UI_PORT`` is read, then the default 5173.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

_LEGACY_BLOCK = (":8501", "chunduri-aditya.github.io")


def _is_legacy(origin: str) -> bool:
    return any(tok in origin for tok in _LEGACY_BLOCK)


def _ui_port() -> int:
    """The Vite port from ``RMX_UI_PORT``, else ``REMIXMATE_UI_PORT`` (an API
    started without ``start.sh``, which unsets it); the default when both are
    unset or the value is not a port."""
    default = 5173
    var = "RMX_UI_PORT"
    raw = os.environ.get(var, "").strip()
    if not raw:
        var = "REMIXMATE_UI_PORT"
        raw = os.environ.get(var, "").strip()
    if not raw:
        return default
    try:
        port = int(raw)
    except ValueError:
        port = 0
    if 1 <= port <= 65535:
        return port
    logger.warning("Ignored %s=%r (not a port from 1 to 65535); using %d", var, raw, default)
    return default


def _dev_origins(port: int) -> List[str]:
    return [
        f"http://localhost:{port}",
        f"https://localhost:{port}",
        f"http://127.0.0.1:{port}",
        f"https://127.0.0.1:{port}",
    ]


def resolve_cors_origins(config_origins: Optional[Sequence[str]] = None) -> List[str]:
    """Return the CORS allow-list for this process.

    Precedence:
    1. ``REMIXMATE_CORS_ORIGINS`` (comma-separated)
    2. ``REMIXMATE_FRONTEND_URL`` (single origin)
    3. non-wildcard ``config_origins`` with legacy entries stripped
    4. Vite localhost origins on ``RMX_UI_PORT``, else ``REMIXMATE_UI_PORT`` (default 5173)
    """
    env = os.environ.get("REMIXMATE_CORS_ORIGINS", "").strip()
    if env:
        origins = [o.strip() for o in env.split(",") if o.strip()]
        if "*" in origins:
            logger.warning("Dropped '*' from REMIXMATE_CORS_ORIGINS; list explicit origins instead")
        return [o for o in origins if o != "*"]

    frontend = os.environ.get("REMIXMATE_FRONTEND_URL", "").strip().rstrip("/")
    if frontend:
        return [frontend]

    if config_origins:
        cleaned = [o for o in config_origins if o and o != "*" and not _is_legacy(o)]
        if cleaned:
            return cleaned

    return _dev_origins(_ui_port())
