"""
scripts/api/cors.py — Resolve CORS allow-origins for prod vs local (Phase 7).

Prod: ``REMIXMATE_CORS_ORIGINS`` (comma-separated) or ``REMIXMATE_FRONTEND_URL``.
Dev: Vite localhost ports only — no Streamlit ``:8501``, no GitHub Pages.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

_DEV_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "https://localhost:5173",
    "http://127.0.0.1:5173",
    "https://127.0.0.1:5173",
)

_LEGACY_BLOCK = (":8501", "chunduri-aditya.github.io")


def _is_legacy(origin: str) -> bool:
    return any(tok in origin for tok in _LEGACY_BLOCK)


def resolve_cors_origins(config_origins: Optional[Sequence[str]] = None) -> List[str]:
    """Return the CORS allow-list for this process.

    Precedence:
    1. ``REMIXMATE_CORS_ORIGINS`` (comma-separated)
    2. ``REMIXMATE_FRONTEND_URL`` (single origin)
    3. non-wildcard ``config_origins`` with legacy entries stripped
    4. Vite localhost defaults
    """
    env = os.environ.get("REMIXMATE_CORS_ORIGINS", "").strip()
    if env:
        return [o.strip() for o in env.split(",") if o.strip()]

    frontend = os.environ.get("REMIXMATE_FRONTEND_URL", "").strip().rstrip("/")
    if frontend:
        return [frontend]

    if config_origins:
        cleaned = [o for o in config_origins if o and o != "*" and not _is_legacy(o)]
        if cleaned:
            return cleaned

    return list(_DEV_ORIGINS)
