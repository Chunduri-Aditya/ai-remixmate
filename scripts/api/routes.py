"""
scripts/api/routes.py — Route aggregator.

Each domain has its own sub-router in scripts/api/routers/. This module
combines them into a single APIRouter that main.py includes.
"""

from fastapi import APIRouter

from scripts.api.routers import (
    analysis,
    crates,
    downloads,
    events,
    generative,
    jobs,
    library,
    remix,
    setlist,
    spotify,
    stems,
    system,
)

router = APIRouter()
router.include_router(system.router)
router.include_router(library.router)
router.include_router(downloads.router)
router.include_router(stems.router)
router.include_router(analysis.router)
router.include_router(remix.router)
router.include_router(generative.router)
router.include_router(spotify.router)
router.include_router(jobs.router)
router.include_router(crates.router)
router.include_router(events.router)
router.include_router(setlist.router)
