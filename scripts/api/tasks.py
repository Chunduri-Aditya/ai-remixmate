"""
scripts/api/tasks.py — Task function re-export shim.

All task functions have been split into domain modules under scripts/api/task_modules/.
This module re-exports them all so that existing imports like:
    from scripts.api.tasks import task_download
continue to work without modification.
"""

# Download tasks
from scripts.api.task_modules.download import (
    task_download,
    task_playlist_download,
)

# Stem tasks
from scripts.api.task_modules.stems import (
    task_batch_compress_stems,
    task_batch_stem_split,
    task_compress_stems,
    task_stem_split,
)

# Remix tasks
from scripts.api.task_modules.remix import (
    task_dj_chain,
    task_dj_remix,
    task_remix_preview,
)

# Analysis tasks
from scripts.api.task_modules.analysis import (
    task_analyze,
    task_analyze_missing,
    task_initialize_library,
    task_rebuild_index,
)

# Generative tasks
from scripts.api.task_modules.generative import (
    task_inpaint_transition,
    task_style_transfer,
    task_tokenize_stems,
)

# Lab tasks
from scripts.api.task_modules.lab import (
    task_instrument_lab,
)

__all__ = [
    # Download
    "task_download",
    "task_playlist_download",
    # Stems
    "task_stem_split",
    "task_batch_stem_split",
    "task_compress_stems",
    "task_batch_compress_stems",
    # Remix
    "task_dj_remix",
    "task_dj_chain",
    "task_remix_preview",
    # Analysis
    "task_analyze",
    "task_analyze_missing",
    "task_rebuild_index",
    "task_initialize_library",
    # Generative
    "task_style_transfer",
    "task_inpaint_transition",
    "task_tokenize_stems",
    # Lab
    "task_instrument_lab",
]
