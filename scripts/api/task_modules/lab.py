"""
scripts/api/task_modules/lab.py — Instrument lab stem swap task function.

task_instrument_lab  — stem swap experiments
"""

import uuid
from typing import Any, Dict, Optional

from scripts.api.jobs import update_job
from scripts.core.audit import log_audit
from scripts.core.paths import OUTPUTS_DIR


def task_instrument_lab(
    job_id: str,
    songs: list,
    mode: str = "targeted",
    swap_stems: Optional[list] = None,
    target_duration: Optional[float] = None,
    include_pure: bool = False,
) -> Dict[str, Any]:
    """
    Render all instrument swap combinations for the given songs.

    For each combo, swaps stems between songs (e.g. vocals from A + drums from B),
    time-stretches to a common BPM, mixes, and saves the result.
    """
    from scripts.core.instrument_lab import render_all_combos, get_songs_with_stems

    log_audit("instrument_lab_start", resource=f"{len(songs)} songs", job_id=job_id,
              metadata={"songs": songs, "mode": mode})
    update_job(job_id, progress=0.05, message=f"Instrument Lab: preparing {len(songs)} songs…")

    # ── Verify stems exist ────────────────────────────────────────────────
    available = set(get_songs_with_stems(min_stems=2))
    missing = [s for s in songs if s not in available]
    if missing:
        raise ValueError(
            f"Songs missing stems (run stem separation first): {missing}"
        )

    # ── Create output directory ───────────────────────────────────────────
    session_id = str(uuid.uuid4())[:8]
    out_dir = OUTPUTS_DIR / f"lab_{session_id}"

    # ── Render all combos with progress reporting ─────────────────────────
    def _progress(idx: int, total: int, label: str, result):
        frac = 0.10 + 0.80 * (idx / max(total, 1))
        status = "✓" if (result and result.success) else "…"
        update_job(job_id, progress=frac,
                   message=f"[{idx+1}/{total}] {status} {label}")

    session = render_all_combos(
        songs=songs,
        output_dir=out_dir,
        mode=mode,
        target_duration=target_duration,
        swap_stems=swap_stems,
        include_pure=include_pure,
        progress_cb=_progress,
    )

    update_job(job_id, progress=0.95,
               message=f"✅ {session.rendered} combos rendered, {session.failed} failed")

    # ── Build result metadata ─────────────────────────────────────────────
    combo_summaries = []
    for r in session.results:
        combo_summaries.append({
            "label":       r.combo.short_label(),
            "mapping":     r.combo.mapping,
            "success":     r.success,
            "output":      str(r.output_path) if r.output_path else None,
            "bpm":         r.bpm,
            "duration_sec": r.duration_sec,
            "lufs":        r.lufs,
            "error":       r.error,
        })

    log_audit("instrument_lab_complete", resource=str(out_dir), job_id=job_id,
              metadata={"rendered": session.rendered, "failed": session.failed,
                        "mode": mode, "songs": songs})

    return {
        "session_id":     session_id,
        "output_dir":     str(out_dir),
        "songs":          songs,
        "mode":           mode,
        "total_combos":   session.total_combos,
        "rendered":       session.rendered,
        "failed":         session.failed,
        "skipped":        session.skipped,
        "combos":         combo_summaries,
    }
