"""
scripts/core/mix_retention.py — bound the size of a user's rendered-mix folder.

`outputs/<user_id>/` accumulates one directory per rendered mix (``dj_<id>``,
``chain_<id>``, ``beat_<id>``, preview sessions). On a shared-catalog hosted
deploy the catalog is a fixed operator asset, but per-user output is unbounded
without a sweep. This module is the sweep.

Two independent limits, applied in order:

1. **TTL** — a mix directory whose most recent mtime is older than
   ``max_age_days`` is removed.
2. **Count cap** — of the directories that survive the TTL pass, if more than
   ``max_count`` remain, the oldest (by mtime) are removed until ``max_count``
   are left. This is LRU by last write.

Pure filesystem + stdlib. No API, config, or job-store imports, so it is cheap
to unit test and safe to call from a task thread. The caller passes the root to
sweep, so it can only ever touch the directory it was handed — one user's
``outputs/<uid>/`` never reaches another's.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts.core.logging_utils import get_logger

_log = get_logger(__name__)

_SECONDS_PER_DAY = 86_400


@dataclass
class RetentionResult:
    """What a sweep did (or, when ``dry_run``, would do)."""

    removed: list[str] = field(default_factory=list)   # session dir names
    removed_by_ttl: list[str] = field(default_factory=list)
    removed_by_cap: list[str] = field(default_factory=list)
    kept: int = 0
    freed_bytes: int = 0
    dry_run: bool = False

    def as_dict(self) -> dict:
        return {
            "removed": self.removed,
            "removed_by_ttl": self.removed_by_ttl,
            "removed_by_cap": self.removed_by_cap,
            "kept": self.kept,
            "freed_bytes": self.freed_bytes,
            "freed_mb": round(self.freed_bytes / 1_048_576, 1),
            "dry_run": self.dry_run,
        }


def _dir_mtime(path: Path) -> float:
    """Most recent mtime anywhere under ``path`` (the dir's own mtime only moves
    when a direct child is added or removed, so a mix whose file was rewritten
    in place would otherwise look stale)."""
    latest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            latest = max(latest, child.stat().st_mtime)
        except OSError:
            continue
    return latest


def _dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def sweep_dir(
    root: Path,
    *,
    max_age_days: int = 30,
    max_count: int = 50,
    now: float | None = None,
    dry_run: bool = False,
) -> RetentionResult:
    """Apply the TTL and count-cap limits to the immediate subdirectories of
    ``root``. A missing or empty ``root`` is a no-op. ``now`` (epoch seconds)
    is injectable for tests."""
    result = RetentionResult(dry_run=dry_run)
    root = Path(root)
    if not root.is_dir():
        return result

    now = time.time() if now is None else now
    sessions = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=_dir_mtime,
    )  # oldest first

    ttl_cutoff = now - max_age_days * _SECONDS_PER_DAY
    survivors: list[Path] = []
    for session in sessions:
        if _dir_mtime(session) < ttl_cutoff:
            _remove(session, result, result.removed_by_ttl)
        else:
            survivors.append(session)

    overflow = len(survivors) - max_count
    if overflow > 0:
        for session in survivors[:overflow]:          # already oldest-first
            _remove(session, result, result.removed_by_cap)
        survivors = survivors[overflow:]

    result.kept = len(survivors)
    return result


def _remove(session: Path, result: RetentionResult, bucket: list[str]) -> None:
    size = _dir_size(session)
    if not result.dry_run:
        try:
            shutil.rmtree(session)
        except OSError as exc:
            _log.warning(f"mix_retention: could not remove {session.name}: {exc}")
            return
    result.removed.append(session.name)
    bucket.append(session.name)
    result.freed_bytes += size
    verb = "would remove" if result.dry_run else "removed"
    _log.info(f"mix_retention: {verb} {session.name} ({size / 1_048_576:.1f} MB)")


def sweep_current_user(*, dry_run: bool = False) -> RetentionResult:
    """Sweep the acting user's ``outputs/<uid>/`` using the configured limits.
    Best-effort: never raises, so a post-render call cannot fail a render."""
    try:
        from scripts.core.config import cfg
        from scripts.core.paths import user_outputs_dir

        rc = cfg.retention
        result = sweep_dir(
            user_outputs_dir(),
            max_age_days=rc.mix_max_age_days,
            max_count=rc.mix_max_count,
            dry_run=dry_run,
        )
        _mark_swept_expired(result)
        return result
    except Exception as exc:  # noqa: BLE001 - a retention failure must not break a render
        _log.warning(f"mix_retention: sweep_current_user failed: {exc}")
        return RetentionResult(dry_run=dry_run)


def _mark_swept_expired(result: RetentionResult) -> None:
    """Flag the job rows of every removed session ``result.expired = true``.
    A dry run removed nothing, so it marks nothing."""
    if result.dry_run:
        return
    from scripts.api import jobs as job_store

    for session_id in result.removed:
        job_store.mark_expired(session_id)


def sweep_all_users() -> dict[str, RetentionResult]:
    """Sweep every ``outputs/<uid>/`` with the configured limits and mark each
    swept session's jobs expired. Independent of the acting user. Best-effort
    per user, like ``sweep_current_user``: one failing user dir is logged and
    the rest still run. Returns the result per user id."""
    from scripts.core import paths
    from scripts.core.config import cfg

    rc = cfg.retention
    results: dict[str, RetentionResult] = {}
    root = paths.OUTPUTS_DIR
    if not root.is_dir():
        return results
    for user_root in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            result = sweep_dir(
                user_root,
                max_age_days=rc.mix_max_age_days,
                max_count=rc.mix_max_count,
            )
            _mark_swept_expired(result)
            results[user_root.name] = result
        except Exception as exc:  # noqa: BLE001 - one user's failure must not stop the sweep
            _log.warning(f"mix_retention: sweep of {user_root.name} failed: {exc}")
    return results
