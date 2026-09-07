#!/usr/bin/env python3
"""
scripts/migrate_uids.py — One-time "local" → real-uid migration (Phase 4).

Phase 3 scoped every per-user write path to a user id that was always
``"local"``. Phase 4 turns on Supabase auth, so the operator's own existing
data has to be re-stamped from ``"local"`` to their real Supabase user id and
their rendered mixes moved under ``outputs/<uid>/``.

What it touches (all rows currently owned by ``"local"``):
  • outputs/local/*            → outputs/<uid>/*        (directory move)
  • data/jobs.db      jobs.meta_json .user_id           → <uid>
  • data/remixmate.db crates / crate_songs / song_tags / song_favorites .user_id → <uid>
  • data/feedback.db  feedback_events.user_id           → <uid>

The shared catalogue (``library/``) is NOT namespaced and is left alone.

Usage:
    python -m scripts.migrate_uids --uid <supabase-uid>              # dry run
    python -m scripts.migrate_uids --uid <supabase-uid> --apply      # do it

Dry run is the default. **Snapshot ``data/`` and ``outputs/`` before --apply** —
the row updates are reversible with a second run only if you keep the old uid,
the directory move is not.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import uuid
from pathlib import Path

_CRATE_TABLES = ("crates", "crate_songs", "song_tags", "song_favorites")


# ---------------------------------------------------------------------------
# outputs/local → outputs/<uid>
# ---------------------------------------------------------------------------

def _plan_outputs_move(outputs_dir: Path, uid: str) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Return (moves, warnings). Each move is (src, dst) for one entry under
    outputs/local/. Collisions with an existing outputs/<uid>/entry are a hard
    error (raised), not a silent overwrite."""
    src_root = outputs_dir / "local"
    dst_root = outputs_dir / uid
    moves: list[tuple[Path, Path]] = []
    warnings: list[str] = []

    if not src_root.is_dir():
        warnings.append(f"no {src_root} — nothing to move")
    else:
        for entry in sorted(src_root.iterdir()):
            dst = dst_root / entry.name
            if dst.exists():
                raise SystemExit(
                    f"refusing to migrate: {dst} already exists "
                    f"(would collide with {entry}). Resolve by hand."
                )
            moves.append((entry, dst))

    strays = [
        d.name for d in sorted(outputs_dir.iterdir())
        if d.is_dir() and d.name not in ("local", uid)
    ] if outputs_dir.is_dir() else []
    if strays:
        warnings.append(
            "left in place (not under outputs/local/): "
            + ", ".join(strays)
            + " — move by hand if these are yours"
        )
    return moves, warnings


def _apply_outputs_move(moves: list[tuple[Path, Path]], uid: str, outputs_dir: Path) -> None:
    if not moves:
        return
    (outputs_dir / uid).mkdir(parents=True, exist_ok=True)
    for src, dst in moves:
        shutil.move(str(src), str(dst))
    # drop the now-empty outputs/local/
    src_root = outputs_dir / "local"
    if src_root.is_dir() and not any(src_root.iterdir()):
        src_root.rmdir()


# ---------------------------------------------------------------------------
# DB row re-stamping
# ---------------------------------------------------------------------------

def _count_and_update_jobs(db: Path, uid: str, apply: bool) -> int:
    """jobs.db stores the owner inside the meta_json blob, not a column."""
    if not db.exists():
        return 0
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT job_id, meta_json FROM jobs").fetchall()
        stale = []
        for job_id, meta_json in rows:
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except (TypeError, json.JSONDecodeError):
                meta = {}
            if (meta.get("user_id") or "local") == "local":
                stale.append((job_id, meta))
        if apply:
            for job_id, meta in stale:
                meta["user_id"] = uid
                conn.execute(
                    "UPDATE jobs SET meta_json = ? WHERE job_id = ?",
                    (json.dumps(meta), job_id),
                )
            conn.commit()
        return len(stale)
    finally:
        conn.close()


def _count_and_update_column(db: Path, tables: tuple[str, ...], uid: str, apply: bool) -> dict[str, int]:
    """Tables that carry a real user_id column (crates.db, feedback.db)."""
    out: dict[str, int] = {}
    if not db.exists():
        return {t: 0 for t in tables}
    conn = sqlite3.connect(str(db))
    try:
        for table in tables:
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE user_id = 'local'"
            ).fetchone()[0]
            out[table] = n
            if apply and n:
                conn.execute(
                    f"UPDATE {table} SET user_id = ? WHERE user_id = 'local'", (uid,)
                )
        if apply:
            conn.commit()
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(uid: str, apply: bool, data_dir: Path, outputs_dir: Path) -> int:
    if not uid or uid == "local":
        raise SystemExit("--uid must be a real user id, not empty or 'local'")
    try:
        uuid.UUID(uid)
    except ValueError:
        print(f"⚠  --uid {uid!r} is not UUID-shaped (fine for a test uid)", file=sys.stderr)

    jobs_db = data_dir / "jobs.db"
    crates_db = data_dir / "remixmate.db"
    feedback_db = data_dir / "feedback.db"

    moves, warnings = _plan_outputs_move(outputs_dir, uid)
    n_jobs = _count_and_update_jobs(jobs_db, uid, apply)
    n_crates = _count_and_update_column(crates_db, _CRATE_TABLES, uid, apply)
    n_feedback = _count_and_update_column(feedback_db, ("feedback_events",), uid, apply)

    verb = "moved" if apply else "would move"
    stamp = "stamped" if apply else "would stamp"
    print(f"Target uid: {uid}")
    print(f"outputs/local → outputs/{uid}:  {verb} {len(moves)} entr{'y' if len(moves) == 1 else 'ies'}")
    for w in warnings:
        print(f"  ⚠  {w}")
    print(f"jobs.db          jobs.meta_json.user_id:   {stamp} {n_jobs}")
    for table, n in n_crates.items():
        print(f"remixmate.db     {table}.user_id:{' ' * (max(0, 20 - len(table)))}{stamp} {n}")
    print(f"feedback.db      feedback_events.user_id:  {stamp} {n_feedback['feedback_events']}")

    if apply:
        _apply_outputs_move(moves, uid, outputs_dir)
        print("\n✅  Migration applied.")
    else:
        total = len(moves) + n_jobs + sum(n_crates.values()) + n_feedback["feedback_events"]
        print(f"\nDry run — nothing changed. {total} item(s) pending. "
              "Snapshot data/ and outputs/, then re-run with --apply.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--uid", required=True, help="the real Supabase user id to migrate 'local' data to")
    parser.add_argument("--apply", action="store_true", help="apply the migration (default: dry run)")
    parser.add_argument("--data-dir", type=Path, default=None, help="override data/ (default: scripts.core.paths.DATA_DIR)")
    parser.add_argument("--outputs-dir", type=Path, default=None, help="override outputs/ (default: scripts.core.paths.OUTPUTS_DIR)")
    args = parser.parse_args()

    if args.data_dir is None or args.outputs_dir is None:
        from scripts.core.paths import DATA_DIR, OUTPUTS_DIR
        data_dir = args.data_dir or DATA_DIR
        outputs_dir = args.outputs_dir or OUTPUTS_DIR
    else:
        data_dir, outputs_dir = args.data_dir, args.outputs_dir

    return run(args.uid, args.apply, Path(data_dir), Path(outputs_dir))


if __name__ == "__main__":
    raise SystemExit(main())
