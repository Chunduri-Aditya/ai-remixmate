"""
tests/test_storage_isolation.py — guards the conftest storage-isolation shim.

Deployment plan Phase 0: a test run must never write into the repo's real
``outputs/`` or ``data/`` trees. The shim in tests/conftest.py sets
REMIXMATE_LIBRARY_OUTPUTS_DIR / REMIXMATE_DATA_DIR to a throwaway dir before the
first ``import scripts.*``. These tests fail if that shim stops taking effect —
including if a new DATA_DIR consumer binds the path in a way the shim misses.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).parents[1].resolve()


def _inside_repo(p: Path) -> bool:
    p = Path(p).resolve()
    return p == _REPO or _REPO in p.parents


def test_outputs_and_data_dirs_are_outside_the_repo() -> None:
    from scripts.core import paths

    assert not _inside_repo(paths.OUTPUTS_DIR), (
        f"OUTPUTS_DIR={paths.OUTPUTS_DIR} is inside the repo — isolation shim did not apply"
    )
    assert not _inside_repo(paths.DATA_DIR), (
        f"DATA_DIR={paths.DATA_DIR} is inside the repo — isolation shim did not apply"
    )
    assert not _inside_repo(paths.EMBEDDINGS_FILE), (
        f"EMBEDDINGS_FILE={paths.EMBEDDINGS_FILE} is inside the repo"
    )


def test_library_dir_still_points_at_the_real_catalog() -> None:
    # Phase 0 isolates OUTPUTS_DIR + DATA_DIR only; LIBRARY_DIR stays global.
    from scripts.core import paths

    assert paths.LIBRARY_DIR.resolve() == (_REPO / "library").resolve()


def test_db_consumers_resolved_to_the_isolated_data_dir() -> None:
    # The real bite: modules bind DATA_DIR at import time. Prove each DB path
    # landed outside the repo, not just the paths.DATA_DIR constant.
    from scripts.api import jobs
    from scripts.core import crates, feedback_store

    for label, db_path in (
        ("jobs._DB_PATH", jobs._DB_PATH),
        ("crates._DB_PATH", crates._DB_PATH),
        ("feedback_store._DB_PATH", feedback_store._DB_PATH),
    ):
        assert not _inside_repo(Path(db_path)), (
            f"{label}={db_path} is inside the repo — DATA_DIR shim missed this consumer"
        )


def test_hardcoded_data_paths_are_also_isolated() -> None:
    # These bypass paths.DATA_DIR entirely: config.py holds them as relative
    # strings ("data/metadata.db", "data/remixmate.db", ...) and audit.py had a
    # bare CWD-relative Path("data/audit.jsonl"). The shim covers the config
    # strings via REMIXMATE_METADATA_* / REMIXMATE_DATABASE_*; audit.py now
    # derives from DATA_DIR.
    from scripts.core import audit
    from scripts.core.config import cfg

    candidates = {
        "cfg.metadata.cache_path": cfg.metadata.cache_path,
        "cfg.database.path": cfg.database.path,
        "cfg.database.embeddings_path": cfg.database.embeddings_path,
        "audit._AUDIT_PATH": audit._AUDIT_PATH,
    }
    for label, raw in candidates.items():
        # relative values resolve against CWD (repo root in a normal run)
        resolved = Path(raw)
        if not resolved.is_absolute():
            resolved = _REPO / resolved
        assert not _inside_repo(resolved), (
            f"{label}={raw!r} resolves inside the repo — not isolated"
        )


def test_writes_land_in_the_isolated_tree() -> None:
    from scripts.core import paths

    paths.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    probe = paths.OUTPUTS_DIR / "isolation_probe.txt"
    probe.write_text("probe", encoding="utf-8")
    try:
        assert probe.exists()
        assert not _inside_repo(probe)
    finally:
        probe.unlink(missing_ok=True)
