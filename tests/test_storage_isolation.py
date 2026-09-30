"""
tests/test_storage_isolation.py — guards the conftest storage-isolation shim.

Deployment plan Phase 0: a test run must never write into the repo's real
``outputs/`` or ``data/`` trees. The shim in tests/conftest.py sets
REMIXMATE_LIBRARY_OUTPUTS_DIR / REMIXMATE_DATA_DIR to a throwaway dir before the
first ``import scripts.*``. These tests fail if that shim stops taking effect —
including if a new DATA_DIR consumer binds the path in a way the shim misses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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
        resolved = Path(str(raw))
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


# ---------------------------------------------------------------------------
# Spec B0.5: the shim must apply even when the caller pre set the storage env
# vars (that was how fixture rows reached the real data/jobs.db). The only opt
# out is REMIXMATE_TESTS_USE_REAL_DATA=1. These cases need the env var set
# *before* conftest is imported, so each one runs a child interpreter.
# ---------------------------------------------------------------------------

_STORE_VARS = (
    "REMIXMATE_DATA_DIR",
    "REMIXMATE_METADATA_CACHE_PATH",
    "REMIXMATE_DATABASE_PATH",
    "REMIXMATE_DATABASE_EMBEDDINGS_PATH",
)

_CHILD_SCRIPT = r"""
import json, os
import tests.conftest  # the shim runs at import, before any scripts.* import
from scripts.core import paths
print(json.dumps({
    "OUTPUTS_DIR": str(paths.OUTPUTS_DIR),
    "DATA_DIR": str(paths.DATA_DIR),
    "env": {k: os.environ.get(k) for k in (
        "REMIXMATE_LIBRARY_OUTPUTS_DIR",
        "REMIXMATE_DATA_DIR",
        "REMIXMATE_METADATA_CACHE_PATH",
        "REMIXMATE_DATABASE_PATH",
        "REMIXMATE_DATABASE_EMBEDDINGS_PATH",
    )},
}))
"""


def _resolve_in_child(preset: dict[str, str]) -> dict[str, Any]:
    """Import conftest then paths in a fresh interpreter with ``preset`` exported.

    Inherited REMIXMATE_* vars are stripped first: the parent pytest's own shim
    already points them at a temp dir, and leaking those in would make every
    case pass for the wrong reason.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("REMIXMATE_")}
    env.update(preset)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, f"child interpreter failed:\n{proc.stderr[-2000:]}"
    result: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def test_preset_outputs_dir_does_not_opt_out_of_isolation() -> None:
    # Red on the old conftest: ``if not os.environ.get("REMIXMATE_LIBRARY_OUTPUTS_DIR")``
    # skipped the whole shim, so OUTPUTS_DIR froze to the real repo tree.
    real_outputs = _REPO / "outputs"
    result = _resolve_in_child({"REMIXMATE_LIBRARY_OUTPUTS_DIR": str(real_outputs)})

    assert not _inside_repo(Path(result["OUTPUTS_DIR"])), (
        f"OUTPUTS_DIR={result['OUTPUTS_DIR']} is inside the repo: a pre set "
        "REMIXMATE_LIBRARY_OUTPUTS_DIR opted the run out of the isolation shim"
    )
    assert not _inside_repo(Path(result["DATA_DIR"])), (
        f"DATA_DIR={result['DATA_DIR']} is inside the repo"
    )
    for var in _STORE_VARS:
        value = result["env"][var]
        assert value, f"{var} was not redirected by the shim"
        assert not _inside_repo(Path(value)), f"{var}={value} points inside the repo"


def test_preset_data_dir_is_redirected_too() -> None:
    # Old conftest used setdefault for the store vars, so a pre set
    # REMIXMATE_DATA_DIR pointing at the real data/ tree survived.
    real_data = _REPO / "data"
    result = _resolve_in_child({
        "REMIXMATE_DATA_DIR": str(real_data),
        "REMIXMATE_DATABASE_PATH": str(real_data / "remixmate.db"),
    })

    assert not _inside_repo(Path(result["DATA_DIR"])), (
        f"DATA_DIR={result['DATA_DIR']} is inside the repo: a pre set "
        "REMIXMATE_DATA_DIR survived the isolation shim"
    )
    for var in _STORE_VARS:
        value = result["env"][var]
        assert value, f"{var} was not redirected by the shim"
        assert not _inside_repo(Path(value)), f"{var}={value} points inside the repo"


def test_exported_outputs_alias_does_not_opt_out_of_isolation() -> None:
    # B8.1 added REMIXMATE_OUTPUTS_DIR as an alias of the generic form; a shell
    # that sourced the deploy env file must not send the suite to real outputs/.
    real_outputs = _REPO / "outputs"
    result = _resolve_in_child({"REMIXMATE_OUTPUTS_DIR": str(real_outputs)})

    assert not _inside_repo(Path(result["OUTPUTS_DIR"])), (
        f"OUTPUTS_DIR={result['OUTPUTS_DIR']} is inside the repo: an exported "
        "REMIXMATE_OUTPUTS_DIR alias opted the run out of the isolation shim"
    )


def test_real_data_opt_out_keeps_preset_outputs_dir() -> None:
    # The paired allow case: REMIXMATE_TESTS_USE_REAL_DATA=1 is the one way to
    # keep the real paths. Read only: importing paths creates nothing.
    real_outputs = _REPO / "outputs"
    result = _resolve_in_child({
        "REMIXMATE_TESTS_USE_REAL_DATA": "1",
        "REMIXMATE_LIBRARY_OUTPUTS_DIR": str(real_outputs),
    })

    assert Path(result["OUTPUTS_DIR"]).resolve() == real_outputs.resolve(), (
        f"OUTPUTS_DIR={result['OUTPUTS_DIR']}: opt out flag did not keep the pre set dir"
    )
    assert result["env"]["REMIXMATE_DATA_DIR"] is None, (
        "opt out flag set, yet the shim still exported REMIXMATE_DATA_DIR"
    )
