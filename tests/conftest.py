"""
tests/conftest.py — Shared pytest configuration and fixtures.

Environment detection
---------------------
Some tests use librosa + numba (via llvmlite) for audio analysis.  In certain
CI/containerised environments numba fails at JIT-compile time due to LLVM
version mismatches or missing cache write permissions.  Rather than letting the
whole suite crash with an opaque ImportError, we:

  1. Probe librosa+numba once at collection time.
  2. Register a custom mark ``dj_analysis`` for tests that need it.
  3. Automatically skip ``dj_analysis``-marked tests when the probe fails,
     with a clear message explaining why.

This keeps the unaffected test slice green while making environment problems
obvious and actionable instead of mysterious.

Supported marks
---------------
``pytest.mark.dj_analysis``
    Tests that call librosa beat tracking or _analyze_impl() from dj_engine.
    Skipped when the librosa/numba environment is broken.

``pytest.mark.integration``
    Tests that spin up the full FastAPI app via TestClient.  Not skipped
    automatically, but labelled so they can be excluded with -m "not integration"
    when running in environments without all deps installed.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``scripts.*`` imports resolve when
# pytest is invoked from any working directory.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Storage isolation — MUST run before anything imports scripts.core.paths
# ---------------------------------------------------------------------------
# scripts/core/paths.py freezes OUTPUTS_DIR (via cfg.library.outputs_dir /
# REMIXMATE_LIBRARY_OUTPUTS_DIR) and DATA_DIR (via REMIXMATE_DATA_DIR) into
# module-level constants at first import, and consumers bind them one level
# deeper still — jobs.py does ``_DB_PATH = DATA_DIR / "jobs.db"`` at import,
# crates.py / feedback_store.py likewise. Reassigning the constant afterwards
# is therefore useless. The only way to keep a test run from writing into the
# real outputs/ and data/ trees is to point the env vars at a throwaway dir
# *before* the first ``import scripts.*``. pytest imports this root conftest
# before test collection and it pulls in no scripts modules itself, so here is
# early enough.
#
# A few stores bypass paths.DATA_DIR entirely — config.py carries them as
# relative strings ("data/metadata.db", "data/remixmate.db",
# "data/song_embeddings.json"), consumed by track_metadata.py / database.py.
# _apply_env_vars() maps REMIXMATE_METADATA_CACHE_PATH -> metadata.cache_path
# and REMIXMATE_DATABASE_* -> database.*, so redirect those too. (audit.py's
# log path derives from DATA_DIR directly, so REMIXMATE_DATA_DIR covers it.)
#
# LIBRARY_DIR is deliberately left pointing at the real shared catalog — tests
# read from it and never write; see the deployment plan, Phase 0. Pre-set
# storage vars are overwritten; the only opt out is REMIXMATE_TESTS_USE_REAL_DATA=1.
if os.environ.get("REMIXMATE_TESTS_USE_REAL_DATA") != "1":
    _TEST_STORAGE_TMP = tempfile.TemporaryDirectory(prefix="remixmate-test-storage-")
    _test_storage_root = Path(_TEST_STORAGE_TMP.name)
    _test_data_dir = _test_storage_root / "data"
    os.environ["REMIXMATE_LIBRARY_OUTPUTS_DIR"] = str(_test_storage_root / "outputs")
    os.environ["REMIXMATE_DATA_DIR"] = str(_test_data_dir)
    os.environ["REMIXMATE_METADATA_CACHE_PATH"] = str(_test_data_dir / "metadata.db")
    os.environ["REMIXMATE_DATABASE_PATH"] = str(_test_data_dir / "remixmate.db")
    os.environ["REMIXMATE_DATABASE_EMBEDDINGS_PATH"] = str(
        _test_data_dir / "song_embeddings.json"
    )


# ---------------------------------------------------------------------------
# Mark registration
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "dj_analysis: marks tests that require a functional librosa/numba environment",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests that start a full FastAPI TestClient",
    )


# ---------------------------------------------------------------------------
# librosa / numba environment probe
# ---------------------------------------------------------------------------

def _probe_librosa() -> tuple[bool, str]:
    """
    Try to import librosa and execute a minimal beat-track call.

    Returns (ok: bool, reason: str).  The reason is shown in the skip message
    when ok is False.
    """
    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        # Disable numba caching to avoid write-permission issues in read-only
        # CI file systems.  This must be set before numba is first imported.
        os.environ.setdefault("NUMBA_DISABLE_JIT", "0")
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / ".cache" / "numba"))

        # Minimal real call — exercises the numba JIT path
        y = np.zeros(22050, dtype=np.float32)
        y[::512] = 1.0  # synthetic transients
        librosa.beat.beat_track(y=y, sr=22050)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


_LIBROSA_OK, _LIBROSA_SKIP_REASON = _probe_librosa()


# ---------------------------------------------------------------------------
# Auto-skip hook for dj_analysis tests
# ---------------------------------------------------------------------------

def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip dj_analysis tests when the librosa/numba probe failed."""
    if not _LIBROSA_OK and item.get_closest_marker("dj_analysis"):
        pytest.skip(
            f"Skipping dj_analysis test: librosa/numba environment is not functional "
            f"in this run environment. Probe error: {_LIBROSA_SKIP_REASON}"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def librosa_available() -> bool:
    """Session-scoped flag: True when librosa+numba works in this environment."""
    return _LIBROSA_OK


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root directory."""
    return _PROJECT_ROOT


@pytest.fixture(autouse=False)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """
    Fixture that clears REMIXMATE_* environment variables before a test
    and restores them afterwards.  Opt-in via ``clean_env`` in test signature.
    """
    keys = [k for k in os.environ if k.startswith("REMIXMATE_")]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    yield
