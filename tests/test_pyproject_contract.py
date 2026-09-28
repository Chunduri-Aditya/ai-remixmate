"""Contract tests for pyproject.toml (spec B1.2).

The core install must not pull download, Streamlit or lyric tooling; the
project targets Python 3.12; ruff configuration lives under [tool.ruff] and
[tool.ruff.lint] and excludes the archived and legacy trees.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# PEP 508 requirement -> normalised distribution name (PEP 503: lowercase,
# runs of -_. collapsed to a single dash).  "yt-dlp>=2025.1" -> "yt-dlp",
# "PyJWT[crypto]>=2.10" -> "pyjwt", "uvicorn[standard]>=0.35" -> "uvicorn".
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _dist_name(requirement: str) -> str:
    match = _NAME_RE.match(requirement)
    assert match, f"unparseable requirement: {requirement!r}"
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def core_deps(pyproject: dict) -> set[str]:
    return {_dist_name(req) for req in pyproject["project"]["dependencies"]}


@pytest.fixture(scope="module")
def wordplay_deps(pyproject: dict) -> set[str]:
    extras = pyproject["project"]["optional-dependencies"]
    return {_dist_name(req) for req in extras["wordplay"]}


# ---------------------------------------------------------------------------
# [project] dependencies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "removed",
    ["yt-dlp", "ytmusicapi", "streamlit", "lyricsgenius", "nltk"],
)
def test_core_dependencies_drop_download_ui_and_lyric_tooling(
    core_deps: set[str], removed: str
) -> None:
    assert removed not in core_deps, (
        f"{removed} is still in [project] dependencies; core deps: {sorted(core_deps)}"
    )


def test_core_dependencies_keep_the_api_stack(core_deps: set[str]) -> None:
    # Allow side control: the removal must not have gutted the real runtime set.
    for kept in ("fastapi", "uvicorn", "librosa", "demucs", "sqlalchemy", "pyjwt"):
        assert kept in core_deps, f"{kept} missing from [project] dependencies"


@pytest.mark.parametrize("extra_dep", ["lyricsgenius", "nltk"])
def test_wordplay_extra_still_carries_lyric_tooling(
    wordplay_deps: set[str], extra_dep: str
) -> None:
    assert extra_dep in wordplay_deps, (
        f"{extra_dep} must stay in [project.optional-dependencies].wordplay; "
        f"got {sorted(wordplay_deps)}"
    )


# ---------------------------------------------------------------------------
# Python version pins
# ---------------------------------------------------------------------------


def test_requires_python_is_3_12_or_newer(pyproject: dict) -> None:
    value = pyproject["project"]["requires-python"]
    assert value.replace(" ", "") == ">=3.12", f"requires-python is {value!r}"


def test_mypy_python_version_is_3_12(pyproject: dict) -> None:
    mypy = pyproject["tool"]["mypy"]
    assert mypy["python_version"] == "3.12", f"[tool.mypy] python_version is {mypy['python_version']!r}"


# ---------------------------------------------------------------------------
# ruff layout
# ---------------------------------------------------------------------------

EXPECTED_EXCLUDES = {
    "archive",
    "ai_remixmate_feature_lab",
    "scripts/ui",
    "remix-env",
    ".venv",
    ".venv-old",
    ".venv-new",
}


def test_ruff_has_lint_subtable(pyproject: dict) -> None:
    ruff = pyproject["tool"]["ruff"]
    assert "lint" in ruff and isinstance(ruff["lint"], dict), (
        f"[tool.ruff.lint] missing; [tool.ruff] keys: {sorted(ruff)}"
    )


def test_ruff_select_and_ignore_live_under_lint(pyproject: dict) -> None:
    ruff = pyproject["tool"]["ruff"]
    # Top level select/ignore are the deprecated layout ruff warns about.
    assert "select" not in ruff, "[tool.ruff].select must move to [tool.ruff.lint]"
    assert "ignore" not in ruff, "[tool.ruff].ignore must move to [tool.ruff.lint]"
    lint = ruff["lint"]
    assert "select" in lint, f"[tool.ruff.lint] has no select; keys: {sorted(lint)}"


@pytest.mark.parametrize("path", sorted(EXPECTED_EXCLUDES))
def test_ruff_extend_exclude_lists_each_legacy_tree(pyproject: dict, path: str) -> None:
    ruff = pyproject["tool"]["ruff"]
    excludes = ruff.get("extend-exclude", [])
    assert path in excludes, (
        f"{path!r} missing from [tool.ruff].extend-exclude; got {excludes!r}"
    )
