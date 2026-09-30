"""
tests/test_env_aliases.py — spec B8.1 (env names the code reads).

Two facts under test:

1. ``scripts/core/paths.py`` honours ``REMIXMATE_LIBRARY_DIR`` and
   ``REMIXMATE_OUTPUTS_DIR`` as aliases of the generic config form
   (``REMIXMATE_LIBRARY_LIBRARY_DIR`` / ``REMIXMATE_LIBRARY_OUTPUTS_DIR``).
   paths.py freezes the constants at first import, so each case runs a child
   interpreter with the variable pre set and reads the resolved constant back.
   The child prints only booleans: never the resolved path, because a
   non honoured alias would fall through to whatever config.local.yaml holds.

2. ``app/deploy/remixmate.env.example`` uses ``REMIXMATE_LIBRARY_DIR=/mnt/catalog``
   and documents every ``REMIXMATE_*`` / ``SUPABASE_*`` name the API code reads.
   "Read" is defined by an AST scan: a string constant under ``scripts/`` that is
   exactly such a name (``os.environ.get("X")``, ``_X_ENV = "X"``, ``_truthy("X")``),
   docstrings excluded. ``scripts/ui/`` (the Streamlit dev UI) is excluded: it is
   not part of the systemd deploy the env file configures.
   "Documented" means a line of the form ``NAME=`` or ``# NAME=`` in the example.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parents[1].resolve()
_SCRIPTS = _REPO / "scripts"
_ENV_EXAMPLE = _REPO / "app" / "deploy" / "remixmate.env.example"

_NAME_RE = re.compile(r"(REMIXMATE|SUPABASE)_[A-Z0-9_]+")

# Storage variables the parent pytest process (tests/conftest.py) or a developer
# shell may already carry; each child starts without them so only the variable
# under test can influence the resolved constant.
_STORAGE_VARS = (
    "REMIXMATE_LIBRARY_LIBRARY_DIR",
    "REMIXMATE_LIBRARY_OUTPUTS_DIR",
    "REMIXMATE_LIBRARY_DIR",
    "REMIXMATE_OUTPUTS_DIR",
)

_CHILD_SCRIPT = r"""
import json, sys
from pathlib import Path
from scripts.core import paths
attr, expected = sys.argv[1], Path(sys.argv[2]).resolve()
actual = Path(getattr(paths, attr)).resolve()
print(json.dumps({
    "matches_expected": actual == expected,
    "is_repo_default": actual == (paths.PROJECT_ROOT / attr.split("_")[0].lower()).resolve(),
}))
"""


def _resolve_in_child(var: str, value: Path, attr: str, tmp_path: Path) -> dict[str, bool]:
    env = {k: v for k, v in os.environ.items() if k not in _STORAGE_VARS}
    env["REMIXMATE_DATA_DIR"] = str(tmp_path / "data")  # keep the child off the real data/ tree
    env[var] = str(value)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT, attr, str(value)],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"child interpreter failed (rc={proc.returncode}): {proc.stderr[-2000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# Fact 1: aliases resolve
# ---------------------------------------------------------------------------

def test_generic_library_form_still_resolves(tmp_path: Path) -> None:
    # Control: proves the child harness sees the variable and paths.py honours
    # the generic form. If this fails the alias results below mean nothing.
    target = tmp_path / "lib-generic"
    result = _resolve_in_child("REMIXMATE_LIBRARY_LIBRARY_DIR", target, "LIBRARY_DIR", tmp_path)
    assert result["matches_expected"], "REMIXMATE_LIBRARY_LIBRARY_DIR (generic form) no longer moves LIBRARY_DIR"


def test_remixmate_library_dir_alias_resolves(tmp_path: Path) -> None:
    target = tmp_path / "lib-alias"
    result = _resolve_in_child("REMIXMATE_LIBRARY_DIR", target, "LIBRARY_DIR", tmp_path)
    assert result["matches_expected"], (
        "REMIXMATE_LIBRARY_DIR is not honoured: paths.LIBRARY_DIR did not follow it "
        f"(fell back to repo default: {result['is_repo_default']})"
    )


def test_remixmate_outputs_dir_alias_resolves(tmp_path: Path) -> None:
    target = tmp_path / "out-alias"
    result = _resolve_in_child("REMIXMATE_OUTPUTS_DIR", target, "OUTPUTS_DIR", tmp_path)
    assert result["matches_expected"], (
        "REMIXMATE_OUTPUTS_DIR is not honoured: paths.OUTPUTS_DIR did not follow it "
        f"(fell back to repo default: {result['is_repo_default']})"
    )


def test_aliases_are_documented_in_paths_py() -> None:
    src = (_SCRIPTS / "core" / "paths.py").read_text(encoding="utf-8")
    for name in ("REMIXMATE_LIBRARY_DIR", "REMIXMATE_OUTPUTS_DIR"):
        assert re.search(rf"\b{name}\b", src), f"{name} is not mentioned in scripts/core/paths.py"


# ---------------------------------------------------------------------------
# Fact 2: the env example documents what the code reads
# ---------------------------------------------------------------------------

def _names_read_under_scripts() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted(_SCRIPTS.rglob("*.py")):
        rel = path.relative_to(_REPO)
        if rel.parts[:2] == ("scripts", "ui"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
        docstrings: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)
                    ):
                        docstrings.add(id(stmt.value))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and _NAME_RE.fullmatch(node.value)
            ):
                found.setdefault(node.value, set()).add(str(rel))
    return found


def _documented_names(text: str) -> set[str]:
    # A variable counts as documented when a line sets it or comments it out
    # as a setting: "NAME=value" or "# NAME=". Prose mentions do not count.
    return set(re.findall(r"(?m)^\s*#?\s*([A-Z][A-Z0-9_]*)=", text))


def test_scan_finds_known_reads() -> None:
    # Control for the scanner itself: names read via three different call
    # shapes must all be found, or a green documentation test proves nothing.
    found = _names_read_under_scripts()
    for name in ("REMIXMATE_DATA_DIR", "SUPABASE_JWT_ISS", "REMIXMATE_AUTH_ENABLED", "REMIXMATE_SMTP_TO"):
        assert name in found, f"scanner missed {name}"
    assert "REMIXMATE_API_URL" not in found, "scripts/ui/ was not excluded"


def test_env_example_uses_remixmate_library_dir_on_the_catalog_mount() -> None:
    text = _ENV_EXAMPLE.read_text(encoding="utf-8")
    assert re.search(r"(?m)^REMIXMATE_LIBRARY_DIR=/mnt/catalog\s*$", text), (
        "remixmate.env.example must set REMIXMATE_LIBRARY_DIR=/mnt/catalog"
    )
    assert not re.search(r"(?m)^LIBRARY_DIR=", text), (
        "remixmate.env.example still sets bare LIBRARY_DIR=, a name nothing under scripts/ reads"
    )


def test_env_example_documents_every_name_the_code_reads() -> None:
    found = _names_read_under_scripts()
    documented = _documented_names(_ENV_EXAMPLE.read_text(encoding="utf-8"))
    missing = sorted(n for n in found if n not in documented)
    assert not missing, "remixmate.env.example does not document: " + ", ".join(
        f"{n} ({', '.join(sorted(found[n]))})" for n in missing
    )


@pytest.mark.parametrize(
    "name",
    [
        "SUPABASE_JWT_ISS",
        "REMIXMATE_SMTP_TO",
        "REMIXMATE_DATA_DIR",
        "REMIXMATE_DEVICE",
        "REMIXMATE_TRUST_PROXY",
        "REMIXMATE_REQUIRE_CATALOG",
        "REMIXMATE_MUSICBRAINZ_ENABLED",
        "ADMIN_ALLOW_UNVERIFIED",
        "GENIUS_TOKEN",
        "VITE_COOKIE_DOMAIN",
    ],
)
def test_env_example_documents_spec_listed_name(name: str) -> None:
    documented = _documented_names(_ENV_EXAMPLE.read_text(encoding="utf-8"))
    assert name in documented, f"remixmate.env.example has no `{name}=` or `# {name}=` line"


@pytest.mark.parametrize("name", ["REMIXMATE_TRUST_PROXY", "REMIXMATE_REQUIRE_CATALOG"])
def test_env_example_shows_the_value_1_for_flag(name: str) -> None:
    text = _ENV_EXAMPLE.read_text(encoding="utf-8")
    assert re.search(rf"(?m)^\s*#?\s*{name}=1\s*(#.*)?$", text), (
        f"remixmate.env.example must show {name}=1 (the code compares the raw string to '1')"
    )


def test_license_enforce_annotated_ingest_only() -> None:
    lines = _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    idx = [i for i, ln in enumerate(lines) if re.match(r"^\s*#?\s*REMIXMATE_LICENSE_ENFORCE=", ln)]
    assert idx, "remixmate.env.example has no REMIXMATE_LICENSE_ENFORCE line"
    i = idx[0]
    window = " ".join(lines[max(0, i - 1): i + 1]).lower()
    assert "ingest only" in window, (
        "REMIXMATE_LICENSE_ENFORCE must be annotated 'ingest only' on its line or the line above"
    )
