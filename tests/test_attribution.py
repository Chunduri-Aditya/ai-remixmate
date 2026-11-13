"""
tests/test_attribution.py — visible author credit in the package metadata.

CITATION.cff, pyproject.toml and frontend/package.json each name the author
and the public repo, and keep the MIT licence. No email address anywhere.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = "Aditya Chunduri"
REPO = "https://github.com/Chunduri-Aditya/ai-remixmate"


def test_citation_cff_parses_with_required_keys():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    cff = yaml.safe_load(text)
    # CFF 1.2.0 required keys
    for key in ("cff-version", "message", "title", "authors"):
        assert key in cff, f"CITATION.cff lacks required key {key!r}"
    assert cff["cff-version"] == "1.2.0"
    assert cff["title"] == "AI RemixMate"
    assert cff["authors"] == [{"family-names": "Chunduri", "given-names": "Aditya"}]
    assert cff["repository-code"] == REPO
    assert cff["url"] == REPO
    assert cff["license"] == "MIT"
    assert "@" not in text, "CITATION.cff must carry no email address"


def test_pyproject_names_author_and_repo():
    with (ROOT / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)["project"]
    assert project["authors"] == [{"name": AUTHOR}]
    assert project["urls"]["Homepage"] == REPO
    assert project["urls"]["Repository"] == REPO
    assert project["license"] == {"file": "LICENSE"}


def test_package_json_names_author_and_repo():
    pkg = json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    assert pkg["author"] == AUTHOR
    assert pkg["homepage"] == REPO
    assert pkg["repository"] == {"type": "git", "url": f"git+{REPO}.git"}
    assert pkg["private"] is True
    assert pkg["license"] == "MIT"
