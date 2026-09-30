"""The agent docs stay true to the tree.

Scans the committed agent docs (the root docs CLAUDE.md, AGENTS.md, RULES.md,
ARCHITECTURE.md, PRD.md, PHASES.md, DESIGN.md and everything in docs/agents/)
for backticked citations. Enforced: every cited repo path is known to git,
tracked or new but not ignored (or sits under a known local only prefix),
every `path::symbol` names a
definition in that file, every backlog id cited anywhere exists in BACKLOG.md,
and backlog ids are unique. Not enforced: whether a sentence about the code is
true. That stays a review job.
"""

from __future__ import annotations

import ast
import fnmatch
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ROOT_DOCS = (
    "CLAUDE.md",
    "AGENTS.md",
    "RULES.md",
    "ARCHITECTURE.md",
    "PRD.md",
    "PHASES.md",
    "DESIGN.md",
)
MAPS = ("docs/agents/UI_MAP.md", "docs/agents/MIXING_MAP.md", "docs/agents/BACKLOG.md")
BACKLOG = "docs/agents/BACKLOG.md"
# Paths that exist only on a working machine, never in a clone or in CI.
LOCAL_ONLY = (
    ".venv/",
    ".claude/",
    "data/",
    "library/",
    "library_quarantine/",
    "outputs/",
    "catalog/",
    "catalog_data/",
    "models/",
    "remix-env/",
    "graphify-out",
    "config.local.yaml",
    "MEMORY.md",
)
EXTENSIONS = (
    ".py", ".ts", ".tsx", ".js", ".mjs", ".json", ".md", ".yaml", ".yml", ".toml",
    ".css", ".sh", ".txt", ".html", ".cfg", ".ini", ".service", ".conf", ".example",
)
TOKEN = re.compile(r"`([^`\n]+)`")
LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")
BACKLOG_ID = re.compile(r"\b(?:UI|MX|OPS|DOC)-\d{2,3}\b")
BACKLOG_HEADING = re.compile(r"^#{2,4}\s+((?:UI|MX|OPS|DOC)-\d{2,3})\b", re.MULTILINE)
SKIP_CHARS = set("<>{}$| =,'\"")
TS_DEF = (
    r"(?:export\s+(?:default\s+)?)?(?:async\s+)?"
    r"(?:function\*?|const|let|var|class|interface|type|enum)\s+{name}\b"
)


@lru_cache(maxsize=1)
def tracked_files() -> frozenset[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git ls-files unavailable; the doc check needs a git checkout")
    return frozenset(out.splitlines())


def _top_level(tracked: frozenset[str]) -> set[str]:
    names = {path.split("/", 1)[0] for path in tracked}
    names.update(prefix.rstrip("/") for prefix in LOCAL_ONLY)
    return names


def _is_local_only(path: str) -> bool:
    return any(path == p.rstrip("/") or path.startswith(p) for p in LOCAL_ONLY)


def path_exists(path: str, tracked: frozenset[str]) -> bool:
    if _is_local_only(path):
        return True
    if any(ch in path for ch in "*?["):
        return any(fnmatch.fnmatch(t, path) for t in tracked)
    if path in tracked:
        return True
    prefix = path.rstrip("/") + "/"
    return any(t.startswith(prefix) for t in tracked)


@lru_cache(maxsize=None)
def _python_names(path: str) -> frozenset[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(f"{node.name}.{item.name}")
                for target in _assigned(item):
                    names.add(f"{node.name}.{target}")
    for item in _module_statements(tree.body):
        names.update(_assigned(item))
    return frozenset(names)


def _module_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """Module level statements, including those inside top level if and try blocks."""
    found: list[ast.stmt] = []
    for node in body:
        found.append(node)
        if isinstance(node, ast.If):
            found += _module_statements(node.body + node.orelse)
        elif isinstance(node, ast.Try):
            handlers = [s for h in node.handlers for s in h.body]
            found += _module_statements(node.body + handlers + node.orelse + node.finalbody)
    return found


def _assigned(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def symbol_exists(path: str, symbol: str) -> bool:
    symbol = symbol.removesuffix("()")
    suffix = Path(path).suffix
    if suffix == ".py":
        names = _python_names(path)
        parts = symbol.split(".")
        return symbol in names or ".".join(parts[-2:]) in names
    text = (ROOT / path).read_text(encoding="utf-8")
    if suffix in (".ts", ".tsx", ".js", ".mjs"):
        head, _, tail = symbol.partition(".")
        if not re.search(TS_DEF.format(name=re.escape(head)), text):
            return False
        last = symbol.rsplit(".", 1)[-1]
        return not tail or re.search(rf"\b{re.escape(last)}\s*[:(=]", text) is not None
    if suffix in (".yaml", ".yml"):
        return all(re.search(rf"^\s*{re.escape(p)}\s*:", text, re.M) for p in symbol.split("."))
    return True  # headings and other anchors are not checked


def citations(text: str, tracked: frozenset[str]) -> list[tuple[str, str | None]]:
    """Return (path, symbol or None) for every backticked token that cites the tree."""
    top = _top_level(tracked)
    found = []
    for raw in TOKEN.findall(text):
        token = raw.strip().removeprefix("./")
        if not token or SKIP_CHARS & set(token) or token.startswith(("/", "~", "http", "-")):
            continue
        path, _, symbol = token.partition("::")
        path = LINE_SUFFIX.sub("", path.split("#", 1)[0])
        if "/" not in path and not path.endswith(EXTENSIONS):
            continue
        if not symbol and path.split("/", 1)[0] not in top and not path.endswith(EXTENSIONS):
            continue  # ratios and pairs such as bpm_a/bpm_b are not paths
        if "/" not in path and path not in tracked and not _is_local_only(path):
            if not symbol:
                continue  # a bare file name such as meta.json names a data file, not a repo path
        found.append((path, symbol or None))
    return found


def problems(text: str, source: str, tracked: frozenset[str], backlog_ids: set[str]) -> list[str]:
    found = []
    for path, symbol in citations(text, tracked):
        if not path_exists(path, tracked):
            found.append(f"{source}: `{path}` is not in the tree")
        elif symbol and not _is_local_only(path) and not symbol_exists(path, symbol):
            found.append(f"{source}: `{path}::{symbol}` names no definition in that file")
    for backlog_id in sorted(set(BACKLOG_ID.findall(text)) - backlog_ids):
        found.append(f"{source}: backlog id {backlog_id} is not in {BACKLOG}")
    return found


def _existing_docs() -> list[str]:
    return [d for d in (*ROOT_DOCS, *MAPS) if (ROOT / d).is_file()]


def _backlog_ids() -> list[str]:
    path = ROOT / BACKLOG
    return BACKLOG_HEADING.findall(path.read_text(encoding="utf-8")) if path.is_file() else []


def test_agent_docs_exist() -> None:
    missing = [d for d in (*ROOT_DOCS, *MAPS) if not (ROOT / d).is_file()]
    assert not missing, f"agent docs missing: {missing}"
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines()
    assert "@RULES.md" in [line.strip() for line in claude], "CLAUDE.md must import @RULES.md"


@pytest.mark.parametrize("doc", _existing_docs())
def test_doc_citations_resolve(doc: str) -> None:
    text = (ROOT / doc).read_text(encoding="utf-8")
    found = problems(text, doc, tracked_files(), set(_backlog_ids()))
    assert not found, "\n".join(found)


def test_backlog_ids_are_unique() -> None:
    ids = _backlog_ids()
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"backlog ids used twice: {dupes}"


def test_checker_flags_planted_errors() -> None:
    planted = "\n".join(
        [
            "real: `scripts/api/jobs.py::create_job` and `scripts/core/dj_engine.py::DJEngine.render_stem_blend`",
            "real: `frontend/src/lib/api.ts::normalizeJob` and `config.yaml` and `library/` and `MX-01`",
            "noise: `bpm_a/bpm_b`, `meta.json`, `/library/storage/prune`, `.venv/bin/python -m pytest`",
            "fake path: `scripts/core/no_such_module.py`",
            "fake symbol: `scripts/api/jobs.py::no_such_function`",
            "untracked: `remixmate.egg-info/SOURCES.txt`",
            "unknown id: MX-999",
        ]
    )
    found = problems(planted, "planted", tracked_files(), {"MX-01"})
    assert found == [
        "planted: `scripts/core/no_such_module.py` is not in the tree",
        "planted: `scripts/api/jobs.py::no_such_function` names no definition in that file",
        "planted: `remixmate.egg-info/SOURCES.txt` is not in the tree",
        "planted: backlog id MX-999 is not in docs/agents/BACKLOG.md",
    ]
