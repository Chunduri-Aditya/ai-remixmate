"""
tests/test_structured_logger_usage.py — Static guard against StructuredLogger misuse.

scripts.core.logging_utils.StructuredLogger.info/warning/debug/error/critical all
have the signature (msg, extra: Optional[dict] = None) — NOT stdlib logging's
(msg, *args) printf-style signature. Passing old %-style positional args (as in
`log.info("x=%s y=%s", x, y)`) either raises TypeError immediately (2+ extra
args) or silently corrupts the call (1 extra non-dict arg becomes `extra=<str>`,
which crashes inside stdlib logging's makeRecord when it tries to iterate the
string as a mapping). Either way the log call raises and the surrounding job
fails — this is not a logging nicety, it's a crash bug.

CLAUDE.md already documents this exact bug class being fixed at 8 sites in
api/jobs.py during the June 28 2026 audit. This test was added 2026-06-30
after the same bug was found live (via job history in data/jobs.db) crashing
every single AI Lab Style Transfer and Inpainting job at style_transfer.py and
inpainting.py — 9 call sites, none caught by the original audit because it
only covered jobs.py. This test scans every file that imports get_logger()
(the StructuredLogger factory) and fails if the anti-pattern reappears
anywhere, not just in the two files fixed this session.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Matches `<name>.info(`, `.warning(`, `.warn(`, `.debug(`, `.error(`, `.critical(`
# where the call's first line contains an old-style % placeholder — a strong
# signal of printf-style logging that assumes a *args signature.
_LOG_CALL_RE = re.compile(
    r"\b\w*log\w*\.(info|warning|warn|debug|error|critical)\(\s*\n?\s*f?[\"']"
)
_PERCENT_PLACEHOLDER_RE = re.compile(r"%[sdif]|%\.\d+f")


def _files_using_structured_logger() -> list[Path]:
    files = []
    for path in SCRIPTS_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if "get_logger(" in text and "logging_utils" in text:
            files.append(path)
    return files


def _offending_calls(path: Path) -> list[str]:
    """
    Return log-call snippets (a few lines each) in `path` that use printf-style
    % placeholders inside a StructuredLogger call. A call is only flagged if a
    comma appears after the format string on the same statement (multi-line
    calls are joined) — i.e. there's at least one extra positional argument,
    which is exactly the pattern StructuredLogger cannot accept.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    offenders = []

    for i, line in enumerate(lines):
        m = _LOG_CALL_RE.search(line)
        if not m:
            continue
        # Join up to 5 lines to capture multi-line calls, stop at the
        # matching close-paren depth returning to 0.
        snippet_lines = []
        depth = 0
        started = False
        for j in range(i, min(i + 6, len(lines))):
            snippet_lines.append(lines[j])
            depth += lines[j].count("(") - lines[j].count(")")
            if "(" in lines[j]:
                started = True
            if started and depth <= 0:
                break
        snippet = "\n".join(snippet_lines)

        if not _PERCENT_PLACEHOLDER_RE.search(snippet):
            continue  # f-strings and plain messages are fine

        # A trailing comma after the quoted format string (before the closing
        # paren) means extra positional args were passed — the broken pattern.
        # f-strings (f"...") never trigger this since %-looking text inside an
        # f-string is just literal text with no separate comma-args to follow
        # UNLESS the call itself still has extra positional args after it.
        after_string = re.search(r'["\'][^"\']*["\']\s*,', snippet)
        if after_string:
            offenders.append(snippet)

    return offenders


@pytest.mark.parametrize("path", _files_using_structured_logger(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_printf_style_structured_logger_calls(path: Path):
    offenders = _offending_calls(path)
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} passes printf-style positional args to a "
        f"StructuredLogger call. StructuredLogger.info/warning/debug/error/critical "
        f"only accept (msg, extra: Optional[dict] = None) — extra positional args "
        f"either raise TypeError or corrupt the call. Use an f-string instead.\n\n"
        + "\n---\n".join(offenders)
    )


def test_at_least_one_file_uses_structured_logger():
    # Sanity check that the scan itself is actually finding files — if this
    # starts failing, the detection heuristic (not the logging code) broke.
    assert len(_files_using_structured_logger()) >= 5
