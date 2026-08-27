"""
Adversarial QA — log-line forgery via user-controlled text.

Song names, tags, search queries and playlist titles flow into
`logging.getLogger(__name__)` calls across scripts/core/. A name containing a
newline splits one entry into two, the second forgeable as a real log line.

The guard is a handler-level filter, so these tests drive it the way the app
does: through configure_logging(), not by calling the filter directly.
"""
from __future__ import annotations

import logging

import pytest

from scripts.core.logging_utils import LogInjectionFilter

FORGED = "Innocent Song\nERROR    scripts.core.library  library wiped by admin"


@pytest.fixture
def captured():
    """A handler carrying the real filter, plus the records it formatted."""
    stream_out: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            stream_out.append(self.format(record))

    h = _Capture()
    h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    h.addFilter(LogInjectionFilter())
    log = logging.getLogger("test_log_injection")
    log.handlers = [h]
    log.setLevel(logging.DEBUG)
    log.propagate = False
    yield log, stream_out
    log.handlers = []


def test_newline_in_printf_arg_cannot_forge_a_line(captured):
    log, out = captured
    log.warning("analysis failed for '%s'", FORGED)
    assert len(out) == 1
    assert "\n" not in out[0], f"forged newline survived: {out[0]!r}"
    assert "\\n" in out[0], "newline should be escaped, not silently dropped"


def test_newline_in_fstring_message_cannot_forge_a_line(captured):
    log, out = captured
    log.warning(f"analysis failed for '{FORGED}'")
    assert len(out) == 1
    assert "\n" not in out[0]


def test_carriage_return_is_escaped_too(captured):
    log, out = captured
    log.info("song %s", "evil\rovewritten")
    assert "\r" not in out[0]
    assert "\\r" in out[0]


def test_dict_style_args_are_scrubbed(captured):
    log, out = captured
    log.info("%(song)s failed", {"song": FORGED})
    assert "\n" not in out[0]


def test_non_string_args_pass_through_unharmed(captured):
    """The filter must not corrupt normal logging."""
    log, out = captured
    log.info("bpm=%s beats=%d ratio=%.2f", 128.5, 64, 1.25)
    assert out[0] == "INFO bpm=128.5 beats=64 ratio=1.25"


def test_configure_logging_installs_the_filter():
    """The filter is only worth anything if the live entry point attaches it.

    main.py calls configure_logging() at startup; this asserts that path
    actually wires the filter onto every handler it creates.
    """
    from scripts.core.config import configure_logging

    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        root.handlers = []
        configure_logging()
        assert root.handlers, "configure_logging() installed no handlers"
        for h in root.handlers:
            assert any(isinstance(f, LogInjectionFilter) for f in h.filters), (
                f"handler {h!r} has no LogInjectionFilter"
            )
    finally:
        root.handlers = saved
