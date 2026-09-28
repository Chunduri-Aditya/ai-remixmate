"""
tests/test_generative_urls.py — B5.7 generative task output contract.

Contract (plans/audit-the-project-and-sorted-flask.md B5.7): the generative
tasks write their output under ``user_outputs_dir()/gen_<id>/`` and return a
``stream_url`` that the existing ``GET /outputs/{session}/{file}`` route
resolves for the acting user. No ``/files`` URL anywhere in the result: the
API has no ``/files`` route, so every such link was a dead link.

Seams: the heavy core modules (``scripts.core.style_transfer``,
``scripts.core.inpainting``) are replaced in ``sys.modules`` by stand ins that
behave like the real ones (write a file under ``OUTPUTS_DIR/<legacy session>/``
unless told where to write) and return the real result dataclass shape. The
task module resolves them lazily inside the task function, so the stand in is
what it imports. ``update_job`` and ``log_audit`` in the task module are
no ops. ``OUTPUTS_DIR`` already points at a throwaway dir (tests/conftest.py).
"""

from __future__ import annotations

import json
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.api.routers import library as library_router
from scripts.api.task_modules import generative as gen
from scripts.core.paths import OUTPUTS_DIR, user_outputs_dir

STREAM_RE = re.compile(r"^/outputs/(gen_[^/]+)/([^/]+)$")


@dataclass
class _FakeStyleResult:
    success: bool
    song_name: str
    description: str
    output_path: Optional[str] = None
    duration_sec: float = 0.0
    lufs: float = 0.0
    source_key: Optional[str] = None
    source_camelot: Optional[str] = None
    generation_time_sec: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeInpaintResult:
    success: bool
    song_a: str
    song_b: str
    output_path: Optional[str] = None
    method: str = "cosine_fallback"
    duration_sec: float = 0.0
    lufs: float = 0.0
    generation_time_sec: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class _AnyConfig:
    """Accepts whatever keyword arguments the task passes, like a dataclass."""

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def _target_dir(config: Any, kw: Dict[str, Any], legacy: Path) -> Path:
    """Where the stand in writes: an explicit output dir if the task passes
    one (either as a kwarg or on the config), else the legacy shared dir the
    real core module uses today."""
    for key in ("output_dir", "out_dir", "outputs_dir"):
        if kw.get(key):
            return Path(kw[key])
        if getattr(config, key, None):
            return Path(getattr(config, key))
    return legacy


@pytest.fixture
def quiet_task_module(monkeypatch):
    monkeypatch.setattr(gen, "update_job", lambda *a, **k: None)
    monkeypatch.setattr(gen, "log_audit", lambda *a, **k: None)


@pytest.fixture
def fake_style_transfer(monkeypatch):
    calls: list = []

    def run_style_transfer(song_name, config, progress_cb=None, **kw):
        out_dir = _target_dir(config, kw, OUTPUTS_DIR / "style_legacy_session")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "style_transfer_1.wav"
        out.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake")
        calls.append({"song": song_name, "out": out})
        return _FakeStyleResult(
            success=True, song_name=song_name, description=config.description,
            output_path=str(out), duration_sec=1.0, lufs=-14.0,
        )

    mod = types.ModuleType("scripts.core.style_transfer")
    mod.StyleTransferConfig = _AnyConfig
    mod.run_style_transfer = run_style_transfer
    monkeypatch.setitem(sys.modules, "scripts.core.style_transfer", mod)
    return calls


@pytest.fixture
def fake_inpainting(monkeypatch):
    calls: list = []

    def run_inpainting(song_a, song_b, config, progress_cb=None, **kw):
        out_dir = _target_dir(config, kw, OUTPUTS_DIR / "inpaint_legacy")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "inpaint_1.wav"
        out.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake")
        calls.append({"a": song_a, "b": song_b, "out": out})
        return _FakeInpaintResult(
            success=True, song_a=song_a, song_b=song_b, output_path=str(out),
            duration_sec=1.0, lufs=-14.0,
        )

    mod = types.ModuleType("scripts.core.inpainting")
    mod.InpaintConfig = _AnyConfig
    mod.run_inpainting = run_inpainting
    monkeypatch.setitem(sys.modules, "scripts.core.inpainting", mod)
    return calls


@pytest.fixture
def outputs_client() -> TestClient:
    app = FastAPI()
    app.include_router(library_router.router)
    return TestClient(app)


def _assert_stream_contract(result: dict, outputs_client: TestClient) -> None:
    blob = json.dumps(result, default=str)
    assert "/files" not in blob, f"dead /files link still emitted: {blob}"

    stream_url = result.get("stream_url")
    assert isinstance(stream_url, str), f"no stream_url in result: {sorted(result)}"
    m = STREAM_RE.match(stream_url)
    assert m, f"stream_url {stream_url!r} is not /outputs/gen_<id>/<file>"
    session, fname = m.group(1), m.group(2)

    on_disk = user_outputs_dir() / session / fname
    assert on_disk.exists(), f"{on_disk} missing: output not written under user_outputs_dir()/gen_<id>/"
    assert on_disk.resolve().parent.parent == user_outputs_dir().resolve()

    r = outputs_client.get(stream_url)
    assert r.status_code == 200, f"GET {stream_url} -> {r.status_code} {r.text[:200]}"


@pytest.mark.usefixtures("quiet_task_module")
def test_style_transfer_writes_under_user_gen_dir_and_returns_stream_url(
    fake_style_transfer, outputs_client
):
    result = gen.task_style_transfer(
        "job-style-1", song_name="Some Song", description="dark techno", duration_sec=1.0
    )
    assert fake_style_transfer, "the task never called run_style_transfer"
    _assert_stream_contract(result, outputs_client)


@pytest.mark.usefixtures("quiet_task_module")
def test_inpaint_writes_under_user_gen_dir_and_returns_stream_url(
    fake_inpainting, outputs_client
):
    result = gen.task_inpaint_transition("job-inpaint-1", song_a="Song A", song_b="Song B")
    assert fake_inpainting, "the task never called run_inpainting"
    _assert_stream_contract(result, outputs_client)


@pytest.mark.usefixtures("quiet_task_module")
def test_two_generative_jobs_do_not_share_an_output_dir(fake_style_transfer, outputs_client):
    r1 = gen.task_style_transfer("job-a", song_name="Some Song", description="x")
    r2 = gen.task_style_transfer("job-b", song_name="Some Song", description="y")
    s1 = STREAM_RE.match(r1.get("stream_url") or "")
    s2 = STREAM_RE.match(r2.get("stream_url") or "")
    assert s1 and s2, (r1.get("stream_url"), r2.get("stream_url"))
    assert s1.group(1) != s2.group(1), "two jobs wrote into the same gen_<id> dir"
