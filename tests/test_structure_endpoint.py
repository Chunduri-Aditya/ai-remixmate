"""
tests/test_structure_endpoint.py — GET /analysis/v1/{name}/structure
(DJ_ENGINE_IMPROVEMENT_PLAN.md §3.1).

Follows tests/test_router_smoke.py's pattern: a minimal FastAPI app mounting
only the analysis router, LIBRARY_DIR patched to a tmp_path, real TestClient
hitting the real route handler and Pydantic serialisation — no mocking of
dj_analysis.analyze_structure() itself, so this exercises the actual
librosa-backed structure computation end to end on a synthetic click track.

Marked both `integration` (TestClient/httpx) and `dj_analysis` (calls
analyze_structure(), which needs a working librosa/numba environment) —
excluded by `pytest -m "not dj_analysis and not integration"` same as every
other TestClient-based router test in this suite.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed (required by TestClient)")

from fastapi import FastAPI
from fastapi.testclient import TestClient

SR = 44100


def _click_track(bpm: float = 128.0, duration_s: float = 32.0, sr: int = SR) -> np.ndarray:
    """A short, real (not silent) audio clip with clear periodic transients
    so librosa's beat tracker has something to lock onto — enough bars for
    analyze_structure() to produce a non-trivial bars/beats/sections list."""
    n = int(duration_s * sr)
    audio = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    click_len = int(0.01 * sr)
    click = (np.hanning(click_len * 2)[click_len:] * 0.9).astype(np.float32)
    for beat_t in np.arange(0, duration_s, period):
        i = int(beat_t * sr)
        if i + click_len <= n:
            audio[i : i + click_len] += click
    return audio


def _app_with(router) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.integration
@pytest.mark.dj_analysis
class TestStructureEndpoint:
    @pytest.fixture(autouse=True)
    def client(self, tmp_path, monkeypatch):
        import soundfile as sf

        import scripts.core.paths as paths_mod
        from scripts.api.routers import analysis as analysis_mod

        lib = tmp_path / "library"
        song = "Click Track"
        song_dir = lib / song
        song_dir.mkdir(parents=True)
        sf.write(str(song_dir / "full.wav"), _click_track(), SR)

        # Two separate bindings need patching: _require_song() (in
        # scripts/api/routers/_helpers.py) reads its own module-level
        # LIBRARY_DIR, and song_dir()/_load_song_audio() (called dynamically
        # inside the endpoint) read scripts.core.paths.LIBRARY_DIR — same
        # dual-patch requirement test_router_smoke.py's router tests hit.
        monkeypatch.setattr(paths_mod, "LIBRARY_DIR", lib)
        with patch("scripts.api.routers._helpers.LIBRARY_DIR", lib):
            app = _app_with(analysis_mod.router)
            with TestClient(app, raise_server_exceptions=True) as c:
                self._client = c
                self._song = song
                yield c

    def test_structure_v1_shape(self):
        r = self._client.get(f"/analysis/v1/{self._song}/structure")
        assert r.status_code == 200
        body = r.json()

        assert body["version"] == "v1"
        assert body["song"] == self._song
        assert body["bpm"] > 0
        assert body["duration"] > 0
        assert body["total_bars"] >= 0
        for key in ("bars", "beats", "sections", "key_segments", "downbeat_times"):
            assert key in body and isinstance(body[key], list)

        # At least one bar/beat should come out of a 32 s, clearly-periodic
        # click track — this isn't a silent/degenerate response.
        assert len(body["beats"]) > 0
        assert len(body["bars"]) > 0
        if body["bars"]:
            start, end = body["bars"][0]
            assert end > start

    def test_include_beats_false_omits_beats_only(self):
        full = self._client.get(f"/analysis/v1/{self._song}/structure").json()
        light = self._client.get(
            f"/analysis/v1/{self._song}/structure?include_beats=false"
        ).json()

        assert light["beats"] == []
        # Everything else is unaffected by include_beats.
        assert light["bars"] == full["bars"]
        assert light["total_bars"] == full["total_bars"]

    def test_unknown_song_404(self):
        r = self._client.get("/analysis/v1/does-not-exist/structure")
        assert r.status_code == 404

    def test_response_matches_documented_schema(self):
        """Guards the versioned contract itself — every key the plan's
        definition of done promises (bars/beats/segments/key+confidence)
        must be present with the documented type, even when a field is
        empty for this particular song."""
        from scripts.api.schemas import StructureV1Response

        r = self._client.get(f"/analysis/v1/{self._song}/structure")
        # Round-trips cleanly through the same Pydantic model the route
        # declares as its response_model — fails loudly if the two drift.
        StructureV1Response.model_validate(r.json())
