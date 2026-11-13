"""
tests/test_scan_mix_e2e.py: the stranger's path, end to end, through HTTP.

    music/ folder with two click tracks
      -> scripts.scan --analyze (real ffmpeg import, real analysis, no stems)
      -> tempo gate on the stored analysis, BEFORE any mix is requested
      -> GET /library lists both songs
      -> POST /dj-remix, poll GET /jobs/{id} until terminal
      -> the WAV the route wrote: location, length, loudness

The tracks are a metronome click plus a tonal pad (tests/synth_audio.py), never
a pure sine: a sine has no transients, so a beat grid on it means nothing.

    Click 120   120 BPM, pad 220 Hz,  60 s, written as FLAC
    Click 124   124 BPM, pad 1500 Hz, 60 s, written as WAV

Expected values come from the task, never from the analyzer: tempos 120 and
124 within 1 BPM, loudness within 1 LU of config.yaml audio.target_lufs
(-14.0). The canary scans a pure 440 Hz sine as "Sine 124" and proves the
tempo gate goes red on it (analysis raises a clear error, no meta.json).

Isolation: the conftest redirects data and outputs; this module redirects the
library root, the library manager and the two route modules that bind
LIBRARY_DIR by name (RULES.md, "Do"), inside one module scoped patch context
that stays open until the render job is terminal.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from urllib.parse import unquote

import numpy as np
import pytest

pytest.importorskip("soundfile")
pytest.importorskip("pyloudnorm")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import pyloudnorm  # noqa: E402
import soundfile as sf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from scripts import scan  # noqa: E402
from scripts.api import jobs as job_store  # noqa: E402
from scripts.api import main as main_mod  # noqa: E402
from scripts.api.auth import DEFAULT_USER  # noqa: E402
from scripts.api.routers import _helpers  # noqa: E402
from scripts.api.routers import library as library_router  # noqa: E402
from scripts.core import library as library_mod  # noqa: E402
from scripts.core import paths  # noqa: E402
from scripts.core.config import cfg  # noqa: E402
from scripts.core.library import LibraryManager  # noqa: E402
from tests.synth_audio import synth_click_track  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.dj_analysis,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg"),
]

ROOT = Path(__file__).resolve().parents[1]
SR = 44100
SOURCE_SECONDS = 60.0
TEMPO_TOLERANCE_BPM = 1.0
LOUDNESS_TOLERANCE_LU = 1.0
TRACKS = (
    # (song name, file name, bpm, pad Hz, seed)
    ("Click 120", "Click 120.flac", 120.0, 220.0, 0),
    ("Click 124", "Click 124.wav", 124.0, 1500.0, 1),
)
TRANSITION_BARS = 8
POLL_SECONDS = 0.5
POLL_CAP_SECONDS = 180.0
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED")


# ---------------------------------------------------------------------------
# Tempo gate
# ---------------------------------------------------------------------------

def _assert_tempo(name: str, expected_bpm: float) -> Dict[str, float]:
    """Fail unless the analysis the app stored for `name` reads `expected_bpm`
    within 1 BPM, in both meta.json and analysis.json. A missing file is a
    failure too: the gate never skips. Returns the stored values."""
    d = paths.song_dir(name)
    stored: Dict[str, float] = {}
    for fname in ("meta.json", "analysis.json"):
        f = d / fname
        assert f.exists(), (
            f"{name}: no {fname} in the isolated song dir; analysis did not run or failed, "
            f"so there is no tempo to check against {expected_bpm}"
        )
        bpm = json.loads(f.read_text(encoding="utf-8")).get("bpm")
        assert bpm is not None, f"{name}: {fname} holds no bpm"
        stored[fname] = float(bpm)
        assert abs(float(bpm) - expected_bpm) <= TEMPO_TOLERANCE_BPM, (
            f"{name}: {fname} bpm {float(bpm):.2f} is {abs(float(bpm) - expected_bpm):.2f} BPM "
            f"from {expected_bpm} (limit {TEMPO_TOLERANCE_BPM})"
        )
    return stored


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iso(tmp_path_factory: pytest.TempPathFactory):
    """Isolated library root, manager and TestClient (routes only, no
    lifespan). Every reader on the scan, route and render path sees `root`."""
    base = tmp_path_factory.mktemp("scan_mix")
    root = base / "library"
    root.mkdir()
    real_data = (ROOT / "data").resolve()
    assert not Path(job_store._DB_PATH).resolve().is_relative_to(real_data), job_store._DB_PATH
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paths, "LIBRARY_DIR", root)
        mgr = LibraryManager(library_dir=root)
        mp.setattr(library_mod, "_manager", mgr)
        mp.setattr(library_mod, "get_library_manager", lambda: mgr)
        mp.setattr(library_router, "LIBRARY_DIR", root)
        mp.setattr(_helpers, "LIBRARY_DIR", root)
        mp.delenv("REMIXMATE_AUTH_ENABLED", raising=False)   # acting user "local"
        assert paths.song_dir("x") == root / "x"              # the redirect is live
        yield SimpleNamespace(base=base, root=root, mgr=mgr, client=TestClient(main_mod.app))


@pytest.fixture(scope="module")
def mixed(iso) -> Dict[str, Any]:
    """Scan, tempo gate, list, POST /dj-remix, wait for the job. The gate runs
    above the POST; `events` records the order for the test that checks it."""
    events: list[str] = []
    music = iso.base / "music"
    music.mkdir()
    for _name, fname, bpm, pad_hz, seed in TRACKS:
        sf.write(str(music / fname), synth_click_track(bpm, SOURCE_SECONDS, pad_hz, seed, SR), SR)

    assert scan.main([str(music), "--analyze"]) == 0
    events.append("scan")

    tempos = {name: _assert_tempo(name, bpm) for name, _f, bpm, _p, _s in TRACKS}
    events.append("tempo_gate")

    client = iso.client
    r = client.get("/library")
    assert r.status_code == 200, r.text[:200]
    listed = {s["name"] for s in r.json()["songs"]}
    assert {name for name, *_ in TRACKS} <= listed, listed

    # Earlier modules in the session (tests/test_render_e2e.py::_new_job) leave
    # rows that never reach a terminal state; clear the acting user's through
    # the public job store API, in the conftest isolated jobs.db, so the cap
    # check in POST /dj-remix does not answer 429.
    job_store.cancel_user_jobs(DEFAULT_USER)

    events.append("post")
    r = client.post("/dj-remix", json={
        "song_a": "Click 120", "song_b": "Click 124",
        "transition_bars": TRANSITION_BARS, "preset": "club",
    })
    assert r.status_code == 202, (r.status_code, r.text[:300])
    job_id = r.json()["job_id"]

    deadline = time.monotonic() + POLL_CAP_SECONDS
    job: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        g = client.get(f"/jobs/{job_id}")
        assert g.status_code == 200, (g.status_code, g.text[:300])
        job = g.json()
        if job["status"] in TERMINAL:
            break
        time.sleep(POLL_SECONDS)
    assert job.get("status") == "COMPLETED", (
        f"job {job_id} ended {job.get('status')!r} after polling: "
        f"error={job.get('error')!r} message={job.get('message')!r}"
    )
    result = job["result"]

    prefix = "/outputs/"
    assert result["stream_url"].startswith(prefix), result["stream_url"]
    wav = paths.user_outputs_dir() / unquote(result["stream_url"][len(prefix):])
    y, sr = sf.read(str(wav), dtype="float32")
    return {"job_id": job_id, "result": result, "wav": wav, "y": y, "sr": sr,
            "tempos": tempos, "events": events}


# ---------------------------------------------------------------------------
# Tests on the one render
# ---------------------------------------------------------------------------

def test_scanned_click_tracks_read_120_and_124_before_the_mix(mixed):
    assert mixed["events"] == ["scan", "tempo_gate", "post"]
    for name, _f, bpm, _p, _s in TRACKS:
        for fname, value in mixed["tempos"][name].items():
            assert abs(value - bpm) <= TEMPO_TOLERANCE_BPM, f"{name} {fname} {value} vs {bpm}"
    assert set(mixed["tempos"]) == {"Click 120", "Click 124"}, mixed["tempos"]


def test_engine_used_the_scanned_tempos(mixed):
    res = mixed["result"]
    assert abs(res["bpm_a"] - 120.0) <= TEMPO_TOLERANCE_BPM, f"bpm_a {res['bpm_a']}"
    assert abs(res["bpm_b"] - 124.0) <= TEMPO_TOLERANCE_BPM, f"bpm_b {res['bpm_b']}"


def test_mix_wav_exists_under_isolated_outputs(mixed):
    wav = Path(mixed["wav"])
    assert wav.is_file(), wav
    conftest_outputs = Path(os.environ["REMIXMATE_LIBRARY_OUTPUTS_DIR"]).resolve()
    assert wav.resolve().is_relative_to(conftest_outputs), (wav, conftest_outputs)
    assert not wav.resolve().is_relative_to((ROOT / "outputs").resolve()), wav


def test_mix_duration_matches_the_plan(mixed):
    """Two checks, both against the plan, not against a measured value.

    1. The cut points the result carries: song A plays alone until
       timeline.overlap_start_s, then song B runs to its end, stretched to
       A's tempo (tempo_ratio = bpm_a / bpm_b), so

           expected = overlap_start_s + SOURCE_SECONDS / tempo_ratio

    2. The spec nominal, a mix out at the end of A with an 8 bar transition
       at 120 BPM (8 * 4 * 60 / 120 = 16 s) and B at 60 * 124 / 120 = 62 s:

           60 + 62 - 16 = 106 s

    Both within 1.0 s. The file length must also equal the reported
    duration_sec within 0.05 s (duration_sec is rounded to 0.1 s)."""
    res = mixed["result"]
    duration = sf.info(str(mixed["wav"])).duration
    assert abs(duration - res["duration_sec"]) <= 0.05, (duration, res["duration_sec"])

    from_cuts = res["timeline"]["overlap_start_s"] + SOURCE_SECONDS / res["tempo_ratio"]
    assert abs(duration - from_cuts) <= 1.0, (duration, from_cuts, res["timeline"])

    transition_s = TRANSITION_BARS * 4 * 60.0 / 120.0
    nominal = SOURCE_SECONDS + SOURCE_SECONDS * 124.0 / 120.0 - transition_s
    assert nominal == pytest.approx(106.0)
    assert abs(duration - nominal) <= 1.0, (duration, nominal)


def test_mix_loudness_is_within_1_lu_of_target(mixed):
    target = cfg.audio.target_lufs
    assert target == -14.0
    y, sr = mixed["y"], mixed["sr"]
    lufs = pyloudnorm.Meter(sr).integrated_loudness(np.asarray(y, dtype=np.float64))
    assert abs(lufs - target) <= LOUDNESS_TOLERANCE_LU, f"mix {lufs:.3f} LUFS, target {target}"


# ---------------------------------------------------------------------------
# Red checks: the tempo gate can fail
# ---------------------------------------------------------------------------

def test_tempo_gate_rejects_the_120_track_as_124(mixed):
    """The numeric branch: a real stored tempo 4 BPM off goes red, with the value."""
    with pytest.raises(AssertionError, match=r"Click 120: meta\.json bpm 1[12]\d\.\d\d is"):
        _assert_tempo("Click 120", 124.0)


def test_pure_sine_in_place_of_the_124_track_fails_the_tempo_gate(iso, capsys):
    """A 60 s constant 440 Hz sine has no beats. Analysis must say so plainly,
    and the tempo gate must fail on it, never skip or pass."""
    sine_dir = iso.base / "sine_music"
    sine_dir.mkdir()
    t = np.arange(int(SOURCE_SECONDS * SR)) / SR
    sf.write(str(sine_dir / "Sine 124.wav"), (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32), SR)

    rc = scan.main([str(sine_dir), "--analyze"])
    out = capsys.readouterr()
    assert rc == 1, (rc, out.out, out.err)
    assert "no beats or bars detected" in out.out + out.err, (out.out, out.err)

    with pytest.raises(AssertionError, match="Sine 124: no meta.json"):
        _assert_tempo("Sine 124", 124.0)
