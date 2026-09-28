"""
tests/test_silent_failures.py — B3.5 "silent failures become real ones",
minus the items owned by the render family (record_render guard,
manifest.session, chain cancel check).

Spec (plans/audit-the-project-and-sorted-flask.md, B3.5). One test per item:

  1. inline CLAP build task: an exception ends the job FAILED
  2. inline LUFS calibration task: an exception ends the job FAILED
  3. analysis fallbacks log at WARNING (not DEBUG)
  4. Serato export writes to a distinct temp path (no SameFileError)
  5. POST /stems/compress accepts a {song} JSON body
  6. POST /stems/compress-batch honours {songs}
  7. GET /setlist/camelot/modulation 422s on an invalid key (real error type)
  8. analyze missing skips playlist artifacts and folders without audio
  9. MusicIndex.upsert_song skips playlist artifacts
 10. MusicBrainz lookups run only with REMIXMATE_MUSICBRAINZ_ENABLED=1
 11. nltk.download is guarded (an offline download never raises)
 12. POST /setlist/wordplay returns 503 "wordplay unavailable" without the extra
 13. a failed Genius fetch is not cached
 14. quality_passed is judged on the pre limiter signal, not the limiter ceiling
 15. beatless audio yields a clean job error "no beats detected"

Nothing here touches the shared catalog: every library read is redirected to
tmp_path, and every job submission is captured or run through the real job
store against the conftest's isolated data dir.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import FastAPI
from fastapi.testclient import TestClient

sf = pytest.importorskip("soundfile")

SR = 44100
TERMINAL = {"done", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _app_with(*routers) -> FastAPI:
    app = FastAPI()
    for r in routers:
        app.include_router(r)
    return app


def _status(job: dict) -> str:
    s = job["status"]
    return s.value if hasattr(s, "value") else str(s)


def _wait_terminal(job_store, job_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = job_store.get_job(job_id)
        if job is not None and _status(job) in TERMINAL:
            return job
        time.sleep(0.02)
    return job_store.get_job(job_id)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), SR)


def _tone(seconds: float, amp: float = 0.1, hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _click_track(seconds: float = 10.0, bpm: float = 120.0) -> np.ndarray:
    """Clicks on the beat plus a quiet pad: beat tracking finds a grid here."""
    n = int(seconds * SR)
    y = _tone(seconds, amp=0.05, hz=220.0)
    step = int(SR * 60.0 / bpm)
    for i in range(0, n, step):
        seg = np.arange(min(2000, n - i))
        y[i:i + len(seg)] += (np.exp(-seg / 300.0)).astype(np.float32)
    return y


@pytest.fixture
def fake_lib(tmp_path, monkeypatch):
    """Redirect every LIBRARY_DIR binding the routes under test read."""
    import scripts.api.routers._helpers as helpers
    import scripts.core.paths as paths

    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setattr(paths, "LIBRARY_DIR", lib)
    monkeypatch.setattr(helpers, "LIBRARY_DIR", lib)
    for modname in ("scripts.api.routers.library", "scripts.api.routers.stems",
                    "scripts.api.routers.analysis", "scripts.api.task_modules.analysis",
                    "scripts.core.music_index"):
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, "LIBRARY_DIR"):
            monkeypatch.setattr(mod, "LIBRARY_DIR", lib)
    return lib


@pytest.fixture
def no_auth(monkeypatch):
    monkeypatch.delenv("REMIXMATE_AUTH_ENABLED", raising=False)


@pytest.fixture
def capture_submit(monkeypatch):
    """Capture job submissions instead of running them; return (calls, real)."""
    import scripts.api.jobs as job_store

    real = job_store.submit_job
    calls: list[tuple[str, object, dict]] = []

    def _fake(job_id, fn, **kwargs):
        calls.append((job_id, fn, kwargs))

    monkeypatch.setattr(job_store, "submit_job", _fake)
    return calls, real


# ---------------------------------------------------------------------------
# 1 + 2. inline tasks propagate exceptions -> job FAILED
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_clap_build_task_failure_marks_job_failed(monkeypatch, capture_submit, no_auth):
    import scripts.api.jobs as job_store
    from scripts.api.routers.analysis import build_clap_index

    calls, real_submit = capture_submit

    fake_digger = types.ModuleType("scripts.core.crate_digger")

    def _boom():
        raise RuntimeError("clap boom")

    fake_digger.get_digger = _boom
    monkeypatch.setitem(sys.modules, "scripts.core.crate_digger", fake_digger)

    build_clap_index(force=False)
    assert len(calls) == 1, "route must submit exactly one job"
    job_id, fn, kwargs = calls[0]

    # Run the inline task through the real job wrapper.
    real_submit(job_id, fn, **kwargs)
    job = _wait_terminal(job_store, job_id)

    assert _status(job) == "failed", f"job ended {_status(job)!r}: {job.get('message')!r}"
    assert "clap boom" in (job.get("error") or job.get("message") or "")


@pytest.mark.integration
def test_lufs_calibration_task_failure_marks_job_failed(monkeypatch, capture_submit, no_auth):
    import scripts.api.jobs as job_store
    import scripts.core.mastering as mastering
    from scripts.api.routers.library import calibrate_stem_lufs

    calls, real_submit = capture_submit

    def _boom():
        raise RuntimeError("lufs boom")

    monkeypatch.setattr(mastering, "analyze_library_stem_targets", _boom)

    calibrate_stem_lufs()
    assert len(calls) == 1
    job_id, fn, kwargs = calls[0]

    real_submit(job_id, fn, **kwargs)
    job = _wait_terminal(job_store, job_id)

    assert _status(job) == "failed", f"job ended {_status(job)!r}: {job.get('message')!r}"
    assert "lufs boom" in (job.get("error") or job.get("message") or "")


# ---------------------------------------------------------------------------
# 3. analysis fallbacks log at WARNING
# ---------------------------------------------------------------------------

@pytest.mark.dj_analysis
def test_analysis_fallback_logs_at_warning(monkeypatch, caplog):
    import scripts.core.key_detection as kd
    from scripts.core.dj_analysis import analyze_structure

    def _boom(*a, **k):
        raise RuntimeError("key boom")

    monkeypatch.setattr(kd, "detect_key", _boom)

    caplog.set_level(logging.DEBUG)
    analyze_structure(_click_track(10.0), SR)

    hits = [r for r in caplog.records if "key boom" in r.getMessage()]
    assert hits, "the key detection fallback logged nothing at all"
    levels = {r.levelno for r in hits}
    assert min(levels) >= logging.WARNING, (
        f"fallback logged at {[logging.getLevelName(lvl) for lvl in levels]}, expected WARNING"
    )


# ---------------------------------------------------------------------------
# 4. Serato export: no SameFileError
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_serato_export_no_same_file_error(fake_lib, no_auth):
    pytest.importorskip("mutagen")
    from scripts.api.routers.library import router

    song = "Serato_Song"
    d = fake_lib / song
    d.mkdir()
    # Minimal MPEG-1 Layer III frames (128 kbps, 44.1 kHz): mutagen can tag it.
    frame = bytes([0xFF, 0xFB, 0x90, 0x00]) + b"\x00" * 413
    (d / "full.mp3").write_bytes(frame * 20)
    (d / "analysis.json").write_text(json.dumps({"bpm": 120.0, "phrase_boundaries": [0.0, 16.0]}))

    with TestClient(_app_with(router), raise_server_exceptions=False) as c:
        r = c.get(f"/library/{song}/export-cues", params={"fmt": "serato"})

    assert r.status_code == 200, f"{r.status_code}: {r.text[:200]}"
    assert r.headers.get("content-type", "").startswith("audio/mpeg")
    assert len(r.content) > 0


# ---------------------------------------------------------------------------
# 5 + 6. /stems/compress body contract
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_stems_compress_accepts_song_json(fake_lib, no_auth, capture_submit):
    from scripts.api.routers.stems import router

    song = "Compress_Me"
    _write_wav(fake_lib / song / "vocals.wav", _tone(1.0))
    calls, _ = capture_submit

    with TestClient(_app_with(router), raise_server_exceptions=False) as c:
        r = c.post("/stems/compress", json={"song": song})

    assert r.status_code == 202, f"{r.status_code}: {r.text[:200]}"
    assert len(calls) == 1
    assert calls[0][2].get("song") == song


@pytest.mark.integration
def test_stems_compress_batch_honours_songs(fake_lib, no_auth, capture_submit):
    from scripts.api.routers.stems import router

    picked = "Picked_Song"
    other = "Other_Song"
    for name in (picked, other):
        _write_wav(fake_lib / name / "vocals.wav", _tone(1.0))
    calls, _ = capture_submit

    with TestClient(_app_with(router), raise_server_exceptions=False) as c:
        r = c.post("/stems/compress-batch", json={"songs": [picked]})

    assert r.status_code == 202, f"{r.status_code}: {r.text[:200]}"
    assert len(calls) == 1
    assert calls[0][2].get("songs") == [picked], (
        f"batch task received songs={calls[0][2].get('songs')!r}; the selection was ignored"
    )


# ---------------------------------------------------------------------------
# 7. camelot modulation 422 on a real invalid key
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_camelot_modulation_invalid_key_422():
    from scripts.api.routers.setlist import router

    with TestClient(_app_with(router), raise_server_exceptions=False) as c:
        bad = c.get("/setlist/camelot/modulation", params={"from_key": "XYZ", "to_key": "8A"})
        good = c.get("/setlist/camelot/modulation", params={"from_key": "8A", "to_key": "8B"})

    assert good.status_code == 200, f"control pair 8A->8B: {good.status_code}: {good.text[:200]}"
    assert bad.status_code == 422, (
        f"invalid key returned {bad.status_code}: {bad.text[:200]}"
    )


# ---------------------------------------------------------------------------
# 8 + 9. playlist artifacts skipped
# ---------------------------------------------------------------------------

def _playlist_dir(lib: Path, name: str, with_audio: bool = True) -> Path:
    d = lib / name
    d.mkdir(parents=True)
    (d / "full.info.json").write_text(json.dumps({
        "title": name, "n_entries": 5, "playlist_count": 5,
    }))
    if with_audio:
        _write_wav(d / "full.wav", _tone(1.0))
    return d


def test_analyze_missing_skips_playlist_and_no_audio(fake_lib, monkeypatch):
    import scripts.api.task_modules.analysis as task_mod
    import scripts.core.analysis_pipeline as pipeline

    real_song = "Real_Song"
    _write_wav(fake_lib / real_song / "full.wav", _tone(1.0))
    _playlist_dir(fake_lib, "Walking_playlist")
    (fake_lib / "No_Audio_Folder").mkdir()
    (fake_lib / "No_Audio_Folder" / "meta.json").write_text("{}")

    analysed: list[str] = []

    def _fake_analysis(name, key_profile="auto", progress_cb=None):
        analysed.append(name)
        return {"bpm": 120.0}

    monkeypatch.setattr(pipeline, "run_song_analysis", _fake_analysis)
    monkeypatch.setattr(task_mod, "_index_upsert", lambda name: None)
    monkeypatch.setattr(task_mod, "update_job", lambda *a, **k: None)
    monkeypatch.setattr(task_mod, "log_audit", lambda *a, **k: None)

    result = task_mod.task_analyze_missing("job-analyze-missing")

    assert real_song in analysed
    assert "Walking_playlist" not in analysed, "playlist artifact was analysed as a song"
    assert "No_Audio_Folder" not in analysed, "folder without audio was analysed"
    failed_names = {f["song"] for f in result["failed"]}
    assert not ({"Walking_playlist", "No_Audio_Folder"} & failed_names), (
        f"skipped folders must not be reported as failures: {result['failed']}"
    )


def test_upsert_song_skips_playlist_artifact(fake_lib, monkeypatch):
    import scripts.core.music_index as mi

    monkeypatch.setattr(mi, "LIBRARY_DIR", fake_lib)
    name = "Walking_playlist"
    d = _playlist_dir(fake_lib, name)
    # A complete-looking meta.json: without the playlist check this indexes fine.
    (d / "meta.json").write_text(json.dumps({
        "bpm": 124.0, "key": "A", "mode": "minor", "energy_mean": 0.6,
    }))

    idx = mi.MusicIndex()
    idx._songs.pop(name, None)

    ok = idx.upsert_song(name)

    assert ok is False, "upsert_song indexed a playlist artifact"
    assert name not in idx._songs


# ---------------------------------------------------------------------------
# 10. MusicBrainz gated by env
# ---------------------------------------------------------------------------

def test_musicbrainz_lookup_gated_by_env(tmp_path, monkeypatch):
    from scripts.core.track_metadata import MetadataClient

    client = MetadataClient(getsongbpm_key="", lastfm_key="",
                            cache_path=tmp_path / "meta.db")
    mb_calls: list[tuple[str, str]] = []

    def _record(title, artist=""):
        mb_calls.append((title, artist))
        return None

    monkeypatch.setattr(client._musicbrainz, "lookup", _record)

    monkeypatch.delenv("REMIXMATE_MUSICBRAINZ_ENABLED", raising=False)
    client.lookup("Some Title", "Some Artist", force_refresh=True)
    assert mb_calls == [], "MusicBrainz was queried without REMIXMATE_MUSICBRAINZ_ENABLED=1"

    monkeypatch.setenv("REMIXMATE_MUSICBRAINZ_ENABLED", "1")
    client.lookup("Some Title", "Some Artist", force_refresh=True)
    assert len(mb_calls) == 1, "MusicBrainz must run when the flag is set"


# ---------------------------------------------------------------------------
# 11 + 12 + 13. wordplay extra
# ---------------------------------------------------------------------------

def test_nltk_download_guarded(monkeypatch):
    import scripts.core.wordplay as wp

    class _Data:
        @staticmethod
        def find(_name):
            raise LookupError("resource not found")

    def _download(*a, **k):
        raise OSError("offline: nltk.download reached the network")

    fake_nltk = types.SimpleNamespace(data=_Data(), download=_download)
    fake_stopwords = types.SimpleNamespace(words=lambda lang: [])

    monkeypatch.setattr(wp, "nltk", fake_nltk, raising=False)
    monkeypatch.setattr(wp, "stopwords", fake_stopwords, raising=False)
    monkeypatch.setattr(wp, "_HAS_NLTK", True)
    monkeypatch.setattr(wp, "_ensure_nltk_data_done", False)
    monkeypatch.setattr(wp, "_STOP_WORDS", set())

    try:
        wp._ensure_nltk_data()
    except OSError as exc:
        pytest.fail(f"nltk.download error escaped _ensure_nltk_data: {exc}")


@pytest.mark.integration
def test_wordplay_route_503_when_extra_missing(monkeypatch):
    import scripts.core.wordplay as wp
    from scripts.api.routers.setlist import router

    monkeypatch.setattr(wp, "_HAS_GENIUS", False)
    monkeypatch.setattr(wp, "_HAS_NLTK", False)

    body = {
        "genius_token": "test-token",
        "tracks": [
            {"track_id": "a", "title": "Song A", "artist": "Artist A"},
            {"track_id": "b", "title": "Song B", "artist": "Artist B"},
        ],
    }
    with TestClient(_app_with(router), raise_server_exceptions=False) as c:
        r = c.post("/setlist/wordplay", json=body)

    assert r.status_code == 503, f"{r.status_code}: {r.text[:200]}"
    assert "wordplay unavailable" in r.text.lower()


def test_failed_genius_fetch_not_cached(tmp_path, monkeypatch):
    import scripts.core.wordplay as wp

    monkeypatch.setattr(wp, "_CACHE_DIR", tmp_path / "lyrics")
    monkeypatch.setattr(wp.time, "sleep", lambda s: None)

    class _FailingGenius:
        def search_song(self, *a, **k):
            raise RuntimeError("genius 502")

    class _Song:
        lyrics = "hello darkness my old friend"

    class _WorkingGenius:
        def search_song(self, *a, **k):
            return _Song()

    assert wp._fetch_lyrics_genius("Title", "Artist", _FailingGenius()) is None
    assert wp._load_from_cache("Title", "Artist") is None, (
        "a failed fetch was written to the lyric cache"
    )
    # The next attempt must reach Genius again instead of replaying the failure.
    assert wp._fetch_lyrics_genius("Title", "Artist", _WorkingGenius()) == _Song.lyrics


# ---------------------------------------------------------------------------
# 14. quality_passed judged against the pre limiter signal
# ---------------------------------------------------------------------------

def test_quality_passed_ignores_limiter_ceiling(monkeypatch):
    pytest.importorskip("scipy")
    import scripts.core.mastering as mastering

    ceiling_db = -1.0
    ceiling = 10.0 ** (ceiling_db / 20.0)

    # A limiter that holds its ceiling exactly, like the 38 stored renders
    # whose report reads "N sample(s) above -1.0 dBFS after limiting" with
    # peak -1.0: the output sits AT the ceiling, never above it.
    def _brickwall(audio, ceiling_db=-1.0, sr=SR, **_):
        c = 10.0 ** (ceiling_db / 20.0)
        return np.clip(audio.astype(np.float32), -c, c)

    monkeypatch.setattr(mastering, "apply_limiter", _brickwall)

    # Clean input (peak well under 0 dBFS) with a crest factor that makes the
    # normaliser push a few clicks over the ceiling, so limiting engages.
    x = _tone(20.0, amp=0.1, hz=1000.0)
    x[::SR // 2] += 0.45
    assert np.abs(x).max() < 1.0

    normalised, _ = mastering.normalize_to_lufs(x, SR, target_lufs=-14.0)
    assert np.abs(normalised).max() > ceiling, "precondition: limiting must engage"

    out, report = mastering.master_mix(x, SR, target_lufs=-14.0, ceiling_db=ceiling_db)

    at_ceiling = int((np.abs(out) >= np.float32(ceiling)).sum())
    assert at_ceiling > 0, "precondition: the limiter output must sit at its ceiling"
    assert abs(report.lufs_integrated - (-14.0)) <= 3.0, report.notes

    assert report.has_clipping is False, (
        f"limiter ceiling counted as clipping: clip_count={report.clip_count}, notes={report.notes}"
    )
    assert report.passed is True, report.notes


# ---------------------------------------------------------------------------
# 15. beatless audio -> clean job error
# ---------------------------------------------------------------------------

@pytest.mark.dj_analysis
def test_beatless_audio_clean_job_error(fake_lib, monkeypatch):
    import scripts.api.task_modules.analysis as task_mod
    import scripts.core.genre as genre_mod

    song = "Silent_Song"
    _write_wav(fake_lib / song / "full.wav", np.zeros(SR * 10, dtype=np.float32))

    monkeypatch.setattr(
        genre_mod, "detect_genre",
        lambda audio, sr=SR: types.SimpleNamespace(genre="ambient", confidence=0.0),
    )
    monkeypatch.setattr(task_mod, "update_job", lambda *a, **k: None)
    monkeypatch.setattr(task_mod, "log_audit", lambda *a, **k: None)

    with pytest.raises(Exception) as ei:
        task_mod.task_analyze("job-beatless", song)

    msg = str(ei.value)
    assert "no beats detected" in msg.lower(), (
        f"beatless audio raised {type(ei.value).__name__}: {msg!r}"
    )
