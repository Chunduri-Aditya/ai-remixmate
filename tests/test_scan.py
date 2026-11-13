"""scripts.scan: folder import into an isolated library.

Demucs and analysis are stubbed. In process tests redirect paths.LIBRARY_DIR
and the library manager to a tmp root; subprocess tests pass the six scratch
storage variables in the child env only (RULES.md, "Do")."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts import scan
from scripts.core import library as library_mod
from scripts.core import paths
from scripts.core.library import LibraryManager

ROOT = Path(__file__).resolve().parents[1]
SR = 44100

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_audio(path: Path, seconds: float = 1.0, hz: float = 440.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * SR)) / SR
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32), SR)
    return path


def _song_dirs(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def _scratch_env(tmp_path: Path) -> dict[str, str]:
    """Child env with all six storage variables under tmp_path and the three
    overriding names removed. Never applied to the pytest process."""
    env = dict(os.environ)
    for k in ("DATABASE_URL", "REMIXMATE_LIBRARY_DIR", "REMIXMATE_OUTPUTS_DIR"):
        env.pop(k, None)
    s = tmp_path / "scratch"
    env.update({
        "REMIXMATE_LIBRARY_LIBRARY_DIR": str(s / "library"),
        "REMIXMATE_LIBRARY_OUTPUTS_DIR": str(s / "outputs"),
        "REMIXMATE_DATA_DIR": str(s / "data"),
        "REMIXMATE_METADATA_CACHE_PATH": str(s / "data" / "metadata.db"),
        "REMIXMATE_DATABASE_PATH": str(s / "data" / "remixmate.db"),
        "REMIXMATE_DATABASE_EMBEDDINGS_PATH": str(s / "data" / "song_embeddings.json"),
    })
    return env


def _run_cli(tmp_path: Path, *args: str, extra_env: dict[str, str] | None = None):
    env = _scratch_env(tmp_path)
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-m", "scripts.scan", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, errors="replace")


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """Isolated library root, read by scan through paths and library_mod."""
    root = tmp_path / "library"
    root.mkdir()
    monkeypatch.setattr(paths, "LIBRARY_DIR", root)
    mgr = LibraryManager(library_dir=root)
    monkeypatch.setattr(library_mod, "_manager", mgr)
    monkeypatch.setattr(library_mod, "get_library_manager", lambda: mgr)
    return SimpleNamespace(root=root, mgr=mgr)


@pytest.fixture
def client(lib, monkeypatch):
    """TestClient (no lifespan) whose library routes read the isolated root.
    scripts/api/routers/library.py and _helpers.py bind LIBRARY_DIR by name."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from scripts.api import main as main_mod
    from scripts.api.routers import _helpers
    from scripts.api.routers import library as library_router

    monkeypatch.setattr(library_router, "LIBRARY_DIR", lib.root)
    monkeypatch.setattr(_helpers, "LIBRARY_DIR", lib.root)
    monkeypatch.delenv("REMIXMATE_AUTH_ENABLED", raising=False)
    return TestClient(main_mod.app)


@pytest.fixture
def music(tmp_path):
    d = tmp_path / "music"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Rescan
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_rescan_same_folder_creates_no_duplicates(lib, music, capsys):
    _write_audio(music / "Track One.wav")
    assert scan.main([str(music)]) == 0
    capsys.readouterr()
    wav = lib.root / "Track One" / "full.wav"
    before = (wav.read_bytes(), wav.stat().st_mtime_ns)

    assert scan.main([str(music)]) == 0
    out = capsys.readouterr()
    assert "already in library" in out.out
    assert list(lib.mgr._load_index()) == ["Track One"]
    assert _song_dirs(lib.root) == ["Track One"]
    assert (wav.read_bytes(), wav.stat().st_mtime_ns) == before


# ---------------------------------------------------------------------------
# Non ASCII names
# ---------------------------------------------------------------------------

@needs_ffmpeg
@pytest.mark.parametrize("name", ["Café del Mar", "Beyoncé", "日本の歌"])
def test_non_ascii_names_survive_and_routes_accept_them(name, lib, music, client):
    """Route under test: GET /library/{name} (scripts/api/routers/library.py),
    which returns _song_info(_require_song(name)). The 404 for a missing name
    proves _require_song ran on the isolated root."""
    _write_audio(music / f"{name}.wav")
    assert scan.main([str(music)]) == 0
    assert _song_dirs(lib.root) == [name]
    assert paths.SONG_NAME_RE.match(name)

    r = client.get("/library")
    assert r.status_code == 200, r.text[:200]
    assert [s["name"] for s in r.json()["songs"]] == [name]

    r = client.get(f"/library/{name}")
    assert r.status_code == 200, r.text[:200]
    assert r.json()["name"] == name
    assert client.get(f"/library/{name} missing").status_code == 404


@needs_ffmpeg
def test_nfd_filename_imports_as_nfc_name(lib, music):
    nfd = unicodedata.normalize("NFD", "Café")
    assert nfd != "Café"
    _write_audio(music / f"{nfd}.wav")
    assert scan.main([str(music)]) == 0
    assert [unicodedata.is_normalized("NFC", n) for n in _song_dirs(lib.root)] == [True]
    assert _song_dirs(lib.root) == ["Café"]
    assert list(lib.mgr._load_index()) == ["Café"]


@needs_ffmpeg
def test_non_utf8_stdout_does_not_crash(tmp_path, music):
    _write_audio(music / "Café.wav")
    out = _run_cli(tmp_path, str(music), extra_env={"PYTHONIOENCODING": "ascii"})
    assert "Traceback" not in out.stderr, out.stderr
    assert out.returncode == 0, out.stderr
    assert _song_dirs(tmp_path / "scratch" / "library") == ["Café"]


# ---------------------------------------------------------------------------
# Folder level errors
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_empty_folder_exits_1_with_clear_message(tmp_path, music):
    (music / "notes.txt").write_text("not audio")
    out = _run_cli(tmp_path, str(music))
    assert out.returncode == 1, (out.stdout, out.stderr)
    assert "Traceback" not in out.stderr, out.stderr
    assert "no audio files found" in out.stderr
    assert str(music) in out.stderr
    for ext in sorted(scan.AUDIO_EXTS):
        assert ext in out.stderr, ext


def test_missing_folder_exits_2(tmp_path, lib, capsys):
    missing = tmp_path / "nope"
    assert scan.main([str(missing)]) == 2
    err = capsys.readouterr().err
    assert f"not a folder: {missing}" in err
    assert _song_dirs(lib.root) == []


def test_missing_ffmpeg_exits_2(lib, music, monkeypatch, capsys):
    _write_audio(music / "Song.wav")
    monkeypatch.setattr(scan.shutil, "which", lambda _name: None)
    assert scan.main([str(music)]) == 2
    assert "ffmpeg" in capsys.readouterr().err
    assert _song_dirs(lib.root) == []


# ---------------------------------------------------------------------------
# Per file failures
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_corrupt_file_reported_and_scan_continues(lib, music, capsys):
    (music / "Broken.mp3").write_bytes(b"this is not an mp3 " * 64)
    _write_audio(music / "Good.wav")

    assert scan.main([str(music)]) == 1
    out = capsys.readouterr()
    assert str(music / "Broken.mp3") in out.err
    assert _song_dirs(lib.root) == ["Good"]
    assert (lib.root / "Good" / "full.wav").exists()
    assert not (lib.root / "Broken").exists()

    assert scan.main([str(music)]) == 1
    out = capsys.readouterr()
    assert str(music / "Broken.mp3") in out.err
    assert "failed" in out.err
    assert not (lib.root / "Broken").exists()


@needs_ffmpeg
def test_stems_flag_calls_separator_once_per_song(lib, music, monkeypatch):
    from scripts.core import stems

    calls: list[tuple[str, bool]] = []

    def fake(name, enhance=True, **_kw):
        calls.append((name, enhance))
        (paths.song_dir(name) / "vocals.wav").write_bytes(b"stub")
        return SimpleNamespace(success=True, stems={}, error=None)

    monkeypatch.setattr(stems, "separate_song_stems", fake)
    _write_audio(music / "Alpha.wav")
    _write_audio(music / "Beta.wav")
    assert scan.main([str(music), "--stems"]) == 0
    assert scan.main([str(music), "--stems"]) == 0   # stems now on disk: no second call
    assert calls == [("Alpha", False), ("Beta", False)]


@needs_ffmpeg
def test_stems_failure_reported_and_folder_continues(lib, music, monkeypatch, capsys):
    from scripts.core import stems

    calls: list[str] = []

    def fake(name, enhance=True, **_kw):
        calls.append(name)
        if len(calls) == 1:
            return SimpleNamespace(success=False, stems={}, error="demucs exploded")
        return SimpleNamespace(success=True, stems={}, error=None)

    monkeypatch.setattr(stems, "separate_song_stems", fake)
    _write_audio(music / "Alpha.wav")
    _write_audio(music / "Beta.wav")
    assert scan.main([str(music), "--stems"]) == 1
    err = capsys.readouterr().err
    assert "Alpha" in err and "demucs exploded" in err
    assert calls == ["Alpha", "Beta"]
    assert sorted(lib.mgr._load_index()) == ["Alpha", "Beta"]
    assert _song_dirs(lib.root) == ["Alpha", "Beta"]


@needs_ffmpeg
def test_analyze_flag_needs_only_full_wav(lib, music, monkeypatch):
    from scripts.core import analysis_pipeline

    calls: list[str] = []
    monkeypatch.setattr(analysis_pipeline, "run_song_analysis", lambda name, **_kw: calls.append(name))
    _write_audio(music / "Solo.wav")
    assert scan.main([str(music), "--analyze"]) == 0
    assert sorted(p.name for p in (lib.root / "Solo").iterdir()) == ["full.wav"]
    assert calls == ["Solo"]


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_name_collision_in_one_scan_is_reported_not_overwritten(lib, music, capsys):
    first = _write_audio(music / "A" / "Song.wav", seconds=1.0)
    second = _write_audio(music / "B" / "song.flac", seconds=2.0)
    assert scan.main([str(music)]) == 1
    err = capsys.readouterr().err
    assert str(first) in err and str(second) in err
    assert _song_dirs(lib.root) == ["Song"]
    assert sf.info(str(lib.root / "Song" / "full.wav")).duration == pytest.approx(1.0, abs=0.05)


@needs_ffmpeg
def test_empty_name_file_skipped_with_message(lib, music, capsys):
    assert paths.sanitize_song_name("★★★") == ""
    star = _write_audio(music / "★★★.wav")
    assert scan.main([str(music)]) == 1
    assert str(star) in capsys.readouterr().err
    assert _song_dirs(lib.root) == []


@needs_ffmpeg
def test_hidden_and_appledouble_files_ignored(lib, music):
    (music / "._Track.mp3").write_bytes(b"\x00\x05\x16\x07AppleDouble" * 8)
    _write_audio(music / ".secret.wav")
    _write_audio(music / ".hidden" / "Hidden Song.wav")
    _write_audio(music / "Track.wav")
    assert [p.name for p in scan.find_audio(music)] == ["Track.wav"]
    assert scan.main([str(music)]) == 0
    assert _song_dirs(lib.root) == ["Track"]


@needs_ffmpeg
def test_files_inside_library_root_not_rescanned(tmp_path, monkeypatch, capsys):
    music = tmp_path / "music"
    root = music / "library"
    root.mkdir(parents=True)
    monkeypatch.setattr(paths, "LIBRARY_DIR", root)
    mgr = LibraryManager(library_dir=root)
    monkeypatch.setattr(library_mod, "_manager", mgr)
    monkeypatch.setattr(library_mod, "get_library_manager", lambda: mgr)

    _write_audio(music / "Song.wav")
    assert scan.main([str(music)]) == 0
    assert scan.main([str(music)]) == 0
    assert _song_dirs(root) == ["Song"]
    assert [p.name for p in scan.find_audio(music)] == ["Song.wav"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@needs_ffmpeg
@pytest.mark.parametrize("variant", ["same_manager", "second_manager"])
def test_registered_song_listed_by_get_library_without_restart(variant, lib, music, client,
                                                               monkeypatch):
    """GET /library is called before the scan (primes any cache) and after.
    second_manager: scan registers through a second LibraryManager on the same
    dir, as a separate process would; the app's manager must still see it."""
    before = client.get("/library")
    assert before.status_code == 200
    assert before.json()["songs"] == []

    if variant == "second_manager":
        other = LibraryManager(library_dir=lib.root)
        monkeypatch.setattr(library_mod, "get_library_manager", lambda: other)

    _write_audio(music / "Local Song.wav")
    assert scan.main([str(music)]) == 0

    after = client.get("/library")
    assert [s["name"] for s in after.json()["songs"]] == ["Local Song"]
    entry = lib.mgr.song_info("Local Song")
    assert entry is not None and entry.source == "local"
    assert entry.has_full_wav is True
    assert not (lib.root / "Local Song" / "license.json").exists()


def test_scan_never_imports_the_job_store(tmp_path):
    code = (
        "import sys\n"
        "import scripts.scan, scripts.core.stems, scripts.core.analysis_pipeline, scripts.core.library\n"
        "bad = sorted(m for m in sys.modules if m == 'scripts.api' or m.startswith('scripts.api.'))\n"
        "print('API_MODULES=' + ','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=_scratch_env(tmp_path),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "API_MODULES=\n" in out.stdout, out.stdout
