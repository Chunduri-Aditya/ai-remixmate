"""
tests/test_stems.py — scripts/core/stems.py, focused on the
DJ_ENGINE_IMPROVEMENT_PLAN.md §2.2 preview-tier separation path.

No real Demucs/torch involved: subprocess.run is monkeypatched to fake a
successful Demucs invocation that writes zero-filled stem files of the
requested length, so these tests exercise the real truncation/output-routing
logic in separate_song_stems() without needing the actual model.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 44100


def _fake_demucs_run(model_name: str):
    """Return a subprocess.run replacement that writes stem files matching
    whatever audio file it was actually asked to process — the key thing
    these tests need to observe: how long is the file Demucs was pointed
    at, and where does it live."""

    def _run(cmd, capture_output=True, text=True):
        # cmd == [python_exe, "-m", "demucs", "-n", model, "-d", dev, "-o", tmp, str(demucs_input)]
        tmp = cmd[cmd.index("-o") + 1]
        demucs_input = Path(cmd[-1])
        input_audio, _ = sf.read(str(demucs_input))
        out_dir = Path(tmp) / model_name / demucs_input.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem in ("vocals", "drums", "bass", "other"):
            sf.write(str(out_dir / f"{stem}.wav"), input_audio, SR)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _run


@pytest.fixture
def fake_library(tmp_path, monkeypatch):
    """A library/ containing one song with a 5-second full.wav, with
    scripts.core.paths.LIBRARY_DIR redirected so scripts.core.stems's
    song_dir() resolves into tmp_path instead of the real library."""
    import scripts.core.paths as paths

    lib = tmp_path / "library"
    song = "Test Song"
    song_path = lib / song
    song_path.mkdir(parents=True)

    duration_s = 5.0
    audio = 0.1 * np.sin(2 * np.pi * 440 * np.arange(int(duration_s * SR)) / SR).astype(np.float32)
    sf.write(str(song_path / "full.wav"), audio, SR)

    monkeypatch.setattr(paths, "LIBRARY_DIR", lib)
    return song, song_path, duration_s


class TestFullEnhancedFallback:
    """A song can end up with ONLY full_enhanced.wav and no full.wav — e.g.
    Demucs failed after enhancement already wrote its output, or full.wav
    was later pruned/evicted before separation ever completed. This is the
    real state discovered in the live library's "unprocessed"/"missing
    stems" backlog (185 songs, all sharing this exact shape). Without this
    fallback, separate_song_stems() can never recover such a song without a
    full re-download from source."""

    def test_falls_back_to_full_enhanced_when_full_wav_missing(
        self, fake_library, monkeypatch
    ):
        from scripts.core import stems as stems_mod

        song, song_path, full_duration = fake_library
        # Simulate the stuck state: full.wav gone, only a previously-written
        # full_enhanced.wav remains (as produced by a prior failed attempt).
        audio, sr = sf.read(str(song_path / "full.wav"))
        (song_path / "full.wav").unlink()
        sf.write(str(song_path / "full_enhanced.wav"), audio, sr, subtype="PCM_24")

        monkeypatch.setattr(stems_mod.subprocess, "run", _fake_demucs_run("htdemucs"))

        result = stems_mod.separate_song_stems(song, enhance=True)

        assert result.success, result.error
        assert result.enhance_info.get("reused_existing_enhanced_file") is True
        for stem_path in result.stems.values():
            assert stem_path.parent == song_path
        produced_audio, out_sr = sf.read(str(result.stems["vocals"]))
        assert len(produced_audio) / out_sr == pytest.approx(full_duration, abs=0.05)

    def test_no_double_enhancement_when_reusing_full_enhanced(
        self, fake_library, monkeypatch
    ):
        """Reusing an already-enhanced file must not re-run enhance_audio()
        — doing so would double-apply LUFS normalisation/compression."""
        from scripts.core import stems as stems_mod

        song, song_path, _dur = fake_library
        audio, sr = sf.read(str(song_path / "full.wav"))
        (song_path / "full.wav").unlink()
        sf.write(str(song_path / "full_enhanced.wav"), audio, sr, subtype="PCM_24")

        monkeypatch.setattr(stems_mod.subprocess, "run", _fake_demucs_run("htdemucs"))

        def _boom(*a, **kw):
            raise AssertionError("enhance_audio() must not be called when reusing full_enhanced.wav")

        monkeypatch.setattr("scripts.core.audio_enhance.enhance_audio", _boom, raising=False)

        result = stems_mod.separate_song_stems(song, enhance=True)
        assert result.success, result.error

    def test_error_when_neither_full_wav_nor_enhanced_exists(self, fake_library):
        from scripts.core import stems as stems_mod

        song, song_path, _dur = fake_library
        (song_path / "full.wav").unlink()

        result = stems_mod.separate_song_stems(song, enhance=True)
        assert not result.success
        assert "no source audio" in (result.error or "").lower()

    def test_preview_seconds_combined_with_fallback(
        self, fake_library, tmp_path, monkeypatch
    ):
        """The fallback and preview-tier separation (§2.2) are independent
        features — this combination (backlog song + a preview request) was
        never exercised by any existing test. Must still truncate correctly
        and must never write into library/<song>/."""
        from scripts.core import stems as stems_mod

        song, song_path, full_duration = fake_library
        audio, sr = sf.read(str(song_path / "full.wav"))
        (song_path / "full.wav").unlink()
        sf.write(str(song_path / "full_enhanced.wav"), audio, sr, subtype="PCM_24")

        monkeypatch.setattr(stems_mod.subprocess, "run", _fake_demucs_run("htdemucs"))

        preview_s = 2.0
        out_dir = tmp_path / "preview_out"
        result = stems_mod.separate_song_stems(
            song, enhance=True, preview_seconds=preview_s, output_dir=out_dir,
        )

        assert result.success, result.error
        assert result.enhance_info.get("reused_existing_enhanced_file") is True
        for stem_path in result.stems.values():
            assert stem_path.parent == out_dir
        produced_audio, out_sr = sf.read(str(result.stems["vocals"]))
        assert len(produced_audio) / out_sr == pytest.approx(preview_s, abs=0.05)
        assert len(produced_audio) / out_sr < full_duration
        # The library's own full_enhanced.wav must be untouched by a preview call.
        assert (song_path / "full_enhanced.wav").exists()
        preview_audio, _ = sf.read(str(song_path / "full_enhanced.wav"))
        assert len(preview_audio) / sr == pytest.approx(full_duration, abs=0.05)

    def test_fallback_with_enhance_false(self, fake_library, monkeypatch):
        """enhance=False + fallback: full_enhanced.wav is fed to Demucs
        as-is (it's already enhanced audio regardless), no re-enhancement
        attempted either way since enhance=False skips that branch entirely."""
        from scripts.core import stems as stems_mod

        song, song_path, full_duration = fake_library
        audio, sr = sf.read(str(song_path / "full.wav"))
        (song_path / "full.wav").unlink()
        sf.write(str(song_path / "full_enhanced.wav"), audio, sr, subtype="PCM_24")

        monkeypatch.setattr(stems_mod.subprocess, "run", _fake_demucs_run("htdemucs"))

        result = stems_mod.separate_song_stems(song, enhance=False)

        assert result.success, result.error
        assert result.enhance_info == {"skipped": True}
        produced_audio, out_sr = sf.read(str(result.stems["vocals"]))
        assert len(produced_audio) / out_sr == pytest.approx(full_duration, abs=0.05)


class TestPreviewSeparationSafety:
    """§2.2's core safety requirement: a preview separation must never be
    able to land in / truncate a song's permanent library stems."""

    def test_preview_seconds_without_output_dir_raises(self, fake_library):
        from scripts.core.stems import separate_song_stems

        song, _song_path, _dur = fake_library
        with pytest.raises(ValueError, match="output_dir"):
            separate_song_stems(song, preview_seconds=30.0)

    def test_preview_writes_to_output_dir_not_library(self, fake_library, tmp_path, monkeypatch):
        from scripts.core import stems as stems_mod

        song, song_path, _dur = fake_library
        monkeypatch.setattr(stems_mod.subprocess, "run", _fake_demucs_run("htdemucs"))

        preview_dir = tmp_path / "preview_out"
        result = stems_mod.separate_song_stems(
            song, enhance=False, preview_seconds=2.0, output_dir=preview_dir,
        )

        assert result.success, result.error
        for stem_path in result.stems.values():
            assert stem_path.parent == preview_dir
        # The permanent library folder must not have gained any stem files.
        assert not (song_path / "vocals.wav").exists()
        assert not (song_path / "drums.wav").exists()


class TestPreviewSeparationTruncation:
    """§2.2's actual speed win: Demucs must only ever see `preview_seconds`
    of audio, in both the enhance=True and enhance=False paths — the bug
    this guards against is a truncation that only reaches Demucs via the
    enhancement chain and silently no-ops when enhance=False."""

    def test_no_enhance_truncates_before_demucs(self, fake_library, tmp_path, monkeypatch):
        from scripts.core import stems as stems_mod

        song, _song_path, full_duration = fake_library
        monkeypatch.setattr(stems_mod.subprocess, "run", _fake_demucs_run("htdemucs"))

        preview_s = 2.0
        out_dir = tmp_path / "preview_out"
        result = stems_mod.separate_song_stems(
            song, enhance=False, preview_seconds=preview_s, output_dir=out_dir,
        )

        assert result.success, result.error
        produced_audio, sr = sf.read(str(result.stems["vocals"]))
        assert sr == SR
        assert len(produced_audio) / sr == pytest.approx(preview_s, abs=0.05)
        assert len(produced_audio) / sr < full_duration

    def test_enhance_failure_falls_back_to_truncated_not_full_audio(
        self, fake_library, tmp_path, monkeypatch
    ):
        """If the enhancement chain itself raises, the pre-§2.2 code fell
        back to wav_path — the ORIGINAL, untruncated file — which would
        silently defeat preview mode. It must fall back to the truncated
        preview input instead."""
        from scripts.core import stems as stems_mod

        song, _song_path, full_duration = fake_library
        monkeypatch.setattr(stems_mod.subprocess, "run", _fake_demucs_run("htdemucs"))

        def _boom(*a, **kw):
            raise RuntimeError("enhancement exploded")

        monkeypatch.setattr("scripts.core.audio_enhance.enhance_audio", _boom, raising=False)

        preview_s = 2.0
        out_dir = tmp_path / "preview_out"
        result = stems_mod.separate_song_stems(
            song, enhance=True, preview_seconds=preview_s, output_dir=out_dir,
        )

        assert result.success, result.error
        produced_audio, sr = sf.read(str(result.stems["vocals"]))
        assert len(produced_audio) / sr == pytest.approx(preview_s, abs=0.05)
        assert len(produced_audio) / sr < full_duration

    def test_full_quality_path_unchanged_when_preview_seconds_omitted(
        self, fake_library, monkeypatch
    ):
        """Default call (no preview_seconds, no output_dir) must behave
        exactly as before §2.2 — full duration, written into library/<song>/."""
        from scripts.core import stems as stems_mod

        song, song_path, full_duration = fake_library
        monkeypatch.setattr(stems_mod.subprocess, "run", _fake_demucs_run("htdemucs"))

        result = stems_mod.separate_song_stems(song, enhance=False)

        assert result.success, result.error
        for stem_path in result.stems.values():
            assert stem_path.parent == song_path
        produced_audio, sr = sf.read(str(result.stems["vocals"]))
        assert len(produced_audio) / sr == pytest.approx(full_duration, abs=0.05)
