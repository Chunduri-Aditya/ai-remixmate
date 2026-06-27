"""
tests/test_core_modules.py — Smoke tests for all new core modules.

Tests cover:
  - Config system (all sections load as typed dataclasses)
  - License classification (all source types)
  - Camelot wheel + compatibility scoring (TrackMetadata)
  - Library manager (in-memory, no disk side effects)
  - Genre detection (synthetic audio)
  - DJ structure analysis (synthetic audio)
  - DJ transition planner
  - python_bridge check_compatibility (metadata-only, no downloads)

Run:
  python -m pytest tests/test_core_modules.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 44100


def _make_audio(
    duration: float = 5.0,
    bpm: float = 128.0,
    freq: float = 440.0,
    sr: int = SR,
) -> np.ndarray:
    """
    Synthetic audio: sine wave with a BPM-aligned amplitude envelope.
    The beat envelope gives librosa real transients to lock onto.
    """
    rng = np.random.default_rng(seed=42)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    beat_hz = bpm / 60.0
    # Hard transient every beat, exponential decay
    envelope = np.zeros_like(t)
    beat_samples = int(sr / beat_hz)
    for onset in range(0, len(t), beat_samples):
        decay_len = min(beat_samples, len(t) - onset)
        envelope[onset:onset + decay_len] += np.exp(
            -np.linspace(0, 6, decay_len)
        )
    audio = (np.sin(2 * np.pi * freq * t) * 0.7 + rng.normal(0, 0.05, len(t)))
    audio = (audio * envelope).astype(np.float32)
    audio /= max(np.abs(audio).max(), 1e-6)
    return audio


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_all_sections_are_dataclasses(self):
        """All config sections must resolve to typed dataclasses, not dicts."""
        from scripts.core.config import cfg
        import dataclasses

        for section_name in ("audio", "remix", "separation", "download",
                              "metadata", "dj", "library", "database",
                              "api", "logging"):
            section = getattr(cfg, section_name)
            assert dataclasses.is_dataclass(section), (
                f"cfg.{section_name} is a {type(section).__name__}, "
                f"expected a dataclass instance"
            )

    def test_audio_values(self):
        from scripts.core.config import cfg
        assert cfg.audio.sample_rate in (44100, 48000)
        assert cfg.audio.bit_depth in (16, 24)
        assert cfg.audio.target_lufs < 0

    def test_dj_values(self):
        from scripts.core.config import cfg
        assert cfg.dj.default_transition_bars in (8, 16, 32)
        assert cfg.dj.hp_filter_start_hz > cfg.dj.hp_filter_end_hz
        assert cfg.dj.bass_crossover_hz > 0

    def test_library_values(self):
        from scripts.core.config import cfg
        assert cfg.library.max_size_gb > 0
        assert isinstance(cfg.library.keep_raw_after_separation, bool)
        assert isinstance(cfg.library.prune_on_download, bool)

    def test_metadata_values(self):
        from scripts.core.config import cfg
        assert isinstance(cfg.metadata.getsongbpm_api_key, str)
        assert isinstance(cfg.metadata.lastfm_api_key, str)
        assert cfg.metadata.cache_ttl_days > 0


# ---------------------------------------------------------------------------
# 2. License system
# ---------------------------------------------------------------------------

class TestLicense:
    def test_youtube_tos(self):
        from scripts.core.license import classify_license, LicenseType
        lic = classify_license("youtube")
        assert lic.license_type == LicenseType.YOUTUBE_TOS
        assert lic.commercial_ok is False
        assert lic.derivatives_ok is False

    def test_jamendo_cc_by(self):
        from scripts.core.license import classify_license, LicenseType
        lic = classify_license(
            "jamendo",
            license_url="https://creativecommons.org/licenses/by/4.0/"
        )
        assert lic.license_type == LicenseType.CC_BY
        assert lic.commercial_ok is True
        assert lic.attribution_required is True

    def test_jamendo_cc_by_nc(self):
        from scripts.core.license import classify_license, LicenseType
        lic = classify_license(
            "jamendo",
            license_url="https://creativecommons.org/licenses/by-nc/4.0/"
        )
        assert lic.license_type == LicenseType.CC_BY_NC
        assert lic.commercial_ok is False

    def test_cc0_fully_free(self):
        from scripts.core.license import classify_license, LicenseType
        lic = classify_license(
            "jamendo",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/"
        )
        assert lic.license_type == LicenseType.CC0
        assert lic.commercial_ok is True
        assert lic.attribution_required is False

    def test_unknown_source_is_conservative(self):
        from scripts.core.license import classify_license, LicenseType
        lic = classify_license("unknown_source")
        assert lic.commercial_ok is False

    def test_youtube_cc_video(self):
        """YouTube videos explicitly marked CC-BY should get CC_BY, not YOUTUBE_TOS."""
        from scripts.core.license import classify_license, LicenseType
        lic = classify_license(
            "youtube",
            license_str="Creative Commons Attribution licence (reuse allowed)"
        )
        assert lic.license_type == LicenseType.CC_BY
        assert lic.commercial_ok is True

    def test_warning_empty_for_cc0(self):
        from scripts.core.license import classify_license, license_warning
        lic = classify_license("jamendo",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/")
        assert license_warning(lic, "Song") == ""

    def test_warning_nonempty_for_youtube(self):
        from scripts.core.license import classify_license, license_warning
        lic = classify_license("youtube")
        warn = license_warning(lic, "Song")
        assert len(warn) > 20
        assert "YouTube" in warn

    def test_parse_cc_url_variants(self):
        from scripts.core.license import parse_cc_url, LicenseType
        cases = [
            ("https://creativecommons.org/licenses/by/4.0/",       LicenseType.CC_BY),
            ("https://creativecommons.org/licenses/by-sa/3.0/",    LicenseType.CC_BY_SA),
            ("https://creativecommons.org/licenses/by-nc-nd/4.0/", LicenseType.CC_BY_NC_ND),
            ("https://creativecommons.org/publicdomain/zero/1.0/", LicenseType.CC0),
        ]
        for url, expected in cases:
            result = parse_cc_url(url)
            assert result == expected, f"Expected {expected} for {url}, got {result}"


# ---------------------------------------------------------------------------
# 3. Track metadata + Camelot wheel
# ---------------------------------------------------------------------------

class TestTrackMetadata:
    def test_camelot_lookup(self):
        from scripts.core.track_metadata import key_to_camelot
        assert key_to_camelot("C", "major") == "8B"
        assert key_to_camelot("A", "minor") == "8A"
        assert key_to_camelot("G", "major") == "9B"
        assert key_to_camelot("F#", "major") == "2B"
        # Enharmonic equivalents
        assert key_to_camelot("Db", "major") == "3B"
        assert key_to_camelot("C#", "major") == "3B"

    def test_camelot_distance_same(self):
        from scripts.core.track_metadata import camelot_distance
        assert camelot_distance("8B", "8B") == 0

    def test_camelot_distance_adjacent(self):
        from scripts.core.track_metadata import camelot_distance
        assert camelot_distance("8B", "9B") == 1
        assert camelot_distance("8B", "7B") == 1

    def test_camelot_distance_circular(self):
        """1B and 12B are adjacent on the wheel (distance = 1, not 11)."""
        from scripts.core.track_metadata import camelot_distance
        assert camelot_distance("1B", "12B") == 1

    def test_camelot_distance_different_mode(self):
        from scripts.core.track_metadata import camelot_distance
        # Major vs minor of very different keys = 999 (different modes)
        dist = camelot_distance("8B", "3A")
        assert dist == 999

    def test_compatibility_perfect_match(self):
        from scripts.core.track_metadata import TrackMetadata, MetadataClient
        client = MetadataClient()
        a = TrackMetadata(bpm=128.0, key="C", mode="major", camelot="8B", energy=0.8)
        b = TrackMetadata(bpm=128.0, key="C", mode="major", camelot="8B", energy=0.8)
        compat = client.compatibility_score(a, b)
        assert compat["key_score"] == 1.0
        assert compat["bpm_score"] == 1.0
        assert compat["compatible"] is True

    def test_compatibility_bpm_close(self):
        from scripts.core.track_metadata import TrackMetadata, MetadataClient
        client = MetadataClient()
        a = TrackMetadata(bpm=128.0, camelot="8B", energy=0.8)
        b = TrackMetadata(bpm=130.0, camelot="8B", energy=0.8)
        compat = client.compatibility_score(a, b)
        # 130/128 = 1.015 — within 2% threshold
        assert compat["bpm_score"] >= 0.75

    def test_compatibility_bpm_mismatch(self):
        from scripts.core.track_metadata import TrackMetadata, MetadataClient
        client = MetadataClient()
        a = TrackMetadata(bpm=128.0, camelot="8B", energy=0.8)
        b = TrackMetadata(bpm=175.0, camelot="8B", energy=0.8)
        compat = client.compatibility_score(a, b)
        assert compat["bpm_score"] < 0.5

    def test_trackmetadata_merge(self):
        from scripts.core.track_metadata import TrackMetadata
        a = TrackMetadata(title="Test", bpm=128.0)
        b = TrackMetadata(key="C", mode="major", genres=["house"])
        merged = a.merge(b)
        assert merged.title == "Test"
        assert merged.bpm == 128.0
        assert merged.key == "C"
        assert merged.genres == ["house"]

    def test_trackmetadata_merge_prefers_self(self):
        from scripts.core.track_metadata import TrackMetadata
        a = TrackMetadata(bpm=128.0)
        b = TrackMetadata(bpm=100.0)
        merged = a.merge(b)
        assert merged.bpm == 128.0  # a's value wins

    def test_is_complete(self):
        from scripts.core.track_metadata import TrackMetadata
        complete = TrackMetadata(bpm=128.0, key="C")
        incomplete = TrackMetadata(bpm=0.0, key="")
        assert complete.is_complete is True
        assert incomplete.is_complete is False

    def test_metadata_cache_roundtrip(self, tmp_path):
        from scripts.core.track_metadata import MetadataCache, TrackMetadata
        cache = MetadataCache(db_path=tmp_path / "test.db", ttl=3600)
        meta = TrackMetadata(title="Test", artist="Artist", bpm=128.0,
                             key="C", mode="major")
        cache.put("test artist", meta)
        retrieved = cache.get("test artist")
        assert retrieved is not None
        assert retrieved.title == "Test"
        assert retrieved.bpm == 128.0

    def test_metadata_cache_ttl_expiry(self, tmp_path):
        from scripts.core.track_metadata import MetadataCache, TrackMetadata
        cache = MetadataCache(db_path=tmp_path / "test.db", ttl=1)
        meta = TrackMetadata(title="Stale", bpm=100.0)
        cache.put("stale", meta)
        time.sleep(1.1)
        result = cache.get("stale")
        assert result is None  # expired


# ---------------------------------------------------------------------------
# 4. Library manager (no disk side effects via tmp_path)
# ---------------------------------------------------------------------------

class TestLibraryManager:
    def test_empty_library(self, tmp_path):
        from scripts.core.library import LibraryManager
        mgr = LibraryManager(library_dir=tmp_path / "lib")
        assert mgr.get_size_bytes() == 0
        assert mgr.get_size_gb() == 0.0
        assert mgr.list_songs() == []

    def test_has_song_false_when_missing(self, tmp_path):
        from scripts.core.library import LibraryManager
        mgr = LibraryManager(library_dir=tmp_path / "lib")
        assert mgr.has_song("nonexistent") is False

    def test_register_and_list(self, tmp_path):
        from scripts.core.library import LibraryManager
        lib = tmp_path / "lib"
        song = lib / "TestSong"
        song.mkdir(parents=True)
        (song / "full.wav").write_bytes(b"\x00" * 1024)

        mgr = LibraryManager(library_dir=lib)
        entry = mgr.register("TestSong", source="youtube")
        assert entry.name == "TestSong"
        assert entry.has_full_wav is True
        assert entry.size_bytes > 0

        songs = mgr.list_songs()
        assert any(s.name == "TestSong" for s in songs)

    def test_touch_updates_last_accessed(self, tmp_path):
        from scripts.core.library import LibraryManager
        lib = tmp_path / "lib"
        song = lib / "A"
        song.mkdir(parents=True)
        (song / "full.wav").write_bytes(b"\x00" * 512)

        mgr = LibraryManager(library_dir=lib)
        mgr.register("A")
        t1 = time.time()
        time.sleep(0.05)
        mgr.touch("A")
        entry = mgr.song_info("A")
        assert entry is not None
        assert entry.last_accessed >= t1

    def test_prune_raw_requires_all_stems(self, tmp_path):
        """prune_raw should do nothing if not all stems are present."""
        from scripts.core.library import LibraryManager
        lib = tmp_path / "lib"
        song = lib / "MySong"
        song.mkdir(parents=True)
        (song / "full.wav").write_bytes(b"\x00" * 1024)
        # Only 2 stems — not enough to prune
        (song / "vocals.wav").write_bytes(b"\x00" * 512)
        (song / "drums.wav").write_bytes(b"\x00" * 512)

        mgr = LibraryManager(library_dir=lib)
        result = mgr.prune_raw("MySong")
        assert result is False
        assert (song / "full.wav").exists()

    def test_prune_raw_succeeds_with_all_stems(self, tmp_path):
        """prune_raw should delete full.wav when all four stems exist."""
        from scripts.core.library import LibraryManager
        lib = tmp_path / "lib"
        song = lib / "FullSong"
        song.mkdir(parents=True)
        (song / "full.wav").write_bytes(b"\x00" * 4096)
        for stem in ("vocals", "drums", "bass", "other"):
            (song / f"{stem}.wav").write_bytes(b"\x00" * 1024)

        mgr = LibraryManager(library_dir=lib)
        mgr.register("FullSong")
        result = mgr.prune_raw("FullSong")
        assert result is True
        assert not (song / "full.wav").exists()

    def test_evict_lru_no_action_when_under_cap(self, tmp_path):
        from scripts.core.library import LibraryManager
        lib = tmp_path / "lib"
        song = lib / "TinySong"
        song.mkdir(parents=True)
        (song / "full.wav").write_bytes(b"\x00" * 100)

        mgr = LibraryManager(library_dir=lib, max_size_gb=100.0)
        evicted = mgr.evict_lru()
        assert evicted == []


# ---------------------------------------------------------------------------
# 5. Genre detection
# ---------------------------------------------------------------------------

class TestGenreDetection:
    def test_returns_valid_genre(self):
        from scripts.core.genre import detect_genre, GENRE_PRESETS
        audio = _make_audio(duration=10.0, bpm=128.0)
        result = detect_genre(audio, SR)
        assert result.genre in GENRE_PRESETS
        assert 0.0 <= result.confidence <= 1.0

    def test_runner_up_is_different(self):
        from scripts.core.genre import detect_genre
        audio = _make_audio(duration=10.0, bpm=128.0)
        result = detect_genre(audio, SR)
        if result.runner_up:
            assert result.runner_up != result.genre

    def test_preset_has_mix_params(self):
        from scripts.core.genre import detect_genre
        audio = _make_audio(duration=10.0, bpm=120.0)
        result = detect_genre(audio, SR)
        preset = result.preset
        assert preset.lufs_target < 0
        assert preset.vocal_hp_filter_hz > 0
        assert 0.0 <= preset.inst_sidechain_amount <= 1.0

    def test_get_preset_by_name(self):
        from scripts.core.genre import get_preset, GENRE_PRESETS
        for name in GENRE_PRESETS:
            p = get_preset(name)
            assert p.genre == name

    def test_legacy_preset_names(self):
        from scripts.core.genre import get_preset
        assert get_preset("radio").genre == "pop"
        assert get_preset("club").genre  == "house"

    def test_auto_preset_override(self):
        from scripts.core.genre import auto_preset
        audio = _make_audio(duration=5.0, bpm=120.0)
        preset = auto_preset(audio, SR, override="techno")
        assert preset.genre == "techno"

    def test_all_ten_genres_have_complete_presets(self):
        from scripts.core.genre import GENRE_PRESETS
        required_genres = {"house", "techno", "hiphop", "trap", "pop",
                           "rnb", "dnb", "ambient", "rock", "jazz"}
        assert set(GENRE_PRESETS.keys()) >= required_genres
        for name, preset in GENRE_PRESETS.items():
            assert preset.bpm_range[0] < preset.bpm_range[1], \
                f"{name}: bpm_range lo >= hi"
            assert preset.lufs_target < 0, f"{name}: lufs_target should be negative"
            assert 0.0 <= preset.inst_sidechain_amount <= 1.0, \
                f"{name}: sidechain out of range"


# ---------------------------------------------------------------------------
# 6. DJ engine — structure analysis
# ---------------------------------------------------------------------------

@pytest.mark.dj_analysis
class TestDJStructureAnalysis:
    def test_analyze_returns_structure(self):
        from scripts.core.dj_engine import _analyze_impl
        audio = _make_audio(duration=30.0, bpm=128.0)
        struct = _analyze_impl(audio, SR)
        assert struct.bpm > 0
        assert struct.total_bars > 0
        assert struct.duration > 0

    def test_beat_count_roughly_correct(self):
        from scripts.core.dj_engine import _analyze_impl
        bpm = 120.0
        duration = 20.0
        expected_beats = bpm / 60 * duration  # ~40
        audio = _make_audio(duration=duration, bpm=bpm)
        struct = _analyze_impl(audio, SR)
        # Allow ±40% tolerance (librosa can drift on synthetic audio)
        assert 0.6 * expected_beats <= len(struct.beats) <= 1.4 * expected_beats, \
            f"Expected ~{expected_beats:.0f} beats, got {len(struct.beats)}"

    def test_sections_are_labelled(self):
        from scripts.core.dj_engine import _analyze_impl
        audio = _make_audio(duration=30.0, bpm=120.0)
        struct = _analyze_impl(audio, SR)
        valid_types = {"intro", "verse", "chorus", "drop", "build", "break", "outro"}
        for sec in struct.sections:
            assert sec.type in valid_types, f"Unknown section type: {sec.type}"

    def test_section_times_are_ordered(self):
        from scripts.core.dj_engine import _analyze_impl
        audio = _make_audio(duration=30.0, bpm=120.0)
        struct = _analyze_impl(audio, SR)
        for i in range(1, len(struct.sections)):
            assert struct.sections[i].start_bar >= struct.sections[i - 1].start_bar

    def test_bar_start_time_in_bounds(self):
        from scripts.core.dj_engine import _analyze_impl
        audio = _make_audio(duration=20.0, bpm=120.0)
        struct = _analyze_impl(audio, SR)
        if struct.total_bars > 0:
            t = struct.bar_start_time(0)
            assert 0.0 <= t < struct.duration

    def test_phrase_boundary_alignment(self):
        from scripts.core.dj_engine import _analyze_impl
        audio = _make_audio(duration=30.0, bpm=120.0)
        struct = _analyze_impl(audio, SR)
        for phrase_size in (4, 8, 16):
            boundary = struct.phrase_boundary(5, phrase_size)
            assert boundary % phrase_size == 0, \
                f"Boundary {boundary} not aligned to phrase size {phrase_size}"


# ---------------------------------------------------------------------------
# 7. DJ engine — transition planner
# ---------------------------------------------------------------------------

class TestDJTransitionPlanner:
    def _make_structure(self, bpm: float, total_bars: int = 64):
        from scripts.core.dj_engine import SongStructure, Section
        bar_sec = 60.0 / bpm * 4
        bars = [(i * bar_sec, (i + 1) * bar_sec) for i in range(total_bars)]
        sections = [
            Section("intro",  0,  8,  bars[0][0],  bars[7][1],  0.3, 0.3),
            Section("verse",  8, 32,  bars[8][0],  bars[31][1], 0.6, 0.5),
            Section("chorus", 32, 48, bars[32][0], bars[47][1], 0.9, 0.8),
            Section("outro",  48, 64, bars[48][0], bars[63][1], 0.3, 0.3),
        ]
        return SongStructure(bpm=bpm, duration=total_bars * bar_sec,
                             bars=bars, sections=sections)

    def test_plan_returns_transition(self):
        from scripts.core.dj_engine import plan_transition
        a = self._make_structure(128.0)
        b = self._make_structure(128.0)
        plan = plan_transition(a, b)
        assert plan.transition_bars > 0
        assert plan.exit_bar_a >= 0
        assert plan.entry_bar_b >= 0

    def test_transition_bars_is_phrase_aligned(self):
        from scripts.core.dj_engine import plan_transition
        a = self._make_structure(128.0)
        b = self._make_structure(128.0)
        plan = plan_transition(a, b, phrase_size=8)
        assert plan.transition_bars % 8 == 0

    def test_exit_bar_in_outro(self):
        """With clear outro section, exit should land in it."""
        from scripts.core.dj_engine import plan_transition
        a = self._make_structure(128.0)
        b = self._make_structure(128.0)
        plan = plan_transition(a, b)
        assert plan.exit_bar_a >= 40, \
            f"Expected exit near outro (bar ≥40), got bar {plan.exit_bar_a}"

    def test_entry_bar_in_intro(self):
        """With clear intro section, entry should land at bar 0."""
        from scripts.core.dj_engine import plan_transition
        a = self._make_structure(128.0)
        b = self._make_structure(128.0)
        plan = plan_transition(a, b)
        assert plan.entry_bar_b == 0

    def test_bass_swap_at_midpoint(self):
        from scripts.core.dj_engine import plan_transition
        a = self._make_structure(128.0)
        b = self._make_structure(128.0)
        plan = plan_transition(a, b, transition_bars=16)
        assert plan.eq.bass_swap_bar == 8  # midpoint of 16

    def test_eq_plan_hp_start_greater_than_end(self):
        from scripts.core.dj_engine import plan_transition
        a = self._make_structure(128.0)
        b = self._make_structure(128.0)
        plan = plan_transition(a, b)
        assert plan.eq.hp_start_hz > plan.eq.hp_end_hz

    def test_tempo_shift_ratio_identity(self):
        """Same BPM → ratio should be 1.0."""
        from scripts.core.dj_engine import plan_transition
        a = self._make_structure(128.0)
        b = self._make_structure(128.0)
        plan = plan_transition(a, b)
        assert abs(plan.tempo_shift_ratio - 1.0) < 0.01

    def test_tempo_shift_ratio_different_bpm(self):
        from scripts.core.dj_engine import plan_transition
        a = self._make_structure(128.0)
        b = self._make_structure(140.0)
        plan = plan_transition(a, b)
        assert abs(plan.tempo_shift_ratio - 140.0 / 128.0) < 0.01


# ---------------------------------------------------------------------------
# 8. DJ engine — renderer (output shape + no NaN)
# ---------------------------------------------------------------------------

@pytest.mark.dj_analysis
class TestDJRenderer:
    def _make_structure(self, bpm: float, total_bars: int = 32):
        from scripts.core.dj_engine import SongStructure, Section
        bar_sec = 60.0 / bpm * 4
        bars = [(i * bar_sec, (i + 1) * bar_sec) for i in range(total_bars)]
        sections = [
            Section("intro",  0,  4,  bars[0][0], bars[3][1],  0.3, 0.3),
            Section("outro",  24, 32, bars[24][0], bars[31][1], 0.3, 0.3),
        ]
        return SongStructure(bpm=bpm, duration=total_bars * bar_sec,
                             bars=bars, sections=sections)

    def test_render_output_is_float32(self):
        from scripts.core.dj_engine import DJEngine, plan_transition
        sr = 22050
        bpm = 120.0
        dur = 20.0
        a = _make_audio(duration=dur, bpm=bpm, sr=sr)
        b = _make_audio(duration=dur, bpm=bpm, sr=sr)
        struct_a = self._make_structure(bpm)
        struct_b = self._make_structure(bpm)
        plan = plan_transition(struct_a, struct_b, transition_bars=8)
        engine = DJEngine(sr=sr)
        mix = engine.render(a, b, plan)
        assert mix.dtype == np.float32

    def test_render_no_nan_or_inf(self):
        from scripts.core.dj_engine import DJEngine, plan_transition
        sr = 22050
        bpm = 120.0
        dur = 20.0
        a = _make_audio(duration=dur, bpm=bpm, sr=sr)
        b = _make_audio(duration=dur, bpm=bpm, sr=sr)
        struct_a = self._make_structure(bpm)
        struct_b = self._make_structure(bpm)
        plan = plan_transition(struct_a, struct_b, transition_bars=8)
        engine = DJEngine(sr=sr)
        mix = engine.render(a, b, plan)
        assert not np.isnan(mix).any(), "Output contains NaN"
        assert not np.isinf(mix).any(), "Output contains Inf"

    def test_render_output_clipped(self):
        """Soft limiter should keep output within [-1, 1] range."""
        from scripts.core.dj_engine import DJEngine, plan_transition
        sr = 22050
        bpm = 120.0
        dur = 20.0
        a = _make_audio(duration=dur, bpm=bpm, sr=sr)
        b = _make_audio(duration=dur, bpm=bpm, sr=sr)
        struct_a = self._make_structure(bpm)
        struct_b = self._make_structure(bpm)
        plan = plan_transition(struct_a, struct_b, transition_bars=8)
        engine = DJEngine(sr=sr)
        mix = engine.render(a, b, plan)
        assert np.abs(mix).max() <= 1.0 + 1e-5, \
            f"Output exceeds [-1,1]: max abs = {np.abs(mix).max()}"

    def test_render_transition_only_shorter_than_full(self):
        """full_output=False should produce less audio than full_output=True."""
        from scripts.core.dj_engine import DJEngine, plan_transition
        sr = 22050
        bpm = 120.0
        dur = 20.0
        a = _make_audio(duration=dur, bpm=bpm, sr=sr)
        b = _make_audio(duration=dur, bpm=bpm, sr=sr)
        struct_a = self._make_structure(bpm)
        struct_b = self._make_structure(bpm)
        plan = plan_transition(struct_a, struct_b, transition_bars=8)
        engine = DJEngine(sr=sr)
        full = engine.render(a, b, plan, full_output=True)
        trans_only = engine.render(a, b, plan, full_output=False)
        assert len(trans_only) < len(full)


# ---------------------------------------------------------------------------
# Shared helper for Gap 1A/1B tests — mirrors TestDJRenderer._make_structure
# but lives at module scope so both test classes can use it without inheriting
# the @pytest.mark.dj_analysis marker unnecessarily.
# ---------------------------------------------------------------------------

def _make_structure_for_render(bpm: float, total_bars: int = 32):
    """Build a minimal SongStructure with labelled sections for render tests."""
    from scripts.core.dj_engine import SongStructure, Section
    bar_sec = 60.0 / bpm * 4
    bars = [(i * bar_sec, (i + 1) * bar_sec) for i in range(total_bars)]
    sections = [
        Section("intro",  0,  8,  bars[0][0],   bars[7][1],  0.3, 0.3),
        Section("verse",  8,  16, bars[8][0],   bars[15][1], 0.6, 0.5),
        Section("chorus", 16, 24, bars[16][0],  bars[23][1], 0.9, 0.7),
        Section("outro",  24, 32, bars[24][0],  bars[31][1], 0.3, 0.3),
    ]
    return SongStructure(bpm=bpm, duration=total_bars * bar_sec,
                         bars=bars, sections=sections)


def _write_wav(path, audio: np.ndarray, sr: int) -> None:
    """Write a mono float32 array as a WAV file."""
    import soundfile as sf
    sf.write(str(path), audio, sr, subtype="FLOAT")


# ---------------------------------------------------------------------------
# Gap 1A — Per-stem LUFS normalization
# ---------------------------------------------------------------------------

@pytest.mark.dj_analysis
class TestStemLufsNormalization:
    """
    Tests for Gap 1A: normalize_stems_to_target() is called before
    the crossfade loop in render_stem_blend(), preventing loud stems
    from overpowering softer ones.
    """

    def _make_stem_dirs(self, tmp_path, sr: int, bpm: float, dur: float,
                        gain_a: float = 1.0, gain_b: float = 1.0):
        """
        Create two stem directories (song A and song B) with synthetic audio.

        gain_a / gain_b let us inject a deliberate level imbalance to verify
        that normalization corrects it.  All four stems share the same content
        so their similarity score will be high (> 0.75), forcing the extended
        blend path which exercises the most code.
        """
        dir_a = tmp_path / "stems_a"
        dir_b = tmp_path / "stems_b"
        dir_a.mkdir()
        dir_b.mkdir()

        base = _make_audio(duration=dur, bpm=bpm, sr=sr)
        for stem in ("vocals", "drums", "bass", "other"):
            _write_wav(dir_a / f"{stem}.wav", base * gain_a, sr)
            _write_wav(dir_b / f"{stem}.wav", base * gain_b, sr)

        return dir_a, dir_b

    def test_normalization_prevents_clipping_with_loud_stems(self, tmp_path):
        """
        Stems at gain=0.9 (just below 0 dBFS) should not clip after mixing
        four of them together. Without per-stem LUFS normalization the sum of
        four near-full-scale stems would clip hard.
        """
        from scripts.core.dj_engine import DJEngine
        from scripts.core.dj_analysis import plan_transition

        sr, bpm, dur = 22050, 120.0, 20.0
        dir_a, dir_b = self._make_stem_dirs(tmp_path, sr, bpm, dur,
                                             gain_a=0.9, gain_b=0.9)
        track_a = _make_audio(duration=dur, bpm=bpm, sr=sr)
        track_b = _make_audio(duration=dur, bpm=bpm, sr=sr)
        struct_a = _make_structure_for_render(bpm)
        struct_b = _make_structure_for_render(bpm)
        plan = plan_transition(struct_a, struct_b, transition_bars=8)

        engine = DJEngine(sr=sr)
        mix = engine.render_stem_blend(
            track_a, track_b, plan,
            stems_dir_a=dir_a, stems_dir_b=dir_b,
            full_output=False,
        )

        assert mix.dtype == np.float32
        assert not np.isnan(mix).any(), "Output contains NaN"
        assert not np.isinf(mix).any(), "Output contains Inf"
        assert np.abs(mix).max() <= 1.0 + 1e-4, \
            f"Output clips: max abs = {np.abs(mix).max():.4f}"

    def test_level_imbalance_is_corrected(self, tmp_path):
        """
        A Song B stem that is 10× louder than Song A should produce a mix
        whose first and second halves are within 12 dB of each other —
        rather than the raw +20 dB jump that would happen without normalization.
        """
        from scripts.core.dj_engine import DJEngine
        from scripts.core.dj_analysis import plan_transition

        sr, bpm, dur = 22050, 120.0, 20.0
        dir_a, dir_b = self._make_stem_dirs(tmp_path, sr, bpm, dur,
                                             gain_a=0.1, gain_b=1.0)
        track_a = _make_audio(duration=dur, bpm=bpm, sr=sr)
        track_b = _make_audio(duration=dur, bpm=bpm, sr=sr)
        struct_a = _make_structure_for_render(bpm)
        struct_b = _make_structure_for_render(bpm)
        plan = plan_transition(struct_a, struct_b, transition_bars=8)

        engine = DJEngine(sr=sr)
        mix = engine.render_stem_blend(
            track_a, track_b, plan,
            stems_dir_a=dir_a, stems_dir_b=dir_b,
            full_output=False,
        )

        n = len(mix)
        rms_first  = float(np.sqrt(np.mean(mix[:n // 2] ** 2)) + 1e-10)
        rms_second = float(np.sqrt(np.mean(mix[n // 2:] ** 2)) + 1e-10)
        level_diff_db = 20 * np.log10(rms_second / rms_first)

        assert abs(level_diff_db) <= 12.0, (
            f"Level jump too large after normalization: {level_diff_db:+.1f} dB "
            f"(first half RMS {rms_first:.4f}, second half RMS {rms_second:.4f}). "
            f"Per-stem LUFS normalization may not be applied."
        )


# ---------------------------------------------------------------------------
# Gap 1B — Bass stem swap-point handoff
# ---------------------------------------------------------------------------

@pytest.mark.dj_analysis
class TestBassSwap:
    """
    Tests for Gap 1B: the bass stem uses a hard swap-point envelope instead
    of the generic similarity-based crossfade. Song A's bass exits by the
    swap sample; Song B's bass enters clean from that same point.
    """

    def test_bass_exits_before_midpoint(self, tmp_path):
        """
        With only a bass stem populated (all other stems silent), Song A's
        contribution should be near-zero in the second half of the transition
        window, confirming that the swap envelope cuts A's bass early.
        """
        from scripts.core.dj_engine import DJEngine
        from scripts.core.dj_analysis import plan_transition

        sr, bpm, dur = 22050, 120.0, 20.0
        dir_a = tmp_path / "stems_a"
        dir_b = tmp_path / "stems_b"
        dir_a.mkdir(); dir_b.mkdir()

        silence = np.zeros(int(sr * dur), dtype=np.float32)

        # Song A: only bass has content; everything else is silence
        bass_a = _make_audio(duration=dur, bpm=bpm, freq=80.0, sr=sr) * 0.5
        _write_wav(dir_a / "bass.wav",   bass_a,  sr)
        _write_wav(dir_a / "vocals.wav", silence, sr)
        _write_wav(dir_a / "drums.wav",  silence, sr)
        _write_wav(dir_a / "other.wav",  silence, sr)

        # Song B: all stems silent (so B's fade-in contributes nothing)
        _write_wav(dir_b / "bass.wav",   silence, sr)
        _write_wav(dir_b / "vocals.wav", silence, sr)
        _write_wav(dir_b / "drums.wav",  silence, sr)
        _write_wav(dir_b / "other.wav",  silence, sr)

        track_a = bass_a.copy()
        track_b = silence.copy()

        struct_a = _make_structure_for_render(bpm)
        struct_b = _make_structure_for_render(bpm)
        plan = plan_transition(struct_a, struct_b, transition_bars=8)

        engine = DJEngine(sr=sr)
        mix = engine.render_stem_blend(
            track_a, track_b, plan,
            stems_dir_a=dir_a, stems_dir_b=dir_b,
            full_output=False,
        )

        n = len(mix)
        # First quarter should have energy (A's bass is playing)
        rms_first_quarter = float(np.sqrt(np.mean(mix[:n // 4] ** 2)))
        # Final quarter should be near-silent (A's bass has been swapped out)
        rms_last_quarter  = float(np.sqrt(np.mean(mix[3 * n // 4:] ** 2)))

        assert rms_first_quarter > 1e-4, \
            "Expected Song A bass energy in the first quarter of the transition"
        assert rms_last_quarter < rms_first_quarter * 0.1, (
            f"Song A bass should be near-silent in the final quarter after swap. "
            f"First-quarter RMS: {rms_first_quarter:.5f}, "
            f"last-quarter RMS: {rms_last_quarter:.5f}"
        )


@pytest.mark.dj_analysis
class TestPitchShift:
    """Tests for pitch_shift_audio() and TransitionPlan.suggested_pitch_shift."""

    def test_pitch_shift_audio_semitone_up(self):
        """Shifting a 440 Hz sine by +12 semitones should peak near 880 Hz."""
        from scripts.core.key_detection import pitch_shift_audio

        sr = 22050
        duration = 1.0
        freq = 440.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

        shifted = pitch_shift_audio(audio, sr, semitones=12.0)

        freqs = np.fft.rfftfreq(len(shifted), d=1.0 / sr)
        magnitudes = np.abs(np.fft.rfft(shifted))
        peak_freq = freqs[np.argmax(magnitudes)]

        assert abs(peak_freq - 880.0) < 20.0, (
            f"Expected peak near 880 Hz after +12 semitones, got {peak_freq:.1f} Hz"
        )

    def test_transition_plan_has_pitch_shift(self):
        """plan_transition() should populate suggested_pitch_shift as a float."""
        from scripts.core.dj_analysis import plan_transition

        struct_a = _make_structure_for_render(120.0)
        struct_b = _make_structure_for_render(120.0)
        struct_a.camelot = "8B"   # C major
        struct_b.camelot = "10B"  # D major

        plan = plan_transition(struct_a, struct_b, transition_bars=8)

        assert isinstance(plan.suggested_pitch_shift, float)

    def test_pitch_shift_zero_when_same_key(self):
        """Same Camelot key for both songs should produce suggested_pitch_shift == 0.0."""
        from scripts.core.dj_analysis import plan_transition

        struct_a = _make_structure_for_render(120.0)
        struct_b = _make_structure_for_render(120.0)
        struct_a.camelot = "8B"
        struct_b.camelot = "8B"

        plan = plan_transition(struct_a, struct_b, transition_bars=8)

        assert plan.suggested_pitch_shift == 0.0
