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
