"""
tests/test_setlist_planner.py — Setlist planner coverage (was: zero tests).

Covers:
  - _arc_target_energy(): all 4 arc models, boundary conditions, ordering
  - transition_cost(): known BPM/energy/harmonic inputs, unit-range invariant
  - MarkovTransitionModel: train/score/save/load, edge cases
  - parse_exportify_csv(): valid CSV, BOM, bytes, empty, bad rows, Camelot resolution
  - SetlistPlanner.optimize(): ordering, arc selection, start_track, Markov blend
  - SetlistPlanner.export_csv(): file written, correct columns, positions start at 1
  - SetlistPlanner.summary(): track count, cost, modulation breakdown, blend counts
  - _SPOTIFY_KEY_CAMELOT: completeness and internal consistency
"""

from __future__ import annotations

import csv
import json
import random

import pytest

from scripts.core.setlist_planner import (
    EnergyArc,
    MarkovTransitionModel,
    SetlistPlanner,
    TrackNode,
    TransitionScore,
    _arc_target_energy,
    _SPOTIFY_KEY_CAMELOT,
    parse_exportify_csv,
    transition_cost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _track(
    name: str = "Track",
    artist: str = "Artist",
    bpm: float = 128.0,
    energy: float = 0.70,
    camelot: str = "8B",
    duration_ms: int = 240_000,
) -> TrackNode:
    return TrackNode(
        name=name,
        artist=artist,
        bpm=bpm,
        energy=energy,
        camelot=camelot,
        duration_ms=duration_ms,
    )


def _pool(n: int = 8) -> list[TrackNode]:
    """Generate a diverse pool of tracks for optimizer tests."""
    camelots = ["8B", "9B", "10B", "11B", "12B", "1B", "2B", "3B"]
    bpms     = [120, 124, 126, 128, 130, 128, 126, 124]
    energies = [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.75]
    return [
        TrackNode(
            name=f"Track {i + 1}",
            artist=f"Artist {i + 1}",
            bpm=float(bpms[i % len(bpms)]),
            energy=energies[i % len(energies)],
            camelot=camelots[i % len(camelots)],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Exportify CSV helpers
# ---------------------------------------------------------------------------

_EXPORTIFY_HEADER = (
    "Spotify ID,Track Name,Artist Name(s),Album Name,"
    "Duration (ms),Popularity,Danceability,Energy,Key,Loudness,Mode,"
    "Speechiness,Acousticness,Instrumentalness,Liveness,Valence,Tempo,Genres"
)


def _csv_row(
    name="Test Song", artist="Test Artist", album="Test Album",
    duration_ms=240000, popularity=75,
    danceability=0.80, energy=0.75, key=0, loudness=-8.5, mode=1,
    speechiness=0.04, acousticness=0.01, instrumentalness=0.50,
    liveness=0.10, valence=0.60, tempo=128.0,
    genres="techno", spotify_id="abc123",
) -> str:
    return (
        f'"{spotify_id}","{name}","{artist}","{album}",'
        f"{duration_ms},{popularity},{danceability},{energy},{key},{loudness},{mode},"
        f"{speechiness},{acousticness},{instrumentalness},{liveness},{valence},{tempo},"
        f'"{genres}"'
    )


def _csv(*rows: str) -> str:
    return "\n".join([_EXPORTIFY_HEADER] + list(rows))


# ---------------------------------------------------------------------------
# 1. _arc_target_energy
# ---------------------------------------------------------------------------

class TestArcTargetEnergy:
    """All arc models must return [0, 1] and respect their intended shape."""

    @pytest.mark.parametrize("arc", list(EnergyArc))
    def test_returns_float_in_unit_range(self, arc):
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = _arc_target_energy(t, arc)
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0, f"Arc {arc} t={t} → {val} (outside [0,1])"

    def test_ramp_up_is_monotonic(self):
        ts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        vals = [_arc_target_energy(t, EnergyArc.RAMP_UP) for t in ts]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1], f"RAMP_UP not monotonic at t={ts[i]}"

    def test_ramp_up_ends_higher_than_start(self):
        assert _arc_target_energy(0.0, EnergyArc.RAMP_UP) < _arc_target_energy(1.0, EnergyArc.RAMP_UP)

    def test_ramp_down_ends_lower_than_start(self):
        assert _arc_target_energy(0.0, EnergyArc.RAMP_DOWN) > _arc_target_energy(1.0, EnergyArc.RAMP_DOWN)

    def test_mountain_peak_near_70_percent(self):
        peak = _arc_target_energy(0.70, EnergyArc.MOUNTAIN)
        assert _arc_target_energy(0.20, EnergyArc.MOUNTAIN) < peak
        assert _arc_target_energy(1.00, EnergyArc.MOUNTAIN) < peak

    def test_wave_oscillates(self):
        vals = [_arc_target_energy(i / 10, EnergyArc.WAVE) for i in range(11)]
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        signs = {1 if d >= 0 else -1 for d in diffs}
        assert len(signs) > 1, "WAVE arc does not oscillate"

    def test_boundary_values_no_crash(self):
        for arc in EnergyArc:
            _arc_target_energy(0.0, arc)
            _arc_target_energy(1.0, arc)


# ---------------------------------------------------------------------------
# 2. transition_cost
# ---------------------------------------------------------------------------

class TestTransitionCost:
    """transition_cost(): 50% harmonic, 30% BPM, 20% energy."""

    def test_returns_transition_score(self):
        assert isinstance(transition_cost(_track(), _track()), TransitionScore)

    def test_identical_tracks_zero_bpm_harmonic_cost(self):
        a = _track(bpm=128.0, camelot="8B")
        b = _track(bpm=128.0, camelot="8B")
        s = transition_cost(a, b, arc_position=0.5)
        assert s.bpm_cost == pytest.approx(0.0)
        assert s.harmonic_cost == pytest.approx(0.0, abs=0.01)

    def test_large_bpm_gap_caps_bpm_cost_at_one(self):
        s = transition_cost(_track(bpm=90.0), _track(bpm=140.0), arc_position=0.5)
        assert s.bpm_cost == pytest.approx(1.0, abs=0.01)

    def test_total_cost_in_unit_range(self):
        rng = random.Random(42)
        codes = ["1A", "6B", "12B", "3A", "8A"]
        for _ in range(25):
            a = _track(bpm=rng.uniform(80, 160), energy=rng.random(), camelot=rng.choice(codes))
            b = _track(bpm=rng.uniform(80, 160), energy=rng.random(), camelot=rng.choice(codes))
            s = transition_cost(a, b, arc_position=rng.random())
            assert 0.0 <= s.total_cost <= 1.0, f"total_cost={s.total_cost}"

    def test_missing_camelot_uninformed_prior(self):
        s = transition_cost(_track(camelot=""), _track(camelot=""), arc_position=0.5)
        assert s.harmonic_cost == pytest.approx(0.5)

    def test_bpm_delta_is_absolute(self):
        s = transition_cost(_track(bpm=140.0), _track(bpm=120.0))
        assert s.bpm_delta == pytest.approx(20.0, abs=0.1)

    def test_pure_harmonic_weight(self):
        """w_harmonic=1, others 0 → total_cost == harmonic_cost."""
        a = _track(camelot="8B")
        b = _track(camelot="8B")
        s = transition_cost(a, b, w_harmonic=1.0, w_bpm=0.0, w_energy=0.0, arc_position=0.5)
        assert s.total_cost == pytest.approx(s.harmonic_cost, abs=0.001)


# ---------------------------------------------------------------------------
# 3. MarkovTransitionModel
# ---------------------------------------------------------------------------

class TestMarkovTransitionModel:

    @staticmethod
    def _ab_setlist() -> list[TrackNode]:
        """A→B→A→B — trains A→B as dominant transition."""
        a = _track(name="A", energy=0.5, camelot="8B")
        b = _track(name="B", energy=0.7, camelot="9B")
        return [a, b, a, b]

    def test_uninformed_prior_is_half(self):
        model = MarkovTransitionModel()
        assert model.score(_track(camelot="8B"), _track(camelot="9B")) == pytest.approx(0.5)

    def test_train_raises_observed_score(self):
        model = MarkovTransitionModel()
        model.train(self._ab_setlist())
        a = _track(name="A", energy=0.5, camelot="8B")
        b = _track(name="B", energy=0.7, camelot="9B")
        assert model.score(a, b) > 0.5

    def test_train_empty_no_crash(self):
        MarkovTransitionModel().train([])

    def test_train_single_track_no_crash(self):
        MarkovTransitionModel().train([_track()])

    def test_score_always_in_unit_range(self):
        model = MarkovTransitionModel()
        model.train(_pool(8))
        for a in _pool(4):
            for b in _pool(4):
                s = model.score(a, b)
                assert 0.0 <= s <= 1.0, f"score={s} out of [0,1]"

    def test_save_load_roundtrip(self, tmp_path):
        model = MarkovTransitionModel()
        model.train(self._ab_setlist())
        path = tmp_path / "markov.json"
        model.save(path)

        data = json.loads(path.read_text())
        assert "counts" in data and "total" in data

        model2 = MarkovTransitionModel()
        model2.load(path)
        a = _track(name="A", energy=0.5, camelot="8B")
        b = _track(name="B", energy=0.7, camelot="9B")
        assert model.score(a, b) == pytest.approx(model2.score(a, b))

    def test_save_counts_are_positive_ints(self, tmp_path):
        model = MarkovTransitionModel()
        model.train(_pool(5))
        path = tmp_path / "model.json"
        model.save(path)
        data = json.loads(path.read_text())
        for _, transitions in data["counts"].items():
            for _, count in transitions.items():
                assert isinstance(count, int) and count > 0


# ---------------------------------------------------------------------------
# 4. parse_exportify_csv
# ---------------------------------------------------------------------------

class TestParseExportifyCsv:

    def test_parses_basic_row(self):
        tracks = parse_exportify_csv(_csv(_csv_row()))
        assert len(tracks) == 1
        t = tracks[0]
        assert t.name == "Test Song"
        assert t.artist == "Test Artist"
        assert t.bpm == pytest.approx(128.0)
        assert t.energy == pytest.approx(0.75)

    def test_resolves_camelot_c_major(self):
        tracks = parse_exportify_csv(_csv(_csv_row(key=0, mode=1)))
        assert tracks[0].camelot == "8B"

    def test_all_24_spotify_keys_resolve(self):
        for (key, mode), expected in _SPOTIFY_KEY_CAMELOT.items():
            tracks = parse_exportify_csv(_csv(_csv_row(key=key, mode=mode)))
            assert len(tracks) == 1
            assert tracks[0].camelot == expected, (
                f"Key={key} Mode={mode}: expected {expected}, got {tracks[0].camelot!r}"
            )

    def test_multiple_rows(self):
        rows = [_csv_row(name=f"Track {i}") for i in range(5)]
        tracks = parse_exportify_csv(_csv(*rows))
        assert len(tracks) == 5

    def test_skips_empty_name_rows(self):
        tracks = parse_exportify_csv(_csv(_csv_row(), ",,,,,,,,,,,,,,,,"))
        assert len(tracks) == 1

    def test_strips_utf8_bom_bytes(self):
        content = ("﻿" + _csv(_csv_row())).encode("utf-8")
        tracks = parse_exportify_csv(content)
        assert len(tracks) == 1
        assert tracks[0].name == "Test Song"

    def test_bytes_input(self):
        tracks = parse_exportify_csv(_csv(_csv_row()).encode("utf-8"))
        assert len(tracks) == 1

    def test_empty_csv_returns_empty_list(self):
        assert parse_exportify_csv(_EXPORTIFY_HEADER + "\n") == []

    def test_genres_split_on_comma(self):
        tracks = parse_exportify_csv(_csv(_csv_row(genres="techno, minimal, dark")))
        assert tracks[0].genres == ["techno", "minimal", "dark"]

    def test_duration_ms_parsed(self):
        tracks = parse_exportify_csv(_csv(_csv_row(duration_ms=200_000)))
        assert tracks[0].duration_ms == 200_000

    def test_popularity_parsed(self):
        tracks = parse_exportify_csv(_csv(_csv_row(popularity=88)))
        assert tracks[0].popularity == 88

    def test_bad_float_does_not_crash(self):
        """Corrupt numeric field must not crash the parser."""
        bad_row = _csv_row().replace(",0.80,", ",NOT_A_FLOAT,")
        tracks = parse_exportify_csv(_csv(bad_row))
        assert len(tracks) == 1  # still parsed, bad field falls back to default


# ---------------------------------------------------------------------------
# 5. SetlistPlanner.optimize
# ---------------------------------------------------------------------------

class TestSetlistPlannerOptimize:

    def test_empty_returns_empty(self):
        assert SetlistPlanner().optimize([]) == []

    def test_single_track(self):
        result = SetlistPlanner().optimize([_track()])
        assert len(result) == 1
        assert result[0]["transition_to_next"] is None

    def test_returns_all_tracks(self):
        pool = _pool(8)
        assert len(SetlistPlanner().optimize(pool)) == len(pool)

    def test_all_names_present(self):
        pool = _pool(8)
        result = SetlistPlanner().optimize(pool)
        assert {r["name"] for r in result} == {t.name for t in pool}

    def test_positions_are_sequential(self):
        pool = _pool(6)
        result = SetlistPlanner().optimize(pool)
        assert [r["position"] for r in result] == list(range(len(pool)))

    def test_start_track_is_first(self):
        pool = _pool(8)
        target = pool[4].name
        result = SetlistPlanner().optimize(pool, start_track=target)
        assert result[0]["name"] == target

    def test_unknown_start_track_fallback(self):
        pool = _pool(5)
        result = SetlistPlanner().optimize(pool, start_track="NONEXISTENT")
        assert len(result) == len(pool)

    @pytest.mark.parametrize("arc", list(EnergyArc))
    def test_all_arcs_complete(self, arc):
        assert len(SetlistPlanner().optimize(_pool(8), arc=arc)) == 8

    def test_transition_fields_complete(self):
        result = SetlistPlanner().optimize(_pool(4))
        for r in result[:-1]:
            t = r["transition_to_next"]
            assert t is not None
            for field in ("total_cost", "harmonic_cost", "bpm_delta", "safe_to_blend",
                          "next_track", "modulation_type"):
                assert field in t, f"Missing field {field!r}"

    def test_with_markov_model(self):
        pool = _pool(8)
        markov = MarkovTransitionModel()
        markov.train(pool)
        result = SetlistPlanner(markov_model=markov).optimize(pool, markov_weight=0.2)
        assert len(result) == len(pool)

    def test_cumulative_cost_non_negative(self):
        for r in SetlistPlanner().optimize(_pool(6)):
            assert r["cumulative_cost"] >= 0.0

    def test_ramp_up_anchors_lowest_energy(self):
        pool = _pool(6)
        pool[3] = TrackNode(name="LowEnergy", bpm=120.0, energy=0.02, camelot="1A")
        result = SetlistPlanner().optimize(pool, arc=EnergyArc.RAMP_UP)
        assert result[0]["name"] == "LowEnergy"

    def test_ramp_down_anchors_highest_energy(self):
        pool = _pool(6)
        pool[1] = TrackNode(name="HighEnergy", bpm=132.0, energy=0.99, camelot="8B")
        result = SetlistPlanner().optimize(pool, arc=EnergyArc.RAMP_DOWN)
        assert result[0]["name"] == "HighEnergy"

    def test_arc_target_energy_in_results(self):
        for r in SetlistPlanner().optimize(_pool(4)):
            assert 0.0 <= r["arc_target_energy"] <= 1.0


# ---------------------------------------------------------------------------
# 6. SetlistPlanner.export_csv
# ---------------------------------------------------------------------------

class TestSetlistPlannerExportCsv:

    def test_creates_file(self, tmp_path):
        result = SetlistPlanner().optimize(_pool(4))
        out = tmp_path / "setlist.csv"
        SetlistPlanner().export_csv(result, out)
        assert out.exists()

    def test_row_count_matches_tracks(self, tmp_path):
        pool = _pool(5)
        result = SetlistPlanner().optimize(pool)
        out = tmp_path / "set.csv"
        SetlistPlanner().export_csv(result, out)
        with open(out, newline="") as f:
            assert len(list(csv.DictReader(f))) == len(pool)

    def test_required_columns_present(self, tmp_path):
        result = SetlistPlanner().optimize(_pool(3))
        out = tmp_path / "set.csv"
        SetlistPlanner().export_csv(result, out)
        with open(out, newline="") as f:
            fieldnames = set(csv.DictReader(f).fieldnames or [])
        required = {"position", "artist", "name", "camelot", "bpm", "energy_level", "total_cost"}
        assert required.issubset(fieldnames)

    def test_positions_start_at_one(self, tmp_path):
        result = SetlistPlanner().optimize(_pool(3))
        out = tmp_path / "set.csv"
        SetlistPlanner().export_csv(result, out)
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["position"] == "1"

    def test_empty_results_no_crash(self, tmp_path):
        out = tmp_path / "empty.csv"
        SetlistPlanner().export_csv([], out)
        assert not out.exists()  # no file written for empty input


# ---------------------------------------------------------------------------
# 7. SetlistPlanner.summary
# ---------------------------------------------------------------------------

class TestSetlistPlannerSummary:

    def test_empty_returns_empty_dict(self):
        assert SetlistPlanner().summary([]) == {}

    def test_total_tracks_correct(self):
        result = SetlistPlanner().optimize(_pool(6))
        assert SetlistPlanner().summary(result)["total_tracks"] == 6

    def test_bpm_range_min_max(self):
        pool = _pool(6)
        result = SetlistPlanner().optimize(pool)
        summary = SetlistPlanner().summary(result)
        bpm_vals = [r["bpm"] for r in result if r.get("bpm")]
        assert summary["bpm_range"] == [round(min(bpm_vals), 1), round(max(bpm_vals), 1)]

    def test_energy_range_ordered(self):
        result = SetlistPlanner().optimize(_pool(6))
        er = SetlistPlanner().summary(result)["energy_range"]
        assert len(er) == 2 and er[0] <= er[1]

    def test_safe_plus_hard_equals_n_minus_one(self):
        pool = _pool(6)
        result = SetlistPlanner().optimize(pool)
        summary = SetlistPlanner().summary(result)
        assert summary["safe_blend_count"] + summary["hard_cut_count"] == len(pool) - 1

    def test_modulation_breakdown_is_dict(self):
        result = SetlistPlanner().optimize(_pool(5))
        assert isinstance(SetlistPlanner().summary(result)["modulation_breakdown"], dict)

    def test_total_cost_non_negative(self):
        result = SetlistPlanner().optimize(_pool(4))
        assert SetlistPlanner().summary(result)["total_cost"] >= 0.0


# ---------------------------------------------------------------------------
# 8. _SPOTIFY_KEY_CAMELOT mapping integrity
# ---------------------------------------------------------------------------

class TestSpotifyKeyCamelotMapping:

    def test_has_24_entries(self):
        assert len(_SPOTIFY_KEY_CAMELOT) == 24

    def test_no_duplicate_camelot_codes(self):
        codes = list(_SPOTIFY_KEY_CAMELOT.values())
        assert len(codes) == len(set(codes)), "Duplicate Camelot codes in _SPOTIFY_KEY_CAMELOT"

    def test_all_codes_valid(self):
        from scripts.core.key_detection import CAMELOT
        valid = set(CAMELOT.values())
        for (key, mode), code in _SPOTIFY_KEY_CAMELOT.items():
            assert code in valid, f"Key={key} Mode={mode} → {code!r} is not a valid Camelot code"

    def test_keys_in_range_0_to_11(self):
        for key, mode in _SPOTIFY_KEY_CAMELOT:
            assert 0 <= key <= 11
            assert mode in (0, 1)
