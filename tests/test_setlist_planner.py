"""tests/test_setlist_planner.py — Unit tests for SetlistPlanner."""
from __future__ import annotations
import pytest


def _make_track(name, bpm=128.0, camelot="8A", energy=0.5):
    from scripts.core.setlist_planner import TrackNode
    return TrackNode(name=name, bpm=bpm, camelot=camelot, energy=energy)


class TestTrackNode:
    def test_energy_level_low(self):
        t = _make_track("a", energy=0.1)
        assert t.energy_level == 1

    def test_energy_level_high(self):
        t = _make_track("a", energy=0.95)
        assert t.energy_level == 10

    def test_energy_level_midrange(self):
        t = _make_track("a", energy=0.5)
        assert 4 <= t.energy_level <= 6

    def test_to_dict_has_required_keys(self):
        d = _make_track("a").to_dict()
        for key in ("name", "bpm", "energy", "camelot"):
            assert key in d


class TestTransitionCost:
    def test_same_track_zero_bpm_cost(self):
        from scripts.core.setlist_planner import transition_cost
        t = _make_track("a", bpm=128.0, camelot="8A")
        cost = transition_cost(t, t)
        assert cost.bpm_cost == pytest.approx(0.0)

    def test_large_bpm_delta_high_cost(self):
        from scripts.core.setlist_planner import transition_cost
        a = _make_track("a", bpm=80.0)
        b = _make_track("b", bpm=160.0)
        cost = transition_cost(a, b)
        assert cost.bpm_cost > 0.5

    def test_adjacent_camelot_lower_harmonic_cost_than_far(self):
        from scripts.core.setlist_planner import transition_cost
        a = _make_track("a", camelot="8A")
        b = _make_track("b", camelot="9A")   # one step adjacent
        cost_adj = transition_cost(a, b)
        c = _make_track("c", camelot="2B")   # far away
        cost_far = transition_cost(a, c)
        assert cost_adj.harmonic_cost < cost_far.harmonic_cost


class TestSetlistPlanner:
    def _make_pool(self):
        return [
            _make_track("a", bpm=128, camelot="8A",  energy=0.3),
            _make_track("b", bpm=130, camelot="8B",  energy=0.5),
            _make_track("c", bpm=132, camelot="9A",  energy=0.7),
            _make_track("d", bpm=140, camelot="9B",  energy=0.9),
            _make_track("e", bpm=128, camelot="10A", energy=0.6),
        ]

    def test_optimize_returns_all_tracks(self):
        from scripts.core.setlist_planner import SetlistPlanner, EnergyArc
        result = SetlistPlanner().optimize(self._make_pool(), arc=EnergyArc.RAMP_UP)
        assert len(result) == 5

    def test_optimize_returns_unique_track_names(self):
        from scripts.core.setlist_planner import SetlistPlanner, EnergyArc
        result = SetlistPlanner().optimize(self._make_pool(), arc=EnergyArc.RAMP_UP)
        names = [r["name"] for r in result]
        assert len(names) == len(set(names))

    def test_rise_arc_energy_trend(self):
        """For RAMP_UP arc, average energy of second half >= first half (loose tolerance)."""
        from scripts.core.setlist_planner import SetlistPlanner, EnergyArc
        result = SetlistPlanner().optimize(self._make_pool(), arc=EnergyArc.RAMP_UP)
        energies = [r["energy"] for r in result]
        mid = len(energies) // 2
        first_avg  = sum(energies[:mid]) / max(len(energies[:mid]), 1)
        second_avg = sum(energies[mid:]) / max(len(energies[mid:]), 1)
        assert second_avg >= first_avg - 0.15

    def test_single_track_returns_one(self):
        from scripts.core.setlist_planner import SetlistPlanner, EnergyArc
        result = SetlistPlanner().optimize([_make_track("solo")], arc=EnergyArc.MOUNTAIN)
        assert len(result) == 1

    def test_empty_pool_returns_empty(self):
        from scripts.core.setlist_planner import SetlistPlanner, EnergyArc
        result = SetlistPlanner().optimize([], arc=EnergyArc.MOUNTAIN)
        assert result == []

    def test_result_dicts_have_position_key(self):
        from scripts.core.setlist_planner import SetlistPlanner, EnergyArc
        result = SetlistPlanner().optimize(self._make_pool(), arc=EnergyArc.MOUNTAIN)
        for r in result:
            assert "position" in r
