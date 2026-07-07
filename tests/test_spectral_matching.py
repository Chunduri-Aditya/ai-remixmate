"""
tests/test_spectral_matching.py — learned spectrogram cue matching.

Covers the deterministic pieces without librosa (similarity math, model
updates, feedback persistence) plus a librosa-backed end-to-end candidate
search (marked dj_analysis, auto-skips where librosa is broken).
"""

import json

import numpy as np
import pytest

from scripts.core.spectral_matching import (
    N_BANDS,
    SpectralMatchModel,
    apply_feedback,
    band_similarity,
    match_score,
    record_render,
)

SR = 44100


# ---------------------------------------------------------------------------
# Similarity math (pure numpy)
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical_profiles_score_one(self):
        p = np.random.default_rng(1).random(N_BANDS) + 0.1
        p = p / p.sum()
        sims = band_similarity(p, p)
        assert np.allclose(sims, 1.0)
        assert match_score(sims, np.ones(N_BANDS)) == pytest.approx(1.0)

    def test_disjoint_profiles_score_zero(self):
        a = np.zeros(N_BANDS)
        a[:12] = 1 / 12
        b = np.zeros(N_BANDS)
        b[12:] = 1 / 12
        sims = band_similarity(a, b)
        assert np.allclose(sims, 0.0)

    def test_sims_bounded(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            a = rng.random(N_BANDS)
            b = rng.random(N_BANDS)
            sims = band_similarity(a / a.sum(), b / b.sum())
            assert np.all(sims >= 0.0) and np.all(sims <= 1.0)

    def test_weights_shift_the_score(self):
        a = np.full(N_BANDS, 1.0 / N_BANDS)
        b = a.copy()
        b[0] *= 3          # band 0 mismatched
        b = b / b.sum()
        sims = band_similarity(a, b)
        w_flat = np.ones(N_BANDS)
        w_band0 = np.ones(N_BANDS)
        w_band0[0] = 3.0
        # Weighting the mismatched band harder must lower the score
        assert match_score(sims, w_band0) < match_score(sims, w_flat)


# ---------------------------------------------------------------------------
# Online learning
# ---------------------------------------------------------------------------

class TestModelLearning:
    def test_liked_boosts_high_similarity_bands(self):
        m = SpectralMatchModel()
        sims = np.full(N_BANDS, 0.5)
        sims[3] = 1.0
        m.update(sims, liked=True, lr=0.5)
        assert m.weights[3] > m.weights[0]
        assert m.n_updates == 1

    def test_disliked_penalizes_high_similarity_bands(self):
        m = SpectralMatchModel()
        sims = np.full(N_BANDS, 0.5)
        sims[3] = 1.0
        m.update(sims, liked=False, lr=0.5)
        assert m.weights[3] < m.weights[0]

    def test_weights_stay_clipped_and_normalized(self):
        m = SpectralMatchModel()
        sims = np.zeros(N_BANDS)
        sims[0] = 1.0
        for _ in range(100):
            m.update(sims, liked=True, lr=1.0)
        assert np.all(m.weights > 0)
        assert float(m.weights.mean()) == pytest.approx(1.0, abs=1e-6)
        # clip band is [0.1, 3.0] pre-normalization → ratio can never exceed 30
        assert float(m.weights.max() / m.weights.min()) <= 30.0 + 1e-9

    def test_bad_sims_shape_is_ignored(self):
        m = SpectralMatchModel()
        before = m.weights.copy()
        m.update(np.ones(5), liked=True)
        assert np.array_equal(m.weights, before)
        assert m.n_updates == 0

    def test_persistence_roundtrip(self, tmp_path):
        m = SpectralMatchModel()
        m.update(np.linspace(0, 1, N_BANDS), liked=True, lr=0.3)
        p = tmp_path / "model.json"
        m.save(p)
        m2 = SpectralMatchModel.load(p)
        assert np.allclose(m.weights, m2.weights)
        assert m2.n_updates == 1

    def test_corrupt_model_file_falls_back_fresh(self, tmp_path):
        p = tmp_path / "model.json"
        p.write_text("{not json", encoding="utf-8")
        m = SpectralMatchModel.load(p)
        assert np.allclose(m.weights, 1.0)


# ---------------------------------------------------------------------------
# Feedback flow (history → model update)
# ---------------------------------------------------------------------------

class TestFeedbackFlow:
    @pytest.fixture(autouse=True)
    def _isolate_data_files(self, tmp_path, monkeypatch):
        import scripts.core.spectral_matching as sm
        monkeypatch.setattr(sm, "model_path", lambda: tmp_path / "model.json")
        monkeypatch.setattr(sm, "history_path", lambda: tmp_path / "history.json")

    def test_record_then_feedback_updates_model(self, tmp_path):
        info = {"sims": [0.9] * N_BANDS, "entry_bar": 8, "score": 0.9}
        record_render("sess-1", info)
        assert json.loads((tmp_path / "history.json").read_text())["sess-1"]["entry_bar"] == 8

        out = apply_feedback("sess-1", liked=True, lr=0.2)
        assert out["updated"] is True
        assert out["n_updates"] == 1
        # record is one-shot: consumed on feedback
        assert "sess-1" not in json.loads((tmp_path / "history.json").read_text())
        assert apply_feedback("sess-1", liked=True)["updated"] is False

    def test_feedback_for_unknown_session(self):
        out = apply_feedback("nope", liked=False)
        assert out["updated"] is False


# ---------------------------------------------------------------------------
# End-to-end candidate search (librosa)
# ---------------------------------------------------------------------------

@pytest.mark.dj_analysis
class TestCandidateSearch:
    def test_picks_spectrally_matching_phrase(self):
        """B's intro is noise, its later phrase is A's timbre — the matcher
        must move B's entry to the matching phrase, not bar 0."""
        from scripts.core.spectral_matching import choose_entry_bar

        rng = np.random.default_rng(3)
        bar_sec = 60.0 / 120.0 * 4          # 2 s bars at 120 BPM
        tone = lambda hz, secs: 0.5 * np.sin(  # noqa: E731
            2 * np.pi * hz * np.linspace(0, secs, int(SR * secs), endpoint=False)
        ).astype(np.float32)

        audio_a = tone(440.0, 16 * bar_sec)
        noise = (0.5 * rng.standard_normal(int(8 * bar_sec * SR))).astype(np.float32)
        audio_b = np.concatenate([noise, tone(440.0, 16 * bar_sec)])

        bars_b = [(i * bar_sec, (i + 1) * bar_sec) for i in range(24)]
        best = choose_entry_bar(
            audio_a, audio_b, SR,
            exit_time_a=16 * bar_sec,
            transition_seconds=4 * bar_sec,
            bars_b=bars_b,
            weights=np.ones(N_BANDS),
        )
        assert best["entry_bar"] == 8, f"expected the tonal phrase at bar 8, got {best}"
        assert best["score"] == max(c["score"] for c in best["candidates"])

    def test_apply_spectral_matching_mutates_plan(self):
        from scripts.core.classic_engine import analyze_structure, plan_transition
        from scripts.core.spectral_matching import apply_spectral_matching

        rng = np.random.default_rng(5)
        t = np.linspace(0, 30.0, SR * 30, endpoint=False)
        env = (1 + np.sign(np.sin(2 * np.pi * 2.0 * t))) / 2
        audio_a = (0.5 * np.sin(2 * np.pi * 440 * t) * env).astype(np.float32)
        audio_b = (0.5 * np.sin(2 * np.pi * 330 * t) * env
                   + 0.05 * rng.standard_normal(len(t))).astype(np.float32)

        sa = analyze_structure(audio_a, SR)
        sb = analyze_structure(audio_b, SR)
        plan = plan_transition(sa, sb, transition_bars=8)
        info = apply_spectral_matching(plan, audio_a, audio_b, sb, SR)
        if info is not None:   # None only if B's bar grid came out too short
            assert plan.entry_bar_b == info["entry_bar"]
            assert plan.entry_time_b == pytest.approx(sb.bars[info["entry_bar"]][0])
            assert 0.0 <= info["score"] <= 1.0


# ---------------------------------------------------------------------------
# Drop-aware entry scoring
# ---------------------------------------------------------------------------

class TestDropAlignment:
    def test_no_drop_is_neutral(self):
        from scripts.core.spectral_matching import _drop_alignment_factor
        factor, delta = _drop_alignment_factor(10.0, None, 30.0, 2.0)
        assert factor == 1.0 and delta is None

    def test_drop_shortly_after_handover_gets_bonus(self):
        from scripts.core.spectral_matching import _drop_alignment_factor
        # entry 10s + 30s overlap → handover at 40s; drop at 44s (2 bars later)
        factor, delta = _drop_alignment_factor(10.0, 44.0, 30.0, 2.0)
        assert factor > 1.0
        assert delta == 4.0

    def test_drop_inside_overlap_is_penalized(self):
        from scripts.core.spectral_matching import _drop_alignment_factor
        # drop at 25s falls inside the 10–40s overlap
        factor, delta = _drop_alignment_factor(10.0, 25.0, 30.0, 2.0)
        assert factor == 0.9
        assert delta == -15.0

    def test_far_away_drop_is_neutral(self):
        from scripts.core.spectral_matching import _drop_alignment_factor
        factor, _ = _drop_alignment_factor(10.0, 400.0, 30.0, 2.0)
        assert factor == 1.0

    def test_closer_drop_beats_farther_drop(self):
        from scripts.core.spectral_matching import _drop_alignment_factor
        near, _ = _drop_alignment_factor(0.0, 32.0, 30.0, 2.0)   # 2s after handover
        far, _ = _drop_alignment_factor(0.0, 58.0, 30.0, 2.0)    # 28s after handover
        assert near > far > 1.0
