from __future__ import annotations

import math

from ai_remixmate_features.neural.feature_registry import (
    FEATURE_MODEL_SPECS,
    SPECTRAL_SIMS_SIZE,
    feature_names,
)
from ai_remixmate_features.research.spectral_entry_match import (
    HandTunedBandModel,
    auc,
    describe_real_renders,
    generate_synthetic_dataset,
    load_real_render_sims,
    make_ground_truth_weights,
    run_experiment,
    sims_to_mlp_input,
)


def test_spectral_entry_match_registered():
    assert "spectral_entry_match" in feature_names()
    spec = FEATURE_MODEL_SPECS["spectral_entry_match"]
    assert spec.input_size == SPECTRAL_SIMS_SIZE == 24


class TestHandTunedBandModel:
    def test_starts_at_equal_weights(self):
        model = HandTunedBandModel()
        assert model.weights == [1.0] * SPECTRAL_SIMS_SIZE
        assert model.n_updates == 0

    def test_score_is_bounded_weighted_mean(self):
        model = HandTunedBandModel()
        sims = [0.5] * SPECTRAL_SIMS_SIZE
        assert model.score(sims) == 0.5   # equal weights -> plain mean

    def test_liked_update_favors_above_average_bands(self):
        model = HandTunedBandModel()
        sims = [0.9] * 4 + [0.1] * (SPECTRAL_SIMS_SIZE - 4)
        model.update(sims, liked=True, lr=0.5)
        # The four above-mean bands should have gained weight relative to the rest.
        assert min(model.weights[:4]) > max(model.weights[4:])
        assert model.n_updates == 1

    def test_disliked_update_flips_direction(self):
        liked_model = HandTunedBandModel()
        disliked_model = HandTunedBandModel()
        sims = [0.9] * 4 + [0.1] * (SPECTRAL_SIMS_SIZE - 4)
        liked_model.update(sims, liked=True, lr=0.5)
        disliked_model.update(sims, liked=False, lr=0.5)
        # Same input, opposite verdict -> the above-mean bands move opposite ways.
        assert liked_model.weights[0] > 1.0
        assert disliked_model.weights[0] < 1.0

    def test_weights_stay_positive_and_mean_renormalized(self):
        """NOTE: clip-then-renormalize (identical order to the real
        SpectralMatchModel.update() this mirrors) does NOT strictly
        guarantee every weight stays inside [0.1, 3.0] after the
        mean-renormalization division — clipping happens first, so a value
        already at a bound can be pushed back outside it by the /mean step.
        Verified against an adversarial one-hot sims vector at high lr; the
        real invariants the algorithm actually guarantees are "weights stay
        positive" and "mean stays at 1" (renormalization's whole point)."""
        model = HandTunedBandModel()
        sims = [1.0] + [0.0] * (SPECTRAL_SIMS_SIZE - 1)
        for _ in range(50):
            model.update(sims, liked=True, lr=0.9)
        assert all(w > 0 for w in model.weights)
        assert math.isclose(sum(model.weights) / len(model.weights), 1.0, abs_tol=1e-6)

    def test_weights_stay_within_bounds_under_realistic_updates(self):
        """Under a moderate, non-adversarial lr and correlated (not one-hot)
        sims vectors — the realistic case, since real per-band trajectory
        similarities are smooth, not maximally extreme — weights do stay
        within the nominal [0.1, 3.0] band in practice."""
        model = HandTunedBandModel()
        rng_sims = [0.6, 0.55, 0.5, 0.45] * (SPECTRAL_SIMS_SIZE // 4)
        for i in range(50):
            model.update(rng_sims, liked=(i % 2 == 0), lr=0.2)
        assert all(0.1 - 1e-9 <= w <= 3.0 + 1e-9 for w in model.weights)

    def test_mismatched_length_is_a_no_op(self):
        model = HandTunedBandModel()
        before = list(model.weights)
        model.update([0.5, 0.5], liked=True)   # wrong length
        assert model.weights == before
        assert model.n_updates == 0


def test_sims_to_mlp_input_maps_zero_one_to_minus_one_one():
    assert sims_to_mlp_input([0.0, 0.5, 1.0]) == [-1.0, 0.0, 1.0]


class TestSyntheticDataset:
    def test_deterministic_under_fixed_seed(self):
        examples_a, truth_a = generate_synthetic_dataset(20, seed=5)
        examples_b, truth_b = generate_synthetic_dataset(20, seed=5)
        assert examples_a == examples_b
        assert truth_a == truth_b

    def test_different_seeds_differ(self):
        examples_a, _ = generate_synthetic_dataset(20, seed=5)
        examples_b, _ = generate_synthetic_dataset(20, seed=6)
        assert examples_a != examples_b

    def test_shape_and_bounds(self):
        examples, truth = generate_synthetic_dataset(15, seed=1)
        assert len(examples) == 15
        assert len(truth) == SPECTRAL_SIMS_SIZE
        for sims, liked in examples:
            assert len(sims) == SPECTRAL_SIMS_SIZE
            assert all(0.0 <= s <= 1.0 for s in sims)
            assert isinstance(liked, bool)

    def test_reusing_ground_truth_produces_consistent_label_distribution(self):
        """A held-out set built from the SAME ground truth should have a
        plausible label mix (not degenerately all-one-class) — sanity check
        that the label noise model is doing something reasonable."""
        _, truth = generate_synthetic_dataset(10, seed=3)
        examples, _ = generate_synthetic_dataset(300, seed=4, ground_truth=truth)
        liked_fraction = sum(1 for _, liked in examples if liked) / len(examples)
        assert 0.05 < liked_fraction < 0.95


class TestAUC:
    def test_perfect_separation_is_one(self):
        assert auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0

    def test_perfect_inversion_is_zero(self):
        assert auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0

    def test_ties_score_half_credit(self):
        assert auc([0.5, 0.5], [1, 0]) == 0.5

    def test_single_class_is_nan(self):
        assert math.isnan(auc([0.1, 0.2, 0.3], [1, 1, 1]))


class TestRealRenderLoading:
    def test_missing_file_returns_empty_list(self, tmp_path):
        assert load_real_render_sims(tmp_path / "does-not-exist.json") == []

    def test_describe_empty_is_zero_count(self):
        assert describe_real_renders([]) == {"count": 0}

    def test_describe_real_history_file_if_present(self):
        """Reads the actual repo data file if it exists — descriptive only,
        never used as labeled training data (see module docstring). Skips
        gracefully (empty list) when running somewhere that file doesn't
        exist yet, e.g. a machine that has never rendered a mix."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        history_path = repo_root / "data" / "spectral_match_history.json"
        entries = load_real_render_sims(history_path)
        summary = describe_real_renders(entries)
        assert "count" in summary
        if entries:
            assert summary["sims_min"] is not None


class TestRunExperiment:
    def test_returns_expected_shape(self):
        result = run_experiment(n_train=40, n_test=20, seed=1)
        assert result["n_train"] == 40
        assert result["n_test"] == 20
        assert len(result["ground_truth_weights"]) == SPECTRAL_SIMS_SIZE
        for key in ("hand_tuned_band_model", "learned_mlp"):
            assert 0.0 <= result[key]["accuracy"] <= 1.0
            metrics_auc = result[key]["auc"]
            assert math.isnan(metrics_auc) or 0.0 <= metrics_auc <= 1.0

    def test_deterministic_under_fixed_seed(self):
        a = run_experiment(n_train=40, n_test=20, seed=3)
        b = run_experiment(n_train=40, n_test=20, seed=3)
        assert a == b

    def test_mlp_actually_trains_not_frozen_at_init(self):
        result = run_experiment(n_train=100, n_test=20, seed=9)
        assert result["learned_mlp"]["examples_seen"] == 100
        assert result["hand_tuned_band_model"]["n_updates"] == 100
