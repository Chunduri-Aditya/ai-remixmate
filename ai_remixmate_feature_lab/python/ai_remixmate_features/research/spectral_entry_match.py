"""
ai_remixmate_features/research/spectral_entry_match.py

Research prototype for DJ_ENGINE_IMPROVEMENT_PLAN.md §5.1: "Prototype a small
differentiable EQ+fader controller (in the spirit of DJTransGAN, arXiv:2110.06525)
... as an alternative candidate generator to compare against — not replace —
spectral_matching.py's existing mechanism."

Scope, precisely
-----------------
scripts/core/spectral_matching.py's real mechanism (read for reference only —
never imported; see this lab's isolation rule) is NOT an EQ/fader controller.
It is 24 per-mel-band SIMILARITY-SCORING weights (`SpectralMatchModel`,
persisted in data/spectral_match_model.json) used only to re-rank B-entry
candidate bars in `choose_entry_bar()` — a linear weighted mean of per-band
trajectory similarities. It never outputs an audio-affecting parameter (no
fade curve, no EQ gain). A learned controller that actually "competes" with
it, in the narrow sense the plan means, is a second scoring function over the
same 24-dim `sims` input, evaluated on whether it separates liked-vs-disliked
mixes better than the existing linear model. `HandTunedBandModel` below is an
independent, read-only-inspired reimplementation of that exact math (never an
import) so the comparison is apples-to-apples without violating the lab's
"nothing here imports from scripts/" rule.

Data availability finding (read before trusting any number below)
-------------------------------------------------------------------
The plan's constraint says to "bootstrap dataset from
data/spectral_match_history.json (session_id-keyed, already has
sims/entry_bar/score/ts) before reaching for external DJ-mix scraping." That
file's *fields* are as described, but it does NOT contain outcome labels, and
by construction never will under the current production code:

    scripts/core/spectral_matching.py:apply_feedback() pops the session's
    history record the instant a verdict (👍/👎) arrives and folds it into
    SpectralMatchModel's aggregate per-band weights. The individual
    (sims_vector, liked) pair is never written back to history.json or
    anywhere else — only the running aggregate survives.

So `spectral_match_history.json` only ever holds renders that are AWAITING a
verdict (6 such entries exist in this repo's data/ as of this prototype, none
labeled). There is no historical labeled dataset to bootstrap a real
"does the learned model beat the hand-tuned one on actual verdicts"
comparison from — that comparison is not something more engineering effort
in this lab could produce; it requires a production logging change (log the
(sims, liked) pair before or alongside folding it into the aggregate) that is
out of scope here (this lab must not modify scripts/core/spectral_matching.py).

What this module does instead
------------------------------
Runs the honest substitute: a synthetic-data methodology check. A fixed,
hidden per-band "ground truth" preference function generates labeled
(sims, liked) pairs with logistic label noise (see `generate_synthetic_dataset`).
Both models train online (one example at a time, matching how both would
actually be updated in production — per-render, not batched) on the same
synthetic stream, then both are evaluated against a held-out synthetic set
using the SAME known ground truth. This answers "can a small gradient-trained
model recover a non-uniform band-importance pattern at least as well as the
existing hand-tuned linear-nudge rule, given labeled data" — a methodology
sanity check, not a production result. See reports/spectral_entry_match_prototype.md
for the actual numbers this produced and what they do/don't imply.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_remixmate_features.neural.core import TinyOnlineMLP
from ai_remixmate_features.neural.feature_registry import (
    FEATURE_MODEL_SPECS,
    SPECTRAL_SIMS_SIZE,
)

_WEIGHT_MIN, _WEIGHT_MAX = 0.1, 3.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


# ---------------------------------------------------------------------------
# Baseline: independent reimplementation of SpectralMatchModel
# ---------------------------------------------------------------------------

@dataclass
class HandTunedBandModel:
    """Independent reimplementation of scripts/core/spectral_matching.py's
    SpectralMatchModel — same update rule, same weighted-mean score, written
    from scratch in pure Python (this lab has no numpy in its base
    dependencies) so the comparison below doesn't require importing
    production code."""

    weights: list[float] = field(default_factory=lambda: [1.0] * SPECTRAL_SIMS_SIZE)
    n_updates: int = 0

    def score(self, sims: list[float]) -> float:
        total = sum(self.weights)
        if total <= 0:
            return sum(sims) / len(sims)
        return sum(w * s for w, s in zip(self.weights, sims)) / total

    def update(self, sims: list[float], liked: bool, lr: float = 0.2) -> None:
        if len(sims) != len(self.weights):
            return
        mean_sim = sum(sims) / len(sims)
        delta = [lr * (s - mean_sim) for s in sims]
        if not liked:
            delta = [-d for d in delta]
        self.weights = [
            _clamp(w + d, _WEIGHT_MIN, _WEIGHT_MAX) for w, d in zip(self.weights, delta)
        ]
        mean_w = sum(self.weights) / len(self.weights)
        if mean_w > 0:
            self.weights = [w / mean_w for w in self.weights]
        self.n_updates += 1


def sims_to_mlp_input(sims: list[float]) -> list[float]:
    """Map [0, 1] similarity values to [-1, 1] — TinyOnlineMLP clamps inputs
    to [-1, 1] (see NeuralFeatureSpec/TrainingEvent.validate)."""
    return [_clamp(2.0 * s - 1.0, -1.0, 1.0) for s in sims]


# ---------------------------------------------------------------------------
# Synthetic labeled dataset (see module docstring — real labels don't exist)
# ---------------------------------------------------------------------------

def make_ground_truth_weights(seed: int) -> list[float]:
    """A fixed, non-uniform hidden band-importance vector used ONLY to
    generate synthetic labels. Deliberately not uniform (unlike the hand-
    tuned model's ones-initialization) so there is something for a learner
    to discover beyond "just average the bands"."""
    rng = random.Random(seed)
    return [rng.uniform(0.3, 2.5) for _ in range(SPECTRAL_SIMS_SIZE)]


def generate_synthetic_dataset(
    n: int,
    seed: int = 0,
    ground_truth: list[float] | None = None,
    threshold: float = 0.6,
    noise_scale: float = 8.0,
) -> tuple[list[tuple[list[float], bool]], list[float]]:
    """Generate n synthetic (sims_vector, liked) examples.

    Each example's sims vector is a noisy realization of a single "candidate
    quality" level (bands trend together for a given candidate, mirroring
    the real band_similarity()/trajectory_similarity() outputs' bounded
    [0, 1], correlated-within-a-window character) plus independent per-band
    noise. The label is drawn from a logistic function of the ground-truth
    weighted score around `threshold`, so labels are noisy — not a
    deterministic function of the score — matching how real verdicts would
    behave (mood, distraction, unrelated taste all inject noise a perfect
    audio-only model could never fully explain).
    """
    rng = random.Random(seed)
    truth = ground_truth or make_ground_truth_weights(seed)
    total_truth = sum(truth)
    examples: list[tuple[list[float], bool]] = []
    for _ in range(n):
        base_quality = rng.uniform(0.2, 0.95)
        sims = [
            _clamp(base_quality + rng.uniform(-0.25, 0.25), 0.0, 1.0)
            for _ in range(SPECTRAL_SIMS_SIZE)
        ]
        true_score = sum(w * s for w, s in zip(truth, sims)) / total_truth
        logit = noise_scale * (true_score - threshold)
        p_liked = 1.0 / (1.0 + math.exp(-_clamp(logit, -30.0, 30.0)))
        liked = rng.random() < p_liked
        examples.append((sims, liked))
    return examples, truth


# ---------------------------------------------------------------------------
# Real (unlabeled) render data — descriptive statistics only
# ---------------------------------------------------------------------------

def load_real_render_sims(history_path: str | Path) -> list[dict[str, Any]]:
    """Read data/spectral_match_history.json for descriptive stats ONLY.

    Do not use this as labeled training data — see the module docstring's
    "Data availability finding". Returns [] if the file is missing or
    unreadable (this must never fail a lab test just because a machine
    hasn't produced any real renders yet).
    """
    try:
        raw = json.loads(Path(history_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    return list(raw.values()) if isinstance(raw, dict) else []


def describe_real_renders(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary stats over whatever real (unlabeled) render records exist —
    context for the report, not evaluation input."""
    if not entries:
        return {"count": 0}
    all_sims = [s for e in entries for s in e.get("sims", [])]
    scores = [float(e["score"]) for e in entries if "score" in e]
    return {
        "count": len(entries),
        "sims_min": round(min(all_sims), 4) if all_sims else None,
        "sims_max": round(max(all_sims), 4) if all_sims else None,
        "sims_mean": round(sum(all_sims) / len(all_sims), 4) if all_sims else None,
        "score_min": round(min(scores), 4) if scores else None,
        "score_max": round(max(scores), 4) if scores else None,
        "score_mean": round(sum(scores) / len(scores), 4) if scores else None,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC (probability a random positive outscores a random
    negative), pure Python — no sklearn in this lab's base dependencies.
    O(n_pos * n_neg); fine at prototype dataset sizes. Returns nan if either
    class is empty (undefined)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _evaluate(predict_fn, examples: list[tuple[list[float], bool]]) -> dict[str, float]:
    scores: list[float] = []
    labels: list[int] = []
    correct = 0
    for sims, liked in examples:
        p = predict_fn(sims)
        scores.append(p)
        labels.append(1 if liked else 0)
        correct += int((p >= 0.5) == liked)
    return {
        "accuracy": round(correct / len(examples), 4) if examples else float("nan"),
        "auc": round(auc(scores, labels), 4),
        "n": len(examples),
    }


def run_experiment(
    n_train: int = 300,
    n_test: int = 150,
    seed: int = 7,
    lr: float = 0.2,
) -> dict[str, Any]:
    """Train HandTunedBandModel and the registered spectral_entry_match
    TinyOnlineMLP online (one example at a time) on the same synthetic
    stream, then evaluate both against a held-out synthetic set generated
    from the SAME ground-truth weights. Returns a JSON-serializable result
    dict — see reports/spectral_entry_match_prototype.md for the actual
    numbers this produced and their interpretation.
    """
    train_examples, truth = generate_synthetic_dataset(n_train, seed=seed)
    test_examples, _ = generate_synthetic_dataset(n_test, seed=seed + 1000, ground_truth=truth)

    hand_tuned = HandTunedBandModel()
    mlp = TinyOnlineMLP(FEATURE_MODEL_SPECS["spectral_entry_match"])

    for sims, liked in train_examples:
        hand_tuned.update(sims, liked, lr=lr)
        mlp.train_one(sims_to_mlp_input(sims), [1.0 if liked else 0.0])

    hand_tuned_metrics = _evaluate(lambda s: hand_tuned.score(s), test_examples)
    mlp_metrics = _evaluate(lambda s: mlp.predict(sims_to_mlp_input(s))[0], test_examples)

    return {
        "n_train": n_train,
        "n_test": n_test,
        "seed": seed,
        "learning_rate": lr,
        "ground_truth_weights": [round(w, 4) for w in truth],
        "hand_tuned_band_model": {**hand_tuned_metrics, "n_updates": hand_tuned.n_updates},
        "learned_mlp": {**mlp_metrics, "examples_seen": mlp.examples_seen, "last_loss": round(mlp.last_loss, 6)},
    }
