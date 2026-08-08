"""
Runner script for the spectral_entry_match prototype
(DJ_ENGINE_IMPROVEMENT_PLAN.md §5.1). See spectral_entry_match.py's module
docstring for full scope and the data-availability finding this is built
around.

Usage
-----
    cd ai_remixmate_feature_lab/python
    python -m ai_remixmate_features.research.run_spectral_entry_match

Prints:
  1. Descriptive stats over whatever real (unlabeled) renders exist in
     data/spectral_match_history.json — context only, never used as
     training data.
  2. A per-seed comparison table (HandTunedBandModel vs. the learned
     TinyOnlineMLP) on synthetic labeled data, plus the across-seed average
     and win-count — the actual research result this prototype produces.

This script does not write anything outside ai_remixmate_feature_lab/ (it
only reads data/spectral_match_history.json for the descriptive section).
"""

from __future__ import annotations

import json
from pathlib import Path

from .spectral_entry_match import describe_real_renders, load_real_render_sims, run_experiment

DEFAULT_SEEDS = (1, 7, 13, 42, 99, 123, 2024, 31337)


def _repo_root() -> Path:
    # ai_remixmate_feature_lab/python/ai_remixmate_features/research/ -> repo root
    return Path(__file__).resolve().parents[4]


def main(seeds: tuple[int, ...] = DEFAULT_SEEDS, n_train: int = 300, n_test: int = 150) -> dict:
    history_path = _repo_root() / "data" / "spectral_match_history.json"
    real_entries = load_real_render_sims(history_path)
    real_summary = describe_real_renders(real_entries)

    print("=" * 72)
    print("Real (unlabeled) render data — descriptive only, NOT used for training")
    print("=" * 72)
    print(json.dumps(real_summary, indent=2))
    print()

    print("=" * 72)
    print("Synthetic labeled-data comparison: HandTunedBandModel vs. learned MLP")
    print("=" * 72)

    rows = []
    for seed in seeds:
        result = run_experiment(n_train=n_train, n_test=n_test, seed=seed)
        row = {
            "seed": seed,
            "hand_tuned_acc": result["hand_tuned_band_model"]["accuracy"],
            "hand_tuned_auc": result["hand_tuned_band_model"]["auc"],
            "mlp_acc": result["learned_mlp"]["accuracy"],
            "mlp_auc": result["learned_mlp"]["auc"],
        }
        rows.append(row)
        print(
            f"seed={seed:>6}  hand_tuned(acc={row['hand_tuned_acc']:.4f} "
            f"auc={row['hand_tuned_auc']:.4f})  "
            f"mlp(acc={row['mlp_acc']:.4f} auc={row['mlp_auc']:.4f})"
        )

    def _avg(key: str) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 4)

    mlp_acc_wins = sum(1 for r in rows if r["mlp_acc"] > r["hand_tuned_acc"])
    mlp_auc_wins = sum(1 for r in rows if r["mlp_auc"] > r["hand_tuned_auc"])

    summary = {
        "n_seeds": len(rows),
        "avg_hand_tuned_acc": _avg("hand_tuned_acc"),
        "avg_hand_tuned_auc": _avg("hand_tuned_auc"),
        "avg_mlp_acc": _avg("mlp_acc"),
        "avg_mlp_auc": _avg("mlp_auc"),
        "mlp_wins_on_accuracy": f"{mlp_acc_wins}/{len(rows)}",
        "mlp_wins_on_auc": f"{mlp_auc_wins}/{len(rows)}",
    }
    print()
    print(json.dumps(summary, indent=2))
    return {"real_render_summary": real_summary, "per_seed": rows, "summary": summary}


if __name__ == "__main__":
    main()
