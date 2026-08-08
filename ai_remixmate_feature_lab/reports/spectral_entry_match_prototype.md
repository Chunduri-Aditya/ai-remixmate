# Spectral Entry-Match Prototype — Research Result

DJ_ENGINE_IMPROVEMENT_PLAN.md §5.1: "Prototype a small differentiable EQ+fader
controller (in the spirit of DJTransGAN, arXiv:2110.06525) ... as an
alternative candidate generator to compare against — not replace —
spectral_matching.py's existing mechanism."

Code: `ai_remixmate_features/research/spectral_entry_match.py`,
`ai_remixmate_features/research/run_spectral_entry_match.py`.
Tests: `tests/test_spectral_entry_match.py` (23 tests, all passing).
Reproduce: `cd ai_remixmate_feature_lab/python && python -m
ai_remixmate_features.research.run_spectral_entry_match`.

## What "the existing mechanism" actually is

`scripts/core/spectral_matching.py`'s `SpectralMatchModel` (read for
reference, never imported into this lab) is **not** an EQ/fader controller —
it is 24 per-mel-band similarity-scoring weights, used only to re-rank
B-entry candidate bars in `choose_entry_bar()`. It never outputs a
render-affecting parameter. So "compare against it" narrows to: does a
second scoring function over the same 24-dim `sims` input separate
liked-vs-disliked candidates better than the existing linear weighted mean?
That is what this prototype tests.

## Data availability finding (read this before the numbers below)

The plan's constraint pointed at `data/spectral_match_history.json`
(session_id-keyed, "already has sims/entry_bar/score/ts") as the bootstrap
dataset. That file's fields are as described, but **it contains no outcome
labels, and cannot under the current production code**:

```
scripts/core/spectral_matching.py:apply_feedback()
    entry = history.get(session_id)
    ...
    model.update(np.asarray(entry["sims"], ...), liked=liked, lr=lr)
    model.save()
    history.pop(session_id, None)          # <- the record is discarded here
    history_path().write_text(json.dumps(history, indent=2), ...)
```

The instant a verdict arrives, the individual `(sims, liked)` pair is folded
into the aggregate weights and the record is removed from history.json —
never persisted anywhere in labeled form. As of this prototype, this repo's
`data/spectral_match_history.json` holds exactly **6 entries**, all
representing renders still **awaiting** a verdict (none labeled):

```json
{"count": 6, "sims_min": 0.127, "sims_max": 0.998, "sims_mean": 0.4564,
 "score_min": 0.3313, "score_max": 0.8911, "score_mean": 0.4755}
```

There is no historical labeled dataset to bootstrap a real "does the learned
model beat the hand-tuned one on actual verdicts" comparison from — more
effort inside this lab cannot produce one. That requires a **production**
change (persist `(sims, liked)` pairs before/alongside folding them into the
aggregate) which is out of scope here: this lab must not modify
`scripts/core/spectral_matching.py`.

## What this prototype does instead: a synthetic methodology check

A fixed, hidden per-band "ground truth" weight vector (uniform-random in
`[0.3, 2.5]` per band, seeded) generates labeled `(sims, liked)` pairs with
logistic label noise around a 0.6 threshold — noisy, not a deterministic
function of the weighted score, the way real verdicts would actually behave.

Both models train **online**, one example at a time (matching how both
would really be updated per-render, not in a batch):

- `HandTunedBandModel` — an independent, faithful reimplementation of
  `SpectralMatchModel`'s exact update rule and weighted-mean score (same
  math, written from scratch in pure Python so nothing here imports
  `scripts/`).
- The registered `spectral_entry_match` `TinyOnlineMLP` (12 hidden units,
  tanh → sigmoid, backprop) — this lab's existing dependency-free neural
  primitive (`ai_remixmate_features/neural/core.py`), reused rather than
  reaching for PyTorch for a 24-input, 1-output problem this small.

Both are then evaluated against a **held-out** synthetic set built from the
**same** ground truth, on accuracy (score ≥ 0.5 vs. actual label) and AUC
(rank-based, pure Python).

## Results (8 seeds, 300 train / 150 test examples each)

| seed | hand-tuned acc | hand-tuned AUC | MLP acc | MLP AUC |
|---|---|---|---|---|
| 1     | 0.7867 | 0.8827 | 0.8067 | 0.8762 |
| 7     | 0.6800 | 0.8050 | 0.7600 | 0.7968 |
| 13    | 0.7867 | 0.8871 | 0.8067 | 0.8830 |
| 42    | 0.8133 | 0.8980 | 0.8400 | 0.8946 |
| 99    | 0.7867 | 0.8590 | 0.7867 | 0.8595 |
| 123   | 0.6667 | 0.7885 | 0.6933 | 0.7848 |
| 2024  | 0.8333 | 0.8969 | 0.8200 | 0.8951 |
| 31337 | 0.7400 | 0.8444 | 0.7400 | 0.8348 |
| **avg** | **0.7617** | **0.8577** | **0.7817** | **0.8531** |

MLP wins on accuracy in **5/8** seeds. Hand-tuned wins on AUC in **7/8**
seeds. Neither model dominates — this is a close, mixed result, not a clean
win for the learned approach.

## Interpretation

[Certain] No real evaluation of "does a learned model beat the hand-tuned
one on actual user verdicts" is possible today — the labeled data doesn't
exist and can't be reconstructed from what's persisted.

[Likely] On synthetic data, the small MLP has a slight edge on thresholded
accuracy (+0.02 average) but the simpler linear hand-tuned rule has a slight
edge on ranking quality (+0.005 average AUC). At this data volume (300
online updates) and this problem size (24 inputs, 1 output), the two
approaches are roughly at parity — there's no evidence here that the added
complexity and reduced interpretability of a small neural model is worth it
over the existing hand-tuned linear weights.

[Guessing — a real limitation of this synthetic check] The ground truth used
to generate labels is itself a **linear** weighted sum of the input bands.
That structurally favors a linear-ish comparison and does not test whether
real preference functions have nonlinear structure (e.g., specific band
*combinations* mattering, not just individual band importance) that only a
model with a hidden layer could exploit. A fairer follow-up would generate
labels from a deliberately nonlinear ground truth (e.g., an XOR-style
interaction between two band groups) to see whether the MLP's extra
capacity shows a bigger advantage there — that would be a more informative
test of whether the added model complexity buys anything, independent of
whether real labeled data ever becomes available.

## Recommendation

Do not build this as a production feature. The research question this
prototype could answer — "does a learned scorer beat the hand-tuned weights
on real verdicts" — remains unanswered because the labeled data doesn't
exist; that is the actual finding worth acting on. If this is worth
revisiting, the real next step is a small, deliberate production change (not
in this lab): have `apply_feedback()` append `{sims, liked, ts}` to a
lightweight labeled-events log *before* folding the verdict into the
aggregate weights (it already has both values in scope at that exact call
site), so a genuine comparison becomes possible after a few weeks of real
usage — mirroring the same "instrument first, model later" discipline
`SELF_IMPROVING_DJ_RESEARCH.md` already established for the separate
setlist-feedback research track.
