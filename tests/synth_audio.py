"""Synthetic test audio shared by the render and scan end to end tests.

Not a test module: pytest collects only test_*.py, *_test.py and
e2e_test_suite.py (pyproject.toml), so this file holds helpers only."""

from __future__ import annotations

import numpy as np


def synth_click_track(bpm: float, seconds: float, pad_hz: float, seed: int,
                      sr: int = 44100) -> np.ndarray:
    """Metronome click (noise burst, accent every 4th beat) over a slowly
    breathing two partial pad. Peak stays under 1.0."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    pad = 0.15 * (np.sin(2 * np.pi * pad_hz * t) + 0.5 * np.sin(2 * np.pi * pad_hz * 2 * t))
    pad *= 0.5 + 0.5 * np.sin(2 * np.pi * t / 8.0) ** 2
    y = pad.astype(np.float32)
    beat = 60.0 / bpm
    click_len = int(0.02 * sr)
    env = np.exp(-np.arange(click_len) / (0.004 * sr)).astype(np.float32)
    burst = rng.standard_normal(click_len).astype(np.float32) * env
    k = 0
    while k * beat < seconds:
        s = int(k * beat * sr)
        e = min(n, s + click_len)
        y[s:e] += (0.9 if k % 4 == 0 else 0.6) * burst[: e - s]
        k += 1
    return np.clip(y, -1.0, 1.0)
