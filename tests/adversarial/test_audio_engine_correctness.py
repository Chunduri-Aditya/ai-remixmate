"""
Adversarial audio-QA — DJ engine correctness (2026-07-12).

Hunts *silent wrong audio* in scripts/core/. Bugs are marked xfail(strict) so
the suite stays green until fixed; verified-correct behaviors are plain passing
"control" tests that lock in the claims so a future regression trips them.

Independent measurement:
  * loudness  → pyloudnorm (a separate BS.1770 implementation), never the
    code-under-test's own compute_lufs.
  * true-peak → 4x polyphase oversample + max|.| computed here.

Run:  pytest tests/adversarial/test_audio_engine_correctness.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as ss

pyln = pytest.importorskip("pyloudnorm")

from scripts.core.mastering import (          # noqa: E402
    apply_limiter, compute_lufs, master_mix, _kweight_coefficients,
)
from scripts.core.dj_engine import (          # noqa: E402
    _stem_bass_ramp, _stem_crossfade_curves, _seam_blend,
)

SR = 44100


def _ind_lufs(x):
    return pyln.Meter(SR).integrated_loudness(x.astype(np.float64))


def _true_peak_db(x, os=4):
    xo = ss.resample_poly(x.astype(np.float64), os, 1)
    return 20 * np.log10(np.max(np.abs(xo)) + 1e-12)


# ===========================================================================
# FIXED [was P1] — K-weighting is now ITU-R BS.1770-4 compliant.
# Stage-1 high-shelf reaches +4 dB at HF and Stage-2 RLB high-pass uses the
# spec Q (0.5003, not Butterworth 0.707). compute_lufs now agrees with a
# reference meter (pyloudnorm) to <0.5 LU across the whole spectrum, so
# master_mix's -14 LUFS target is met in reality, independent of spectral
# balance. These tests are regression guards for the fix (mastering.py:52-90).
# ===========================================================================

def test_kweight_highshelf_gain_matches_spec():
    b1, a1, _, _ = _kweight_coefficients(SR)
    _, h = ss.freqz(b1, a1, worN=[np.pi * 0.99])   # near Nyquist
    hf_gain_db = 20 * np.log10(abs(h[0]))
    assert abs(hf_gain_db - 4.0) < 0.5, f"HF shelf gain {hf_gain_db:.2f} dB, spec +4.0"


def test_kweight_matches_published_48k_reference_coefficients():
    b1, a1, b2, a2 = _kweight_coefficients(48000)
    assert abs(b1[0] - 1.53512485958697) < 1e-4      # stage-1 shelf
    assert abs(a1[2] - 0.73248077421585) < 1e-4
    assert abs(a2[1] - (-1.99004745483398)) < 1e-6   # stage-2 RLB high-pass
    assert abs(a2[2] - 0.99007225036621) < 1e-6


def test_compute_lufs_agrees_with_reference_across_spectrum():
    t = np.arange(SR * 4) / SR
    worst = 0.0
    for f in (30, 50, 100, 500, 1000, 4000, 12000):
        x = (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)
        worst = max(worst, abs(compute_lufs(x, SR) - _ind_lufs(x)))
    assert worst < 0.5, f"max |code-vs-reference| LUFS error = {worst:.2f} LU"


def test_master_mix_true_loudness_is_spectrum_independent():
    """After the K-weight fix, two mixes normalized to the same -14 target land
    at the same TRUE loudness (within ~0.5 LU) regardless of spectral balance —
    the meter is compliant. Before the fix this gap was several LU."""
    t = np.arange(SR * 6) / SR
    bassy = (0.3 * np.sin(2 * np.pi * 80 * t)).astype(np.float32)
    bright = (0.3 * np.sin(2 * np.pi * 9000 * t)).astype(np.float32)
    out_b, _ = master_mix(bassy, SR, target_lufs=-14.0)
    out_h, _ = master_mix(bright, SR, target_lufs=-14.0)
    true_gap = abs(_ind_lufs(out_b) - _ind_lufs(out_h))
    assert true_gap < 0.6, f"true-loudness gap {true_gap:.2f} LU — meter not spectrum-flat"


# ===========================================================================
# BUG 2 [P2] — render() silently produces a transition into SILENCE when
# entry_time_b lands near/after the end of B. entry_sample_b is clamped to
# len-1, so B's transition slice is 1 sample + zero padding: the incoming
# track is absent for the whole second half. No error, no warning.
# (Requires an out-of-range plan; plan_transition normally avoids it.)
# ===========================================================================

@pytest.mark.xfail(reason="dj_engine.py:846 clamps entry_sample_b but never "
                          "guards entry past end of B -> transition into "
                          "silence with no error/warning", strict=True)
def test_render_entry_past_end_of_b_is_not_silent():
    from scripts.core.dj_engine import DJEngine
    from scripts.core.dj_analysis import TransitionPlan, EQPlan
    eng = DJEngine(sr=SR)
    A = (0.3 * np.sin(2 * np.pi * 220 * np.arange(SR * 10) / SR)).astype(np.float32)
    B = (0.3 * np.sin(2 * np.pi * 330 * np.arange(SR * 10) / SR)).astype(np.float32)
    bars = 8
    eq = EQPlan(400, 80, bars // 2, bars // 2, 150, 0, bars, 0, bars)
    plan = TransitionPlan(0, 4.0, 0, 9.9, bars, bars * (60 / 120) * 4,
                          120.0, 120.0, 1.0, eq, 0.8)   # entry 9.9s into a 10s B
    out = eng.render(A, B, plan, full_output=True)
    exit_s = int(4.0 * SR); ts = int(bars * (60 / 120) * 4 * SR); mid = ts // 2
    second_half = out[exit_s + mid: exit_s + ts]
    rms_db = 20 * np.log10(np.sqrt(np.mean(second_half.astype(np.float64) ** 2)) + 1e-12)
    # B should be rising to full here; -74 dB means it never entered.
    assert rms_db > -40, f"transition 2nd-half RMS {rms_db:.1f} dB — B is absent (silent transition)"


# ===========================================================================
# CONTROL — verified-correct behaviors. These must keep PASSING.
# ===========================================================================

def test_limiter_true_peak_never_exceeds_ceiling():
    t = np.arange(SR) / SR
    x = np.sin(2*np.pi*997*t) + 0.9*np.sin(2*np.pi*5001*t) + 0.7*np.sin(2*np.pi*11003*t)
    x = (x / np.max(np.abs(x))).astype(np.float32)   # 0 dBFS, inter-sample peaks > 0
    for ceil in (-1.0, -0.3):
        y = apply_limiter(x, ceiling_db=ceil, sr=SR)
        assert _true_peak_db(y) <= ceil + 0.05, f"ceil {ceil}: TP {_true_peak_db(y):.3f}"
        assert np.all(np.isfinite(y))


def test_near_silence_not_pumped_and_finite():
    t = np.arange(SR * 4) / SR
    quiet = (0.001 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # ~-60 dBFS
    out, _ = master_mix(quiet, SR, target_lufs=-14.0)
    assert _ind_lufs(out) < -30 and np.all(np.isfinite(out))


def test_bass_ramp_no_simultaneous_overlap():
    n = SR * 4; t = np.arange(n) / SR
    da, db = _stem_bass_ramp((0.5*np.sin(2*np.pi*60*t)).astype(np.float32),
                             (0.5*np.sin(2*np.pi*90*t)).astype(np.float32),
                             n // 2, int(0.10 * SR))
    win = 256
    ea = np.array([np.sqrt(np.mean(da[i*win:(i+1)*win]**2)) for i in range(n//win)])
    eb = np.array([np.sqrt(np.mean(db[i*win:(i+1)*win]**2)) for i in range(n//win)])
    both = ((20*np.log10(ea+1e-12) > -40) & (20*np.log10(eb+1e-12) > -40)).sum()
    assert both == 0, f"{both} windows with both bass stems audible simultaneously"


def test_medium_crossfade_is_equal_power():
    rng = np.random.default_rng(7); n = SR * 4
    xa = rng.standard_normal(n).astype(np.float32) * 0.3
    xb = rng.standard_normal(n).astype(np.float32) * 0.3   # uncorrelated
    fo, fi = _stem_crossfade_curves(n, 0.5)
    mix = xa * fo + xb * fi
    win = 2048
    env = np.array([20*np.log10(np.sqrt(np.mean(mix[i*win:(i+1)*win].astype(np.float64)**2))+1e-12)
                    for i in range(n // win)])
    src = 20*np.log10(np.sqrt(np.mean(xa.astype(np.float64)**2)))
    dip = src - env[len(env)//10:-len(env)//10].min()
    assert dip < 1.0, f"midpoint power dip {dip:.2f} dB (equal-gain would be ~3)"


def test_seam_blend_transparent_on_correlated():
    rng = np.random.default_rng(1)
    seg = rng.standard_normal(int(0.03 * SR)).astype(np.float32) * 0.4
    assert np.max(np.abs(_seam_blend(seg, seg) - seg)) < 1e-6


def test_stems_enter_at_identical_sample_under_equal_similarity():
    n = SR * 4
    onsets = {int(np.argmax(_stem_crossfade_curves(n, 0.55)[1] > 1e-4)) for _ in range(3)}
    assert len(onsets) == 1


def test_tiv_score_within_documented_floor_on_key_templates():
    from scripts.core.tiv_scoring import tiv_harmonic_score, _key_chroma
    tpl = [_key_chroma(r, m) for r in range(12) for m in ("major", "minor")]
    lo = min(tiv_harmonic_score(a, b) for a in tpl for b in tpl)
    hi = max(tiv_harmonic_score(a, b) for a in tpl for b in tpl)
    assert 0.35 <= lo and hi <= 1.0, f"TIV range [{lo:.3f},{hi:.3f}] vs documented [0.36,1.0]"


def test_pitch_shift_for_camelot_roundtrip_and_range():
    from scripts.core.key_detection import pitch_shift_for_camelot, CAMELOT
    codes = sorted(set(CAMELOT.values()))
    for a in codes:
        for b in codes:
            s = pitch_shift_for_camelot(a, b)
            assert -6 <= s <= 6
            assert (s + pitch_shift_for_camelot(b, a)) % 12 == 0
