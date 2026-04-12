"""
scripts/core/beat_synth.py — Procedural drum beat synthesizer for AI RemixMate.

Generates a bridge beat that sits underneath the transition between two songs,
giving the mix energy and forward momentum while both tracks cross-fade.

Everything here is pure numpy/scipy synthesis — no sample files needed.

Instruments
-----------
  kick   — sine with pitch envelope (150 Hz → 50 Hz) + click transient
  snare  — bandpass-filtered noise burst + short sine body
  hihat  — highpass-filtered noise, fast decay (closed)
  openhat— same but slower decay (open hi-hat)

Genre presets
-------------
  techno  — 4-on-the-floor kick, 2&4 snare, straight 16th hats
  house   — 4-on-the-floor kick, 2&4 snare, offbeat 8th hats
  hiphop  — boom-bap kick on 1&3, snare on 2&4, swung 16th hats
  trap    — sparse kick, steady hihat 32nds, snare on 3
  dnb     — fast syncopated kick, snare on 2.5&4, dense hats
  ambient — just open hats on 1, no kick or snare
  auto    — alias: techno (safe default)

Usage
-----
  from scripts.core.beat_synth import render_beat, strudel_code

  audio = render_beat(bpm=128.0, genre="techno", bars=16, sr=22050)
  code  = strudel_code(bpm=128.0, genre="techno")
"""

from __future__ import annotations

from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# Instrument synthesis (numpy only — no scipy dependency at module level)
# ---------------------------------------------------------------------------

def _kick(sr: int) -> np.ndarray:
    """
    Kick drum: sine with falling pitch envelope + short noise click.
    Pitch drops 150 Hz → 50 Hz over ~80 ms (classic 808-style thump).
    """
    duration = 0.55
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr

    # Pitch envelope: exponential fall
    pitch = 50.0 + 100.0 * np.exp(-t / 0.07)
    phase = np.cumsum(2.0 * np.pi * pitch / sr)
    body  = np.sin(phase)

    # Amplitude envelope: punchy attack, smooth decay
    amp = np.exp(-t / 0.25)

    # Click transient (very short noise burst on the front)
    click_n = max(1, int(sr * 0.004))
    click   = np.random.default_rng(0).standard_normal(click_n).astype(np.float32)
    click  *= np.exp(-np.linspace(0, 15, click_n, dtype=np.float32))

    kick = body * amp
    kick[:click_n] += click * 0.4

    peak = float(np.abs(kick).max()) or 1.0
    return (kick / peak).astype(np.float32)


def _snare(sr: int) -> np.ndarray:
    """
    Snare: bandpass-filtered noise + 180 Hz body sine.
    """
    try:
        from scipy.signal import butter, lfilter
        _SCIPY = True
    except ImportError:
        _SCIPY = False

    duration = 0.22
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr

    # Body
    body = np.sin(2.0 * np.pi * 180.0 * t) * np.exp(-t / 0.025)

    # Snare buzz (filtered noise)
    rng   = np.random.default_rng(1)
    noise = rng.standard_normal(n).astype(np.float32)
    if _SCIPY:
        nyq = sr / 2.0
        lo  = min(max(200.0 / nyq, 0.001), 0.99)
        hi  = min(max(8000.0 / nyq, 0.001), 0.99)
        if lo < hi:
            b, a  = butter(2, [lo, hi], btype="band")
            noise = lfilter(b, a, noise).astype(np.float32)
    buzz = noise * np.exp(-t / 0.07)

    snare = body * 0.5 + buzz * 0.7
    peak  = float(np.abs(snare).max()) or 1.0
    return (snare / peak).astype(np.float32)


def _hihat(sr: int, open_: bool = False) -> np.ndarray:
    """
    Hi-hat: highpass-filtered white noise with fast (closed) or slow (open) decay.
    """
    try:
        from scipy.signal import butter, lfilter
        _SCIPY = True
    except ImportError:
        _SCIPY = False

    decay = 0.10 if open_ else 0.018
    n     = int(sr * max(decay * 5, 0.01))
    t     = np.arange(n, dtype=np.float32) / sr

    rng   = np.random.default_rng(2 if open_ else 3)
    noise = rng.standard_normal(n).astype(np.float32)

    if _SCIPY:
        nyq  = sr / 2.0
        cut  = min(max(8000.0 / nyq, 0.001), 0.99)
        b, a = butter(4, cut, btype="high")
        noise = lfilter(b, a, noise).astype(np.float32)

    amp = np.exp(-t / decay)
    hat = noise * amp
    peak = float(np.abs(hat).max()) or 1.0
    return (hat / peak).astype(np.float32)


# ---------------------------------------------------------------------------
# Genre patterns (16-step, 1 step = 1 sixteenth note)
# Each list contains (instrument, step_index, velocity 0-1) tuples.
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, list[tuple[str, int, float]]] = {
    "techno": [
        # 4-on-the-floor kick
        ("kick",  0, 0.9), ("kick",  4, 0.85), ("kick",  8, 0.9), ("kick", 12, 0.85),
        # Snare on 2 & 4
        ("snare", 4, 0.75), ("snare", 12, 0.75),
        # Straight 16th hi-hats
        *[("hh", i, 0.5 if i % 4 == 0 else 0.3) for i in range(16)],
    ],
    "house": [
        # 4-on-the-floor kick
        ("kick",  0, 0.9), ("kick",  4, 0.85), ("kick",  8, 0.9), ("kick", 12, 0.85),
        # Snare on 2 & 4
        ("snare", 4, 0.7), ("snare", 12, 0.7),
        # Offbeat 8th hi-hats
        ("hh", 2, 0.5), ("hh", 6, 0.5), ("hh", 10, 0.5), ("hh", 14, 0.5),
        # Open hat on the + of 3
        ("oh", 10, 0.4),
    ],
    "hiphop": [
        # Boom-bap kick (1 & 3-and)
        ("kick",  0, 0.9), ("kick",  6, 0.7), ("kick",  8, 0.85), ("kick", 14, 0.6),
        # Snare on 2 & 4
        ("snare", 4, 0.8), ("snare", 12, 0.75),
        # Swung 8th hi-hats
        ("hh", 0, 0.45), ("hh", 3, 0.3), ("hh", 4, 0.45), ("hh", 7, 0.3),
        ("hh", 8, 0.45), ("hh", 11, 0.3), ("hh", 12, 0.45), ("hh", 15, 0.3),
    ],
    "trap": [
        # Sparse kick
        ("kick",  0, 0.9), ("kick",  6, 0.65), ("kick", 10, 0.7),
        # Snare on 3 (bar 2 of 4/4 in trap feel)
        ("snare", 8, 0.85),
        # Dense 32nd hi-hats (every 16th step, simulated with rapid hh)
        *[("hh", i, 0.35 + 0.1 * (i % 4 == 0)) for i in range(16)],
        # Open hat rolls
        ("oh", 4, 0.5), ("oh", 12, 0.5),
    ],
    "dnb": [
        # Syncopated DnB kick
        ("kick",  0, 0.9), ("kick",  5, 0.7), ("kick", 11, 0.75),
        # Snare on 2.5 (step 10) & 4
        ("snare", 10, 0.85), ("snare", 12, 0.65),
        # Busy 16th hats (every step, alternating velocity)
        *[("hh", i, 0.55 if i % 2 == 0 else 0.25) for i in range(16)],
    ],
    "ambient": [
        # Just open hi-hat on downbeats, very sparse
        ("oh", 0, 0.3), ("oh", 8, 0.25),
        ("hh", 4, 0.2), ("hh", 12, 0.2),
    ],
}

# Map genre aliases
_PATTERNS["auto"]  = _PATTERNS["techno"]
_PATTERNS["rnb"]   = _PATTERNS["hiphop"]
_PATTERNS["pop"]   = _PATTERNS["house"]
_PATTERNS["rock"]  = _PATTERNS["hiphop"]
_PATTERNS["jazz"]  = _PATTERNS["ambient"]


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_beat(
    bpm: float,
    genre: str = "auto",
    bars: int = 4,
    sr: int = 22050,
    intensity: float = 0.45,
    build_up: bool = True,
) -> np.ndarray:
    """
    Render a synthesized drum beat loop as a mono float32 numpy array.

    Parameters
    ----------
    bpm : float
        Tempo in beats per minute.
    genre : str
        Genre preset key (see _PATTERNS).
    bars : int
        Length of the loop in bars (1 bar = 16 sixteenth-note steps).
    sr : int
        Sample rate (Hz).
    intensity : float
        Overall gain scaling 0.0–1.0.
    build_up : bool
        If True, apply a linear fade-in so the beat "builds up" from silence.

    Returns
    -------
    np.ndarray — mono float32, length = bars × 4 beats × (sr * 60/bpm) samples.
    """
    bpm = max(60.0, min(200.0, float(bpm)))
    genre = genre.lower() if genre.lower() in _PATTERNS else "auto"

    # Duration of one sixteenth note in samples
    step_samples = int(sr * 60.0 / bpm / 4)   # bpm beats/min → /4 for 16th
    total_steps  = bars * 16
    total_samples = total_steps * step_samples

    # Pre-render instrument sounds
    sounds = {
        "kick":  _kick(sr),
        "snare": _snare(sr),
        "hh":    _hihat(sr, open_=False),
        "oh":    _hihat(sr, open_=True),
    }

    pattern = _PATTERNS[genre]
    output  = np.zeros(total_samples, dtype=np.float32)

    # Stamp each hit for each bar in the loop
    for bar in range(bars):
        for inst, step, vel in pattern:
            sound    = sounds[inst]
            position = (bar * 16 + step) * step_samples
            end      = min(position + len(sound), total_samples)
            length   = end - position
            if length > 0:
                output[position:end] += sound[:length] * vel

    # Normalise to prevent clipping before intensity scaling
    peak = float(np.abs(output).max())
    if peak > 0.0:
        output = output / peak

    output *= intensity

    # Build-up: linear fade-in over the first 2 bars
    if build_up and bars > 2:
        fade_samples = min(2 * 16 * step_samples, total_samples)
        fade         = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        output[:fade_samples] *= fade

    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Strudel code generator
# ---------------------------------------------------------------------------

_STRUDEL_PATTERNS: dict[str, str] = {
    "techno": """\
  // 4-on-the-floor kick
  s("bd bd bd bd").gain(0.9),
  // Snare on 2 & 4
  s("~ sd ~ sd").gain(0.75),
  // 16th hi-hats
  s("[hh*4]").gain(0.45).room(0.1),""",

    "house": """\
  // 4-on-the-floor kick
  s("bd bd bd bd").gain(0.85),
  // Snare on 2 & 4
  s("~ sd ~ sd").gain(0.7),
  // Offbeat 8th hi-hats + open hat on the + of 3
  s("~ hh ~ hh ~ hh ~ hh").gain(0.5),
  s("~ ~ ~ ~ ~ ~ oh ~").gain(0.4),""",

    "hiphop": """\
  // Boom-bap kick
  s("bd ~ ~ bd ~ ~ bd ~").gain(0.9),
  // Snare on 2 & 4
  s("~ sd ~ sd").gain(0.8),
  // Swung 8th hi-hats
  s("hh ~ hh ~ hh ~ hh ~").swing(0.1).gain(0.45),""",

    "trap": """\
  // Sparse trap kick
  s("bd ~ ~ ~ ~ bd ~ ~").gain(0.9),
  // Snare clap on 3
  s("~ ~ sd ~").gain(0.85),
  // Dense 16th hi-hats with velocity variation
  s("[hh*4]").gain(0.3).room(0.05),
  // Open hat rolls
  s("~ oh ~ oh").gain(0.4),""",

    "dnb": """\
  // Syncopated DnB kick
  s("bd ~ [~ bd] ~ bd ~ [~ bd] ~").fast(2).gain(0.9),
  // Amen-style snare
  s("~ ~ sd ~ ~ sd sd ~").fast(2).gain(0.8),
  // Busy hats
  s("[hh*4]").fast(2).gain(0.4),""",

    "ambient": """\
  // Minimal pulse — open hats only
  s("oh ~ ~ oh ~ ~ oh ~").gain(0.35).room(0.5).slow(2),
  s("hh ~ hh ~").gain(0.2).room(0.8),""",
}

_STRUDEL_PATTERNS["auto"]  = _STRUDEL_PATTERNS["techno"]
_STRUDEL_PATTERNS["rnb"]   = _STRUDEL_PATTERNS["hiphop"]
_STRUDEL_PATTERNS["pop"]   = _STRUDEL_PATTERNS["house"]
_STRUDEL_PATTERNS["rock"]  = _STRUDEL_PATTERNS["hiphop"]
_STRUDEL_PATTERNS["jazz"]  = _STRUDEL_PATTERNS["ambient"]


def strudel_code(bpm: float, genre: str = "auto") -> str:
    """
    Generate a Strudel REPL pattern string for the given BPM and genre.

    The returned string is valid Strudel syntax — paste it directly into
    https://strudel.cc and press Ctrl+Enter to hear it.

    setcps(bpm/60/4) sets cycles-per-second where 1 cycle = 1 bar of 4 beats.
    """
    genre   = genre.lower() if genre.lower() in _STRUDEL_PATTERNS else "auto"
    pattern = _STRUDEL_PATTERNS[genre]
    cps     = round(bpm / 60.0 / 4.0, 5)

    return f"""// AI RemixMate — generated bridge beat
// Genre: {genre}  |  BPM: {bpm:.1f}
//
// Paste into https://strudel.cc and press Ctrl+Enter to play.
// Tweak velocities (.gain), reverb (.room), filters (.lpf/.hpf),
// or swap drum sounds (bd, sd, hh, oh, cp, mt, ht, lt).

setcps({cps})

stack(
{pattern}
)"""
