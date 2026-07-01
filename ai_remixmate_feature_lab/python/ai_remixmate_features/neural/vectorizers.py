from __future__ import annotations

import math
from typing import Any

TRACK_VECTOR_SIZE = 16
PAIR_VECTOR_SIZE = 40


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _camelot(value: str | None) -> tuple[float, float, float]:
    if not value:
        return 0.0, 0.0, 1.0
    text = str(value).strip().upper()
    if len(text) < 2 or text[-1] not in {"A", "B"}:
        return 0.0, 0.0, 1.0
    try:
        number = int(text[:-1])
    except ValueError:
        return 0.0, 0.0, 1.0
    if not 1 <= number <= 12:
        return 0.0, 0.0, 1.0
    return number / 12.0, 1.0 if text[-1] == "B" else -1.0, 0.0


def _energy(track: dict[str, Any]) -> list[float]:
    return [float(value) for value in track.get("energy_curve", track.get("energyCurve", []))]


def _timbre(track: dict[str, Any]) -> list[float]:
    return [float(value) for value in track.get("timbre_vector", track.get("timbreVector", []))]


def _vocal(track: dict[str, Any]) -> list[float]:
    return [float(value) for value in track.get("vocal_activity", track.get("vocalActivity", []))]


def _beatgrid(track: dict[str, Any]) -> dict[str, Any]:
    return dict(track.get("beatgrid", {}))


def _beat_times(grid: dict[str, Any]) -> list[float]:
    values = grid.get("beatTimes", grid.get("beat_times", []))
    return [float(value) for value in values]


def _stem_availability(track: dict[str, Any]) -> float:
    manifest = track.get("stemManifest") or track.get("stem_manifest") or {}
    stems = manifest.get("stems", []) if isinstance(manifest, dict) else []
    if not stems:
        return 0.0
    available = [1.0 for stem in stems if stem.get("available")]
    return len(available) / len(stems)


def _cosine(a: list[float], b: list[float]) -> float:
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    dot = sum(a[index] * b[index] for index in range(length))
    norm_a = math.sqrt(sum(a[index] * a[index] for index in range(length)))
    norm_b = math.sqrt(sum(b[index] * b[index] for index in range(length)))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return _clamp(dot / (norm_a * norm_b), -1.0, 1.0)


def track_feature_vector(track: dict[str, Any]) -> list[float]:
    """Create a bounded vector for one track-level neural feature model."""

    bpm = float(track.get("bpm", 0.0) or 0.0)
    duration = float(track.get("duration_sec", track.get("durationSec", 0.0)) or 0.0)
    key_number, key_mode, key_unknown = _camelot(track.get("camelot"))
    energy = [_clamp(value, 0.0, 1.0) for value in _energy(track)]
    timbre = _timbre(track)
    vocal = [_clamp(value, 0.0, 1.0) for value in _vocal(track)]
    grid = _beatgrid(track)
    beat_times = _beat_times(grid)
    downbeats = [float(value) for value in grid.get("downbeats", [])]

    vector = [
        _clamp(bpm / 200.0, 0.0, 1.0),
        _clamp(duration / 600.0, 0.0, 1.0),
        key_number,
        key_mode,
        key_unknown,
        _mean(energy, 0.5),
        _clamp(_std(energy), 0.0, 1.0),
        energy[0] if energy else 0.5,
        energy[-1] if energy else 0.5,
        _clamp(_mean(timbre, 0.0) / 50.0, -1.0, 1.0),
        _clamp(_mean([abs(value) for value in timbre], 0.0) / 50.0, 0.0, 1.0),
        _mean(vocal, 0.0),
        max(vocal) if vocal else 0.0,
        _clamp(float(grid.get("confidence", 0.5)), 0.0, 1.0),
        _clamp(len(beat_times) / 1024.0, 0.0, 1.0),
        _clamp(len(downbeats) / 256.0 + _stem_availability(track) * 0.25, 0.0, 1.0),
    ]
    return [_clamp(value, -1.0, 1.0) for value in vector]


def pair_feature_vector(a: dict[str, Any], b: dict[str, Any]) -> list[float]:
    """Create a bounded vector for pairwise neural remix features."""

    va = track_feature_vector(a)
    vb = track_feature_vector(b)
    bpm_a = max(1e-6, float(a.get("bpm", 0.0) or 0.0))
    bpm_b = max(1e-6, float(b.get("bpm", 0.0) or 0.0))
    while bpm_b / bpm_a > 1.5:
        bpm_b /= 2
    while bpm_b / bpm_a < 0.67:
        bpm_b *= 2
    percent = ((bpm_b / bpm_a) - 1.0) * 100.0
    key_a = _camelot(a.get("camelot"))
    key_b = _camelot(b.get("camelot"))
    energy_diff = _mean(_energy(b), 0.5) - _mean(_energy(a), 0.5)
    vocal_overlap = _mean([x * y for x, y in zip(_vocal(a), _vocal(b))], 0.0)
    duration_a = max(1.0, float(a.get("duration_sec", a.get("durationSec", 0.0)) or 0.0))
    duration_b = max(1.0, float(b.get("duration_sec", b.get("durationSec", 0.0)) or 0.0))
    extra = [
        _clamp(percent / 50.0, -1.0, 1.0),
        1.0 if abs(percent) <= 2.0 else 0.0,
        1.0 if key_a[:2] == key_b[:2] and key_a[2] == 0.0 else 0.0,
        1.0 if key_a[0] == key_b[0] and key_a[1] != key_b[1] and key_a[2] == 0.0 else 0.0,
        _clamp(energy_diff, -1.0, 1.0),
        _cosine(_timbre(a), _timbre(b)),
        _clamp(vocal_overlap, 0.0, 1.0),
        _clamp((duration_b / duration_a) - 1.0, -1.0, 1.0),
    ]
    vector = va + vb + extra
    if len(vector) != PAIR_VECTOR_SIZE:
        raise AssertionError(f"pair vector size mismatch: {len(vector)}")
    return [_clamp(value, -1.0, 1.0) for value in vector]


TRACK_LEVEL_FEATURES = {"beatgrid_confidence", "stem_quality", "waveform_interest"}


def build_feature_input(
    feature_name: str,
    primary_track: dict[str, Any],
    secondary_track: dict[str, Any] | None = None,
) -> list[float]:
    if feature_name in TRACK_LEVEL_FEATURES or secondary_track is None:
        return track_feature_vector(primary_track)
    return pair_feature_vector(primary_track, secondary_track)
