from __future__ import annotations

import math
from typing import Any


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_target_bpm(source_bpm: float, target_bpm: float) -> float:
    if source_bpm <= 0 or target_bpm <= 0:
        raise ValueError("BPM values must be positive")
    adjusted = float(target_bpm)
    while adjusted / source_bpm > 1.5:
        adjusted /= 2
    while adjusted / source_bpm < 0.67:
        adjusted *= 2
    return adjusted


def bpm_compatibility(source_bpm: float, target_bpm: float, max_safe_percent: float = 8.0) -> dict[str, Any]:
    normalized = _normalize_target_bpm(source_bpm, target_bpm)
    percent = ((normalized / source_bpm) - 1.0) * 100.0
    distance = abs(percent)
    score = _clamp01(1.0 - distance / 16.0)
    warnings = [] if distance <= max_safe_percent else [f"Tempo shift exceeds {max_safe_percent:.1f}%."]
    explanation = ["BPMs are effectively identical." if distance < 0.1 else f"Tempo shift is {percent:.2f}%."]
    if abs(normalized - target_bpm) > 0.01:
        explanation.append("Half/double BPM relation was normalized before scoring.")
    return {"score": score, "percentChange": percent, "normalizedTargetBpm": normalized, "safe": not warnings, "explanation": explanation, "warnings": warnings}


def _parse_camelot(value: str | None) -> tuple[int, str] | None:
    if not value:
        return None
    text = value.strip().upper()
    if len(text) < 2 or text[-1] not in {"A", "B"}:
        return None
    try:
        number = int(text[:-1])
    except ValueError:
        return None
    if not 1 <= number <= 12:
        return None
    return number, text[-1]


def _wrapped_delta(a: int, b: int) -> int:
    raw = abs(a - b)
    return min(raw, 12 - raw)


def key_compatibility(from_camelot: str | None, to_camelot: str | None) -> dict[str, Any]:
    a = _parse_camelot(from_camelot)
    b = _parse_camelot(to_camelot)
    if not a or not b:
        return {"score": 0.5, "relation": "unknown", "explanation": ["Unknown key; using neutral harmonic score."], "warnings": ["Missing or invalid Camelot key."]}
    if a == b:
        return {"score": 1.0, "relation": "same_key", "explanation": ["Same Camelot key."], "warnings": []}
    if a[0] == b[0] and a[1] != b[1]:
        return {"score": 0.92, "relation": "relative_major_minor", "explanation": ["Relative major/minor movement."], "warnings": []}
    if a[1] == b[1] and _wrapped_delta(a[0], b[0]) == 1:
        return {"score": 0.85, "relation": "adjacent", "explanation": ["Adjacent Camelot movement."], "warnings": []}
    if a[1] == b[1] and ((b[0] - a[0] + 12) % 12) == 2:
        return {"score": 0.72, "relation": "energy_boost", "explanation": ["Forward +2 Camelot energy boost."], "warnings": ["Energy boost movement is less neutral than same or adjacent key."]}
    return {"score": 0.35, "relation": "distant", "explanation": ["Distant Camelot movement."], "warnings": ["Potential harmonic clash."]}


def _sample(values: list[float], index: int, length: int) -> float:
    if not values:
        return 0.5
    if len(values) == 1 or length <= 1:
        return _clamp01(values[0])
    position = (index / (length - 1)) * (len(values) - 1)
    left = math.floor(position)
    right = min(len(values) - 1, math.ceil(position))
    frac = position - left
    return _clamp01(values[left] + (values[right] - values[left]) * frac)


def energy_compatibility(a: list[float], b: list[float]) -> dict[str, Any]:
    length = max(4, min(64, max(len(a), len(b))))
    distance = sum(abs(_sample(a, i, length) - _sample(b, i, length)) for i in range(length)) / length
    score = _clamp01(1.0 - distance)
    warnings = ["Energy curves differ strongly."] if score < 0.55 else []
    return {"score": score, "explanation": [f"Average normalized energy distance is {distance:.3f}."], "warnings": warnings}


def timbre_compatibility(a: list[float], b: list[float]) -> dict[str, Any]:
    length = min(len(a), len(b))
    if length == 0:
        return {"score": 0.5, "explanation": ["No timbre vectors; using neutral score."], "warnings": ["Missing timbre data."]}
    dot = sum(a[i] * b[i] for i in range(length))
    norm_a = math.sqrt(sum(a[i] * a[i] for i in range(length)))
    norm_b = math.sqrt(sum(b[i] * b[i] for i in range(length)))
    if norm_a <= 0 or norm_b <= 0:
        return {"score": 0.5, "explanation": ["Zero timbre vector; using neutral score."], "warnings": ["Invalid timbre vector."]}
    score = _clamp01((dot / (norm_a * norm_b) + 1.0) / 2.0)
    return {"score": score, "explanation": [f"Timbre cosine similarity mapped to {score:.3f}."], "warnings": []}


def vocal_clash_risk(a: list[float], b: list[float]) -> dict[str, Any]:
    length = min(len(a), len(b))
    if length == 0:
        return {"risk": 0.0, "warnings": []}
    risk = _clamp01(sum(_clamp01(a[i]) * _clamp01(b[i]) for i in range(length)) / length)
    warnings = ["Both tracks have active vocals in the same region."] if risk > 0.35 else []
    return {"risk": risk, "warnings": warnings}


def compatibility_score(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    bpm = bpm_compatibility(float(a["bpm"]), float(b["bpm"]))
    key = key_compatibility(a.get("camelot"), b.get("camelot"))
    energy = energy_compatibility(list(a.get("energy_curve", [])), list(b.get("energy_curve", [])))
    timbre = timbre_compatibility(list(a.get("timbre_vector", [])), list(b.get("timbre_vector", [])))
    vocal = vocal_clash_risk(list(a.get("vocal_activity", [])), list(b.get("vocal_activity", [])))
    overall = _clamp01(bpm["score"] * 0.30 + key["score"] * 0.25 + energy["score"] * 0.20 + timbre["score"] * 0.15 + (1.0 - vocal["risk"]) * 0.10)
    return {
        "overall": overall,
        "bpmScore": bpm["score"],
        "keyScore": key["score"],
        "energyScore": energy["score"],
        "timbreScore": timbre["score"],
        "vocalClashPenalty": vocal["risk"],
        "explanation": [*bpm["explanation"], *key["explanation"], *energy["explanation"], *timbre["explanation"]],
        "warnings": [*bpm["warnings"], *key["warnings"], *energy["warnings"], *timbre["warnings"], *vocal["warnings"]],
    }
