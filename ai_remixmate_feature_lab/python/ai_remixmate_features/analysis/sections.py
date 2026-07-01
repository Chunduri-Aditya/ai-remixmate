from __future__ import annotations

from .energy_curve import extract_energy_curve


def estimate_sections(audio_path: str) -> dict:
    """Estimate coarse intro/body/outro sections from duration and energy.

    TODO: Replace fixed thirds with novelty and phrase-boundary segmentation.
    """
    energy = extract_energy_curve(audio_path)
    curve = energy["energyCurve"]
    if not curve:
        return {"audioPath": audio_path, "sections": []}
    total_buckets = len(curve)
    labels = ["intro", "body", "outro"]
    sections = []
    for idx, label in enumerate(labels):
        start = int((idx / 3) * total_buckets)
        end = int(((idx + 1) / 3) * total_buckets)
        segment = curve[start:end] or [0.0]
        sections.append({"label": label, "startBucket": start, "endBucket": end, "meanEnergy": float(sum(segment) / len(segment)), "confidence": 0.35})
    return {"audioPath": audio_path, "sections": sections}
