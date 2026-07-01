from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BeatGrid:
    bpm: float
    beat_times: list[float]
    downbeats: list[float]
    beats_per_bar: int = 4
    confidence: float = 0.5
    source: str = "feature_lab"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CuePoint:
    id: str
    track_id: str
    timestamp_sec: float
    label: str
    type: str = "memory"
    color: str | None = None
    beat_index: int | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LoopState:
    active: bool
    start_sec: float
    end_sec: float
    length_beats: float
    quantized: bool = True
    source_cue_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrackAnalysis:
    id: str
    title: str
    duration_sec: float
    bpm: float
    beatgrid: BeatGrid
    artist: str | None = None
    key: str | None = None
    mode: str = "unknown"
    camelot: str | None = None
    key_confidence: float = 0.0
    energy_curve: list[float] = field(default_factory=list)
    timbre_vector: list[float] = field(default_factory=list)
    vocal_activity: list[float] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    cue_points: list[CuePoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CompatibilityScore:
    overall: float
    bpm_score: float
    key_score: float
    energy_score: float
    timbre_score: float
    vocal_clash_penalty: float
    explanation: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TransitionPlan:
    id: str
    from_track_id: str
    to_track_id: str
    entry_time_sec: float
    exit_time_sec: float
    transition_length_bars: int
    compatibility: CompatibilityScore
    eq_automation_notes: list[str] = field(default_factory=list)
    filter_automation_notes: list[str] = field(default_factory=list)
    cue_suggestions: list[CuePoint] = field(default_factory=list)
    loop_suggestions: list[LoopState] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RemixRecipe:
    id: str
    transition_plan_id: str
    method: str
    stem_suggestions: list[str]
    timing_suggestion: str
    risk_warnings: list[str]
    steps: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
