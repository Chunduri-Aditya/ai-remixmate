from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .core import TinyOnlineMLP, TrainingEvent
from .feature_registry import FEATURE_MODEL_SPECS, create_model_registry


class LearningController:
    """Owns adaptive feature models and applies explicit training events."""

    def __init__(self, models: dict[str, TinyOnlineMLP] | None = None) -> None:
        self.models = models or create_model_registry()
        self.event_count = 0

    def predict(self, feature_name: str, input_vector: list[float]) -> list[float]:
        return self._model(feature_name).predict(input_vector)

    def learn(self, event: TrainingEvent) -> dict[str, Any]:
        model = self._model(event.feature_name)
        loss = model.train_event(event)
        self.event_count += 1
        return {
            "featureName": event.feature_name,
            "loss": loss,
            "examplesSeen": model.examples_seen,
            "prediction": model.predict(event.input_vector),
        }

    def learn_score(
        self,
        feature_name: str,
        input_vector: list[float],
        target_score: float,
        *,
        event_id: str | None = None,
        source: str = "user_feedback",
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = TrainingEvent(
            id=event_id or f"{feature_name}-{int(time.time() * 1000)}",
            feature_name=feature_name,
            input_vector=input_vector,
            target_vector=[max(0.0, min(1.0, float(target_score)))],
            source=source,
            weight=weight,
            metadata=metadata or {},
        )
        return self.learn(event)

    def learn_many(self, events: list[TrainingEvent]) -> list[dict[str, Any]]:
        return [self.learn(event) for event in events]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "1.0",
            "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "eventCount": self.event_count,
            "models": {name: model.to_dict() for name, model in sorted(self.models.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningController":
        models = {
            name: TinyOnlineMLP.from_dict(model_data)
            for name, model_data in dict(data.get("models", {})).items()
        }
        controller = cls(models=models or create_model_registry())
        controller.event_count = int(data.get("eventCount", 0))
        return controller

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LearningController":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def append_event_log(self, path: str | Path, event: TrainingEvent) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

    def _model(self, feature_name: str) -> TinyOnlineMLP:
        if feature_name not in self.models:
            if feature_name in FEATURE_MODEL_SPECS:
                self.models[feature_name] = TinyOnlineMLP(FEATURE_MODEL_SPECS[feature_name])
            else:
                raise KeyError(f"unknown neural feature model: {feature_name}")
        return self.models[feature_name]
