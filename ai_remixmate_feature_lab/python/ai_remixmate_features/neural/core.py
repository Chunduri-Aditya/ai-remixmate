from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def stable_seed(text: str) -> int:
    total = 17
    for index, char in enumerate(text):
        total = (total * 31 + (index + 1) * ord(char)) % 2_147_483_647
    return total


@dataclass(slots=True)
class NeuralFeatureSpec:
    """Shape and training metadata for one adaptive feature model."""

    name: str
    input_size: int
    hidden_size: int = 12
    output_size: int = 1
    learning_rate: float = 0.05
    description: str = ""
    version: int = 1

    def validate(self) -> None:
        if not self.name:
            raise ValueError("feature model name is required")
        if not 1 <= self.input_size <= 128:
            raise ValueError("input_size must be in [1, 128]")
        if not 1 <= self.hidden_size <= 128:
            raise ValueError("hidden_size must be in [1, 128]")
        if not 1 <= self.output_size <= 16:
            raise ValueError("output_size must be in [1, 16]")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")


@dataclass(slots=True)
class TrainingEvent:
    """One explicit supervised learning event.

    The event is intentionally small and bounded so the model never learns from
    hidden background state. Callers decide when feedback is trustworthy enough
    to become a training event.
    """

    id: str
    feature_name: str
    input_vector: list[float]
    target_vector: list[float]
    source: str = "user_feedback"
    created_at: float = field(default_factory=time.time)
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, input_size: int, output_size: int) -> None:
        if not self.id:
            raise ValueError("training event id is required")
        if not self.feature_name:
            raise ValueError("feature_name is required")
        if len(self.input_vector) != input_size:
            raise ValueError(f"input_vector must have {input_size} values")
        if len(self.target_vector) != output_size:
            raise ValueError(f"target_vector must have {output_size} values")
        for value in self.input_vector:
            if not -1 <= value <= 1:
                raise ValueError("input_vector values must be in [-1, 1]")
        for value in self.target_vector:
            if not 0 <= value <= 1:
                raise ValueError("target_vector values must be in [0, 1]")
        if self.weight < 0:
            raise ValueError("weight must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = float(self.created_at)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingEvent":
        return cls(
            id=str(data["id"]),
            feature_name=str(data["feature_name"]),
            input_vector=[float(v) for v in data["input_vector"]],
            target_vector=[float(v) for v in data["target_vector"]],
            source=str(data.get("source", "user_feedback")),
            created_at=float(data.get("created_at", time.time())),
            weight=float(data.get("weight", 1.0)),
            metadata=dict(data.get("metadata", {})),
        )


class TinyOnlineMLP:
    """Small one-hidden-layer neural network for bounded online updates.

    This class is intentionally dependency-free. It is not a replacement for a
    production PyTorch model; it gives the lab a testable feedback loop and JSON
    persistence contract.
    """

    def __init__(
        self,
        spec: NeuralFeatureSpec,
        *,
        w1: list[list[float]] | None = None,
        b1: list[float] | None = None,
        w2: list[list[float]] | None = None,
        b2: list[float] | None = None,
        examples_seen: int = 0,
        last_loss: float = 0.0,
    ) -> None:
        spec.validate()
        self.spec = spec
        rng = random.Random(stable_seed(spec.name))
        limit1 = 1.0 / math.sqrt(spec.input_size)
        limit2 = 1.0 / math.sqrt(spec.hidden_size)
        self.w1 = w1 or [
            [rng.uniform(-limit1, limit1) for _ in range(spec.input_size)]
            for _ in range(spec.hidden_size)
        ]
        self.b1 = b1 or [0.0 for _ in range(spec.hidden_size)]
        self.w2 = w2 or [
            [rng.uniform(-limit2, limit2) for _ in range(spec.hidden_size)]
            for _ in range(spec.output_size)
        ]
        self.b2 = b2 or [0.0 for _ in range(spec.output_size)]
        self.examples_seen = int(examples_seen)
        self.last_loss = float(last_loss)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if len(self.w1) != self.spec.hidden_size:
            raise ValueError("w1 hidden dimension mismatch")
        if any(len(row) != self.spec.input_size for row in self.w1):
            raise ValueError("w1 input dimension mismatch")
        if len(self.b1) != self.spec.hidden_size:
            raise ValueError("b1 dimension mismatch")
        if len(self.w2) != self.spec.output_size:
            raise ValueError("w2 output dimension mismatch")
        if any(len(row) != self.spec.hidden_size for row in self.w2):
            raise ValueError("w2 hidden dimension mismatch")
        if len(self.b2) != self.spec.output_size:
            raise ValueError("b2 dimension mismatch")

    def _input(self, values: list[float]) -> list[float]:
        if len(values) != self.spec.input_size:
            raise ValueError(f"{self.spec.name} expects {self.spec.input_size} inputs")
        return [clamp(value, -1.0, 1.0) for value in values]

    def _target(self, values: list[float]) -> list[float]:
        if len(values) != self.spec.output_size:
            raise ValueError(f"{self.spec.name} expects {self.spec.output_size} targets")
        return [clamp(value, 0.0, 1.0) for value in values]

    def _forward_internal(self, values: list[float]) -> tuple[list[float], list[float]]:
        x = self._input(values)
        hidden = []
        for row, bias in zip(self.w1, self.b1):
            hidden.append(math.tanh(sum(weight * value for weight, value in zip(row, x)) + bias))
        outputs = []
        for row, bias in zip(self.w2, self.b2):
            outputs.append(sigmoid(sum(weight * value for weight, value in zip(row, hidden)) + bias))
        return hidden, outputs

    def predict(self, values: list[float]) -> list[float]:
        return self._forward_internal(values)[1]

    def train_one(self, values: list[float], target: list[float], *, weight: float = 1.0) -> float:
        x = self._input(values)
        y = self._target(target)
        event_weight = clamp(weight, 0.0, 10.0)
        hidden, outputs = self._forward_internal(x)
        loss = sum((out - truth) ** 2 for out, truth in zip(outputs, y)) / len(y)

        output_delta = [
            event_weight * (out - truth) * out * (1.0 - out)
            for out, truth in zip(outputs, y)
        ]
        old_w2 = [row[:] for row in self.w2]

        for output_index in range(self.spec.output_size):
            for hidden_index in range(self.spec.hidden_size):
                self.w2[output_index][hidden_index] -= (
                    self.spec.learning_rate * output_delta[output_index] * hidden[hidden_index]
                )
            self.b2[output_index] -= self.spec.learning_rate * output_delta[output_index]

        hidden_delta = []
        for hidden_index in range(self.spec.hidden_size):
            downstream = sum(
                output_delta[output_index] * old_w2[output_index][hidden_index]
                for output_index in range(self.spec.output_size)
            )
            hidden_delta.append((1.0 - hidden[hidden_index] ** 2) * downstream)

        for hidden_index in range(self.spec.hidden_size):
            for input_index in range(self.spec.input_size):
                self.w1[hidden_index][input_index] -= (
                    self.spec.learning_rate * hidden_delta[hidden_index] * x[input_index]
                )
            self.b1[hidden_index] -= self.spec.learning_rate * hidden_delta[hidden_index]

        self.examples_seen += 1
        self.last_loss = float(loss)
        return self.last_loss

    def train_event(self, event: TrainingEvent) -> float:
        event.validate(self.spec.input_size, self.spec.output_size)
        if event.feature_name != self.spec.name:
            raise ValueError(f"event is for {event.feature_name}, not {self.spec.name}")
        return self.train_one(event.input_vector, event.target_vector, weight=event.weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "inputSize": self.spec.input_size,
            "hiddenSize": self.spec.hidden_size,
            "outputSize": self.spec.output_size,
            "learningRate": self.spec.learning_rate,
            "version": self.spec.version,
            "examplesSeen": self.examples_seen,
            "lastLoss": self.last_loss,
            "weights": {
                "w1": self.w1,
                "b1": self.b1,
                "w2": self.w2,
                "b2": self.b2,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TinyOnlineMLP":
        spec = NeuralFeatureSpec(
            name=str(data["name"]),
            input_size=int(data["inputSize"]),
            hidden_size=int(data["hiddenSize"]),
            output_size=int(data["outputSize"]),
            learning_rate=float(data["learningRate"]),
            description=str(data.get("description", "")),
            version=int(data.get("version", 1)),
        )
        weights = dict(data.get("weights", {}))
        return cls(
            spec,
            w1=[[float(v) for v in row] for row in weights["w1"]],
            b1=[float(v) for v in weights["b1"]],
            w2=[[float(v) for v in row] for row in weights["w2"]],
            b2=[float(v) for v in weights["b2"]],
            examples_seen=int(data.get("examplesSeen", 0)),
            last_loss=float(data.get("lastLoss", 0.0)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TinyOnlineMLP":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
