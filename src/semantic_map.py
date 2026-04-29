"""Object-level semantic map with simple centroid association."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


BASE_LABELS = (
    "chair",
    "table",
    "sofa",
    "cabinet",
    "tv_monitor",
    "refrigerator",
    "sink",
    "stove",
    "oven",
    "washer",
)

LABEL_ALIASES = {
    "tv": "tv_monitor",
    "television": "tv_monitor",
    "monitor": "tv_monitor",
    "tv monitor": "tv_monitor",
    "couch": "sofa",
    "fridge": "refrigerator",
}


@dataclass
class ObjectObservation:
    label: str
    source_label: str
    centroid_m: np.ndarray
    confidence: float
    frame_id: str
    mask_area: int
    box_xyxy: list[float]


@dataclass
class Object3D:
    object_id: str
    label: str
    centroid_m: np.ndarray
    confidence_max: float
    observation_count: int = 0
    seen_frame_ids: list[str] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)

    def add(self, observation: ObjectObservation) -> None:
        count = self.observation_count
        self.centroid_m = (self.centroid_m * count + observation.centroid_m) / (count + 1)
        self.confidence_max = max(self.confidence_max, observation.confidence)
        self.observation_count += 1
        if observation.frame_id not in self.seen_frame_ids:
            self.seen_frame_ids.append(observation.frame_id)
        self.observations.append(
            {
                "frame_id": observation.frame_id,
                "source_label": observation.source_label,
                "confidence": observation.confidence,
                "centroid_m": observation.centroid_m.round(4).tolist(),
                "mask_area": observation.mask_area,
                "box_xyxy": [round(float(v), 3) for v in observation.box_xyxy],
            }
        )


class SemanticMap:
    def __init__(self, *, association_distance_m: float = 0.6) -> None:
        self.association_distance_m = association_distance_m
        self.objects: list[Object3D] = []
        self._next_id = 1

    def add_observation(self, observation: ObjectObservation) -> Object3D:
        match = self._find_match(observation)
        if match is None:
            match = Object3D(
                object_id=f"{observation.label}_{self._next_id:03d}",
                label=observation.label,
                centroid_m=observation.centroid_m.astype(float),
                confidence_max=observation.confidence,
            )
            self._next_id += 1
            self.objects.append(match)
        match.add(observation)
        return match

    def to_dict(self) -> dict[str, Any]:
        return {
            "association_distance_m": self.association_distance_m,
            "object_count": len(self.objects),
            "objects": [
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "centroid_m": obj.centroid_m.round(4).tolist(),
                    "confidence_max": round(float(obj.confidence_max), 4),
                    "observation_count": obj.observation_count,
                    "seen_frame_ids": obj.seen_frame_ids,
                    "observations": obj.observations,
                }
                for obj in self.objects
            ],
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def _find_match(self, observation: ObjectObservation) -> Object3D | None:
        candidates = [obj for obj in self.objects if obj.label == observation.label]
        if not candidates:
            return None
        distances = [
            float(np.linalg.norm(obj.centroid_m.astype(float) - observation.centroid_m.astype(float)))
            for obj in candidates
        ]
        best_idx = int(np.argmin(distances))
        if distances[best_idx] <= self.association_distance_m:
            return candidates[best_idx]
        return None


def normalize_label(raw_label: str, base_labels: tuple[str, ...] = BASE_LABELS) -> str | None:
    """Map Grounding DINO phrases into a stable semantic-map label."""
    normalized = raw_label.lower().replace("-", " ").replace("_", " ").strip()
    if normalized in LABEL_ALIASES:
        return LABEL_ALIASES[normalized]

    words = normalized.split()
    matches: list[str] = []
    for label in base_labels:
        label_text = label.replace("_", " ")
        if label_text == normalized or label_text in normalized or label in words:
            matches.append(label)
    if not matches:
        return None
    return matches[0]
