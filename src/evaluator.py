"""Evaluate predicted object centroids against object-level GT centroids."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class ObjectPrediction:
    object_id: str
    label: str
    centroid_m: np.ndarray
    observation_count: int
    confidence_max: float


@dataclass(frozen=True)
class ObjectGroundTruth:
    uid: str
    label: str
    centroid_m: np.ndarray


def load_predictions(
    map_path: str | Path,
    *,
    min_observations: int = 1,
    labels: set[str] | None = None,
) -> list[ObjectPrediction]:
    data = json.load(open(map_path))
    predictions = []
    for item in data.get("objects", []):
        if item["observation_count"] < min_observations:
            continue
        if labels is not None and item["label"] not in labels:
            continue
        predictions.append(
            ObjectPrediction(
                object_id=item["object_id"],
                label=item["label"],
                centroid_m=np.asarray(item["centroid_m"], dtype=float),
                observation_count=int(item["observation_count"]),
                confidence_max=float(item["confidence_max"]),
            )
        )
    return predictions


def match_by_label(
    predictions: list[ObjectPrediction],
    gt_objects: list[ObjectGroundTruth],
    *,
    distance_threshold_m: float = 1.0,
) -> dict[str, Any]:
    labels = sorted({obj.label for obj in predictions} | {obj.label for obj in gt_objects})
    matches = []
    unmatched_predictions = []
    unmatched_gt = []

    for label in labels:
        pred_for_label = [obj for obj in predictions if obj.label == label]
        gt_for_label = [obj for obj in gt_objects if obj.label == label]
        if not pred_for_label:
            unmatched_gt.extend(gt_for_label)
            continue
        if not gt_for_label:
            unmatched_predictions.extend(pred_for_label)
            continue

        costs = np.zeros((len(pred_for_label), len(gt_for_label)), dtype=float)
        for i, pred in enumerate(pred_for_label):
            for j, gt in enumerate(gt_for_label):
                costs[i, j] = np.linalg.norm(pred.centroid_m - gt.centroid_m)

        row_ind, col_ind = linear_sum_assignment(costs)
        matched_pred_indices = set()
        matched_gt_indices = set()
        for i, j in zip(row_ind, col_ind):
            distance = float(costs[i, j])
            if distance <= distance_threshold_m:
                pred = pred_for_label[i]
                gt = gt_for_label[j]
                matches.append(
                    {
                        "label": label,
                        "prediction_id": pred.object_id,
                        "gt_uid": gt.uid,
                        "distance_m": round(distance, 4),
                        "pred_centroid_m": pred.centroid_m.round(4).tolist(),
                        "gt_centroid_m": gt.centroid_m.round(4).tolist(),
                        "observation_count": pred.observation_count,
                        "confidence_max": round(pred.confidence_max, 4),
                    }
                )
                matched_pred_indices.add(i)
                matched_gt_indices.add(j)

        unmatched_predictions.extend(
            pred for idx, pred in enumerate(pred_for_label) if idx not in matched_pred_indices
        )
        unmatched_gt.extend(gt for idx, gt in enumerate(gt_for_label) if idx not in matched_gt_indices)

    distances = [match["distance_m"] for match in matches]
    precision = len(matches) / len(predictions) if predictions else 0.0
    recall = len(matches) / len(gt_objects) if gt_objects else 0.0
    duplicate_rate = max(0, len(predictions) - len(matches)) / len(predictions) if predictions else 0.0

    return {
        "distance_threshold_m": distance_threshold_m,
        "num_predictions": len(predictions),
        "num_gt": len(gt_objects),
        "num_matches": len(matches),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mean_l2_m": round(float(np.mean(distances)), 4) if distances else None,
        "median_l2_m": round(float(np.median(distances)), 4) if distances else None,
        "duplicate_rate": round(duplicate_rate, 4),
        "matches": sorted(matches, key=lambda item: (item["label"], item["distance_m"])),
        "unmatched_predictions": [
            {
                "object_id": pred.object_id,
                "label": pred.label,
                "centroid_m": pred.centroid_m.round(4).tolist(),
                "observation_count": pred.observation_count,
                "confidence_max": round(pred.confidence_max, 4),
            }
            for pred in unmatched_predictions
        ],
        "unmatched_gt": [
            {
                "uid": gt.uid,
                "label": gt.label,
                "centroid_m": gt.centroid_m.round(4).tolist(),
            }
            for gt in unmatched_gt
        ],
    }
