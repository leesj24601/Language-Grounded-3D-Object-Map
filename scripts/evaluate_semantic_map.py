"""Evaluate a semantic map JSON against ARKitScenes GT annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.arkitscenes_adapter import ARKitScenesAdapter
from src.evaluator import ObjectGroundTruth, load_predictions, match_by_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", default="data/arkitscenes/3dod/Training/41098076")
    parser.add_argument("--map", default="outputs/maps/41098076_semantic_map_20frames_text035.json")
    parser.add_argument("--out", default="outputs/metrics_41098076_text035.json")
    parser.add_argument("--distance-threshold-m", type=float, default=1.0)
    parser.add_argument("--min-observations", type=int, default=1)
    parser.add_argument(
        "--labels",
        default="chair,table,sofa,cabinet,tv_monitor,refrigerator,sink,stove,oven,washer",
        help="Comma-separated labels to evaluate.",
    )
    args = parser.parse_args()

    labels = {item.strip() for item in args.labels.split(",") if item.strip()}
    scene = ARKitScenesAdapter(args.scene_dir)
    gt_objects = [
        ObjectGroundTruth(uid=obj.uid, label=obj.label, centroid_m=np.asarray(obj.centroid, dtype=float))
        for obj in scene.load_gt_objects()
        if obj.label in labels
    ]
    predictions = load_predictions(args.map, min_observations=args.min_observations, labels=labels)

    result = match_by_label(
        predictions,
        gt_objects,
        distance_threshold_m=args.distance_threshold_m,
    )
    result["scene_id"] = scene.scene_id
    result["map_path"] = args.map
    result["min_observations"] = args.min_observations
    result["labels"] = sorted(labels)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"scene_id: {scene.scene_id}")
    print(f"map: {args.map}")
    print(f"min_observations: {args.min_observations}")
    print(f"predictions: {result['num_predictions']}")
    print(f"gt: {result['num_gt']}")
    print(f"matches: {result['num_matches']}")
    print(f"precision: {result['precision']}")
    print(f"recall: {result['recall']}")
    print(f"mean_l2_m: {result['mean_l2_m']}")
    print(f"median_l2_m: {result['median_l2_m']}")
    print(f"duplicate_rate: {result['duplicate_rate']}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
