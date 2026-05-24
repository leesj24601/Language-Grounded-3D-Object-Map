"""Build a small semantic map from selected ARKitScenes frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.arkitscenes_adapter import ARKitScenesAdapter
from src.detector import GroundingDinoDetector
from src.projector import mask_to_world_projection
from src.segmentor import SamSegmentor
from src.semantic_map import ObjectObservation, SemanticMap, normalize_label


def parse_frame_indices(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", default="data/arkitscenes/3dod/Training/41098076")
    parser.add_argument("--frame-indices", default="250,300,350")
    parser.add_argument(
        "--caption",
        default="cabinet . chair . table . sofa . oven . refrigerator . washer . sink . tv monitor . stove .",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-mask-area", type=int, default=250)
    parser.add_argument("--association-distance-m", type=float, default=0.6)
    parser.add_argument("--out", default="outputs/gt_aligned_10_label/maps/41098076_semantic_map_demo.json")
    args = parser.parse_args()

    scene = ARKitScenesAdapter(args.scene_dir)
    detector = GroundingDinoDetector()
    segmentor = SamSegmentor()
    semantic_map = SemanticMap(association_distance_m=args.association_distance_m)

    for frame_index in parse_frame_indices(args.frame_indices):
        frame = scene[frame_index]
        image_source, detections = detector.predict_file(
            frame.rgb_path,
            args.caption,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        print(f"frame {frame_index} / {frame.frame_id}: detections={len(detections)}")
        if not detections:
            continue

        boxes_xyxy = np.stack([det.box_xyxy for det in detections], axis=0)
        masks = segmentor.predict_boxes(image_source, boxes_xyxy)
        for det, mask in zip(detections, masks):
            label = normalize_label(det.label)
            if label is None:
                print(f"  skip unknown label: {det.label}")
                continue
            if int(mask.sum()) < args.min_mask_area:
                print(f"  skip small mask: {det.label} area={int(mask.sum())}")
                continue
            try:
                projection = mask_to_world_projection(mask, frame.depth_m, frame.intrinsics, frame.t_cam_to_world)
            except ValueError as exc:
                print(f"  skip projection error: {det.label} {exc}")
                continue
            obj = semantic_map.add_observation(
                ObjectObservation(
                    label=label,
                    source_label=det.label,
                    centroid_m=projection.world_centroid,
                    confidence=det.score,
                    frame_id=frame.frame_id,
                    mask_area=int(mask.sum()),
                    box_xyxy=det.box_xyxy.tolist(),
                )
            )
            print(
                f"  {det.label:16s} -> {obj.object_id:16s} "
                f"score={det.score:.3f} centroid={projection.world_centroid.round(3).tolist()}"
            )

    output_path = Path(args.out)
    semantic_map.save_json(output_path)
    print(f"saved_map: {output_path}")
    print(f"object_count: {len(semantic_map.objects)}")


if __name__ == "__main__":
    main()
