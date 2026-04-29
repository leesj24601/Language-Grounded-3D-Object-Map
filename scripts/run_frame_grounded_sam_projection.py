"""Run Grounding DINO + SAM + 3D projection on one ARKitScenes frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.arkitscenes_adapter import ARKitScenesAdapter
from src.detector import GroundingDinoDetector
from src.projector import mask_to_world_projection
from src.segmentor import SamSegmentor


def draw_overlay(
    image_rgb: np.ndarray,
    detections: list,
    masks: np.ndarray,
    output_path: Path,
) -> None:
    canvas = image_rgb.astype(np.float32).copy()
    colors = np.asarray(
        [
            [255, 64, 64],
            [64, 180, 255],
            [255, 200, 64],
            [160, 96, 255],
            [80, 220, 120],
            [255, 128, 32],
        ],
        dtype=np.float32,
    )
    for idx, mask in enumerate(masks):
        color = colors[idx % len(colors)]
        canvas[mask] = canvas[mask] * 0.55 + color * 0.45

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label = f"{det.label} {det.score:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", default="data/arkitscenes/3dod/Training/41098076")
    parser.add_argument("--frame-index", type=int, default=300)
    parser.add_argument(
        "--caption",
        default="chair . table . sofa . cabinet . tv monitor . refrigerator . sink . stove .",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = ARKitScenesAdapter(args.scene_dir)
    frame = scene[args.frame_index]
    prefix = f"{scene.scene_id}_{frame.frame_id.replace('.', '_')}"

    detector = GroundingDinoDetector()
    image_source, detections = detector.predict_file(
        frame.rgb_path,
        args.caption,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    print(f"scene_id: {scene.scene_id}")
    print(f"frame_id: {frame.frame_id}")
    print(f"caption: {args.caption}")
    print(f"detections: {len(detections)}")
    if not detections:
        return

    boxes_xyxy = np.stack([det.box_xyxy for det in detections], axis=0)
    segmentor = SamSegmentor()
    masks = segmentor.predict_boxes(image_source, boxes_xyxy)

    results = []
    for idx, (det, mask) in enumerate(zip(detections, masks)):
        item = {
            "label": det.label,
            "score": det.score,
            "box_xyxy": [round(float(v), 3) for v in det.box_xyxy],
            "mask_area": int(mask.sum()),
        }
        try:
            projection = mask_to_world_projection(mask, frame.depth_m, frame.intrinsics, frame.t_cam_to_world)
            item["camera_centroid_m"] = [round(float(v), 4) for v in projection.camera_centroid]
            item["world_centroid_m"] = [round(float(v), 4) for v in projection.world_centroid]
            item["projected_points"] = int(len(projection.world_points))
        except ValueError as exc:
            item["projection_error"] = str(exc)
        results.append(item)
        print(idx, item)

    overlay_path = out_dir / f"{prefix}_grounded_sam_projection.jpg"
    json_path = out_dir / f"{prefix}_detections_3d.json"
    draw_overlay(image_source, detections, masks, overlay_path)
    with json_path.open("w") as f:
        json.dump(
            {
                "scene_id": scene.scene_id,
                "frame_id": frame.frame_id,
                "caption": args.caption,
                "detections": results,
            },
            f,
            indent=2,
        )
    print(f"saved_overlay: {overlay_path}")
    print(f"saved_json: {json_path}")


if __name__ == "__main__":
    main()
