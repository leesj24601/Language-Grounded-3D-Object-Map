"""Inspect a downloaded ARKitScenes 3DOD scene."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.arkitscenes_adapter import ARKitScenesAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-dir",
        default="data/arkitscenes/3dod/Training/41098076",
        help="Path to one ARKitScenes 3DOD scene directory.",
    )
    parser.add_argument("--frame-index", type=int, default=0)
    args = parser.parse_args()

    scene = ARKitScenesAdapter(args.scene_dir)
    frame = scene[args.frame_index]
    gt_objects = scene.load_gt_objects()
    label_counts = Counter(obj.label for obj in gt_objects)

    valid_depth = frame.depth_m[frame.depth_m > 0]
    print(f"scene_id: {scene.scene_id}")
    print(f"frames: {len(scene)}")
    print(f"frame_id: {frame.frame_id}")
    print(f"rgb_shape: {frame.rgb.shape} dtype={frame.rgb.dtype}")
    print(f"depth_shape: {frame.depth_m.shape} dtype={frame.depth_m.dtype}")
    print(
        "depth_m: "
        f"valid={valid_depth.size} "
        f"min={valid_depth.min():.3f} "
        f"median={np.median(valid_depth):.3f} "
        f"max={valid_depth.max():.3f}"
    )
    print(f"intrinsics:\n{frame.intrinsics}")
    print(f"t_cam_to_world:\n{frame.t_cam_to_world}")
    print(f"gt_objects: {len(gt_objects)}")
    print("gt_labels: " + ", ".join(f"{label}:{count}" for label, count in label_counts.most_common()))


if __name__ == "__main__":
    main()
