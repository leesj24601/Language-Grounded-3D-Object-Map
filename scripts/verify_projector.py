"""Verify depth projection on a known case and one ARKitScenes frame."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.arkitscenes_adapter import ARKitScenesAdapter
from src.projector import mask_to_camera_points, mask_to_world_projection, transform_points


def verify_known_projection() -> None:
    intrinsics = np.asarray([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]])
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 2.0
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    camera_points = mask_to_camera_points(mask, depth, intrinsics)
    expected_camera = np.asarray([[0.0, 0.0, 2.0]])
    if not np.allclose(camera_points, expected_camera):
        raise AssertionError(f"known camera projection failed: {camera_points}")

    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    world_points = transform_points(camera_points, transform)
    expected_world = np.asarray([[1.0, 2.0, 5.0]])
    if not np.allclose(world_points, expected_world):
        raise AssertionError(f"known world projection failed: {world_points}")
    print("known_projection: ok")


def verify_scene_projection(scene_dir: str, frame_index: int) -> None:
    scene = ARKitScenesAdapter(scene_dir)
    frame = scene[frame_index]
    height, width = frame.depth_m.shape

    mask = np.zeros((height, width), dtype=bool)
    y0 = height // 2 - 20
    y1 = height // 2 + 20
    x0 = width // 2 - 20
    x1 = width // 2 + 20
    mask[y0:y1, x0:x1] = True

    result = mask_to_world_projection(mask, frame.depth_m, frame.intrinsics, frame.t_cam_to_world)
    print(f"scene_id: {scene.scene_id}")
    print(f"frame_id: {frame.frame_id}")
    print(f"mask_pixels: {int(mask.sum())}")
    print(f"valid_projected_points: {len(result.camera_points)}")
    print(f"camera_centroid_m: {np.round(result.camera_centroid, 4).tolist()}")
    print(f"world_centroid_m: {np.round(result.world_centroid, 4).tolist()}")
    print(
        "world_bounds_m: "
        f"min={np.round(result.world_points.min(axis=0), 4).tolist()} "
        f"max={np.round(result.world_points.max(axis=0), 4).tolist()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", default="data/arkitscenes/3dod/Training/41098076")
    parser.add_argument("--frame-index", type=int, default=300)
    args = parser.parse_args()

    verify_known_projection()
    verify_scene_projection(args.scene_dir, args.frame_index)


if __name__ == "__main__":
    main()
