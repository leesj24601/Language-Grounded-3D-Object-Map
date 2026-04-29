"""ARKitScenes 3DOD scene loader.

The adapter exposes RGB, depth, intrinsics, and camera-to-world pose in a
single frame object. Ground-truth annotations are loaded separately and should
only be used for evaluation or scene selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ARKitFrame:
    scene_id: str
    frame_id: str
    rgb_path: Path
    depth_path: Path
    intrinsics_path: Path
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    t_cam_to_world: np.ndarray


@dataclass(frozen=True)
class ARKitGTObject:
    uid: str
    label: str
    centroid: np.ndarray
    axes_lengths: np.ndarray
    normalized_axes: np.ndarray


class ARKitScenesAdapter:
    """Load one downloaded ARKitScenes 3DOD scene."""

    def __init__(self, scene_dir: str | Path) -> None:
        self.scene_dir = Path(scene_dir)
        self.scene_id = self.scene_dir.name
        self.frames_dir = self.scene_dir / f"{self.scene_id}_frames"
        self.rgb_dir = self.frames_dir / "lowres_wide"
        self.depth_dir = self.frames_dir / "lowres_depth"
        self.intrinsics_dir = self.frames_dir / "lowres_wide_intrinsics"
        self.traj_path = self.frames_dir / "lowres_wide.traj"
        self.annotation_path = self.scene_dir / f"{self.scene_id}_3dod_annotation.json"

        self._validate_scene()
        self._poses = self._load_poses()
        self.frame_ids = self._discover_frame_ids()

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __iter__(self) -> Iterable[ARKitFrame]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> ARKitFrame:
        frame_id = self.frame_ids[idx]
        filename = f"{self.scene_id}_{frame_id}.png"
        rgb_path = self.rgb_dir / filename
        depth_path = self.depth_dir / filename
        intrinsics_path = self._find_intrinsics_path(frame_id)

        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise FileNotFoundError(f"Could not read depth image: {depth_path}")
        depth_m = depth_mm.astype(np.float32) / 1000.0

        return ARKitFrame(
            scene_id=self.scene_id,
            frame_id=frame_id,
            rgb_path=rgb_path,
            depth_path=depth_path,
            intrinsics_path=intrinsics_path,
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=self._load_intrinsics(intrinsics_path),
            t_cam_to_world=self._find_pose(frame_id),
        )

    def load_gt_objects(self) -> list[ARKitGTObject]:
        """Load object-level GT boxes for evaluation only."""
        with self.annotation_path.open() as f:
            annotation = json.load(f)

        objects: list[ARKitGTObject] = []
        for item in annotation.get("data", []):
            obb = item.get("segments", {}).get("obbAligned") or item.get("segments", {}).get("obb")
            if not obb:
                continue
            objects.append(
                ARKitGTObject(
                    uid=item["uid"],
                    label=item["label"],
                    centroid=np.asarray(obb["centroid"], dtype=np.float32),
                    axes_lengths=np.asarray(obb["axesLengths"], dtype=np.float32),
                    normalized_axes=np.asarray(obb["normalizedAxes"], dtype=np.float32).reshape(3, 3),
                )
            )
        return objects

    def _validate_scene(self) -> None:
        required = [
            self.rgb_dir,
            self.depth_dir,
            self.intrinsics_dir,
            self.traj_path,
            self.annotation_path,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing ARKitScenes files:\n" + "\n".join(missing))

    def _discover_frame_ids(self) -> list[str]:
        frame_ids = []
        for depth_path in sorted(self.depth_dir.glob(f"{self.scene_id}_*.png")):
            frame_ids.append(depth_path.stem.split("_", 1)[1])
        if not frame_ids:
            raise FileNotFoundError(f"No depth frames found in {self.depth_dir}")
        return frame_ids

    def _load_poses(self) -> dict[str, np.ndarray]:
        poses: dict[str, np.ndarray] = {}
        with self.traj_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                tokens = line.split()
                if len(tokens) != 7:
                    raise ValueError(f"Invalid trajectory row: {line}")
                timestamp = f"{float(tokens[0]):.3f}"
                angle_axis = np.asarray([float(x) for x in tokens[1:4]], dtype=np.float64)
                translation = np.asarray([float(x) for x in tokens[4:7]], dtype=np.float64)

                t_world_to_cam = np.eye(4, dtype=np.float64)
                t_world_to_cam[:3, :3] = Rotation.from_rotvec(angle_axis).as_matrix()
                t_world_to_cam[:3, 3] = translation
                poses[timestamp] = np.linalg.inv(t_world_to_cam)
        if not poses:
            raise ValueError(f"No poses found in {self.traj_path}")
        return poses

    def _find_pose(self, frame_id: str) -> np.ndarray:
        return self._find_by_timestamp(frame_id, self._poses)

    def _find_intrinsics_path(self, frame_id: str) -> Path:
        candidates = {
            f"{float(frame_id):.3f}": self.intrinsics_dir / f"{self.scene_id}_{float(frame_id):.3f}.pincam",
            f"{float(frame_id) - 0.001:.3f}": self.intrinsics_dir
            / f"{self.scene_id}_{float(frame_id) - 0.001:.3f}.pincam",
            f"{float(frame_id) + 0.001:.3f}": self.intrinsics_dir
            / f"{self.scene_id}_{float(frame_id) + 0.001:.3f}.pincam",
        }
        for path in candidates.values():
            if path.exists():
                return path

        target = float(frame_id)
        nearby = sorted(
            self.intrinsics_dir.glob(f"{self.scene_id}_*.pincam"),
            key=lambda p: abs(float(p.stem.split("_", 1)[1]) - target),
        )
        if nearby and abs(float(nearby[0].stem.split("_", 1)[1]) - target) < 0.005:
            return nearby[0]
        raise FileNotFoundError(f"No intrinsics file close to timestamp {frame_id}")

    @staticmethod
    def _load_intrinsics(path: Path) -> np.ndarray:
        width, height, fx, fy, cx, cy = np.loadtxt(path, dtype=np.float64)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid intrinsics file: {path}")
        return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def _find_by_timestamp(frame_id: str, values: dict[str, np.ndarray]) -> np.ndarray:
        rounded = f"{float(frame_id):.3f}"
        if rounded in values:
            return values[rounded].copy()

        target = float(frame_id)
        key = min(values, key=lambda candidate: abs(float(candidate) - target))
        if abs(float(key) - target) < 0.005:
            return values[key].copy()
        raise KeyError(f"No trajectory pose close to timestamp {frame_id}")
