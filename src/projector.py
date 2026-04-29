"""Depth unprojection and camera-to-world projection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectionResult:
    camera_points: np.ndarray
    world_points: np.ndarray
    camera_centroid: np.ndarray
    world_centroid: np.ndarray


def mask_to_camera_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    *,
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
    stride: int = 1,
) -> np.ndarray:
    """Unproject valid masked depth pixels into camera-frame XYZ points.

    Args:
        mask: Boolean-like image mask, same HxW as depth.
        depth_m: Depth image in meters.
        intrinsics: 3x3 camera matrix with fx/fy/cx/cy.
        min_depth_m: Minimum accepted depth in meters.
        max_depth_m: Maximum accepted depth in meters.
        stride: Optional pixel subsampling stride.

    Returns:
        Array of shape (N, 3), where each row is (x, y, z) in camera frame.
    """
    if mask.shape != depth_m.shape:
        raise ValueError(f"mask shape {mask.shape} must match depth shape {depth_m.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {intrinsics.shape}")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    mask_bool = mask.astype(bool)
    valid_depth = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    valid = mask_bool & valid_depth
    if stride > 1:
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[::stride, ::stride] = True
        valid &= stride_mask

    v, u = np.nonzero(valid)
    if v.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = depth_m[v, u].astype(np.float64)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    return np.column_stack((x, y, z))


def transform_points(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to 3D points."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if transform_4x4.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {transform_4x4.shape}")
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    points_h = np.ones((points.shape[0], 4), dtype=np.float64)
    points_h[:, :3] = points.astype(np.float64)
    transformed = points_h @ transform_4x4.astype(np.float64).T
    return transformed[:, :3]


def centroid(points: np.ndarray, *, method: str = "median") -> np.ndarray:
    """Compute a robust object centroid from projected points."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if points.size == 0:
        raise ValueError("cannot compute centroid of zero points")
    if method == "median":
        return np.median(points, axis=0)
    if method == "mean":
        return np.mean(points, axis=0)
    raise ValueError(f"unknown centroid method: {method}")


def mask_to_world_projection(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    t_cam_to_world: np.ndarray,
    *,
    centroid_method: str = "median",
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
    stride: int = 1,
) -> ProjectionResult:
    """Project a 2D object mask into camera/world points and centroids."""
    camera_points = mask_to_camera_points(
        mask,
        depth_m,
        intrinsics,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        stride=stride,
    )
    if camera_points.size == 0:
        raise ValueError("mask has no valid depth pixels")

    world_points = transform_points(camera_points, t_cam_to_world)
    return ProjectionResult(
        camera_points=camera_points,
        world_points=world_points,
        camera_centroid=centroid(camera_points, method=centroid_method),
        world_centroid=centroid(world_points, method=centroid_method),
    )
