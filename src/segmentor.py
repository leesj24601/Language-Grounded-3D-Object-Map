"""SAM segmentation wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


class SamSegmentor:
    def __init__(
        self,
        checkpoint_path: str | Path = "models/sam/sam_vit_b_01ec64.pth",
        *,
        model_type: str = "vit_b",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    def predict_boxes(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Return one binary mask per xyxy box."""
        if boxes_xyxy.size == 0:
            return np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)
        self.predictor.set_image(image_rgb)
        boxes_t = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=self.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_t, image_rgb.shape[:2])
        masks_t, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks_t[:, 0].detach().cpu().numpy().astype(bool)
