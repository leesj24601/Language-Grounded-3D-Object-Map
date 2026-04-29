"""Grounding DINO detection wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from groundingdino.util.inference import load_image, load_model, predict


@dataclass(frozen=True)
class Detection2D:
    label: str
    score: float
    box_cxcywh_norm: np.ndarray
    box_xyxy: np.ndarray


def boxes_cxcywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert normalized cx/cy/w/h boxes into pixel xyxy boxes."""
    scaled = boxes * torch.tensor([width, height, width, height], dtype=boxes.dtype)
    return torch.stack(
        [
            scaled[:, 0] - scaled[:, 2] / 2,
            scaled[:, 1] - scaled[:, 3] / 2,
            scaled[:, 0] + scaled[:, 2] / 2,
            scaled[:, 1] + scaled[:, 3] / 2,
        ],
        dim=1,
    )


class GroundingDinoDetector:
    def __init__(
        self,
        config_path: str | Path = "models/groundingdino/GroundingDINO_SwinT_OGC.py",
        checkpoint_path: str | Path = "models/groundingdino/groundingdino_swint_ogc.pth",
        *,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(str(config_path), str(checkpoint_path), device=self.device)

    def predict_file(
        self,
        image_path: str | Path,
        caption: str,
        *,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> tuple[np.ndarray, list[Detection2D]]:
        """Run Grounding DINO on an image file."""
        image_source, image = load_image(str(image_path))
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        height, width, _ = image_source.shape
        if len(boxes) == 0:
            return image_source, []

        boxes_xyxy = boxes_cxcywh_to_xyxy(boxes, width, height).detach().cpu().numpy()
        boxes_norm = boxes.detach().cpu().numpy()
        scores = logits.detach().cpu().numpy()
        detections = [
            Detection2D(
                label=str(label),
                score=float(score),
                box_cxcywh_norm=box_norm.astype(float),
                box_xyxy=box_xyxy_item.astype(float),
            )
            for label, score, box_norm, box_xyxy_item in zip(phrases, scores, boxes_norm, boxes_xyxy)
        ]
        return image_source, detections
