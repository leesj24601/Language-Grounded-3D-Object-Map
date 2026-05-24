from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from groundingdino.util.inference import load_image, load_model, predict
from segment_anything import SamPredictor, sam_model_registry


def boxes_cxcywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
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


def save_box_overlay(
    image_rgb: np.ndarray,
    boxes_xyxy: torch.Tensor,
    phrases: list[str],
    logits: torch.Tensor,
    output_path: Path,
) -> None:
    canvas = image_rgb.copy()
    for phrase, score, box in zip(phrases, logits, boxes_xyxy.detach().cpu().numpy()):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{phrase} {float(score):.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_mask_overlay(
    image_rgb: np.ndarray,
    boxes_xyxy: torch.Tensor,
    masks: np.ndarray,
    phrases: list[str],
    logits: torch.Tensor,
    output_path: Path,
) -> None:
    canvas = image_rgb.astype(np.float32).copy()
    colors = np.array(
        [
            [255, 64, 64],
            [64, 180, 255],
            [255, 200, 64],
            [160, 96, 255],
            [80, 220, 120],
        ],
        dtype=np.float32,
    )

    for idx, mask in enumerate(masks):
        color = colors[idx % len(colors)]
        canvas[mask] = canvas[mask] * 0.55 + color * 0.45

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    for phrase, score, box in zip(phrases, logits, boxes_xyxy.detach().cpu().numpy()):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label = f"{phrase} {float(score):.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/test/room.jpg")
    parser.add_argument("--caption", default="chair . table . sofa . couch .")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--out-dir", default="outputs/gt_aligned_10_label/figures")
    parser.add_argument("--prefix", default="room")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"image: {image_path}")
    print(f"caption: {args.caption}")

    model = load_model(
        "models/groundingdino/GroundingDINO_SwinT_OGC.py",
        "models/groundingdino/groundingdino_swint_ogc.pth",
        device=device,
    )
    image_source, image = load_image(str(image_path))
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=args.caption,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
    )

    print(f"detections: {len(boxes)}")
    for idx, (phrase, score, box) in enumerate(zip(phrases, logits, boxes)):
        rounded_box = [round(float(value), 4) for value in box]
        print(f"{idx}: {phrase} score={float(score):.3f} box_cxcywh={rounded_box}")

    height, width, _ = image_source.shape
    if len(boxes) == 0:
        empty = torch.empty((0, 4), dtype=torch.float32)
        save_box_overlay(
            image_source,
            empty,
            [],
            torch.empty(0),
            out_dir / f"{args.prefix}_groundingdino_boxes.jpg",
        )
        print("No detections; saved original image as box overlay.")
        return

    boxes_output = out_dir / f"{args.prefix}_groundingdino_boxes.jpg"
    masks_output = out_dir / f"{args.prefix}_grounded_sam_masks.jpg"

    boxes_xyxy = boxes_cxcywh_to_xyxy(boxes, width, height)
    save_box_overlay(
        image_source,
        boxes_xyxy,
        phrases,
        logits,
        boxes_output,
    )

    sam = sam_model_registry["vit_b"](checkpoint="models/sam/sam_vit_b_01ec64.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_source)

    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_xyxy.to(device),
        image_source.shape[:2],
    )
    masks_t, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    masks = masks_t[:, 0].detach().cpu().numpy()

    print(f"masks: {len(masks)}")
    for idx, mask in enumerate(masks):
        print(f"mask {idx}: area={int(mask.sum())}")

    save_mask_overlay(
        image_source,
        boxes_xyxy,
        masks,
        phrases,
        logits,
        masks_output,
    )
    print(f"saved: {boxes_output}")
    print(f"saved: {masks_output}")


if __name__ == "__main__":
    main()
