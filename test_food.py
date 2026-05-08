#!/usr/bin/env python3
"""
Offline food segmentation test — no ROS2 required.
Runs SAM on a JPG/PNG image, classifies each food region,
and prints bounding boxes + fork/spoon recommendation.

Usage:
  python3 test_food.py --image food.jpg --checkpoint ~/sam_checkpoints/sam_vit_b_01ec64.pth
  python3 test_food.py --image food.jpg --checkpoint sam2.1_hiera_small.pt --save result.jpg
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 0.60
NMS_IOU_THR    = 0.80

_PALETTE = [
    (0, 255, 0), (255, 80, 0), (0, 128, 255),
    (255, 0, 255), (0, 255, 255), (128, 255, 0),
    (0, 200, 255), (200, 0, 255),
]


def load_sam(checkpoint: str, device: str = "cpu"):
    ck = Path(checkpoint).expanduser().resolve()
    if not ck.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ck}")

    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        sam = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(ck), device=device)
        gen = SAM2AutomaticMaskGenerator(
            model=sam,
            points_per_side=8,          # 64 decode passes instead of 1024 — CPU-viable
            pred_iou_thresh=0.75,
            stability_score_thresh=0.85,
            crop_n_layers=0,            # no multi-scale crops; halves run time
            min_mask_region_area=100,
        )
        print(f"SAM2 loaded: {ck.name} on {device}")
        return gen
    except Exception as e:
        print(f"SAM2 unavailable ({e}), trying SAM1...")

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    mtype = "vit_b" if "vit_b" in str(ck) else ("vit_l" if "vit_l" in str(ck) else "vit_h")
    sam = sam_model_registry[mtype](checkpoint=str(ck))
    sam.to(device=device)
    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.85,
        min_mask_region_area=100,
    )
    print(f"SAM1 loaded: {ck.name} ({mtype}) on {device}")
    return gen


def _ensure_mask(raw, hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if raw is None:
        return None
    arr = np.squeeze(np.asarray(raw))
    if arr.ndim != 2:
        return None
    h, w = hw
    if arr.shape != (h, w):
        arr = cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    arr = arr.astype(bool)
    return arr if arr.any() else None


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _nms(candidates: List[Tuple[np.ndarray, float]], iou_thr: float) -> List[int]:
    order = sorted(range(len(candidates)), key=lambda i: candidates[i][1], reverse=True)
    kept: List[int] = []
    for i in order:
        m_i = candidates[i][0]
        suppressed = any(
            (np.logical_and(m_i, candidates[j][0]).sum() /
             max(1, np.logical_or(m_i, candidates[j][0]).sum())) >= iou_thr
            for j in kept
        )
        if not suppressed:
            kept.append(i)
    return kept


def classify_food(bgr: np.ndarray, mask: np.ndarray) -> Tuple[str, str]:
    """
    solid     → high gradient + colour variation     → fork
    semi-solid → moderate gradient (rice, mash, etc) → spoon
    liquid     → smooth, fills bbox evenly           → spoon
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mean_grad = float(np.sqrt(gx**2 + gy**2)[mask].mean())

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat_std = float(hsv[:, :, 1].astype(np.float32)[mask].std())

    x1, y1, x2, y2 = _mask_bbox(mask)
    fill_ratio = float(mask.sum()) / max(1, (x2 - x1) * (y2 - y1))

    if mean_grad < 12.0 and fill_ratio > 0.68:
        food_type = "liquid"
    elif mean_grad > 26.0 and sat_std > 18.0:
        food_type = "solid"
    else:
        food_type = "semi-solid"

    utensil = "fork" if food_type == "solid" else "spoon"
    return food_type, utensil


def segment(bgr: np.ndarray, sam_gen) -> List[dict]:
    """Run SAM + filter + NMS. Returns list of result dicts."""
    h, w     = bgr.shape[:2]
    img_area = h * w

    import time
    print(f"Running SAM on {w}×{h} image...")
    t0  = time.monotonic()
    raw = sam_gen.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    print(f"SAM returned {len(raw)} raw masks  ({time.monotonic()-t0:.1f}s)")

    candidates: List[Tuple[np.ndarray, float, Tuple]] = []
    for item in raw:
        mask = _ensure_mask(item.get("segmentation"), (h, w))
        if mask is None:
            continue
        ratio = float(mask.sum()) / img_area
        if not (MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
            continue
        score = float(item.get("predicted_iou", item.get("pred_iou", 0.0)) or 0.0)
        candidates.append((mask, score, _mask_bbox(mask)))

    kept_idx = _nms([(m, s) for m, s, _ in candidates], NMS_IOU_THR)
    accepted = [candidates[i] for i in kept_idx]
    accepted.sort(key=lambda c: (c[2][1], c[2][0]))

    results = []
    for mask, score, bbox in accepted:
        food_type, utensil = classify_food(bgr, mask)
        results.append({
            "mask":      mask,
            "bbox":      bbox,
            "score":     score,
            "food_type": food_type,
            "utensil":   utensil,
        })
    return results


def draw_overlay(bgr: np.ndarray, results: List[dict]) -> np.ndarray:
    overlay = bgr.copy()
    for i, r in enumerate(results):
        mask  = r["mask"]
        x1, y1, x2, y2 = r["bbox"]
        color = _PALETTE[i % len(_PALETTE)]

        coloured = np.zeros_like(bgr)
        coloured[mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, coloured, 0.38, 0)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"{i+1}. {r['food_type']}  |  {r['utensil']}"
        cv2.putText(overlay, label, (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        # Centroid dot
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(overlay, (cx, cy), 4, (255, 255, 255), -1)
        cv2.circle(overlay, (cx, cy), 4, color, 1)

    return overlay


def print_results(results: List[dict]):
    print(f"\nDetected {len(results)} food item(s):\n")
    print(f"  {'#':<4} {'type':<13} {'utensil':<8} {'bbox (x1,y1,x2,y2)':<28} score")
    print("  " + "-" * 62)
    for i, r in enumerate(results):
        x1, y1, x2, y2 = r["bbox"]
        print(f"  {i+1:<4} {r['food_type']:<13} {r['utensil']:<8} "
              f"({x1:4d}, {y1:4d}, {x2:4d}, {y2:4d})       {r['score']:.3f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",      required=True,  help="Input image path (JPG/PNG)")
    ap.add_argument("--checkpoint", required=True,  help="SAM checkpoint (.pt)")
    ap.add_argument("--device",     default="cpu",  choices=["cpu", "cuda", "mps"])
    ap.add_argument("--save",       default=None,   help="Save overlay to this path")
    ap.add_argument("--no-display", action="store_true", help="Skip cv2.imshow window")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    print(f"Image: {args.image}  ({bgr.shape[1]}×{bgr.shape[0]})")

    sam = load_sam(args.checkpoint, args.device)
    results = segment(bgr, sam)

    if not results:
        print("No food items detected. Try --device cuda or a different checkpoint.")
        return

    print_results(results)
    overlay = draw_overlay(bgr, results)

    if args.save:
        cv2.imwrite(args.save, overlay)
        print(f"Overlay saved → {args.save}")

    if not args.no_display:
        cv2.imshow("Food Segmentation", overlay)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
