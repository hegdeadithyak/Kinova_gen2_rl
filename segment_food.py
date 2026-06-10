#!/usr/bin/env python3
"""
Food segmentation CLI.

Usage:
  python3 segment_food.py <image_path> [--conf 0.25] [--weights path/to/best.pt]
"""
import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, '/home/amma/coco_food')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image', help='Path to input image')
    ap.add_argument('--conf',    type=float, default=0.25,
                    help='Confidence threshold (default 0.25)')
    ap.add_argument('--weights', default='/home/amma/coco_food/runs/yolo11n_cafg-3/weights/best.pt',
                    help='YOLO weights path')
    ap.add_argument('--out',     default=None,
                    help='Output path (default: <input>_seg.png)')
    args = ap.parse_args()

    import food_cafg  # noqa — must import before YOLO
    from ultralytics import YOLO

    img = cv2.imread(args.image)
    if img is None:
        print(f'ERROR: cannot read {args.image}')
        sys.exit(1)

    h, w = img.shape[:2]
    print(f'Image: {w}x{h}  |  conf={args.conf}  |  weights={args.weights}')

    model   = YOLO(args.weights)
    results = model(img, conf=args.conf, verbose=False)

    PALETTE = [(0,220,80),(255,100,0),(0,140,255),(220,0,220),(0,220,220),(255,200,0),(200,0,100)]
    out = img.copy()
    n   = 0

    r = results[0]
    if r.masks is not None:
        # Build candidate list
        candidates = []
        for mask_t, box in zip(r.masks.data, r.boxes):
            mask = cv2.resize(mask_t.cpu().numpy().astype(np.uint8),
                              (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            candidates.append((mask, float(box.conf)))

        # Filter containers: if >65% of mask-j lives inside mask-i, mask-i is a bowl/container
        n_c = len(candidates)
        is_container = [False] * n_c
        for i in range(n_c):
            for j in range(n_c):
                if i == j: continue
                overlap = np.logical_and(candidates[i][0], candidates[j][0]).sum()
                if overlap / max(1, candidates[j][0].sum()) > 0.65:
                    is_container[i] = True
                    break
        candidates = [(m, c) for k, (m, c) in enumerate(candidates) if not is_container[k]]
        print(f'After container filter: {len(candidates)} / {n_c} detections kept')

        for i, (mask, conf) in enumerate(candidates):
            col = PALETTE[i % len(PALETTE)]

            # Semi-transparent fill
            colored = np.zeros_like(out)
            colored[mask] = col
            out = cv2.addWeighted(out, 1.0, colored, 0.45, 0)

            # Contour
            m8 = mask.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, col, 2)

            # Centroid label
            M = cv2.moments(m8)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                label = f'item{i+1} {conf:.2f}'
                cv2.putText(out, label, (max(cx-40,0), max(cy,14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            n += 1
    else:
        print('No masks detected.')

    out_path = args.out or args.image.rsplit('.', 1)[0] + '_seg.png'
    cv2.imwrite(out_path, out)
    print(f'{n} item(s) detected  →  {out_path}')


if __name__ == '__main__':
    main()
