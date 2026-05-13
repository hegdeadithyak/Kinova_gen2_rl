#!/usr/bin/env python3
import argparse
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

<<<<<<< HEAD
=======
# ── ChromaRefine (CCMR) — optional novel post-processing ─────────────────────
_CCMR_ENABLED = False

def _enable_ccmr():
    global _CCMR_ENABLED
    _CCMR_ENABLED = True

def _apply_ccmr_batch(bgr: np.ndarray, candidates: list) -> list:
    """
    Refine all candidate masks together so CCMR never expands one mask
    into territory already owned by another (prevents bleed between
    adjacent same-coloured items like two touching idlis).
    """
    if not _CCMR_ENABLED or not candidates:
        return candidates
    try:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent /
                               "Downloads" / "coco_food"))
        from chromarefine import ccmr_batch
        masks   = [c[0] for c in candidates]
        refined = ccmr_batch(bgr, masks)
        return [(refined[i], candidates[i][1], candidates[i][2])
                for i in range(len(candidates))]
    except Exception:
        return candidates

# ── YOLO backend (optional — used when --yolo is passed instead of --checkpoint) ──
def _load_yolo(weights_path: str, device: str = "cpu"):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    model.to(device)
    print(f"[food_segmenter] YOLO loaded: {weights_path} on {device}")
    return model


def _segment_yolo(model, bgr: np.ndarray):
    """
    Run the YOLO segmentation model and return candidates in the same format
    as _segment() (SAM backend): list of (mask, score, bbox) tuples.

    The model is trained with a single 'food' class, so it is class-agnostic —
    it segments any food region regardless of what it is, just like SAM.
    """
    h, w     = bgr.shape[:2]
    img_area = h * w

    results = model(bgr, verbose=False, conf=0.15, iou=0.4)[0]
    if results.masks is None:
        return []

    candidates: List[Tuple[np.ndarray, float, Tuple]] = []
    masks_data = results.masks.data.cpu().numpy()   # (N, H', W')
    boxes_data = results.boxes

    for i in range(len(masks_data)):
        # Bilinear upsample then threshold — smoother boundaries than INTER_NEAREST
        mask_small = masks_data[i]
        mask_f = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
        mask   = mask_f > 0.5

        ratio = float(mask.sum()) / img_area
        if not (MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
            continue
        if _is_background(mask):
            continue

        score = float(boxes_data.conf[i].item()) if boxes_data is not None else 0.8
        bbox  = _mask_bbox(mask)
        if bbox is None:
            continue
        candidates.append((mask, score, bbox))

    if not candidates:
        return []

    kept     = _nms([(m, s) for m, s, _ in candidates], NMS_IOU_THR)
    accepted = [candidates[i] for i in kept]
    accepted = _filter_containers(accepted)
    # CCMR: refine all masks together — each mask blocked from bleeding into neighbours
    accepted = _apply_ccmr_batch(bgr, accepted)
    accepted.sort(key=lambda c: (c[2][1], c[2][0]))
    return accepted

>>>>>>> 4ebc471 (food)
# ROS imports — only needed in live-camera mode
try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import Image
    from tf2_ros import Buffer, TransformListener
    from visualization_msgs.msg import Marker, MarkerArray
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False

# Camera intrinsics — calibrated values from click_pointer.py
FX, FY = 603.6312, 603.0632
CX, CY = 319.0870, 236.3678

CAM_FRAME  = "camera_color_optical_frame"
BASE_FRAME = "j2s6s200_link_base"

<<<<<<< HEAD
MIN_AREA_RATIO = 0.005   # ignore fragments smaller than 0.5% of image
=======
MIN_AREA_RATIO = 0.002   # ignore fragments smaller than 0.2% of image
>>>>>>> 4ebc471 (food)
MAX_AREA_RATIO = 0.60    # ignore large background regions
NMS_IOU_THR    = 0.80    # mask IoU threshold for duplicate suppression

# BGR colours for each detected food item in the overlay
_PALETTE = [
    (0, 255, 0), (255, 80, 0), (0, 128, 255),
    (255, 0, 255), (0, 255, 255), (128, 255, 0),
    (0, 200, 255), (200, 0, 255),
]


def _load_sam(checkpoint: str, device: str = "cpu"):
    """SAM2 preferred; SAM1 (segment_anything) as fallback — mirrors gist pattern."""
    from pathlib import Path
    ck = Path(checkpoint).expanduser().resolve()
    if not ck.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {ck}")

    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        sam = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(ck), device=device)
        gen = SAM2AutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.88,
            min_mask_region_area=20,
        )
        print(f"[food_segmenter] SAM2 loaded: {ck.name} on {device}")
        return gen
    except Exception as e:
        print(f"[food_segmenter] SAM2 unavailable ({e}), trying SAM1 fallback...")

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    mtype = "vit_b" if "vit_b" in str(ck) else ("vit_l" if "vit_l" in str(ck) else "vit_h")
    sam = sam_model_registry[mtype](checkpoint=str(ck))
    sam.to(device=device)
    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.88,
        min_mask_region_area=20,
    )
    print(f"[food_segmenter] SAM1 loaded: {ck.name} ({mtype}) on {device}")
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


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _nms(candidates: List[Tuple[np.ndarray, float]], iou_thr: float) -> List[int]:
    """Mask-IoU NMS. Returns indices of surviving candidates sorted by score (desc)."""
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


def _is_background(mask: np.ndarray) -> bool:
    """True if the mask looks like background — large region hugging image borders."""
    h, w = mask.shape
    border_pixels = (
        int(mask[0, :].sum()) + int(mask[-1, :].sum()) +
        int(mask[:, 0].sum()) + int(mask[:, -1].sum())
    )
    border_coverage = border_pixels / (2 * (h + w))
    area_ratio      = float(mask.sum()) / (h * w)
    # background covers a big chunk of the border AND a meaningful image area
    return border_coverage > 0.25 and area_ratio > 0.08


def _filter_containers(candidates: List[Tuple[np.ndarray, float, Tuple]]) -> List[Tuple]:
    """Remove masks that contain other masks — bowls, boxes, plates."""
    n = len(candidates)
    is_container = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            overlap = np.logical_and(candidates[i][0], candidates[j][0]).sum()
            # if >65% of mask-j lives inside mask-i, mask-i is a container
            if overlap / max(1, candidates[j][0].sum()) > 0.65:
                is_container[i] = True
                break
    return [c for k, c in enumerate(candidates) if not is_container[k]]


def pick_utensil(n_items: int) -> str:
    """Many discrete pieces → fork (fruits, cut idlis, chapati). Single mass → spoon."""
    return "fork" if n_items >= 2 else "spoon"


def classify_food(bgr_img: np.ndarray, mask: np.ndarray) -> str:
    """
    Classify food texture as solid / semi-solid / liquid.

    Uses Sobel gradient magnitude (edge sharpness) and HSV saturation
    variance. Liquids are smooth and uniform; solids are textured and
    colourful; semi-solids (rice, mash) are in between.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mean_grad = float(np.sqrt(gx**2 + gy**2)[mask].mean())

    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    sat_std = float(hsv[:, :, 1].astype(np.float32)[mask].std())

    x1, y1, x2, y2 = _mask_bbox(mask)
    fill_ratio = float(mask.sum()) / max(1, (x2 - x1) * (y2 - y1))

    if mean_grad < 12.0 and fill_ratio > 0.68:
        return "liquid"
    if mean_grad > 26.0 and sat_std > 18.0:
        return "solid"
    return "semi-solid"


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*y*y-2*z*z,   2*x*y-2*z*w,   2*x*z+2*y*w],
        [  2*x*y+2*z*w, 1-2*x*x-2*z*z,   2*y*z-2*x*w],
        [  2*x*z-2*y*w,   2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ])


def _pixel_to_base(u: int, v: int, depth: np.ndarray,
                   tf_buffer) -> Optional[np.ndarray]:
    """
    Pixel + aligned-depth → 3-D point in BASE_FRAME.
    Identical projection model to click_pointer.py.
    Uses a 9×9 patch median to reject noisy depth readings.
    """
    y0, y1 = max(0, v - 4), min(depth.shape[0], v + 5)
    x0, x1 = max(0, u - 4), min(depth.shape[1], u + 5)
    valid = depth[y0:y1, x0:x1]
    valid = valid[valid > 0]
    if len(valid) < 4:
        return None
    z = float(np.median(valid)) * 0.001     # mm → m
    if z < 0.05:
        return None

    x_cam = (u - CX) * z / FX
    y_cam = (v - CY) * z / FY
    cam_pt = np.array([x_cam, y_cam, z])

    try:
        tf = tf_buffer.lookup_transform(BASE_FRAME, CAM_FRAME, Time(),
                                        Duration(seconds=0.5))
        R = _quat_to_matrix(tf.transform.rotation)
        t = np.array([tf.transform.translation.x,
                      tf.transform.translation.y,
                      tf.transform.translation.z])
        return R @ cam_pt + t
    except Exception:
        return None


def _draw_and_label(bgr, overlay, mask, bbox, food_type, utensil, color):
    x1, y1, x2, y2 = bbox
    coloured = np.zeros_like(bgr)
    coloured[mask] = color
    overlay[:] = cv2.addWeighted(overlay, 1.0, coloured, 0.38, 0)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    label = f"{food_type}  {utensil}"
    cv2.putText(overlay, label, (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 3)
    cv2.putText(overlay, label, (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)


def _segment(sam, bgr: np.ndarray):
    """Run SAM and return filtered, NMS-deduplicated candidates."""
    h, w     = bgr.shape[:2]
    img_area = h * w
    raw = sam.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    candidates: List[Tuple[np.ndarray, float, Tuple]] = []
    for item in raw:
        mask = _ensure_mask(item.get("segmentation"), (h, w))
        if mask is None:
            continue
        ratio = float(mask.sum()) / img_area
        if not (MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
            continue
        if _is_background(mask):
            continue
        score = float(item.get("predicted_iou", item.get("pred_iou", 0.0)) or 0.0)
        bbox  = _mask_bbox(mask)
        if bbox is None:
            continue
        candidates.append((mask, score, bbox))

    if not candidates:
        return []

    kept     = _nms([(m, s) for m, s, _ in candidates], NMS_IOU_THR)
    accepted = [candidates[i] for i in kept]
    accepted = _filter_containers(accepted)
    accepted.sort(key=lambda c: (c[2][1], c[2][0]))
    return accepted


<<<<<<< HEAD
def _run_on_image(sam, image_path: str):
=======
def _run_on_image(model, image_path: str, use_yolo: bool = False):
>>>>>>> 4ebc471 (food)
    """Standalone mode: segment a single image file, no ROS required."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

<<<<<<< HEAD
    h, w = bgr.shape[:2]
    print(f"[food_segmenter] Running SAM on {image_path} ({w}x{h})...")
    t0       = time.monotonic()
    accepted = _segment(sam, bgr)
    print(f"[food_segmenter] SAM done ({time.monotonic()-t0:.1f}s) — "
=======
    h, w    = bgr.shape[:2]
    backend = "YOLO" if use_yolo else "SAM"
    print(f"[food_segmenter] Running {backend} on {image_path} ({w}x{h})...")
    t0       = time.monotonic()
    accepted = _segment_yolo(model, bgr) if use_yolo else _segment(model, bgr)
    print(f"[food_segmenter] {backend} done ({time.monotonic()-t0:.1f}s) — "
>>>>>>> 4ebc471 (food)
          f"{len(accepted)} item(s) after filtering")

    if not accepted:
        print("[food_segmenter] No food items detected.")
        return

<<<<<<< HEAD
    utensil = pick_utensil(len(accepted))
    overlay = bgr.copy()
    header  = f"  {'#':<3}  {'type':<12}  {'bbox (x1,y1,x2,y2)':<26}"
=======
    # Multiple discrete pieces → all solid, use fork.
    # Single mass/blob → texture-classify it, use spoon.
    is_multi  = len(accepted) >= 2
    utensil   = pick_utensil(len(accepted))
    overlay   = bgr.copy()
    header    = f"  {'#':<3}  {'type':<12}  {'bbox (x1,y1,x2,y2)':<26}"
>>>>>>> 4ebc471 (food)
    print(f"\n[food_segmenter] {len(accepted)} food item(s) → use {utensil.upper()}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, (mask, score, bbox) in enumerate(accepted):
<<<<<<< HEAD
        food_type = classify_food(bgr, mask)
=======
        food_type = "solid" if is_multi else classify_food(bgr, mask)
>>>>>>> 4ebc471 (food)
        x1, y1, x2, y2 = bbox
        color = _PALETTE[i % len(_PALETTE)]
        _draw_and_label(bgr, overlay, mask, bbox, food_type, utensil, color)
        print(f"  {i+1:<3}  {food_type:<12}  ({x1:4d},{y1:4d},{x2:4d},{y2:4d})")

    out_path = image_path.rsplit(".", 1)[0] + "_overlay.jpg"
    cv2.imwrite(out_path, overlay)
    print(f"\n[food_segmenter] Overlay saved → {out_path}")


if _ROS_AVAILABLE:
    class FoodSegmenterNode(Node):

<<<<<<< HEAD
        def __init__(self, sam, interval: float):
            super().__init__("food_segmenter")
            self._sam      = sam
=======
        def __init__(self, model, interval: float, use_yolo: bool = False):
            super().__init__("food_segmenter")
            self._model    = model
            self._use_yolo = use_yolo
>>>>>>> 4ebc471 (food)
            self._interval = interval

            self._lock      = threading.Lock()
            self._color_img: Optional[np.ndarray] = None
            self._depth_img: Optional[np.ndarray] = None
            self._busy      = False

            self._bridge    = CvBridge()
            self._tf_buffer = Buffer()
            TransformListener(self._tf_buffer, self)

            self.create_subscription(Image, "/camera/camera/color/image_raw",
                                     self._color_cb, 2)
            self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw",
                                     self._depth_cb, 2)

            self._marker_pub  = self.create_publisher(MarkerArray, "/food_segmenter/markers", 10)
            self._overlay_pub = self.create_publisher(Image, "/food_segmenter/overlay", 2)

            self.create_timer(interval, self._tick)
            self.get_logger().info(f"FoodSegmenter ready — running every {interval:.0f}s")

        def _color_cb(self, msg):
            with self._lock:
                self._color_img = self._bridge.imgmsg_to_cv2(msg, "bgr8")

        def _depth_cb(self, msg):
            with self._lock:
                self._depth_img = self._bridge.imgmsg_to_cv2(msg, "passthrough")

        def _tick(self):
            if self._busy:
                return
            with self._lock:
                color = self._color_img.copy() if self._color_img is not None else None
                depth = self._depth_img.copy() if self._depth_img is not None else None
            if color is None or depth is None:
                self.get_logger().warn("Waiting for camera frames...")
                return
            self._busy = True
            threading.Thread(target=self._run, args=(color, depth), daemon=True).start()

        def _run(self, bgr: np.ndarray, depth: np.ndarray):
            try:
                t_start  = time.monotonic()
<<<<<<< HEAD
                accepted = _segment(self._sam, bgr)
                self.get_logger().info(f"SAM: {len(accepted)} item(s) "
=======
                if self._use_yolo:
                    accepted = _segment_yolo(self._model, bgr)
                else:
                    accepted = _segment(self._model, bgr)
                backend = "YOLO" if self._use_yolo else "SAM"
                self.get_logger().info(f"{backend}: {len(accepted)} item(s) "
>>>>>>> 4ebc471 (food)
                                       f"({time.monotonic()-t_start:.1f}s)")

                if not accepted:
                    print("[FoodSegmenter] No food items detected.")
                    return

<<<<<<< HEAD
                utensil = pick_utensil(len(accepted))
                overlay = bgr.copy()
                markers = MarkerArray()
=======
                # Multiple discrete pieces → all solid, use fork.
                # Single mass/blob → texture-classify it, use spoon.
                is_multi = len(accepted) >= 2
                utensil  = pick_utensil(len(accepted))
                overlay  = bgr.copy()
                markers  = MarkerArray()
>>>>>>> 4ebc471 (food)

                print(f"\n[FoodSegmenter] {len(accepted)} food item(s) → use {utensil.upper()}")
                header = (f"  {'#':<3}  {'type':<12}  "
                          f"{'bbox (x1,y1,x2,y2)':<26}  3D centroid (m, base frame)")
                print(header)
                print("  " + "-" * (len(header) - 2))

                for i, (mask, score, bbox) in enumerate(accepted):
<<<<<<< HEAD
                    food_type = classify_food(bgr, mask)
=======
                    food_type = "solid" if is_multi else classify_food(bgr, mask)
>>>>>>> 4ebc471 (food)
                    x1, y1, x2, y2 = bbox
                    cx_px = (x1 + x2) // 2
                    cy_px = (y1 + y2) // 2
                    xyz   = _pixel_to_base(cx_px, cy_px, depth, self._tf_buffer)
                    color = _PALETTE[i % len(_PALETTE)]

                    _draw_and_label(bgr, overlay, mask, bbox, food_type, utensil, color)

                    if xyz is not None:
                        m = Marker()
                        m.header.frame_id    = BASE_FRAME
                        m.header.stamp       = self.get_clock().now().to_msg()
                        m.ns, m.id           = "food_items", i
                        m.type, m.action     = Marker.SPHERE, Marker.ADD
                        m.pose.position.x    = float(xyz[0])
                        m.pose.position.y    = float(xyz[1])
                        m.pose.position.z    = float(xyz[2])
                        m.pose.orientation.w = 1.0
                        m.scale.x = m.scale.y = m.scale.z = 0.04
                        m.color.r, m.color.g, m.color.b = [c / 255.0 for c in color]
                        m.color.a = 0.85
                        m.lifetime.sec = int(self._interval * 2)
                        markers.markers.append(m)
                        coord = f"({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f})"
                    else:
                        coord = "depth unavailable"

                    print(f"  {i+1:<3}  {food_type:<12}  "
                          f"({x1:4d},{y1:4d},{x2:4d},{y2:4d})    {coord}")

                print()
                if markers.markers:
                    self._marker_pub.publish(markers)
                self._overlay_pub.publish(self._bridge.cv2_to_imgmsg(overlay, "bgr8"))

            except Exception as e:
                self.get_logger().error(f"Segmentation run failed: {e}")
            finally:
                self._busy = False


def main():
<<<<<<< HEAD
    ap = argparse.ArgumentParser(description="SAM food segmenter for Kinova feeding task")
    ap.add_argument("--checkpoint", required=True,
                    help="SAM checkpoint path (.pt) — SAM2 small or SAM1 vit_b/vit_h")
=======
    ap = argparse.ArgumentParser(description="SAM / YOLO food segmenter for Kinova feeding task")
    # SAM backend
    ap.add_argument("--checkpoint", default=None,
                    help="SAM checkpoint path (.pt) — SAM2 small or SAM1 vit_b/vit_h")
    # YOLO backend (class-agnostic food segmenter, no class labels required)
    ap.add_argument("--yolo", default=None,
                    help="Path to trained YOLO segmentation weights (.pt). "
                         "Segments any food class-agnostically, like SAM.")
>>>>>>> 4ebc471 (food)
    ap.add_argument("--device",   default="cpu",  choices=["cpu", "cuda", "mps"])
    ap.add_argument("--interval", default=3.0,    type=float,
                    help="Seconds between segmentation passes (default 3)")
    ap.add_argument("--image",    default=None,
                    help="Path to a single image file — runs without ROS")
<<<<<<< HEAD
    known, ros_args = ap.parse_known_args()

    sam = _load_sam(known.checkpoint, known.device)

    if known.image:
        _run_on_image(sam, known.image)
=======
    ap.add_argument("--refine",   action="store_true",
                    help="Apply ChromaRefine CCMR post-processing to improve mask boundaries")
    known, ros_args = ap.parse_known_args()

    if not known.checkpoint and not known.yolo:
        ap.error("Provide either --checkpoint (SAM) or --yolo (YOLO weights).")

    if known.refine:
        _enable_ccmr()
        print("[food_segmenter] ChromaRefine CCMR enabled")

    use_yolo = bool(known.yolo)
    model = _load_yolo(known.yolo, known.device) if use_yolo else _load_sam(known.checkpoint, known.device)

    if known.image:
        _run_on_image(model, known.image, use_yolo=use_yolo)
>>>>>>> 4ebc471 (food)
        return

    if not _ROS_AVAILABLE:
        raise RuntimeError("ROS 2 is not available. Use --image for standalone mode.")

<<<<<<< HEAD
    rclpy.init(args=ros_args if ros_args else None)
    node = FoodSegmenterNode(sam, known.interval)
=======
    if use_yolo:
        # wrap YOLO so FoodSegmenterNode._segment call works transparently
        class _YOLOWrapper:
            def __init__(self, m): self._m = m
            def generate(self, rgb):
                # FoodSegmenterNode calls sam.generate() internally via _segment()
                # — not reached when use_yolo is True (node overrides _segment)
                raise NotImplementedError
        _node_model = model
    else:
        _node_model = model

    rclpy.init(args=ros_args if ros_args else None)
    node = FoodSegmenterNode(_node_model, known.interval, use_yolo=use_yolo)
>>>>>>> 4ebc471 (food)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
