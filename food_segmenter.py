#!/usr/bin/env python3
import argparse
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

# Camera intrinsics — calibrated values from click_pointer.py
FX, FY = 603.6312, 603.0632
CX, CY = 319.0870, 236.3678

CAM_FRAME  = "camera_color_optical_frame"
BASE_FRAME = "j2s6s200_link_base"

MIN_AREA_RATIO = 0.005   # ignore fragments smaller than 0.5% of image
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


def classify_food(bgr_img: np.ndarray, mask: np.ndarray) -> Tuple[str, str]:
    """
    Classify food region as solid / semi-solid / liquid from texture.

    Uses Sobel gradient magnitude (edge sharpness) and HSV saturation
    variance. Liquids are smooth and uniform; solids are textured and
    colourful; semi-solids (rice, mash) are in between.

    Returns (food_type, utensil) where utensil is 'fork' or 'spoon'.
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
        food_type = "liquid"
    elif mean_grad > 26.0 and sat_std > 18.0:
        food_type = "solid"
    else:
        food_type = "semi-solid"

    utensil = "fork" if food_type == "solid" else "spoon"
    return food_type, utensil


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*y*y-2*z*z,   2*x*y-2*z*w,   2*x*z+2*y*w],
        [  2*x*y+2*z*w, 1-2*x*x-2*z*z,   2*y*z-2*x*w],
        [  2*x*z-2*y*w,   2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ])


def _pixel_to_base(u: int, v: int, depth: np.ndarray,
                   tf_buffer: Buffer) -> Optional[np.ndarray]:
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


class FoodSegmenterNode(Node):

    def __init__(self, sam, interval: float):
        super().__init__("food_segmenter")
        self._sam      = sam
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
            h, w     = bgr.shape[:2]
            img_area = h * w
            t_start  = time.monotonic()

            raw = self._sam.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            self.get_logger().info(f"SAM produced {len(raw)} raw masks "
                                   f"({time.monotonic()-t_start:.1f}s)")

            candidates: List[Tuple[np.ndarray, float, Tuple]] = []
            for item in raw:
                mask = _ensure_mask(item.get("segmentation"), (h, w))
                if mask is None:
                    continue
                ratio = float(mask.sum()) / img_area
                if not (MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
                    continue
                score = float(item.get("predicted_iou", item.get("pred_iou", 0.0)) or 0.0)
                bbox  = _mask_bbox(mask)
                if bbox is None:
                    continue
                candidates.append((mask, score, bbox))

            if not candidates:
                print("[FoodSegmenter] No food items detected.")
                return

            kept = _nms([(m, s) for m, s, _ in candidates], NMS_IOU_THR)
            accepted = [candidates[i] for i in kept]
            # Sort top-left → bottom-right for stable ordering
            accepted.sort(key=lambda c: (c[2][1], c[2][0]))

            overlay  = bgr.copy()
            markers  = MarkerArray()

            print(f"\n[FoodSegmenter] {len(accepted)} food item(s) detected:")
            header = f"  {'#':<3}  {'type':<12}  {'utensil':<7}  "
            header += f"{'bbox (x1,y1,x2,y2)':<26}  3D centroid (m, base frame)"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for i, (mask, score, bbox) in enumerate(accepted):
                food_type, utensil = classify_food(bgr, mask)
                x1, y1, x2, y2    = bbox
                cx_px = (x1 + x2) // 2
                cy_px = (y1 + y2) // 2
                xyz   = _pixel_to_base(cx_px, cy_px, depth, self._tf_buffer)
                color = _PALETTE[i % len(_PALETTE)]

                # Overlay: translucent mask + solid border + label
                coloured = np.zeros_like(bgr)
                coloured[mask] = color
                overlay = cv2.addWeighted(overlay, 1.0, coloured, 0.38, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label   = f"{food_type}  {utensil}"
                cv2.putText(overlay, label, (x1, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 3)
                cv2.putText(overlay, label, (x1, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

                # RViz marker at 3-D centroid
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

                print(f"  {i+1:<3}  {food_type:<12}  {utensil:<7}  "
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
    ap = argparse.ArgumentParser(description="SAM food segmenter for Kinova feeding task")
    ap.add_argument("--checkpoint", required=True,
                    help="SAM checkpoint path (.pt) — SAM2 small or SAM1 vit_b/vit_h")
    ap.add_argument("--device",   default="cpu",  choices=["cpu", "cuda", "mps"])
    ap.add_argument("--interval", default=3.0,    type=float,
                    help="Seconds between segmentation passes (default 3)")
    known, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args if ros_args else None)
    sam  = _load_sam(known.checkpoint, known.device)
    node = FoodSegmenterNode(sam, known.interval)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
