#!/usr/bin/env python3
"""
Intrinsic calibration using ArUco GridBoard (4x3, DICT_5X5_250).
Subscribes to /camera/camera/color/image_raw via ROS 2.
Press SPACE to capture, Q to quit and calibrate.
Need at least 20 good captures, aim for 50.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys, threading

# ── Board definition ──────────────────────────────────────────────────────────
MARKER_LENGTH   = 0.0690   # metres (measured from print)
MARKER_SEP      = 0.0069   # metres
BOARD_COLS      = 4
BOARD_ROWS      = 3

aruco_dict      = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
board           = cv2.aruco.GridBoard(
    size=(BOARD_COLS, BOARD_ROWS),
    markerLength=MARKER_LENGTH,
    markerSeparation=MARKER_SEP,
    dictionary=aruco_dict,
)
detector_params = cv2.aruco.DetectorParameters()
detector        = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

TARGET_CAPTURES = 50
MIN_MARKERS     = 4        # minimum markers visible per capture

# ── State ─────────────────────────────────────────────────────────────────────
all_corners  = []
all_ids      = []
img_size     = None
n_captured   = 0
latest_frame = None
frame_lock   = threading.Lock()
capture_flag = threading.Event()
quit_flag    = threading.Event()


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('aruco_intrinsic_calib')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_cb,
            10,
        )

    def image_cb(self, msg):
        global latest_frame
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with frame_lock:
            latest_frame = frame.copy()


def calibrate_and_save():
    global all_corners, all_ids, img_size

    print(f"\nCalibrating with {n_captured} frames...")

    obj_points, img_points = [], []
    for corners, ids in zip(all_corners, all_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is not None and len(obj_pts) >= MIN_MARKERS:
            obj_points.append(obj_pts)
            img_points.append(img_pts)

    if len(obj_points) < 10:
        print(f"Only {len(obj_points)} usable frames after filtering. Need at least 10.")
        return

    print(f"Using {len(obj_points)} frames for calibration...")
    flags = (
        cv2.CALIB_RATIONAL_MODEL        # 8-param distortion (k1-k6, p1, p2)
    )
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None, flags=flags
    )

    print(f"\n{'='*50}")
    print(f"Reprojection error: {ret:.4f} px")
    if ret > 1.0:
        print("WARNING: error > 1.0px — recapture with more varied poses")
    elif ret > 0.5:
        print("Acceptable — good enough for hand-eye calibration")
    else:
        print("Excellent calibration")

    print(f"\nK =\n{np.array2string(K, precision=4, suppress_small=True)}")
    print(f"\nD = {D.ravel()}")

    # ── Write camera_info.yaml ────────────────────────────────────────────────
    d = D.ravel()
    # Pad or truncate to 5 coefficients (plumb_bob: k1 k2 p1 p2 k3)
    d5 = list(d[:5]) + [0.0] * max(0, 5 - len(d))

    yaml_str = f"""image_width: {img_size[0]}
image_height: {img_size[1]}
camera_name: camera
camera_matrix:
  rows: 3
  cols: 3
  data: [{K[0,0]:.8f}, 0.0, {K[0,2]:.8f},
         0.0, {K[1,1]:.8f}, {K[1,2]:.8f},
         0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [{', '.join(f'{x:.8f}' for x in d5)}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [{K[0,0]:.8f}, 0.0, {K[0,2]:.8f}, 0.0,
         0.0, {K[1,1]:.8f}, {K[1,2]:.8f}, 0.0,
         0.0, 0.0, 1.0, 0.0]
"""
    out = "/tmp/camera_info.yaml"
    with open(out, "w") as f:
        f.write(yaml_str)
    print(f"\nSaved: {out}")
    print("\nNext step: copy to your ROS config and set camera_info_url in the RealSense launch.")


def display_loop():
    global latest_frame, n_captured, img_size

    cv2.namedWindow("ArUco Intrinsic Calibration", cv2.WINDOW_NORMAL)

    while not quit_flag.is_set():
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            if cv2.waitKey(30) & 0xFF == ord('q'):
                quit_flag.set()
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]  # (width, height)

        corners, ids, rejected = detector.detectMarkers(gray)
        display = frame.copy()

        n_visible = 0
        if ids is not None:
            n_visible = len(ids)
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            color = (0, 255, 0) if n_visible >= MIN_MARKERS else (0, 165, 255)
        else:
            color = (0, 0, 255)

        status = f"Markers: {n_visible} | Captured: {n_captured}/{TARGET_CAPTURES}"
        hint   = "SPACE=capture  Q=quit+calibrate" if n_visible >= MIN_MARKERS else "Move board — need >= 4 markers"

        cv2.putText(display, status, (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, hint,   (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Coverage guide overlay
        h, w = display.shape[:2]
        thirds_x = [w//3, 2*w//3]
        thirds_y = [h//3, 2*h//3]
        for x in thirds_x:
            cv2.line(display, (x, 0), (x, h), (60, 60, 60), 1)
        for y in thirds_y:
            cv2.line(display, (0, y), (w, y), (60, 60, 60), 1)

        if n_captured >= TARGET_CAPTURES:
            cv2.putText(display, "TARGET REACHED — press Q to calibrate",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("ArUco Intrinsic Calibration", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            if ids is not None and n_visible >= MIN_MARKERS:
                all_corners.append(corners)
                all_ids.append(ids)
                n_captured += 1
                print(f"[{n_captured:>3}/{TARGET_CAPTURES}] Captured — {n_visible} markers visible")
            else:
                print(f"Skipped — only {n_visible} markers visible (need {MIN_MARKERS})")

        elif key == ord('q'):
            quit_flag.set()

    cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = ImageSubscriber()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("="*50)
    print("ArUco Intrinsic Calibration")
    print(f"Board: {BOARD_COLS}x{BOARD_ROWS}, marker={MARKER_LENGTH*1000:.1f}mm, sep={MARKER_SEP*1000:.1f}mm")
    print(f"Target: {TARGET_CAPTURES} captures")
    print("="*50)
    print("\nCoverage guide — try to capture the board in ALL of these positions:")
    print("  - Center, flat facing camera")
    print("  - Left / Right / Top / Bottom of frame")
    print("  - Close (fills frame) / Far (board small)")
    print("  - Tilted left ~30° / right ~30° / up ~30° / down ~30°")
    print("  - Each corner of the frame, board tilted toward camera")
    print("\nPress SPACE to capture, Q to quit and run calibration.\n")

    display_loop()

    node.destroy_node()
    rclpy.shutdown()

    if n_captured >= 10:
        calibrate_and_save()
    else:
        print(f"Only {n_captured} captures — need at least 10. Rerun.")


if __name__ == "__main__":
    main()