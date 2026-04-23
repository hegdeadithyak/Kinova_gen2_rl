#!/usr/bin/env python3
"""
mouth_detector_node.py  (MediaPipe >= 0.10 / Task API)
=======================================================
ROS2 Node: Detects patient mouth position using MediaPipe FaceLandmarker
(new Task API, mediapipe >= 0.10), then provides a /feed_trigger service.
When triggered, publishes the 3D mouth goal pose to /mouth_pose.

Subscribed Topics:
  /camera/color/image_raw        (sensor_msgs/Image)
  /camera/depth/image_rect_raw   (sensor_msgs/Image)
  /camera/color/camera_info      (sensor_msgs/CameraInfo)

Published Topics:
  /mouth_pose                    (geometry_msgs/PoseStamped)
  /mouth_detection/image         (sensor_msgs/Image)  — annotated debug view

Services:
  /feed_trigger                  (std_srvs/Trigger)

Usage:
  ros2 launch <your_package> feeding_system.launch.py
  ros2 service call /feed_trigger std_srvs/srv/Trigger {}
"""

import os
import urllib.request

import rclpy
from rclpy.node import Node

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

# ─────────────────────────────────────────────────────────────────────
# Mouth landmark indices (same topology across all mediapipe versions)
# 13  = upper lip top center
# 14  = lower lip bottom center
# 61  = left mouth corner
# 291 = right mouth corner
# 78  = inner left
# 308 = inner right
# ─────────────────────────────────────────────────────────────────────
MOUTH_INDICES = [13, 14, 61, 291, 78, 308]

# Lip outline connections for debug drawing
LIP_CONNECTIONS = [
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),
    (405,321),(321,375),(375,291),(61,185),(185,40),(40,39),(39,37),
    (37,0),(0,267),(267,269),(269,270),(270,409),(409,291),
    (78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),
    (402,318),(318,324),(324,308),(78,191),(191,80),(80,81),(81,82),
    (82,13),(13,312),(312,311),(311,310),(310,415),(415,308),
]

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.expanduser("~/.cache/mediapipe/face_landmarker.task")


def _ensure_model():
    """Download the FaceLandmarker .task model file if not already cached."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"[mouth_detector] Downloading FaceLandmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[mouth_detector] Download complete.")
    return MODEL_PATH


class MouthDetectorNode(Node):
    def __init__(self):
        super().__init__('mouth_detector_node')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('approach_offset_z',        0.05)
        self.declare_parameter('min_detection_confidence', 0.6)
        self.declare_parameter('min_tracking_confidence',  0.5)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')

        self.approach_offset_z = self.get_parameter('approach_offset_z').value
        self.camera_frame      = self.get_parameter('camera_frame').value
        det_conf = self.get_parameter('min_detection_confidence').value
        trk_conf = self.get_parameter('min_tracking_confidence').value

        # ── Build MediaPipe FaceLandmarker (Task API) ────────────────
        model_path = _ensure_model()

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        options   = FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=det_conf,
            min_face_presence_confidence=det_conf,
            min_tracking_confidence=trk_conf,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self.detector     = FaceLandmarker.create_from_options(options)
        self._frame_ts_ms = 0   # monotonic ms counter required by VIDEO mode

        # ── State ────────────────────────────────────────────────────
        self.bridge       = CvBridge()
        self.latest_rgb   = None
        self.latest_depth = None
        self.camera_info  = None
        self.mouth_pixel  = None
        self.mouth_3d     = None

        # ── Subscribers ──────────────────────────────────────────────
        self.create_subscription(Image,      '/camera/camera/color/image_raw',
                                 self.rgb_callback,   10)
        self.create_subscription(Image,      '/camera/camera/depth/image_rect_raw',
                                 self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self.info_callback,  10)

        # ── Publishers ───────────────────────────────────────────────
        self.mouth_pose_pub  = self.create_publisher(
            PoseStamped, '/mouth_pose',            10)
        self.debug_image_pub = self.create_publisher(
            Image,       '/mouth_detection/image', 10)

        # ── Service ──────────────────────────────────────────────────
        self.srv = self.create_service(
            Trigger, '/feed_trigger', self.feed_trigger_callback)

        # ── 15 Hz processing timer ───────────────────────────────────
        self.create_timer(1.0 / 15.0, self.process_frame)

        self.get_logger().info(
            '✅  MouthDetectorNode ready (mediapipe Task API). '
            'Call /feed_trigger to feed.')

    # ── ROS Callbacks ────────────────────────────────────────────────

    def rgb_callback(self, msg):   self.latest_rgb   = msg
    def depth_callback(self, msg): self.latest_depth = msg

    def info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info('Camera intrinsics received.')

    # ── Main Processing ──────────────────────────────────────────────

    def process_frame(self):
        if self.latest_rgb is None:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(
                self.latest_rgb, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'RGB conversion failed: {e}')
            return

        h, w = bgr.shape[:2]

        # Task API requires mp.Image in RGB
        rgb_np   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)

        # Monotonically increasing timestamp in ms
        self._frame_ts_ms += 67   # ~15 Hz

        result   = self.detector.detect_for_video(mp_image, self._frame_ts_ms)
        annotated = bgr.copy()

        if result.face_landmarks:
            lms = result.face_landmarks[0]   # first detected face

            # Draw all landmarks (faint dots)
            for lm in lms:
                cv2.circle(annotated,
                           (int(lm.x * w), int(lm.y * h)),
                           1, (160, 160, 160), -1)

            # Draw lip outline
            for a, b in LIP_CONNECTIONS:
                if a < len(lms) and b < len(lms):
                    p1 = (int(lms[a].x * w), int(lms[a].y * h))
                    p2 = (int(lms[b].x * w), int(lms[b].y * h))
                    cv2.line(annotated, p1, p2, (0, 200, 255), 1)

            # Mouth center
            valid_mouth = [i for i in MOUTH_INDICES if i < len(lms)]
            mx = np.mean([lms[i].x for i in valid_mouth])
            my = np.mean([lms[i].y for i in valid_mouth])
            u, v = int(mx * w), int(my * h)
            self.mouth_pixel = (u, v)

            cv2.circle(annotated, (u, v), 8,  (0, 255, 0),   -1)
            cv2.circle(annotated, (u, v), 14, (0, 200, 255),   2)
            cv2.putText(annotated, 'MOUTH TARGET',
                        (u + 16, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Back-project to 3D
            mouth_3d = self._pixel_to_3d(u, v)
            if mouth_3d is not None:
                self.mouth_3d = mouth_3d
                x, y, z = mouth_3d
                cv2.putText(annotated,
                            f'3D  x={x:.3f} y={y:.3f} z={z:.3f} m',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 200), 2)
            else:
                cv2.putText(annotated, 'No depth at mouth',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 165, 255), 2)
        else:
            self.mouth_pixel = None
            self.mouth_3d    = None
            cv2.putText(annotated, 'NO FACE DETECTED',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        try:
            dbg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            dbg.header = self.latest_rgb.header
            self.debug_image_pub.publish(dbg)
        except Exception as e:
            self.get_logger().warn(f'Debug publish failed: {e}')

    # ── Depth back-projection ─────────────────────────────────────────

    def _pixel_to_3d(self, u, v):
        if self.latest_depth is None or self.camera_info is None:
            return None
        try:
            depth_img = self.bridge.imgmsg_to_cv2(
                self.latest_depth, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Depth conversion: {e}')
            return None

        dh, dw = depth_img.shape[:2]
        r = 5
        patch = depth_img[
            max(0, v-r):min(dh, v+r+1),
            max(0, u-r):min(dw, u+r+1)
        ].astype(np.float32)
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        depth_m = float(np.median(valid)) / 1000.0   # mm → m

        if not (0.1 < depth_m < 3.0):
            self.get_logger().warn(
                f'Depth {depth_m:.3f}m out of plausible range, skipping.')
            return None

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        return ((u - cx) * depth_m / fx,
                (v - cy) * depth_m / fy,
                depth_m)

    # ── /feed_trigger service ─────────────────────────────────────────

    def feed_trigger_callback(self, request, response):
        if self.mouth_3d is None:
            msg = ('❌  No mouth detected. '
                   'Ensure the patient face is visible to the camera.')
            self.get_logger().error(msg)
            response.success = False
            response.message = msg
            return response

        x, y, z = self.mouth_3d

        pose = PoseStamped()
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.header.frame_id = self.camera_frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z - self.approach_offset_z  # stop in front
        pose.pose.orientation.w = 1.0   # identity quaternion

        self.mouth_pose_pub.publish(pose)

        msg = (f'✅  Feed triggered! Mouth=({x:.3f},{y:.3f},{z:.3f})m '
               f'Goal published to /mouth_pose')
        self.get_logger().info(msg)
        response.success = True
        response.message = msg
        return response

    def destroy_node(self):
        self.detector.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MouthDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()