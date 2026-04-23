#!/usr/bin/env python3


import math
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

# Offset applied to the raw 3-D mouth point in camera_color_optical_frame (metres).
# X: right, Y: down, Z: depth (away from camera).
POINTER_OFFSET = np.array([
    -0.108,   # X
    -0.368,   # Y
    -0.347,   # Z (depth)
])
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

FaceLandmarksConnections = mp_vision.FaceLandmarksConnections

_MODEL_PATH = os.path.expanduser('~/.mediapipe/face_landmarker.task')
_MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        print(f'Downloading FaceLandmarker model to {_MODEL_PATH} …')
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print('Download complete.')


class MouthTracker(Node):
    def __init__(self):
        super().__init__('mouth_tracker')

        self.declare_parameter('mouth_point_topic', '/mouth_3d_point')
        mouth_point_topic = self.get_parameter('mouth_point_topic').value

        # ── MediaPipe ──────────────────────────────────────────────────
        _ensure_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.8,
            min_tracking_confidence=0.8,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        self.bridge       = CvBridge()
        self.cam_model    = PinholeCameraModel()
        self._got_info    = False
        self.point_pub = self.create_publisher(
            PointStamped, mouth_point_topic, 10)
        self.debug_pub = self.create_publisher(
            Image, '/mouth_tracker/debug_image', 10)

        self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info',
            self._info_cb, 1)

        # ── Synchronized color + aligned depth ────────────────────────
        color_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        depth_sub = Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.05)
        sync.registerCallback(self._frame_cb)

        self.get_logger().info(
            'MouthTracker ready — waiting for /camera/camera/color/image_raw …')

    # ------------------------------------------------------------------
    def _info_cb(self, msg: CameraInfo):
        if not self._got_info:
            self.cam_model.fromCameraInfo(msg)
            self._got_info = True
            self.get_logger().info(
                f'Camera intrinsics loaded: '
                f'fx={self.cam_model.fx():.1f} fy={self.cam_model.fy():.1f} '
                f'cx={self.cam_model.cx():.1f} cy={self.cam_model.cy():.1f}')

    # ------------------------------------------------------------------
    def _frame_cb(self, color_msg: Image, depth_msg: Image):
        if not self._got_info:
            return

        try:
            color_bgr  = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            depth_raw  = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        display = color_bgr.copy()
        h, w    = color_bgr.shape[:2]

        rgb_image = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = MpImage(image_format=ImageFormat.SRGB, data=rgb_image)
        result    = self.face_landmarker.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            valid, reason = self._face_proportion_valid(lm)
            if not valid:
                cv2.putText(display, f'REJECTED: {reason}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                self._draw_connections(
                    display, lm,
                    FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    (80, 110, 10), h, w)
                self._draw_connections(
                    display, lm,
                    FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    (255, 255, 255), h, w)

                # Mouth centre (landmarks 13 = upper lip, 14 = lower lip)
                pt13, pt14 = lm[13], lm[14]
                u = int((pt13.x + pt14.x) / 2 * w)
                v = int((pt13.y + pt14.y) / 2 * h)
                u = max(0, min(w - 1, u))
                v = max(0, min(h - 1, v))

                # Depth in metres (image is 16-bit millimetres)
                depth_mm = float(depth_raw[v, u])
                if depth_mm > 0:
                    depth_m = depth_mm / 1000.0 -0.25
                    x, y, z = self._deproject(u, v, depth_m)
                    x += POINTER_OFFSET[0]
                    y += POINTER_OFFSET[1]
                    z += POINTER_OFFSET[2]

                    msg = PointStamped()
                    msg.header.stamp    = color_msg.header.stamp
                    msg.header.frame_id = 'camera_color_optical_frame'
                    msg.point.x = x
                    msg.point.y = y
                    msg.point.z = z
                    self.point_pub.publish(msg)

                    color = (0, 255, 255) if z > 0.15 else (0, 165, 255)
                    cv2.drawMarker(display, (u, v), color,
                                   cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(display, f'{z:.2f}m', (u + 10, v - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        debug_msg = self.bridge.cv2_to_imgmsg(display, 'bgr8')
        debug_msg.header.stamp    = color_msg.header.stamp
        debug_msg.header.frame_id = 'camera_color_optical_frame'
        self.debug_pub.publish(debug_msg)

    # ------------------------------------------------------------------
    def _deproject(self, u: int, v: int, depth_m: float):
        """Convert pixel + depth to 3-D point (camera_color_optical_frame)."""
        fx = self.cam_model.fx()
        fy = self.cam_model.fy()
        cx = self.cam_model.cx()
        cy = self.cam_model.cy()
        x  = (u - cx) * depth_m / fx
        y  = (v - cy) * depth_m / fy
        z  = depth_m
        return float(x), float(y), float(z)

    # ------------------------------------------------------------------
    def _face_proportion_valid(self, lm):
        nose_tip  = lm[1]
        left_eye  = lm[33]
        right_eye = lm[263]
        upper_lip = lm[13]

        if left_eye.y > nose_tip.y or right_eye.y > nose_tip.y:
            return False, 'Eyes below nose'
        if nose_tip.y > upper_lip.y:
            return False, 'Nose below mouth'

        eye_dist       = math.hypot(left_eye.x - right_eye.x,
                                    left_eye.y - right_eye.y)
        eye_mouth_dist = upper_lip.y - (left_eye.y + right_eye.y) / 2
        if eye_mouth_dist < eye_dist * 0.3:
            return False, 'Face squashed (nostrils-as-eyes risk)'
        return True, 'Valid'

    # ------------------------------------------------------------------
    def _draw_connections(self, image, lm, connections, color, h, w):
        for conn in connections:
            x0 = int(lm[conn.start].x * w)
            y0 = int(lm[conn.start].y * h)
            x1 = int(lm[conn.end].x * w)
            y1 = int(lm[conn.end].y * h)
            cv2.line(image, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)


def main(args=None):
    rclpy.init(args=args)
    node = MouthTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
