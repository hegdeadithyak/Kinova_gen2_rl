import math
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from rclpy.node import Node
from sensor_msgs.msg import Image

# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.expanduser('~/.mediapipe/face_landmarker.task')
_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)

FaceLandmarksConnections = mp_vision.FaceLandmarksConnections


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        print(f'Downloading FaceLandmarker model to {_MODEL_PATH} …')
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print('Download complete.')


# ---------------------------------------------------------------------------
class MouthTracker(Node):
    def __init__(self):
        super().__init__('mouth_tracker')

        self.declare_parameter('mouth_point_topic', '/mouth_3d_point')
        mouth_point_topic = self.get_parameter('mouth_point_topic').value

        # ── MediaPipe FaceLandmarker (Tasks API, mediapipe ≥ 0.10) ──────
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

        # ── RealSense ────────────────────────────────────────────────────
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.intrinsics = (
                profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            self.get_logger().info('RealSense D435i initialised.')
        except Exception as e:
            self.get_logger().error(f'Could not initialise RealSense: {e}')
            raise

        self.bridge = CvBridge()
        self.point_pub = self.create_publisher(PointStamped, mouth_point_topic, 10)
        self.debug_pub = self.create_publisher(Image, '/mouth_tracker/debug_image', 10)
        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)

    # ------------------------------------------------------------------
    def _is_face_proportion_valid(self, lm):
        """Reject detections where nostrils are confused for eyes."""
        nose_tip  = lm[1]
        left_eye  = lm[33]
        right_eye = lm[263]
        upper_lip = lm[13]

        if left_eye.y > nose_tip.y or right_eye.y > nose_tip.y:
            return False, 'Eyes below nose'
        if nose_tip.y > upper_lip.y:
            return False, 'Nose below mouth'

        eye_dist       = math.hypot(left_eye.x - right_eye.x, left_eye.y - right_eye.y)
        eye_mouth_dist = upper_lip.y - (left_eye.y + right_eye.y) / 2
        if eye_mouth_dist < eye_dist * 0.3:
            return False, 'Face squashed (nostrils-as-eyes risk)'

        return True, 'Valid'

    # ------------------------------------------------------------------
    def _draw_connections(self, image, lm, connections, color, h, w):
        for conn in connections:
            start = lm[conn.start]
            end   = lm[conn.end]
            x0, y0 = int(start.x * w), int(start.y * h)
            x1, y1 = int(end.x * w),   int(end.y * h)
            cv2.line(image, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def process_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                return

            color_image   = np.asanyarray(color_frame.get_data())
            display_image = color_image.copy()
            h, w          = color_image.shape[:2]

            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            mp_image  = MpImage(image_format=ImageFormat.SRGB, data=rgb_image)
            result    = self.face_landmarker.detect(mp_image)

            if result.face_landmarks:
                lm = result.face_landmarks[0]

                valid, reason = self._is_face_proportion_valid(lm)
                if not valid:
                    cv2.putText(display_image, f'REJECTED: {reason}', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # Draw mesh
                    self._draw_connections(
                        display_image, lm,
                        FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                        (80, 110, 10), h, w)
                    self._draw_connections(
                        display_image, lm,
                        FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                        (255, 255, 255), h, w)

                    # Mouth centre (landmarks 13 = upper lip, 14 = lower lip)
                    pt13, pt14 = lm[13], lm[14]
                    u = int((pt13.x + pt14.x) / 2 * w)
                    v = int((pt13.y + pt14.y) / 2 * h)
                    u = max(0, min(w - 1, u))
                    v = max(0, min(h - 1, v))

                    distance = depth_frame.get_distance(u, v)
                    if distance > 0:
                        point_3d = rs.rs2_deproject_pixel_to_point(
                            self.intrinsics, [u, v], distance)
                        x, y, z = point_3d

                        msg = PointStamped()
                        msg.header.stamp    = self.get_clock().now().to_msg()
                        msg.header.frame_id = 'camera_color_optical_frame'
                        msg.point.x = float(x)
                        msg.point.y = float(y)
                        msg.point.z = float(z)
                        self.point_pub.publish(msg)

                        color = (0, 255, 255) if z > 0.15 else (0, 165, 255)
                        cv2.drawMarker(display_image, (u, v), color,
                                       cv2.MARKER_CROSS, 20, 2)
                        cv2.putText(display_image, f'DIST: {z:.2f}m',
                                    (u + 10, v - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            debug_msg = self.bridge.cv2_to_imgmsg(display_image, 'bgr8')
            debug_msg.header.stamp    = self.get_clock().now().to_msg()
            debug_msg.header.frame_id = 'camera_color_optical_frame'
            self.debug_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f'Error in process_frame: {e}')

    def __del__(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()


# ---------------------------------------------------------------------------
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
