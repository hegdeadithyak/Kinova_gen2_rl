#!/usr/bin/env python3
"""FaceMesh mouth detector — press SPACE to send mouth 3D position to /goal_point."""
import os
import sys
import threading

os.environ["QT_LOGGING_RULES"] = "qt.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
for _d in ("/usr/share/fonts", "/usr/share/fonts/truetype", "/usr/local/share/fonts"):
    if os.path.isdir(_d):
        os.environ.setdefault("QT_QPA_FONTDIR", _d)
        break

import numpy as np
import cv2
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs  # noqa: F401

# Reuse face_overlay from click_pointer
sys.path.insert(0, '/home/amma/Kinova_gen2_rl/src/click_pointer/scripts')
from face_overlay import build_face_mesh, draw_face_overlay, MOUTH_CENTER_IDS


class MouthGoalNode(Node):
    def __init__(self):
        super().__init__('mouth_goal_node')

        self._cb   = ReentrantCallbackGroup()
        self._lock = threading.Lock()

        self._bridge      = CvBridge()
        self._K           = None
        self._color_frame = None
        self._depth_frame = None
        self._depth_enc   = '16UC1'
        self._mouth_px    = None
        self._mouth_lms   = None
        self._mouth_hw    = None
        self._tf_ready    = False

        self.declare_parameter('cam_offset_x', 0.0)
        self.declare_parameter('cam_offset_y', 0.0)
        self.declare_parameter('cam_offset_z', 0.0)

        self._mp_face = build_face_mesh()

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._goal_pub    = self.create_publisher(PointStamped, '/goal_point', 10)

        self.create_subscription(CameraInfo,
            '/camera/camera/color/camera_info', self._info_cb, 1,
            callback_group=self._cb)
        self.create_subscription(Image,
            '/camera/camera/color/image_raw', self._color_cb, 1,
            callback_group=self._cb)
        self.create_subscription(Image,
            '/camera/camera/aligned_depth_to_color/image_raw', self._depth_cb, 1,
            callback_group=self._cb)

        self.create_timer(1.0, self._check_tf)
        self.get_logger().info('MouthGoal ready — SPACE to send mouth to arm')

    def _check_tf(self):
        if self._tf_ready:
            return
        try:
            self._tf_buffer.lookup_transform(
                'root', 'camera_color_optical_frame',
                rclpy.time.Time(), timeout=Duration(seconds=0.1))
            self._tf_ready = True
            self.get_logger().info('TF ready')
        except Exception:
            pass

    def _info_cb(self, msg):
        with self._lock:
            if self._K is None:
                self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _color_cb(self, msg):
        frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w  = frame.shape[:2]
        res   = self._mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mouth_px, lms, hw = None, None, None
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            mu  = int(np.mean([lms[i].x * w for i in MOUTH_CENTER_IDS]))
            mv  = int(np.mean([lms[i].y * h for i in MOUTH_CENTER_IDS]))
            mouth_px = (mu, mv)
            hw = (h, w)
        with self._lock:
            self._color_frame = frame
            self._mouth_px    = mouth_px
            self._mouth_lms   = lms
            self._mouth_hw    = hw

    def _depth_cb(self, msg):
        with self._lock:
            self._depth_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self._depth_enc   = msg.encoding

    def get_frame_data(self):
        with self._lock:
            return (self._color_frame, self._mouth_px,
                    self._mouth_lms, self._mouth_hw)

    def send_mouth_goal(self):
        with self._lock:
            mpx   = self._mouth_px
            depth = self._depth_frame
            K     = self._K
            enc   = self._depth_enc

        if mpx is None:
            self.get_logger().warn('No face detected')
            return
        if depth is None or K is None:
            self.get_logger().warn('Camera not ready')
            return

        u, v = mpx
        h, w = depth.shape[:2]
        u0, u1 = max(0, u - 4), min(w, u + 5)
        v0, v1 = max(0, v - 4), min(h, v + 5)
        patch = depth[v0:v1, u0:u1].astype(np.float32)
        valid = patch[patch > 0]
        if len(valid) < 4:
            self.get_logger().warn('Invalid depth at mouth')
            return
        raw = float(np.median(valid))
        depth_m = raw / 1000.0 if enc == '16UC1' else raw
        if not (0.1 < depth_m < 5.0):
            self.get_logger().warn(f'Depth out of range: {depth_m:.3f}m')
            return

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        pt = PointStamped()
        pt.header.frame_id = 'camera_color_optical_frame'
        pt.header.stamp    = rclpy.time.Time().to_msg()
        pt.point.x = (u - cx) * depth_m / fx + self.get_parameter('cam_offset_x').value
        pt.point.y = (v - cy) * depth_m / fy + self.get_parameter('cam_offset_y').value
        pt.point.z = depth_m + self.get_parameter('cam_offset_z').value

        try:
            pt_root = self._tf_buffer.transform(pt, 'root', timeout=Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'TF failed: {e}')
            return

        self._goal_pub.publish(pt_root)
        self.get_logger().info(
            f'Mouth → root: ({pt_root.point.x:.3f}, '
            f'{pt_root.point.y:.3f}, {pt_root.point.z:.3f})')


def run_ui(node: MouthGoalNode):
    winname = 'MouthGoal | Kinova Feeding'
    _saved_stderr = os.dup(2)
    _devnull      = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)

    try:
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        frame_count = 0

        while rclpy.ok():
            frame, mouth_px, lms, hw = node.get_frame_data()

            if frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, 'Waiting for camera...', (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(winname, blank)
            else:
                disp = frame.copy()

                # Draw face overlay using click_pointer's neon style
                disp = draw_face_overlay(disp, lms, hw, frame_count)

                # Highlight mouth centre
                if mouth_px:
                    cv2.circle(disp, mouth_px, 10, (0, 255, 0), -1)
                    cv2.circle(disp, mouth_px, 13, (255, 255, 255), 2)

                # HUD
                if mouth_px and node._tf_ready:
                    status, color = 'SPACE = feed mouth', (0, 255, 0)
                else:
                    status, color = 'No face detected', (0, 0, 255)
                cv2.putText(disp, status, (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1, cv2.LINE_AA)

                cv2.imshow(winname, disp)
                frame_count += 1

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == 32:  # SPACE
                threading.Thread(target=node.send_mouth_goal, daemon=True).start()

    finally:
        os.dup2(_saved_stderr, 2)
        os.close(_saved_stderr)
        cv2.destroyAllWindows()


def main():
    rclpy.init(args=sys.argv)
    node = MouthGoalNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        run_ui(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
