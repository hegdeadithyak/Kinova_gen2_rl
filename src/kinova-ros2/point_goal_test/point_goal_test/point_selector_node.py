#!/usr/bin/env python3
"""Show live color feed; click a pixel → publish 3D goal point in root frame."""
import sys
import numpy as np
import cv2
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs  # noqa: F401


class PointSelectorNode(Node):
    def __init__(self):
        super().__init__('point_selector_node')
        self._bridge = CvBridge()
        self._K = None
        self._color_frame = None
        self._depth_frame = None
        self._clicked_px = None

        self.declare_parameter('cam_offset_x', 0.0)  # + = camera right, - = camera left
        self.declare_parameter('cam_offset_y', 0.0)  # + = camera down,  - = camera up

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._goal_pub = self.create_publisher(PointStamped, '/goal_point', 10)

        self.create_subscription(CameraInfo,
            '/camera/camera/color/camera_info', self._info_cb, 1)
        self.create_subscription(Image,
            '/camera/camera/color/image_raw', self._color_cb, 1)
        self.create_subscription(Image,
            '/camera/camera/aligned_depth_to_color/image_raw', self._depth_cb, 1)

        self._tf_ready = False

        cv2.namedWindow('PointSelector', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('PointSelector', self._mouse_cb)
        self.create_timer(1.0, self._check_tf)
        self.get_logger().info('PointSelector started — waiting for TF tree...')

    def _check_tf(self):
        if self._tf_ready:
            return
        try:
            self._tf_buffer.lookup_transform(
                'root', 'camera_color_optical_frame',
                rclpy.time.Time(), timeout=Duration(seconds=0.1))
            self._tf_ready = True
            self.get_logger().info('TF ready — CLICK A PIXEL TO MOVE THE ARM')
        except Exception:
            self.get_logger().info('Waiting for TF (root ↔ camera_color_optical_frame)...', once=True)

    def _info_cb(self, msg):
        if self._K is None:
            self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f'Camera intrinsics received: fx={self._K[0,0]:.1f}')

    def _color_cb(self, msg):
        try:
            self._color_frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Color decode error: {e}')

    def _depth_cb(self, msg):
        try:
            self._depth_frame = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth decode error: {e}')

    def _mouse_cb(self, event, u, v, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not self._tf_ready:
            self.get_logger().warn('TF not ready yet — wait for "CLICK A PIXEL" message')
            return
        if self._K is None:
            self.get_logger().warn('No camera info yet')
            return
        if self._depth_frame is None:
            self.get_logger().warn('No depth frame yet')
            return

        raw = int(self._depth_frame[v, u])
        depth_m = raw / 1000.0

        if raw == 0 or depth_m > 2.0:
            print(f'Pixel ({u},{v}) depth={depth_m:.3f}m — skipped (invalid)')
            return

        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]
        x_c = (u - cx) * depth_m / fx + self.get_parameter('cam_offset_x').value
        y_c = (v - cy) * depth_m / fy + self.get_parameter('cam_offset_y').value
        z_c = depth_m

        print(f'\n--- Click ({u},{v}) ---')
        print(f'  raw depth:    {raw}  →  {depth_m:.4f} m')
        print(f'  camera frame: x={x_c:.4f}  y={y_c:.4f}  z={z_c:.4f}  (offsets applied)')

        pt = PointStamped()
        pt.header.frame_id = 'camera_color_optical_frame'
        pt.header.stamp = rclpy.time.Time().to_msg()  # Time(0) = latest available
        pt.point.x, pt.point.y, pt.point.z = x_c, y_c, z_c

        try:
            pt_root = self._tf_buffer.transform(
                pt, 'root', timeout=Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'TF failed: {e}')
            return

        rx, ry, rz = pt_root.point.x, pt_root.point.y, pt_root.point.z
        print(f'  root frame:   x={rx:.4f}  y={ry:.4f}  z={rz:.4f}')

        self._goal_pub.publish(pt_root)
        self._clicked_px = (u, v)

    def display_once(self):
        if self._color_frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, 'Waiting for camera...', (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('PointSelector', blank)
        else:
            frame = self._color_frame.copy()
            if self._clicked_px is not None:
                cv2.circle(frame, self._clicked_px, 8, (0, 0, 255), 2)
                cv2.circle(frame, self._clicked_px, 2, (0, 0, 255), -1)
            cv2.imshow('PointSelector', frame)
        return cv2.waitKey(1)


def main():
    rclpy.init(args=sys.argv)
    node = PointSelectorNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            key = node.display_once()
            if key == 27:  # ESC to quit
                break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
