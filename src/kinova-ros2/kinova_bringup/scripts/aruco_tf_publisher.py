#!/usr/bin/env python3
"""Detect a single ArUco marker (DICT_5X5_250) and publish its pose to TF."""
import sys
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class ArucoTfPublisher(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher')
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('marker_size', 0.0594)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('marker_frame', 'aruco_marker_frame')

        self._marker_id = self.get_parameter('marker_id').value
        self._marker_size = self.get_parameter('marker_size').value
        self._camera_frame = self.get_parameter('camera_frame').value
        self._marker_frame = self.get_parameter('marker_frame').value

        self._bridge = CvBridge()
        self._tf_broadcaster = TransformBroadcaster(self)
        self._camera_matrix = None
        self._dist_coeffs = None

        self._dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self._params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)

        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self._camera_info_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._image_cb, 10)
        self.get_logger().info(
            f'Detecting marker id={self._marker_id} size={self._marker_size}m '
            f'dict=DICT_5X5_250 -> TF {self._camera_frame} -> {self._marker_frame}')

    def _camera_info_cb(self, msg):
        self._camera_matrix = np.array(msg.k).reshape(3, 3)
        self._dist_coeffs = np.array(msg.d)

    def _image_cb(self, msg):
        if self._camera_matrix is None:
            return
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return
        for i, mid in enumerate(ids.flatten()):
            if mid != self._marker_id:
                continue
            half = self._marker_size / 2.0
            obj_pts = np.array([
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0],
            ], dtype=np.float32)
            img_pts = corners[i].reshape(4, 2)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, self._camera_matrix, self._dist_coeffs)
            if not ok:
                continue
            rvec = rvec.flatten()
            tvec = tvec.flatten()
            rot, _ = cv2.Rodrigues(rvec)
            # rotation matrix -> quaternion
            trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                w = 0.25 / s
                x = (rot[2, 1] - rot[1, 2]) * s
                y = (rot[0, 2] - rot[2, 0]) * s
                z = (rot[1, 0] - rot[0, 1]) * s
            elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
                w = (rot[2, 1] - rot[1, 2]) / s
                x = 0.25 * s
                y = (rot[0, 1] + rot[1, 0]) / s
                z = (rot[0, 2] + rot[2, 0]) / s
            elif rot[1, 1] > rot[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
                w = (rot[0, 2] - rot[2, 0]) / s
                x = (rot[0, 1] + rot[1, 0]) / s
                y = 0.25 * s
                z = (rot[1, 2] + rot[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
                w = (rot[1, 0] - rot[0, 1]) / s
                x = (rot[0, 2] + rot[2, 0]) / s
                y = (rot[1, 2] + rot[2, 1]) / s
                z = 0.25 * s
            t = TransformStamped()
            t.header.stamp = msg.header.stamp
            t.header.frame_id = self._camera_frame
            t.child_frame_id = self._marker_frame
            t.transform.translation.x = float(tvec[0])
            t.transform.translation.y = float(tvec[1])
            t.transform.translation.z = float(tvec[2])
            t.transform.rotation.x = float(x)
            t.transform.rotation.y = float(y)
            t.transform.rotation.z = float(z)
            t.transform.rotation.w = float(w)
            self._tf_broadcaster.sendTransform(t)
            return


def main():
    rclpy.init(args=sys.argv)
    node = ArucoTfPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
