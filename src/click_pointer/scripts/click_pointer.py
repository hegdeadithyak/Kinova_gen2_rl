#!/usr/bin/env python3
from __future__ import annotations

import os
import threading
import time
import warnings

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tui import TUI
from face_overlay import build_face_mesh, draw_face_overlay, MOUTH_CENTER_IDS

os.environ["QT_LOGGING_RULES"] = "qt.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
for _d in ("/usr/share/fonts", "/usr/share/fonts/truetype", "/usr/local/share/fonts"):
    if os.path.isdir(_d):
        os.environ.setdefault("QT_QPA_FONTDIR", _d)
        break
warnings.filterwarnings("ignore")

FX = 603.6312;  FY = 603.0632
CX = 319.0870;  CY = 236.3678

EE_LINK    = "j2s6s200_end_effector"
CAM_FRAME  = "camera_color_optical_frame"
BASE_FRAME = "j2s6s200_link_base"

ARM_JOINT_NAMES = [
    "j2s6s200_joint_1", "j2s6s200_joint_2", "j2s6s200_joint_3",
    "j2s6s200_joint_4", "j2s6s200_joint_5", "j2s6s200_joint_6",
]

CAM_Y_OFFSET = -0.1
CAM_X_OFFSET = -0.24
CAM_Z_OFFSET = -0.15
STEP_RADS    = 0.05
CART_DELTA_X = 0.08
CART_DELTA_Y = 0.07
CART_DELTA_Z = 0.06
TICK_DUR_S   = 0.6
ERR_TOL_M    = 0.05


def _draw_hud(disp: np.ndarray, node, frame_count: int) -> np.ndarray:
    h, w = disp.shape[:2]

    overlay = disp.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.72, disp, 0.28, 0, disp)

    if node.busy:
        pill_bgr, pill_txt, txt_bgr = (160, 30, 0), " ● FEEDING... ", (255, 255, 200)
    elif node.pending_target_cam is not None:
        pill_bgr, pill_txt, txt_bgr = (0, 100, 180), " ▶ CONFIRM? ", (80, 220, 255)
    else:
        pill_bgr, pill_txt, txt_bgr = (0, 60, 0), " ● READY ", (80, 255, 140)

    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(pill_txt, font, 0.52, 1)
    px, py = 8, 6
    cv2.rectangle(disp, (px - 2, py - 2), (px + tw + 10, py + th + 10), pill_bgr, -1)
    cv2.rectangle(disp, (px - 2, py - 2), (px + tw + 10, py + th + 10), (100, 120, 120), 1)
    cv2.putText(disp, pill_txt, (px + 4, py + th + 3), font, 0.52, txt_bgr, 1, cv2.LINE_AA)

    if not node.busy:
        hint = ("Y = feed   N = cancel" if node.pending_target_cam is not None
                else "SPACE = mouth    CLICK = target    Q = quit")
        cv2.putText(disp, hint, (px + tw + 22, py + th + 3), font, 0.38, (140, 145, 145), 1, cv2.LINE_AA)

    mouth_lms, mouth_hw = node.get_landmark_data()
    disp = draw_face_overlay(disp, mouth_lms, mouth_hw, frame_count)

    if node.pending_target_cam is not None:
        tc = node.pending_target_cam
        coord_str = f"target  x={tc[0]:+.2f}  y={tc[1]:+.2f}  z={tc[2]:+.2f} m"
        ov2 = disp.copy()
        cv2.rectangle(ov2, (0, h - 28), (w, h), (10, 10, 20), -1)
        cv2.addWeighted(ov2, 0.68, disp, 0.32, 0, disp)
        cv2.putText(disp, coord_str, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 255), 1, cv2.LINE_AA)

    if node.busy:
        phase_str = f"Phase: {getattr(node, '_current_phase', 'moving...')}"
        (ptw, _), _ = cv2.getTextSize(phase_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        ov3 = disp.copy()
        cv2.rectangle(ov3, (w - ptw - 18, h - 28), (w, h), (10, 10, 20), -1)
        cv2.addWeighted(ov3, 0.68, disp, 0.32, 0, disp)
        cv2.putText(disp, phase_str, (w - ptw - 10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 180, 50), 1, cv2.LINE_AA)

    if not node.busy:
        cx2, cy2 = w // 2, h // 2
        gap, arm, clr = 6, 18, (0, 200, 160)
        cv2.line(disp, (cx2 - arm - gap, cy2), (cx2 - gap, cy2), clr, 1, cv2.LINE_AA)
        cv2.line(disp, (cx2 + gap, cy2), (cx2 + arm + gap, cy2), clr, 1, cv2.LINE_AA)
        cv2.line(disp, (cx2, cy2 - arm - gap), (cx2, cy2 - gap), clr, 1, cv2.LINE_AA)
        cv2.line(disp, (cx2, cy2 + gap), (cx2, cy2 + arm + gap), clr, 1, cv2.LINE_AA)
        cv2.circle(disp, (cx2, cy2), 2, clr, -1, cv2.LINE_AA)

    return disp


class ClickPointer(Node):
    def __init__(self, tui: TUI):
        super().__init__("click_pointer")
        self.tui      = tui
        self.cb_group = ReentrantCallbackGroup()

        self.bridge    = CvBridge()
        self.lock      = threading.Lock()
        self.color_img = None
        self.depth_img = None
        self.joints    = {n: 0.0 for n in ARM_JOINT_NAMES}
        self.busy      = False

        self.mouth_px  = None
        self.mouth_lms = None
        self.mouth_hw  = None
        self.mp_face   = build_face_mesh()

        self.pending_target_cam = None
        self._current_phase     = ""

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, "/camera/camera/color/image_raw",
                                 self.color_cb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw",
                                 self.depth_cb, 10, callback_group=self.cb_group)
        self.create_subscription(JointState, "/joint_states",
                                 self.js_cb, 10, callback_group=self.cb_group)

        self.marker_pub   = self.create_publisher(Marker, "/target_marker", 10)
        self._traj_client = ActionClient(self, FollowJointTrajectory,
                                         "/arm_controller/follow_joint_trajectory",
                                         callback_group=self.cb_group)

        self.tui.info("Waiting for trajectory controller ...")
        if not self._traj_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("Timed out waiting for trajectory action server")

        self.create_subscription(Bool, '/eye_confirm', self._eye_cb, 10,
                                 callback_group=self.cb_group)
        self.tui.ready_prompt()

    # ── Subscribers ───────────────────────────────────────────────────────────

    def color_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        mp_, ml_, mh_ = self._detect_mouth(frame)
        with self.lock:
            self.color_img = frame
            self.mouth_px  = mp_
            self.mouth_lms = ml_
            self.mouth_hw  = mh_

    def depth_cb(self, msg):
        with self.lock:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def js_cb(self, msg):
        with self.lock:
            for n, p in zip(msg.name, msg.position):
                if n in self.joints:
                    self.joints[n] = p

    # ── Face detection ────────────────────────────────────────────────────────

    def _detect_mouth(self, frame):
        h, w = frame.shape[:2]
        res  = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None, None, None
        lms = res.multi_face_landmarks[0].landmark
        return (
            int(np.mean([lms[i].x * w for i in MOUTH_CENTER_IDS])),
            int(np.mean([lms[i].y * h for i in MOUTH_CENTER_IDS]))
        ), lms, (h, w)

    # ── Thread-safe getters ───────────────────────────────────────────────────

    def get_frame_data(self):
        with self.lock:
            return self.color_img

    def get_mouth_px(self):
        with self.lock:
            return self.mouth_px

    def get_landmark_data(self):
        with self.lock:
            return self.mouth_lms, self.mouth_hw

    # ── Input events ──────────────────────────────────────────────────────────

    def on_click(self, u, v):
        if not self.busy:
            self.process_click(u, v)

    def on_space(self):
        with self.lock:
            mpix = self.mouth_px
        if mpix is None:
            self.tui.warn("No mouth detected — look at the camera")
            return
        self.process_click(*mpix)

    def confirm_feed(self):
        if self.pending_target_cam is None or self.busy:
            return
        tgt = self.pending_target_cam
        self.pending_target_cam = None
        threading.Thread(target=self.move_to_point_cam, args=(tgt,), daemon=True).start()

    def _eye_cb(self, msg: Bool):
        if msg.data:
            threading.Thread(target=self._eye_confirm_sequence, daemon=True).start()
        else:
            self.cancel_feed()

    def _eye_confirm_sequence(self):
        if self.pending_target_cam is None and not self.busy:
            self.on_space()
        self.confirm_feed()

    def cancel_feed(self):
        if self.pending_target_cam is not None:
            self.pending_target_cam = None
            self.delete_marker()
            self.tui.info("Feed cancelled.")

    # ── Click → 3-D target ────────────────────────────────────────────────────

    def process_click(self, u, v):
        if self.busy:
            return
        with self.lock:
            depth = self.depth_img
        if depth is None:
            self.tui.warn("No depth image yet — waiting for camera")
            return
        z_raw = float(depth[v, u]) * 0.001
        if z_raw <= 0.01:
            self.tui.warn(f"Bad depth at ({u},{v}): {z_raw*1000:.0f} mm — click on the subject")
            return

        x_cam = (u - CX) * z_raw / FX
        y_cam = (v - CY) * z_raw / FY
        target_cam = np.array([x_cam + CAM_X_OFFSET, y_cam - CAM_Y_OFFSET, z_raw + CAM_Z_OFFSET])

        self.tui.separator()
        self.tui.info(f"Click at pixel ({u}, {v}),  depth = {z_raw*100:.1f} cm")
        self.tui.coord_block("Target (cam-frame):", *target_cam, color=TUI.CYN)

        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, CAM_FRAME, Time(), Duration(seconds=1.0))
            R  = self._quat_to_matrix(tf.transform.rotation)
            t  = np.array([tf.transform.translation.x,
                           tf.transform.translation.y,
                           tf.transform.translation.z])
            target_base = R @ target_cam + t
        except Exception as e:
            self.tui.warn(f"TF camera→base failed ({e}), skipping RViz marker")
            target_base = target_cam

        self.publish_marker(target_base, BASE_FRAME)
        self.tui.coord_block("Target (base-frame):", *target_base, color=TUI.BLU)

        self.pending_target_cam = target_cam
        self.tui.info(f"dx = {target_cam[0]:+.3f} m  dy = {target_cam[1]:+.3f} m")
        self.tui.prompt("Press  Y  to feed   ·   N  to cancel")


    def publish_marker(self, pt, fid):
        m = Marker()
        m.header.frame_id = fid
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns = "target"; m.id = 0
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x = float(pt[0])
        m.pose.position.y = float(pt[1])
        m.pose.position.z = float(pt[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color.r = 1.0; m.color.a = 1.0
        m.lifetime.sec = 0
        self.marker_pub.publish(m)

    def delete_marker(self):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns = "target"; m.id = 0; m.action = Marker.DELETE
        self.marker_pub.publish(m)

    @staticmethod
    def _quat_to_matrix(q):
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1-2*y*y-2*z*z,   2*x*y-2*z*w,   2*x*z+2*y*w],
            [  2*x*y+2*z*w, 1-2*x*x-2*z*z,   2*y*z-2*x*w],
            [  2*x*z-2*y*w,   2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])

    # ── Phased joint stepping ─────────────────────────────────────────────────

    def move_to_point_cam(self, target_cam):
        if self.busy:
            return
        self.busy = True
        try:
            try:
                tf_ee_cam = self.tf_buffer.lookup_transform(
                    CAM_FRAME, EE_LINK, Time(), timeout=Duration(seconds=1.0))
            except Exception as e:
                self.tui.error(f"TF EE→cam failed: {e}")
                return

            start = np.array([tf_ee_cam.transform.translation.x,
                              tf_ee_cam.transform.translation.y,
                              tf_ee_cam.transform.translation.z])
            self.tui.separator()
            self.tui.coord_block("EE start (cam):", *start, color=TUI.GRY)
            self.tui.coord_block("Target   (cam):", *target_cam, color=TUI.CYN)

            self._perform_phased(start, target_cam)

            self.tui.separator()
            self.tui.success("Feeding sequence complete.")
            self.tui.separator()
            time.sleep(0.5)

        except Exception as e:
            self.tui.error(f"move_to_point_cam: {e}")
        finally:
            self.busy = False
            self._current_phase = ""

    def _perform_phased(self, startpoint, endpoint):
        MIN_DUR  = 1.5
        dy       = endpoint[1] - startpoint[1]
        dz       = endpoint[2] - startpoint[2]
        dx_cam   = endpoint[0]

        self.tui.phase_header(1, "Y-axis  →  J2 (elbow pitch)", f"error = {dy:+.3f} m")
        self._current_phase = "Y → J2"
        if abs(dy) > ERR_TOL_M:
            delta_J2 = -(dy / CART_DELTA_Y) * STEP_RADS
            self._dispatch_smooth({1: delta_J2}, max(MIN_DUR, abs(dy) / CART_DELTA_Y * TICK_DUR_S), "Y-axis")
        self.tui.phase_done("Y-axis")

        px_offset = endpoint[0] * FX / endpoint[2]
        self.tui.phase_header(2, "X-axis  →  J3 (forearm rotation)",
                              f"target = {dx_cam:+.3f} m from cam-centre  ({px_offset:+.0f} px)")
        self._current_phase = "X → J3"
        if abs(dx_cam) > ERR_TOL_M:
            delta_J3 = -(dx_cam / CART_DELTA_X) * STEP_RADS
            self._dispatch_smooth({2: delta_J3}, max(MIN_DUR, abs(dx_cam) / CART_DELTA_X * TICK_DUR_S), "X-axis")
        self.tui.phase_done("X-axis")

        self.tui.phase_header(3, "Z-axis  →  J1 + J4 (depth approach)", f"error = {dz:+.3f} m")
        self._current_phase = "Z → J1+J4"
        if dz > ERR_TOL_M:
            delta = (dz / CART_DELTA_Z) * STEP_RADS * 2.0
            self._dispatch_smooth({0: delta, 3: delta}, max(MIN_DUR, dz / CART_DELTA_Z * TICK_DUR_S), "Z-axis")
        self.tui.phase_done("Z-axis")

        self.tui.info("All three phases converged.")

    # ── Trajectory dispatch ───────────────────────────────────────────────────

    def _make_trajectory(self, current_pos, target_pos, duration_s):
        pt0 = JointTrajectoryPoint()
        pt0.positions       = list(current_pos)
        pt0.time_from_start = DurationMsg(sec=0, nanosec=0)

        dur_sec = int(duration_s)
        pt1 = JointTrajectoryPoint()
        pt1.positions       = target_pos
        pt1.time_from_start = DurationMsg(sec=dur_sec, nanosec=int((duration_s - dur_sec) * 1e9))

        traj              = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = ARM_JOINT_NAMES
        traj.points       = [pt0, pt1]
        return traj

    def _dispatch_smooth(self, deltas: dict, duration_s: float, label: str = "moving"):
        # Blocks until action server completes — required to prevent phases from running concurrently.
        with self.lock:
            current_pos = [self.joints[n] for n in ARM_JOINT_NAMES]
        target_pos = list(current_pos)
        for idx, delta in deltas.items():
            target_pos[idx] += delta

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self._make_trajectory(current_pos, target_pos, duration_s)

        done_event = threading.Event()

        def _goal_resp(future):
            gh = future.result()
            if not gh.accepted:
                done_event.set()
                return
            gh.get_result_async().add_done_callback(lambda _: done_event.set())

        self._traj_client.send_goal_async(goal).add_done_callback(_goal_resp)

        t0 = time.time()
        while not done_event.wait(timeout=0.08):
            self.tui.moving(label, time.time() - t0, duration_s)
        print()


def run_ui(node: ClickPointer, tui: TUI):
    winname = "Click-to-Feed | Kinova Jaco2"

    # Redirect stderr for window lifetime — Qt font/platform warnings pollute the TUI (stdout only)
    _saved_stderr = os.dup(2)
    _devnull      = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)

    try:
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(winname,
                             lambda event, x, y, flags, param:
                                 node.on_click(x, y) if event == cv2.EVENT_LBUTTONDOWN else None,
                             node)

        frame_count = 0
        while rclpy.ok():
            frame = node.get_frame_data()
            if frame is not None:
                cv2.imshow(winname, _draw_hud(frame.copy(), node, frame_count))
                frame_count += 1

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                threading.Thread(target=node.on_space, daemon=True).start()
            elif key in (ord("y"), ord("Y")):
                node.confirm_feed()
            elif key in (ord("n"), ord("N")):
                node.cancel_feed()

    finally:
        os.dup2(_saved_stderr, 2)
        os.close(_saved_stderr)
        cv2.destroyAllWindows()


def main(args=None):
    tui = TUI()
    tui.banner()

    rclpy.init(args=args)
    node = ClickPointer(tui)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        run_ui(node, tui)
    finally:
        tui.info("Cleaning up ROS2 node ...")
        node.destroy_node()
        rclpy.shutdown()
        tui.success("Shutdown complete.")


if __name__ == "__main__":
    main()