#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import threading
import time

os.environ["QT_LOGGING_RULES"] = "qt.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import MoveItErrorCodes
from visualization_msgs.msg import Marker
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Int32MultiArray
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

FX, FY = 383.04962158203125,383.04962158203125
CX, CY = 318.07012939453125 , 236.87078857421875


BASE_FRAME = "j2s6s200_link_base"
CAM_FRAME  = "camera_color_optical_frame"
EE_LINK    = "j2s6s200_end_effector"
ARM_GROUP  = "arm"

HOVER_HEIGHT    = 0.08
PENETRATE_DEPTH = 0.02
DEPTH_PATCH_PX  = 4
HOVER_TOLERANCE = 0.015

HOVER_VEL_SCALE  = 0.05
CARTESIAN_SPEED  = 0.04
PLUNGE_SPEED     = 0.07
MIN_PATH_FRAC    = 0.00
CART_TIMEOUT_S   = 10.0
MOVE_TIMEOUT_S   = 30.0
EXEC_TIMEOUT_S   = 30.0

ARM_JOINT_NAMES = [
    "j2s6s200_joint_1", "j2s6s200_joint_2", "j2s6s200_joint_3",
    "j2s6s200_joint_4", "j2s6s200_joint_5", "j2s6s200_joint_6",
]
# Camera-to-EE physical offset — matches click_pointer.py exactly
CAM_X_OFFSET = -0.24
CAM_Y_OFFSET =  -0.1
CAM_Z_OFFSET = -0.15
STEP_RADS    =  0.05
CART_DELTA_X =  0.08
CART_DELTA_Y =  0.07
CART_DELTA_Z =  0.06
TICK_DUR_S   =  0.6
ERR_TOL_M    =  0.05


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y],
    ])


def _make_pose(xyz, q) -> Pose:
    p = Pose()
    p.position    = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
    p.orientation = Quaternion(x=float(q.x), y=float(q.y),
                               z=float(q.z), w=float(q.w))
    return p


def _build_trajectory(cur_pos, tgt_pos, duration_s) -> JointTrajectory:
    traj = JointTrajectory()
    traj.joint_names = ARM_JOINT_NAMES
    pt0 = JointTrajectoryPoint()
    pt0.positions = cur_pos
    pt0.time_from_start = DurationMsg(sec=0, nanosec=0)
    pt1 = JointTrajectoryPoint()
    pt1.positions = tgt_pos
    sec = int(duration_s)
    pt1.time_from_start = DurationMsg(sec=sec, nanosec=int((duration_s - sec) * 1e9))
    traj.points = [pt0, pt1]
    return traj


def _stamp_trajectory(traj: JointTrajectory, speed: float) -> JointTrajectory:
    pts      = traj.points
    total    = 0.0
    prev_pos = list(pts[0].positions) if pts else []
    for i, pt in enumerate(pts):
        if i == 0:
            pt.time_from_start = DurationMsg(sec=0, nanosec=0)
            prev_pos = list(pt.positions)
            continue
        delta  = max((abs(a - b) for a, b in zip(pt.positions, prev_pos)), default=0.01)
        total += max(delta / speed, 0.02)
        s      = int(total)
        pt.time_from_start = DurationMsg(sec=s, nanosec=int((total - s) * 1e9))
        prev_pos = list(pt.positions)
    return traj


class ForkAcquirer(Node):

    def __init__(self, backend: str = "moveit"):
        super().__init__("fork_acquirer")
        self._cb      = ReentrantCallbackGroup()
        self._lock    = threading.Lock()
        self.bridge   = CvBridge()
        self._backend = backend

        self._depth_img  = None
        self._color_img  = None
        self._last_bbox  = None
        self._joints     = {n: 0.0 for n in ARM_JOINT_NAMES}
        self.busy        = False
        self._confirming = False

        self._tf = Buffer()
        TransformListener(self._tf, self)

        self.create_subscription(
            Image, "/camera/camera/color/image_raw",
            self._color_cb, 2, callback_group=self._cb)
        self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw",
            self._depth_cb, 10, callback_group=self._cb)
        self.create_subscription(
            Int32MultiArray, "/fork_acquisition/bbox",
            self._bbox_cb, 10, callback_group=self._cb)
        self.create_subscription(
            Point, "/fork_acquisition/target_xyz",
            self._xyz_cb, 10, callback_group=self._cb)
        self.create_subscription(
            JointState, "/joint_states",
            self._js_cb, 10, callback_group=self._cb)

        self._traj = ActionClient(self, FollowJointTrajectory,
                                  "/arm_controller/follow_joint_trajectory",
                                  callback_group=self._cb)
        print("Waiting for /arm_controller/follow_joint_trajectory ...")
        if not self._traj.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("Trajectory controller not available")

        if self._backend == "moveit":
            self._cart = self.create_client(GetCartesianPath,
                                            "/compute_cartesian_path",
                                            callback_group=self._cb)
            self._ik = self.create_client(GetPositionIK, "/compute_ik",
                                           callback_group=self._cb)
            print("Waiting for /compute_ik ...")
            if not self._ik.wait_for_service(timeout_sec=30.0):
                raise RuntimeError("/compute_ik unavailable — is MoveIt running?")
            print("Waiting for /compute_cartesian_path ...")
            if not self._cart.wait_for_service(timeout_sec=30.0):
                raise RuntimeError("/compute_cartesian_path unavailable")

        self._marker_pub = self.create_publisher(Marker, "/fork_acquisition/target_marker", 10)

        print(f"\n{'='*55}")
        print(f"  ForkAcquirer ready  —  backend: {self._backend.upper()}")
        print(f"{'='*55}\n")

    # ── Subscribers ───────────────────────────────────────────────────────────

    def _color_cb(self, msg):
        with self._lock:
            self._color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def _depth_cb(self, msg):
        with self._lock:
            self._depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def _js_cb(self, msg):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._joints:
                    self._joints[name] = pos

    def _bbox_cb(self, msg):
        if len(msg.data) != 4:
            return
        with self._lock:
            self._last_bbox = tuple(msg.data)
        # joints backend: trigger directly from bbox pixel — no base-frame round-trip
        if self._backend == "joints" and not self.busy and not self._confirming:
            self._confirming = True
            bbox = tuple(msg.data)
            threading.Thread(
                target=self._confirm_and_acquire_joints, args=(bbox,), daemon=True).start()

    def _xyz_cb(self, msg):
        # moveit backend only — base-frame point from food_segmenter
        if self._backend != "moveit" or self.busy or self._confirming:
            return
        self._confirming = True
        food_base = np.array([msg.x, msg.y, msg.z])
        print(f"dx={msg.x:+.3f}  dy={msg.y:+.3f}  dz={msg.z:+.3f} m")
        self._publish_marker(food_base)
        threading.Thread(
            target=self._confirm_and_acquire, args=(food_base,), daemon=True).start()

    # ── Pixel → camera-frame target (identical to click_pointer.process_click) ──

    def _pixel_to_cam(self, u: int, v: int, depth: np.ndarray):
        """Back-project bbox centre pixel to camera-frame 3-D with physical EE offset."""
        h, w = depth.shape[:2]
        r    = DEPTH_PATCH_PX
        y0 = max(0, v - r);  y1 = min(h, v + r + 1)
        x0 = max(0, u - r);  x1 = min(w, u + r + 1)
        patch = depth[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 10]
        if valid.size < 4:
            print(f"[WARN] no valid depth at ({u},{v})")
            return None
        z_raw = float(np.median(valid)) * 0.001   # mm → m
        if not (0.05 < z_raw < 1.5):
            print(f"[WARN] depth {z_raw*100:.1f} cm out of range")
            return None
        x_cam = (u - CX) * z_raw / FX
        y_cam = (v - CY) * z_raw / FY
        # apply same offsets as click_pointer — moves aim-point to physical EE tip
        target_cam = np.array([x_cam + CAM_X_OFFSET,
                                y_cam - CAM_Y_OFFSET,
                                z_raw + CAM_Z_OFFSET])
        print(f"pixel ({u},{v})  depth={z_raw*100:.1f} cm  "
              f"cam [{target_cam[0]:+.3f}, {target_cam[1]:+.3f}, {target_cam[2]:+.3f}] m")
        return target_cam

    # ── Entry point ───────────────────────────────────────────────────────────

    def acquire(self, u_min: int, v_min: int, u_max: int, v_max: int):
        if self.busy or self._confirming:
            return
        with self._lock:
            depth = self._depth_img
        if depth is None:
            print("[WARN] no depth image yet")
            return

        food_base = self._project_to_base(u_min, v_min, u_max, v_max, depth)
        if food_base is None:
            return

        self._publish_marker(food_base)
        threading.Thread(
            target=self._confirm_and_acquire, args=(food_base,), daemon=True).start()

    def _confirm_and_acquire(self, food_base: np.ndarray):
        try:
            self._show_bbox_window()
            try:
                ans = input("  RViz marker + bbox shown — proceed? [y/n]: ").strip().lower()
            except EOFError:
                ans = "n"
            cv2.destroyWindow("Fork Target")
            cv2.waitKey(1)
            if ans == "y":
                self._run_acquisition(food_base)
            else:
                self._delete_marker()
                print("  cancelled.")
        finally:
            self._confirming = False

    def _confirm_and_acquire_joints(self, bbox: tuple):
        """Joints-backend path: project bbox centre pixel → cam-frame → phased stepping."""
        try:
            with self._lock:
                depth = self._depth_img.copy() if self._depth_img is not None else None
            if depth is None:
                print("[WARN] no depth image yet")
                return
            x1, y1, x2, y2 = bbox
            u, v = (x1 + x2) // 2, (y1 + y2) // 2
            target_cam = self._pixel_to_cam(u, v, depth)
            if target_cam is None:
                return

            # ── preview: EE position + planned phase deltas ────────────────────
            try:
                tf_ee = self._tf.lookup_transform(
                    CAM_FRAME, EE_LINK, Time(), Duration(seconds=1.0))
                ee = np.array([tf_ee.transform.translation.x,
                               tf_ee.transform.translation.y,
                               tf_ee.transform.translation.z])
                print(f"EE  cam [{ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f}] m")
                dy     = target_cam[1] - ee[1]
                dz     = target_cam[2] - ee[2]
                dx_cam = target_cam[0]
                print(f"errors  dx={dx_cam:+.3f}  dy={dy:+.3f}  dz={dz:+.3f} m")
                dz_hover = dz - HOVER_HEIGHT
                dz_plunge = HOVER_HEIGHT + PENETRATE_DEPTH
                print(f"[1/5] Y-align  dy={dy:+.3f} m  → J3")
                print(f"[2/5] X-align  dx={dx_cam:+.3f} m  → J1")
                print(f"[3/5] Z-hover  dz={dz_hover:+.3f} m  → J1+J4")
                print(f"[4/5] plunge   dz={dz_plunge:.3f} m  → J2+J3")
                print(f"[5/5] retract")
            except Exception as e:
                print(f"[WARN] preview TF failed ({e}), proceeding anyway")

            self._show_bbox_window()
            try:
                ans = input("  proceed? [y/n]: ").strip().lower()
            except EOFError:
                ans = "n"
            cv2.destroyWindow("Fork Target")
            cv2.waitKey(1)
            if ans == "y":
                self.busy = True
                try:
                    self._run_joints(target_cam)
                except Exception as e:
                    print(f"[ERR] acquisition failed: {e}")
                finally:
                    self._delete_marker()
                    self.busy = False
            else:
                print("  cancelled.")
        finally:
            self._confirming = False

    def _show_bbox_window(self):
        with self._lock:
            frame = self._color_img.copy() if self._color_img is not None else None
            bbox  = self._last_bbox
        if frame is None:
            print("[WARN] no color frame yet")
            return
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 6, (0, 200, 0), -1)
            cv2.putText(frame, "fork target", (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        # suppress Qt/OpenCV stderr noise for the duration of the window
        _saved = os.dup(2)
        _null  = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_null, 2); os.close(_null)
        try:
            cv2.imshow("Fork Target", frame)
            for _ in range(5):
                cv2.waitKey(20)
        finally:
            os.dup2(_saved, 2); os.close(_saved)

    # ── Marker helpers ────────────────────────────────────────────────────────

    def _publish_marker(self, xyz: np.ndarray):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns = "fork_target"; m.id = 0
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x = float(xyz[0])
        m.pose.position.y = float(xyz[1])
        m.pose.position.z = float(xyz[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color.r = 1.0; m.color.g = 0.4; m.color.b = 0.0; m.color.a = 0.9
        m.lifetime.sec = 0
        self._marker_pub.publish(m)

    def _delete_marker(self):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns = "fork_target"; m.id = 0; m.action = Marker.DELETE
        self._marker_pub.publish(m)

    # ── Pixel → base-frame 3D ─────────────────────────────────────────────────

    def _project_to_base(self, u_min, v_min, u_max, v_max,
                         depth: np.ndarray) -> np.ndarray | None:
        u_c = (u_min + u_max) // 2
        v_c = (v_min + v_max) // 2
        h, w = depth.shape[:2]
        r    = DEPTH_PATCH_PX

        u0 = max(u_min + 4, u_c - r, 0)
        u1 = min(u_max - 4, u_c + r, w - 1)
        v0 = max(v_min + 4, v_c - r, 0)
        v1 = min(v_max - 4, v_c + r, h - 1)

        patch = depth[v0:v1+1, u0:u1+1].astype(np.float32)
        valid = patch[patch > 10]
        if valid.size < 4:
            print("[ERR] depth patch has no valid readings")
            return None

        z_m = float(np.median(valid)) * 0.001
        if not (0.05 < z_m < 1.5):
            print(f"[ERR] depth {z_m*100:.1f} cm out of range")
            return None

        x_cam = (u_c - CX) * z_m / FX
        y_cam = (v_c - CY) * z_m / FY
        pt_cam = np.array([x_cam, y_cam, z_m])
        print(f"pixel ({u_c},{v_c})  depth={z_m*100:.1f} cm  "
              f"cam [{pt_cam[0]:+.3f}, {pt_cam[1]:+.3f}, {pt_cam[2]:+.3f}] m")

        try:
            tf  = self._tf.lookup_transform(BASE_FRAME, CAM_FRAME,
                                            Time(), Duration(seconds=1.0))
            R   = _quat_to_matrix(tf.transform.rotation)
            t   = np.array([tf.transform.translation.x,
                            tf.transform.translation.y,
                            tf.transform.translation.z])
            pt_base = R @ pt_cam + t
        except Exception as e:
            print(f"[ERR] TF cam→base failed: {e}")
            return None

        reach = math.sqrt(pt_base[0]**2 + pt_base[1]**2)
        print(f"base [{pt_base[0]:+.3f}, {pt_base[1]:+.3f}, {pt_base[2]:+.3f}] m  "
              f"reach={reach:.3f} m")
        return pt_base

    # ── Current EE orientation ────────────────────────────────────────────────

    def _get_ee_orientation(self):
        try:
            tf = self._tf.lookup_transform(BASE_FRAME, EE_LINK,
                                           Time(), Duration(seconds=1.0))
            return tf.transform.rotation
        except Exception as e:
            print(f"[ERR] TF EE→base failed: {e}")
            return None

    # ── Stage 1: hover approach (joint-space IK) ──────────────────────────────

    def _move_to_hover(self, hover_xyz, q) -> bool:
        with self._lock:
            current_joints = {n: self._joints[n] for n in ARM_JOINT_NAMES}

        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name       = ARM_GROUP
        ik_req.ik_request.ik_link_name     = EE_LINK
        ik_req.ik_request.avoid_collisions = True
        ik_req.ik_request.timeout.sec      = 5
        ik_req.ik_request.pose_stamped.header.frame_id = BASE_FRAME
        ik_req.ik_request.pose_stamped.header.stamp    = self.get_clock().now().to_msg()
        ik_req.ik_request.pose_stamped.pose.position.x = float(hover_xyz[0])
        ik_req.ik_request.pose_stamped.pose.position.y = float(hover_xyz[1])
        ik_req.ik_request.pose_stamped.pose.position.z = float(hover_xyz[2])
        ik_req.ik_request.pose_stamped.pose.orientation.x = float(q.x)
        ik_req.ik_request.pose_stamped.pose.orientation.y = float(q.y)
        ik_req.ik_request.pose_stamped.pose.orientation.z = float(q.z)
        ik_req.ik_request.pose_stamped.pose.orientation.w = float(q.w)
        ik_req.ik_request.robot_state.joint_state.name     = ARM_JOINT_NAMES
        ik_req.ik_request.robot_state.joint_state.position = [current_joints[n]
                                                               for n in ARM_JOINT_NAMES]

        ik_future = self._ik.call_async(ik_req)
        ik_done = threading.Event()
        ik_future.add_done_callback(lambda _: ik_done.set())
        if not ik_done.wait(timeout=10.0):
            print("[ERR] IK timed out")
            return False

        ik_result = ik_future.result()
        if ik_result.error_code.val != MoveItErrorCodes.SUCCESS:
            print(f"[ERR] IK failed — code {ik_result.error_code.val}")
            return False

        js     = ik_result.solution.joint_state
        ik_map = dict(zip(js.name, js.position))
        print("[hover] IK solved — executing")

        cur_pos = [current_joints[n] for n in ARM_JOINT_NAMES]
        raw_tgt = [ik_map.get(n, current_joints[n]) for n in ARM_JOINT_NAMES]
        tgt_pos = []
        for cur, tgt in zip(cur_pos, raw_tgt):
            diff = (tgt - cur + math.pi) % (2 * math.pi) - math.pi
            tgt_pos.append(cur + diff)

        max_delta = max(abs(a - b) for a, b in zip(tgt_pos, cur_pos))
        duration  = max(6.0, max_delta / 0.05)

        traj = _build_trajectory(cur_pos, tgt_pos, duration)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        done = threading.Event()
        ok   = [False]

        def _gr(f):
            gh = f.result()
            if not gh.accepted:
                print("[hover] trajectory rejected")
                done.set()
                return
            def _rr(_): ok[0] = True; done.set()
            gh.get_result_async().add_done_callback(_rr)

        self._traj.send_goal_async(goal).add_done_callback(_gr)
        done.wait(timeout=duration + 15.0)
        return ok[0]

    # ── Stage 2: Cartesian plunge / retract ───────────────────────────────────

    def _cartesian_path(self, waypoints, speed) -> JointTrajectory | None:
        req = GetCartesianPath.Request()
        req.header.frame_id  = BASE_FRAME
        req.header.stamp     = self.get_clock().now().to_msg()
        req.group_name       = ARM_GROUP
        req.link_name        = EE_LINK
        req.waypoints        = waypoints
        req.max_step         = 0.004
        req.jump_threshold   = 0.0
        req.avoid_collisions = False

        future = self._cart.call_async(req)
        done   = threading.Event()
        future.add_done_callback(lambda _: done.set())

        if not done.wait(timeout=CART_TIMEOUT_S):
            print("[ERR] GetCartesianPath timed out")
            return None

        resp = future.result()
        if resp.fraction < MIN_PATH_FRAC:
            print(f"[ERR] Cartesian path {resp.fraction*100:.0f}% feasible "
                  f"(need {MIN_PATH_FRAC*100:.0f}%)")
            return None

        print(f"Cartesian path {resp.fraction*100:.0f}%")
        return _stamp_trajectory(resp.solution.joint_trajectory, speed)

    def _execute(self, traj: JointTrajectory, label: str) -> bool:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        done    = threading.Event()
        success = [False]

        def _on_goal(f):
            gh = f.result()
            if not gh.accepted:
                print(f"[{label}] trajectory rejected")
                done.set()
                return
            def _on_result(rf):
                success[0] = (rf.result().result.error_code == 0)
                done.set()
            gh.get_result_async().add_done_callback(_on_result)

        self._traj.send_goal_async(goal).add_done_callback(_on_goal)
        done.wait(timeout=EXEC_TIMEOUT_S)
        return success[0]

    # ── Acquisition dispatch ──────────────────────────────────────────────────

    def _run_acquisition(self, food_base: np.ndarray):
        self.busy = True
        try:
            if self._backend == "moveit":
                self._run_moveit(food_base)
            else:
                self._run_joints(food_base)
        except Exception as e:
            print(f"[ERR] acquisition failed: {e}")
        finally:
            self._delete_marker()
            self.busy = False

    # ── Cam-frame entry point ─────────────────────────────────────────────────

    def move_to_cam_point(self, pt_cam: np.ndarray) -> bool:
        """Move EE to a 3D point specified in camera_color_optical_frame."""
        try:
            tf = self._tf.lookup_transform(
                BASE_FRAME, CAM_FRAME, Time(), Duration(seconds=1.0))
            R = _quat_to_matrix(tf.transform.rotation)
            t = np.array([tf.transform.translation.x,
                          tf.transform.translation.y,
                          tf.transform.translation.z])
            pt_base = R @ pt_cam + t
        except Exception as e:
            self.get_logger().error(f"TF cam→base failed: {e}")
            return False

        self._publish_marker(pt_base)
        self._run_moveit(pt_base)
        return True

    # ── Backend A: MoveIt ─────────────────────────────────────────────────────

    def _run_moveit(self, food_base: np.ndarray):
        q = self._get_ee_orientation()
        if q is None:
            return

        hover_xyz  = food_base.copy(); hover_xyz[2]  = food_base[2] + HOVER_HEIGHT
        plunge_xyz = food_base.copy(); plunge_xyz[2] = food_base[2] 

        dx = food_base[0]; dy = food_base[1]; dz = food_base[2]
        print(f"food  dx={dx:+.3f}  dy={dy:+.3f}  dz={dz:+.3f} m")

        print(f"[1/3] hover    z={hover_xyz[2]:+.3f} m")
        if not self._move_to_hover(hover_xyz, q):
            print("[ERR] hover failed — aborting")
            return

        q = self._get_ee_orientation()
        if q is None:
            return

        print(f"[2/3] plunge   z={plunge_xyz[2]:+.3f} m")
        traj = self._cartesian_path(
            [_make_pose(hover_xyz, q), _make_pose(plunge_xyz, q)], PLUNGE_SPEED)
        if traj is None:
            print("[ERR] plunge path failed — aborting")
            return
        self._execute(traj, "plunge")
        time.sleep(0.25)

        print(f"[3/3] retract  z={hover_xyz[2]:+.3f} m")
        traj = self._cartesian_path(
            [_make_pose(plunge_xyz, q), _make_pose(hover_xyz, q)], CARTESIAN_SPEED)
        if traj:
            self._execute(traj, "retract")

        print("fork acquisition complete.")

    # ── Backend B: joint stepping (exact algorithm from click_pointer.py) ────────

    def _run_joints(self, target_cam: np.ndarray):
        """
        Phased proportional joint stepping — same control law as click_pointer._perform_phased.

        target_cam: 3-D point in camera frame, already offset to physical EE tip
                    (produced by _pixel_to_cam which applies CAM_X/Y/Z_OFFSET).

        Phases:
          1  Y-error  → J2 (elbow pitch)       — largest error, settled first
          2  X-error  → J3 (forearm rotation)  — camera-centred X, zero when target centred
          3  Z-hover  → J1+J4 (approach)       — stop HOVER_HEIGHT short of food
          4  Plunge   → J1+J4 (fork into food)
          5  Retract  → J1+J4 (lift out)
        """
        try:
            tf_ee_cam = self._tf.lookup_transform(
                CAM_FRAME, EE_LINK, Time(), Duration(seconds=1.0))
        except Exception as e:
            print(f"[ERR] TF EE→cam failed: {e}")
            return

        ee = np.array([tf_ee_cam.transform.translation.x,
                       tf_ee_cam.transform.translation.y,
                       tf_ee_cam.transform.translation.z])
        print(f"EE  cam [{ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f}] m")

        # errors in camera frame — same decomposition as click_pointer._perform_phased
        dy     = target_cam[1] - ee[1]   # vertical: cam-Y points down
        dz     = target_cam[2] - ee[2]   # depth: cam-Z points forward
        dx_cam = target_cam[0]            # horizontal: already cam-centred

        print(f"errors  dx={dx_cam:+.3f}  dy={dy:+.3f}  dz={dz:+.3f} m")

        # ── Phase 1: Y → J3 ───────────────────────────────────────────────────
        print(f"[1/5] Y-align  dy={dy:+.3f} m  → J3")
        if abs(dy) > ERR_TOL_M:
            delta = -(dy / CART_DELTA_Y) * STEP_RADS
            self._step({2: delta}, max(1.5, abs(dy) / CART_DELTA_Y * TICK_DUR_S),
                       "Y-align → J3")

        # ── Phase 2: X → J1 ───────────────────────────────────────────────────
        print(f"[2/5] X-align  dx={dx_cam:+.3f} m  → J1")
        if abs(dx_cam) > ERR_TOL_M:
            delta = -(dx_cam / CART_DELTA_X) * STEP_RADS
            self._step({0: delta}, max(1.5, abs(dx_cam) / CART_DELTA_X * TICK_DUR_S),
                       "X-align → J1")

        # ── Phase 3: Z hover approach → J1+J4 ─────────────────────────────────
        dz_hover = dz - HOVER_HEIGHT   # stop HOVER_HEIGHT short so we don't ram food
        print(f"[3/5] Z-hover  dz={dz_hover:+.3f} m  → J1+J4")
        if dz_hover > ERR_TOL_M:
            delta = (dz_hover / CART_DELTA_Z) * STEP_RADS * 2.0
            self._step({0: delta, 3: delta},
                       max(1.5, dz_hover / CART_DELTA_Z * TICK_DUR_S),
                       "Z-hover → J1+J4")

        # ── Phase 4: Plunge → J2+J3 (Z via elbow+forearm, not base+wrist) ───────
        dz_plunge = HOVER_HEIGHT + PENETRATE_DEPTH + 0.1
        plunge_j  = (dz_plunge / CART_DELTA_Z) * STEP_RADS * 2.0
        print(f"[4/5] plunge   dz={dz_plunge:.3f} m  → J2+J3")
        self._step({1: plunge_j, 2: plunge_j},
                   max(0.8, dz_plunge / CART_DELTA_Z * TICK_DUR_S * 0.5),
                   "Plunge → J2+J3")
        time.sleep(0.3)

        # ── Phase 5: Retract → J2+J3 ──────────────────────────────────────────
        print(f"[5/5] retract")
        self._step({1: -plunge_j, 2: -plunge_j},
                   max(1.0, dz_plunge / CART_DELTA_Z * TICK_DUR_S),
                   "Retract → J2+J3")

        print("fork acquisition complete.")

    _SPIN = ['|', '/', '-', '\\']

    def _step(self, deltas: dict, duration_s: float, label: str = "moving"):
        with self._lock:
            cur = [self._joints[n] for n in ARM_JOINT_NAMES]
        tgt = list(cur)
        for idx, d in deltas.items():
            tgt[idx] += d

        pt0 = JointTrajectoryPoint()
        pt0.positions       = cur
        pt0.time_from_start = DurationMsg(sec=0, nanosec=0)
        s   = int(duration_s)
        pt1 = JointTrajectoryPoint()
        pt1.positions       = tgt
        pt1.time_from_start = DurationMsg(sec=s, nanosec=int((duration_s - s) * 1e9))

        traj              = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = ARM_JOINT_NAMES
        traj.points       = [pt0, pt1]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        done = threading.Event()

        def _on_goal(f):
            gh = f.result()
            if gh.accepted:
                gh.get_result_async().add_done_callback(lambda _: done.set())
            else:
                done.set()

        self._traj.send_goal_async(goal).add_done_callback(_on_goal)

        t0 = time.time()
        i  = 0
        while not done.wait(timeout=0.08):
            elapsed = time.time() - t0
            filled  = min(20, int(20 * elapsed / max(duration_s, 0.01)))
            bar     = '█' * filled + '░' * (20 - filled)
            print(f"\r  {self._SPIN[i % 4]}  [{bar}]  {label}  {elapsed:.1f}s / {duration_s:.1f}s",
                  end='', flush=True)
            i += 1
        print()


def main():
    ap = argparse.ArgumentParser(description="Fork acquisition node")
    ap.add_argument("--backend", choices=["moveit", "joints"], default=None)
    args, ros_args = ap.parse_known_args()

    if args.backend is None:
        print("\n" + "="*55)
        print("  Fork Acquisition — choose motion backend")
        print("="*55)
        print("  [1] moveit  — MoveGroup + Cartesian IK")
        print("  [2] joints  — Direct joint stepping")
        print("="*55)
        while True:
            choice = input("  Enter 1 or 2: ").strip()
            if choice == "1":   args.backend = "moveit"; break
            elif choice == "2": args.backend = "joints"; break
            print("  Please enter 1 or 2.")

    print(f"\n  Starting with backend: {args.backend.upper()}\n")

    rclpy.init(args=ros_args if ros_args else None)
    node = ForkAcquirer(backend=args.backend)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
