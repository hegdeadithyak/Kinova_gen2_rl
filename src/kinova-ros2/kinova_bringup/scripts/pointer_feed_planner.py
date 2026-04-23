#!/usr/bin/env python3
"""
pointer_feed_planner.py — ROS2 port of SingleClickDepthXYZPointer.

Mouth replaces mouse click.  Algorithm mirrors the ROS1 version exactly:

  pointer  = ee_pos + R @ SPOON_OFFSET        (spoon tip in base frame)
  err_base = target - pointer
  err_tool = R.T @ err_base
  step_tool = err_tool/|err| * min(STEP_SIZE,|err|)
  step_tool[:2] clamped to MAX_LATERAL_M      (safety — mirrors ROS1 clamp)
  step_base = R @ step_tool
  new_ee    = ee_pos + step_base

Speed: the EE target is computed ONCE at trigger time as
  ee_target = mouth_approach - R_init @ SPOON_OFFSET
and we step toward that fixed point, reusing the Jacobian every N steps.
This gives ~4 FK calls per step instead of 7, halving loop time.

SPOON_OFFSET: vector from EE frame origin to spoon/fork tip, in EE frame.
  Start with [0,0,0] — pure EE homing — then tune once the arm reaches the
  correct position visually.

Call /feed_trigger to start.
"""

import copy
import math
import time
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_point
from tf2_ros import (Buffer, TransformListener,
                     LookupException, ExtrapolationException, ConnectivityException)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ── Frame / joint names ──────────────────────────────────────────────────
BASE_FRAME = 'root'
EE_LINK    = 'j2s6s200_end_effector'
CAM_FRAME  = 'camera_color_optical_frame'

ARM_JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]

# Only J1–J3 are used in the per-step Jacobian (they control 3-D position).
# Reduces FK calls from 7 → 4 per iteration, halving loop latency.
JACOBIAN_JOINTS = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
]

# ── Spoon / tool tip offset in EE frame (metres) ─────────────────────────
# Same meaning as POINTER_OFFSET in your ROS1 node.
# Set to [0,0,0] to home the EE directly, then tune once motion looks correct.
SPOON_OFFSET = np.array([0.0, 0.0, 0.0])

# ── Approach: stop this far short of the mouth ───────────────────────────
APPROACH_M = 0.10   # 10 cm gap (spoon length + safety)

# ── Step-controller ──────────────────────────────────────────────────────
STEP_SIZE      = 0.03    # m — Cartesian step per iteration (3 cm, was 1 cm)
STOP_THRESHOLD = 0.015   # m — convergence criterion
MAX_LATERAL_M  = 0.015   # lateral clamp in tool frame (mirrors ROS1)

# ── Jacobian rebuild frequency ────────────────────────────────────────────
# Recompute Jacobian every N steps.  N=1 = fresh every step (accurate, slow).
# N=3 = fast — valid because steps are small and J doesn't change much.
JACOBIAN_REBUILD_EVERY = 3

# ── Trajectory timing ────────────────────────────────────────────────────
STEP_DURATION_S = 0.50   # seconds per step trajectory
MAX_JOINT_VEL   = 0.35   # rad/s safety cap
RETURN_DUR_S    = 5.0
HOLD_SECONDS    = 5.0

# ── Numerical Jacobian ───────────────────────────────────────────────────
FK_PERTURB_RAD = 0.05
DELTA_CLIP_RAD = 0.40


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


class PointerFeedPlanner(Node):

    def __init__(self):
        super().__init__('pointer_feed_planner')
        self._cb = ReentrantCallbackGroup()

        self._tf_buf      = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        self._latest_mouth: Optional[PointStamped] = None
        self._latest_js:    Optional[JointState]   = None
        self._busy = False

        self.create_subscription(PointStamped, '/mouth_3d_point',
                                 self._mouth_cb, 10, callback_group=self._cb)
        self.create_subscription(JointState, '/joint_states',
                                 self._js_cb, 10, callback_group=self._cb)

        self._fk_client = self.create_client(
            GetPositionFK, '/compute_fk', callback_group=self._cb)
        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            callback_group=self._cb)

        self.create_service(Trigger, '/feed_trigger', self._trigger_cb,
                            callback_group=self._cb)

        self.get_logger().info('Waiting for services …')
        if not self._fk_client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError('Timed out waiting for /compute_fk')
        if not self._traj_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError('Timed out waiting for trajectory controller')
        self.get_logger().info('PointerFeedPlanner ready — call /feed_trigger')

    # ── Callbacks ──────────────────────────────────────────────────────
    def _mouth_cb(self, msg):
        self._latest_mouth = msg

    def _js_cb(self, msg):
        self._latest_js = msg

    def _trigger_cb(self, _req, response):
        if self._busy:
            response.success = False
            response.message = 'Feed already in progress.'
            return response
        if self._latest_mouth is None:
            response.success = False
            response.message = 'No mouth point — is mouth_tracker running?'
            return response
        self._busy = True
        try:
            ok, msg = self._execute_feed()
            response.success = ok
            response.message  = msg
        except Exception as exc:
            response.success = False
            response.message  = f'Exception: {exc}'
            self.get_logger().error(response.message)
        finally:
            self._busy = False
        return response

    # ── Core feed ───────────────────────────────────────────────────────
    def _execute_feed(self) -> Tuple[bool, str]:
        if self._latest_js is None:
            return False, 'No joint states received yet.'

        # ── Mouth → base frame ────────────────────────────────────────
        snap      = self._latest_mouth
        src_frame = snap.header.frame_id or CAM_FRAME
        stamp     = Time.from_msg(snap.header.stamp)
        try:
            tf_cam = self._lookup_tf(BASE_FRAME, src_frame, stamp=stamp)
        except RuntimeError:
            self.get_logger().warn('Timestamped TF miss; using latest.')
            try:
                tf_cam = self._lookup_tf(BASE_FRAME, src_frame)
            except RuntimeError as e:
                return False, str(e)

        mouth_pt  = do_transform_point(snap, tf_cam).point
        mx, my, mz = mouth_pt.x, mouth_pt.y, mouth_pt.z
        self.get_logger().info(f'Mouth (base): ({mx:+.3f},{my:+.3f},{mz:+.3f})')

        # ── Approach point: APPROACH_M back from mouth ────────────────
        horiz = math.sqrt(mx*mx + my*my)
        if horiz < 1e-3:
            return False, 'Mouth directly above base — cannot approach.'
        ux, uy  = mx / horiz, my / horiz
        approach = np.array([mx - ux*APPROACH_M, my - uy*APPROACH_M, mz])
        self.get_logger().info(
            f'Approach (base): ({approach[0]:+.3f},{approach[1]:+.3f},{approach[2]:+.3f})')

        # ── EE target = approach − R_init @ SPOON_OFFSET ─────────────
        # Compute once using current EE orientation; fixed for entire feed.
        fk_init = self._fk_pose(self._latest_js)
        if fk_init is None:
            return False, 'Initial FK failed.'
        ee_pos_init, ee_quat_init = fk_init
        R_init = _quat_to_R(*ee_quat_init)
        ee_target = approach - R_init @ SPOON_OFFSET

        self.get_logger().info(
            f'EE target    (base): ({ee_target[0]:+.3f},{ee_target[1]:+.3f},{ee_target[2]:+.3f})')
        self.get_logger().info(
            f'EE current   (base): ({ee_pos_init[0]:+.3f},{ee_pos_init[1]:+.3f},{ee_pos_init[2]:+.3f})')
        self.get_logger().info(
            f'Distance EE→target:  {np.linalg.norm(ee_target - ee_pos_init)*100:.1f} cm')

        start_js = copy.deepcopy(self._latest_js)

        # ── Step loop ─────────────────────────────────────────────────
        # J_cached holds the last Jacobian so we don't recompute every step.
        J_cached:    Optional[np.ndarray] = None
        ptr_cached:  Optional[np.ndarray] = None
        max_iters    = int(60.0 / STEP_DURATION_S)

        for iteration in range(max_iters):
            fk_cur = self._fk_pose(self._latest_js)
            if fk_cur is None:
                self._safe_return(start_js)
                return False, f'FK failed at iteration {iteration}.'
            ee_cur, ee_quat_cur = fk_cur
            R_cur = _quat_to_R(*ee_quat_cur)

            pointer = ee_cur + R_cur @ SPOON_OFFSET

            # Error in tool frame (same as ROS1 algorithm)
            err_base = ee_target - pointer
            err_tool = R_cur.T @ err_base
            dist     = float(np.linalg.norm(err_tool))

            # Also show physical EE-to-mouth distance for intuition
            ee_to_mouth = float(np.linalg.norm(np.array([mx,my,mz]) - ee_cur))
            self.get_logger().info(
                f'[iter {iteration:3d}] pointer→target={dist*100:5.1f} cm  '
                f'EE→mouth={ee_to_mouth*100:5.1f} cm  '
                f'err_tool=({err_tool[0]:+.3f},{err_tool[1]:+.3f},{err_tool[2]:+.3f})')

            if dist < STOP_THRESHOLD:
                self.get_logger().info('Reached approach target.')
                break

            # Step in tool frame, clamped laterally
            step_tool    = err_tool / dist * min(STEP_SIZE, dist)
            step_tool[0] = float(np.clip(step_tool[0], -MAX_LATERAL_M, MAX_LATERAL_M))
            step_tool[1] = float(np.clip(step_tool[1], -MAX_LATERAL_M, MAX_LATERAL_M))
            step_base    = R_cur @ step_tool
            new_ee       = ee_cur + step_base

            # Rebuild Jacobian only every N steps
            rebuild = (J_cached is None or
                       iteration % JACOBIAN_REBUILD_EVERY == 0)
            if rebuild:
                J_cached, ptr_cached = self._build_jacobian(ee_cur, R_cur)
                if J_cached is None:
                    self._safe_return(start_js)
                    return False, f'Jacobian build failed at iteration {iteration}.'

            # Δpointer = new_ee - ee_cur (R is approximately constant per step)
            dp = new_ee - ee_cur
            dq = np.linalg.pinv(J_cached) @ dp
            dq = np.clip(dq, -DELTA_CLIP_RAD, DELTA_CLIP_RAD)

            new_js  = copy.deepcopy(self._latest_js)
            new_pos = [float(p) for p in new_js.position]
            js_names = list(new_js.name)
            for jname, delta in zip(JACOBIAN_JOINTS, dq):
                if jname in js_names:
                    new_pos[js_names.index(jname)] += float(delta)
            new_js.position = new_pos

            if not self._move_to_joints(new_js, min_dur=STEP_DURATION_S):
                self._safe_return(start_js)
                return False, f'Trajectory failed at iteration {iteration}.'

        else:
            self.get_logger().warn('Step loop hit max iterations.')

        # ── Hold ──────────────────────────────────────────────────────
        self.get_logger().info(f'Holding {HOLD_SECONDS:.0f} s …')
        time.sleep(HOLD_SECONDS)

        # ── Return ────────────────────────────────────────────────────
        ok = self._safe_return(start_js)
        return True, 'Feed complete.' if ok else 'Feed complete (return failed).'

    # ── Build Jacobian of pointer position for JACOBIAN_JOINTS ──────────
    def _build_jacobian(
            self,
            ee_cur: np.ndarray,
            R_cur:  np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._latest_js is None:
            return None, None
        js_names = list(self._latest_js.name)
        js_pos   = [float(p) for p in self._latest_js.position]
        ptr_cur  = ee_cur + R_cur @ SPOON_OFFSET

        J_cols = []
        for jname in JACOBIAN_JOINTS:
            if jname not in js_names:
                self.get_logger().error(f'Joint {jname} not in states.')
                return None, None
            idx        = js_names.index(jname)
            perturbed  = js_pos.copy()
            perturbed[idx] += FK_PERTURB_RAD

            p_js          = JointState()
            p_js.name     = js_names
            p_js.position = perturbed

            fk_pert = self._fk_pose(p_js)
            if fk_pert is None:
                self.get_logger().error(f'FK failed for {jname}.')
                return None, None

            ee_pert, q_pert = fk_pert
            R_pert          = _quat_to_R(*q_pert)
            ptr_pert        = ee_pert + R_pert @ SPOON_OFFSET
            J_cols.append((ptr_pert - ptr_cur) / FK_PERTURB_RAD)

        return np.array(J_cols).T, ptr_cur   # (3, N)

    # ── FK: (pos_np, (qx,qy,qz,qw)) or None ────────────────────────────
    def _fk_pose(self, joint_state: JointState
                 ) -> Optional[Tuple[np.ndarray, tuple]]:
        req = GetPositionFK.Request()
        req.header.frame_id = BASE_FRAME
        req.fk_link_names   = [EE_LINK]
        rs                  = RobotState()
        rs.joint_state      = joint_state
        req.robot_state     = rs

        future = self._fk_client.call_async(req)
        if not self._wait(future, timeout_sec=3.0):
            self.get_logger().error('FK timed out.')
            return None
        res = future.result()
        if res is None or res.error_code.val != 1 or not res.pose_stamped:
            self.get_logger().error(
                f'FK failed, code: {res.error_code.val if res else "None"}')
            return None
        pose = res.pose_stamped[0].pose
        pos  = np.array([pose.position.x, pose.position.y, pose.position.z])
        q    = pose.orientation
        return pos, (q.x, q.y, q.z, q.w)

    # ── Trajectory execution ─────────────────────────────────────────────
    def _move_to_joints(self, target_js: JointState,
                        min_dur: float = STEP_DURATION_S) -> bool:
        names:     List[str]   = []
        positions: List[float] = []
        for name, pos in zip(target_js.name, target_js.position):
            if name in ARM_JOINT_NAMES:
                names.append(name)
                positions.append(float(pos))
        if not names:
            return False

        duration = min_dur
        cur_positions: List[float] = []
        if self._latest_js is not None:
            cur_map = dict(zip(self._latest_js.name, self._latest_js.position))
            for name, tgt in zip(names, positions):
                if name in cur_map:
                    delta    = abs(tgt - float(cur_map[name]))
                    duration = max(duration, delta / MAX_JOINT_VEL)
            cur_positions = [float(cur_map.get(n, p)) for n, p in zip(names, positions)]
        else:
            cur_positions = list(positions)

        sec  = int(duration)
        nsec = int((duration - sec) * 1e9)

        pt_start = JointTrajectoryPoint()
        pt_start.positions      = cur_positions
        pt_start.time_from_start = DurationMsg(sec=0, nanosec=0)

        pt_end = JointTrajectoryPoint()
        pt_end.positions      = positions
        pt_end.time_from_start = DurationMsg(sec=sec, nanosec=nsec)

        traj              = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = names
        traj.points       = [pt_start, pt_end]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = self._traj_client.send_goal_async(goal)
        if not self._wait(future, timeout_sec=5.0):
            return False
        gh = future.result()
        if gh is None or not gh.accepted:
            return False

        res_f = gh.get_result_async()
        if not self._wait(res_f, timeout_sec=duration + 10.0):
            return False
        rw = res_f.result()
        return rw is not None and rw.result.error_code == 0

    def _safe_return(self, start_js: Optional[JointState]) -> bool:
        if start_js is None:
            return False
        return self._move_to_joints(start_js, min_dur=RETURN_DUR_S)

    # ── TF ───────────────────────────────────────────────────────────────
    def _lookup_tf(self, target: str, source: str,
                   timeout: float = 2.0, stamp=None):
        if stamp is None:
            stamp = Time()
        try:
            return self._tf_buf.lookup_transform(
                target, source, stamp, timeout=Duration(seconds=timeout))
        except ConnectivityException as e:
            frames = self._tf_buf.all_frames_as_string()
            self.get_logger().error(
                f'TF broken {source}→{target}\nFrames:\n{frames}')
            raise RuntimeError(f'TF {source}→{target}: {e}') from e
        except (LookupException, ExtrapolationException) as e:
            raise RuntimeError(f'TF {source}→{target}: {e}') from e

    def _wait(self, future, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while not future.done():
            if time.time() > deadline:
                return False
            time.sleep(0.01)
        return True


def main(args=None):
    rclpy.init(args=args)
    node = PointerFeedPlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
