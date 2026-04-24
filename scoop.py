#!/usr/bin/env python3
"""
feeding_loop.py

Orchestrates a 3-cycle feeding sequence on the Kinova j2s6s200:

  for cycle in 1..3:
      MOVE  → SCOOP_START          (joint-space, direct to bridge)
      SCOOP → joint-5 ramp          (joint-space, direct to bridge)
      MOVE  → TRANSIT               (joint-space, direct to bridge)
      MOVE  → FEEDING               (Cartesian + orientation-locked via MoveIt;
                                     falls back to joint-space if planner fails)
      DWELL 5 s
      MOVE  → TRANSIT               (Cartesian + orientation-locked; same fallback)

Joint-space moves bypass MoveIt entirely and go straight to the bridge's
FollowJointTrajectory action. Cartesian moves use MoveIt's
compute_cartesian_path service to GENERATE the trajectory, then dispatch the
resulting joint trajectory directly to the bridge — bypassing
move_group's controller manager (which is the source of every -4 error
we've been chasing).
"""

import math
import sys
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint
from geometry_msgs.msg import PoseStamped, Pose


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]
NJ = len(JOINT_NAMES)

# Captured live from /j2s6s200_driver/out/joint_state — DO NOT convert from deg
SCOOP_START_RAD: List[float] = [
    6.318579250698958,
    3.851496857986220,
    1.031682174422878,
    4.466520622226394,
    3.192789795017694,
    6.220801823908915,
]

TRANSIT_RAD: List[float] = [
    6.318555814881365,
    3.919515323807389,
    1.032239574039031,
    4.559317937777987,
    4.822317345964326,
    6.220804487070007,
]

FEEDING_RAD: List[float] = [
    3.072234617740315,   # j2s6s200_joint_1
    3.889442376145996,   # j2s6s200_joint_2
    0.9398291492147997,  # j2s6s200_joint_3
    1.4061254865861905,  # j2s6s200_joint_4
    4.960991872350278,   # j2s6s200_joint_5
    6.566627672525591,   # j2s6s200_joint_6
]

# Joint-5 scoop ramp: (time_s, j5_deg). All other joints pinned.
SCOOP_J5_RAMP_DEG = [
    (0.00, 192.63),
    (1.00, 192.63),
    (1.20, 192.75),
    (1.40, 196.30),
    (1.60, 201.85),
    (1.80, 207.78),
    (2.00, 213.93),
    (2.20, 220.18),
    (2.40, 226.58),
    (2.60, 232.97),
    (2.80, 239.39),
    (3.00, 245.75),
    (3.20, 250.82),
    (3.40, 251.15),
    (5.40, 251.18),
]
DEG2RAD = math.pi / 180.0
J5 = 4

# Move durations (seconds). Sized for the deltas; conservative.
DUR_TO_SCOOP_START = 8.0
DUR_TO_TRANSIT     = 4.0
DUR_TO_FEEDING     = 10.0     # ~125 deg sweep on j1 — needs time
DWELL_AT_FEEDING   = 5.0
NUM_CYCLES         = 3

# Cartesian planning parameters
CARTESIAN_EEF_STEP   = 0.01      # 1 cm interpolation step
CARTESIAN_JUMP_THR   = 0.0       # disabled (only meaningful in MoveIt 1)
CARTESIAN_MIN_FRAC   = 0.90      # require at least 90% of path to be valid
ORIENTATION_TOL_RAD  = 0.35      # ~20 deg per axis — generous; tighten later
EEF_LINK             = 'j2s6s200_link_6'
PLANNING_FRAME       = 'root'    # confirm with `ros2 run tf2_ros tf2_echo root j2s6s200_link_6`
PLANNING_GROUP       = 'arm'     # must match SRDF

ACTION_TOPIC      = '/arm_controller/follow_joint_trajectory'
JOINT_STATE_TOPIC = '/j2s6s200_driver/out/joint_state'


# ---------------------------------------------------------------------------
class FeedingOrchestrator(Node):
    def __init__(self):
        super().__init__('feeding_orchestrator')

        self._traj_client = ActionClient(self, FollowJointTrajectory, ACTION_TOPIC)

        self._latest_q: Optional[List[float]] = None
        self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._js_cb, 10
        )

        self._cart_cli = self.create_client(GetCartesianPath, 'compute_cartesian_path')
        self._fk_cli   = self.create_client(GetPositionFK,    'compute_fk')

    # ------------------------------------------------------------------
    def _js_cb(self, msg: JointState):
        try:
            idx = [msg.name.index(j) for j in JOINT_NAMES]
        except ValueError:
            return
        self._latest_q = [msg.position[i] for i in idx]

    def _wait_for_state(self, timeout=3.0) -> bool:
        t0 = time.monotonic()
        while self._latest_q is None and (time.monotonic() - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self._latest_q is not None

    # ------------------------------------------------------------------
    # Joint-space single-waypoint move (direct to bridge)
    # ------------------------------------------------------------------
    def move_joint_space(self, target_rad: List[float], duration_s: float,
                         label: str) -> bool:
        self.get_logger().info(
            f'[{label}] joint-space move over {duration_s:.1f}s'
        )
        if not self._traj_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Trajectory action server not available')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions  = list(target_rad)
        pt.velocities = [0.0] * NJ
        sec = int(duration_s)
        pt.time_from_start = Duration(sec=sec, nanosec=int((duration_s - sec) * 1e9))
        goal.trajectory.points.append(pt)

        return self._send_and_wait(goal, label)

    # ------------------------------------------------------------------
    # Joint-5-only scoop sweep
    # ------------------------------------------------------------------
    def execute_scoop(self) -> bool:
        if not self._wait_for_state():
            self.get_logger().error('No joint state for scoop build')
            return False
        q_now = list(self._latest_q)
        ramp = list(SCOOP_J5_RAMP_DEG)
        ramp[0] = (ramp[0][0], q_now[J5] / DEG2RAD)

        self.get_logger().info(
            f'[SCOOP] j5 sweep {ramp[0][1]:.1f} -> {ramp[-1][1]:.1f} deg'
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        for i, (t_sec, j5_deg) in enumerate(ramp):
            pt = JointTrajectoryPoint()
            positions = list(q_now)
            positions[J5] = j5_deg * DEG2RAD
            pt.positions = positions

            velocities = [0.0] * NJ
            if i + 1 < len(ramp):
                t_next, j5_next = ramp[i + 1]
                dt = t_next - t_sec
                if dt > 1e-6:
                    velocities[J5] = (j5_next - j5_deg) * DEG2RAD / dt
            pt.velocities = velocities

            sec = int(t_sec)
            pt.time_from_start = Duration(
                sec=sec, nanosec=int((t_sec - sec) * 1e9)
            )
            goal.trajectory.points.append(pt)

        return self._send_and_wait(goal, 'SCOOP')

    # ------------------------------------------------------------------
    # Cartesian + orientation-locked move via compute_cartesian_path
    # ------------------------------------------------------------------
    def move_cartesian_orientation_locked(self,
                                          target_joint_rad: List[float],
                                          duration_s: float,
                                          label: str) -> bool:
        """
        Plan a Cartesian path from the CURRENT eef pose to the eef pose
        implied by target_joint_rad, with the eef ORIENTATION held fixed
        at the current orientation. If planning succeeds with >= MIN_FRAC
        coverage, dispatch the joint trajectory directly to the bridge.
        Otherwise return False so caller can fall back.
        """
        if not self._cart_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                '[%s] compute_cartesian_path service unavailable; '
                'falling back to joint-space' % label)
            return False
        if not self._fk_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                '[%s] compute_fk service unavailable; falling back' % label)
            return False
        if not self._wait_for_state():
            self.get_logger().error('[%s] no joint state' % label)
            return False

        # 1. FK on current pose -> get start eef pose (we'll lock orientation here)
        start_pose = self._fk(self._latest_q)
        if start_pose is None:
            self.get_logger().warn(
                '[%s] FK on current state failed; fallback' % label)
            return False

        # 2. FK on target joint config -> get target eef position
        target_pose = self._fk(target_joint_rad)
        if target_pose is None:
            self.get_logger().warn(
                '[%s] FK on target failed; fallback' % label)
            return False

        # 3. Build the waypoint list: just the target position, with the
        #    orientation overridden to the START orientation (lock).
        locked_target = Pose()
        locked_target.position = target_pose.position
        locked_target.orientation = start_pose.orientation

        # 4. Orientation constraint applied along the path
        oc = OrientationConstraint()
        oc.header.frame_id = PLANNING_FRAME
        oc.link_name = EEF_LINK
        oc.orientation = start_pose.orientation
        oc.absolute_x_axis_tolerance = ORIENTATION_TOL_RAD
        oc.absolute_y_axis_tolerance = ORIENTATION_TOL_RAD
        oc.absolute_z_axis_tolerance = ORIENTATION_TOL_RAD
        oc.weight = 1.0

        path_constraints = Constraints()
        path_constraints.orientation_constraints.append(oc)

        req = GetCartesianPath.Request()
        req.header.frame_id = PLANNING_FRAME
        req.start_state.joint_state.name = JOINT_NAMES
        req.start_state.joint_state.position = list(self._latest_q)
        req.group_name = PLANNING_GROUP
        req.link_name = EEF_LINK
        req.waypoints = [locked_target]
        req.max_step = CARTESIAN_EEF_STEP
        req.jump_threshold = CARTESIAN_JUMP_THR
        req.avoid_collisions = False
        req.path_constraints = path_constraints

        future = self._cart_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if not future.done():
            self.get_logger().warn('[%s] cartesian planner timed out' % label)
            return False

        resp = future.result()
        frac = resp.fraction
        n_pts = len(resp.solution.joint_trajectory.points)
        self.get_logger().info(
            '[%s] cartesian plan: fraction=%.2f, %d points'
            % (label, frac, n_pts))

        if frac < CARTESIAN_MIN_FRAC or n_pts < 2:
            self.get_logger().warn(
                '[%s] cartesian plan insufficient (frac=%.2f); fallback'
                % (label, frac))
            return False

        # 5. Retime: rescale time_from_start so total duration = duration_s
        traj = resp.solution.joint_trajectory
        last = traj.points[-1].time_from_start
        last_t = last.sec + last.nanosec * 1e-9
        if last_t < 1e-3:
            self.get_logger().warn(
                '[%s] degenerate trajectory time; fallback' % label)
            return False
        scale = duration_s / last_t
        for p in traj.points:
            t = (p.time_from_start.sec + p.time_from_start.nanosec * 1e-9) * scale
            p.time_from_start.sec = int(t)
            p.time_from_start.nanosec = int((t - int(t)) * 1e9)
            # Clear velocities; bridge will recompute via finite diff
            p.velocities = []
            p.accelerations = []

        # Reorder joint_names to canonical and rebuild positions
        if list(traj.joint_names) != JOINT_NAMES:
            try:
                idx_map = [traj.joint_names.index(j) for j in JOINT_NAMES]
            except ValueError as e:
                self.get_logger().warn(
                    '[%s] joint name mismatch in plan: %s' % (label, e))
                return False
            for p in traj.points:
                p.positions = [p.positions[k] for k in idx_map]
            traj.joint_names = list(JOINT_NAMES)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        return self._send_and_wait(goal, label + ' (cartesian)')

    # ------------------------------------------------------------------
    def _fk(self, joint_positions: List[float]) -> Optional[Pose]:
        req = GetPositionFK.Request()
        req.header.frame_id = PLANNING_FRAME
        req.fk_link_names = [EEF_LINK]
        rs = RobotState()
        rs.joint_state.name = JOINT_NAMES
        rs.joint_state.position = list(joint_positions)
        req.robot_state = rs
        fut = self._fk_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        if not fut.done():
            return None
        resp = fut.result()
        if not resp.pose_stamped:
            return None
        return resp.pose_stamped[0].pose

    # ------------------------------------------------------------------
    def _send_and_wait(self, goal, label: str) -> bool:
        send_fut = self._traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        gh = send_fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('[%s] goal rejected' % label)
            return False
        result_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut)
        result = result_fut.result().result
        ok = (result.error_code == FollowJointTrajectory.Result.SUCCESSFUL)
        if ok:
            self.get_logger().info('[%s] complete' % label)
        else:
            self.get_logger().error(
                '[%s] failed, error_code=%d' % (label, result.error_code))
        return ok

    # ------------------------------------------------------------------
    # Cartesian-with-fallback wrapper
    # ------------------------------------------------------------------
    def move_with_orientation_lock(self, target_rad: List[float],
                                   duration_s: float, label: str) -> bool:
        ok = self.move_cartesian_orientation_locked(
            target_rad, duration_s, label)
        if ok:
            return True
        self.get_logger().warn(
            '[%s] falling back to joint-space (spoon may tip)' % label)
        return self.move_joint_space(target_rad, duration_s, label)

    # ------------------------------------------------------------------
    # Top-level cycle
    # ------------------------------------------------------------------
    def run_cycles(self, n: int) -> int:
        successes = 0
        for cycle in range(1, n + 1):
            self.get_logger().info('====== CYCLE %d / %d ======' % (cycle, n))

            if not self.move_joint_space(
                    SCOOP_START_RAD, DUR_TO_SCOOP_START, 'TO_SCOOP_START'):
                break
            if not self.execute_scoop():
                break
            if not self.move_joint_space(
                    TRANSIT_RAD, DUR_TO_TRANSIT, 'TO_TRANSIT'):
                break
            if not self.move_with_orientation_lock(
                    FEEDING_RAD, DUR_TO_FEEDING, 'TO_FEEDING'):
                break

            self.get_logger().info(
                '[DWELL] holding feeding pose for %.1fs' % DWELL_AT_FEEDING)
            t0 = time.monotonic()
            while (time.monotonic() - t0) < DWELL_AT_FEEDING and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)

            if not self.move_with_orientation_lock(
                    TRANSIT_RAD, DUR_TO_TRANSIT, 'TO_TRANSIT_RETURN'):
                break

            successes += 1
            self.get_logger().info('====== CYCLE %d done ======' % cycle)

        return successes


# ---------------------------------------------------------------------------
def main():
    rclpy.init()
    node = FeedingOrchestrator()
    try:
        n_ok = node.run_cycles(NUM_CYCLES)
        node.get_logger().info(
            'Completed %d/%d cycles' % (n_ok, NUM_CYCLES))
        sys.exit(0 if n_ok == NUM_CYCLES else 1)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
        sys.exit(130)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()