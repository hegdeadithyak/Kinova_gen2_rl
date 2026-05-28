#!/usr/bin/env python3
import json
import math
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
from geometry_msgs.msg import Pose


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    'j2s6s200_joint_1',
    'j2s6s200_joint_2',
    'j2s6s200_joint_3',
    'j2s6s200_joint_4',
    'j2s6s200_joint_5',
    'j2s6s200_joint_6',
]

NJ = len(JOINT_NAMES)

J5 = 1

SCOOP_J5_RAMP_DEG = [
    (0.05, 232.97),
    (0.15, 232.97),
    (0.25, 232.97),
    (0.35, 232.97),
    (0.45, 232.97),
    (0.55, 232.97),
    (0.65, 232.97),
    (0.75, 232.97),
    (0.85, 232.97),
    (0.95, 232.97),
    (1.05, 232.97),
    (1.15, 232.97),
    (1.25, 232.97),
    (1.35, 232.97),
    (1.45, 232.97),
    (1.55, 232.97),
    (1.65, 232.97),
    (1.75, 232.97),
    (1.85, 232.97),
    (1.95, 233.01),
    (2.05, 233.01),
    (2.15, 233.75),
    (2.25, 234.14),
    (2.35, 234.55),
    (2.45, 234.55),
    (2.55, 234.93),
    (2.65, 235.22),
    (2.75, 235.53),
    (2.85, 235.90),
    (2.95, 236.17),
    (3.05, 236.50),
    (3.15, 236.86),
    (3.25, 237.51),
    (3.35, 237.51),
    (3.45, 237.88),
    (3.55, 238.25),
    (3.65, 238.60),
    (3.75, 239.43),
    (3.85, 239.43),
    (3.95, 239.89),
]

DEG2RAD = math.pi / 180.0

DUR_TO_SCOOP_START = 5.0
DUR_RETURN_TO_SCOOP_START = 2.5
DUR_TO_TRANSIT = 2.5
DUR_TO_FEEDING = 5.0
DWELL_AT_FEEDING = 3.0
NUM_CYCLES = 3

CARTESIAN_EEF_STEP = 0.01
CARTESIAN_JUMP_THR = 0.0
CARTESIAN_MIN_FRAC = 0.90
ORIENTATION_TOL_RAD = 0.35

EEF_LINK = 'j2s6s200_link_6'
PLANNING_FRAME = 'root'
PLANNING_GROUP = 'arm'

ACTION_TOPIC = '/arm_controller/follow_joint_trajectory'
JOINT_STATE_TOPIC = '/j2s6s200_driver/out/joint_state'


# ---------------------------------------------------------------------------
# Robot Node
# ---------------------------------------------------------------------------

class FeedingOrchestrator(Node):
    def __init__(self):
        super().__init__('feeding_orchestrator')

        self._traj_client = ActionClient(self, FollowJointTrajectory, ACTION_TOPIC)
        self._latest_q: Optional[List[float]] = None

        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self._cart_cli = self.create_client(GetCartesianPath, 'compute_cartesian_path')
        self._fk_cli = self.create_client(GetPositionFK, 'compute_fk')

    def _js_cb(self, msg: JointState):
        try:
            idx = [msg.name.index(j) for j in JOINT_NAMES]
        except ValueError:
            return
        self._latest_q = [msg.position[i] for i in idx]

    def _wait_for_state(self, timeout: float = 3.0) -> bool:
        t0 = time.monotonic()
        while self._latest_q is None and (time.monotonic() - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self._latest_q is not None

    def move_joint_space(self, target_rad: List[float], duration_s: float, label: str) -> bool:
        self.get_logger().info(f'[{label}] joint-space move over {duration_s:.1f}s')

        if not self._traj_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Trajectory action server not available')
            return False

        if self._wait_for_state():
            TWO_PI = 2.0 * math.pi
            positions = [
                t - round((t - c) / TWO_PI) * TWO_PI
                for c, t in zip(self._latest_q, target_rad)
            ]
        else:
            self.get_logger().warn(f'[{label}] no joint state; using target angles as-is')
            positions = list(target_rad)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.velocities = [0.0] * NJ

        sec = int(duration_s)
        pt.time_from_start = Duration(sec=sec, nanosec=int((duration_s - sec) * 1e9))
        goal.trajectory.points.append(pt)

        return self._send_and_wait(goal, label)

    def execute_scoop(self) -> bool:
        if not self._wait_for_state():
            self.get_logger().error('No joint state for scoop build')
            return False

        q_now = list(self._latest_q)
        ramp = list(SCOOP_J5_RAMP_DEG)
        ramp[0] = (ramp[0][0], q_now[J5] / DEG2RAD)

        self.get_logger().info(f'[SCOOP] j5 sweep {ramp[0][1]:.1f} -> {ramp[-1][1]:.1f} deg')

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
                    velocities[J5] = min((j5_next - j5_deg) * DEG2RAD / dt, 55.0)
            pt.velocities = velocities

            sec = int(t_sec)
            pt.time_from_start = Duration(sec=sec, nanosec=int((t_sec - sec) * 1e9))
            goal.trajectory.points.append(pt)

        return self._send_and_wait(goal, 'SCOOP')

    def move_cartesian_orientation_locked(
        self, target_joint_rad: List[float], duration_s: float, label: str
    ) -> bool:
        if not self._cart_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(f'[{label}] compute_cartesian_path unavailable; fallback')
            return False

        if not self._fk_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(f'[{label}] compute_fk unavailable; fallback')
            return False

        if not self._wait_for_state():
            self.get_logger().error(f'[{label}] no joint state')
            return False

        start_pose = self._fk(self._latest_q)
        if start_pose is None:
            self.get_logger().warn(f'[{label}] FK on current state failed; fallback')
            return False

        target_pose = self._fk(target_joint_rad)
        if target_pose is None:
            self.get_logger().warn(f'[{label}] FK on target failed; fallback')
            return False

        locked_target = Pose()
        locked_target.position = target_pose.position
        locked_target.orientation = start_pose.orientation

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
            self.get_logger().warn(f'[{label}] cartesian planner timed out')
            return False

        resp = future.result()
        frac = resp.fraction
        n_pts = len(resp.solution.joint_trajectory.points)
        self.get_logger().info(f'[{label}] cartesian plan: fraction={frac:.2f}, {n_pts} points')

        if frac < CARTESIAN_MIN_FRAC or n_pts < 2:
            self.get_logger().warn(f'[{label}] cartesian plan insufficient (frac={frac:.2f}); fallback')
            return False

        traj = resp.solution.joint_trajectory
        last = traj.points[-1].time_from_start
        last_t = last.sec + last.nanosec * 1e-9

        if last_t < 1e-3:
            self.get_logger().warn(f'[{label}] degenerate trajectory time; fallback')
            return False

        scale = duration_s / last_t
        for point in traj.points:
            old_t = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            new_t = old_t * scale
            point.time_from_start.sec = int(new_t)
            point.time_from_start.nanosec = int((new_t - int(new_t)) * 1e9)
            point.velocities = []
            point.accelerations = []

        if list(traj.joint_names) != JOINT_NAMES:
            try:
                idx_map = [traj.joint_names.index(j) for j in JOINT_NAMES]
            except ValueError as e:
                self.get_logger().warn(f'[{label}] joint name mismatch in plan: {e}')
                return False
            for point in traj.points:
                point.positions = [point.positions[k] for k in idx_map]
            traj.joint_names = list(JOINT_NAMES)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        return self._send_and_wait(goal, label + ' (cartesian)')

    def _fk(self, joint_positions: List[float]) -> Optional[Pose]:
        req = GetPositionFK.Request()
        req.header.frame_id = PLANNING_FRAME
        req.fk_link_names = [EEF_LINK]

        robot_state = RobotState()
        robot_state.joint_state.name = JOINT_NAMES
        robot_state.joint_state.position = list(joint_positions)
        req.robot_state = robot_state

        future = self._fk_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if not future.done() or not future.result().pose_stamped:
            return None
        return future.result().pose_stamped[0].pose

    def _send_and_wait(self, goal, label: str) -> bool:
        send_future = self._traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f'[{label}] goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        ok = result.error_code == FollowJointTrajectory.Result.SUCCESSFUL

        if ok:
            self.get_logger().info(f'[{label}] complete')
        else:
            self.get_logger().error(f'[{label}] failed, error_code={result.error_code}')
        return ok

    def execute_recorded(self, trajectory_file: str) -> bool:
        """Replay a trajectory saved by record.py."""
        try:
            with open(trajectory_file) as f:
                waypoints = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.get_logger().error(f'[RECORDED] cannot load {trajectory_file}: {e}')
            return False

        if len(waypoints) < 2:
            self.get_logger().error('[RECORDED] trajectory has fewer than 2 waypoints')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        for i, wp in enumerate(waypoints):
            pt = JointTrajectoryPoint()
            pt.positions = list(wp['positions'])
            t = float(wp['time_s'])

            # finite-difference velocities for smooth inter-waypoint motion
            velocities = [0.0] * NJ
            if i + 1 < len(waypoints):
                nxt = waypoints[i + 1]
                dt = nxt['time_s'] - wp['time_s']
                if dt > 1e-6:
                    for j in range(NJ):
                        velocities[j] = (nxt['positions'][j] - wp['positions'][j]) / dt
            pt.velocities = velocities

            pt.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
            goal.trajectory.points.append(pt)

        dur = waypoints[-1]['time_s']
        self.get_logger().info(
            f'[RECORDED] replaying {len(waypoints)} waypoints over {dur:.1f}s')
        return self._send_and_wait(goal, 'RECORDED')

    def move_with_orientation_lock(self, target_rad: List[float], duration_s: float, label: str) -> bool:
        ok = self.move_cartesian_orientation_locked(target_rad, duration_s, label)
        if ok:
            return True
        self.get_logger().warn(f'[{label}] falling back to joint-space; spoon may tip')
        return self.move_joint_space(target_rad, duration_s, label)

    def run_cycles(self, n: int, scoop_start: List[float], transit: List[float], feeding: List[float]) -> int:
        successes = 0
        for cycle in range(1, n + 1):
            self.get_logger().info(f'====== CYCLE {cycle} / {n} ======')

            if not self.move_joint_space(scoop_start, DUR_TO_SCOOP_START, 'TO_SCOOP_START'):
                break
            if not self.execute_scoop():
                break
            if not self.move_joint_space(scoop_start, DUR_RETURN_TO_SCOOP_START, 'RETURN_TO_SCOOP_START'):
                break
            if not self.move_joint_space(transit, DUR_TO_TRANSIT, 'TO_TRANSIT'):
                break
            if not self.move_with_orientation_lock(feeding, DUR_TO_FEEDING, 'TO_FEEDING'):
                break

            self.get_logger().info(f'[DWELL] holding feeding pose for {DWELL_AT_FEEDING:.1f}s')
            t0 = time.monotonic()
            while (time.monotonic() - t0) < DWELL_AT_FEEDING and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)

            if not self.move_with_orientation_lock(transit, DUR_TO_TRANSIT, 'TO_TRANSIT_RETURN'):
                break

            successes += 1
            self.get_logger().info(f'====== CYCLE {cycle} done ======')

        return successes
