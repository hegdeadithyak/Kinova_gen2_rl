#!/usr/bin/env python3
"""
kinova_trajectory_bridge.py

FollowJointTrajectory action server -> Kinova j2s6s200 joint velocity interface.

Control architecture:
  - 100 Hz unconditional velocity heartbeat (DSP watchdog stays fed).
  - 50 Hz inner control loop interpolates between waypoints in real time.
    Single-waypoint goals produce smooth motion via a synthetic t=0 waypoint
    pulled from the measured pose.
  - Two-phase controller:
      TRACK  (t < t_end):  v = v_ff + KP*err          (no integral; tracking
                                                       lag during a ramp is
                                                       expected and would
                                                       wind up the integrator)
      SETTLE (t >= t_end): v =        KP*err + KI*∫err (integral rejects
                                                        gravity load on a
                                                        static target)
  - Anti-windup clamp on the integrator.
  - Runaway watchdog: if commanded |v| stays above RUNAWAY_VEL for
    RUNAWAY_TIME while error is growing, abort and halt.
  - Path tolerance check during execution; goal tolerance check at end.
  - SUCCESS reported only if measured joint error is within tolerance.
"""

import math
import threading
import time
from typing import List, Optional

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from kinova_msgs.msg import JointVelocity

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
PUBLISH_HZ          = 100.0
PUBLISH_DT          = 1.0 / PUBLISH_HZ

CTRL_HZ             = 50.0
CTRL_DT             = 1.0 / CTRL_HZ

JOINT_STATE_TIMEOUT = 0.5                  # s; abort if state goes stale

# Control gains
KP                  = 2.0                  # rad/s per rad of position error
KI                  = 0.4                  # rad/s per (rad·s) of accumulated err
INTEGRAL_LIMIT      = math.radians(8.0)    # anti-windup clamp on accumulator

# Limits
MAX_JOINT_VEL       = math.radians(60.0)   # rad/s safety clamp
GOAL_TOLERANCE      = math.radians(2.0)    # final position tolerance per joint
PATH_TOLERANCE      = math.radians(20.0)   # in-flight tracking error per joint
SETTLE_TIME         = 0.5                  # s convergence time after t_end

# Runaway watchdog
RUNAWAY_VEL         = math.radians(50.0)   # rad/s; sustained commands above this
RUNAWAY_TIME        = 0.5                  # s; for this long while err grows

JOINT_STATE_TOPIC = '/j2s6s200_driver/out/joint_state'
VEL_CMD_TOPIC     = '/j2s6s200_driver/in/joint_velocity'

JOINT_NAMES = [
    'j2s6s200_joint_1',
    'j2s6s200_joint_2',
    'j2s6s200_joint_3',
    'j2s6s200_joint_4',
    'j2s6s200_joint_5',
    'j2s6s200_joint_6',
]
NJ = len(JOINT_NAMES)


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ---------------------------------------------------------------------------
class KinovaTrajectoryBridge(Node):
    def __init__(self):
        super().__init__('kinova_trajectory_bridge')

        self._cb_group = ReentrantCallbackGroup()

        # Shared command (deg/s; Kinova API unit)
        self._cmd_lock = threading.Lock()
        self._current_cmd_deg = [0.0] * NJ

        # Joint state cache (rad)
        self._js_lock = threading.Lock()
        self._latest_q: Optional[List[float]] = None
        self._latest_q_stamp: float = 0.0

        # Preemption: one execution at a time.
        # When a new goal arrives, _stop_event is set so the running
        # _execute_cb exits within one CTRL_DT (~20 ms), then _exec_lock
        # is acquired exclusively by the new goal's thread.
        self._exec_lock  = threading.Lock()
        self._stop_event = threading.Event()

        # I/O
        self._vel_pub = self.create_publisher(JointVelocity, VEL_CMD_TOPIC, 10)

        self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._joint_state_cb, 10,
            callback_group=self._cb_group,
        )

        # Unconditional 100 Hz heartbeat
        self.create_timer(PUBLISH_DT, self._heartbeat_cb,
                          callback_group=self._cb_group)

        self._action_server = ActionServer(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f'Kinova trajectory bridge ready. '
            f'Heartbeat={PUBLISH_HZ:.0f}Hz CtrlLoop={CTRL_HZ:.0f}Hz '
            f'KP={KP} KI={KI} (settle-only)'
        )

    # ------------------------------------------------------------------
    # Heartbeat — always runs
    # ------------------------------------------------------------------
    def _heartbeat_cb(self):
        with self._cmd_lock:
            v = list(self._current_cmd_deg)
        msg = JointVelocity()
        msg.joint1, msg.joint2, msg.joint3 = v[0], v[1], v[2]
        msg.joint4, msg.joint5, msg.joint6 = v[3], v[4], v[5]
        msg.joint7 = 0.0
        self._vel_pub.publish(msg)

    # ------------------------------------------------------------------
    # Joint state cache
    # ------------------------------------------------------------------
    def _joint_state_cb(self, msg: JointState):
        try:
            idx = [msg.name.index(j) for j in JOINT_NAMES]
        except ValueError:
            return
        q = [msg.position[i] for i in idx]
        with self._js_lock:
            self._latest_q = q
            self._latest_q_stamp = time.monotonic()

    def _get_q(self) -> Optional[List[float]]:
        with self._js_lock:
            if self._latest_q is None:
                return None
            if (time.monotonic() - self._latest_q_stamp) > JOINT_STATE_TIMEOUT:
                return None
            return list(self._latest_q)

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------
    def _set_cmd_rad(self, v_rad: List[float]):
        clamped = [clamp(v, -MAX_JOINT_VEL, MAX_JOINT_VEL) for v in v_rad]
        with self._cmd_lock:
            self._current_cmd_deg = [math.degrees(v) for v in clamped]

    def _zero_cmd(self):
        with self._cmd_lock:
            self._current_cmd_deg = [0.0] * NJ

    # ------------------------------------------------------------------
    # Goal / cancel callbacks
    # ------------------------------------------------------------------
    def _goal_cb(self, goal_request):
        # Accept every incoming goal; the execute wrapper below will
        # preempt whatever is currently running.
        self._stop_event.set()
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        return CancelResponse.ACCEPT

    # ------------------------------------------------------------------
    # Action callback — thin wrapper that enforces exclusive execution
    # ------------------------------------------------------------------
    def _execute_cb(self, goal_handle: ServerGoalHandle):
        # _goal_cb already set _stop_event; wait for the previous
        # execution to exit and release _exec_lock (~one CTRL_DT delay).
        with self._exec_lock:
            self._stop_event.clear()      # we own the lock; reset for our run
            return self._run_trajectory(goal_handle)

    def _run_trajectory(self, goal_handle: ServerGoalHandle):
        result = FollowJointTrajectory.Result()
        traj = goal_handle.request.trajectory
        points = traj.points
        joint_order = traj.joint_names

        if not points:
            self.get_logger().warn('Empty trajectory; nothing to do.')
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            return result

        try:
            idx_map = [joint_order.index(j) for j in JOINT_NAMES]
        except ValueError as e:
            self.get_logger().error(f'Joint name mismatch: {e}')
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            return result

        q_start = self._get_q()
        if q_start is None:
            self.get_logger().error('No fresh joint state; aborting.')
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            return result

        # ---- Pre-extract waypoints in canonical order ------------------
        wp_t: List[float]       = []
        wp_q: List[List[float]] = []
        for p in points:
            t = p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
            q = [p.positions[idx_map[j]] for j in range(NJ)]
            wp_t.append(t)
            wp_q.append(q)

        # ---- Normalize trajectory to driver's angle frame --------------
        # MoveIt may plan in a different 2π cycle than the driver reports
        # (e.g. planner uses [-π, π] while driver reports [0, 2π]).
        # Compute per-joint integer-2π offsets so the trajectory start
        # aligns with the measured position, then apply to all waypoints.
        if wp_q:
            for j in range(NJ):
                diff = q_start[j] - wp_q[0][j]
                offset = round(diff / (2.0 * math.pi)) * (2.0 * math.pi)
                if abs(offset) > 1e-6:
                    self.get_logger().info(
                        f'Joint {j+1}: applying 2π offset of '
                        f'{math.degrees(offset):.1f} deg to trajectory'
                    )
                    for wp in wp_q:
                        wp[j] += offset

        # Synthetic t=0 from measured pose -> smooth single-waypoint goals
        if wp_t[0] > 1e-3:
            wp_t.insert(0, 0.0)
            wp_q.insert(0, list(q_start))
        else:
            wp_q[0] = list(q_start)

        t_end = wp_t[-1]
        q_goal = wp_q[-1]

        self.get_logger().info(
            f'Executing trajectory: {len(wp_t)} waypoints, t_end={t_end:.2f}s'
        )

        # ---- 50 Hz control loop ----------------------------------------
        err_integral    = [0.0] * NJ
        prev_max_err    = 0.0
        runaway_t_start = None

        start_time = time.monotonic()
        log_counter = 0

        try:
            while True:
                # Preempted by a newer goal arriving
                if self._stop_event.is_set():
                    self.get_logger().info('Preempted by new goal — aborting.')
                    self._zero_cmd()
                    goal_handle.abort()
                    result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    return result

                # Client-side cancel request
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Cancel requested — stopping.')
                    self._zero_cmd()
                    goal_handle.canceled()
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    return result

                t_now = time.monotonic() - start_time
                if t_now >= t_end + SETTLE_TIME:
                    break

                in_settle = (t_now >= t_end)

                # ---- Target & feedforward ------------------------------
                if in_settle:
                    q_target = q_goal
                    v_ff     = [0.0] * NJ
                else:
                    k = 0
                    while k + 1 < len(wp_t) and wp_t[k + 1] < t_now:
                        k += 1
                    if k + 1 >= len(wp_t):
                        q_target = q_goal
                        v_ff     = [0.0] * NJ
                    else:
                        t0, t1 = wp_t[k], wp_t[k + 1]
                        dt_seg = max(t1 - t0, 1e-6)
                        alpha = (t_now - t0) / dt_seg
                        alpha = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)
                        q_target = [
                            wp_q[k][j] * (1 - alpha) + wp_q[k + 1][j] * alpha
                            for j in range(NJ)
                        ]
                        v_ff = [
                            (wp_q[k + 1][j] - wp_q[k][j]) / dt_seg
                            for j in range(NJ)
                        ]

                # ---- Measurement ---------------------------------------
                q_meas = self._get_q()
                if q_meas is None:
                    self.get_logger().error('Joint state went stale; aborting.')
                    self._zero_cmd()
                    goal_handle.abort()
                    result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    return result

                err = [q_target[j] - q_meas[j] for j in range(NJ)]
                # Shortest-path normalization — handles any residual wrap
                for j in range(NJ):
                    while err[j] >  math.pi:
                        err[j] -= 2.0 * math.pi
                    while err[j] < -math.pi:
                        err[j] += 2.0 * math.pi
                max_err = max(abs(e) for e in err)

                if max_err > PATH_TOLERANCE:
                    self.get_logger().error(
                        f'PATH TOLERANCE VIOLATED at t={t_now:.2f}s, '
                        f'max_err={math.degrees(max_err):.2f} deg. '
                        f'err_deg={[round(math.degrees(e), 2) for e in err]}'
                    )
                    self._zero_cmd()
                    goal_handle.abort()
                    result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    return result

                # ---- Integrator: settle phase ONLY ---------------------
                if in_settle:
                    for j in range(NJ):
                        err_integral[j] += err[j] * CTRL_DT
                        if err_integral[j] >  INTEGRAL_LIMIT:
                            err_integral[j] =  INTEGRAL_LIMIT
                        elif err_integral[j] < -INTEGRAL_LIMIT:
                            err_integral[j] = -INTEGRAL_LIMIT
                    i_term = [KI * err_integral[j] for j in range(NJ)]
                else:
                    err_integral = [0.0] * NJ
                    i_term = [0.0] * NJ

                # ---- Control law ---------------------------------------
                v_cmd = [v_ff[j] + KP * err[j] + i_term[j] for j in range(NJ)]
                self._set_cmd_rad(v_cmd)

                # ---- Runaway watchdog ----------------------------------
                cmd_max = max(abs(v) for v in v_cmd)
                err_growing = (max_err > prev_max_err + math.radians(0.05))
                if cmd_max > RUNAWAY_VEL and err_growing:
                    if runaway_t_start is None:
                        runaway_t_start = t_now
                    elif (t_now - runaway_t_start) > RUNAWAY_TIME:
                        self.get_logger().error(
                            f'RUNAWAY DETECTED at t={t_now:.2f}s. '
                            f'cmd_max={math.degrees(cmd_max):.1f} deg/s, '
                            f'err_deg={[round(math.degrees(e), 2) for e in err]}. '
                            f'Halting.'
                        )
                        self._zero_cmd()
                        goal_handle.abort()
                        result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                        return result
                else:
                    runaway_t_start = None
                prev_max_err = max_err

                # ---- Logging -------------------------------------------
                log_counter += 1
                if log_counter % 25 == 0:
                    phase = 'SETTLE' if in_settle else 'TRACK'
                    self.get_logger().info(
                        f'[{phase}] t={t_now:.2f}s '
                        f'err_deg={[round(math.degrees(e), 2) for e in err]} '
                        f'i_deg={[round(math.degrees(i), 2) for i in err_integral]}'
                    )

                time.sleep(CTRL_DT)

            # ---- Final goal tolerance check -----------------------------
            self._zero_cmd()
            time.sleep(0.05)

            q_meas = self._get_q()
            if q_meas is None:
                self.get_logger().error('No joint state at goal check; aborting.')
                goal_handle.abort()
                result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                return result

            final_err = [q_goal[j] - q_meas[j] for j in range(NJ)]
            max_final_err = max(abs(e) for e in final_err)

            self.get_logger().info(
                f'Final tracking error (deg): '
                f'{[round(math.degrees(e), 2) for e in final_err]}'
            )

            if max_final_err > GOAL_TOLERANCE:
                self.get_logger().error(
                    f'GOAL TOLERANCE VIOLATED: '
                    f'max_err={math.degrees(max_final_err):.2f} deg > '
                    f'{math.degrees(GOAL_TOLERANCE):.2f} deg'
                )
                goal_handle.abort()
                result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                return result

            self.get_logger().info('Trajectory complete and verified.')
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            return result

        except Exception as e:
            self.get_logger().error(f'Execution exception: {e}')
            self._zero_cmd()
            try:
                goal_handle.abort()
            except Exception:
                pass
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            return result


# ---------------------------------------------------------------------------
def main():
    rclpy.init()
    node = KinovaTrajectoryBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._zero_cmd()
        time.sleep(0.05)
        rclpy.shutdown()


if __name__ == '__main__':
    main()