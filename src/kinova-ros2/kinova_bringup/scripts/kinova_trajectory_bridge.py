#!/usr/bin/env python3
import math
import threading
import time
from typing import List, Optional

import rclpy
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from kinova_msgs.msg import JointVelocity

PUBLISH_HZ          = 100.0
PUBLISH_DT          = 1.0 / PUBLISH_HZ
CTRL_HZ             = 50.0
CTRL_DT             = 1.0 / CTRL_HZ
JOINT_STATE_TIMEOUT = 0.5

KP             = 2.0
KI             = 0.4
INTEGRAL_LIMIT = math.radians(8.0)

MAX_JOINT_VEL  = math.radians(180.0)
GOAL_TOLERANCE = math.radians(5.0)
PATH_TOLERANCE = math.radians(6000.0)
SETTLE_TIME    = 1.0

JOINT_STATE_TOPIC = '/j2s6s200_driver/out/joint_state'
VEL_CMD_TOPIC     = '/j2s6s200_driver/in/joint_velocity'

JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]
NJ = len(JOINT_NAMES)


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class KinovaTrajectoryBridge(Node):
    def __init__(self):
        super().__init__('kinova_trajectory_bridge')

        self._cb_group = ReentrantCallbackGroup()

        self._cmd_lock = threading.Lock()
        self._current_cmd_deg = [0.0] * NJ

        self._js_lock = threading.Lock()
        self._latest_q: Optional[List[float]] = None
        self._latest_q_stamp: float = 0.0

        self._vel_pub = self.create_publisher(JointVelocity, VEL_CMD_TOPIC, 10)

        self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._joint_state_cb, 10,
            callback_group=self._cb_group)

        self.create_timer(PUBLISH_DT, self._heartbeat_cb, callback_group=self._cb_group)

        self._action_server = ActionServer(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            execute_callback=self._execute_cb,
            callback_group=self._cb_group)

        self.get_logger().info(
            f'Kinova trajectory bridge ready. '
            f'Heartbeat={PUBLISH_HZ:.0f}Hz CtrlLoop={CTRL_HZ:.0f}Hz KP={KP} KI={KI}')

    def _heartbeat_cb(self):
        with self._cmd_lock:
            v = list(self._current_cmd_deg)
        msg = JointVelocity()
        msg.joint1, msg.joint2, msg.joint3 = v[0], v[1], v[2]
        msg.joint4, msg.joint5, msg.joint6 = v[3], v[4], v[5]
        msg.joint7 = 0.0
        self._vel_pub.publish(msg)

    def _joint_state_cb(self, msg: JointState):
        try:
            idx = [msg.name.index(j) for j in JOINT_NAMES]
        except ValueError:
            return
        with self._js_lock:
            self._latest_q = [msg.position[i] for i in idx]
            self._latest_q_stamp = time.monotonic()

    def _get_q(self) -> Optional[List[float]]:
        with self._js_lock:
            if self._latest_q is None:
                return None
            if (time.monotonic() - self._latest_q_stamp) > JOINT_STATE_TIMEOUT:
                return None
            return list(self._latest_q)

    def _set_cmd_rad(self, v_rad: List[float]):
        clamped = [clamp(v, -MAX_JOINT_VEL, MAX_JOINT_VEL) for v in v_rad]
        with self._cmd_lock:
            self._current_cmd_deg = [math.degrees(v) for v in clamped]

    def _zero_cmd(self):
        with self._cmd_lock:
            self._current_cmd_deg = [0.0] * NJ

    def _execute_cb(self, goal_handle: ServerGoalHandle):
        result = FollowJointTrajectory.Result()
        traj   = goal_handle.request.trajectory
        points = traj.points

        if not points:
            self.get_logger().warn('Empty trajectory; nothing to do.')
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            return result

        try:
            idx_map = [traj.joint_names.index(j) for j in JOINT_NAMES]
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

        wp_t: List[float]       = []
        wp_q: List[List[float]] = []
        for p in points:
            wp_t.append(p.time_from_start.sec + p.time_from_start.nanosec * 1e-9)
            wp_q.append([p.positions[idx_map[j]] for j in range(NJ)])

        # Align trajectory to driver's angle frame — MoveIt may use a different 2π cycle.
        for j in range(NJ):
            offset = round((q_start[j] - wp_q[0][j]) / (2.0 * math.pi)) * (2.0 * math.pi)
            if abs(offset) > 1e-6:
                self.get_logger().info(
                    f'Joint {j+1}: 2π offset {math.degrees(offset):.1f} deg')
                for wp in wp_q:
                    wp[j] += offset

        # Synthetic t=0 from measured pose — gives single-waypoint goals a smooth start.
        if wp_t[0] > 1e-3:
            wp_t.insert(0, 0.0)
            wp_q.insert(0, list(q_start))
        else:
            wp_q[0] = list(q_start)

        t_end  = wp_t[-1]
        q_goal = wp_q[-1]

        self.get_logger().info(
            f'Executing trajectory: {len(wp_t)} waypoints, t_end={t_end:.2f}s')

        err_integral = [0.0] * NJ
        start_time   = time.monotonic()
        log_counter  = 0

        try:
            while True:
                t_now     = time.monotonic() - start_time
                in_settle = t_now >= t_end

                if t_now >= t_end + SETTLE_TIME:
                    break

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
                        t0, t1   = wp_t[k], wp_t[k + 1]
                        dt_seg   = max(t1 - t0, 1e-6)
                        alpha    = max(0.0, min(1.0, (t_now - t0) / dt_seg))
                        q_target = [wp_q[k][j] * (1 - alpha) + wp_q[k+1][j] * alpha for j in range(NJ)]
                        v_ff     = [(wp_q[k+1][j] - wp_q[k][j]) / dt_seg for j in range(NJ)]

                q_meas = self._get_q()
                if q_meas is None:
                    self.get_logger().error('Joint state went stale; aborting.')
                    self._zero_cmd()
                    goal_handle.abort()
                    result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    return result

                err = [q_target[j] - q_meas[j] for j in range(NJ)]
                # Shortest-path wrap — handles residual angle discontinuities.
                for j in range(NJ):
                    while err[j] >  math.pi: err[j] -= 2.0 * math.pi
                    while err[j] < -math.pi: err[j] += 2.0 * math.pi
                max_err = max(abs(e) for e in err)

                if max_err > PATH_TOLERANCE:
                    self.get_logger().error(
                        f'PATH TOLERANCE VIOLATED at t={t_now:.2f}s, '
                        f'max_err={math.degrees(max_err):.2f} deg')
                    self._zero_cmd()
                    goal_handle.abort()
                    result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    return result

                if in_settle and max_err < GOAL_TOLERANCE:
                    self.get_logger().info(
                        f'Converged at t={t_now:.2f}s, max_err={math.degrees(max_err):.2f} deg')
                    break

                if in_settle:
                    for j in range(NJ):
                        err_integral[j] = max(-INTEGRAL_LIMIT,
                                              min(INTEGRAL_LIMIT,
                                                  err_integral[j] + err[j] * CTRL_DT))
                    i_term = [KI * err_integral[j] for j in range(NJ)]
                else:
                    err_integral = [0.0] * NJ
                    i_term       = [0.0] * NJ

                self._set_cmd_rad([v_ff[j] + KP * err[j] + i_term[j] for j in range(NJ)])

                log_counter += 1
                if log_counter % 25 == 0:
                    phase = 'SETTLE' if in_settle else 'TRACK'
                    self.get_logger().info(
                        f'[{phase}] t={t_now:.2f}s '
                        f'err={[round(math.degrees(e), 2) for e in err]}')

                time.sleep(CTRL_DT)

            self._zero_cmd()
            self.get_logger().info('Trajectory complete.')
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
