#!/usr/bin/env python3
"""
keyboard_teleop.py — Pure Position Keyboard Control for Kinova j2s6s200

Controls individual joints using step-based absolute position commands in radians.

Key Bindings:
  J1: '1' (+), 'q' (-)
  J2: '2' (+), 'w' (-)
  J3: '3' (+), 'e' (-)
  J4: '4' (+), 'r' (-)
  J5: '5' (+), 't' (-)
  J6: '6' (+), 'y' (-)

  CTRL+C to quit.
"""

import sys
import select
import termios
import time
import tty
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# Configuration
STEP_SIZE_RAD = 0.05  # ~3 degrees per key press
MOVE_DUR_S    = 0.5   # Time allowed to reach the step

JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]
NJ = len(JOINT_NAMES)

ACTION_TOPIC      = '/arm_controller/follow_joint_trajectory'
JOINT_STATE_TOPIC = '/j2s6s200_driver/out/joint_state'

KEY_MAP = {
    '1': (0,  1.0), 'q': (0, -1.0),
    '2': (1,  1.0), 'w': (1, -1.0),
    '3': (2,  1.0), 'e': (2, -1.0),
    '4': (3,  1.0), 'r': (3, -1.0),
    '5': (4,  1.0), 't': (4, -1.0),
    '6': (5,  1.0), 'y': (5, -1.0),
}


class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')

        self._latest_q: Optional[List[float]] = None

        self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self._js_cb,
            10,
        )

        self._traj_client = ActionClient(self, FollowJointTrajectory, ACTION_TOPIC)

        self.get_logger().info('Waiting for trajectory controller...')
        if not self._traj_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Trajectory controller not found!')
            sys.exit(1)

        self.get_logger().info('Ready. Press keys to move joints. CTRL-C to exit.')

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

    def dispatch_step(self, joint_idx: int, direction: float) -> bool:
        if not self._wait_for_state():
            self.get_logger().error('No joint state available')
            return False

        target_pos = list(self._latest_q)
        target_pos[joint_idx] += STEP_SIZE_RAD * direction

        joint_name = JOINT_NAMES[joint_idx]
        print(f"\rMoving {joint_name} to {target_pos[joint_idx]:.3f} rad", end="", flush=True)

        pt = JointTrajectoryPoint()
        pt.positions = target_pos
        pt.velocities = [0.0] * NJ
        sec = int(MOVE_DUR_S)
        pt.time_from_start = Duration(
            sec=sec,
            nanosec=int((MOVE_DUR_S - sec) * 1e9),
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        goal.trajectory.points = [pt]

        return self._send_and_wait(goal, joint_name)

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
        if not ok:
            self.get_logger().error(f'[{label}] failed, error_code={result.error_code}')
        return ok


# ── Terminal Input Handling ────────────────────────────────────────────────

def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main():
    rclpy.init()
    node = KeyboardTeleopNode()

    settings = termios.tcgetattr(sys.stdin)

    print("\n--- Kinova Jaco2 Joint Teleop ---")
    print("1/q: Joint 1 (+/-)")
    print("2/w: Joint 2 (+/-)")
    print("3/e: Joint 3 (+/-)")
    print("4/r: Joint 4 (+/-)")
    print("5/t: Joint 5 (+/-)")
    print("6/y: Joint 6 (+/-)")
    print("---------------------------------\n")

    try:
        while rclpy.ok():
            key = get_key(settings)

            if key == '\x03':  # CTRL+C
                break

            if key in KEY_MAP:
                joint_idx, direction = KEY_MAP[key]
                node.dispatch_step(joint_idx, direction)
            else:
                # Keep joint state fresh while idle
                rclpy.spin_once(node, timeout_sec=0.0)

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
