#!/usr/bin/env python3
"""
joy_teleop.py — Smooth Joystick Position Control for Kinova j2s6s200

Fixes stop-and-go motion by using:
- Non-blocking trajectory sending
- Continuous control loop (timer-based)

Axis Mapping:
  Joint 1: axis[3]
  Joint 2: axis[2]
  Joint 3: axis[6]
  Joint 4: axis[7]

Button Mapping:
  Joint 5: button[5] (+), button[1] (-)
  Joint 6: button[4] (+), button[2] (-)
"""

import sys
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState, Joy
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


# ================= CONFIG =================
STEP_SIZE_RAD = 0.05     # Step per update (~3 deg)
MOVE_DUR_S    = 0.3      # Shorter = smoother
AXIS_DEADZONE = 0.5

TIMER_PERIOD  = 0.1      # Control loop frequency (10 Hz)

JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]
NJ = len(JOINT_NAMES)

ACTION_TOPIC      = '/arm_controller/follow_joint_trajectory'
JOINT_STATE_TOPIC = '/j2s6s200_driver/out/joint_state'
JOY_TOPIC         = '/joy'

# Axis → joint mapping
AXIS_MAP = {
    3: 0,
    2: 1,
    6: 2,
    7: 3,
}

# Button → (joint, direction)
BUTTON_MAP = {
    4: (4, +1.0),
    0: (4, -1.0),
    3: (5, +1.0),
    1: (5, -1.0),
}


# ================= NODE =================
class JoyTeleopNode(Node):
    def __init__(self):
        super().__init__('joy_teleop')

        self._latest_q: Optional[List[float]] = None
        self._last_joy_msg: Optional[Joy] = None

        # Subscribers
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self.create_subscription(Joy, JOY_TOPIC, self._joy_cb, 10)

        # Action client
        self._traj_client = ActionClient(self, FollowJointTrajectory, ACTION_TOPIC)

        self.get_logger().info('Waiting for trajectory controller...')
        if not self._traj_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Trajectory controller not found!')
            sys.exit(1)

        # Continuous control loop
        self.create_timer(TIMER_PERIOD, self.control_loop)

        self.get_logger().info('Joystick teleop ready (smooth mode).')

    # ---------- Callbacks ----------
    def _js_cb(self, msg: JointState):
        try:
            idx = [msg.name.index(j) for j in JOINT_NAMES]
        except ValueError:
            return
        self._latest_q = [msg.position[i] for i in idx]

    def _joy_cb(self, msg: Joy):
        self._last_joy_msg = msg

    # ---------- Control Loop ----------
    def control_loop(self):
        if self._latest_q is None or self._last_joy_msg is None:
            return

        msg = self._last_joy_msg
        target_pos = list(self._latest_q)
        moved = False

        # ---- AXIS CONTROL (continuous) ----
        for axis_idx, joint_idx in AXIS_MAP.items():
            if axis_idx >= len(msg.axes):
                continue

            val = msg.axes[axis_idx]

            if abs(val) > AXIS_DEADZONE:
                direction = 1.0 if val > 0 else -1.0
                target_pos[joint_idx] += STEP_SIZE_RAD * direction
                moved = True

        # ---- BUTTON CONTROL ----
        for btn_idx, (joint_idx, direction) in BUTTON_MAP.items():
            if btn_idx < len(msg.buttons) and msg.buttons[btn_idx] == 1:
                target_pos[joint_idx] += STEP_SIZE_RAD * direction
                moved = True

        if not moved:
            return

        self.send_goal(target_pos)

    # ---------- Send Goal (NON-BLOCKING) ----------
    def send_goal(self, target_pos: List[float]):
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

        # 🔥 Non-blocking call (critical fix)
        self._traj_client.send_goal_async(goal)


# ================= MAIN =================
def main():
    rclpy.init()
    node = JoyTeleopNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
