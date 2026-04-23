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
import tty
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as DurationMsg

# Configuration
STEP_SIZE_RAD = 0.1  # How far the joint moves per key press
MOVE_DUR_S = 0.5      # Time allowed to reach the step

ARM_JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]

# Map characters to (joint_index, direction_multiplier)
KEY_MAP = {
    '1': (0, 1.0),  'q': (0, -1.0),
    '2': (1, 1.0),  'w': (1, -1.0),
    '3': (2, 1.0),  'e': (2, -1.0),
    '4': (3, 1.0),  'r': (3, -1.0),
    '5': (4, 1.0),  't': (4, -1.0),
    '6': (5, 1.0),  'y': (5, -1.0),
}

class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self._cb_group = ReentrantCallbackGroup()
        
        # State tracking
        self._current_positions = {name: 0.0 for name in ARM_JOINT_NAMES}
        self._state_lock = threading.Lock()

        # Subscriber for current positions
        self.create_subscription(
            JointState, 
            '/joint_states', 
            self._joint_state_cb, 
            10, 
            callback_group=self._cb_group
        )

        # Action Client for executing moves
        self._traj_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/arm_controller/follow_joint_trajectory', 
            callback_group=self._cb_group
        )
        
        self.get_logger().info('Waiting for trajectory controller...')
        if not self._traj_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Trajectory controller not found!')
            sys.exit(1)

        self.get_logger().info('Ready. Press keys to move joints. CTRL-C to exit.')

    def _joint_state_cb(self, msg: JointState):
        """Always keep an updated record of exactly where the arm is in pure radians."""
        with self._state_lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._current_positions:
                    self._current_positions[name] = pos

    def dispatch_step(self, joint_idx: int, direction: float):
        """Calculates the new target and dispatches the absolute position command."""
        joint_name = ARM_JOINT_NAMES[joint_idx]
        
        with self._state_lock:
            # We base the step on the *actual* current position, preventing runaway queues
            current_pos = [self._current_positions[name] for name in ARM_JOINT_NAMES]
        
        # Calculate target array (only modifying the requested joint)
        target_pos = list(current_pos)
        target_pos[joint_idx] += (STEP_SIZE_RAD * direction)

        print(f"\rMoving {joint_name} to {target_pos[joint_idx]:.3f} rad", end="")

        pt = JointTrajectoryPoint()
        pt.positions = target_pos
        pt.time_from_start = DurationMsg(sec=0, nanosec=int(MOVE_DUR_S * 1e9))

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ARM_JOINT_NAMES
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # Fire and forget asynchronous goal
        self._traj_client.send_goal_async(goal)

# ── Terminal Input Handling ────────────────────────────────────────────────
def get_key(settings):
    """Reads a single character from standard input."""
    tty.setraw(sys.stdin.fileno())
    # Non-blocking read
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    rclpy.init()
    node = KeyboardTeleopNode()
    
    # Run ROS spin in a background thread so we don't block the main input loop
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

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

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()