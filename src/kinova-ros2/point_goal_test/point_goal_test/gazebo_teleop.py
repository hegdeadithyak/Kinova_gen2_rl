#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import curses
import threading
import time

JOINT_NAMES = [
    'j2s6s200_joint_1',
    'j2s6s200_joint_2',
    'j2s6s200_joint_3',
    'j2s6s200_joint_4',
    'j2s6s200_joint_5',
    'j2s6s200_joint_6',
]

TRAJ_TOPIC = '/arm_controller/joint_trajectory'
JS_TOPIC = '/joint_states'
STEP = 0.05  # Radians per keypress

class GazeboTeleop(Node):
    def __init__(self):
        super().__init__('gazebo_teleop')
        self.traj_pub = self.create_publisher(JointTrajectory, TRAJ_TOPIC, 10)
        self.create_subscription(JointState, JS_TOPIC, self._js_cb, 10)
        
        self.current_pos = [0.0] * 6
        self.target_pos = [0.0] * 6
        self.selected_joint = 0
        self.js_received = False

    def _js_cb(self, msg):
        try:
            # Map joint states to our order
            indices = [msg.name.index(name) for name in JOINT_NAMES]
            self.current_pos = [msg.position[i] for i in indices]
            if not self.js_received:
                self.target_pos = list(self.current_pos)
                self.js_received = True
        except ValueError:
            pass

    def jog(self, delta):
        if not self.js_received:
            return
        
        self.target_pos[self.selected_joint] += delta
        
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        
        point = JointTrajectoryPoint()
        point.positions = list(self.target_pos)
        point.time_from_start = Duration(sec=0, nanosec=100000000) # 0.1s
        
        msg.points.append(point)
        self.traj_pub.publish(msg)

def main_loop(stdscr, node):
    curses.curs_set(0)
    stdscr.timeout(100)
    
    while rclpy.ok():
        stdscr.erase()
        stdscr.addstr(0, 0, "Gazebo Joint Teleop (j2s6s200)", curses.A_BOLD)
        stdscr.addstr(1, 0, "-------------------------------")
        
        if not node.js_received:
            stdscr.addstr(3, 0, "Waiting for /joint_states...")
        else:
            for i, name in enumerate(JOINT_NAMES):
                attr = curses.A_REVERSE if i == node.selected_joint else 0
                stdscr.addstr(3 + i, 2, f"{i+1}: {name:<20} {node.current_pos[i]:>8.3f} rad", attr)
            
            stdscr.addstr(10, 0, "Controls:")
            stdscr.addstr(11, 2, "1-6: Select Joint")
            stdscr.addstr(12, 2, "W/S: Increase/Decrease Position")
            stdscr.addstr(13, 2, "Q  : Quit")
        
        stdscr.refresh()
        
        key = stdscr.getch()
        if key == ord('q') or key == ord('Q'):
            break
        elif ord('1') <= key <= ord('6'):
            node.selected_joint = key - ord('1')
        elif key == ord('w') or key == ord('W'):
            node.jog(STEP)
        elif key == ord('s') or key == ord('S'):
            node.jog(-STEP)

def main():
    rclpy.init()
    node = GazeboTeleop()
    
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    try:
        curses.wrapper(main_loop, node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
