#!/usr/bin/env python3
"""
Kinova j2s6s200 — Keyboard Joint Teleop + Recorder
Run:  python3 kinova_keyboard_teleop.py
"""

import rclpy
from rclpy.node import Node
from kinova_msgs.msg import JointVelocity, JointAngles
import curses
import threading
import time

ROBOT         = "j2s6s200"
VEL_TOPIC     = f"/{ROBOT}_driver/in/joint_velocity"
ANGLE_TOPIC   = f"/{ROBOT}_driver/out/joint_angles"
VEL_STEP      = 0.05   # rad/s per keypress
MAX_VEL       = 0.4    # rad/s safety cap
RECORD_HZ     = 10

JOINT_NAMES = ["J1 Shoulder", "J2 Upper Arm", "J3 Elbow",
               "J4 Forearm",  "J5 Wrist",     "J6 Hand"]

# ── ROS Node ─────────────────────────────────────────────────────────────────

class KinovaNode(Node):
    def __init__(self):
        super().__init__("kinova_kb_teleop")
        self.vel_pub = self.create_publisher(JointVelocity, VEL_TOPIC, 10)
        self.create_subscription(JointAngles, ANGLE_TOPIC, self._angle_cb, 10)
        self.create_timer(0.02, self._pub_vel)         # 50 Hz publish
        self.create_timer(1/RECORD_HZ, self._record)  # angle sampling

        self.velocities = [0.0] * 6
        self.angles     = [0.0] * 6
        self.selected   = 0
        self.recording  = False
        self.waypoints  = []   # list of (time_s, [j1..j6])
        self.rec_start  = 0.0

    def _angle_cb(self, msg):
        self.angles = [msg.joint1, msg.joint2, msg.joint3,
                       msg.joint4, msg.joint5, msg.joint6]

    def _pub_vel(self):
        msg = JointVelocity()
        (msg.joint1, msg.joint2, msg.joint3,
         msg.joint4, msg.joint5, msg.joint6) = self.velocities
        self.vel_pub.publish(msg)

    def _record(self):
        if self.recording:
            t = round(time.time() - self.rec_start, 2)
            self.waypoints.append((t, list(self.angles)))

    # ── helpers ──────────────────────────────────────────────────────────────

    def adjust_vel(self, delta):
        j = self.selected
        self.velocities[j] = round(
            max(-MAX_VEL, min(MAX_VEL, self.velocities[j] + delta)), 4)

    def stop(self):
        self.velocities[self.selected] = 0.0

    def stop_all(self):
        self.velocities = [0.0] * 6

    def toggle_record(self):
        self.recording = not self.recording
        if self.recording:
            self.rec_start = time.time()
            self.waypoints = []

    def trajectory_text(self):
        if not self.waypoints:
            return "# No waypoints recorded yet.\n"
        lines = [
            "# Kinova j2s6s200 scooping trajectory",
            "# angles in degrees, time in seconds",
            "SCOOPING_TRAJECTORY = [",
            "    # (time_s, [j1, j2, j3, j4, j5, j6])",
        ]
        for t, a in self.waypoints:
            r = [round(x, 2) for x in a]
            lines.append(f"    ({t:.2f}, {r}),")
        lines.append("]")
        return "\n".join(lines) + "\n"


# ── Curses UI ─────────────────────────────────────────────────────────────────

def draw(stdscr, node: KinovaNode, msg: str):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    row  = [0]

    def put(text, attr=0):
        if row[0] >= h - 1:
            return
        try:
            stdscr.addstr(row[0], 0, text[:w - 1], attr)
        except curses.error:
            pass
        row[0] += 1

    put(f"  Kinova {ROBOT} — Keyboard Teleop + Recorder",
        curses.A_BOLD | curses.color_pair(1))
    put("  " + "─" * min(w - 4, 56))

    put("  JOINT            vel (r/s)    angle (deg)", curses.A_DIM)
    for i, name in enumerate(JOINT_NAMES):
        sel   = i == node.selected
        mark  = "►" if sel else " "
        attr  = curses.A_BOLD | curses.color_pair(2) if sel else 0
        put(f"  {mark} {name:<15} {node.velocities[i]:+.3f}       {node.angles[i]:7.2f}°", attr)

    put("")
    put("  " + "─" * min(w - 4, 56))
    put("  1-6  select joint   W/S  faster/slower   SPACE stop joint", curses.A_DIM)
    put("  A    stop ALL       R    record toggle    P     print+save", curses.A_DIM)
    put("  C    clear record   Q    quit", curses.A_DIM)
    put("")

    rec_label = "● REC" if node.recording else "  ---"
    rec_attr  = curses.A_BOLD | curses.color_pair(3) if node.recording else curses.A_DIM
    put(f"  {rec_label}  {len(node.waypoints)} waypoints", rec_attr)

    if msg:
        put("")
        put(f"  {msg}", curses.color_pair(4) | curses.A_BOLD)

    stdscr.refresh()


def show_popup(stdscr, text: str):
    lines = text.splitlines()
    h, w  = stdscr.getmaxyx()
    win_h = min(len(lines) + 4, h - 2)
    win_w = min(max((len(l) for l in lines), default=20) + 4, w - 2)
    win   = curses.newwin(win_h, win_w, 1, 2)
    win.border()
    for i, line in enumerate(lines[:win_h - 2]):
        try:
            win.addstr(i + 1, 2, line[:win_w - 4])
        except curses.error:
            pass
    try:
        win.addstr(win_h - 1, 2, "[ press any key to close ]")
    except curses.error:
        pass
    win.refresh()
    win.getch()


def main_loop(stdscr, node: KinovaNode):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN,   -1)   # title
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # selected joint
    curses.init_pair(3, curses.COLOR_RED,    -1)   # recording
    curses.init_pair(4, curses.COLOR_GREEN,  -1)   # status message

    stdscr.timeout(100)   # refresh every 100 ms even with no key

    msg = "Ready — press 1-6 to select a joint, then W/S to move it."

    while True:
        draw(stdscr, node, msg)
        key = stdscr.getch()

        if key == -1:
            continue

        ch = chr(key) if 32 <= key <= 126 else ""

        if ch in "123456":
            node.selected = int(ch) - 1
            msg = f"Selected {JOINT_NAMES[node.selected]}"

        elif ch in ("w", "W"):
            node.adjust_vel(+VEL_STEP)
            msg = f"J{node.selected+1} velocity → {node.velocities[node.selected]:+.3f} r/s"

        elif ch in ("s", "S"):
            node.adjust_vel(-VEL_STEP)
            msg = f"J{node.selected+1} velocity → {node.velocities[node.selected]:+.3f} r/s"

        elif key == ord(" "):
            node.stop()
            msg = f"J{node.selected+1} stopped"

        elif ch in ("a", "A"):
            node.stop_all()
            msg = "ALL joints stopped"

        elif ch in ("r", "R"):
            node.toggle_record()
            if node.recording:
                msg = "Recording STARTED — move through the scooping motion"
            else:
                msg = f"Recording STOPPED — {len(node.waypoints)} waypoints captured"

        elif ch in ("c", "C"):
            node.waypoints = []
            node.recording = False
            msg = "Recording cleared"

        elif ch in ("p", "P"):
            traj = node.trajectory_text()
            show_popup(stdscr, traj)
            fname = "scooping_trajectory.py"
            with open(fname, "w") as f:
                f.write(traj)
            msg = f"Saved to {fname}  ({len(node.waypoints)} waypoints)"

        elif ch in ("q", "Q"):
            node.stop_all()
            break


def main():
    rclpy.init()
    node = KinovaNode()

    spin_th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_th.start()

    try:
        curses.wrapper(main_loop, node)
    finally:
        node.stop_all()
        time.sleep(0.15)
        node.destroy_node()
        rclpy.shutdown()
        print("\nExited cleanly.")


if __name__ == "__main__":
    main()