#!/usr/bin/env python3
"""
feed_recorder.py — Kinova j2s6s200  Keyboard Teleop + Feed-Pose Recorder

Like teleop.py but for ROS2.  Recordings persist to disk — record once,
use forever.  demo_feed_planner.py loads the same file on every launch.

Run:
    python3 feed_recorder.py          (or via ros2 run after installing)

Keys
----
  1-6   select joint
  W/S   nudge selected joint  +2° / −2°  (hold for continuous motion)
  A     stop all motion
  R     record current (mouth position + joints) to file
  D     delete last recording
  C     clear all recordings
  L     list all recordings
  Q     quit

File
----
  ~/.ros2/kinova_feed_recordings.json
  Human-readable JSON.  Copy it to another machine or back it up as needed.
"""

import curses
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from tf2_geometry_msgs import do_transform_point
from tf2_ros import (Buffer, TransformListener,
                     ConnectivityException, ExtrapolationException, LookupException)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ── Config ───────────────────────────────────────────────────────────────
RECORDINGS_FILE = Path.home() / '.ros2' / 'kinova_feed_recordings.json'
BASE_FRAME      = 'root'
CAM_FRAME       = 'camera_color_optical_frame'
ARM_JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]
JOINT_LABELS = ['J1 Shoulder', 'J2 Upper Arm', 'J3 Elbow',
                'J4 Forearm',  'J5 Wrist',     'J6 Hand']
NUDGE_DEG    = 2.0            # degrees per keypress
NUDGE_RAD    = math.radians(NUDGE_DEG)
NUDGE_DUR_S  = 0.35           # trajectory duration per nudge
MAX_VEL_RAD  = 0.35           # rad/s safety cap for duration


# ── Persistent storage ───────────────────────────────────────────────────

def load_recordings() -> list:
    """Load recordings from disk.  Returns list of dicts."""
    if not RECORDINGS_FILE.exists():
        return []
    try:
        with open(RECORDINGS_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_recordings(recs: list):
    """Save recordings list to disk."""
    RECORDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RECORDINGS_FILE, 'w') as f:
        json.dump(recs, f, indent=2)


# ── ROS node ─────────────────────────────────────────────────────────────

class RecorderNode(Node):

    def __init__(self):
        super().__init__('feed_recorder')
        self._cb = ReentrantCallbackGroup()

        self._tf_buf = Buffer()
        TransformListener(self._tf_buf, self)

        self._joints: dict  = {}          # name → rad
        self._mouth_raw: Optional[PointStamped] = None
        self._mouth_base: Optional[np.ndarray]  = None
        self._status: str   = ''
        self._lock = threading.Lock()

        # Load recordings from disk
        self._recs: list = load_recordings()

        self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10,
            callback_group=self._cb)
        self.create_subscription(
            PointStamped, '/mouth_3d_point', self._mouth_cb, 10,
            callback_group=self._cb)

        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            callback_group=self._cb)

        # Background TF refresh for mouth_base
        self.create_timer(0.1, self._refresh_mouth_base,
                          callback_group=self._cb)

    # ── Callbacks ──────────────────────────────────────────────────────
    def _js_cb(self, msg: JointState):
        with self._lock:
            for n, p in zip(msg.name, msg.position):
                self._joints[n] = float(p)

    def _mouth_cb(self, msg: PointStamped):
        with self._lock:
            self._mouth_raw = msg

    def _refresh_mouth_base(self):
        with self._lock:
            snap = self._mouth_raw
        if snap is None:
            return
        src   = snap.header.frame_id or CAM_FRAME
        stamp = Time.from_msg(snap.header.stamp)
        try:
            tf = self._tf_buf.lookup_transform(
                BASE_FRAME, src, stamp, timeout=Duration(seconds=0.1))
        except Exception:
            try:
                tf = self._tf_buf.lookup_transform(
                    BASE_FRAME, src, Time(), timeout=Duration(seconds=0.1))
            except Exception:
                return
        p = do_transform_point(snap, tf).point
        with self._lock:
            self._mouth_base = np.array([p.x, p.y, p.z])

    # ── Nudge joint ────────────────────────────────────────────────────
    def nudge_joint(self, joint_name: str, delta_rad: float):
        with self._lock:
            if joint_name not in self._joints:
                self._status = f'{joint_name} not in joint states'
                return
            current = dict(self._joints)

        target = dict(current)
        target[joint_name] = current[joint_name] + delta_rad

        names         = [n for n in ARM_JOINT_NAMES if n in target]
        positions     = [target[n] for n in names]
        cur_positions = [current.get(n, target[n]) for n in names]

        duration = max(NUDGE_DUR_S,
                       abs(delta_rad) / MAX_VEL_RAD)

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

        goal            = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # Fire-and-forget — don't block the UI
        self._traj_client.send_goal_async(goal)

    # ── Record ─────────────────────────────────────────────────────────
    def record(self) -> str:
        with self._lock:
            mouth = self._mouth_base
            js    = dict(self._joints)

        if mouth is None:
            return 'No mouth point — is mouth_tracker + camera running?'
        if not js:
            return 'No joint states received.'

        entry = {
            'mouth':  [round(float(v), 4) for v in mouth],
            'joints': {n: round(js[n], 5)
                       for n in ARM_JOINT_NAMES if n in js},
        }
        self._recs.append(entry)
        save_recordings(self._recs)

        n = len(self._recs)
        return (f'Saved #{n}: '
                f'mouth=({mouth[0]:+.3f},{mouth[1]:+.3f},{mouth[2]:+.3f})')

    def delete_last(self) -> str:
        if not self._recs:
            return 'No recordings to delete.'
        self._recs.pop()
        save_recordings(self._recs)
        return f'Deleted last — {len(self._recs)} remaining.'

    def clear_all(self) -> str:
        n = len(self._recs)
        self._recs.clear()
        save_recordings(self._recs)
        return f'Cleared {n} recording(s).'

    # ── State snapshot for UI ──────────────────────────────────────────
    def snapshot(self):
        with self._lock:
            return dict(self._joints), self._mouth_base, list(self._recs)


# ── Curses UI ─────────────────────────────────────────────────────────────

def ui(stdscr, node: RecorderNode):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN,   -1)   # title
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # selected joint
    curses.init_pair(3, curses.COLOR_GREEN,  -1)   # status / mouth
    curses.init_pair(4, curses.COLOR_RED,    -1)   # warning
    stdscr.timeout(150)   # ms between redraws

    selected = 0
    status   = f'Loaded {len(node._recs)} recording(s) from {RECORDINGS_FILE}'

    while True:
        joints, mouth, recs = node.snapshot()
        h, w = stdscr.getmaxyx()
        stdscr.erase()
        r = [0]

        def put(text, attr=0):
            if r[0] >= h - 1:
                return
            try:
                stdscr.addstr(r[0], 0, text[:w - 1], attr)
            except curses.error:
                pass
            r[0] += 1

        put('  Kinova j2s6s200 — Feed Recorder',
            curses.A_BOLD | curses.color_pair(1))
        put('  ' + '─' * min(w - 4, 60))
        put('  JOINT            angle (deg)', curses.A_DIM)

        for i, (name, label) in enumerate(zip(ARM_JOINT_NAMES, JOINT_LABELS)):
            sel  = (i == selected)
            mark = '►' if sel else ' '
            attr = curses.A_BOLD | curses.color_pair(2) if sel else 0
            deg  = math.degrees(joints.get(name, 0.0))
            put(f'  {mark} {label:<15}  {deg:+8.2f}°', attr)

        put('')
        if mouth is not None:
            put(f'  Mouth (base):  ({mouth[0]:+.3f}, {mouth[1]:+.3f}, {mouth[2]:+.3f}) m',
                curses.color_pair(3))
        else:
            put('  Mouth (base):  -- waiting for camera --', curses.A_DIM)

        put('')
        put(f'  Recordings saved: {len(recs)}   ({RECORDINGS_FILE})',
            curses.A_DIM)
        for i, rec in enumerate(recs[-5:]):   # show last 5
            m = rec['mouth']
            idx = len(recs) - min(5, len(recs)) + i + 1
            put(f'    #{idx}: mouth=({m[0]:+.3f},{m[1]:+.3f},{m[2]:+.3f})',
                curses.A_DIM)

        put('')
        put('  ' + '─' * min(w - 4, 60))
        put('  1-6 select   W/S nudge joint   R record pose   D del last',
            curses.A_DIM)
        put('  C clear all  L list all        Q quit',
            curses.A_DIM)
        put('')
        put(f'  {status}', curses.color_pair(3) | curses.A_BOLD)

        stdscr.refresh()

        key = stdscr.getch()
        if key == -1:
            continue
        ch = chr(key) if 32 <= key <= 126 else ''

        if ch in '123456':
            selected = int(ch) - 1
            status   = f'Selected {JOINT_LABELS[selected]}'

        elif ch in ('w', 'W'):
            node.nudge_joint(ARM_JOINT_NAMES[selected], +NUDGE_RAD)
            status = f'J{selected+1} +{NUDGE_DEG:.0f}°'

        elif ch in ('s', 'S'):
            node.nudge_joint(ARM_JOINT_NAMES[selected], -NUDGE_RAD)
            status = f'J{selected+1} -{NUDGE_DEG:.0f}°'

        elif ch in ('a', 'A'):
            status = 'Stop not applicable in position mode — release key'

        elif ch in ('r', 'R'):
            status = node.record()

        elif ch in ('d', 'D'):
            status = node.delete_last()

        elif ch in ('c', 'C'):
            status = node.clear_all()

        elif ch in ('l', 'L'):
            _, _, recs_now = node.snapshot()
            if not recs_now:
                status = 'No recordings.'
            else:
                lines = [f'{len(recs_now)} recording(s):']
                for i, rec in enumerate(recs_now):
                    m = rec['mouth']
                    lines.append(
                        f'  #{i+1}: mouth=({m[0]:+.3f},{m[1]:+.3f},{m[2]:+.3f})')
                # Show in a popup
                _popup(stdscr, '\n'.join(lines))
                status = f'{len(recs_now)} recording(s) listed.'

        elif ch in ('q', 'Q'):
            break


def _popup(stdscr, text: str):
    lines = text.splitlines()
    h, w  = stdscr.getmaxyx()
    pw    = min(max(len(l) for l in lines) + 4, w - 4)
    ph    = min(len(lines) + 4, h - 2)
    win   = curses.newwin(ph, pw, 1, 2)
    win.border()
    for i, line in enumerate(lines[:ph - 2]):
        try:
            win.addstr(i + 1, 2, line[:pw - 4])
        except curses.error:
            pass
    try:
        win.addstr(ph - 1, 2, '[ any key to close ]')
    except curses.error:
        pass
    win.refresh()
    win.getch()


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = RecorderNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_th = threading.Thread(target=executor.spin, daemon=True)
    spin_th.start()

    try:
        curses.wrapper(ui, node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print(f'\nRecordings saved to {RECORDINGS_FILE}')
        print(f'Total recordings: {len(load_recordings())}')


if __name__ == '__main__':
    main()
