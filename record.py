#!/usr/bin/env python3
"""
record.py — Record arm movements for replay by scoop.py.

Usage:
  python3 record.py
  Move the arm with the Kinova joystick.
  Press Ctrl+C to stop and save → recorded_scoop.json
  Then: python3 scoop.py
"""
import json
import os
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]
JOINT_STATE_TOPIC = '/j2s6s200_driver/out/joint_state'
SAMPLE_HZ         = 10
OUTPUT_FILE       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recorded_scoop.json')


class RecordNode(Node):
    def __init__(self):
        super().__init__('record_trajectory')
        self._latest_q: Optional[List[float]] = None
        self._waypoints: List[dict] = []
        self._t0 = time.monotonic()

        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self.create_timer(1.0 / SAMPLE_HZ, self._sample)

        print(f"\n{'='*55}")
        print("  Record Arm Trajectory")
        print("="*55)
        print("  Move the arm with the Kinova joystick.")
        print("  Press Ctrl+C to stop and save.")
        print(f"  Output → {OUTPUT_FILE}")
        print(f"{'='*55}\n")
        print("  [REC ●] Recording ...")

    def _js_cb(self, msg: JointState):
        try:
            idx = [msg.name.index(j) for j in JOINT_NAMES]
        except ValueError:
            return
        self._latest_q = [msg.position[i] for i in idx]

    def _sample(self):
        if self._latest_q is None:
            return
        elapsed = time.monotonic() - self._t0
        self._waypoints.append({
            'time_s':    round(elapsed, 4),
            'positions': [round(v, 6) for v in self._latest_q],
        })
        n = len(self._waypoints)
        print(f"\r  [REC ●] {n:4d} pts   t = {elapsed:6.1f}s", end='', flush=True)

    def save(self):
        if not self._waypoints:
            print("\n  Nothing recorded — no file written.")
            return
        print(f"\n\n  Saving {len(self._waypoints)} waypoints ...", end=' ', flush=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(self._waypoints, f, indent=2)
        dur = self._waypoints[-1]['time_s']
        print("done.")
        print(f"  Duration : {dur:.1f} s")
        print(f"  File     : {OUTPUT_FILE}")
        print(f"  Run 'python3 scoop.py' to replay.\n")


def main():
    rclpy.init()
    node = RecordNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
