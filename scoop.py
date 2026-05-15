#!/usr/bin/env python3
import json
import os
import sys
import rclpy
from robot import FeedingOrchestrator, NUM_CYCLES, DUR_TO_FEEDING

RECORDED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recorded_scoop.json')

SCOOP_START_RAD = [
    2.905325523793451,
    3.943063526798088,
    2.134145711139827,
    0.4043288065734776,
    4.5630889738815394,
    1.4033809659248093,
]

TRANSIT_RAD = [
    2.6178218377850753,
    4.10288062276207,
    1.2267091228958793,
    7.314518407762582,
    5.08876767829119,
    6.604563869621552,
]

FEEDING_RAD = [
    3.210582640576623,
    4.127362796347059,
    1.0460198347835374,
    7.942973292328352,
    5.007654183073993,
    6.418817438336232,
]

GOTO_START_DUR = 5.0   # seconds to move to the recording's start position


def main():
    rclpy.init()
    node = FeedingOrchestrator()

    try:
        if os.path.exists(RECORDED_FILE):
            node.get_logger().info(f'Found recorded trajectory: {RECORDED_FILE}')

            # Move to the first position of the recording before replaying
            with open(RECORDED_FILE) as f:
                waypoints = json.load(f)
            start_pos = waypoints[0]['positions']
            node.get_logger().info('Moving to recording start position ...')
            if not node.move_joint_space(start_pos, GOTO_START_DUR, 'TO_RECORDED_START'):
                node.get_logger().error('Failed to reach start position')
                sys.exit(1)

            ok = node.execute_recorded(RECORDED_FILE)
            sys.exit(0 if ok else 1)

        else:
            node.get_logger().info(
                'No recorded_scoop.json found — running built-in scoop cycles.')
            n_ok = node.run_cycles(NUM_CYCLES, SCOOP_START_RAD, TRANSIT_RAD, FEEDING_RAD)
            node.get_logger().info(f'Completed {n_ok}/{NUM_CYCLES} cycles')
            sys.exit(0 if n_ok == NUM_CYCLES else 1)

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
        sys.exit(130)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
