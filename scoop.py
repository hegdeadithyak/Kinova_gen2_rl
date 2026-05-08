#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, NUM_CYCLES

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


def main():
    rclpy.init()
    node = FeedingOrchestrator()

    try:
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
