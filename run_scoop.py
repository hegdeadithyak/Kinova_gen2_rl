#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, DUR_TO_FEEDING

SEQUENCE = [
    # pre_scoop
    [2.3099705357840574, 4.094124948046216,  1.1965476251282414,
     1.6528094378283227, 4.644246677572832,  -5.926436237035841],
    # after_pre_scoop
    [2.3105399196251164, 4.094141992277192,  1.1965613404078552,
     1.5149829950915064, 4.2316473909446,    -5.92817900965319],
    # scoopv0
    [2.3104768027072815, 4.09414012806443,   1.1965596093531468,
     1.5361475357485297, 4.213065983386899,  -4.987037587811254],
    # feed
    [2.6095740278890998, 4.124237044175633,  1.2520163437865752,
     1.0394402289943878, 4.879454401887541,   7.525569663093617],
]

LABELS = ['pre_scoop', 'after_pre_scoop', 'scoopv0', 'feed']


def main():
    rclpy.init()
    node = FeedingOrchestrator()

    try:
        for label, positions in zip(LABELS, SEQUENCE):
            ok = node.move_joint_space(positions, DUR_TO_FEEDING, label)
            if not ok:
                node.get_logger().error(f'Failed at step: {label}')
                sys.exit(1)

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
        sys.exit(130)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
