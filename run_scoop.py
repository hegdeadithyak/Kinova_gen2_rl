#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, DUR_TO_FEEDING

SEQUENCE = [
    # pre_scoop
    [2.2336191720700853, 3.8615886406209667, 1.1309358579841773,
     -4.578315863730177, 3.9497097116144895, 5.488212547031085],
    # after_pre_scoop
    [2.2621428257673246, 3.981902269187476,  1.1588577704331129,
     -4.4203089183591215, 3.8565006713583703, 5.372606854639618],
    # scoopv0
    [2.2621428257673246, 3.981920911315107,  1.1588610993844757,
     -4.4203089183591215, 4.598469068963242, 5.372604191478527],
    # feed
    [2.551642551012715, 4.219014686313809, 1.2363867839811182,
     1.0415872028861821, 4.821501886038544, 5.33597454121336],
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
