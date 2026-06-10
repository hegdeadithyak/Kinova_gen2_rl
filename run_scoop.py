#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, DUR_TO_FEEDING

SEQUENCE = [
    # pre_scoop
    [2.473036155645718, 3.860704204822951, 1.1286894816046955,
     7.973092578992713, 3.928636916857019, 5.490991821944691],
    # after_pre_scoop
    [2.5326994861789816, 4.036655263302851, 1.2444419141722265,
     1.6899869003297199, 3.9285500978054824, 5.4910413567409675],
    # scoopv0
    [2.2621428257673246, 3.981920911315107,  1.1588610993844757,
     -4.4203089183591215, 4.598469068963242, 5.372604191478527],
    # feed
    [2.69807965934403, 4.241848895816258, 1.2520215369507008,
     1.2014285336160841, 4.778086501315855, 5.36437821950349],
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
