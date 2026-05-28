#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, DUR_TO_FEEDING

SEQUENCE = [
    # pre_scoop
    [2.5744562541747698, 3.974333032737253, 1.1287371521882077,
     1.6905496262680537, 3.928425994498685, 5.489848793204829],
    # after_pre_scoop
    [2.503284605622947, 4.1830481631600716, 1.3273229494049092,
     1.7916721165452887, 3.9988194673799864, 5.487007732953933],
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
