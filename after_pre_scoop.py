#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, DUR_TO_FEEDING

FEEDING_RAD = [
      2.503284605622947
, 4.1830481631600716
, 1.3273229494049092
, 1.7916721165452887
, 3.9988194673799864
, 5.487007732953933
]


def main():
    rclpy.init()
    node = FeedingOrchestrator()

    try:
        ok = node.move_joint_space(FEEDING_RAD, DUR_TO_FEEDING, 'TO_FEEDING')
        sys.exit(0 if ok else 1)

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
        sys.exit(130)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
