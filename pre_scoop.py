#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, DUR_TO_FEEDING

PRE_SCOOP_RAD = [
      2.473036155645718
, 3.860704204822951
, 1.1286894816046955
, 7.973092578992713
, 3.928636916857019
, 5.490991821944691
]


def main():
    rclpy.init()
    node = FeedingOrchestrator()

    try:
        ok = node.move_joint_space(PRE_SCOOP_RAD, DUR_TO_FEEDING, 'TO_PRE_SCOOP')
        if ok:
            ok = node.move_joint_space(PRE_SCOOP_RAD, DUR_TO_FEEDING, 'TO_PRE_SCOOP_2')
        sys.exit(0 if ok else 1)

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
        sys.exit(130)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
