#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, DUR_TO_FEEDING

AFTER_PRE_SCOOP_RAD = [
      2.5326994861789816
, 4.036655263302851
, 1.2444419141722265
, 1.6899869003297199
, 3.9285500978054824
, 5.4910413567409675
]


def main():
    rclpy.init()
    node = FeedingOrchestrator()

    try:
        ok = node.move_joint_space(AFTER_PRE_SCOOP_RAD, DUR_TO_FEEDING, 'TO_AFTER_PRE_SCOOP')
        sys.exit(0 if ok else 1)

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
        sys.exit(130)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
