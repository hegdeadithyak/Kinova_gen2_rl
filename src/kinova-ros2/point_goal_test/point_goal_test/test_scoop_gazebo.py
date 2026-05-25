#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2

JOINT_NAMES = [
    'j2s6s200_joint_1',
    'j2s6s200_joint_2',
    'j2s6s200_joint_3',
    'j2s6s200_joint_4',
    'j2s6s200_joint_5',
    'j2s6s200_joint_6',
]

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
    [2.551642551012715, 4.219014686313809, 1.2363867839811182,
     1.0415872028861821, 4.821501886038544, 5.33597454121336],
]

LABELS = ['pre_scoop', 'after_pre_scoop', 'scoopv0', 'feed']

class ScoopTester(Node):
    def __init__(self):
        super().__init__('scoop_tester')
        self._moveit2 = MoveIt2(
            node=self,
            joint_names=JOINT_NAMES,
            base_link_name='root',
            end_effector_name='j2s6s200_end_effector',
            group_name='arm',
        )

    def run(self):
        self.get_logger().info('Starting scoop transition sequence...')
        for label, positions in zip(LABELS, SEQUENCE):
            self.get_logger().info(f'Moving to {label}...')
            
            # Plan and execute using MoveIt 2
            # We use joint positions directly
            traj = self._moveit2.plan(joint_positions=positions)
            
            if traj is not None:
                self._moveit2.execute(traj)
                self._moveit2.wait_until_executed()
                self.get_logger().info(f'Reached {label}')
            else:
                self.get_logger().error(f'Failed to plan to {label}')
                break
        
        self.get_logger().info('Scoop sequence complete.')

def main():
    rclpy.init()
    node = ScoopTester()
    
    # We need a spinner for MoveIt 2 to receive updates
    from threading import Thread
    thread = Thread(target=rclpy.spin, args=(node,))
    thread.start()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()
    thread.join()

if __name__ == '__main__':
    main()
