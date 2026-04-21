#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import math

# Standard MoveIt messages (already installed with ROS 2 Humble MoveIt)
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint, PlanningOptions

class PreScoopPlanner(Node):
    def __init__(self):
        super().__init__('prescoop_planner')
        # Create an action client to talk to MoveIt's move_group server
        self._action_client = ActionClient(self, MoveGroup, 'move_action')

    def send_goal(self, target_angles_rad):
        self.get_logger().info('Waiting for MoveIt move_action server...')
        self._action_client.wait_for_server()
        
        goal_msg = MoveGroup.Goal()
        
        # 1. Setup the Planning Request
        request = MotionPlanRequest()
        request.group_name = 'arm'  # Must match your SRDF group name
        request.num_planning_attempts = 10
        request.allowed_planning_time = 15.0
        request.max_velocity_scaling_factor = 1.0
        request.max_acceleration_scaling_factor = 1.0

        # 2. Define the Target Joint Constraints
        constraints = Constraints()
        
        # IMPORTANT: These joint names must match your Kinova URDF exactly!
        joint_names = [
            'j2s6s200_joint_1', 
            'j2s6s200_joint_2', 
            'j2s6s200_joint_3', 
            'j2s6s200_joint_4', 
            'j2s6s200_joint_5', 
            'j2s6s200_joint_6'
        ]
                       
        for name, angle in zip(joint_names, target_angles_rad):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = angle
            jc.tolerance_above = 0.01  # Tolerance in Radians
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
            
        request.goal_constraints.append(constraints)
        goal_msg.request = request

        # 3. Setup Planning Options (Plan AND Execute)
        options = PlanningOptions()
        options.plan_only = False  # Set False so MoveIt executes the trajectory bridge!
        options.look_around = False
        options.replan = True
        options.replan_attempts = 3
        goal_msg.planning_options = options

        self.get_logger().info('Sending goal to MoveIt...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('MoveIt rejected the planning goal.')
            return

        self.get_logger().info('MoveIt accepted the goal, planning and executing...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        # error_code 1 means SUCCESS in MoveIt
        if result.error_code.val == 1:
            self.get_logger().info('Successfully reached PRE_SCOOP position!')
        else:
            self.get_logger().error(f'Failed with MoveIt error code: {result.error_code.val}')
        
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    # Your target in degrees
    PRE_SCOOP_DEG = [372.75, 217.16, 57.26, 295.66, 192.63, 251.18]
    # Converted to radians for ROS 2
    PRE_SCOOP_RAD = [math.radians(deg) for deg in PRE_SCOOP_DEG]

    node = PreScoopPlanner()
    node.send_goal(PRE_SCOOP_RAD)
    
    # Spin to wait for the action server callbacks
    rclpy.spin(node)

if __name__ == '__main__':
    main()