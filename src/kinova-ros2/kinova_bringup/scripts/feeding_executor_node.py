#!/usr/bin/env python3
"""
feeding_executor_node.py
========================
ROS2 Node: Subscribes to /mouth_pose (published by mouth_detector_node
when /feed_trigger is called) and commands the Kinova arm to execute
the feeding motion using MoveIt2 or the Kinova Kortex API.

Subscribed Topics:
  /mouth_pose   (geometry_msgs/PoseStamped)  — target mouth goal

This node uses the MoveIt2 MoveGroupInterface pattern via rclpy action client.
If you are using the Kinova Kortex ROS2 driver directly, see the commented
section at the bottom for the kortex_driver approach.

Prerequisites:
  pip install moveit_msgs  (or build from source for ROS2 Humble/Iron)
  Your MoveIt2 move_group node must be running.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, WorkspaceParameters,
    Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume, RobotState,
)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
import tf2_ros
import tf2_geometry_msgs   # needed for do_transform_pose


class FeedingExecutorNode(Node):
    def __init__(self):
        super().__init__('feeding_executor_node')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('planning_group',  'arm')        # MoveIt group name
        self.declare_parameter('end_effector_link', 'tool_frame')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('planning_time',   10.0)         # seconds
        self.declare_parameter('vel_scale',       0.2)          # 0-1, slow for safety
        self.declare_parameter('acc_scale',       0.1)

        self.planning_group    = self.get_parameter('planning_group').value
        self.ee_link           = self.get_parameter('end_effector_link').value
        self.base_frame        = self.get_parameter('robot_base_frame').value
        self.planning_time     = self.get_parameter('planning_time').value
        self.vel_scale         = self.get_parameter('vel_scale').value
        self.acc_scale         = self.get_parameter('acc_scale').value

        # ── TF2 ─────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── MoveIt2 Action Client ────────────────────────────────────
        self._move_action = ActionClient(self, MoveGroup, '/move_action')

        # ── Subscriber ───────────────────────────────────────────────
        self.create_subscription(PoseStamped, '/mouth_pose',
                                 self.mouth_pose_callback, 1)

        self.get_logger().info('FeedingExecutorNode ready — waiting for /mouth_pose...')

    # ── Mouth pose received → execute ────────────────────────────────

    def mouth_pose_callback(self, msg: PoseStamped):
        self.get_logger().info(
            f'Received mouth pose in frame [{msg.header.frame_id}]: '
            f'({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, '
            f'{msg.pose.position.z:.3f})')

        # Transform pose into robot base frame
        goal_in_base = self._transform_pose(msg)
        if goal_in_base is None:
            self.get_logger().error('TF transform failed. Aborting feed.')
            return

        self.get_logger().info(
            f'Goal in [{self.base_frame}]: '
            f'({goal_in_base.pose.position.x:.3f}, '
            f' {goal_in_base.pose.position.y:.3f}, '
            f' {goal_in_base.pose.position.z:.3f})')

        self._send_moveit_goal(goal_in_base)

    # ── TF transform ─────────────────────────────────────────────────

    def _transform_pose(self, pose_stamped: PoseStamped):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                pose_stamped.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=2.0),
            )
            return tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
        except Exception as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return None

    # ── MoveIt2 motion request ────────────────────────────────────────

    def _send_moveit_goal(self, goal_pose: PoseStamped):
        if not self._move_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available!')
            return

        # Build position constraint (small tolerance sphere around mouth goal)
        pos_constraint          = PositionConstraint()
        pos_constraint.header   = goal_pose.header
        pos_constraint.link_name = self.ee_link
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0
        pos_constraint.weight   = 1.0

        bounding_vol            = BoundingVolume()
        primitive               = SolidPrimitive()
        primitive.type          = SolidPrimitive.SPHERE
        primitive.dimensions    = [0.01]    # 1 cm tolerance
        bounding_vol.primitives.append(primitive)
        bounding_vol.primitive_poses.append(goal_pose.pose)
        pos_constraint.constraint_region = bounding_vol

        # Build orientation constraint (keep upright, large tolerance for flexibility)
        ori_constraint              = OrientationConstraint()
        ori_constraint.header       = goal_pose.header
        ori_constraint.link_name    = self.ee_link
        ori_constraint.orientation  = goal_pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.4
        ori_constraint.absolute_y_axis_tolerance = 0.4
        ori_constraint.absolute_z_axis_tolerance = 0.4
        ori_constraint.weight       = 0.5

        goal_constraints            = Constraints()
        goal_constraints.position_constraints.append(pos_constraint)
        goal_constraints.orientation_constraints.append(ori_constraint)

        # Build motion plan request
        plan_request                           = MotionPlanRequest()
        plan_request.group_name                = self.planning_group
        plan_request.num_planning_attempts     = 5
        plan_request.allowed_planning_time     = self.planning_time
        plan_request.max_velocity_scaling_factor   = self.vel_scale
        plan_request.max_acceleration_scaling_factor = self.acc_scale
        plan_request.goal_constraints.append(goal_constraints)

        # Wrap in MoveGroup goal
        move_goal                  = MoveGroup.Goal()
        move_goal.request          = plan_request
        move_goal.planning_options.plan_only           = False
        move_goal.planning_options.replan              = True
        move_goal.planning_options.replan_attempts     = 3
        move_goal.planning_options.replan_delay        = 2.0

        self.get_logger().info('🦾  Sending MoveGroup goal...')
        future = self._move_action.send_goal_async(
            move_goal,
            feedback_callback=self._feedback_cb,
        )
        future.add_done_callback(self._goal_response_cb)

    def _feedback_cb(self, feedback_msg):
        pass   # optionally log planning state

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('MoveGroup goal rejected.')
            return
        self.get_logger().info('MoveGroup goal accepted — executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        result = future.result().result
        error_code = result.error_code.val
        if error_code == 1:   # MoveItErrorCodes.SUCCESS
            self.get_logger().info('✅  Feeding motion completed successfully!')
        else:
            self.get_logger().error(f'❌  MoveGroup failed with error code: {error_code}')


def main(args=None):
    rclpy.init(args=args)
    node = FeedingExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()