#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import RobotTrajectory
from moveit_msgs.srv import GetCartesianPath
from control_msgs.action import FollowJointTrajectory
from std_srvs.srv import Trigger
from rclpy.action import ActionClient
from tf_transformations import quaternion_from_euler
from builtin_interfaces.msg import Duration
import math


class ScoopAction(Node):
    def __init__(self):
        super().__init__('scoop_action')

        # ReentrantCallbackGroup allows service + action to run concurrently
        self.cb_group = ReentrantCallbackGroup()

        self.cartesian_client = self.create_client(
            GetCartesianPath,
            '/compute_cartesian_path',
            callback_group=self.cb_group
        )

        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            callback_group=self.cb_group
        )

        self.get_logger().info('Waiting for MoveIt services...')
        self.cartesian_client.wait_for_service(timeout_sec=10.0)
        self.trajectory_client.wait_for_server(timeout_sec=10.0)

        self.srv = self.create_service(
            Trigger,
            '/trigger_scoop',
            self.trigger_callback,
            callback_group=self.cb_group
        )

        self.is_scooping = False
        self.get_logger().info('Ready. Call /trigger_scoop to scoop.')

    # ------------------------------------------------------------------ #
    #  Service callback                                                   #
    # ------------------------------------------------------------------ #
    def trigger_callback(self, request, response):
        if self.is_scooping:
            response.success = False
            response.message = 'Scoop already in progress.'
            return response

        self.is_scooping = True
        try:
            success = self.execute_scoop()
            response.success = success
            response.message = 'Scoop done.' if success else 'Scoop failed — check logs.'
        except Exception as e:
            response.success = False
            response.message = f'Exception: {str(e)}'
            self.get_logger().error(response.message)
        finally:
            self.is_scooping = False

        return response

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #
    def make_pose(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        pose = Pose()
        pose.position = Point(x=x, y=y, z=z)
        q = quaternion_from_euler(roll, pitch, yaw)
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pose

    def compute_cartesian_path(self, waypoints, eef_step=0.005):
        req = GetCartesianPath.Request()
        req.header.frame_id = 'world'
        req.group_name = 'arm'
        req.link_name = 'j2s6s200_end_effector'
        req.waypoints = waypoints
        req.max_step = eef_step
        req.jump_threshold = 0.0
        req.avoid_collisions = True

        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()

        self.get_logger().info(f'Cartesian coverage: {res.fraction * 100:.1f}%')
        if res.fraction < 0.9:
            self.get_logger().error('Coverage too low — aborting phase.')
            return None

        return res.solution  # RobotTrajectory

    def add_timestamps(self, robot_trajectory: RobotTrajectory, speed: float = 0.05):
        """
        Cartesian paths have no time stamps — the controller will reject them.
        This adds linearly spaced timestamps based on desired speed (m/s).
        """
        traj = robot_trajectory.joint_trajectory
        points = traj.points

        if not points:
            return robot_trajectory

        total_time = 0.0
        prev = points[0]

        for i, point in enumerate(points):
            if i == 0:
                point.time_from_start = Duration(sec=0, nanosec=0)
                prev = point
                continue

            # Estimate travel time from max joint delta / speed
            if prev.positions and point.positions:
                max_delta = max(
                    abs(a - b) for a, b in zip(point.positions, prev.positions)
                )
            else:
                max_delta = 0.01

            dt = max(max_delta / speed, 0.05)   # min 50 ms between points
            total_time += dt

            secs    = int(total_time)
            nanosec = int((total_time - secs) * 1e9)
            point.time_from_start = Duration(sec=secs, nanosec=nanosec)
            prev = point

        robot_trajectory.joint_trajectory.points = points
        return robot_trajectory

    def execute_trajectory(self, robot_trajectory: RobotTrajectory):
        # Add timestamps before sending
        robot_trajectory = self.add_timestamps(robot_trajectory, speed=0.05)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = robot_trajectory.joint_trajectory

        future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Trajectory rejected by controller.')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        self.get_logger().info(f'Trajectory done. Error code: {result.error_code}')
        return result.error_code == 0

    # ------------------------------------------------------------------ #
    #  Scoop sequence                                                     #
    # ------------------------------------------------------------------ #
    def execute_scoop(self):
        # ── Tune these to your setup ─────────────────────────────────────
        start_x, start_y, start_z = 0.45, 0.0, 0.25
        plunge_depth     = 0.06
        scoop_forward    = 0.08
        scoop_arc_dip    = 0.02
        wrist_tilt_pitch = 0.3
        lift_height      = 0.12
        base = dict(roll=math.pi, pitch=0.0, yaw=0.0)
        # ────────────────────────────────────────────────────────────────

        # Phase 1 — Plunge
        self.get_logger().info('Phase 1: Plunge')
        traj = self.compute_cartesian_path([
            self.make_pose(start_x, start_y, start_z - plunge_depth, **base)
        ])
        if not traj or not self.execute_trajectory(traj):
            return False

        # Phase 2 — Scoop arc
        self.get_logger().info('Phase 2: Scoop arc')
        arc_waypoints = []
        for i in range(1, 9):
            t = i / 8
            arc_z = -scoop_arc_dip * 4 * t * (1 - t)
            arc_waypoints.append(
                self.make_pose(start_x + scoop_forward * t,
                               start_y,
                               (start_z - plunge_depth) + arc_z,
                               **base)
            )
        traj = self.compute_cartesian_path(arc_waypoints, eef_step=0.003)
        if not traj or not self.execute_trajectory(traj):
            return False

        # Phase 3 — Tilt wrist
        self.get_logger().info('Phase 3: Tilt wrist')
        end_x = start_x + scoop_forward
        traj = self.compute_cartesian_path([
            self.make_pose(end_x, start_y, start_z - plunge_depth,
                           roll=base['roll'],
                           pitch=base['pitch'] + wrist_tilt_pitch,
                           yaw=base['yaw'])
        ])
        if not traj or not self.execute_trajectory(traj):
            return False

        # Phase 4 — Lift
        self.get_logger().info('Phase 4: Lift')
        traj = self.compute_cartesian_path([
            self.make_pose(end_x, start_y, start_z - plunge_depth + lift_height,
                           roll=base['roll'],
                           pitch=base['pitch'] + wrist_tilt_pitch,
                           yaw=base['yaw'])
        ])
        if not traj or not self.execute_trajectory(traj):
            return False

        self.get_logger().info('Scoop complete!')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = ScoopAction()

    # MultiThreadedExecutor — prevents deadlock inside service callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()