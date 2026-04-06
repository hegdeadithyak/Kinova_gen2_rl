#!/usr/bin/env python3
"""
feeding_node.py
═══════════════════════════════════════════════════════════════════════════
Autonomous feeding pipeline for Kinova j2n6s300 — ROS2 Humble.

Uses moveit_msgs actions/services directly — NO moveit_py bindings needed.
  • /move_group                 (moveit_msgs/action/MoveGroup)
  • /compute_cartesian_path     (moveit_msgs/srv/GetCartesianPath)
  • /execute_trajectory         (moveit_msgs/action/ExecuteTrajectory)
  • /finger_trajectory_controller/follow_joint_trajectory

USAGE
─────
  # Terminal 1
  ros2 launch kinova_bringup moveit_robot_launch.py

  # Terminal 2  (after move_group prints "Ready to take commands")
  ros2 run kinova_bringup feeding_node.py
"""

import time
import threading
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Vector3
from std_msgs.msg import String, Int32
from visualization_msgs.msg import Marker, MarkerArray
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from shape_msgs.msg import SolidPrimitive

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, JointConstraint,
    BoundingVolume, OrientationConstraint, PositionConstraint,
    MoveItErrorCodes,
)


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

ROBOT_NAME    = 'j2n6s300'
ARM_GROUP     = 'arm'
EE_LINK       = f'{ROBOT_NAME}_end_effector'
BASE_FRAME    = 'world'

ARM_JOINTS    = [f'{ROBOT_NAME}_joint_{i}' for i in range(1, 7)]
FINGER_JOINTS = [f'{ROBOT_NAME}_joint_finger_{i}' for i in range(1, 4)]

HOME_JOINTS   = [4.71, 2.71, 1.57, 4.71, 0.0, 3.14]
GRIPPER_OPEN  = [0.15, 0.15, 0.15]
GRIPPER_CLOSE = [0.70, 0.70, 0.70]

SPOON_HANDLE_XYZ   = (0.469, -0.147, 0.800)
PRE_GRASP_XYZ      = (0.469, -0.147, 0.870)
LIFT_XYZ           = (0.469, -0.147, 0.880)
ABOVE_BOWL_XYZ     = (0.550, -0.250, 0.895)
SCOOP_DOWN_XYZ     = (0.550, -0.250, 0.838)
SCOOP_UP_XYZ       = (0.550, -0.250, 0.905)
MOUTH_APPROACH_XYZ = (0.580,  0.000, 0.990)
MOUTH_DELIVER_XYZ  = (0.640,  0.000, 0.990)
MOUTH_RETRACT_XYZ  = (0.520,  0.000, 0.990)

Q_SCOOP = (0.0,  0.7071, 0.0, 0.7071)
Q_FEED  = (0.0, -0.7071, 0.0, 0.7071)

VEL_SLOW   = 0.08
VEL_NORMAL = 0.20
VEL_FAST   = 0.35

DWELL_AT_MOUTH_SEC = 2.5
BITE_RETRACT_PAUSE = 0.8
MAX_FEEDING_CYCLES = 3

PLAN_TIMEOUT    = 15.0
EXECUTE_TIMEOUT = 30.0
CART_TIMEOUT    = 10.0


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def make_pose(xyz, quat) -> Pose:
    p = Pose()
    p.position    = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
    p.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]),
                               z=float(quat[2]), w=float(quat[3]))
    return p


def make_stamped_pose(xyz, quat, frame='world') -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose = make_pose(xyz, quat)
    return ps


def error_code_str(code: int) -> str:
    names = {v: k for k, v in vars(MoveItErrorCodes).items()
             if isinstance(v, int)}
    return names.get(code, str(code))


class FeedingState(Enum):
    IDLE            = auto()
    HOMING          = auto()
    OPENING_GRIPPER = auto()
    PRE_GRASP       = auto()
    GRASPING        = auto()
    CLOSING_GRIPPER = auto()
    LIFTING         = auto()
    ABOVE_BOWL      = auto()
    SCOOPING        = auto()
    LIFTING_FOOD    = auto()
    APPROACHING     = auto()
    DELIVERING      = auto()
    DWELLING        = auto()
    RETRACTING      = auto()
    RETURNING       = auto()
    DONE            = auto()
    ERROR           = auto()


# ═══════════════════════════════════════════════════════════════════════
# FEEDING NODE
# ═══════════════════════════════════════════════════════════════════════

class FeedingNode(Node):

    def __init__(self):
        super().__init__('feeding_node')
        self.get_logger().info('╔══════════════════════════════════╗')
        self.get_logger().info('║  Kinova Feeding Node — Starting  ║')
        self.get_logger().info('╚══════════════════════════════════╝')

        # Use ReentrantCallbackGroup so the background thread can call
        # spin_until_future_complete while the executor is also spinning.
        self._cb = ReentrantCallbackGroup()

        # ── Publishers ────────────────────────────────────────────────
        self._status_pub = self.create_publisher(String,      '/feeding/status',      10)
        self._state_pub  = self.create_publisher(String,      '/feeding/state',       10)
        self._cycle_pub  = self.create_publisher(Int32,       '/feeding/cycle_count', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/feeding/waypoints',   10)

        self._state       = FeedingState.IDLE
        self._cycle_count = 0
        self._food_items  = MAX_FEEDING_CYCLES

        # ── Action / service clients (created here, NOT waited on yet) ─
        self._move_group_client = ActionClient(
            self, MoveGroup, '/move_action',
            callback_group=self._cb)
        self._execute_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory',
            callback_group=self._cb)
        self._finger_client = ActionClient(
            self, FollowJointTrajectory,
            '/finger_trajectory_controller/follow_joint_trajectory',
            callback_group=self._cb)
        self._cartesian_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path',
            callback_group=self._cb)

        # ── Background thread: waits for servers THEN runs sequence ───
        # Waiting happens here (after spin starts) to avoid the deadlock
        # where wait_for_server() blocks __init__ before spin() is called.
        self._feed_thread = threading.Thread(
            target=self._wait_then_run, daemon=True)
        self._feed_thread.start()

    # ── Wait for all servers, then start the feeding loop ─────────────

    def _wait_then_run(self):
        """
        Called in background thread AFTER rclpy.spin() has started.
        Safe to call wait_for_server() here because the executor is running.
        """
        self.get_logger().info('Waiting for /move_group action server...')
        self._move_group_client.wait_for_server()
        self.get_logger().info('Waiting for /execute_trajectory action server...')
        self._execute_client.wait_for_server()
        self.get_logger().info('Waiting for /compute_cartesian_path service...')
        self._cartesian_client.wait_for_service()
        self.get_logger().info('Waiting for finger controller action server...')
        self._finger_client.wait_for_server()
        self.get_logger().info('All servers ready ✓')

        self._publish_waypoint_markers()
        self._feeding_loop()

    # ── State / logging ───────────────────────────────────────────────

    def _set_state(self, state: FeedingState, msg: str = ''):
        self._state = state
        s = String(); s.data = state.name
        self._state_pub.publish(s)
        status = String(); status.data = f'[{state.name}] {msg}'
        self._status_pub.publish(status)
        self.get_logger().info(f'  ► {state.name}: {msg}')

    # ── MoveGroup action helper ───────────────────────────────────────

    def _send_move_group_goal(self, request: MotionPlanRequest,
                              vel_scale: float = VEL_NORMAL) -> bool:
        request.max_velocity_scaling_factor     = vel_scale
        request.max_acceleration_scaling_factor = 0.1
        request.allowed_planning_time           = PLAN_TIMEOUT
        request.num_planning_attempts           = 5

        goal = MoveGroup.Goal()
        goal.request                          = request
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3
        goal.planning_options.replan_delay    = 2.0

        future = self._move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future,
                                         timeout_sec=PLAN_TIMEOUT + 5.0)

        if not future.result() or not future.result().accepted:
            self.get_logger().warn('  ✗ MoveGroup goal rejected')
            return False

        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future,
                                         timeout_sec=EXECUTE_TIMEOUT)

        if result_future.result() is None:
            self.get_logger().warn('  ✗ MoveGroup result timeout')
            return False

        code = result_future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(
                f'  ✗ MoveGroup failed: {error_code_str(code)} ({code})')
            return False
        return True

    # ── Joint-space motion ────────────────────────────────────────────

    def _go_joints(self, joint_values: list,
                   vel: float = VEL_NORMAL, label: str = '') -> bool:
        req = MotionPlanRequest()
        req.group_name = ARM_GROUP
        req.workspace_parameters.header.frame_id = BASE_FRAME
        req.workspace_parameters.min_corner = Vector3(x=-2.0, y=-2.0, z=-2.0)
        req.workspace_parameters.max_corner = Vector3(x= 2.0, y= 2.0, z= 2.0)

        constraints = Constraints()
        for name, value in zip(ARM_JOINTS, joint_values):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(value)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)

        ok = self._send_move_group_goal(req, vel_scale=vel)
        if not ok:
            self.get_logger().warn(f'  ✗ Joint motion FAILED: {label}')
        return ok

    # ── Pose-space motion ─────────────────────────────────────────────

    def _go_pose(self, xyz, quat,
                 vel: float = VEL_NORMAL, label: str = '') -> bool:
        req = MotionPlanRequest()
        req.group_name = ARM_GROUP
        req.workspace_parameters.header.frame_id = BASE_FRAME
        req.workspace_parameters.min_corner = Vector3(x=-2.0, y=-2.0, z=-2.0)
        req.workspace_parameters.max_corner = Vector3(x= 2.0, y= 2.0, z= 2.0)

        target = make_stamped_pose(xyz, quat, frame=BASE_FRAME)
        target.header.stamp = self.get_clock().now().to_msg()

        pos_c = PositionConstraint()
        pos_c.header    = target.header
        pos_c.link_name = EE_LINK
        pos_c.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        prim = SolidPrimitive()
        prim.type       = SolidPrimitive.SPHERE
        prim.dimensions = [0.01]
        region = BoundingVolume()
        region.primitives.append(prim)
        region.primitive_poses.append(target.pose)
        pos_c.constraint_region = region
        pos_c.weight = 1.0

        ori_c = OrientationConstraint()
        ori_c.header                    = target.header
        ori_c.link_name                 = EE_LINK
        ori_c.orientation               = target.pose.orientation
        ori_c.absolute_x_axis_tolerance = 0.05
        ori_c.absolute_y_axis_tolerance = 0.05
        ori_c.absolute_z_axis_tolerance = 0.05
        ori_c.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_c)
        constraints.orientation_constraints.append(ori_c)
        req.goal_constraints.append(constraints)

        ok = self._send_move_group_goal(req, vel_scale=vel)
        if not ok:
            self.get_logger().warn(f'  ✗ Pose motion FAILED: {label}')
        return ok

    # ── Cartesian path ────────────────────────────────────────────────

    def _go_cartesian(self, waypoints_xyz, quat,
                      vel_scale: float = 0.05, label: str = '') -> bool:
        req = GetCartesianPath.Request()
        req.header.frame_id  = BASE_FRAME
        req.header.stamp     = self.get_clock().now().to_msg()
        req.group_name       = ARM_GROUP
        req.link_name        = EE_LINK
        req.waypoints        = [make_pose(xyz, quat) for xyz in waypoints_xyz]
        req.max_step         = 0.005
        req.jump_threshold   = 0.0
        req.avoid_collisions = True

        future = self._cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=CART_TIMEOUT)

        if not future.result():
            self.get_logger().warn(f'  ✗ Cartesian service timeout: {label}')
            return False

        resp = future.result()
        if resp.fraction < 0.9:
            self.get_logger().warn(
                f'  ✗ Cartesian path only {resp.fraction*100:.0f}%: {label}')
            return False

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = resp.solution

        exec_future = self._execute_client.send_goal_async(exec_goal)
        rclpy.spin_until_future_complete(self, exec_future,
                                         timeout_sec=EXECUTE_TIMEOUT)

        if not exec_future.result() or not exec_future.result().accepted:
            self.get_logger().warn(f'  ✗ Execute rejected: {label}')
            return False

        result_future = exec_future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future,
                                         timeout_sec=EXECUTE_TIMEOUT)

        if result_future.result() is None:
            self.get_logger().warn(f'  ✗ Execute timeout: {label}')
            return False

        code = result_future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(
                f'  ✗ Execute failed: {error_code_str(code)}: {label}')
            return False
        return True

    # ── Gripper ───────────────────────────────────────────────────────

    def _set_gripper(self, positions: list, label: str = '',
                     duration_sec: float = 2.0) -> bool:
        if not self._finger_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('  Finger action server not available')
            return False

        traj = JointTrajectory()
        traj.joint_names = FINGER_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions  = [float(p) for p in positions]
        pt.velocities = [0.0] * 3
        s  = int(duration_sec)
        ns = int((duration_sec - s) * 1e9)
        pt.time_from_start = Duration(sec=s, nanosec=ns)
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = self._finger_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if not future.result() or not future.result().accepted:
            self.get_logger().warn(f'  Gripper goal rejected: {label}')
            return False

        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)
        self.get_logger().info(f'  Gripper {label} ✓')
        return True

    def _open_gripper(self)  -> bool: return self._set_gripper(GRIPPER_OPEN,  'OPEN')
    def _close_gripper(self) -> bool: return self._set_gripper(GRIPPER_CLOSE, 'CLOSE')
    def _go_home(self)       -> bool: return self._go_joints(HOME_JOINTS, VEL_FAST, 'home')

    # ── RViz markers ──────────────────────────────────────────────────

    def _publish_waypoint_markers(self):
        ma = MarkerArray()
        waypoints = [
            ('home',        (0.40, -0.50, 1.20),  (0.5, 0.5, 0.5)),
            ('pre_grasp',   PRE_GRASP_XYZ,         (0.8, 0.8, 0.0)),
            ('grasp',       SPOON_HANDLE_XYZ,      (0.0, 1.0, 0.0)),
            ('above_bowl',  ABOVE_BOWL_XYZ,        (0.0, 0.6, 1.0)),
            ('scoop_down',  SCOOP_DOWN_XYZ,        (0.0, 0.3, 0.8)),
            ('approach',    MOUTH_APPROACH_XYZ,    (1.0, 0.5, 0.0)),
            ('deliver',     MOUTH_DELIVER_XYZ,     (1.0, 0.0, 0.0)),
        ]
        for i, (name, xyz, color) in enumerate(waypoints):
            m = Marker()
            m.header.frame_id = 'world'
            m.ns, m.id = 'feeding_waypoints', i
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose   = make_pose(xyz, (0, 0, 0, 1))
            m.scale  = Vector3(x=0.035, y=0.035, z=0.035)
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 0.85
            ma.markers.append(m)

            t = Marker()
            t.header.frame_id = 'world'
            t.ns, t.id = 'feeding_labels', i
            t.type   = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose   = make_pose((xyz[0], xyz[1], xyz[2] + 0.055), (0, 0, 0, 1))
            t.scale  = Vector3(x=0.04, y=0.04, z=0.04)
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text   = name
            ma.markers.append(t)

        self._marker_pub.publish(ma)

    # ── MAIN FEEDING LOOP ─────────────────────────────────────────────

    def _feeding_loop(self):
        self.get_logger().info('\n' + '═' * 50)
        self.get_logger().info('  Starting feeding sequence')
        self.get_logger().info('═' * 50)

        while self._food_items > 0:
            self._cycle_count += 1
            c = Int32(); c.data = self._cycle_count
            self._cycle_pub.publish(c)
            self.get_logger().info(
                f'\n  ══ CYCLE {self._cycle_count} / {MAX_FEEDING_CYCLES} ══')

            if not self._execute_cycle():
                self.get_logger().error('  Cycle failed — recovering...')
                self._set_state(FeedingState.ERROR, 'Returning home')
                self._go_home()
                time.sleep(2.0)
                continue

            self._food_items -= 1
            self.get_logger().info(
                f'  ✓ Bite delivered! Remaining: {self._food_items}')

        self._set_state(FeedingState.DONE, 'All food delivered.')
        self._go_home()
        self.get_logger().info('\n' + '═' * 50)
        self.get_logger().info('  ✓ Feeding complete!')
        self.get_logger().info('═' * 50)

    def _execute_cycle(self) -> bool:

        # 1. HOME
        self._set_state(FeedingState.HOMING, 'Moving to home')
        if not self._go_home(): return False
        time.sleep(0.3)

        # 2. OPEN GRIPPER
        self._set_state(FeedingState.OPENING_GRIPPER, 'Opening fingers')
        self._open_gripper()
        time.sleep(0.5)

        # 3. PRE-GRASP
        self._set_state(FeedingState.PRE_GRASP, f'Above spoon {PRE_GRASP_XYZ}')
        if not self._go_pose(PRE_GRASP_XYZ, Q_SCOOP, VEL_NORMAL, 'pre_grasp'):
            return False
        time.sleep(0.2)

        # 4. GRASP DESCENT (Cartesian)
        self._set_state(FeedingState.GRASPING, f'Descend to {SPOON_HANDLE_XYZ}')
        if not self._go_cartesian(
            [PRE_GRASP_XYZ, SPOON_HANDLE_XYZ], Q_SCOOP,
            vel_scale=0.04, label='grasp_descent'
        ):
            if not self._go_pose(SPOON_HANDLE_XYZ, Q_SCOOP, VEL_SLOW, 'grasp_fb'):
                return False
        time.sleep(0.3)

        # 5. CLOSE GRIPPER
        self._set_state(FeedingState.CLOSING_GRIPPER, 'Gripping spoon')
        self._close_gripper()
        time.sleep(0.8)

        # 6. LIFT (Cartesian)
        self._set_state(FeedingState.LIFTING, f'Lifting to {LIFT_XYZ}')
        if not self._go_cartesian(
            [SPOON_HANDLE_XYZ, LIFT_XYZ], Q_SCOOP,
            vel_scale=0.05, label='lift'
        ): return False
        time.sleep(0.2)

        # 7. ABOVE BOWL
        self._set_state(FeedingState.ABOVE_BOWL, f'Over bowl {ABOVE_BOWL_XYZ}')
        if not self._go_pose(ABOVE_BOWL_XYZ, Q_SCOOP, VEL_NORMAL, 'above_bowl'):
            return False
        time.sleep(0.2)

        # 8. SCOOP DOWN (Cartesian)
        self._set_state(FeedingState.SCOOPING, f'Scoop at {SCOOP_DOWN_XYZ}')
        if not self._go_cartesian(
            [ABOVE_BOWL_XYZ, SCOOP_DOWN_XYZ], Q_SCOOP,
            vel_scale=0.04, label='scoop_down'
        ): return False
        time.sleep(0.4)

        # 9. LIFT FOOD (Cartesian)
        self._set_state(FeedingState.LIFTING_FOOD, f'Lift food {SCOOP_UP_XYZ}')
        if not self._go_cartesian(
            [SCOOP_DOWN_XYZ, SCOOP_UP_XYZ], Q_SCOOP,
            vel_scale=0.04, label='scoop_up'
        ): return False
        time.sleep(0.3)

        # 10. APPROACH MOUTH
        self._set_state(FeedingState.APPROACHING, f'Approach {MOUTH_APPROACH_XYZ}')
        if not self._go_pose(MOUTH_APPROACH_XYZ, Q_FEED, VEL_NORMAL, 'approach'):
            return False
        time.sleep(0.3)

        # 11. DELIVER (Cartesian, slow)
        self._set_state(FeedingState.DELIVERING, f'Deliver to {MOUTH_DELIVER_XYZ}')
        if not self._go_cartesian(
            [MOUTH_APPROACH_XYZ, MOUTH_DELIVER_XYZ], Q_FEED,
            vel_scale=VEL_SLOW, label='deliver'
        ):
            self._go_pose(MOUTH_DELIVER_XYZ, Q_FEED, VEL_SLOW, 'deliver_fb')

        # 12. DWELL
        self._set_state(FeedingState.DWELLING, f'Holding {DWELL_AT_MOUTH_SEC}s')
        time.sleep(DWELL_AT_MOUTH_SEC)

        # 13. RETRACT (Cartesian)
        self._set_state(FeedingState.RETRACTING, f'Retract to {MOUTH_RETRACT_XYZ}')
        time.sleep(BITE_RETRACT_PAUSE)
        if not self._go_cartesian(
            [MOUTH_DELIVER_XYZ, MOUTH_RETRACT_XYZ], Q_FEED,
            vel_scale=VEL_SLOW, label='retract'
        ):
            self._go_pose(MOUTH_RETRACT_XYZ, Q_FEED, VEL_SLOW, 'retract_fb')

        # 14. RETURN TO BOWL
        self._set_state(FeedingState.RETURNING, 'Returning to bowl')
        self._go_pose(ABOVE_BOWL_XYZ, Q_SCOOP, VEL_FAST, 'return_bowl')

        return True


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT — uses MultiThreadedExecutor so the background thread
# can call spin_until_future_complete while spin() is also running.
# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = FeedingNode()

    # MultiThreadedExecutor is required: the feeding thread calls
    # spin_until_future_complete() concurrently with the main spin loop.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('\n  Interrupted — returning home...')
        node._go_home()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()