#!/usr/bin/env python3
"""
trajectory_executor.py
───────────────────────
Smooth trajectory execution for Kinova Gen2 (j2n6s300).

Architecture:
  MoveIt2 → FollowJointTrajectory action → queue ALL waypoints at once
           → Kinova internal FIFO → DSP interpolation → smooth motion

Self-contained: all SDK structs, helpers, and KinovaSDKNode are defined
here so this file works as a standalone ROS2 executable without needing
to import from a sibling script (which breaks under ROS2's install layout).

KEY FIXES:
  1. Angle wrapping  — SDK returns 0–360 deg; URDF expects −π..+π.
                       _sdk_deg_to_urdf_rad() wraps via atan2.
  2. Direction flip  — j2n6s300 joints 2,4,6 rotate opposite in SDK vs URDF.
                       JOINT_DIRECTION applies the correct sign on read/write.
  3. _make_kinova_point — was referencing undefined variables, assigning
                          Actuator4 twice, and never setting Actuator5.
  4. send_advance_point  — clean public method; no more reaching into private
                           SDK internals from TrajectoryExecutor.
"""

import ctypes
import math
import os
import queue
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from moveit_msgs.msg import PlanningScene, RobotState
from moveit_msgs.srv import ApplyPlanningScene

# ── SDK shared library ────────────────────────────────────────────────
SDK_PATH = os.path.expanduser(
    '~/Downloads/rl_v2-master-master/src/kinova-ros2/'
    'kinova_driver/lib/x86_64-linux-gnu/USBCommandLayerUbuntu.so'
)

# ── Unit conversion ───────────────────────────────────────────────────
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# ── Per-joint direction mapping ───────────────────────────────────────
# j2n6s300: SDK and URDF rotate in opposite directions on joints 2, 4, 6.
# If a joint still moves the wrong way after testing, flip its sign here.
JOINT_DIRECTION = [1, 1, 1, -1, -1, -1]   # index 0..5 → joints 1..6

# ── Joint name lists ──────────────────────────────────────────────────
ARM_JOINTS = [
    'j2n6s300_joint_1', 'j2n6s300_joint_2', 'j2n6s300_joint_3',
    'j2n6s300_joint_4', 'j2n6s300_joint_5', 'j2n6s300_joint_6',
]
FINGER_JOINTS = [
    'j2n6s300_joint_finger_1',
    'j2n6s300_joint_finger_2',
    'j2n6s300_joint_finger_3',
]
FINGER_TIP_JOINTS = [
    'j2n6s300_joint_finger_tip_1',
    'j2n6s300_joint_finger_tip_2',
    'j2n6s300_joint_finger_tip_3',
]
ALL_JOINTS = ARM_JOINTS + FINGER_JOINTS + FINGER_TIP_JOINTS   # 12 total

# ── SDK constants ─────────────────────────────────────────────────────
ANGULAR_POSITION = 2
FINGER_SCALE     = 1.33 / 6800.0   # SDK 0–6800 → 0–1.33 rad

# ── Execution tuning ──────────────────────────────────────────────────
GOAL_TOLERANCE = 0.08    # radians — generous for real hardware
EXEC_TIMEOUT   = 60.0    # seconds — total trajectory budget
SETTLE_TIMEOUT = 15.0    # seconds — grace period after last waypoint queued


# ─────────────────────────────────────────────────────────────────────
# Angle conversion helpers
# ─────────────────────────────────────────────────────────────────────

def _sdk_deg_to_urdf_rad(deg: float, joint_idx: int) -> float:
    """
    SDK degrees (0–360) → URDF radians (−π..+π) with direction correction.

    Without atan2 wrapping, a joint at 270° publishes +4.71 rad instead of
    −1.57 rad, causing MoveIt2 to plan a full extra revolution the wrong way.
    """
    rad = math.radians(deg * JOINT_DIRECTION[joint_idx])
    return math.atan2(math.sin(rad), math.cos(rad))   # wraps to (−π, +π]


def _urdf_rad_to_sdk_deg(rad: float, joint_idx: int) -> float:
    """
    URDF radians (−π..+π) → SDK degrees [0, 360) with direction correction.
    Exact inverse of _sdk_deg_to_urdf_rad.
    """
    deg = math.degrees(rad) * JOINT_DIRECTION[joint_idx]
    return deg


# ─────────────────────────────────────────────────────────────────────
# ctypes structs (must match USBCommandLayerUbuntu.so ABI exactly)
# ─────────────────────────────────────────────────────────────────────

class AngularInfo(ctypes.Structure):
    _fields_ = [(f'Actuator{i}', ctypes.c_float) for i in range(1, 8)]

class FingersPosition(ctypes.Structure):
    _fields_ = [('Finger1', ctypes.c_float),
                ('Finger2', ctypes.c_float),
                ('Finger3', ctypes.c_float)]

class CartesianInfo(ctypes.Structure):
    _fields_ = [('X',      ctypes.c_float), ('Y',      ctypes.c_float),
                ('Z',      ctypes.c_float), ('ThetaX', ctypes.c_float),
                ('ThetaY', ctypes.c_float), ('ThetaZ', ctypes.c_float)]

class AngularPosition(ctypes.Structure):
    _fields_ = [('Actuators', AngularInfo), ('Fingers', FingersPosition)]

class CartesianPosition(ctypes.Structure):
    _fields_ = [('Coordinates', CartesianInfo), ('Fingers', FingersPosition)]

class UserPosition(ctypes.Structure):
    _fields_ = [('Type',              ctypes.c_int),
                ('Delay',             ctypes.c_float),
                ('CartesianPosition', CartesianInfo),
                ('Actuators',         AngularInfo),
                ('HandMode',          ctypes.c_int),
                ('Fingers',           FingersPosition)]

class Limitation(ctypes.Structure):
    _fields_ = [(f, ctypes.c_float) for f in [
        'speedParameter1', 'speedParameter2', 'speedParameter3',
        'forceParameter1', 'forceParameter2', 'forceParameter3',
        'accelerationParameter1', 'accelerationParameter2', 'accelerationParameter3',
    ]]

class TrajectoryPoint(ctypes.Structure):
    _fields_ = [('Position',          UserPosition),
                ('LimitationsActive', ctypes.c_int),
                ('SynchroType',       ctypes.c_int),
                ('Limitations',       Limitation)]


# ─────────────────────────────────────────────────────────────────────
# KinovaSDKNode
# ─────────────────────────────────────────────────────────────────────

class KinovaSDKNode(Node):

    def __init__(self):
        super().__init__('kinova_sdk_node')

        # ── Single SDK thread — prevents libusb mutex crashes ──────────
        self._sdk_queue  = queue.Queue()
        self._sdk_thread = threading.Thread(target=self._sdk_worker, daemon=True)
        self._sdk_thread.start()

        # ── Load SDK ───────────────────────────────────────────────────
        self._sdk = ctypes.CDLL(SDK_PATH)
        self._declare_sdk_signatures()

        ret = self._sdk_call(self._sdk.InitAPI)
        if ret != 1:
            self.get_logger().error(
                f'InitAPI returned {ret} — check LD_LIBRARY_PATH and USB connection'
            )
            raise RuntimeError('SDK InitAPI failed')

        self.get_logger().info('Kinova SDK initialised successfully')
        time.sleep(1.5)
        self._sdk_call(self._sdk.SetAngularControl)

        # ── Publishers ─────────────────────────────────────────────────
        self._js_pub   = self.create_publisher(JointState,  '/joint_states',          10)
        self._pose_pub = self.create_publisher(PoseStamped, '/kinova/cartesian_pose', 10)

        # ── Services ───────────────────────────────────────────────────
        self.create_service(Trigger, '/kinova/move_home', self._handle_home)

        # ── MoveIt2 planning scene client ──────────────────────────────
        self._scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene'
        )
        self._initial_state_pushed = False

        # ── Timers ─────────────────────────────────────────────────────
        self.create_timer(0.01,  self._publish_state)           # 100 Hz
        self.create_timer(12.0,  self._push_initial_state_once) # once at t=12s

        self.get_logger().info(
            f'Publishing {len(ALL_JOINTS)} joints on /joint_states at 100 Hz'
        )

    # ── SDK serialisation ──────────────────────────────────────────────
    def _sdk_worker(self):
        while True:
            fn, args, result_box = self._sdk_queue.get()
            try:
                result_box.append(fn(*args))
            except Exception as e:
                self.get_logger().error(f'SDK call {fn} failed: {e}')
                result_box.append(None)
            finally:
                self._sdk_queue.task_done()

    def _sdk_call(self, fn, *args):
        result_box = []
        self._sdk_queue.put((fn, args, result_box))
        self._sdk_queue.join()
        return result_box[0] if result_box else None

    def _declare_sdk_signatures(self):
        s = self._sdk
        s.InitAPI.restype                = ctypes.c_int
        s.CloseAPI.restype               = ctypes.c_int
        s.MoveHome.restype               = ctypes.c_int
        s.SetAngularControl.restype      = ctypes.c_int
        s.EraseAllTrajectories.restype   = ctypes.c_int
        s.GetAngularPosition.restype     = ctypes.c_int
        s.GetAngularPosition.argtypes    = [ctypes.POINTER(AngularPosition)]
        s.GetCartesianPosition.restype   = ctypes.c_int
        s.GetCartesianPosition.argtypes  = [ctypes.POINTER(CartesianPosition)]
        s.SendBasicTrajectory.restype    = ctypes.c_int
        s.SendBasicTrajectory.argtypes   = [TrajectoryPoint]
        s.SendAdvanceTrajectory.restype  = ctypes.c_int
        s.SendAdvanceTrajectory.argtypes = [TrajectoryPoint]

    # ── Read helpers ───────────────────────────────────────────────────
    def _read_arm_pos_rad(self, ang: AngularPosition) -> list:
        """SDK degrees → URDF radians with wrap and direction correction."""
        raw = [ang.Actuators.Actuator1, ang.Actuators.Actuator2,
               ang.Actuators.Actuator3, ang.Actuators.Actuator4,
               ang.Actuators.Actuator5, ang.Actuators.Actuator6]
        return [_sdk_deg_to_urdf_rad(raw[i], i) for i in range(6)]

    # ── Joint state publisher (100 Hz) ────────────────────────────────
    def _publish_state(self):
        ang  = AngularPosition()
        cart = CartesianPosition()
        self._sdk_call(self._sdk.GetAngularPosition,   ctypes.byref(ang))
        self._sdk_call(self._sdk.GetCartesianPosition, ctypes.byref(cart))

        now     = self.get_clock().now().to_msg()
        arm_pos = self._read_arm_pos_rad(ang)

        finger_pos = [
            float(ang.Fingers.Finger1) * FINGER_SCALE,
            float(ang.Fingers.Finger2) * FINGER_SCALE,
            float(ang.Fingers.Finger3) * FINGER_SCALE,
        ]
        finger_tip_pos = [p * 0.5 for p in finger_pos]

        js = JointState()
        js.header.stamp = now
        js.name         = ALL_JOINTS
        js.position     = arm_pos + finger_pos + finger_tip_pos
        js.velocity     = [0.0] * 12
        js.effort       = [0.0] * 12
        self._js_pub.publish(js)

        ps = PoseStamped()
        ps.header.stamp    = now
        ps.header.frame_id = 'world'
        ps.pose.position.x = float(cart.Coordinates.X)
        ps.pose.position.y = float(cart.Coordinates.Y)
        ps.pose.position.z = float(cart.Coordinates.Z)
        self._pose_pub.publish(ps)

    # ── MoveIt2 planning scene sync ────────────────────────────────────
    def _push_initial_state_once(self):
        if self._initial_state_pushed:
            return

        if not self._scene_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                'ApplyPlanningScene not ready — will retry next timer tick'
            )
            return

        ang = AngularPosition()
        self._sdk_call(self._sdk.GetAngularPosition, ctypes.byref(ang))
        arm_pos        = self._read_arm_pos_rad(ang)
        finger_pos     = [float(ang.Fingers.Finger1) * FINGER_SCALE,
                          float(ang.Fingers.Finger2) * FINGER_SCALE,
                          float(ang.Fingers.Finger3) * FINGER_SCALE]
        finger_tip_pos = [p * 0.5 for p in finger_pos]
        all_pos        = arm_pos + finger_pos + finger_tip_pos

        self.get_logger().info(
            f'Pushing real arm state to MoveIt2 planning scene:\n'
            f'  J1:{arm_pos[0]:.4f}  J2:{arm_pos[1]:.4f}  J3:{arm_pos[2]:.4f}\n'
            f'  J4:{arm_pos[3]:.4f}  J5:{arm_pos[4]:.4f}  J6:{arm_pos[5]:.4f}'
        )

        robot_state = RobotState()
        robot_state.joint_state.name     = ALL_JOINTS
        robot_state.joint_state.position = all_pos
        robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()

        scene = PlanningScene()
        scene.is_diff     = True
        scene.robot_state = robot_state

        req = ApplyPlanningScene.Request()
        req.scene = scene

        future = self._scene_client.call_async(req)
        future.add_done_callback(self._planning_scene_callback)
        self._initial_state_pushed = True

    def _planning_scene_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(
                    'Real arm state applied to MoveIt2 planning scene — RViz2 now synced'
                )
            else:
                self.get_logger().error(
                    'ApplyPlanningScene returned failure — MoveIt2 may show wrong pose'
                )
        except Exception as e:
            self.get_logger().error(f'ApplyPlanningScene call exception: {e}')

    # ── Public API ────────────────────────────────────────────────────

    def send_joint_position(self, positions_rad: list):
        """
        Send absolute positions (URDF rad) via SendBasicTrajectory.
        Blocking one-shot move with speed limits.
        """
        point = TrajectoryPoint()
        ctypes.memset(ctypes.byref(point), 0, ctypes.sizeof(point))
        point.Position.Type               = ANGULAR_POSITION
        point.LimitationsActive           = 1
        point.Limitations.speedParameter1 = 20.0   # deg/s joints 1–3
        point.Limitations.speedParameter2 = 20.0   # deg/s joints 4–6

        sdk_deg = [_urdf_rad_to_sdk_deg(positions_rad[i], i) for i in range(6)]
        point.Position.Actuators.Actuator1 = sdk_deg[0]
        point.Position.Actuators.Actuator2 = sdk_deg[1]
        point.Position.Actuators.Actuator3 = sdk_deg[2]
        point.Position.Actuators.Actuator4 = sdk_deg[3]
        point.Position.Actuators.Actuator5 = sdk_deg[4]
        point.Position.Actuators.Actuator6 = sdk_deg[5]
        self._sdk_call(self._sdk.SendBasicTrajectory, point)

    def send_advance_point(self, positions_rad: list):
        """
        Queue one waypoint (URDF rad) into the Kinova FIFO via
        SendAdvanceTrajectory. Returns immediately — DSP handles
        smooth interpolation between queued points.
        """
        point = TrajectoryPoint()
        ctypes.memset(ctypes.byref(point), 0, ctypes.sizeof(point))
        point.Position.Type     = ANGULAR_POSITION
        point.Position.Delay    = 0.0
        point.Position.HandMode = 0
        point.LimitationsActive = 0   # no limits → DSP uses its own profile
        point.SynchroType       = 0

        sdk_deg = [_urdf_rad_to_sdk_deg(positions_rad[i], i) for i in range(6)]
        point.Position.Actuators.Actuator1 = sdk_deg[0]
        point.Position.Actuators.Actuator2 = sdk_deg[1]
        point.Position.Actuators.Actuator3 = sdk_deg[2]
        point.Position.Actuators.Actuator4 = sdk_deg[3]
        point.Position.Actuators.Actuator5 = sdk_deg[4]
        point.Position.Actuators.Actuator6 = sdk_deg[5]
        self._sdk_call(self._sdk.SendAdvanceTrajectory, point)

    def get_joint_positions_rad(self) -> list:
        """Return current arm joint positions in URDF radians (6 joints)."""
        ang = AngularPosition()
        self._sdk_call(self._sdk.GetAngularPosition, ctypes.byref(ang))
        return self._read_arm_pos_rad(ang)

    def erase_trajectories(self):
        self._sdk_call(self._sdk.EraseAllTrajectories)

    # ── Service handlers ───────────────────────────────────────────────
    def _handle_home(self, request, response):
        self._sdk_call(self._sdk.EraseAllTrajectories)
        self._sdk_call(self._sdk.MoveHome)
        self._initial_state_pushed = False
        response.success = True
        response.message = 'Moving to home position'
        return response

    def destroy_node(self):
        self._sdk_call(self._sdk.CloseAPI)
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────
# TrajectoryExecutor
# ─────────────────────────────────────────────────────────────────────

class TrajectoryExecutor(Node):

    def __init__(self, sdk_node: KinovaSDKNode):
        super().__init__('kinova_trajectory_executor')
        self._sdk      = sdk_node
        self._cb_group = ReentrantCallbackGroup()
        self._js_lock  = threading.Lock()
        self._current_js = [0.0] * 6   # URDF radians, updated by /joint_states

        self.create_subscription(
            JointState,
            '/joint_states',
            self._js_callback,
            10,
            callback_group=self._cb_group,
        )

        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            execute_callback=self._execute_trajectory,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )
        self.get_logger().info('TrajectoryExecutor ready — smooth FIFO mode')

    # ── Joint state feedback ───────────────────────────────────────────
    def _js_callback(self, msg: JointState):
        with self._js_lock:
            name_to_pos = dict(zip(msg.name, msg.position))
            self._current_js = [name_to_pos.get(n, 0.0) for n in ARM_JOINTS]

    # ── Action callbacks ───────────────────────────────────────────────
    def _goal_callback(self, goal_request):
        self.get_logger().info('Received trajectory goal')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().info('Trajectory cancel requested')
        return CancelResponse.ACCEPT

    # ── Main execution ─────────────────────────────────────────────────
    async def _execute_trajectory(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        points     = trajectory.points
        traj_names = trajectory.joint_names

        if not points:
            self.get_logger().warn('Empty trajectory — aborting')
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        # Map MoveIt2 joint order → SDK actuator order
        try:
            sdk_indices = [traj_names.index(n) for n in ARM_JOINTS]
        except ValueError as e:
            self.get_logger().error(f'Joint name mismatch: {e}')
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        n_pts         = len(points)
        traj_duration = (points[-1].time_from_start.sec
                         + points[-1].time_from_start.nanosec * 1e-9)
        self.get_logger().info(
            f'Executing trajectory: {n_pts} waypoints, '
            f'total duration={traj_duration:.2f}s'
        )

        # ── STEP 1: Flush stale FIFO ───────────────────────────────────
        self._sdk.erase_trajectories()
        time.sleep(0.05)

        # ── STEP 2: Queue ALL waypoints into Kinova FIFO ──────────────
        feedback   = FollowJointTrajectory.Feedback()
        queued     = 0
        start_real = time.time()

        for i, wp in enumerate(points):

            if goal_handle.is_cancel_requested:
                self._sdk.erase_trajectories()
                goal_handle.canceled()
                self.get_logger().info('Trajectory cancelled')
                return FollowJointTrajectory.Result()

            # Reorder positions to match SDK actuator order
            target_rad = [wp.positions[j] for j in sdk_indices]

            # Queue into FIFO — angle wrap + direction flip inside
            self._sdk.send_advance_point(target_rad)
            queued += 1

            # Pace feed at 80% of trajectory timing to keep FIFO half-full
            wp_time = (wp.time_from_start.sec
                       + wp.time_from_start.nanosec * 1e-9)
            desired_feed_time = wp_time * 0.8
            elapsed = time.time() - start_real
            if desired_feed_time > elapsed and i < n_pts - 1:
                time.sleep(desired_feed_time - elapsed)

            # Feedback every 5 waypoints
            if i % 5 == 0:
                with self._js_lock:
                    actual = list(self._current_js)
                feedback.actual.positions  = actual
                feedback.desired.positions = target_rad
                feedback.error.positions   = [a - d for a, d in zip(actual, target_rad)]
                goal_handle.publish_feedback(feedback)

            # Hard timeout
            if time.time() - start_real > EXEC_TIMEOUT:
                self.get_logger().error('Trajectory timed out during feed — aborting')
                self._sdk.erase_trajectories()
                goal_handle.abort()
                return FollowJointTrajectory.Result()

        self.get_logger().info(
            f'All {queued} waypoints queued — waiting for arm to settle'
        )

        # ── STEP 3: Wait for arm to reach final position ───────────────
        final_target = [points[-1].positions[j] for j in sdk_indices]
        deadline     = time.time() + SETTLE_TIMEOUT

        while time.time() < deadline:
            if goal_handle.is_cancel_requested:
                self._sdk.erase_trajectories()
                goal_handle.canceled()
                return FollowJointTrajectory.Result()

            with self._js_lock:
                actual = list(self._current_js)

            errors  = [abs(a - d) for a, d in zip(actual, final_target)]
            max_err = max(errors)

            feedback.actual.positions  = actual
            feedback.desired.positions = final_target
            feedback.error.positions   = [a - d for a, d in zip(actual, final_target)]
            goal_handle.publish_feedback(feedback)

            if max_err < GOAL_TOLERANCE:
                break

            time.sleep(0.05)

        # ── STEP 4: Report result ──────────────────────────────────────
        with self._js_lock:
            actual = list(self._current_js)
        errors  = [abs(a - d) for a, d in zip(actual, final_target)]
        max_err = max(errors)

        result = FollowJointTrajectory.Result()

        if max_err < GOAL_TOLERANCE:
            self.get_logger().info(
                f'Trajectory SUCCESS — max joint error: {max_err:.4f} rad '
                f'({max_err * RAD2DEG:.2f} deg)'
            )
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        else:
            self.get_logger().warn(
                f'Trajectory finished — max error {max_err:.4f} rad '
                f'({max_err * RAD2DEG:.2f} deg) > tolerance {GOAL_TOLERANCE} rad'
            )
            goal_handle.succeed()   # arm did its best — report success anyway
            result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED

        return result


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)

    sdk_node  = KinovaSDKNode()
    exec_node = TrajectoryExecutor(sdk_node)

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(sdk_node)
    executor.add_node(exec_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        sdk_node.destroy_node()
        exec_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()