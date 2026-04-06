#!/usr/bin/env python3
"""
kinova_sdk_node.py
──────────────────
ROS2 node that owns the Kinova USB SDK and:
  - Publishes /joint_states at 100Hz (all 12 joints — 6 arm + 3 finger + 3 tip)
  - Publishes /kinova/cartesian_pose at 100Hz
  - Explicitly pushes real arm state into MoveIt2 planning scene on startup
  - Serves /kinova/move_home (Trigger)
  - Exposes send_joint_position() for trajectory_executor

All SDK calls are serialised through a single thread to prevent libusb crashes.
"""

import ctypes
import os
import queue
import time
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from moveit_msgs.msg import PlanningScene, RobotState
from moveit_msgs.srv import ApplyPlanningScene

SDK_PATH = os.path.expanduser(
    '~/Downloads/rl_v2-master-master/src/kinova-ros2/'
    'kinova_driver/lib/x86_64-linux-gnu/USBCommandLayerUbuntu.so'
)

DEG2RAD = 3.14159265358979 / 180.0
RAD2DEG = 180.0 / 3.14159265358979

# ── All joints MoveIt2 expects ────────────────────────────────────────
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

ANGULAR_POSITION = 2


# ── ctypes structs ────────────────────────────────────────────────────
class AngularInfo(ctypes.Structure):
    _fields_ = [(f'Actuator{i}', ctypes.c_float) for i in range(1, 8)]

class FingersPosition(ctypes.Structure):
    _fields_ = [('Finger1', ctypes.c_float),
                ('Finger2', ctypes.c_float),
                ('Finger3', ctypes.c_float)]

class CartesianInfo(ctypes.Structure):
    _fields_ = [('X',      ctypes.c_float), ('Y',     ctypes.c_float),
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
        'speedParameter1','speedParameter2','speedParameter3',
        'forceParameter1','forceParameter2','forceParameter3',
        'accelerationParameter1','accelerationParameter2','accelerationParameter3']]

class TrajectoryPoint(ctypes.Structure):
    _fields_ = [('Position',          UserPosition),
                ('LimitationsActive', ctypes.c_int),
                ('SynchroType',       ctypes.c_int),
                ('Limitations',       Limitation)]


class KinovaSDKNode(Node):

    def __init__(self):
        super().__init__('kinova_sdk_node')

        # ── Single SDK thread — prevents libusb mutex crashes ─────────
        self._sdk_queue  = queue.Queue()
        self._sdk_thread = threading.Thread(target=self._sdk_worker, daemon=True)
        self._sdk_thread.start()

        # ── Load SDK ──────────────────────────────────────────────────
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

        # ── Publishers ────────────────────────────────────────────────
        self._js_pub   = self.create_publisher(JointState,   '/joint_states',           10)
        self._pose_pub = self.create_publisher(PoseStamped,  '/kinova/cartesian_pose',  10)

        # ── Services ──────────────────────────────────────────────────
        self.create_service(Trigger, '/kinova/move_home', self._handle_home)

        # ── MoveIt2 planning scene client ─────────────────────────────
        self._scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene'
        )
        self._initial_state_pushed = False

        # ── Timers ────────────────────────────────────────────────────
        # Publish joint states at 100Hz — keeps MoveIt2 in continuous sync
        self.create_timer(0.01, self._publish_state)

        # Push real state into MoveIt2 planning scene once at t=12s
        # (move_group starts at t=8s so it has 4s to be ready)
        self.create_timer(12.0, self._push_initial_state_once)

        self.get_logger().info(
            f'Publishing {len(ALL_JOINTS)} joints on /joint_states at 100Hz'
        )

    # ── SDK serialisation ─────────────────────────────────────────────
    def _sdk_worker(self):
        """All SDK ctypes calls run on this single thread — no race conditions."""
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
        """Submit a call to the SDK thread and block until complete."""
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

    # ── Joint state publisher (100Hz) ─────────────────────────────────
    def _publish_state(self):
        ang  = AngularPosition()
        cart = CartesianPosition()
        self._sdk_call(self._sdk.GetAngularPosition,  ctypes.byref(ang))
        self._sdk_call(self._sdk.GetCartesianPosition, ctypes.byref(cart))

        now = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now
        js.name = ALL_JOINTS   # all 12 joints

        # ── 6 arm joints (degrees → radians) ─────────────────────────
        arm_pos = [
            ang.Actuators.Actuator1 * DEG2RAD,
            ang.Actuators.Actuator2 * DEG2RAD,
            ang.Actuators.Actuator3 * DEG2RAD,
            ang.Actuators.Actuator4 * DEG2RAD,
            ang.Actuators.Actuator5 * DEG2RAD,
            ang.Actuators.Actuator6 * DEG2RAD,
        ]

        # ── 3 finger joints (SDK returns 0–6800 range → convert to rad) 
        # Kinova finger range: 0 (open) to ~6800 (closed), maps to 0–1.33 rad
        FINGER_SCALE = 1.33 / 6800.0
        finger_pos = [
            float(ang.Fingers.Finger1) * FINGER_SCALE,
            float(ang.Fingers.Finger2) * FINGER_SCALE,
            float(ang.Fingers.Finger3) * FINGER_SCALE,
        ]

        # ── 3 finger tip joints — not directly readable, mirror fingers ─
        finger_tip_pos = [p * 0.5 for p in finger_pos]

        js.position = arm_pos + finger_pos + finger_tip_pos   # 12 values
        js.velocity = [0.0] * 12
        js.effort   = [0.0] * 12
        self._js_pub.publish(js)

        # ── Cartesian pose ────────────────────────────────────────────
        ps = PoseStamped()
        ps.header.stamp    = now
        ps.header.frame_id = 'world'
        ps.pose.position.x = float(cart.Coordinates.X)
        ps.pose.position.y = float(cart.Coordinates.Y)
        ps.pose.position.z = float(cart.Coordinates.Z)
        self._pose_pub.publish(ps)

    # ── MoveIt2 planning scene sync ───────────────────────────────────
    def _push_initial_state_once(self):
        """
        Explicitly sets MoveIt2's planning scene to the real arm position.
        Called once at t=12s. Without this, MoveIt2 may briefly show the
        robot at URDF default pose until CurrentStateMonitor catches up.
        """
        if self._initial_state_pushed:
            return

        if not self._scene_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                'ApplyPlanningScene service not ready — will retry next tick'
            )
            return

        # Read real positions directly from hardware
        ang = AngularPosition()
        self._sdk_call(self._sdk.GetAngularPosition, ctypes.byref(ang))

        arm_pos = [
            ang.Actuators.Actuator1 * DEG2RAD,
            ang.Actuators.Actuator2 * DEG2RAD,
            ang.Actuators.Actuator3 * DEG2RAD,
            ang.Actuators.Actuator4 * DEG2RAD,
            ang.Actuators.Actuator5 * DEG2RAD,
            ang.Actuators.Actuator6 * DEG2RAD,
        ]
        FINGER_SCALE = 1.33 / 6800.0
        finger_pos = [
            float(ang.Fingers.Finger1) * FINGER_SCALE,
            float(ang.Fingers.Finger2) * FINGER_SCALE,
            float(ang.Fingers.Finger3) * FINGER_SCALE,
        ]
        finger_tip_pos = [p * 0.5 for p in finger_pos]
        all_pos = arm_pos + finger_pos + finger_tip_pos

        self.get_logger().info(
            f'Pushing real arm state to MoveIt2 planning scene:\n'
            f'  J1:{arm_pos[0]:.4f}  J2:{arm_pos[1]:.4f}  J3:{arm_pos[2]:.4f}\n'
            f'  J4:{arm_pos[3]:.4f}  J5:{arm_pos[4]:.4f}  J6:{arm_pos[5]:.4f}'
        )

        # Build RobotState with ALL joints
        robot_state = RobotState()
        robot_state.joint_state.name     = ALL_JOINTS
        robot_state.joint_state.position = all_pos
        robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()

        # Apply as a diff to the existing planning scene
        scene = PlanningScene()
        scene.is_diff      = True
        scene.robot_state  = robot_state

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

    # ── Public API for trajectory_executor ───────────────────────────
    def send_joint_position(self, positions_rad: list):
        """Send absolute joint positions in radians to hardware."""
        point = TrajectoryPoint()
        ctypes.memset(ctypes.byref(point), 0, ctypes.sizeof(point))
        point.Position.Type                = ANGULAR_POSITION
        point.LimitationsActive            = 1
        point.Limitations.speedParameter1  = 20.0   # deg/s joints 1–3
        point.Limitations.speedParameter2  = 20.0   # deg/s joints 4–6

        acts = [p * RAD2DEG for p in positions_rad]
        point.Position.Actuators.Actuator1 = acts[0]
        point.Position.Actuators.Actuator2 = acts[1]
        point.Position.Actuators.Actuator3 = acts[2]
        point.Position.Actuators.Actuator4 = acts[3]
        point.Position.Actuators.Actuator5 = acts[4]
        point.Position.Actuators.Actuator6 = acts[5]

        self._sdk_call(self._sdk.SendBasicTrajectory, point)

    def get_joint_positions_rad(self) -> list:
        """Return current arm joint positions in radians (6 joints)."""
        ang = AngularPosition()
        self._sdk_call(self._sdk.GetAngularPosition, ctypes.byref(ang))
        return [
            ang.Actuators.Actuator1 * DEG2RAD,
            ang.Actuators.Actuator2 * DEG2RAD,
            ang.Actuators.Actuator3 * DEG2RAD,
            ang.Actuators.Actuator4 * DEG2RAD,
            ang.Actuators.Actuator5 * DEG2RAD,
            ang.Actuators.Actuator6 * DEG2RAD,
        ]

    # ── Service handlers ──────────────────────────────────────────────
    def _handle_home(self, request, response):
        self._sdk_call(self._sdk.EraseAllTrajectories)
        self._sdk_call(self._sdk.MoveHome)
        self._initial_state_pushed = False   # re-sync after homing
        response.success = True
        response.message = 'Moving to home position'
        return response

    def destroy_node(self):
        self._sdk_call(self._sdk.CloseAPI)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KinovaSDKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()