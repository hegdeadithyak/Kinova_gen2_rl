#!/usr/bin/env python3
"""
kinova_sdk_node.py
──────────────────
Standalone ROS2 node: owns the Kinova USB SDK for the j2s6s200
(6-DOF, 2-finger) and exposes robot state to the rest of the stack.

Publishers
  /joint_states           sensor_msgs/JointState    100 Hz — 10 joints
  /kinova/cartesian_pose  geometry_msgs/PoseStamped 100 Hz

Services
  /kinova/move_home       std_srvs/Trigger

Angle convention — CRITICAL
  ──────────────────────────
  The URDF joint limits for joints 2, 3, 5 are entirely in the positive
  range (e.g. joint 5: 30°→330° ≈ 0.52→5.76 rad). This means the URDF
  is designed for raw SDK angles converted directly to radians — exactly
  what the official Kinova ROS driver (kinova_arm.cpp) does:

      joint_state.position[i] = sdk_degrees * π / 180

  No direction flipping. No 2-complement unwrapping. No π offsets.
  The joint origin RPY values in the URDF already encode the physical
  axis directions.

  For the continuous joints (1, 4, 6) both representations are
  equivalent since MoveIt2 handles full-rotation joints correctly
  regardless of the sign.

  Write path (URDF radians → SDK degrees):
      sdk_deg = degrees(rad) % 360

Joint model  (j2s6s200 = 200 = 2-finger)
  ──────────────────────────────────────
  6 arm joints
  2 finger proximal
  2 finger tip (underactuated — mirrored at 50% of proximal)
  ─────────────
  10 joints total
"""

import ctypes
import math
import os
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from moveit_msgs.msg import PlanningScene, RobotState
from moveit_msgs.srv import ApplyPlanningScene


# ── SDK library ───────────────────────────────────────────────────────
SDK_PATH = os.path.expanduser(
    '~/kinova_ws/rl_v2-master/src/kinova-ros2/'
    'kinova_driver/lib/x86_64-linux-gnu/USBCommandLayerUbuntu.so'
)

# ── Angle constants ───────────────────────────────────────────────────
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# ── Joint names ───────────────────────────────────────────────────────
ARM_JOINTS = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]
FINGER_JOINTS = [
    'j2s6s200_joint_finger_1',
    'j2s6s200_joint_finger_2',
]
FINGER_TIP_JOINTS = [
    'j2s6s200_joint_finger_tip_1',
    'j2s6s200_joint_finger_tip_2',
]
ALL_JOINTS = ARM_JOINTS + FINGER_JOINTS + FINGER_TIP_JOINTS   # 10 total

# ── SDK constants ─────────────────────────────────────────────────────
ANGULAR_POSITION = 2
FINGER_SCALE     = 1.33 / 6800.0   # SDK counts [0, 6800] → radians [0, 1.33]


# ─────────────────────────────────────────────────────────────────────
# Angle conversion helpers
# ─────────────────────────────────────────────────────────────────────

def sdk_deg_to_urdf_rad(deg: float) -> float:
    """
    SDK degrees [0, 360) → URDF radians.

    The URDF joint limits for the j2s6s200 are expressed in the raw SDK
    angle space (e.g. joint 5: 0.52–5.76 rad = 30°–330°). A straight
    degree-to-radian conversion is therefore all that is needed — no
    sign flipping, no 2-complement unwrapping.

    This matches the official Kinova ROS driver (kinova_arm.cpp):
        joint_state.position[i] = kinova_angles.jointN * M_PI / 180
    """
    return math.radians(float(deg))


def urdf_rad_to_sdk_deg(rad: float) -> float:
    """
    URDF radians → SDK degrees [0, 360).

    % 360.0 maps any value back into [0, 360) as the SDK expects.
    """
    return math.degrees(float(rad)) % 360.0


# ─────────────────────────────────────────────────────────────────────
# ctypes structs — must match USBCommandLayerUbuntu.so ABI exactly
# ─────────────────────────────────────────────────────────────────────

class AngularInfo(ctypes.Structure):
    _fields_ = [(f'Actuator{i}', ctypes.c_float) for i in range(1, 8)]


class FingersPosition(ctypes.Structure):
    _fields_ = [
        ('Finger1', ctypes.c_float),
        ('Finger2', ctypes.c_float),
        ('Finger3', ctypes.c_float),
    ]


class CartesianInfo(ctypes.Structure):
    _fields_ = [
        ('X',      ctypes.c_float), ('Y',      ctypes.c_float),
        ('Z',      ctypes.c_float), ('ThetaX', ctypes.c_float),
        ('ThetaY', ctypes.c_float), ('ThetaZ', ctypes.c_float),
    ]


class AngularPosition(ctypes.Structure):
    _fields_ = [('Actuators', AngularInfo), ('Fingers', FingersPosition)]


class CartesianPosition(ctypes.Structure):
    _fields_ = [('Coordinates', CartesianInfo), ('Fingers', FingersPosition)]


class UserPosition(ctypes.Structure):
    _fields_ = [
        ('Type',              ctypes.c_int),
        ('Delay',             ctypes.c_float),
        ('CartesianPosition', CartesianInfo),
        ('Actuators',         AngularInfo),
        ('HandMode',          ctypes.c_int),
        ('Fingers',           FingersPosition),
    ]


class Limitation(ctypes.Structure):
    _fields_ = [(f, ctypes.c_float) for f in (
        'speedParameter1', 'speedParameter2', 'speedParameter3',
        'forceParameter1', 'forceParameter2', 'forceParameter3',
        'accelerationParameter1', 'accelerationParameter2', 'accelerationParameter3',
    )]


class TrajectoryPoint(ctypes.Structure):
    _fields_ = [
        ('Position',          UserPosition),
        ('LimitationsActive', ctypes.c_int),
        ('SynchroType',       ctypes.c_int),
        ('Limitations',       Limitation),
    ]


# ─────────────────────────────────────────────────────────────────────
# KinovaSDKNode
# ─────────────────────────────────────────────────────────────────────

class KinovaSDKNode(Node):

    def __init__(self):
        super().__init__('kinova_sdk_node')

        # Single SDK serialisation thread — libusb is not thread-safe
        self._sdk_q = queue.Queue()
        threading.Thread(target=self._sdk_worker, daemon=True).start()

        # Load and initialise SDK
        self._lib = ctypes.CDLL(SDK_PATH)
        self._declare_signatures()

        ret = self._call(self._lib.InitAPI)
        if ret != 1:
            raise RuntimeError(
                f'InitAPI returned {ret}. '
                'Check USB connection and LD_LIBRARY_PATH.'
            )
        self.get_logger().info('Kinova SDK initialised')
        time.sleep(1.5)   # allow firmware to settle after init
        self._call(self._lib.SetAngularControl)

        # Publishers
        self._js_pub   = self.create_publisher(JointState,  '/joint_states',          10)
        self._pose_pub = self.create_publisher(PoseStamped, '/kinova/cartesian_pose', 10)

        # Services
        self.create_service(Trigger, '/kinova/move_home', self._handle_move_home)

        # MoveIt2 planning scene client — polls every 2 s, cancels on success
        self._scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene'
        )
        self._scene_timer = self.create_timer(2.0, self._try_push_planning_scene)

        # 100 Hz state loop
        self.create_timer(0.01, self._publish_state)

        self.get_logger().info(
            f'Publishing {len(ALL_JOINTS)} joints on /joint_states @ 100 Hz'
        )

    # ── SDK serialisation ──────────────────────────────────────────────

    def _sdk_worker(self):
        """Drain the SDK queue on this dedicated thread. Never call SDK elsewhere."""
        while True:
            fn, args, box = self._sdk_q.get()
            try:
                box.append(fn(*args))
            except Exception as exc:
                self.get_logger().error(f'SDK call {fn.__name__} raised: {exc}')
                box.append(None)
            finally:
                self._sdk_q.task_done()

    def _call(self, fn, *args):
        """Submit one SDK call and block until the worker thread completes it."""
        box = []
        self._sdk_q.put((fn, args, box))
        self._sdk_q.join()
        return box[0] if box else None

    def _declare_signatures(self):
        lib = self._lib
        lib.InitAPI.restype                = ctypes.c_int
        lib.CloseAPI.restype               = ctypes.c_int
        lib.MoveHome.restype               = ctypes.c_int
        lib.SetAngularControl.restype      = ctypes.c_int
        lib.EraseAllTrajectories.restype   = ctypes.c_int
        lib.GetAngularPosition.restype     = ctypes.c_int
        lib.GetAngularPosition.argtypes    = [ctypes.POINTER(AngularPosition)]
        lib.GetCartesianPosition.restype   = ctypes.c_int
        lib.GetCartesianPosition.argtypes  = [ctypes.POINTER(CartesianPosition)]
        lib.SendBasicTrajectory.restype    = ctypes.c_int
        lib.SendBasicTrajectory.argtypes   = [TrajectoryPoint]
        lib.SendAdvanceTrajectory.restype  = ctypes.c_int
        lib.SendAdvanceTrajectory.argtypes = [TrajectoryPoint]

    # ── Read helpers ───────────────────────────────────────────────────

    def _arm_rad(self, ang: AngularPosition) -> list:
        """6 arm joint positions: raw SDK degrees → radians."""
        return [
            sdk_deg_to_urdf_rad(ang.Actuators.Actuator1),
            sdk_deg_to_urdf_rad(ang.Actuators.Actuator2),
            sdk_deg_to_urdf_rad(ang.Actuators.Actuator3),
            sdk_deg_to_urdf_rad(ang.Actuators.Actuator4),
            sdk_deg_to_urdf_rad(ang.Actuators.Actuator5),
            sdk_deg_to_urdf_rad(ang.Actuators.Actuator6),
        ]

    def _finger_rad(self, ang: AngularPosition) -> list:
        """2 proximal finger joints: SDK counts → radians."""
        return [
            float(ang.Fingers.Finger1) * FINGER_SCALE,
            float(ang.Fingers.Finger2) * FINGER_SCALE,
        ]

    def _all_positions(self, ang: AngularPosition) -> list:
        """All 10 joint positions in URDF radians."""
        arm    = self._arm_rad(ang)
        finger = self._finger_rad(ang)
        tip    = [p * 0.5 for p in finger]   # tips: underactuated at 50%
        return arm + finger + tip

    # ── 100 Hz state publisher ─────────────────────────────────────────

    def _publish_state(self):
        ang  = AngularPosition()
        cart = CartesianPosition()
        self._call(self._lib.GetAngularPosition,   ctypes.byref(ang))
        self._call(self._lib.GetCartesianPosition, ctypes.byref(cart))

        now = self.get_clock().now().to_msg()

        js              = JointState()
        js.header.stamp = now
        js.name         = ALL_JOINTS
        js.position     = self._all_positions(ang)
        js.velocity     = [0.0] * len(ALL_JOINTS)
        js.effort       = [0.0] * len(ALL_JOINTS)
        self._js_pub.publish(js)

        ps                 = PoseStamped()
        ps.header.stamp    = now
        ps.header.frame_id = 'world'
        ps.pose.position.x = float(cart.Coordinates.X)
        ps.pose.position.y = float(cart.Coordinates.Y)
        ps.pose.position.z = float(cart.Coordinates.Z)
        self._pose_pub.publish(ps)

    # ── MoveIt2 planning scene sync ────────────────────────────────────

    def _try_push_planning_scene(self):
        """
        Push the real arm pose to MoveIt2's planning scene.
        Polls every 2 s; self-cancels on the first confirmed success.
        """
        if not self._scene_client.service_is_ready():
            return   # move_group not up yet — retry on next tick

        ang = AngularPosition()
        self._call(self._lib.GetAngularPosition, ctypes.byref(ang))
        all_pos = self._all_positions(ang)
        arm     = all_pos[:6]

        self.get_logger().info(
            'Pushing arm state to MoveIt2 planning scene:\n'
            f'  J1={arm[0]:+.3f}  J2={arm[1]:+.3f}  J3={arm[2]:+.3f}\n'
            f'  J4={arm[3]:+.3f}  J5={arm[4]:+.3f}  J6={arm[5]:+.3f}'
        )

        rs                          = RobotState()
        rs.joint_state.header.stamp = self.get_clock().now().to_msg()
        rs.joint_state.name         = ALL_JOINTS
        rs.joint_state.position     = all_pos

        scene             = PlanningScene()
        scene.is_diff     = True
        scene.robot_state = rs

        req       = ApplyPlanningScene.Request()
        req.scene = scene

        future = self._scene_client.call_async(req)
        future.add_done_callback(self._on_scene_push_done)

    def _on_scene_push_done(self, future):
        """Cancel the timer only on confirmed success so failures auto-retry."""
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(
                f'ApplyPlanningScene raised an exception: {exc} — retrying'
            )
            return

        if result.success:
            self._scene_timer.cancel()
            self.get_logger().info(
                'MoveIt2 planning scene synced — RViz2 shows real arm pose'
            )
        else:
            self.get_logger().warn(
                'ApplyPlanningScene returned failure — retrying in 2 s'
            )

    # ── Public API (used by trajectory_executor and other nodes) ───────

    def get_joint_positions_rad(self) -> list:
        """Current 6 arm joint positions in URDF radians."""
        ang = AngularPosition()
        self._call(self._lib.GetAngularPosition, ctypes.byref(ang))
        return self._arm_rad(ang)

    def send_joint_position(self, positions_rad: list):
        """
        Speed-limited one-shot angular move via SendBasicTrajectory.
        Blocks until the SDK returns (not until the arm arrives).
        Use for homing and single-point supervised moves only.
        """
        pt = self._build_point(positions_rad, speed_limited=True)
        self._call(self._lib.SendBasicTrajectory, pt)

    def send_advance_point(self, positions_rad: list):
        """
        Enqueue one waypoint in the Kinova DSP FIFO.
        Returns immediately. DSP smoothly interpolates between queued points.
        Never mix with SendBasicTrajectory during trajectory execution.
        """
        pt = self._build_point(positions_rad, speed_limited=False)
        self._call(self._lib.SendAdvanceTrajectory, pt)

    def erase_trajectories(self):
        """Flush the DSP FIFO. Always call before starting a new trajectory."""
        self._call(self._lib.EraseAllTrajectories)

    def _build_point(self, positions_rad: list, speed_limited: bool) -> TrajectoryPoint:
        pt = TrajectoryPoint()
        ctypes.memset(ctypes.byref(pt), 0, ctypes.sizeof(pt))
        pt.Position.Type = ANGULAR_POSITION

        if speed_limited:
            pt.LimitationsActive           = 1
            pt.Limitations.speedParameter1 = 20.0   # deg/s, joints 1–3
            pt.Limitations.speedParameter2 = 20.0   # deg/s, joints 4–6
        else:
            pt.LimitationsActive = 0
            pt.SynchroType       = 0

        pt.Position.Actuators.Actuator1 = urdf_rad_to_sdk_deg(positions_rad[0])
        pt.Position.Actuators.Actuator2 = urdf_rad_to_sdk_deg(positions_rad[1])
        pt.Position.Actuators.Actuator3 = urdf_rad_to_sdk_deg(positions_rad[2])
        pt.Position.Actuators.Actuator4 = urdf_rad_to_sdk_deg(positions_rad[3])
        pt.Position.Actuators.Actuator5 = urdf_rad_to_sdk_deg(positions_rad[4])
        pt.Position.Actuators.Actuator6 = urdf_rad_to_sdk_deg(positions_rad[5])
        return pt

    # ── Service handlers ───────────────────────────────────────────────

    def _handle_move_home(self, _req, response):
        self._call(self._lib.EraseAllTrajectories)
        self._call(self._lib.MoveHome)
        # Re-arm scene sync so RViz2 reflects the home pose
        self._scene_timer = self.create_timer(2.0, self._try_push_planning_scene)
        response.success = True
        response.message = 'Moving to home position'
        return response

    def destroy_node(self):
        self._call(self._lib.CloseAPI)
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

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