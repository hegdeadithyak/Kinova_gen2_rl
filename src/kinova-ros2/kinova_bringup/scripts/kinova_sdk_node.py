#!/usr/bin/env python3
import ctypes
import math
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
    '/home/amma/kinova_ws/rl_v2-master/src/kinova-ros2/'
    'kinova_driver/lib/x86_64-linux-gnu/USBCommandLayerUbuntu.so'
)

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# Flip axes where SDK and URDF disagree; index 0..5 → joints 1..6.
JOINT_DIRECTION = [1, 1, 1, 1, 1, 1]

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
ALL_JOINTS = ARM_JOINTS + FINGER_JOINTS + FINGER_TIP_JOINTS

ANGULAR_POSITION = 2
FINGER_SCALE     = 1.33 / 6800.0  # SDK range 0–6800 → 0–1.33 rad


def _sdk_deg_to_urdf_rad(deg: float, joint_idx: int) -> float:
    rad = math.radians(deg * JOINT_DIRECTION[joint_idx])
    return math.atan2(math.sin(rad), math.cos(rad))


def _urdf_rad_to_sdk_deg(rad: float, joint_idx: int) -> float:
    return (math.degrees(rad) * JOINT_DIRECTION[joint_idx]) % 360.0


class AngularInfo(ctypes.Structure):
    _fields_ = [(f'Actuator{i}', ctypes.c_float) for i in range(1, 8)]

class FingersPosition(ctypes.Structure):
    _fields_ = [('Finger1', ctypes.c_float),
                ('Finger2', ctypes.c_float),
                ('Finger3', ctypes.c_float)]

class CartesianInfo(ctypes.Structure):
    _fields_ = [('X', ctypes.c_float), ('Y', ctypes.c_float), ('Z', ctypes.c_float),
                ('ThetaX', ctypes.c_float), ('ThetaY', ctypes.c_float), ('ThetaZ', ctypes.c_float)]

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


def _raw_deg(ang: AngularPosition) -> list:
    a = ang.Actuators
    return [a.Actuator1, a.Actuator2, a.Actuator3, a.Actuator4, a.Actuator5, a.Actuator6]


class KinovaSDKNode(Node):

    def __init__(self):
        super().__init__('kinova_sdk_node')

        self._sdk_queue  = queue.Queue()
        self._sdk_thread = threading.Thread(target=self._sdk_worker, daemon=True)
        self._sdk_thread.start()

        self._sdk = ctypes.CDLL(SDK_PATH)
        self._declare_sdk_signatures()

        ret = self._sdk_call(self._sdk.InitAPI)
        if ret != 1:
            self.get_logger().error(
                f'InitAPI returned {ret} — check LD_LIBRARY_PATH and USB connection')
            raise RuntimeError('SDK InitAPI failed')

        self.get_logger().info('Kinova SDK initialised successfully')
        time.sleep(1.5)
        self._sdk_call(self._sdk.SetAngularControl)

        self._js_pub   = self.create_publisher(JointState,  '/joint_states',          10)
        self._pose_pub = self.create_publisher(PoseStamped, '/kinova/cartesian_pose', 10)

        self.create_service(Trigger, '/kinova/move_home', self._handle_home)

        self._scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self._initial_state_pushed = False

        self.create_timer(0.01,  self._publish_state)
        self.create_timer(12.0,  self._push_initial_state_once)  # t=12s: move_group ready by then

        self.get_logger().info(f'Publishing {len(ALL_JOINTS)} joints on /joint_states at 100 Hz')

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

    def _read_arm_pos_rad(self, ang: AngularPosition) -> list:
        return [_sdk_deg_to_urdf_rad(d, i) for i, d in enumerate(_raw_deg(ang))]

    def _read_all_positions(self, ang: AngularPosition) -> list:
        arm     = self._read_arm_pos_rad(ang)
        fingers = [float(ang.Fingers.Finger1) * FINGER_SCALE,
                   float(ang.Fingers.Finger2) * FINGER_SCALE,
                   float(ang.Fingers.Finger3) * FINGER_SCALE]
        tips    = [p * 0.5 for p in fingers]  # not directly readable; mirrored at 50%
        return arm + fingers + tips

    def _publish_state(self):
        ang  = AngularPosition()
        cart = CartesianPosition()
        self._sdk_call(self._sdk.GetAngularPosition,   ctypes.byref(ang))
        self._sdk_call(self._sdk.GetCartesianPosition, ctypes.byref(cart))

        now = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now
        js.name         = ALL_JOINTS
        js.position     = self._read_all_positions(ang)
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

    def _push_initial_state_once(self):
        if self._initial_state_pushed:
            return
        if not self._scene_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('ApplyPlanningScene not ready — will retry next timer tick')
            return

        ang = AngularPosition()
        self._sdk_call(self._sdk.GetAngularPosition, ctypes.byref(ang))
        all_pos = self._read_all_positions(ang)

        arm_pos = all_pos[:6]
        self.get_logger().info(
            f'Pushing real arm state to MoveIt2 planning scene:\n'
            f'  J1:{arm_pos[0]:.4f}  J2:{arm_pos[1]:.4f}  J3:{arm_pos[2]:.4f}\n'
            f'  J4:{arm_pos[3]:.4f}  J5:{arm_pos[4]:.4f}  J6:{arm_pos[5]:.4f}')

        robot_state = RobotState()
        robot_state.joint_state.name          = ALL_JOINTS
        robot_state.joint_state.position      = all_pos
        robot_state.joint_state.header.stamp  = self.get_clock().now().to_msg()

        scene             = PlanningScene()
        scene.is_diff     = True
        scene.robot_state = robot_state

        req       = ApplyPlanningScene.Request()
        req.scene = scene
        self._scene_client.call_async(req).add_done_callback(self._planning_scene_callback)
        self._initial_state_pushed = True

    def _planning_scene_callback(self, future):
        try:
            if future.result().success:
                self.get_logger().info('Real arm state applied to MoveIt2 planning scene.')
            else:
                self.get_logger().error('ApplyPlanningScene returned failure.')
        except Exception as e:
            self.get_logger().error(f'ApplyPlanningScene exception: {e}')

    def send_joint_position(self, positions_rad: list):
        point = TrajectoryPoint()
        ctypes.memset(ctypes.byref(point), 0, ctypes.sizeof(point))
        point.Position.Type               = ANGULAR_POSITION
        point.LimitationsActive           = 1
        point.Limitations.speedParameter1 = 20.0
        point.Limitations.speedParameter2 = 20.0
        sdk_deg = [_urdf_rad_to_sdk_deg(positions_rad[i], i) for i in range(6)]
        point.Position.Actuators.Actuator1 = sdk_deg[0]
        point.Position.Actuators.Actuator2 = sdk_deg[1]
        point.Position.Actuators.Actuator3 = sdk_deg[2]
        point.Position.Actuators.Actuator4 = sdk_deg[3]
        point.Position.Actuators.Actuator5 = sdk_deg[4]
        point.Position.Actuators.Actuator6 = sdk_deg[5]
        self._sdk_call(self._sdk.SendBasicTrajectory, point)

    def send_advance_point(self, positions_rad: list):
        point = TrajectoryPoint()
        ctypes.memset(ctypes.byref(point), 0, ctypes.sizeof(point))
        point.Position.Type     = ANGULAR_POSITION
        point.Position.Delay    = 0.0
        point.Position.HandMode = 0
        point.LimitationsActive = 0
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
        ang = AngularPosition()
        self._sdk_call(self._sdk.GetAngularPosition, ctypes.byref(ang))
        return self._read_arm_pos_rad(ang)

    def erase_trajectories(self):
        self._sdk_call(self._sdk.EraseAllTrajectories)

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
