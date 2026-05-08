#!/usr/bin/env python3
from __future__ import annotations

import copy
import time
from threading import Lock

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Pose
from moveit_msgs.action import MoveGroup as MoveGroupAction
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    PositionIKRequest,
    RobotState,
    WorkspaceParameters,
)
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401

PLANNING_GROUP     = "arm"
EE_LINK            = "j2s6s200_end_effector"
BASE_FRAME         = "root"
CAMERA_FRAME       = "camera_color_optical_frame"

RRTSTAR_PLANNING_TIME  = 8.0
RRTSTAR_ATTEMPTS       = 3
FALLBACK_PLANNER       = "RRTConnectkConfigDefault"
FALLBACK_PLANNING_TIME = 5.0

POSITION_TOL    = 0.01
ORIENTATION_TOL = 0.05
APPROACH_STANDOFF = 0.08

JOINT_NAMES = [
    "j2s6s200_joint_1", "j2s6s200_joint_2", "j2s6s200_joint_3",
    "j2s6s200_joint_4", "j2s6s200_joint_5", "j2s6s200_joint_6",
]


class DemoFeedPlanner(Node):

    def __init__(self):
        super().__init__("demo_feed_planner")

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._mouth_lock  = Lock()
        self._mouth_point: PointStamped | None = None
        self.create_subscription(PointStamped, "/mouth_3d_point", self._mouth_cb, 10)

        self._js_lock = Lock()
        self._joint_state: JointState | None = None
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)

        self._recorded_pose:   Pose | None       = None
        self._recorded_joints: list[float] | None = None

        self._ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self._mg_client: ActionClient = ActionClient(self, MoveGroupAction, "/move_group")

        self.create_service(Trigger, "/record_feed_pose", self._record_cb)
        self.create_service(Trigger, "/feed_trigger",     self._feed_cb)

        self.get_logger().info("DemoFeedPlanner ready.")

    def _mouth_cb(self, msg: PointStamped):
        with self._mouth_lock:
            self._mouth_point = msg

    def _js_cb(self, msg: JointState):
        with self._js_lock:
            self._joint_state = msg

    def _record_cb(self, _req, res):
        js = self._current_joints()
        if js is None:
            res.success = False
            res.message = "No joint state received yet."
            return res

        try:
            t = self.tf_buffer.lookup_transform(
                BASE_FRAME, EE_LINK, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
        except Exception as e:
            res.success = False
            res.message = f"TF lookup failed: {e}"
            return res

        pose             = Pose()
        pose.position.x  = t.transform.translation.x
        pose.position.y  = t.transform.translation.y
        pose.position.z  = t.transform.translation.z
        pose.orientation = t.transform.rotation

        self._recorded_pose   = pose
        self._recorded_joints = js
        self.get_logger().info(
            f"Recorded feed pose: ({pose.position.x:.3f}, "
            f"{pose.position.y:.3f}, {pose.position.z:.3f})")
        res.success = True
        res.message = "Feed pose recorded."
        return res

    def _feed_cb(self, _req, res):
        if self._recorded_pose is None:
            res.success = False
            res.message = "No pose recorded. Call /record_feed_pose first."
            return res

        mouth_base = self._mouth_in_base()
        if mouth_base is None:
            res.success = False
            res.message = "Could not get mouth position in base frame."
            return res

        target_pose = copy.deepcopy(self._recorded_pose)
        target_pose.position.x = mouth_base[0]
        target_pose.position.y = mouth_base[1]
        target_pose.position.z = mouth_base[2]

        approach_pose   = copy.deepcopy(target_pose)
        approach_pose.position.z -= APPROACH_STANDOFF  # back off in base-frame Z; adjust axis if needed

        current_joints = self._current_joints()
        if current_joints is None:
            res.success = False
            res.message = "No joint state available."
            return res

        self.get_logger().info("Moving to approach pose…")
        if not self._plan_and_execute(approach_pose, current_joints, "approach"):
            res.success = False
            res.message = "Approach motion failed."
            return res

        self.get_logger().info("Moving to feed pose…")
        if not self._plan_and_execute(target_pose, self._current_joints(), "feed"):
            res.success = False
            res.message = "Feed motion failed."
            return res

        time.sleep(1.5)

        self.get_logger().info("Retracting…")
        self._plan_and_execute(approach_pose, self._current_joints(), "retract")

        res.success = True
        res.message = "Feed sequence complete."
        return res

    def _plan_and_execute(self, target_pose: Pose, seed_joints: list[float], label: str) -> bool:
        goal_joints = self._ik_seed(target_pose, seed_joints)
        if goal_joints is None:
            self.get_logger().warn(f"[{label}] IK failed.")
            return False

        for planner, planning_time in [
            ("RRTstarkConfigDefault", RRTSTAR_PLANNING_TIME),
            (FALLBACK_PLANNER,        FALLBACK_PLANNING_TIME),
        ]:
            self.get_logger().info(f"[{label}] Planning with {planner} ({planning_time}s)…")
            if self._moveit_joint_goal(goal_joints, planner, planning_time, label):
                return True
            self.get_logger().warn(f"[{label}] {planner} failed, trying next…")
        return False

    def _ik_seed(self, pose: Pose, seed_joints: list[float]) -> list[float] | None:
        if not self._ik_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("IK service not available.")
            return None

        req                                  = GetPositionIK.Request()
        req.ik_request                       = PositionIKRequest()
        req.ik_request.group_name            = PLANNING_GROUP
        req.ik_request.ik_link_name          = EE_LINK
        req.ik_request.avoid_collisions      = True
        req.ik_request.timeout.sec           = 1

        seed_state                           = RobotState()
        seed_state.joint_state.name         = JOINT_NAMES
        seed_state.joint_state.position     = seed_joints
        req.ik_request.robot_state          = seed_state

        req.ik_request.pose_stamped.header.frame_id = BASE_FRAME
        req.ik_request.pose_stamped.pose            = pose

        future = self._ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done():
            self.get_logger().error("IK timed out.")
            return None

        result = future.result()
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(f"IK error code {result.error_code.val}")
            return None

        js = result.solution.joint_state
        positions = []
        for name in JOINT_NAMES:
            if name not in js.name:
                self.get_logger().error(f"Joint {name} missing from IK solution.")
                return None
            positions.append(js.position[js.name.index(name)])
        return positions

    def _moveit_joint_goal(
        self, goal_joints: list[float], planner_id: str, planning_time: float, label: str
    ) -> bool:
        if not self._mg_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup action server not available.")
            return False

        joint_constraints = []
        for name, pos in zip(JOINT_NAMES, goal_joints):
            jc                 = JointConstraint()
            jc.joint_name      = name
            jc.position        = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            joint_constraints.append(jc)

        goal_constraint                   = Constraints()
        goal_constraint.joint_constraints = joint_constraints

        req                                   = MotionPlanRequest()
        req.group_name                        = PLANNING_GROUP
        req.planner_id                        = planner_id
        req.allowed_planning_time             = planning_time
        req.num_planning_attempts             = RRTSTAR_ATTEMPTS if "star" in planner_id.lower() else 1
        req.max_velocity_scaling_factor       = 0.3
        req.max_acceleration_scaling_factor   = 0.2
        req.goal_constraints                  = [goal_constraint]
        req.workspace_parameters             = self._workspace()

        current_js = self._current_joint_state_msg()
        if current_js is not None:
            req.start_state.joint_state = current_js

        goal_msg                                            = MoveGroupAction.Goal()
        goal_msg.request                                    = req
        goal_msg.planning_options.plan_only                 = False
        goal_msg.planning_options.replan                    = False
        goal_msg.planning_options.replan_attempts           = 0
        goal_msg.planning_options.planning_scene_diff.is_diff = True

        future = self._mg_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=planning_time + 15.0)

        if not future.done() or future.result() is None:
            self.get_logger().error(f"[{label}] Goal rejected or timed out.")
            return False

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"[{label}] Goal not accepted.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)

        if not result_future.done():
            self.get_logger().error(f"[{label}] Execution timed out.")
            return False

        result = result_future.result().result
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f"[{label}] Motion complete.")
            return True

        self.get_logger().warn(f"[{label}] MoveGroup error: {result.error_code.val}")
        return False

    def _mouth_in_base(self) -> np.ndarray | None:
        with self._mouth_lock:
            mouth = self._mouth_point
        if mouth is None:
            self.get_logger().warn("No mouth point received yet.")
            return None
        try:
            transformed = self.tf_buffer.transform(
                mouth, BASE_FRAME, timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f"TF mouth transform failed: {e}")
            return None
        p = transformed.point
        return np.array([p.x, p.y, p.z])

    def _current_joints(self) -> list[float] | None:
        with self._js_lock:
            js = self._joint_state
        if js is None:
            return None
        positions = []
        for name in JOINT_NAMES:
            if name not in js.name:
                return None
            positions.append(js.position[js.name.index(name)])
        return positions

    def _current_joint_state_msg(self):
        with self._js_lock:
            return self._joint_state

    @staticmethod
    def _workspace() -> WorkspaceParameters:
        ws = WorkspaceParameters()
        ws.header.frame_id = BASE_FRAME
        ws.min_corner.x = ws.min_corner.y = ws.min_corner.z = -1.5
        ws.max_corner.x = ws.max_corner.y = ws.max_corner.z =  1.5
        return ws


def main(args=None):
    rclpy.init(args=args)
    node = DemoFeedPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
