#!/usr/bin/env python3
"""
mouth_feeding_planner.py
========================
Adapted from your original MouthFeedingPlanner.
Subscribes to /mouth_3d_point (PointStamped published by mouth_tracker_node),
and executes a 3-phase feeding motion on /feed_trigger.

Robot: Kinova j2s6s300 (6DOF, 3-finger)
Arm group: 'arm'  |  EE link: 'j2s6s200_end_effector'  ← adjust if needed

Motion sequence on /feed_trigger:
  Phase 1 · Height   — raise/lower EE to mouth height (keep x,y)
  Phase 2 · Lateral  — move to approach point 10 cm in front of mouth
  Phase 3 · Feed     — advance spoon tip to 5 cm from mouth
  Hold               — wait HOLD_SECONDS at feed position
  Return             — return to saved start joint configuration
"""

import copy
import math
import time
from typing import Optional, List

import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, Quaternion
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_point
from tf2_ros import (Buffer, TransformListener,
                     LookupException, ExtrapolationException, ConnectivityException)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker

# ── Constants ────────────────────────────────────────────────────────
SPOON_LENGTH_M      = 0.05
SAFETY_GAP_M        = 0.05
STOP_BEFORE_MOUTH_M = SPOON_LENGTH_M + SAFETY_GAP_M   # 0.10 m approach waypoint
FEED_GAP_M          = SAFETY_GAP_M                    # 0.05 m spoon tip at mouth
HOLD_SECONDS        = 5.0

BASE_FRAME  = 'j2s6s200_link_base'        # Kinova base TF frame
CAM_FRAME   = 'camera_color_optical_frame'
EE_LINK     = 'j2s6s200_end_effector'       # ← change to j2s6s300_end_effector if needed
ARM_GROUP   = 'arm'

ARM_JOINT_NAMES = [
    'j2s6s200_joint_1',
    'j2s6s200_joint_2',
    'j2s6s200_joint_3',
    'j2s6s200_joint_4',
    'j2s6s200_joint_5',
    'j2s6s200_joint_6',
]

MAX_REACH           = 0.85
MIN_MOVE_EPS_M      = 1e-3
MAX_JOINT_VEL_RAD_S = 0.25
MIN_PHASE_DUR_S     = 2.0
RETURN_DUR_S        = 5.0


class MouthFeedingPlanner(Node):
    def __init__(self):
        super().__init__('mouth_feeding_planner')
        self._cb = ReentrantCallbackGroup()

        self._tf_buf      = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        self._latest_mouth: Optional[PointStamped] = None
        self._latest_js:    Optional[JointState]   = None
        self._busy = False

        # ── Subscribers ──────────────────────────────────────────────
        self.create_subscription(
            PointStamped, '/mouth_3d_point', self._mouth_cb, 10,
            callback_group=self._cb)
        self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10,
            callback_group=self._cb)

        # ── Publishers ───────────────────────────────────────────────
        self._marker_pub = self.create_publisher(Marker, '/feeding_marker', 10)

        # ── IK service ───────────────────────────────────────────────
        self._ik_client = self.create_client(
            GetPositionIK, '/compute_ik',
            callback_group=self._cb)

        # ── Trajectory action ────────────────────────────────────────
        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            callback_group=self._cb)

        # ── Trigger service ──────────────────────────────────────────
        self.create_service(
            Trigger, '/feed_trigger', self._trigger_cb,
            callback_group=self._cb)

        self.get_logger().info('Waiting for IK and trajectory services …')
        if not self._ik_client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError('Timed out waiting for /compute_ik')
        if not self._traj_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError('Timed out waiting for /arm_controller/follow_joint_trajectory')

        self.get_logger().info(
            'MouthFeedingPlanner ready.\n'
            '  Waiting for /mouth_3d_point from mouth_tracker_node …\n'
            '  Trigger with: ros2 service call /feed_trigger std_srvs/srv/Trigger {}')

    # ── Callbacks ────────────────────────────────────────────────────

    def _mouth_cb(self, msg: PointStamped):
        self._latest_mouth = msg
        self._publish_marker_cam(msg)   # diagnostic sphere in RViz

    def _js_cb(self, msg: JointState):
        self._latest_js = msg

    def _trigger_cb(self, _req, response):
        if self._busy:
            response.success = False
            response.message = 'Feed already in progress — please wait.'
            return response
        if self._latest_mouth is None:
            response.success = False
            response.message = (
                'No mouth point yet. '
                'Is mouth_tracker_node running and detecting a face? '
                'Check: ros2 topic echo /mouth_3d_point')
            return response

        self._busy = True
        try:
            ok, msg = self._execute_feed()
            response.success = ok
            response.message = msg
        except Exception as exc:
            response.success = False
            response.message = f'Exception: {exc}'
            self.get_logger().error(response.message)
        finally:
            self._busy = False
        return response

    # ── Core feeding sequence ─────────────────────────────────────────

    def _execute_feed(self):
        snap      = self._latest_mouth
        src_frame = snap.header.frame_id or CAM_FRAME
        stamp     = rclpy.time.Time.from_msg(snap.header.stamp)

        # Transform mouth point → base frame
        try:
            tf_cam = self._lookup_tf(BASE_FRAME, src_frame, stamp=stamp)
        except RuntimeError:
            self.get_logger().warn('Timestamped TF failed — using latest.')
            try:
                tf_cam = self._lookup_tf(BASE_FRAME, src_frame)
            except RuntimeError as e:
                return False, str(e)

        mp_pt  = do_transform_point(snap, tf_cam).point
        mx, my, mz = mp_pt.x, mp_pt.y, mp_pt.z
        self.get_logger().info(
            f'Mouth in [{BASE_FRAME}]: ({mx:+.3f}, {my:+.3f}, {mz:+.3f}) m')
        self._pub_sphere(BASE_FRAME, mx, my, mz,
                         mid=0, rgba=(1.0, 0.4, 0.7, 0.9), d=0.05)

        # Current EE pose
        try:
            tf_ee = self._lookup_tf(BASE_FRAME, EE_LINK)
        except RuntimeError as e:
            return False, str(e)

        ex = tf_ee.transform.translation.x
        ey = tf_ee.transform.translation.y
        ez = tf_ee.transform.translation.z
        cur_ori = tf_ee.transform.rotation
        self.get_logger().info(
            f'EE start in [{BASE_FRAME}]: ({ex:+.3f}, {ey:+.3f}, {ez:+.3f}) m')

        # Save start joints for return
        start_js = copy.deepcopy(self._latest_js)

        # Horizontal approach direction (base x-y plane, pointing toward mouth)
        horiz = math.sqrt(mx*mx + my*my)
        if horiz < MIN_MOVE_EPS_M:
            return False, 'Mouth directly above base origin — cannot compute approach.'
        uax, uay = mx / horiz, my / horiz

        # ── Waypoints ────────────────────────────────────────────────
        # Phase 1: height — keep EE x,y, move to mouth z
        p1x, p1y, p1z = self._clamp(ex, ey, mz)

        # Phase 2: approach — 10 cm in front of mouth, same height
        p2x, p2y, p2z = self._clamp(
            mx - uax * STOP_BEFORE_MOUTH_M,
            my - uay * STOP_BEFORE_MOUTH_M,
            mz)

        # Phase 3: feed — spoon tip 5 cm from mouth
        p3x, p3y, p3z = self._clamp(
            mx - uax * FEED_GAP_M,
            my - uay * FEED_GAP_M,
            mz)

        self.get_logger().info(
            f'WP1 height  ({p1x:+.3f},{p1y:+.3f},{p1z:+.3f})\n'
            f'WP2 approach({p2x:+.3f},{p2y:+.3f},{p2z:+.3f})\n'
            f'WP3 feed    ({p3x:+.3f},{p3y:+.3f},{p3z:+.3f})')

        self._pub_sphere(BASE_FRAME, p1x,p1y,p1z, mid=3, rgba=(1.0,1.0,0.0,0.9), d=0.03)
        self._pub_sphere(BASE_FRAME, p2x,p2y,p2z, mid=1, rgba=(0.2,1.0,0.2,0.9), d=0.04)
        self._pub_sphere(BASE_FRAME, p3x,p3y,p3z, mid=4, rgba=(1.0,0.5,0.0,0.9), d=0.03)

        # ── Phase 1: Height ───────────────────────────────────────────
        if abs(p1z - ez) > MIN_MOVE_EPS_M:
            self._phase(self._pose(p1x, p1y, p1z, cur_ori),
                        'height-alignment', required=False)
        else:
            self.get_logger().info('Phase 1 skipped — already at mouth height.')

        # ── Phase 2: Lateral approach ─────────────────────────────────
        lat = math.sqrt((p2x-p1x)**2 + (p2y-p1y)**2)
        if lat > MIN_MOVE_EPS_M:
            if not self._phase(self._pose(p2x, p2y, p2z, cur_ori),
                               'lateral-approach', required=True):
                self._safe_return(start_js)
                return False, 'Phase 2 (lateral approach) failed — returned to start.'
        else:
            self.get_logger().info('Phase 2 skipped — already at approach position.')

        # ── Phase 3: Forward feed ─────────────────────────────────────
        fwd = math.sqrt((p3x-p2x)**2 + (p3y-p2y)**2)
        if fwd > MIN_MOVE_EPS_M:
            if not self._phase(self._pose(p3x, p3y, p3z, cur_ori),
                               'forward-feed', required=True):
                self._safe_return(start_js)
                return False, 'Phase 3 (forward feed) failed — returned to start.'
        else:
            self.get_logger().info('Phase 3 skipped — already at feed position.')

        # ── Hold ──────────────────────────────────────────────────────
        self.get_logger().info(f'Holding at feed position for {HOLD_SECONDS:.0f} s …')
        time.sleep(HOLD_SECONDS)

        # ── Return ────────────────────────────────────────────────────
        self.get_logger().info('Returning to start configuration …')
        ok = self._safe_return(start_js)
        return True, ('Feed complete.' if ok
                      else 'Feed complete — WARNING: return failed.')

    # ── Phase: IK → joint trajectory ─────────────────────────────────

    def _phase(self, target_pose: Pose, name: str, required: bool = True) -> bool:
        self.get_logger().info(f'[{name}] Solving IK …')

        js = self._compute_ik(target_pose, avoid_collisions=True)
        if js is None:
            self.get_logger().warn(
                f'[{name}] IK with collision check failed — retrying without.')
            js = self._compute_ik(target_pose, avoid_collisions=False)

        if js is None:
            if not required:
                self.get_logger().warn(
                    f'[{name}] IK failed — skipping optional phase.')
                return True
            self.get_logger().error(f'[{name}] IK failed — no solution.')
            return False

        ok = self._move_to_joints(js, min_dur=MIN_PHASE_DUR_S)
        if ok:
            self.get_logger().info(f'[{name}] ✓')
        else:
            self.get_logger().error(f'[{name}] Joint move failed.')
        return ok

    # ── IK call ───────────────────────────────────────────────────────

    def _compute_ik(self, target_pose: Pose,
                    avoid_collisions: bool = True) -> Optional[JointState]:
        req = GetPositionIK.Request()
        req.ik_request.group_name       = ARM_GROUP
        req.ik_request.ik_link_name     = EE_LINK
        req.ik_request.avoid_collisions = avoid_collisions
        req.ik_request.timeout.sec      = 3

        if self._latest_js is not None:
            req.ik_request.robot_state.joint_state = self._latest_js

        ps = PoseStamped()
        ps.header.frame_id = BASE_FRAME
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.pose            = target_pose
        req.ik_request.pose_stamped = ps

        future = self._ik_client.call_async(req)
        if not self._wait(future, 6.0):
            self.get_logger().error('IK timed out.')
            return None

        res = future.result()
        if res is None or res.error_code.val != 1:
            code = res.error_code.val if res else 'None'
            self.get_logger().error(
                f'IK error {code} | avoid_col={avoid_collisions} | '
                f'target=({target_pose.position.x:+.3f},'
                f'{target_pose.position.y:+.3f},'
                f'{target_pose.position.z:+.3f})')
            return None

        return res.solution.joint_state

    # ── Send joint trajectory ─────────────────────────────────────────

    def _move_to_joints(self, target_js: JointState,
                        min_dur: float = MIN_PHASE_DUR_S) -> bool:
        names: List[str]   = []
        positions: List[float] = []
        for n, p in zip(target_js.name, target_js.position):
            if n in ARM_JOINT_NAMES:
                names.append(n)
                positions.append(float(p))

        if not names:
            self.get_logger().error('No arm joints in IK solution.')
            return False

        duration = min_dur
        cur_positions: List[float] = []
        if self._latest_js is not None:
            js_names = list(self._latest_js.name)
            for n, tgt in zip(names, positions):
                if n in js_names:
                    delta    = abs(tgt - float(
                        self._latest_js.position[js_names.index(n)]))
                    duration = max(duration, delta / MAX_JOINT_VEL_RAD_S)
            cur_positions = [
                float(self._latest_js.position[js_names.index(n)])
                if n in js_names else p
                for n, p in zip(names, positions)
            ]
        else:
            cur_positions = list(positions)

        self.get_logger().info(
            f'Sending {len(names)} joints, duration={duration:.2f} s')

        sec  = int(duration)
        nsec = int((duration - sec) * 1e9)

        pt_start = JointTrajectoryPoint()
        pt_start.positions      = cur_positions
        pt_start.time_from_start = DurationMsg(sec=0, nanosec=0)

        pt_end = JointTrajectoryPoint()
        pt_end.positions      = positions
        pt_end.time_from_start = DurationMsg(sec=sec, nanosec=nsec)

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = names
        traj.points       = [pt_start, pt_end]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = self._traj_client.send_goal_async(goal)
        if not self._wait(future, 5.0):
            self.get_logger().error('Trajectory goal send timed out.')
            return False

        gh = future.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('Trajectory rejected.')
            return False

        res_f = gh.get_result_async()
        if not self._wait(res_f, duration + 10.0):
            self.get_logger().error('Trajectory execution timed out.')
            return False

        rw = res_f.result()
        return rw is not None and rw.result.error_code == 0

    def _safe_return(self, start_js: Optional[JointState]) -> bool:
        if start_js is None:
            self.get_logger().warn('No start JS saved — cannot return.')
            return False
        return self._move_to_joints(start_js, min_dur=RETURN_DUR_S)

    # ── Helpers ───────────────────────────────────────────────────────

    def _wait(self, future, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while not future.done():
            if time.time() > deadline:
                return False
            time.sleep(0.01)
        return True

    def _lookup_tf(self, target, source, timeout=2.0, stamp=None):
        if stamp is None:
            stamp = Time()
        try:
            return self._tf_buf.lookup_transform(
                target, source, stamp, timeout=Duration(seconds=timeout))
        except ConnectivityException as e:
            frames = self._tf_buf.all_frames_as_string()
            self.get_logger().error(
                f'TF broken {source}→{target}.\n'
                f'Active frames:\n{frames}\n'
                'Check arm driver: ros2 topic hz /joint_states')
            raise RuntimeError(str(e)) from e
        except (LookupException, ExtrapolationException) as e:
            raise RuntimeError(f'TF {source}→{target}: {e}') from e

    def _clamp(self, x, y, z):
        norm = math.sqrt(x*x + y*y + z*z)
        if norm > MAX_REACH:
            s = MAX_REACH / norm
            x, y, z = x*s, y*s, z*s
            self.get_logger().warn(
                f'Clamped to reach limit: ({x:+.3f},{y:+.3f},{z:+.3f})')
        return x, y, z

    def _pose(self, x, y, z, ori) -> Pose:
        return Pose(
            position=Point(x=float(x), y=float(y), z=float(z)),
            orientation=Quaternion(
                x=ori.x, y=ori.y, z=ori.z, w=ori.w))

    def _pub_sphere(self, frame, x, y, z, mid, rgba, d=0.04):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns, m.id        = 'feeding', mid
        m.type            = Marker.SPHERE
        m.action          = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = d
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.lifetime.sec = 0
        self._marker_pub.publish(m)

    def _publish_marker_cam(self, mouth_cam: PointStamped):
        try:
            tf_wc = self._lookup_tf('world', CAM_FRAME, timeout=0.2)
        except RuntimeError:
            return
        mw = do_transform_point(mouth_cam, tf_wc)
        self._pub_sphere('world', mw.point.x, mw.point.y, mw.point.z,
                         mid=10, rgba=(1.0, 0.4, 0.7, 0.9), d=0.05)


def main(args=None):
    rclpy.init(args=args)
    node = MouthFeedingPlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()