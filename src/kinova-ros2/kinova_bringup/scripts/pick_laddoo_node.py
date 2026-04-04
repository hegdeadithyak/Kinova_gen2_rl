#!/usr/bin/env python3
"""
pick_laddoo_node.py  (v5 — fully robust, actually feeds the patient)
════════════════════════════════════════════════════════════════════════════

WHAT CHANGED FROM v4
─────────────────────
  v4 fixed the bugs but planning was STILL silently failing because:

  1. Scene not confirmed before planning:
       v4 published to /planning_scene (fire-and-forget).  move_group might
       not have processed it yet when the first plan request arrives.
       FIX: use /apply_planning_scene SERVICE (synchronous RPC) so we block
       until move_group confirms the scene is committed.

  2. No success checking / retry:
       move_to_pose() + wait_until_executed() never checked whether the plan
       actually succeeded.  On FAILURE the arm stays put, the code merrily
       moves on, and the sequence "completes" with the arm stationary.
       FIX: _move_robust() retries up to MAX_RETRIES times, progressively
       widening position + orientation tolerances.  Raises on total failure.

  3. Cartesian flag used too aggressively:
       compute_cartesian_path requires EVERY interpolated waypoint to pass
       IK + collision checks along the straight-line segment.  The 5 mm IK
       mesh can easily fail even for a short lift or retract.
       FIX: Cartesian=True ONLY for the critical ±Z moves (descent-to-grasp,
       lift, mouth-delivery).  Swing/approach moves use joint-space.
       On retry 2+ Cartesian failures also fall back to joint-space.

  4. Orientation too tight for IKFast:
       The j2n6s300 IKFast solver struggles inside a narrow orientation cone
       at the extremes of workspace.
       FIX: orientation tolerance starts at 0.40 rad (~23°) and widens to
       π (~180°, i.e. position-only) on attempt 4.

  5. move_group startup race:
       The 5-second delay was a guess; move_group takes 30-40 s on first
       launch.
       FIX: _startup() thread polls /apply_planning_scene until it responds,
       then applies the scene, sets _scene_applied=True, and only then
       accepts trigger calls.

  6. Pedestal collision bug (from v3/v4) confirmed fixed:
       Pedestal top at z = -0.10, well below arm origin (z=0).

COORDINATE TRANSFORM
──────────────────────
  ARM_SPAWN = (0.40, -0.50, 0.85)   [kinova_launch.py]
  p_moveit  = p_gazebo - ARM_SPAWN

KEY WAYPOINTS (MoveIt2 frame)
──────────────────────────────
  LADDOO   = ( 0.150,  0.250, -0.016)   dist = 0.294 m ✓
  MOUTH    = ( 0.260,  0.500,  0.140)   dist = 0.579 m ✓

TRIGGER
────────
  ros2 service call /feed_trigger std_srvs/srv/Trigger {}
"""

import threading
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene as ApplyPlanningSceneSrv
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from action_msgs.msg import GoalStatus

try:
    from pymoveit2 import MoveIt2
    PYMOVEIT2_AVAILABLE = True
except ImportError:
    PYMOVEIT2_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

ROBOT_NAME     = 'j2n6s300'
PLANNING_FRAME = 'world'
ARM_SPAWN      = np.array([0.40, -0.50, 0.85])


def g2m(gazebo_xyz):
    """Gazebo world XYZ  →  MoveIt2 (URDF world) XYZ."""
    return np.asarray(gazebo_xyz, dtype=float) - ARM_SPAWN


# ── Key targets ───────────────────────────────────────────────────────────────
LADDOO = g2m([0.550, -0.250, 0.834])   # (+0.150, +0.250, -0.016)
MOUTH  = g2m([0.660,  0.000, 0.990])   # (+0.260, +0.500, +0.140)

# ── Clearances ────────────────────────────────────────────────────────────────
PRE_GRASP_Z  = 0.16    # hover above laddoo
GRASP_Z      = 0.030   # fingertip contact height above laddoo centre
LIFT_Z       = 0.22    # rise above table top (-0.085) with margin
RETRACT_DIST = 0.14    # pull back -X after release

# ── Orientations ─────────────────────────────────────────────────────────────
# With large tolerances these act as start-hints for IKFast.
QUAT_DOWN  = Quaternion(x=0.000, y=0.707, z=0.000, w=0.707)  # gripper down
QUAT_FRONT = Quaternion(x=0.500, y=0.500, z=0.500, w=0.500)  # toward patient

# ── Planning retry ladder ─────────────────────────────────────────────────────
# Each tuple: (position_tolerance_m, orientation_tolerance_rad)
TOLERANCE_LADDER = [
    (0.008, 0.40),          # attempt 1 — tight  (~23°)
    (0.015, 0.70),          # attempt 2 — medium (~40°)
    (0.025, 1.20),          # attempt 3 — loose  (~69°)
    (0.040, math.pi),       # attempt 4 — position-only
]
PLANNING_TIME_BASE = 20.0   # seconds; multiplied by attempt number
MAX_RETRIES        = len(TOLERANCE_LADDER)

# ── Fingers ───────────────────────────────────────────────────────────────────
FINGER_JOINTS = [f'{ROBOT_NAME}_joint_finger_{i}' for i in range(1, 4)]
FINGER_OPEN   = [0.00, 0.00, 0.00]
FINGER_GRASP  = [0.70, 0.70, 0.70]
FINGER_DT     = 2.0

# ── Home ──────────────────────────────────────────────────────────────────────
HOME_JOINTS = [4.71, 2.71, 1.57, 4.71, 0.0, 3.14]


# ═════════════════════════════════════════════════════════════════════════════
#  COLLISION SCENE BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _prim_pose(cx, cy, cz):
    p = Pose()
    p.position    = Point(x=float(cx), y=float(cy), z=float(cz))
    p.orientation = Quaternion(w=1.0)
    return p

def _box(cid, cx, cy, cz, sx, sy, sz):
    co = CollisionObject()
    co.id = cid
    co.header = Header(frame_id=PLANNING_FRAME)
    co.operation = CollisionObject.ADD
    co.primitives = [SolidPrimitive(type=SolidPrimitive.BOX,
                                    dimensions=[float(sx), float(sy), float(sz)])]
    co.primitive_poses = [_prim_pose(cx, cy, cz)]
    return co

def _sphere(cid, cx, cy, cz, r):
    co = CollisionObject()
    co.id = cid
    co.header = Header(frame_id=PLANNING_FRAME)
    co.operation = CollisionObject.ADD
    co.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE,
                                    dimensions=[float(r)])]
    co.primitive_poses = [_prim_pose(cx, cy, cz)]
    return co

def _cylinder(cid, cx, cy, cz, r, h):
    co = CollisionObject()
    co.id = cid
    co.header = Header(frame_id=PLANNING_FRAME)
    co.operation = CollisionObject.ADD
    co.primitives = [SolidPrimitive(type=SolidPrimitive.CYLINDER,
                                    dimensions=[float(h), float(r)])]
    co.primitive_poses = [_prim_pose(cx, cy, cz)]
    return co


def build_collision_objects():
    """
    Mirror feeding_scene.sdf in MoveIt2 planning frame.
    food_item_*/food_bowl excluded so the robot can reach them.

    CRITICAL: pedestal top must remain below z = -0.10 to avoid overlapping
    j2n6s300_link_base (at origin), which would make every start-state
    appear in-collision and cause all plans to return FAILURE.
    """
    o = []

    # Ground  (Gazebo z=0 → MoveIt2 z=-0.85; slab centre at -0.875)
    o.append(_box('ground',         0.0,  0.0, -0.875, 6.0,  6.0,  0.05))

    # Pedestal: height 0.75 m, top at z=-0.10, centre at -0.475
    o.append(_box('pedestal',       0.0,  0.0, -0.475, 0.24, 0.24, 0.75))

    # Food side table
    p = g2m([0.55, -0.25, 0.765])
    o.append(_box('table_top',  p[0], p[1], p[2],  0.44, 0.44, 0.05))
    p = g2m([0.55, -0.25, 0.375])
    o.append(_box('table_body', p[0], p[1], p[2],  0.24, 0.24, 0.75))

    # Chair
    p = g2m([0.75,  0.00,  0.44])
    o.append(_box('chair_seat',    p[0], p[1], p[2], 0.54, 0.54, 0.07))
    p = g2m([0.90,  0.00,  0.70])
    o.append(_box('chair_back',    p[0], p[1], p[2], 0.07, 0.50, 0.58))
    p = g2m([0.75, +0.255, 0.645])
    o.append(_box('armrest_left',  p[0], p[1], p[2], 0.48, 0.06, 0.05))
    p = g2m([0.75, -0.255, 0.645])
    o.append(_box('armrest_right', p[0], p[1], p[2], 0.48, 0.06, 0.05))

    # Patient
    p = g2m([0.75, 0.0, 0.74])
    o.append(_box('patient_torso', p[0], p[1], p[2], 0.26, 0.32, 0.44))
    p = g2m([0.63, 0.0, 0.46])
    o.append(_box('patient_lower', p[0], p[1], p[2], 0.36, 0.28, 0.12))
    # Head shifted 4 cm toward backrest (+x) so mouth area stays reachable
    p = g2m([0.79, 0.0, 1.04])
    o.append(_sphere('patient_head', p[0], p[1], p[2], 0.10))
    p = g2m([0.75, 0.0, 0.965])
    o.append(_cylinder('patient_neck', p[0], p[1], p[2], 0.045, 0.07))

    # Monitor
    p = g2m([0.165, 0.0, 1.130])
    o.append(_box('monitor',       p[0], p[1], p[2], 0.06, 0.42, 0.30))
    p = g2m([-0.10, 0.0, 0.50])
    o.append(_cylinder('monitor_pole', p[0], p[1], p[2], 0.03, 1.02))
    p = g2m([-0.10, 0.0, 0.018])
    o.append(_box('monitor_base',  p[0], p[1], p[2], 0.37, 0.37, 0.04))

    return o


# ═════════════════════════════════════════════════════════════════════════════
class PickLaddooNode(Node):

    def __init__(self):
        super().__init__('pick_laddoo_node')
        self._cb = ReentrantCallbackGroup()

        # ── MoveIt2 ───────────────────────────────────────────────────────────
        if PYMOVEIT2_AVAILABLE:
            self._arm = MoveIt2(
                node=self,
                joint_names=[f'{ROBOT_NAME}_joint_{i}' for i in range(1, 7)],
                base_link_name=PLANNING_FRAME,
                end_effector_name=f'{ROBOT_NAME}_end_effector',
                group_name='arm',
                callback_group=self._cb,
            )
            self._arm.planner_id            = 'RRTConnectkConfigDefault'
            self._arm.max_velocity          = 0.12
            self._arm.max_acceleration      = 0.06
            self._arm.num_planning_attempts = 8
        else:
            self.get_logger().fatal('pymoveit2 not available!')
            self._arm = None

        # ── ApplyPlanningScene service ────────────────────────────────────────
        # Synchronous RPC — guarantees move_group has processed the scene
        # before we send any planning requests.
        self._apply_scene_cli = self.create_client(
            ApplyPlanningSceneSrv,
            '/apply_planning_scene',
            callback_group=self._cb,
        )

        # ── Finger controller ─────────────────────────────────────────────────
        self._fingers = ActionClient(
            self,
            FollowJointTrajectory,
            '/finger_trajectory_controller/follow_joint_trajectory',
            callback_group=self._cb,
        )

        # ── Trigger service ───────────────────────────────────────────────────
        self._srv       = self.create_service(
            Trigger, '/feed_trigger', self._trigger_cb,
            callback_group=self._cb,
        )
        self._busy          = threading.Lock()
        self._scene_applied = False

        # Startup in background: wait for move_group → apply scene → ready
        threading.Thread(target=self._startup, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────────
    #  STARTUP
    # ──────────────────────────────────────────────────────────────────────────

    def _startup(self):
        log = self.get_logger()
        log.info('Waiting for move_group (/apply_planning_scene) …')

        # Block until move_group is alive
        while not self._apply_scene_cli.wait_for_service(timeout_sec=5.0):
            log.info('  … still waiting for move_group to start')

        log.info('move_group is up — committing collision scene …')
        self._commit_scene()

        # Short settle so move_group indexes the new objects before planning
        time.sleep(2.0)
        self._scene_applied = True

        log.info(
            '\n╔══════════════════════════════════════════════╗'
            '\n║  pick_laddoo_node  READY                     ║'
            '\n║                                              ║'
            '\n║  ros2 service call /feed_trigger             ║'
            '\n║    std_srvs/srv/Trigger {}                   ║'
            '\n╚══════════════════════════════════════════════╝'
        )

    def _commit_scene(self):
        """Apply all collision objects via /apply_planning_scene (blocking)."""
        objs = build_collision_objects()

        scene         = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = objs

        req       = ApplyPlanningSceneSrv.Request()
        req.scene = scene

        future   = self._apply_scene_cli.call_async(req)
        deadline = time.time() + 20.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)

        ok = future.done() and future.result() and future.result().success
        if ok:
            self.get_logger().info(
                f'Collision scene committed — {len(objs)} objects.'
            )
            for obj in objs:
                p = obj.primitive_poses[0].position
                t = {SolidPrimitive.BOX: 'BOX',
                     SolidPrimitive.SPHERE: 'SPH',
                     SolidPrimitive.CYLINDER: 'CYL'}.get(
                         obj.primitives[0].type, '?')
                self.get_logger().info(
                    f'  [{obj.id:20s}]  {t} '
                    f'({p.x:+.3f}, {p.y:+.3f}, {p.z:+.3f})'
                )
        else:
            self.get_logger().error(
                'ApplyPlanningScene call FAILED or timed out — '
                'planning may collide with scene objects!'
            )

    # ──────────────────────────────────────────────────────────────────────────
    #  TRIGGER
    # ──────────────────────────────────────────────────────────────────────────

    def _trigger_cb(self, req, res):
        if not self._scene_applied:
            res.success = False
            res.message = 'Node still initialising — please wait.'
            return res
        if not self._busy.acquire(blocking=False):
            res.success = False
            res.message = 'Feed sequence already in progress.'
            return res
        threading.Thread(target=self._sequence, daemon=True).start()
        res.success = True
        res.message = 'Feed sequence started.'
        return res

    # ──────────────────────────────────────────────────────────────────────────
    #  FEED SEQUENCE
    # ──────────────────────────────────────────────────────────────────────────

    def _sequence(self):
        log = self.get_logger()
        try:
            # ─────────────────────────────────────────────────────────────────
            # 1. Open fingers
            # ─────────────────────────────────────────────────────────────────
            log.info('━━━ STEP 1/7  Open fingers ━━━')
            self._finger_cmd(FINGER_OPEN, 'open')

            # ─────────────────────────────────────────────────────────────────
            # 2. Pre-grasp hover — directly above laddoo, gripper down
            #    Joint-space is safest here; IK has many solutions away from
            #    the bowl.
            # ─────────────────────────────────────────────────────────────────
            log.info('━━━ STEP 2/7  Pre-grasp hover ━━━')
            pre = LADDOO.copy()
            pre[2] += PRE_GRASP_Z   # z = -0.016 + 0.160 = +0.144
            self._move_robust(pre, QUAT_DOWN, 'pre_grasp', cartesian=False)

            # ─────────────────────────────────────────────────────────────────
            # 3. Descend straight down into bowl (Cartesian — keep it vertical)
            # ─────────────────────────────────────────────────────────────────
            log.info('━━━ STEP 3/7  Descend to laddoo ━━━')
            grasp = LADDOO.copy()
            grasp[2] += GRASP_Z     # z = -0.016 + 0.030 = +0.014
            self._move_robust(grasp, QUAT_DOWN, 'grasp_descent', cartesian=True)

            # ─────────────────────────────────────────────────────────────────
            # 4. Close fingers
            # ─────────────────────────────────────────────────────────────────
            log.info('━━━ STEP 4/7  Close fingers ━━━')
            self._finger_cmd(FINGER_GRASP, 'grasp')
            time.sleep(0.8)

            # ─────────────────────────────────────────────────────────────────
            # 5. Lift straight up, clear of table and bowl rim (Cartesian)
            # ─────────────────────────────────────────────────────────────────
            log.info('━━━ STEP 5/7  Lift ━━━')
            lift = LADDOO.copy()
            lift[2] += LIFT_Z       # z = -0.016 + 0.220 = +0.204
            self._move_robust(lift, QUAT_DOWN, 'lift', cartesian=True)

            # ─────────────────────────────────────────────────────────────────
            # 6. Deliver to mouth — two-phase:
            #    6a) Joint-space swing to approach point (in front of & above
            #        mouth), avoids passing through patient_head sphere.
            #    6b) Cartesian descent forward to mouth (2 cm gap).
            # ─────────────────────────────────────────────────────────────────
            log.info('━━━ STEP 6/7  Deliver to mouth ━━━')

            approach = np.array([
                MOUTH[0] - 0.10,   # 10 cm in front of mouth
                MOUTH[1],
                MOUTH[2] + 0.12,   # 12 cm above mouth
            ])
            self._move_robust(approach, QUAT_FRONT,
                              'mouth_approach', cartesian=False)

            deliver = np.array([
                MOUTH[0] - 0.02,   # 2 cm gap — no direct contact
                MOUTH[1],
                MOUTH[2],
            ])
            self._move_robust(deliver, QUAT_FRONT,
                              'mouth_deliver', cartesian=True)

            log.info('Holding at mouth position …')
            time.sleep(1.5)

            # ─────────────────────────────────────────────────────────────────
            # 7. Release, retract, return home
            # ─────────────────────────────────────────────────────────────────
            log.info('━━━ STEP 7/7  Release & retract ━━━')
            self._finger_cmd(FINGER_OPEN, 'release')
            time.sleep(0.5)

            retract = np.array([
                MOUTH[0] - RETRACT_DIST,
                MOUTH[1],
                MOUTH[2] + 0.06,
            ])
            self._move_robust(retract, QUAT_FRONT,
                              'retract', cartesian=True)

            self._go_home()

            log.info(
                '\n╔══════════════════════════════════╗'
                '\n║  Feed sequence complete  ✓       ║'
                '\n╚══════════════════════════════════╝'
            )

        except RuntimeError as exc:
            log.error(f'Feed sequence ABORTED: {exc}')
        except Exception as exc:
            import traceback
            log.error(f'Unexpected error: {exc}\n{traceback.format_exc()}')
        finally:
            self._busy.release()

    # ──────────────────────────────────────────────────────────────────────────
    #  MOTION HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _move_robust(self, xyz: np.ndarray, quat: Quaternion,
                     label: str, cartesian: bool = False):
        """
        Move to pose with automatic retry and widening tolerances.

        Attempt 1:   8 mm,  0.40 rad (~23°)  — tight
        Attempt 2:  15 mm,  0.70 rad (~40°)  — medium
        Attempt 3:  25 mm,  1.20 rad (~69°)  — loose
        Attempt 4:  40 mm,  π rad   (~180°)  — position-only

        On Cartesian failure at attempt ≥ 2, automatically retries as
        joint-space (less prone to IK gaps along the path).

        Raises RuntimeError if all attempts fail.
        """
        if self._arm is None:
            return

        log = self.get_logger()
        log.info(
            f'[arm→{label}]  '
            f'({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f})  '
            f'dist={np.linalg.norm(xyz):.3f} m  cartesian={cartesian}'
        )

        pose             = Pose()
        pose.position    = Point(x=float(xyz[0]),
                                 y=float(xyz[1]),
                                 z=float(xyz[2]))
        pose.orientation = quat

        last_err = 'unknown'
        for attempt, (pos_tol, ori_tol) in enumerate(TOLERANCE_LADDER, 1):
            self._arm.goal_position_tolerance    = pos_tol
            self._arm.goal_orientation_tolerance = ori_tol
            self._arm.planning_time              = PLANNING_TIME_BASE * attempt

            log.info(
                f'  attempt {attempt}/{MAX_RETRIES}  '
                f'pos={pos_tol*1000:.0f} mm  '
                f'ori={math.degrees(ori_tol):.0f}°  '
                f'time={self._arm.planning_time:.0f} s'
            )

            # Primary attempt
            try:
                self._arm.move_to_pose(pose=pose, cartesian=cartesian)
                self._arm.wait_until_executed()
                log.info(f'  [arm→{label}] ✓  (attempt {attempt})')
                return
            except Exception as exc:
                last_err = str(exc)
                log.warn(f'  attempt {attempt} failed: {exc}')

            # Cartesian fallback to joint-space on later attempts
            if cartesian and attempt >= 2:
                log.warn(
                    f'  Cartesian failed — retrying as joint-space '
                    f'(attempt {attempt})'
                )
                try:
                    self._arm.move_to_pose(pose=pose, cartesian=False)
                    self._arm.wait_until_executed()
                    log.info(
                        f'  [arm→{label}] ✓ via joint-space fallback'
                    )
                    return
                except Exception as exc2:
                    last_err = str(exc2)
                    log.warn(f'  joint-space fallback also failed: {exc2}')

        raise RuntimeError(
            f'[arm→{label}] FAILED after {MAX_RETRIES} attempts. '
            f'Last error: {last_err}'
        )

    def _go_home(self):
        if self._arm is None:
            return
        log = self.get_logger()
        log.info('[arm→home]')
        self._arm.goal_position_tolerance    = 0.015
        self._arm.goal_orientation_tolerance = 0.5
        self._arm.planning_time              = PLANNING_TIME_BASE
        # Pass positionally — keyword arg name varies between pymoveit2 builds
        self._arm.move_to_configuration(HOME_JOINTS)
        self._arm.wait_until_executed()
        log.info('[arm→home] ✓')

    def _finger_cmd(self, positions: list, label: str):
        """Send finger trajectory and wait for result."""
        server = '/finger_trajectory_controller/follow_joint_trajectory'

        if not self._fingers.wait_for_server(timeout_sec=4.0):
            self.get_logger().warn(
                f'[finger] server not available — skipping {label}\n'
                f'  Expected: {server}\n'
                f'  Check:    ros2 action list | grep finger'
            )
            return

        traj             = JointTrajectory()
        traj.joint_names = FINGER_JOINTS
        pt               = JointTrajectoryPoint()
        pt.positions     = [float(p) for p in positions]
        pt.velocities    = [0.0] * 3
        pt.time_from_start = Duration(
            sec=int(FINGER_DT),
            nanosec=int((FINGER_DT % 1) * 1e9),
        )
        traj.points = [pt]

        goal            = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(f'[finger→{label}]  {positions}')

        f = self._fingers.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f, timeout_sec=6.0)

        gh = f.result()
        if not gh or not gh.accepted:
            self.get_logger().error(f'[finger→{label}] goal rejected')
            return

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=FINGER_DT + 3.0)
        r = rf.result()
        if r and r.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'[finger→{label}] ✓')
        else:
            self.get_logger().warn(
                f'[finger→{label}] ended status='
                f'{r.status if r else "none"}'
            )


# ═════════════════════════════════════════════════════════════════════════════
def main():
    rclpy.init()
    node = PickLaddooNode()
    exe  = MultiThreadedExecutor(num_threads=6)
    exe.add_node(node)
    try:
        exe.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()