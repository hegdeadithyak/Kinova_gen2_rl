#!/usr/bin/env python3
"""
click_pointer.py  —  Auto-mouth detection + Click-to-Feed
==========================================================

Flow:
  1. MediaPipe FaceLandmarker runs at 15 Hz on the colour stream.
     When a mouth is found its pixel (u,v) is highlighted in the
     live window.  Press  [SPACE]  to auto-use that pixel, or just
     left-click any pixel as before.

  2. Pixel (u,v) + aligned depth  →  3D point in camera frame.
     target_y  -=  CAM_Y_OFFSET  (camera sits above EE in cam-Y).

  3. Confirm target, then choose motion mode:
       [1] Custom Algorithm  — phased proportional joint stepping
       [2] BFMT Planner      — MoveIt2 ABITstar optimal path planning

Jaco2 joint convention (j2s6s200):
  J1  base yaw          J2  shoulder pitch   J3  elbow pitch
  J4  wrist pitch       J5  wrist roll       J6  finger roll

Phase-3 note  (reach toward mouth):
  The arm first fully pre-tilts J4 (wrist pitch) to level the spoon,
  then sweeps J1 (base yaw) to translate the spoon toward the mouth.
  J5 is held at its locked value to prevent roll drift.
"""

import os
import threading
import time
import urllib.request

import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped
from moveit_msgs.msg import (
    BoundingVolume, Constraints, MotionPlanRequest,
    MoveItErrorCodes, PositionConstraint, RobotState,
)
from moveit_msgs.srv import GetMotionPlan
from sensor_msgs.msg import Image, JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ══════════════════════════════════════════════════════════════════════════════
#  Terminal colour palette (ANSI)
# ══════════════════════════════════════════════════════════════════════════════
class C:
    RESET   = '\033[0m';  BOLD    = '\033[1m';  DIM     = '\033[2m'
    RED     = '\033[91m'; GREEN   = '\033[92m'; YELLOW  = '\033[93m'
    BLUE    = '\033[94m'; MAGENTA = '\033[95m'; CYAN    = '\033[96m'
    WHITE   = '\033[97m'; ORANGE  = '\033[38;5;208m'

W = 62

def banner(title):
    pad = W - len(title) - 4; l, r = pad // 2, pad - pad // 2
    print(f"\n{C.CYAN}{C.BOLD}╔{'═'*(W-2)}╗\n║  {' '*l}{title}{' '*r}  ║\n╚{'═'*(W-2)}╝{C.RESET}")

def section(title, color=C.YELLOW):
    print(f"\n{color}{C.BOLD}  ┌─ {title} {'─'*(W-len(title)-6)}┐{C.RESET}")

def row(label, value, color=C.WHITE):
    print(f"{C.DIM}  │{C.RESET}  {C.CYAN}{label:<18}{C.RESET}{color}{value}{C.RESET}")

def done_row(label, value): row(label, f"✓  {value}", C.GREEN)
def warn_row(label, value): row(label, f"⚠  {value}", C.YELLOW)
def err_row(label, value):  row(label, f"✗  {value}", C.RED)

def sep(): print(f"{C.DIM}  └{'─'*(W-4)}┘{C.RESET}")

def cprint(msg, color=C.WHITE, bold=False):
    b = C.BOLD if bold else ''
    print(f"  {b}{color}{msg}{C.RESET}")

def spinner_msg(msg, color=C.MAGENTA):
    print(f"\n  {color}{C.BOLD}⟳  {msg}{C.RESET}")

# ══════════════════════════════════════════════════════════════════════════════
#  MediaPipe model
# ══════════════════════════════════════════════════════════════════════════════
_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
_MODEL_PATH = os.path.expanduser("~/.cache/mediapipe/face_landmarker.task")

# Mouth landmark indices (MediaPipe 478-point topology)
_MOUTH_IDX = [13, 14, 61, 291, 78, 308]

# Lip outline connections for debug overlay
_LIP_CONN = [
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),
    (405,321),(321,375),(375,291),(61,185),(185,40),(40,39),(39,37),
    (37,0),(0,267),(267,269),(269,270),(270,409),(409,291),
    (78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),
    (402,318),(318,324),(324,308),(78,191),(191,80),(80,81),(81,82),
    (82,13),(13,312),(312,311),(311,310),(310,415),(415,308),
]

def _ensure_model():
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    if not os.path.exists(_MODEL_PATH):
        cprint(f'Downloading FaceLandmarker model → {_MODEL_PATH} …', C.CYAN)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        cprint('Model download complete.', C.GREEN)
    return _MODEL_PATH

# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════
FX_REF, FY_REF = 603.6312, 603.0632
REF_W,  REF_H  = 640, 480

BASE_FRAME = 'j2s6s200_link_base'
EE_LINK    = 'j2s6s200_end_effector'
CAM_FRAME  = 'camera_color_optical_frame'

ARM_JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]

CAM_Y_OFFSET = 0.09   # camera 9 cm above EE in camera-Y

# ── Phase 1 (Y / J2) ─────────────────────────────────────────────────────────
STEP_RADS_Y  = 0.04       # smaller → smoother
CART_DELTA_Y = 0.16       # metres of Cartesian travel per unit step
TICK_Y_S     = 0.35       # seconds per step

# ── Phase 2 (X / J3) ─────────────────────────────────────────────────────────
STEP_RADS_X  = 0.08
CART_DELTA_X = 0.06
TICK_X_S     = 0.30

# ── Phase 3 (Z reach — J1 + J4 coupled + J5 locked) ─────────────────────────
J4_PER_J1    = 0.55       # J4 correction per radian of J1 motion (flipped to reverse)
STEP_J1_RAD  = 0.035      # base yaw step per tick  (gentle!)
CART_DELTA_Z = 0.018      # metres of Z advance per J1 step
TICK_Z_S     = 0.28       # seconds per step  (smooth, no jerk)
ERR_TOL_M    = 0.04       # convergence threshold for all axes

# ── BFMT / MoveIt2 ───────────────────────────────────────────────────────────
MOVEIT_GROUP     = 'arm'
MOVEIT_PLANNER   = 'BFMTkConfigDefault'
MOVEIT_PLAN_TIME = 15.0
MOVEIT_ATTEMPTS  = 3
MOVEIT_VEL_SCALE = 0.3
MOVEIT_ACC_SCALE = 0.3
MOVEIT_TOL_M     = 0.05

# ── Depth median patch ────────────────────────────────────────────────────────
DEPTH_PATCH_R = 7          # sample radius in pixels (larger → more robust)

# ══════════════════════════════════════════════════════════════════════════════
#  Node
# ══════════════════════════════════════════════════════════════════════════════
class ClickPointerNode(Node):

    def __init__(self):
        super().__init__('click_pointer')
        self._cb = ReentrantCallbackGroup()

        self._bridge    = CvBridge()
        self._lock      = threading.Lock()
        self._color_img = None
        self._depth_img = None
        self._busy      = False
        self._current_positions = {n: 0.0 for n in ARM_JOINT_NAMES}

        # MediaPipe mouth detection state
        self._mouth_pixel  = None   # (u, v) in colour image coords
        self._mouth_locked = False  # True while "use mouth" was pressed
        self._mp_ts_ms     = 0      # monotonic timestamp for VIDEO mode

        self._tf_buf = Buffer()
        TransformListener(self._tf_buf, self)

        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._color_cb, 10, callback_group=self._cb)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw',
                                 self._depth_cb, 10, callback_group=self._cb)
        self.create_subscription(JointState, '/joint_states',
                                 self._js_cb, 10, callback_group=self._cb)

        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            callback_group=self._cb,
        )

        # Build MediaPipe FaceLandmarker
        cprint('Loading MediaPipe FaceLandmarker …', C.CYAN)
        _path = _ensure_model()
        _opts = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.55,
            min_face_presence_confidence=0.55,
            min_tracking_confidence=0.45,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._face_det = FaceLandmarker.create_from_options(_opts)

        # 15 Hz mouth-detection timer
        self.create_timer(1.0 / 15.0, self._detect_mouth, callback_group=self._cb)

        banner('Kinova Click-to-Feed  v3  (MediaPipe + J4 Pre-Tilt)')
        spinner_msg('Waiting for trajectory controller …', C.CYAN)
        if not self._traj_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError('Timed out waiting for trajectory controller')
        cprint('Trajectory controller  ✓', C.GREEN, bold=True)
        cprint('\nSPACE = use detected mouth   Click = manual pixel   Q = quit', C.CYAN, bold=True)

    # ── Callbacks ────────────────────────────────────────────────────────────
    def _color_cb(self, msg):
        with self._lock:
            self._color_img = self._bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _depth_cb(self, msg):
        with self._lock:
            self._depth_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _js_cb(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._current_positions:
                    self._current_positions[name] = pos

    # ── MediaPipe mouth detection (15 Hz) ────────────────────────────────────
    def _detect_mouth(self):
        with self._lock:
            frame = self._color_img
        if frame is None:
            return

        h, w = frame.shape[:2]
        rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)

        self._mp_ts_ms += 67   # ~15 Hz
        result = self._face_det.detect_for_video(mp_img, self._mp_ts_ms)

        if result.face_landmarks:
            lms = result.face_landmarks[0]
            valid = [i for i in _MOUTH_IDX if i < len(lms)]
            mx = np.mean([lms[i].x for i in valid])
            my = np.mean([lms[i].y for i in valid])
            with self._lock:
                self._mouth_pixel = (int(mx * w), int(my * h))
                self._mouth_lms   = lms
                self._mouth_hw    = (h, w)
        else:
            with self._lock:
                self._mouth_pixel = None
                self._mouth_lms   = None
                self._mouth_hw    = None

    # ── Click / space handling ────────────────────────────────────────────────
    def on_click(self, u: int, v: int):
        if self._busy:
            cprint('Arm busy — input ignored.', C.RED)
            return
        self._start_pipeline(u, v, source='click')

    def on_space(self):
        """Use the currently detected mouth centre."""
        if self._busy:
            cprint('Arm busy — input ignored.', C.RED)
            return
        with self._lock:
            mp = self._mouth_pixel
        if mp is None:
            cprint('No mouth detected yet — position the patient in frame.', C.YELLOW)
            return
        self._start_pipeline(mp[0], mp[1], source='mouth')

    def _start_pipeline(self, u: int, v: int, source: str):
        with self._lock:
            depth_img = self._depth_img
        if depth_img is None:
            cprint('No depth frame yet.', C.YELLOW)
            return

        # ── Robust depth via median patch ────────────────────────────────────
        dh, dw = depth_img.shape[:2]
        r = DEPTH_PATCH_R
        patch = depth_img[max(0, v-r):min(dh, v+r+1),
                          max(0, u-r):min(dw, u+r+1)].astype(np.float32)
        valid = patch[patch > 0]
        if valid.size == 0:
            cprint('No depth at selected pixel (patch all zeros).', C.YELLOW)
            return
        raw = float(np.median(valid))
        if raw == 0:
            cprint('Zero depth — try clicking a closer region.', C.YELLOW)
            return

        h, w = depth_img.shape[:2]
        fx = FX_REF * (w / REF_W)
        fy = FY_REF * (h / REF_H)
        cx, cy = w / 2.0, h / 2.0

        z = raw * 0.001

        # Symmetric X re-mapping (handles non-square fields of view)
        x_left  = (0       - cx) * z / fx
        x_right = (w - 1   - cx) * z / fx
        half    = min(abs(x_left), x_right)
        x_raw   = (u - cx) * z / fx
        x = x_raw * half / (abs(x_left) if x_raw < 0 else x_right)
        y = (v - cy) * z / fy

        endpoint_cam = np.array([x, y - CAM_Y_OFFSET, z])

        # ── EE start in camera frame ─────────────────────────────────────────
        try:
            tf_ee_cam = self._tf_buf.lookup_transform(
                CAM_FRAME, EE_LINK, Time(), timeout=Duration(seconds=1.0))
        except Exception as e:
            err_row('TF lookup', str(e)); return

        startpoint = np.array([
            tf_ee_cam.transform.translation.x,
            tf_ee_cam.transform.translation.y,
            tf_ee_cam.transform.translation.z,
        ])

        dx = endpoint_cam[0] - startpoint[0]
        dy = endpoint_cam[1] - startpoint[1]
        dz = endpoint_cam[2] - startpoint[2]

        section('Target Summary', C.CYAN)
        row('Source',     source.upper())
        row('Pixel',      f'u={u}  v={v}')
        row('Depth',      f'{z:.3f} m')
        row('Target cam', f'X={endpoint_cam[0]:+.3f}  Y={endpoint_cam[1]:+.3f}  Z={endpoint_cam[2]:+.3f}', C.GREEN)
        row('EE start',   f'X={startpoint[0]:+.3f}  Y={startpoint[1]:+.3f}  Z={startpoint[2]:+.3f}')
        row('Δ error',    f'dx={dx:+.3f}  dy={dy:+.3f}  dz={dz:+.3f}',
            C.RED if max(abs(dx), abs(dy), abs(dz)) > 0.3 else C.YELLOW)
        sep()

        ans = input(f'\n  {C.BOLD}{C.WHITE}Proceed to this target? {C.DIM}[y/N]{C.RESET}: ').strip().lower()
        if ans != 'y':
            cprint('Skipped.', C.DIM)
            return

        print(f"""
  {C.CYAN}{C.BOLD}╔{'═'*44}╗
  ║   Select Motion Planning Mode             ║
  ╠{'═'*44}╣
  ║   {C.GREEN}[1]{C.CYAN}  Custom Algorithm  {C.DIM}(Phased IK)      {C.CYAN} ║
  ║   {C.MAGENTA}[2]{C.CYAN}  BFMT Planner      {C.DIM}(MoveIt2)        {C.CYAN} ║
  ╚{'═'*44}╝{C.RESET}""")

        mode = input(f'  {C.BOLD}Choice [1/2]{C.RESET}: ').strip()

        if mode == '2':
            spinner_msg('Launching BFMT planner …', C.MAGENTA)
            threading.Thread(target=self._perform_moveit,
                             args=(endpoint_cam,), daemon=True).start()
        else:
            spinner_msg('Launching custom phased algorithm …', C.GREEN)
            threading.Thread(target=self._perform_custom,
                             args=(startpoint, endpoint_cam), daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  MODE 1 — Custom phased algorithm
    # ══════════════════════════════════════════════════════════════════════════
    def _perform_custom(self, startpoint: np.ndarray, endpoint: np.ndarray):
        self._busy = True
        try:
            banner('Custom Algorithm — Phased IK  (Feeding Mode)')
            currpoint = startpoint.copy()
            dx = endpoint[0] - currpoint[0]
            dy = endpoint[1] - currpoint[1]
            dz = endpoint[2] - currpoint[2]
            row('Start',  f'({startpoint[0]:+.3f}, {startpoint[1]:+.3f}, {startpoint[2]:+.3f})')
            row('Target', f'({endpoint[0]:+.3f}, {endpoint[1]:+.3f}, {endpoint[2]:+.3f})', C.GREEN)

            # ── Phase 1 · Y axis → J2 ────────────────────────────────────────
            section('Phase 1 · Y  (J2 shoulder pitch)', C.YELLOW)
            n = 0
            while abs(dy) > ERR_TOL_M:
                frac = min(abs(dy) / CART_DELTA_Y, 1.0)
                step = frac * STEP_RADS_Y
                if dy > 0:
                    self.decrease_j2(step); currpoint[1] += frac * CART_DELTA_Y
                    row(f'  step {n:02d}', f'dy={dy:+.3f}  → J2 ▼  ({step:.4f} rad)', C.CYAN)
                else:
                    self.increase_j2(step); currpoint[1] -= frac * CART_DELTA_Y
                    row(f'  step {n:02d}', f'dy={dy:+.3f}  → J2 ▲  ({step:.4f} rad)', C.CYAN)
                time.sleep(TICK_Y_S)
                dy = endpoint[1] - currpoint[1]
                n += 1
            done_row('Phase 1', f'{n} steps  residual dy={dy:+.4f}')
            sep()

            # ── Phase 2 · X axis → J3 ────────────────────────────────────────
            section('Phase 2 · X  (J3 elbow pitch)', C.YELLOW)
            n = 0
            while abs(dx) > ERR_TOL_M:
                frac = min(abs(dx) / CART_DELTA_X, 1.0)
                step = frac * STEP_RADS_X
                if dx < 0:
                    self.decrease_j3(step * 0.8); currpoint[0] -= frac * CART_DELTA_X
                    row(f'  step {n:02d}', f'dx={dx:+.3f}  → J3 ◄  ({step*0.8:.4f} rad)', C.CYAN)
                else:
                    self.increase_j3(step * 0.8); currpoint[0] += frac * CART_DELTA_X
                    row(f'  step {n:02d}', f'dx={dx:+.3f}  → J3 ►  ({step*0.8:.4f} rad)', C.CYAN)
                time.sleep(TICK_X_S)
                dx = endpoint[0] - currpoint[0]
                n += 1
            done_row('Phase 2', f'{n} steps  residual dx={dx:+.4f}')
            sep()

            # ── Phase 3 · Z reach — J4 pre-tilt, then J1 sweep ───────────────
            section('Phase 3 · Z reach  (J4 pre-tilt, then J1 sweep, J5 locked)', C.YELLOW)
            with self._lock:
                j5_hold = self._current_positions['j2s6s200_joint_5']

            # Calculate total expected movements based on remaining Z distance
            total_j1_expected = (dz / CART_DELTA_Z) * STEP_J1_RAD
            total_j4_expected = total_j1_expected * J4_PER_J1

            row('J5 lock', f'{j5_hold:.4f} rad  (spoon roll locked)', C.MAGENTA)
            row('J4 pre-tilt target', f'{total_j4_expected:+.4f} rad', C.MAGENTA)

            # --- STEP 1: Move J4 entirely first ---
            n_j4 = 0
            j4_remaining = total_j4_expected
            # Keep J4 motion smooth by limiting its step size per tick
            max_j4_step = STEP_J1_RAD * abs(J4_PER_J1)

            while abs(j4_remaining) > 0.001:
                # Determine direction and magnitude for this tick
                step_sign = 1.0 if j4_remaining > 0 else -1.0
                j4_step = step_sign * min(abs(j4_remaining), max_j4_step)

                row(f'  pre-tilt {n_j4:02d}', 
                    f'J4Δ={j4_step:+.4f}  (remaining: {j4_remaining-j4_step:+.4f})', 
                    C.CYAN)

                # Dispatch only J4; J1 is 0.0, J5 is held
                self._dispatch_phase3_step(0.0, j4_step*5.2, j5_hold)
                
                j4_remaining -= j4_step
                time.sleep(TICK_Z_S)
                
                # Refresh J5 lock against gravity
                with self._lock:
                    j5_hold = self._current_positions['j2s6s200_joint_5']
                n_j4 += 1

            done_row('J4 Pre-tilt', f'complete in {n_j4} steps')
            sep()

            # --- STEP 2: Sweep J1 to reach Z target ---
            n_j1 = 0
            total_j1 = 0.0

            while dz > ERR_TOL_M:
                # Fractional step — slow down as we approach the mouth
                frac = min(dz / (CART_DELTA_Z * 3.0), 1.0)
                j1_step = frac * STEP_J1_RAD

                row(f'  reach {n_j1:02d}',
                    f'dz={dz:+.3f}  J1Δ={j1_step:+.4f}  J4Δ=0.0000  J5={j5_hold:.3f}',
                    C.CYAN)

                # Dispatch only J1; J4 is already positioned, J5 is held
                self._dispatch_phase3_step(j1_step*3.8, 0.0, j5_hold)
                
                currpoint[2] += frac * CART_DELTA_Z
                total_j1 += j1_step
                time.sleep(TICK_Z_S)

                dz = endpoint[2] - currpoint[2]
                with self._lock:
                    j5_hold = self._current_positions['j2s6s200_joint_5']
                n_j1 += 1

            done_row('Phase 3', f'{n_j1} steps  total J1={total_j1:+.4f} rad  residual dz={dz:+.4f}')
            sep()

            print(f'\n  {C.GREEN}{C.BOLD}🍴  Fed the patient!{C.RESET}\n')
            # section('Phase 3 · Step 3 (J4 Return / Tip Bowl)', C.YELLOW)
            
            # # The exact inverse of the pre-tilt
            # j4_return_delta = -total_j4_expected
            
            # if abs(j4_return_delta) <= 0.05:
            #     done_row('J4 Return', f'Delta {j4_return_delta:+.4f} rad is <= 0.05. Skipping.')
            # else:
            #     row('J4 Return Target', f'{j4_return_delta:+.4f} rad (undoing pre-tilt)', C.MAGENTA)
                
            #     n_j4_ret = 0
            #     j4_rem = j4_return_delta
                
            #     while abs(j4_rem) > 0.001:
            #         step_sign =-1.0 if j4_rem > 0 else -1.0
            #         j4_step = step_sign * min(abs(j4_rem), max_j4_step)
                    
            #         row(f'  return {n_j4_ret:02d}', 
            #             f'J4Δ={j4_step:+.4f}  (remaining: {j4_rem-j4_step:+.4f})', 
            #             C.CYAN)

            #         self._dispatch_phase3_step(0.0, j4_step * 5.2, j5_hold)
                    
            #         j4_rem -= j4_step
            #         time.sleep(TICK_Z_S)
                    
            #         with self._lock:
            #             j5_hold = self._current_positions['j2s6s200_joint_5']
            #         n_j4_ret += 1

            #     done_row('J4 Return', f'complete in {n_j4_ret} steps')
            # sep()

            # print(f'\n  {C.GREEN}{C.BOLD}🍴  Fed the patient!{C.RESET}\n')
            self.decrease_j5(0.2)  # open fingers to tip bowl
        except Exception as e:
            err_row('Custom algorithm error', str(e))
            import traceback; traceback.print_exc()
        finally:
            self._busy = False

    # ══════════════════════════════════════════════════════════════════════════
    #  MODE 2 — BFMT via MoveIt2
    # ══════════════════════════════════════════════════════════════════════════
    def _perform_moveit(self, endpoint_cam: np.ndarray):
        self._busy = True
        joint_traj = None
        try:
            banner('BFMT Planner  (MoveIt2 — shortest path)')

            spinner_msg('TF: camera → base …', C.CYAN)
            try:
                tf_cb = self._tf_buf.lookup_transform(
                    BASE_FRAME, CAM_FRAME, Time(), timeout=Duration(seconds=2.0))
            except Exception as e:
                err_row('TF lookup failed', str(e)); return

            R = Rotation.from_quat([
                tf_cb.transform.rotation.x, tf_cb.transform.rotation.y,
                tf_cb.transform.rotation.z, tf_cb.transform.rotation.w,
            ]).as_matrix()
            t_vec = np.array([tf_cb.transform.translation.x,
                              tf_cb.transform.translation.y,
                              tf_cb.transform.translation.z])
            target_base = R @ endpoint_cam + t_vec
            done_row('Target (base)',
                     f'X={target_base[0]:+.3f}  Y={target_base[1]:+.3f}  Z={target_base[2]:+.3f}')

            prim = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[MOVEIT_TOL_M])
            bv   = BoundingVolume(
                primitives=[prim],
                primitive_poses=[Pose(position=Point(
                    x=target_base[0], y=target_base[1], z=target_base[2]))])
            pos_c = PositionConstraint(
                header=self._tf_buf.lookup_transform(BASE_FRAME, CAM_FRAME, Time()).header,
                link_name=EE_LINK, constraint_region=bv, weight=1.0)
            pos_c.header.frame_id = BASE_FRAME
            goal_c = Constraints(position_constraints=[pos_c])

            req = MotionPlanRequest()
            req.group_name                      = MOVEIT_GROUP
            req.planner_id                      = MOVEIT_PLANNER
            req.allowed_planning_time           = MOVEIT_PLAN_TIME
            req.num_planning_attempts           = MOVEIT_ATTEMPTS
            req.max_velocity_scaling_factor     = MOVEIT_VEL_SCALE
            req.max_acceleration_scaling_factor = MOVEIT_ACC_SCALE
            req.goal_constraints                = [goal_c]

            start_state = RobotState()
            start_state.joint_state.header.stamp = self.get_clock().now().to_msg()
            start_state.joint_state.name = list(ARM_JOINT_NAMES)
            with self._lock:
                start_state.joint_state.position = [
                    float(self._current_positions[n]) for n in ARM_JOINT_NAMES]
            req.start_state = start_state

            row('Planner',   MOVEIT_PLANNER, C.MAGENTA)
            row('Plan time', f'{MOVEIT_PLAN_TIME:.0f} s')
            row('Velocity',  f'{MOVEIT_VEL_SCALE*100:.0f}%')
            row('Tolerance', f'{MOVEIT_TOL_M*100:.0f} mm sphere')

            spinner_msg('Waiting for /plan_kinematic_path …', C.MAGENTA)
            mini = rclpy.create_node('cp_bfmt_mini')
            try:
                plan_cli = mini.create_client(GetMotionPlan, '/plan_kinematic_path')
                from rclpy.executors import SingleThreadedExecutor as _STE
                mini_exec = _STE()
                mini_exec.add_node(mini)

                deadline = time.time() + 30.0
                while not plan_cli.service_is_ready():
                    if time.time() > deadline:
                        err_row('MoveIt', '/plan_kinematic_path not available'); return
                    mini_exec.spin_once(timeout_sec=0.5)

                done_row('Planning service', 'connected')
                spinner_msg('BFMT planning …', C.MAGENTA)

                plan_req = GetMotionPlan.Request()
                plan_req.motion_plan_request = req
                future = plan_cli.call_async(plan_req)
                mini_exec.spin_until_future_complete(future, timeout_sec=MOVEIT_PLAN_TIME + 5.0)

                if not future.done():
                    err_row('Planning', 'timed out'); return

                plan_resp = future.result().motion_plan_response
                if plan_resp.error_code.val != MoveItErrorCodes.SUCCESS:
                    err_row('Planning failed', f'MoveIt code {plan_resp.error_code.val}'); return

                joint_traj = plan_resp.trajectory.joint_trajectory
                done_row('Path found', f'{len(joint_traj.points)} waypoints')
            finally:
                mini_exec.shutdown(timeout_sec=1.0)
                mini.destroy_node()

            if not joint_traj or not joint_traj.points:
                err_row('Planning', 'empty trajectory'); return

            spinner_msg('Executing trajectory …', C.MAGENTA)
            fj_goal = FollowJointTrajectory.Goal()
            fj_goal.trajectory = joint_traj
            exec_future = self._traj_client.send_goal_async(fj_goal)
            while not exec_future.done():
                time.sleep(0.05)

            gh = exec_future.result()
            if not gh.accepted:
                err_row('Execution', 'trajectory goal rejected'); return

            result_future = gh.get_result_async()
            while not result_future.done():
                time.sleep(0.1)

            if result_future.result().result.error_code == 0:
                print(f'\n  {C.GREEN}{C.BOLD}🍴  BFMT execution complete — fed the patient!{C.RESET}\n')
            else:
                err_row('Execution failed', f'code {result_future.result().result.error_code}')

        except Exception as e:
            err_row('MoveIt error', str(e))
            import traceback; traceback.print_exc()
        finally:
            self._busy = False

    # ══════════════════════════════════════════════════════════════════════════
    #  Joint helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _dispatch_phase3_step(self, j1_delta: float, j4_delta: float, j5_lock: float):
        """
        Send ONE smooth trajectory segment:
          • J1 += j1_delta    (base yaw, advances Z)
          • J4 += j4_delta    (wrist pitch correction, keeps spoon level)
          • J5  = j5_lock     (absolute — fights roll drift)
          All other joints held at current position.

        Two waypoints (current → target) with TICK_Z_S duration give the
        trajectory controller enough information to interpolate smoothly.
        A third mid-point could be added here for even finer blending if
        the arm still feels jerky on your hardware.
        """
        with self._lock:
            cur = [self._current_positions[n] for n in ARM_JOINT_NAMES]

        tgt = list(cur)
        tgt[0] += j1_delta     # J1  base yaw
        tgt[3] += j4_delta     # J4  wrist pitch counter-rotation
        tgt[4]  = j5_lock      # J5  absolute roll lock

        dur_ns = int(TICK_Z_S * 1e9)

        pt0 = JointTrajectoryPoint(
            positions=cur,
            time_from_start=DurationMsg(sec=0, nanosec=0))

        # Optional: add a mid-point at half duration for smoother interpolation
        mid = [(a + b) / 2 for a, b in zip(cur, tgt)]
        pt_mid = JointTrajectoryPoint(
            positions=mid,
            time_from_start=DurationMsg(sec=0, nanosec=dur_ns // 2))

        pt1 = JointTrajectoryPoint(
            positions=tgt,
            time_from_start=DurationMsg(sec=0, nanosec=dur_ns))

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = ARM_JOINT_NAMES
        traj.points       = [pt0, pt_mid, pt1]   # 3-point for smooth interpolation

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self._traj_client.send_goal_async(goal)

    def _dispatch_step(self, joint_idx: int, delta_rad: float):
        with self._lock:
            cur = [self._current_positions[n] for n in ARM_JOINT_NAMES]
        tgt = list(cur)
        tgt[joint_idx] += delta_rad
        dur_ns = int(0.30 * 1e9)

        pt0 = JointTrajectoryPoint(
            positions=cur, time_from_start=DurationMsg(sec=0, nanosec=0))
        pt1 = JointTrajectoryPoint(
            positions=tgt, time_from_start=DurationMsg(sec=0, nanosec=dur_ns))

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = ARM_JOINT_NAMES
        traj.points       = [pt0, pt1]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self._traj_client.send_goal_async(goal)

    def increase_j1(self, s): self._dispatch_step(0,  abs(s))
    def decrease_j1(self, s): self._dispatch_step(0, -abs(s))
    def increase_j2(self, s): self._dispatch_step(1,  abs(s))
    def decrease_j2(self, s): self._dispatch_step(1, -abs(s))
    def increase_j3(self, s): self._dispatch_step(2,  abs(s))
    def decrease_j3(self, s): self._dispatch_step(2, -abs(s))
    def increase_j4(self, s): self._dispatch_step(3,  abs(s))
    def decrease_j4(self, s): self._dispatch_step(3, -abs(s))
    def increase_j5(self, s): self._dispatch_step(4,  abs(s))
    def decrease_j5(self, s): self._dispatch_step(4, -abs(s))
    def increase_j6(self, s): self._dispatch_step(5,  abs(s))
    def decrease_j6(self, s): self._dispatch_step(5, -abs(s))

    def get_frame(self):
        with self._lock:
            return self._color_img, self._mouth_pixel, getattr(self, '_mouth_lms', None), getattr(self, '_mouth_hw', None)


# ══════════════════════════════════════════════════════════════════════════════
#  OpenCV UI
# ══════════════════════════════════════════════════════════════════════════════
def run_ui(node: ClickPointerNode):
    WIN = 'Click-to-Feed  |  SPACE=use mouth  Q=quit'
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, lambda e, u, v, *_:
                         node.on_click(u, v) if e == cv2.EVENT_LBUTTONDOWN else None)

    while rclpy.ok():
        frame, mouth_px, mouth_lms, mouth_hw = node.get_frame()
        if frame is not None:
            disp = frame.copy()
            h, w = disp.shape[:2]

            # ── Draw lip outline if mouth detected ──────────────────────────
            if mouth_lms is not None and mouth_hw is not None:
                mh, mw = mouth_hw
                for a, b in _LIP_CONN:
                    if a < len(mouth_lms) and b < len(mouth_lms):
                        p1 = (int(mouth_lms[a].x * mw), int(mouth_lms[a].y * mh))
                        p2 = (int(mouth_lms[b].x * mw), int(mouth_lms[b].y * mh))
                        cv2.line(disp, p1, p2, (0, 220, 255), 1)

            # ── Mouth target circle ─────────────────────────────────────────
            if mouth_px is not None:
                mu, mv = mouth_px
                cv2.circle(disp, (mu, mv), 10, (0, 255, 80),  -1)
                cv2.circle(disp, (mu, mv), 16, (0, 220, 255),  2)
                cv2.putText(disp, 'MOUTH', (mu + 20, mv - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)
                cv2.putText(disp, 'SPACE to feed',
                            (mu + 20, mv + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

            # ── Status pill ─────────────────────────────────────────────────
            if node._busy:
                label, bg, fg = 'FEEDING…', (0, 0, 180), (255, 255, 255)
            elif mouth_px is not None:
                label, bg, fg = 'Mouth detected — SPACE or click', (0, 100, 0), (255, 255, 255)
            else:
                label, bg, fg = 'No face — click to aim manually', (60, 40, 0), (220, 200, 100)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(disp, (6, 6), (tw + 20, th + 20), bg, -1)
            cv2.putText(disp, label, (12, th + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2)

            # ── Crosshair ───────────────────────────────────────────────────
            if not node._busy:
                cv2.line(disp, (w//2 - 14, h//2), (w//2 + 14, h//2), (0, 255, 200), 1)
                cv2.line(disp, (w//2, h//2 - 14), (w//2, h//2 + 14), (0, 255, 200), 1)

            cv2.imshow(WIN, disp)

        key = cv2.waitKey(33)
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            threading.Thread(target=node.on_space, daemon=True).start()

    cv2.destroyAllWindows()
    rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ClickPointerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        run_ui(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()