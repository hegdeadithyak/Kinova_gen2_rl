#!/usr/bin/env python3
"""Image-based path planner: centre a bounding box, then approach in depth.

Works for ANY arm configuration (any direction the wrist camera is facing),
because the joint->camera mapping is rebuilt every step from the LIVE TF tree
that the real robot driver publishes — not from a fixed/simulated kinematics
model.  This is the key difference from click_pointer.py, whose hand-tuned
J2/J3/J1+J4 mapping is only valid at the narrow feeding pose.

How it works each control step
------------------------------
1. TARGET     A bounding box (from a detector topic, or a manually drawn box
              tracked with OpenCV) gives a centre pixel (u, v).  The aligned
              depth image gives metric depth Z at that pixel.

2. BACK-PROJECT   Pinhole model -> target position in the camera optical frame:
                  x_cam = (u-cx)*Z/fx,  y_cam = (v-cy)*Z/fy,  z_cam = Z.

3. DESIRED MOVE   To put the target on the optical axis at the standoff depth
                  s* = (0, 0, DESIRED_Z), the camera must translate (in its own
                  optical frame) by
                      d_cam = (x_cam, y_cam, w_z * (z_cam - DESIRED_Z)).
                  w_z gates the forward (depth) term to ~0 until the box is
                  centred  ->  "centre first, then approach".  If the box drifts
                  off-centre during approach, w_z drops and centring re-engages.
                  The step is capped so the box can never jump out of frame, and
                  forward motion is disabled when the box nears any frame edge.

4. LIVE JACOBIAN  From TF (base->link_i, base->camera) build the geometric
                  Jacobian of the camera ORIGIN in the base frame:
                      J[:, i] = z_i x (p_cam - p_i)
                  where z_i is joint i's rotation axis (every Kinova joint spins
                  about its child-link z) and p_i its origin, both from TF.

5. SOLVE          Rotate the desired move into the base frame with the live
                  camera orientation (also from TF), then damped least-squares:
                      d_base = R_base_cam @ d_cam
                      dq     = GAIN * J^T (J J^T + lambda^2 I)^-1 d_base
                  Whatever joints actually move the camera in THIS pose are the
                  ones that get used — fully pose-independent, signs automatic.

6. ACTUATE        Clamp dq, clamp target to joint limits, and send a blocking
                  FollowJointTrajectory goal (position control, same controller
                  click_pointer.py uses).  Re-observe, repeat until converged.

Bounding-box source (pick one)
  * Topic:  /target_bbox  (std_msgs/Int32MultiArray, data = [x1, y1, x2, y2]).
  * Manual: drag a rectangle in the window; an OpenCV tracker follows it as the
            camera moves (re-localisation needed for the closed loop).

Keys (in the OpenCV window)
  drag-LMB  draw a box        ENTER / s  start servoing
  c         cancel / clear    q / ESC    quit
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

# Reuse the exact MediaPipe face mesh + lip landmarks that click_pointer.py uses.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src", "click_pointer", "scripts"))
from face_overlay import build_face_mesh, MOUTH_CENTER_IDS, LIP_CONN  # noqa: E402

# Every unique landmark index on the lip contour -> used to bound the mouth box.
MOUTH_LIP_IDS = sorted({i for pair in LIP_CONN for i in pair})

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Int32MultiArray
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

os.environ.setdefault("QT_LOGGING_RULES", "qt.*=false")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

# ── Frames (real TF names, from j2s6s200 URDF) ───────────────────────────────
BASE_FRAME = "j2s6s200_link_base"
LINK_FRAMES = [f"j2s6s200_link_{i}" for i in range(1, 7)]   # one per joint
CAM_FRAME = "camera_color_optical_frame"

ARM_JOINT_NAMES = [f"j2s6s200_joint_{i}" for i in range(1, 7)]
NJ = 6

ACTION_TOPIC = "/arm_controller/follow_joint_trajectory"
# Use the DRIVER's joint feedback (native convention, no wrap/relay mismatch) —
# same topic the tested teleop uses.  /joint_states can be stale/re-wrapped,
# which corrupts continuous joints (J1/J4/J6) -> the J4 fly-out.
JOINT_STATE_TOPIC = "/j2s6s200_driver/out/joint_state"
BBOX_TOPIC = "/target_bbox"

# RealSense publishes images as BEST_EFFORT sensor data — a default (RELIABLE)
# subscription receives NOTHING from it (rqt works because it auto-adapts).
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

# Joint limits (rad) — from kinova_env_config.yaml / URDF.  Continuous joints
# (1, 4, 6) are effectively unbounded; we still clamp at +/-2pi as a guard.
JOINT_POS_LIMITS = np.array([
    [-6.283185307, 6.283185307],
    [0.820304748, 5.462880558],
    [0.331612558, 5.951572749],
    [-6.283185307, 6.283185307],
    [0.523598776, 5.759586532],
    [-6.283185307, 6.283185307],
])
JOINT_VEL_LIMITS = np.array([0.6283, 0.6283, 0.6283, 0.8378, 0.8378, 0.8378])
# Joints 1, 4, 6 are CONTINUOUS — the driver reports them as accumulated angles
# that can sit outside +/-2pi, so clipping them to +/-2pi causes a huge jump
# (the J4 "jump").  Only joints 2, 3, 5 have real position limits to clip.
CONTINUOUS_IDX = (0, 3, 5)

# Fallback intrinsics (overwritten by camera_info) — calibrated in click_pointer.
FX_DEF, FY_DEF = 603.6312, 603.0632
CX_DEF, CY_DEF = 319.0870, 236.3678
W_DEF, H_DEF = 640, 480

# ── Controller tuning ────────────────────────────────────────────────────────
# Centring error is in NORMALISED image coords x=(u-cx)/fx: 0 on the optical
# axis, ~±0.4-0.7 at the frame edges.  Depth is in metres.
DESIRED_Z = 0.20          # m  — final camera->target standoff
# End-effector offset from the depth camera, in the CAMERA optical frame (metres):
# +x = image-right, +y = image-down, +z = forward.  The EE is 5 cm BELOW the
# camera -> +y.  The target is servo'd to where the EE (not the lens) lines up,
# i.e. it ends up ~5 cm-equivalent below image centre.  Flip the sign if your
# image is mounted upside-down.
EE_OFFSET_X = 0.00        # m
EE_OFFSET_Y = 0.05        # m  (EE 5 cm below camera)
CENTER_TOL_N = 0.035      # norm — centred when hypot(x,y) below this (~3% frame)
CENTER_ENGAGE_N = 0.10    # norm — above this centring error, forward motion = 0
Z_TOL = 0.020             # m  — depth error counted as converged
EDGE_FRAC = 0.10          # box within this frac of any border -> no forward step
GAIN = 0.5                # closed-loop gain on the feature error (<1, stable)
DLS_LAMBDA = 0.08         # damped least-squares regularisation (rad-scale)
DS_MAX_XY = 0.06          # norm — max image-feature move per step (keeps in frame)
DS_MAX_Z = 0.030          # m  — max depth move commanded per step
MAX_DQ = 0.12             # rad — hard cap on per-joint motion per step
MIN_DUR = 1.0             # s  — trajectory floor duration
VEL_FRAC = 0.5            # fraction of joint vel limit used to time a step
SETTLE_S = 0.30           # s  — settle/re-observe pause after each step
DEPTH_WIN = 4             # px half-window for robust median depth
MAX_LOST = 40             # consecutive bad observations before aborting
MOUTH_PAD = 6             # px padding around the lip contour for the mouth box
BBOX_STALE_S = 0.7        # s — topic/mouth detection older than this -> hold

# On-robot calibration: probe each joint and MEASURE how the image feature
# s=[x,y,Z] responds.  This needs no TF / hand-eye / URDF, so it is immune to a
# bad camera-orientation calibration (the cause of the wrist fly-out).
# The probe amplitude is chosen ADAPTIVELY per joint: sensitive joints (e.g. the
# wrist J4, which rotates the camera) get a small jog, slow joints a larger one,
# so no single probe ever sweeps the target out of frame.
CALIB_DELTA_START = 0.025  # rad — initial probe amplitude
CALIB_DELTA_MAX   = 0.08   # rad — upper bound on probe amplitude
CALIB_DELTA_MIN   = 0.005  # rad — lower bound (very sensitive joints)
CALIB_IMG_TARGET  = 0.06   # norm — desired image-feature move per probe
CALIB_IMG_MAX     = 0.12   # norm — too big -> shrink the jog (avoid leaving frame)
CALIB_IMG_MIN     = 0.02   # norm — too small -> grow the jog (better signal)
CALIB_SETTLE = 0.5         # s   — settle before sampling features after a move
CALIB_FRAMES = 6           # feature samples median-averaged per measurement

# Record-and-fit calibration: jog the arm around (you keep the target in view),
# record (joint_state, feature) pairs, then least-squares fit J from the deltas.
# Robust for roll-heavy joints (J4) because YOU control the motion.
JOG_STEP        = 0.030    # rad — per key-press jog of the selected joint
MIN_FIT_SAMPLES = 8        # samples needed before a Jacobian fit is allowed


def damped_pinv_solve(J: np.ndarray, e: np.ndarray, lam: float) -> np.ndarray:
    """Damped least-squares: dq = J^T (J J^T + lam^2 I)^-1 e  (min-norm, stable)."""
    m = J.shape[0]
    return J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(m), e)


class BBoxPlanner(Node):
    def __init__(self, desired_z: float, source: str = "mouth",
                 ee_off_x: float = EE_OFFSET_X, ee_off_y: float = EE_OFFSET_Y):
        super().__init__("bbox_planner")
        self.desired_z = desired_z
        self.source = source                  # "mouth" | "drag" | "topic"
        self.ee_off_x = ee_off_x              # EE-vs-camera offset (cam frame, m)
        self.ee_off_y = ee_off_y
        self.cb = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # MediaPipe face mesh for the mouth bounding box (same as click_pointer.py)
        self.mp_face = build_face_mesh() if source == "mouth" else None
        self.mouth_lms = None                 # latest lip landmarks (for the HUD)

        # measured image Jacobian (3x6: dq -> d[x,y,Z]); None until calibrated
        self.J_img = None
        self.calibrated = False
        self.save_calib_path = None
        self.calib_samples = []               # (q, s) pairs for record-and-fit
        self.jog_joint = 0                     # active joint for keyboard jogging

        # camera / intrinsics
        self.color = None
        self.depth = None
        self.fx, self.fy, self.cx, self.cy = FX_DEF, FY_DEF, CX_DEF, CY_DEF
        self.img_w, self.img_h = W_DEF, H_DEF
        self.got_info = False

        # joint state
        self.q = None

        # target bounding box (x1, y1, x2, y2) in full-frame pixels + timestamp
        self.bbox = None
        self.bbox_t = 0.0
        self.bbox_src = None              # "mouth" | "topic" | "drag"

        # control state (for HUD)
        self.busy = False
        self.servo_on = False
        self.phase = "idle"
        self.last_info = ""

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, "/camera/camera/color/image_raw",
                                 self._color_cb, SENSOR_QOS, callback_group=self.cb)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw",
                                 self._depth_cb, SENSOR_QOS, callback_group=self.cb)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info",
                                 self._info_cb, SENSOR_QOS, callback_group=self.cb)
        self.create_subscription(JointState, JOINT_STATE_TOPIC,
                                 self._js_cb, 10, callback_group=self.cb)
        self.create_subscription(Int32MultiArray, BBOX_TOPIC,
                                 self._bbox_cb, 10, callback_group=self.cb)

        self.marker_pub = self.create_publisher(Marker, "/target_marker", 10)
        self.traj_client = ActionClient(self, FollowJointTrajectory, ACTION_TOPIC,
                                        callback_group=self.cb)

    # ── subscribers ──────────────────────────────────────────────────────────
    def _color_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        box, lms = (None, None)
        if self.source == "mouth":
            box, lms = self._detect_mouth(frame)
        with self.lock:
            self.color = frame
            self.img_h, self.img_w = frame.shape[:2]
            if self.source == "mouth":
                self.mouth_lms = lms
                if box is not None:
                    self.bbox = box
                    self.bbox_t = time.monotonic()
                    self.bbox_src = "mouth"

    def _detect_mouth(self, frame):
        """Return (mouth_bbox (x1,y1,x2,y2), lip_landmarks) or (None, None)."""
        h, w = frame.shape[:2]
        res = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None, None
        lms = res.multi_face_landmarks[0].landmark
        n = len(lms)
        xs = [lms[i].x * w for i in MOUTH_LIP_IDS if i < n]
        ys = [lms[i].y * h for i in MOUTH_LIP_IDS if i < n]
        if len(xs) < 4:
            return None, None
        x1 = max(0, int(min(xs)) - MOUTH_PAD)
        y1 = max(0, int(min(ys)) - MOUTH_PAD)
        x2 = min(w, int(max(xs)) + MOUTH_PAD)
        y2 = min(h, int(max(ys)) + MOUTH_PAD)
        return (x1, y1, x2, y2), lms

    def _depth_cb(self, msg):
        with self.lock:
            self.depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def _info_cb(self, msg):
        if msg.k[0] > 0.0 and not self.got_info:
            with self.lock:
                self.fx, self.fy = float(msg.k[0]), float(msg.k[4])
                self.cx, self.cy = float(msg.k[2]), float(msg.k[5])
                self.got_info = True
            self.get_logger().info(
                f"camera_info fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx:.1f} cy={self.cy:.1f}")

    def _js_cb(self, msg):
        try:
            idx = [msg.name.index(j) for j in ARM_JOINT_NAMES]
        except ValueError:
            return
        with self.lock:
            self.q = np.array([msg.position[i] for i in idx], np.float64)

    def _bbox_cb(self, msg):
        d = list(msg.data)
        if len(d) >= 4 and self.source == "topic":
            with self.lock:
                self.bbox = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
                self.bbox_t = time.monotonic()
                self.bbox_src = "topic"

    # ── thread-safe getters / setters ────────────────────────────────────────
    def get_color(self):
        with self.lock:
            return None if self.color is None else self.color.copy()

    def set_bbox(self, box):
        with self.lock:
            self.bbox = box
            self.bbox_t = time.monotonic()
            self.bbox_src = "drag"

    def clear_bbox(self):
        with self.lock:
            self.bbox = None

    # ── observation: box centre + metric depth ───────────────────────────────
    def _sample_depth(self, u: int, v: int) -> float:
        with self.lock:
            depth = None if self.depth is None else self.depth
            if depth is None:
                return 0.0
            h, w = depth.shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                return 0.0
            x0, x1 = max(0, u - DEPTH_WIN), min(w, u + DEPTH_WIN + 1)
            y0, y1 = max(0, v - DEPTH_WIN), min(h, v + DEPTH_WIN + 1)
            patch = depth[y0:y1, x0:x1].astype(np.float32).ravel()
        valid = patch[patch > 0.0]
        if valid.size < 6:
            return 0.0
        return float(np.median(valid)) * 0.001   # mm -> m

    def observe(self):
        """Return (u, v, Z, box, q, intrinsics) or None if not ready."""
        with self.lock:
            box = self.bbox
            box_t = self.bbox_t
            q = None if self.q is None else self.q.copy()
            have_color = self.color is not None
            have_depth = self.depth is not None
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
            W, H = self.img_w, self.img_h
        # precise idle reasons (shown in HUD + logged) so failures are debuggable
        if not have_color:
            self.last_info = "NO camera frames — QoS/topic? check /camera/.../image_raw"
            return None
        if q is None:
            self.last_info = "NO /joint_states yet"
            return None
        if not have_depth:
            self.last_info = "NO depth frames yet"
            return None
        if box is None:
            self.last_info = ("NO mouth detected — get your face into the camera view"
                              if self.source == "mouth" else "no bbox received yet")
            return None
        # mouth / topic detections must be fresh; a dragged box is static.
        if self.bbox_src in ("mouth", "topic") and (time.monotonic() - box_t) > BBOX_STALE_S:
            self.last_info = "detection STALE — target lost"
            return None
        x1, y1, x2, y2 = box
        u = int(round((x1 + x2) / 2.0))
        v = int(round((y1 + y2) / 2.0))
        Z = self._sample_depth(u, v)
        return dict(u=u, v=v, Z=Z, box=box, q=q,
                    fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H)

    # ── live Jacobian of the camera origin (base frame) from TF ───────────────
    def _tf_pose(self, target_frame, source_frame):
        """Return (t[3], R[3x3]) of source expressed in target, or None."""
        try:
            tr = self.tf_buffer.lookup_transform(
                target_frame, source_frame, Time(), Duration(seconds=0.5))
        except Exception as e:
            self.last_info = f"TF {target_frame}<-{source_frame}: {e}"
            return None
        q = tr.transform.rotation
        t = tr.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ])
        return np.array([t.x, t.y, t.z]), R

    def camera_twist_jacobian(self):
        """6x6 geometric Jacobian of the CAMERA frame, expressed in the camera
        optical frame, built live from TF.

        Rows 0:3 = linear vel of the camera origin, rows 3:6 = angular vel.
        Column i (revolute joint about its child-link z axis z_i at p_i):
            linear  = z_i x (p_cam - p_i)
            angular = z_i
        Both parts are then rotated from base into the camera optical frame by
        R_cb = R_base_cam^T, so the result maps q_dot -> camera-frame twist.
        This is what makes wrist ROTATION (e.g. J4) correctly modelled — the
        thing the position-only version got catastrophically wrong.
        """
        cam = self._tf_pose(BASE_FRAME, CAM_FRAME)
        if cam is None:
            return None, None
        p_cam, R_bc = cam
        R_cb = R_bc.T
        J = np.zeros((6, NJ))
        for i, link in enumerate(LINK_FRAMES):
            pose = self._tf_pose(BASE_FRAME, link)
            if pose is None:
                return None, None
            p_i, R_i = pose
            z_i = R_i[:, 2]                          # joint axis (base frame)
            J[0:3, i] = R_cb @ np.cross(z_i, p_cam - p_i)   # linear  -> cam frame
            J[3:6, i] = R_cb @ z_i                          # angular -> cam frame
        return J, R_bc

    # ── feature extraction (measured image-space servoing) ───────────────────
    def _feature(self):
        """Current feature s=[x, y, Z]: normalised image coords + depth (m)."""
        obs = self.observe()
        if obs is None:
            return None
        if obs["Z"] <= 0.02:
            return None
        x = (obs["u"] - obs["cx"]) / obs["fx"]
        y = (obs["v"] - obs["cy"]) / obs["fy"]
        return np.array([x, y, obs["Z"]], np.float64)

    def _feature_median(self, n=CALIB_FRAMES):
        """Robust feature sample: median of several frames (rejects jitter)."""
        samples = []
        for _ in range(n * 4):
            f = self._feature()
            if f is not None:
                samples.append(f)
            if len(samples) >= n:
                break
            time.sleep(0.05)
        if len(samples) < max(3, n // 2):
            return None
        return np.median(np.array(samples), axis=0)

    def _step_dur(self, qa, qb):
        return max(MIN_DUR, float(np.max(np.abs(np.asarray(qb) - np.asarray(qa)) /
                                        (VEL_FRAC * JOINT_VEL_LIMITS))))

    @staticmethod
    def _clip_target(q):
        """Clip ONLY the revolute joints (2,3,5) to their limits; leave the
        continuous joints (1,4,6) alone so the driver's accumulated angle isn't
        snapped to +/-2pi (which causes the J4 jump)."""
        out = np.asarray(q, np.float64).copy()
        for i in range(NJ):
            if i not in CONTINUOUS_IDX:
                out[i] = float(np.clip(out[i], JOINT_POS_LIMITS[i, 0], JOINT_POS_LIMITS[i, 1]))
        return out

    def _probe(self, q0, j, delta):
        """Move joint j by delta from q0, settle, return the measured feature."""
        qj = q0.copy()
        qj[j] = qj[j] + delta
        qj = self._clip_target(qj)
        self.dispatch(qj, self._step_dur(q0, qj))
        time.sleep(CALIB_SETTLE)
        return self._feature_median()

    def _measure_side(self, q0, s0, j, sign):
        """Jog joint j in one direction, SHRINKING the amplitude until the target
        stays in view and the image move is bounded.  Never grows (a roll-heavy
        joint legitimately barely moves the centre — growing would roll it out).
        Returns (feature, signed_delta) or (None, None)."""
        delta = CALIB_DELTA_START
        for _ in range(5):
            s = self._probe(q0, j, sign * delta)
            if s is None:                                   # left view -> too big
                delta *= 0.5
                if delta < CALIB_DELTA_MIN:
                    return None, None
                continue
            d_img = float(np.linalg.norm(s[:2] - s0[:2]))
            if d_img > CALIB_IMG_MAX and delta > CALIB_DELTA_MIN:
                delta = max(CALIB_DELTA_MIN, delta * max(0.4, CALIB_IMG_TARGET / d_img))
                continue
            return s, sign * delta                          # in view and bounded
        return None, None

    def _calib_column(self, q0, s0, j):
        """Measure J[:,j] = d[x,y,Z]/dq_j from a two-sided probe (central diff),
        falling back to one-sided if only one direction stays in frame.
        Returns (column, amplitude) or (None, None)."""
        sp, dp = self._measure_side(q0, s0, j, +1)
        self.dispatch(q0, self._step_dur(q0, q0) + 0.3)     # recentre between sides
        time.sleep(CALIB_SETTLE)
        sm, dm = self._measure_side(q0, s0, j, -1)
        self.dispatch(q0, self._step_dur(q0, q0) + 0.3)     # back to start
        time.sleep(CALIB_SETTLE)
        if sp is not None and sm is not None:               # slope over (dp - dm), dm<0
            return (sp - sm) / (dp - dm), abs(dp)
        if sp is not None:
            return (sp - s0) / dp, abs(dp)
        if sm is not None:
            return (sm - s0) / dm, abs(dm)
        return None, None

    def calibrate(self) -> bool:
        """Probe all 6 joints and MEASURE the image Jacobian J (3x6).

        Needs a static target roughly centred in view.  Each joint is jogged
        +/-CALIB_DELTA and the feature response gives one Jacobian column —
        no TF, no hand-eye, no kinematics, so a bad camera-orientation
        calibration cannot corrupt it.
        """
        if self.busy:
            return False
        self.busy = True
        self.phase = "calibrating"
        try:
            if not self.traj_client.wait_for_server(timeout_sec=5.0):
                self.last_info = "calib: trajectory server unavailable"
                return False
            with self.lock:
                q0 = None if self.q is None else self.q.copy()
            if q0 is None:
                self.last_info = "calib: no /joint_states"
                return False
            if self._feature_median() is None:
                self.last_info = "calib: target not visible — centre a static target first"
                return False

            J = np.zeros((3, NJ))
            for j in range(NJ):
                self.last_info = f"calibrating joint {j+1}/6 ..."
                s0 = self._feature_median()
                if s0 is None:
                    self.last_info = (f"calib joint {j+1}: target lost — re-centre a "
                                      f"static target and retry")
                    self.get_logger().error(self.last_info)
                    return False
                col, delta = self._calib_column(q0, s0, j)
                if col is None:
                    self.last_info = (f"calib FAILED on joint {j+1} — keep the target "
                                      f"centred and in view, then retry")
                    self.get_logger().error(self.last_info)
                    return False
                J[:, j] = col
                self.last_info = f"calibrated joint {j+1}/6 (jog {delta:.3f} rad)"
                self.get_logger().info(
                    f"[calib] J[:,{j}] (d[x,y,Z]/dq) = {np.round(col, 4)}  jog={delta:.3f}")

            self.J_img = J
            self.calibrated = True
            self.get_logger().info(f"[calib] image Jacobian:\n{np.round(J, 4)}")
            self.last_info = "CALIBRATION COMPLETE — press ENTER to servo"
            if self.save_calib_path:
                np.savez(self.save_calib_path, J=J, q=q0, desired_z=self.desired_z)
                self.get_logger().info(f"[calib] saved -> {self.save_calib_path}")
            return True
        finally:
            self.busy = False
            self.phase = "idle"

    def load_calibration(self, path: str) -> bool:
        try:
            data = np.load(path)
            self.J_img = np.array(data["J"], np.float64)
            self.calibrated = True
            self.get_logger().info(f"[calib] loaded {path}\n{np.round(self.J_img, 4)}")
            return True
        except Exception as e:
            self.get_logger().error(f"[calib] load failed: {e}")
            return False

    # ── record-and-fit calibration (you move the arm, keeping target in view) ─
    def record_sample(self) -> bool:
        """Snapshot the current (joint_state, feature) pair."""
        f = self._feature()
        with self.lock:
            q = None if self.q is None else self.q.copy()
        if f is None or q is None:
            self.last_info = "record: no valid target/feature in view"
            return False
        self.calib_samples.append((q, f))
        self.last_info = f"recorded sample {len(self.calib_samples)} (need >= {MIN_FIT_SAMPLES})"
        self.get_logger().info(self.last_info)
        return True

    def clear_samples(self):
        self.calib_samples = []
        self.last_info = "cleared recorded samples"

    def jog(self, sign: int):
        """Jog the selected joint by +/-JOG_STEP, then auto-record a sample."""
        if self.busy:
            return
        with self.lock:
            q = None if self.q is None else self.q.copy()
        if q is None:
            return
        self.busy = True
        try:
            j = self.jog_joint
            tgt = q.copy()
            tgt[j] = tgt[j] + sign * JOG_STEP
            tgt = self._clip_target(tgt)
            self.dispatch(tgt, self._step_dur(q, tgt))
            time.sleep(0.3)
        finally:
            self.busy = False
        self.record_sample()                       # capture the new pose

    def fit_jacobian(self) -> bool:
        """Least-squares fit J (3x6) from recorded (q, s): solve dS = dQ J^T."""
        n = len(self.calib_samples)
        if n < MIN_FIT_SAMPLES:
            self.last_info = f"need >= {MIN_FIT_SAMPLES} samples (have {n}); jog + record more"
            return False
        Q = np.array([q for q, _ in self.calib_samples])
        S = np.array([s for _, s in self.calib_samples])
        dQ = Q - Q.mean(axis=0)
        dS = S - S.mean(axis=0)
        spread = dQ.std(axis=0)
        weak = [i + 1 for i, v in enumerate(spread) if v < 0.008]   # barely-moved joints
        JT, _res, rank, sv = np.linalg.lstsq(dQ, dS, rcond=None)
        if rank < NJ:
            self.last_info = (f"fit underdetermined (rank {rank}/{NJ}) — "
                              f"jog joint(s) {weak} more, then fit")
            self.get_logger().error(self.last_info)
            return False
        self.J_img = JT.T
        self.calibrated = True
        cond = float(sv[0] / max(sv[-1], 1e-9))
        self.last_info = f"FIT OK from {n} samples (cond {cond:.1f})"
        if weak:
            self.last_info += f"; weak joints {weak}"
        self.get_logger().info(f"[fit] image Jacobian:\n{np.round(self.J_img, 4)}")
        self.get_logger().info(self.last_info)
        if self.save_calib_path:
            np.savez(self.save_calib_path, J=self.J_img)
            self.get_logger().info(f"[fit] saved -> {self.save_calib_path}")
        return True

    # ── trajectory dispatch (blocking) ───────────────────────────────────────
    def dispatch(self, target_q: np.ndarray, duration_s: float) -> bool:
        # Single-point goal (absolute target) — the controller interpolates from
        # the arm's ACTUAL current position.  Matches the tested teleop; avoids
        # asserting a (possibly wrong) start point that can cause a jump.
        pt = JointTrajectoryPoint()
        pt.positions = list(target_q)
        pt.velocities = [0.0] * NJ
        sec = int(duration_s)
        pt.time_from_start = DurationMsg(sec=sec, nanosec=int((duration_s - sec) * 1e9))
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINT_NAMES
        traj.points = [pt]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        with self.lock:
            cur = None if self.q is None else self.q.copy()
        if cur is not None:
            self.get_logger().info(
                f"[dispatch] cur ={np.round(cur, 3)}\n"
                f"[dispatch] tgt ={np.round(np.asarray(target_q), 3)}\n"
                f"[dispatch] d   ={np.round(np.asarray(target_q) - cur, 3)}  dur={duration_s:.2f}s")

        done = threading.Event()
        ok = [False]

        def _gc(fut):
            gh = fut.result()
            if not gh.accepted:
                done.set()
                return

            def _rc(r):
                ok[0] = (r.result().result.error_code ==
                         FollowJointTrajectory.Result.SUCCESSFUL)
                done.set()
            gh.get_result_async().add_done_callback(_rc)

        self.traj_client.send_goal_async(goal).add_done_callback(_gc)
        done.wait(timeout=duration_s + 10.0)
        return ok[0]

    def publish_marker(self, p_base):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "bbox_target"; m.id = 0
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p_base)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.04
        m.color.g = 1.0; m.color.a = 1.0
        self.marker_pub.publish(m)

    # ── one closed-loop control step ─────────────────────────────────────────
    def step(self) -> str:
        """Return 'moving' | 'converged' | 'wait' | 'lost'."""
        obs = self.observe()
        if obs is None:
            return "wait"
        u, v, Z = obs["u"], obs["v"], obs["Z"]
        fx, fy, cx, cy = obs["fx"], obs["fy"], obs["cx"], obs["cy"]
        W, H = obs["W"], obs["H"]
        x1, y1, x2, y2 = obs["box"]
        q = obs["q"]

        if not self.traj_client.wait_for_server(timeout_sec=2.0):
            self.last_info = "trajectory server unavailable"
            return "wait"
        if Z <= 0.02:
            self.last_info = f"no valid depth at box centre ({u},{v})"
            return "lost"

        # normalised image coords of the box centre (feature s = [x, y]); x,y are
        # 0 at the optical axis and ~±0.4-0.7 at the frame edges.
        x = (u - cx) / fx
        y = (v - cy) / fy
        # aim-point shifted so the END-EFFECTOR (not the lens) lines up with the
        # target: the EE offset is fixed in the camera frame, so in normalised
        # image coords it scales as offset/Z.
        x_star = self.ee_off_x / Z
        y_star = self.ee_off_y / Z
        ex = x - x_star
        ey = y - y_star
        lateral = float(np.hypot(ex, ey))               # error to the EE aim-point
        depth_err = Z - self.desired_z

        # convergence
        if lateral < CENTER_TOL_N and abs(depth_err) < Z_TOL:
            self.phase = "converged"
            return "converged"

        # centre-first gating + edge safety for the forward (depth) term
        w_z = float(np.clip((CENTER_ENGAGE_N - lateral) /
                            (CENTER_ENGAGE_N - CENTER_TOL_N), 0.0, 1.0))
        margin = EDGE_FRAC * min(W, H)
        if x1 < margin or y1 < margin or x2 > W - margin or y2 > H - margin:
            w_z = 0.0
        self.phase = "centre" if w_z < 1.0 else "approach"

        # image Jacobian J_img: dq -> d[x, y, Z].  Prefer the MEASURED one from
        # on-robot calibration (immune to a bad hand-eye orientation); otherwise
        # fall back to the analytic interaction-matrix x camera-twist Jacobian.
        if self.calibrated:
            J_img = self.J_img
        else:
            L = np.array([
                [-1.0 / Z, 0.0,     x / Z,  x * y,      -(1 + x * x), y],
                [0.0,     -1.0 / Z, y / Z,  1 + y * y,  -x * y,      -x],
                [0.0,      0.0,    -1.0,   -y * Z,       x * Z,       0.0],
            ], np.float64)
            Jc, _ = self.camera_twist_jacobian()
            if Jc is None:
                return "wait"
            J_img = L @ Jc

        # feature error e = s - s*  ->  drive box to the EE aim-point; depth to
        # standoff (gated).  ex,ey already include the end-effector offset.
        e = np.array([ex, ey, w_z * depth_err], np.float64)
        # desired feature change this step = -gain*e, capped so the image moves a
        # bounded amount per step (this is what keeps the box inside the frame).
        ds = -GAIN * e
        ds[0] = float(np.clip(ds[0], -DS_MAX_XY, DS_MAX_XY))
        ds[1] = float(np.clip(ds[1], -DS_MAX_XY, DS_MAX_XY))
        ds[2] = float(np.clip(ds[2], -DS_MAX_Z, DS_MAX_Z))

        dq = damped_pinv_solve(J_img, ds, DLS_LAMBDA)
        dq = np.clip(dq, -MAX_DQ, MAX_DQ)

        target = self._clip_target(q + dq)
        dur = max(MIN_DUR, float(np.max(np.abs(target - q) / (VEL_FRAC * JOINT_VEL_LIMITS))))

        self.last_info = (f"phase={self.phase} err={lateral:.3f} "
                          f"dz={depth_err*100:+5.1f}cm |dq|={np.linalg.norm(dq):.3f} "
                          f"dq4={dq[3]:+.3f}")
        self.get_logger().info(self.last_info)
        self.dispatch(target, dur)
        time.sleep(SETTLE_S)
        return "moving"

    # ── servo loop (runs in its own thread) ──────────────────────────────────
    def run_servo(self):
        if self.busy:
            return
        if not self.calibrated:
            self.last_info = "NOT calibrated — press 'k' (centre a static target) or --load-calib"
            self.get_logger().warn(self.last_info)
            return
        self.busy = True
        self.servo_on = True
        lost = 0
        last_log = 0.0
        try:
            while rclpy.ok() and self.servo_on:
                res = self.step()
                if res in ("wait", "lost") and (time.monotonic() - last_log) > 1.0:
                    self.get_logger().warn(f"idle: {self.last_info}")
                    last_log = time.monotonic()
                if res == "converged":
                    self.get_logger().info("✓ centred and at standoff depth")
                    self.last_info = "CONVERGED — target centred at standoff"
                    break
                if res == "lost":
                    lost += 1
                    if lost > MAX_LOST:
                        self.last_info = "ABORT — target/depth lost"
                        break
                    time.sleep(0.05)
                    continue
                if res == "wait":
                    time.sleep(0.05)
                    continue
                lost = 0
        finally:
            self.busy = False
            self.servo_on = False
            self.phase = "idle"


# ── OpenCV UI: manual box selection + tracker + HUD ──────────────────────────
def _make_tracker():
    for ctor in ("TrackerCSRT_create", "TrackerKCF_create"):
        if hasattr(cv2, ctor):
            return getattr(cv2, ctor)()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, ctor):
            return getattr(cv2.legacy, ctor)()
    return None


class UI:
    def __init__(self, node: BBoxPlanner):
        self.node = node
        self.win = "BBox Planner | Kinova Jaco2"
        self.drag0 = None
        self.drag1 = None
        self.dragging = False
        self.tracker = None

    def on_mouse(self, event, x, y, flags, _):
        if self.node.source != "drag":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag0 = (x, y); self.drag1 = (x, y); self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x1, y1 = self.drag0; x2, y2 = self.drag1
            box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            if (box[2] - box[0]) * (box[3] - box[1]) > 100:
                self._init_tracker(box)

    def _init_tracker(self, box):
        frame = self.node.get_color()
        if frame is None:
            return
        self.tracker = _make_tracker()
        x1, y1, x2, y2 = box
        if self.tracker is not None:
            try:
                self.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            except Exception:
                self.tracker = None
        self.node.set_bbox(box)

    def _update_tracker(self, frame):
        if self.tracker is None or self.node.source != "drag":
            return
        try:
            ok, r = self.tracker.update(frame)
        except Exception:
            ok = False
        if ok:
            x, y, w, h = (int(v) for v in r)
            self.node.set_bbox((x, y, x + w, y + h))

    def loop(self):
        # silence Qt/plugin chatter on stderr while the window lives
        saved = os.dup(2); devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2); os.close(devnull)
        try:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.win, self.on_mouse)
            while rclpy.ok():
                frame = self.node.get_color()
                if frame is not None:
                    self._update_tracker(frame)
                    cv2.imshow(self.win, self._hud(frame))
                k = cv2.waitKey(30) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k in (13, 10, ord("s")) and not self.node.busy:   # Enter (CR/LF) or 's'
                    threading.Thread(target=self.node.run_servo, daemon=True).start()
                if k == ord("k") and not self.node.busy:             # auto-probe calibrate
                    threading.Thread(target=self.node.calibrate, daemon=True).start()
                if ord("1") <= k <= ord("6"):                        # select jog joint
                    self.node.jog_joint = k - ord("1")
                if k in (ord("."), ord("=")) and not self.node.busy:  # jog + & record
                    threading.Thread(target=self.node.jog, args=(+1,), daemon=True).start()
                if k in (ord(","), ord("-")) and not self.node.busy:  # jog - & record
                    threading.Thread(target=self.node.jog, args=(-1,), daemon=True).start()
                if k == ord("r"):                                    # record a sample
                    self.node.record_sample()
                if k == ord("f") and not self.node.busy:             # fit Jacobian
                    threading.Thread(target=self.node.fit_jacobian, daemon=True).start()
                if k == ord("x"):                                    # clear samples
                    self.node.clear_samples()
                if k == ord("c"):
                    self.node.servo_on = False
                    self.tracker = None
                    self.node.clear_bbox()
        finally:
            os.dup2(saved, 2); os.close(saved)
            cv2.destroyAllWindows()

    def _hud(self, frame):
        disp = frame.copy()
        H, W = disp.shape[:2]
        # EE aim-point (where the box should land so the end-effector lines up),
        # drawn at the standoff depth; faint dot = the optical centre.
        zc = max(self.node.desired_z, 1e-3)
        ax = int(self.node.cx + (self.node.ee_off_x / zc) * self.node.fx)
        ay = int(self.node.cy + (self.node.ee_off_y / zc) * self.node.fy)
        cv2.drawMarker(disp, (ax, ay), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(disp, "EE", (ax + 14, ay + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(disp, (int(self.node.cx), int(self.node.cy)),
                       (90, 90, 90), cv2.MARKER_CROSS, 10, 1)
        m = int(EDGE_FRAC * min(W, H))
        cv2.rectangle(disp, (m, m), (W - m, H - m), (60, 60, 60), 1)

        with self.node.lock:
            box = self.node.bbox
        if self.dragging and self.drag0 and self.drag1:
            cv2.rectangle(disp, self.drag0, self.drag1, (0, 200, 255), 1)
        # draw the detected lip landmarks in mouth mode
        with self.node.lock:
            lms = self.node.mouth_lms
        if lms is not None:
            n = len(lms)
            for i in MOUTH_LIP_IDS:
                if i < n:
                    cv2.circle(disp, (int(lms[i].x * W), int(lms[i].y * H)), 1,
                               (0, 200, 255), -1)
        if box is not None:
            x1, y1, x2, y2 = box
            cu, cv_ = (x1 + x2) // 2, (y1 + y2) // 2
            col = (0, 255, 120) if self.node.phase == "converged" else (0, 180, 255)
            cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)
            cv2.circle(disp, (cu, cv_), 4, col, -1)
            cv2.line(disp, (ax, ay), (cu, cv_), (120, 120, 120), 1)
            label = "MOUTH" if self.node.source == "mouth" else "TARGET"
            cv2.putText(disp, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        col, 1, cv2.LINE_AA)

        cal = "CAL" if self.node.calibrated else "UNCAL"
        cal_col = (120, 230, 160) if self.node.calibrated else (90, 160, 255)
        status = (f"[{cal}] {'RUNNING' if self.node.busy else 'READY'}  "
                  f"J{self.node.jog_joint + 1} smp={len(self.node.calib_samples)}  "
                  f"{self.node.last_info}")
        cv2.rectangle(disp, (0, 0), (W, 24), (12, 12, 18), -1)
        cv2.putText(disp, status, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    cal_col if not self.node.busy else (120, 230, 160), 1, cv2.LINE_AA)
        hint = ("1-6=joint  .,=jog+rec  f=FIT  x=clear  k=auto  ENTER=servo  q=quit"
                if not self.node.calibrated else
                "ENTER=servo  .,=jog+rec  f=refit  k=recal  c=cancel  q=quit")
        cv2.putText(disp, hint, (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (140, 145, 145), 1, cv2.LINE_AA)
        return disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desired-z", type=float, default=DESIRED_Z,
                    help="final camera->target standoff depth (m)")
    ap.add_argument("--source", choices=["mouth", "drag", "topic"], default="mouth",
                    help="bbox source: mouth = MediaPipe lips (default), "
                         "drag = draw + track, topic = /target_bbox")
    ap.add_argument("--save-calib", default=None, metavar="PATH",
                    help="save the measured image Jacobian (.npz) after calibration")
    ap.add_argument("--load-calib", default=None, metavar="PATH",
                    help="load a previously measured image Jacobian (.npz) and skip 'k'")
    ap.add_argument("--ee-offset-y", type=float, default=EE_OFFSET_Y,
                    help="EE offset below the camera in the cam frame (m); "
                         "default 0.05 = EE 5 cm below the depth camera")
    ap.add_argument("--ee-offset-x", type=float, default=EE_OFFSET_X,
                    help="EE horizontal offset from the camera in the cam frame (m)")
    args = ap.parse_args()

    rclpy.init()
    node = BBoxPlanner(args.desired_z, source=args.source,
                       ee_off_x=args.ee_offset_x, ee_off_y=args.ee_offset_y)
    node.save_calib_path = args.save_calib
    if args.load_calib:
        node.load_calibration(args.load_calib)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    nxt = "press ENTER to servo" if node.calibrated else "press 'k' to calibrate (static target)"
    node.get_logger().info(
        f"bbox_planner up — source={args.source}, standoff {args.desired_z:.2f} m, "
        f"calibrated={node.calibrated}. {nxt}.")
    ui = UI(node)
    try:
        ui.loop()
    finally:
        node.servo_on = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
