#!/usr/bin/env python3
"""
Click-to-Feed for Kinova Jaco2  —  Phased Joint Stepping (no IK / MoveIt)
==========================================================================

ALGORITHM — FULL EXPLANATION
─────────────────────────────────────────────────────────────────────────────

STEP 1 · Pixel → Camera-frame 3-D target
  Standard pinhole back-projection converts a clicked pixel (u, v) plus its
  aligned-depth value Z_mm into a metric 3-D point in the camera optical frame:

      X_cam = (u − cx) · Z / fx      ← camera-right
      Y_cam = (v − cy) · Z / fy      ← camera-down
      Z_cam = Z_mm · 0.001           ← depth (towards scene)

  cx, cy, fx, fy are the RealSense D435 colour-camera intrinsics.
  The depth image is already aligned to the colour frame by the ROS driver, so
  depth[v, u] is the exact depth at the clicked colour pixel — no rectification
  needed.

  A fixed vertical correction (CAM_Y_OFFSET = 9 cm) is subtracted from Y_cam.
  The camera is mounted ~9 cm above the EE tip, so without the offset the arm
  would stop 9 cm too high.  We subtract in the Y direction because camera Y
  points downward: making it smaller moves the aim-point downward (toward the
  EE).

STEP 2 · Confirmation gate
  The 3-D target is stored but NOT executed immediately.  The operator presses
  Y to go or N to abort.  This single gate prevents accidental limb motion if
  the click was mis-aimed at background clutter.

STEP 3 · Phased proportional joint stepping
  Instead of solving IK (which needs a complete kinematic model and can fail
  near singularities), the error is decomposed into three camera-frame axes and
  each axis is corrected by the one or two joints whose primary coupling is to
  that axis at the arm's nominal pose:

    Phase 1 — Y-axis (vertical)     → J2 (elbow pitch)
      Pitching the elbow up/down changes EE height almost linearly.
      Handled first because vertical error is largest at meal-time (bowl on
      table ↔ mouth height) and must be cleared before X to avoid shoulder
      singularities that arise when the arm is outstretched and twisted.

    Phase 2 — X-axis (horizontal)   → J3 (forearm rotation)
      Rotating the forearm sweeps the EE left/right.
      Y is already settled, so J3 changes only X cleanly.

    Phase 3 — Z-axis (depth)        → J1 (base rotation) + J4 (wrist pitch)
      J1 extends the arm forward (primary Z effect).
      J4 compensates for the wrist-angle change that J1 introduces, keeping
      the spoon/fork level.
      Only forward approach (dz > 0) is automated; retract is done by reset.

  Control law — proportional clamped step (same in every phase):
      frac  = clamp(|error| / CART_DELTA,  0, 1.0)
      step  = frac × STEP_RADS               ← joint increment sent to arm
      track = frac × CART_DELTA              ← added to open-loop EE estimate

  When |error| ≫ CART_DELTA  →  frac = 1  (maximum-speed step, fast approach)
  When |error| < CART_DELTA  →  frac < 1  (scaled-down step, prevents overshoot)

  WHY OPEN-LOOP POSITION TRACKING?
    Live TF queries inside the control loop add ≥100 ms latency per step.  At
    TICK_DUR_S = 0.6 s that dominates the settling time and halves throughput.
    Instead, the code accumulates (frac × CART_DELTA) after each joint command.
    The CART_DELTA constants are tuned empirically on the real arm so one unit
    of frac × STEP_RADS at J2/J3/J1 moves the EE by CART_DELTA metres in the
    corresponding camera axis.  This approximation is valid because:
      • The arm operates in a narrow workspace envelope (meal-time reach).
      • We only need ~5 cm accuracy (ERR_TOL_M); exact IK is overkill.
"""
from __future__ import annotations

import math
import os
import threading
import time
import warnings
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import Image, JointState
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ── suppress Qt / OpenCV noise that pollutes the terminal dashboard ────────────
os.environ["QT_LOGGING_RULES"] = "qt.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
# Point Qt to a real system font dir so it stops complaining about the missing
# cv2 pip-package fonts directory (the warning appears on every Qt draw call).
for _d in ("/usr/share/fonts", "/usr/share/fonts/truetype", "/usr/local/share/fonts"):
    if os.path.isdir(_d):
        os.environ.setdefault("QT_QPA_FONTDIR", _d)
        break
warnings.filterwarnings("ignore")

# ── Camera intrinsics (RealSense D435 factory defaults – tune if needed) ──────
FX = 603.6312;  FY = 603.0632
CX = 319.0870;  CY = 236.3678

# ── Robot frames & joint names ────────────────────────────────────────────────
EE_LINK    = "j2s6s200_end_effector"
CAM_FRAME  = "camera_color_optical_frame"
BASE_FRAME = "j2s6s200_link_base"

ARM_JOINT_NAMES = [
    "j2s6s200_joint_1", "j2s6s200_joint_2", "j2s6s200_joint_3",
    "j2s6s200_joint_4", "j2s6s200_joint_5", "j2s6s200_joint_6",
]

# ── Camera-to-EE vertical offset: camera is 9 cm above EE in camera-Y ────────
CAM_Y_OFFSET = 0.09

# ── Stepping parameters (empirically tuned on the real arm) ──────────────────
STEP_RADS    = 0.05   # max joint radians per command
CART_DELTA_X = 0.15   # metres EE moves in camera-X per STEP_RADS on J3
CART_DELTA_Y = 0.12   # metres EE moves in camera-Y per STEP_RADS on J2
CART_DELTA_Z = 0.06   # metres EE moves in camera-Z per STEP_RADS on J1/J4
TICK_DUR_S   = 0.6    # trajectory settling time (s) between joint commands
ERR_TOL_M    = 0.05   # convergence threshold: 5 cm

# ── MediaPipe face-mesh landmark connection tables ────────────────────────────
MOUTH_CENTER_IDS = [13, 14, 0, 17]

LIP_CONN = [
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),
    (314,405),(405,321),(321,375),(375,291),(61,185),(185,40),
    (40,39),(39,37),(37,0),(0,267),(267,269),(269,270),
    (270,409),(409,291),(78,95),(95,88),(88,178),(178,87),
    (87,14),(14,317),(317,402),(402,318),(318,324),(324,308),
    (78,191),(191,80),(80,81),(81,82),(82,13),(13,312),
    (312,311),(311,310),(310,415),(415,308),
]

FACE_OVAL_CONN = [
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),
    (389,356),(356,454),(454,323),(323,361),(361,288),(288,397),
    (397,365),(365,379),(379,378),(378,400),(400,377),(377,152),
    (152,148),(148,176),(176,149),(149,150),(150,136),(136,172),
    (172,58),(58,132),(132,93),(93,234),(234,127),(127,162),
    (162,21),(21,54),(54,103),(103,67),(67,109),(109,10),
]

# Both eyes combined (person's right eye + person's left eye)
EYE_CONN = [
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),
    (154,155),(155,133),(133,173),(173,157),(157,158),(158,159),
    (159,160),(160,161),(161,246),(246,33),
    (263,249),(249,390),(390,373),(373,374),(374,380),(380,381),
    (381,382),(382,362),(362,398),(398,384),(384,385),(385,386),
    (386,387),(387,388),(388,466),(466,263),
]

# Both eyebrows combined
EYEBROW_CONN = [
    (46,53),(53,52),(52,65),(65,55),(55,70),(70,63),(63,105),(105,66),(66,107),
    (276,283),(283,282),(282,295),(295,285),(285,300),(300,293),(293,334),(334,296),(296,336),
]

# Nose bridge + tip
NOSE_CONN = [
    (168,6),(6,197),(197,195),(195,5),(5,4),(4,1),(1,19),(19,94),(94,2),
]


class TUI:
    """ANSI-powered terminal dashboard for Click-to-Feed."""

    # ── ANSI escape codes ─────────────────────────────────────────────────────
    # Using 8-bit "bright" codes (9x) rather than 3x: more vivid on dark
    # terminals, still degrade gracefully to the 3x equivalent on light ones.
    RST = "\033[0m";  BLD = "\033[1m";  DIM = "\033[2m"

    RED = "\033[91m"; GRN = "\033[92m"; YLW = "\033[93m"
    BLU = "\033[94m"; MAG = "\033[95m"; CYN = "\033[96m"
    WHT = "\033[97m"; GRY = "\033[90m"

    W = 70  # total border width (characters)

    def __init__(self):
        self._lock = threading.Lock()

    @staticmethod
    def _ts() -> str:
        return time.strftime("%H:%M:%S")

    # ── box primitives ────────────────────────────────────────────────────────
    def _top(self) -> str:
        return f"{self.BLD}{self.CYN}╔{'═' * (self.W - 2)}╗{self.RST}"

    def _bot(self) -> str:
        return f"{self.BLD}{self.CYN}╚{'═' * (self.W - 2)}╝{self.RST}"

    def _sep(self) -> str:
        return f"{self.BLD}{self.CYN}╠{'═' * (self.W - 2)}╣{self.RST}"

    def _row(self, text: str, color: str = "") -> str:
        inner = self.W - 2
        return (f"{self.BLD}{self.CYN}║{self.RST}"
                f"{color}{text:^{inner}}{self.RST}"
                f"{self.BLD}{self.CYN}║{self.RST}")

    # ── public: layout ────────────────────────────────────────────────────────
    def banner(self):
        """Print the startup header box — call once at launch."""
        print(f"\n{self._top()}")
        print(self._row(
            "  CLICK-TO-FEED  ·  Kinova Jaco2  ·  Phased Joint Stepping  ",
            f"{self.BLD}{self.WHT}"))
        print(self._sep())
        print(self._row(
            "  SPACE = mouth   CLICK = target   Y = confirm   N = cancel   Q = quit  ",
            self.GRY))
        print(self._bot())
        print()

    def separator(self):
        """Thin horizontal rule between logical sections."""
        with self._lock:
            print(f"  {self.GRY}{'─' * (self.W - 4)}{self.RST}")

    # ── public: log messages ──────────────────────────────────────────────────
    def _emit(self, symbol: str, color: str, msg: str):
        # Format: "  HH:MM:SS  ● message"
        # Timestamp in dim grey → easy to scan, not distracting.
        # Symbol in colour conveys severity at a glance before reading text.
        with self._lock:
            print(f"  {self.GRY}{self._ts()}{self.RST}  "
                  f"{color}{symbol}{self.RST}  {msg}")

    def info(self, msg: str):
        self._emit("●", self.CYN, msg)

    def warn(self, msg: str):
        self._emit("▲", self.YLW, f"{self.YLW}{msg}{self.RST}")

    def error(self, msg: str):
        self._emit("✖", self.RED, f"{self.RED}{msg}{self.RST}")

    def success(self, msg: str):
        self._emit("✔", self.GRN, f"{self.BLD}{self.GRN}{msg}{self.RST}")

    def prompt(self, msg: str):
        """Bold yellow prompt for operator decisions."""
        with self._lock:
            print(f"\n  {self.BLD}{self.YLW}▶  {msg}{self.RST}\n")

    # ── public: phased algorithm progress ────────────────────────────────────
    def phase_header(self, phase_num: int, name: str, detail: str = ""):
        """
        Print a visually distinct section break for each phase.
        Circled numbers ①②③ let the operator instantly know which phase is
        running without reading the full label.
        """
        icons = {1: "①", 2: "②", 3: "③"}
        icon  = icons.get(phase_num, f"[{phase_num}]")
        label = f" {icon}  Phase {phase_num}  ·  {name} "
        pad   = max(2, (self.W - 4 - len(label)) // 2)
        with self._lock:
            print(f"\n  {self.BLD}{self.MAG}"
                  f"{'─' * pad}{label}{'─' * pad}{self.RST}")
            if detail:
                print(f"  {self.DIM}{self.GRY}  {detail}{self.RST}")

    def moving(self, label: str, elapsed: float, total: float):
        """
        Live \\r progress bar for a single smooth trajectory.

        Safe to use \\r here because during a phase sleep the motion thread is
        the ONLY thing writing to stdout — ROS callbacks update internal state
        only, and the TUI lock blocks any other concurrent print.  The bar
        shows elapsed / total time so the operator sees exactly how far the
        arm is through its trajectory.
        """
        BAR = 26
        frac   = min(elapsed / max(total, 0.001), 1.0)
        filled = int(BAR * frac)
        bar    = (f"{self.GRN}{'█' * filled}"
                  f"{self.GRY}{'░' * (BAR - filled)}{self.RST}")
        remain = max(0.0, total - elapsed)
        with self._lock:
            print(f"\r    {self.CYN}{label:<10}{self.RST}"
                  f"[{bar}]  "
                  f"{self.GRY}{elapsed:.1f}s / {total:.1f}s  "
                  f"({self.WHT}{remain:.1f}s{self.GRY} left){self.RST}   ",
                  end="", flush=True)

    def phase_done(self, label: str):
        """Overwrite the \\r line with a full green bar and move to next line."""
        BAR = 26
        bar = f"{self.GRN}{'█' * BAR}{self.RST}"
        with self._lock:
            print(f"\r    {self.BLD}{self.GRN}{label:<10}{self.RST}"
                  f"[{bar}]  {self.BLD}{self.GRN}✔  complete{self.RST}"
                  + " " * 20)   # trailing spaces clear any leftover chars

    def coord_block(self, label: str, x: float, y: float, z: float, color: str = ""):
        """Single-line coordinate triplet display."""
        c = color or self.GRY
        with self._lock:
            print(f"  {c}{label:<18}{self.RST}"
                  f"{self.WHT}x={x:+.3f}  y={y:+.3f}  z={z:+.3f}{self.RST}  m")

    def ready_prompt(self):
        """Displayed once after initialisation."""
        self.separator()
        self.prompt("Ready — click the image or press SPACE to target the mouth")
        self.separator()


# ══════════════════════════════════════════════════════════════════════════════
#  MediaPipe loader
#  Handles both old (mp.solutions) and new (FaceLandmarker) MediaPipe APIs.
# ══════════════════════════════════════════════════════════════════════════════
def _build_face_mesh():
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    import pathlib, tempfile, urllib.request
    model_url = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task")
    model_path = pathlib.Path(tempfile.gettempdir()) / "face_landmarker.task"
    if not model_path.exists():
        urllib.request.urlretrieve(model_url, model_path)

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        running_mode=RunningMode.IMAGE)
    landmarker = FaceLandmarker.create_from_options(options)

    class Wrapper:
        def process(self, rgb):
            h, w = rgb.shape[:2]
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            except AttributeError:
                from mediapipe.tasks.python.components.containers.image import Image as MPImage
                mp_img = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)
            if not result.face_landmarks:
                return type("R", (), {"multi_face_landmarks": None})()
            class Landmark:
                def __init__(self, x, y, z=0.0): self.x=x; self.y=y; self.z=z
            class LandmarkList:
                def __init__(self, lms): self.landmark = lms
            all_faces = []
            for face in result.face_landmarks:
                all_faces.append(LandmarkList([Landmark(lm.x, lm.y, lm.z) for lm in face]))
            return type("R", (), {"multi_face_landmarks": all_faces})()

    return Wrapper()


# ══════════════════════════════════════════════════════════════════════════════
#  Neon face overlay
#  ──────────────────────────────────────────────────────────────────────────
#  Technique: draw coloured strokes on a black canvas, blur the canvas with
#  GaussianBlur to simulate light spreading from each line (glow), then blend
#  that glow layer over the camera image.  Finally, draw sharp crisp lines on
#  top so the contours read clearly against the soft glow.
#
#  Why GaussianBlur for glow and not real alpha-blending per line?
#    OpenCV has no per-line transparency.  Blurring a black canvas with bright
#    strokes produces exactly the same visual as a glowing neon tube — the blur
#    spreads the brightness outward while the intensity falls off with distance,
#    matching how real light diffuses.
# ══════════════════════════════════════════════════════════════════════════════
def _draw_face_cool(disp: np.ndarray, lms, hw, frame_count: int) -> np.ndarray:
    """Render neon-wireframe face landmarks over the camera image."""
    if not lms or not hw:
        return disp
    mh, mw = hw
    n = len(lms)

    def px(i):
        return (int(lms[i].x * mw), int(lms[i].y * mh))

    def seg(canvas, conn_list, colour, thick):
        for a, b in conn_list:
            if a < n and b < n:
                cv2.line(canvas, px(a), px(b), colour, thick, cv2.LINE_AA)

    # ── BGR colour palette ────────────────────────────────────────────────────
    # Each region gets its own hue so the operator can distinguish features
    # at a glance even when the arm is moving.
    OVAL_C  = (30,  210, 255)   # gold / amber  — face boundary
    EYE_C   = (50,  255, 160)   # spring green  — eye contours
    BROW_C  = (255, 60,  220)   # magenta        — eyebrows
    NOSE_C  = (50,  190, 255)   # orange         — nose
    LIP_C   = (0,   120, 255)   # deep amber     — lips

    # ── Glow pass — thick strokes on black, then blurred ─────────────────────
    glow = np.zeros_like(disp)
    seg(glow, FACE_OVAL_CONN, OVAL_C, 4)
    seg(glow, EYE_CONN,       EYE_C,  3)
    seg(glow, EYEBROW_CONN,   BROW_C, 3)
    seg(glow, NOSE_CONN,      NOSE_C, 2)
    seg(glow, LIP_CONN,       LIP_C,  3)

    # Kernel (17,17) spreads ~8 px — gives visible halo without being muddy.
    glow = cv2.GaussianBlur(glow, (17, 17), 0)
    cv2.addWeighted(disp, 1.0, glow, 0.70, 0, disp)

    # ── Sharp lines on top (crisp core) ──────────────────────────────────────
    seg(disp, FACE_OVAL_CONN, (60,  235, 255), 1)
    seg(disp, EYE_CONN,       (80,  255, 180), 1)
    seg(disp, EYEBROW_CONN,   (255, 80,  240), 1)
    seg(disp, NOSE_CONN,      (80,  210, 255), 1)
    seg(disp, LIP_CONN,       (0,   180, 255), 1)

    # ── Landmark dots on contour vertices ────────────────────────────────────
    # 1-px dots mark every contour vertex so the mesh looks like a point cloud
    # at the facial boundaries.  Only contour points (not all 468) so it stays
    # readable rather than a solid cluster.
    for conn in (FACE_OVAL_CONN, EYE_CONN, EYEBROW_CONN):
        seen = set()
        for a, b in conn:
            for i in (a, b):
                if i < n and i not in seen:
                    cv2.circle(disp, px(i), 1, (220, 255, 255), -1, cv2.LINE_AA)
                    seen.add(i)

    # ── Iris circles (landmarks 468-477, only when refine_landmarks=True) ────
    # Radius = distance from iris centre landmark to one edge landmark, which
    # directly encodes the visible iris diameter in the image.
    def draw_iris(center_i, edge_i, max_i):
        if max_i >= n:
            return
        c, e = px(center_i), px(edge_i)
        r = max(3, int(math.hypot(c[0] - e[0], c[1] - e[1])))
        # outer dim halo
        cv2.circle(disp, c, r + 5, (20, 50, 50), 1, cv2.LINE_AA)
        # iris ring — bright cyan
        cv2.circle(disp, c, r, (220, 255, 255), 1, cv2.LINE_AA)
        # pupil — solid black with white specular dot
        cv2.circle(disp, c, max(2, r // 3), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(disp, (c[0] - r//5, c[1] - r//5), max(1, r//6),
                   (255, 255, 255), -1, cv2.LINE_AA)

    draw_iris(468, 469, 472)   # left iris
    draw_iris(473, 474, 477)   # right iris

    # ── Animated scan line across the face ───────────────────────────────────
    # A horizontal line sweeps top-to-bottom across the face bounding box every
    # 80 frames (~2.6 s at 30 fps).  The fade at the edges (alpha∝sin) prevents
    # the hard appearance of a line suddenly appearing/disappearing.
    oval_px = [px(a) for a, _ in FACE_OVAL_CONN if a < n]
    if oval_px:
        fy0 = min(p[1] for p in oval_px)
        fy1 = max(p[1] for p in oval_px)
        fx0 = min(p[0] for p in oval_px)
        fx1 = max(p[0] for p in oval_px)
        period = 80
        t = (frame_count % period) / period          # 0 → 1
        sy = int(fy0 + (fy1 - fy0) * t)
        alpha = max(0.0, math.sin(t * math.pi))      # 0→1→0 fade
        if alpha > 0.05:
            scan_col = (int(60 * alpha), int(220 * alpha), int(60 * alpha))
            cv2.line(disp, (fx0, sy), (fx1, sy), scan_col, 1, cv2.LINE_AA)

    # ── Mouth centre pulse ────────────────────────────────────────────────────
    # Mouth landmarks 13,14,0,17 average to the centre of the lips.
    # Two concentric pulsing rings show the target point that SPACE will use.
    mc_xs = [lms[i].x * mw for i in MOUTH_CENTER_IDS if i < n]
    mc_ys = [lms[i].y * mh for i in MOUTH_CENTER_IDS if i < n]
    if mc_xs:
        mx, my = int(np.mean(mc_xs)), int(np.mean(mc_ys))
        # outer static halo
        for r, b in [(28, 20), (21, 50), (15, 90)]:
            cv2.circle(disp, (mx, my), r, (0, b, b // 2), 1, cv2.LINE_AA)
        # pulsing ring — period ~2 s
        pr = int(11 + 4 * math.sin(frame_count * 0.10))
        cv2.circle(disp, (mx, my), pr, (0, 255, 180), 2, cv2.LINE_AA)
        # solid centre dot
        cv2.circle(disp, (mx, my), 5, (50, 255, 120), -1, cv2.LINE_AA)
        cv2.putText(disp, "MOUTH", (mx - 22, my - 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 230, 180), 1, cv2.LINE_AA)

    return disp


# ══════════════════════════════════════════════════════════════════════════════
#  OpenCV HUD
#  ──────────────────────────────────────────────────────────────────────────
#  Design choices:
#   • cv2.addWeighted alpha-blend: gives dark semi-transparent panels without
#     needing actual OpenGL transparency — pure CPU, no extra deps.
#   • Pulsing mouth ring: sin(frame_count) radius oscillation signals "live
#     face tracking" vs a frozen dot, at zero computation cost.
#   • Colour semantics:
#       Cyan / teal  = informational / idle
#       Green        = good / confirmed
#       Amber/yellow = warning / awaiting decision
#       Red-ish blue = active motion (FEEDING)
#   • No text inside the camera image unless strictly necessary — the terminal
#     provides all the verbose data; the image shows spatial context only.
# ══════════════════════════════════════════════════════════════════════════════
def _draw_hud(disp: np.ndarray, node, frame_count: int) -> np.ndarray:
    h, w = disp.shape[:2]

    # ── 1. Semi-transparent top status bar ────────────────────────────────────
    # We blend a solid dark rectangle (alpha=0.72) over the image top.
    # This keeps the bar legible against any background without a hard border.
    overlay = disp.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.72, disp, 0.28, 0, disp)

    # ── 2. Status pill ────────────────────────────────────────────────────────
    # A filled rounded-rect (approximated with two rectangles) acts as a
    # coloured "badge" — colour encodes state at a glance before reading text.
    if node.busy:
        pill_bgr  = (160, 30,  0)      # deep blue (BGR)
        pill_txt  = " ● FEEDING... "
        txt_bgr   = (255, 255, 200)
    elif node.pending_target_cam is not None:
        pill_bgr  = (0, 100, 180)      # amber
        pill_txt  = " ▶ CONFIRM? "
        txt_bgr   = (80, 220, 255)
    else:
        pill_bgr  = (0, 60, 0)         # dark green
        pill_txt  = " ● READY "
        txt_bgr   = (80, 255, 140)

    font   = cv2.FONT_HERSHEY_DUPLEX
    fscale = 0.52
    thick  = 1
    (tw, th), _ = cv2.getTextSize(pill_txt, font, fscale, thick)
    px, py = 8, 6
    # pill background
    cv2.rectangle(disp, (px - 2, py - 2), (px + tw + 10, py + th + 10), pill_bgr, -1)
    # subtle bright border
    cv2.rectangle(disp, (px - 2, py - 2), (px + tw + 10, py + th + 10), (100, 120, 120), 1)
    cv2.putText(disp, pill_txt, (px + 4, py + th + 3), font, fscale, txt_bgr, thick,
                cv2.LINE_AA)

    # ── 3. Hint text (right of pill) ──────────────────────────────────────────
    if not node.busy:
        if node.pending_target_cam is not None:
            hint = "Y = feed   N = cancel"
        else:
            hint = "SPACE = mouth    CLICK = target    Q = quit"
        cv2.putText(disp, hint,
                    (px + tw + 22, py + th + 3), font, 0.38, (140, 145, 145), 1, cv2.LINE_AA)

    # ── 4. Full neon face overlay ─────────────────────────────────────────────
    mouth_lms, mouth_hw = node.get_landmark_data()
    disp = _draw_face_cool(disp, mouth_lms, mouth_hw, frame_count)

    # ── 5. Target coordinates overlay ────────────────────────────────────────
    # When a target is pending, show its camera-frame coordinates at the
    # bottom of the image so the operator can sanity-check depth vs. distance.
    if node.pending_target_cam is not None:
        tc = node.pending_target_cam
        coord_str = f"target  x={tc[0]:+.2f}  y={tc[1]:+.2f}  z={tc[2]:+.2f} m"
        # semi-transparent bottom strip
        ov2 = disp.copy()
        cv2.rectangle(ov2, (0, h - 28), (w, h), (10, 10, 20), -1)
        cv2.addWeighted(ov2, 0.68, disp, 0.32, 0, disp)
        cv2.putText(disp, coord_str, (10, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 255), 1, cv2.LINE_AA)

    # ── 7. Phase indicator (bottom-right during FEEDING) ──────────────────────
    if node.busy:
        phase_name = getattr(node, "_current_phase", "moving...")
        phase_str  = f"Phase: {phase_name}"
        (ptw, pth), _ = cv2.getTextSize(phase_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        ov3 = disp.copy()
        cv2.rectangle(ov3, (w - ptw - 18, h - 28), (w, h), (10, 10, 20), -1)
        cv2.addWeighted(ov3, 0.68, disp, 0.32, 0, disp)
        cv2.putText(disp, phase_str, (w - ptw - 10, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 180, 50), 1, cv2.LINE_AA)

    # ── 8. Crosshair (centre of frame) ───────────────────────────────────────
    # Shown only when idle — hidden during feeding to reduce visual noise.
    if not node.busy:
        cx2, cy2 = w // 2, h // 2
        gap = 6   # gap in the centre keeps the crosshair airy
        arm = 18  # arm length
        clr = (0, 200, 160)
        cv2.line(disp, (cx2 - arm - gap, cy2), (cx2 - gap, cy2), clr, 1, cv2.LINE_AA)
        cv2.line(disp, (cx2 + gap, cy2), (cx2 + arm + gap, cy2), clr, 1, cv2.LINE_AA)
        cv2.line(disp, (cx2, cy2 - arm - gap), (cx2, cy2 - gap), clr, 1, cv2.LINE_AA)
        cv2.line(disp, (cx2, cy2 + gap), (cx2, cy2 + arm + gap), clr, 1, cv2.LINE_AA)
        # tiny centre dot
        cv2.circle(disp, (cx2, cy2), 2, clr, -1, cv2.LINE_AA)

    return disp


# ══════════════════════════════════════════════════════════════════════════════
#  ROS2 Node
# ══════════════════════════════════════════════════════════════════════════════
class ClickPointer(Node):
    def __init__(self, tui: TUI):
        super().__init__("click_pointer")
        self.tui      = tui
        self.cb_group = ReentrantCallbackGroup()

        self.bridge    = CvBridge()
        self.lock      = threading.Lock()
        self.color_img = None
        self.depth_img = None
        self.joints    = {n: 0.0 for n in ARM_JOINT_NAMES}
        self.busy      = False

        self.mouth_px  = None
        self.mouth_lms = None
        self.mouth_hw  = None
        self.mp_face   = _build_face_mesh()

        self.pending_target_cam = None

        # Phase label read by the HUD to show which phase is running
        self._current_phase = ""

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, "/camera/camera/color/image_raw",
                                 self.color_cb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw",
                                 self.depth_cb, 10, callback_group=self.cb_group)
        self.create_subscription(JointState, "/joint_states",
                                 self.js_cb, 10, callback_group=self.cb_group)

        self.marker_pub = self.create_publisher(Marker, "/target_marker", 10)

        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
            callback_group=self.cb_group)

        self.tui.info("Waiting for trajectory controller ...")
        if not self._traj_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("Timed out waiting for trajectory action server")

        self.tui.ready_prompt()

    # ── Subscribers ───────────────────────────────────────────────────────────
    def color_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        mp_, ml_, mh_ = self._detect_mouth(frame)
        with self.lock:
            self.color_img = frame
            self.mouth_px  = mp_
            self.mouth_lms = ml_
            self.mouth_hw  = mh_

    def depth_cb(self, msg):
        with self.lock:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def js_cb(self, msg):
        with self.lock:
            for n, p in zip(msg.name, msg.position):
                if n in self.joints:
                    self.joints[n] = p

    # ── Face / mouth detection ─────────────────────────────────────────────────
    def _detect_mouth(self, frame):
        h, w = frame.shape[:2]
        res  = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None, None, None
        lms = res.multi_face_landmarks[0].landmark
        return (
            int(np.mean([lms[i].x * w for i in MOUTH_CENTER_IDS])),
            int(np.mean([lms[i].y * h for i in MOUTH_CENTER_IDS]))
        ), lms, (h, w)

    # ── Thread-safe getters for the HUD ───────────────────────────────────────
    def get_frame_data(self):
        with self.lock:
            return self.color_img

    def get_mouth_px(self):
        with self.lock:
            return self.mouth_px

    def get_landmark_data(self):
        with self.lock:
            return self.mouth_lms, self.mouth_hw

    # ── Input events ──────────────────────────────────────────────────────────
    def on_click(self, u, v):
        if not self.busy:
            self.process_click(u, v)

    def on_space(self):
        with self.lock:
            mpix = self.mouth_px
        if mpix is None:
            self.tui.warn("No mouth detected — look at the camera")
            return
        self.process_click(*mpix)

    def confirm_feed(self):
        if self.pending_target_cam is None or self.busy:
            return
        tgt = self.pending_target_cam
        self.pending_target_cam = None
        threading.Thread(target=self.move_to_point_cam, args=(tgt,), daemon=True).start()

    def cancel_feed(self):
        if self.pending_target_cam is not None:
            self.pending_target_cam = None
            self.delete_marker()
            self.tui.info("Feed cancelled.")

    # ── Click → 3-D target in camera frame ────────────────────────────────────
    def process_click(self, u, v):
        if self.busy:
            return
        with self.lock:
            depth = self.depth_img
        if depth is None:
            self.tui.warn("No depth image yet — waiting for camera")
            return
        z_raw = float(depth[v, u]) * 0.001
        if z_raw <= 0.01:
            self.tui.warn(f"Bad depth at ({u},{v}): {z_raw*1000:.0f} mm — click on the subject")
            return

        # Back-project pixel to camera-frame 3-D point
        x_cam = (u - CX) * z_raw / FX
        y_cam = (v - CY) * z_raw / FY
        # Apply camera-to-EE vertical offset (camera sits above EE)
        target_cam = np.array([x_cam, y_cam - CAM_Y_OFFSET, z_raw])

        self.tui.separator()
        self.tui.info(f"Click at pixel ({u}, {v}),  depth = {z_raw*100:.1f} cm")
        self.tui.coord_block("Target (cam-frame):", *target_cam, color=TUI.CYN)

        # Publish RViz marker in base frame for spatial reference
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, CAM_FRAME, Time(), Duration(seconds=1.0))
            R = self._quat_to_matrix(tf.transform.rotation)
            t = np.array([tf.transform.translation.x,
                          tf.transform.translation.y,
                          tf.transform.translation.z])
            target_base = R @ target_cam + t
        except Exception as e:
            self.tui.warn(f"TF camera→base failed ({e}), skipping RViz marker")
            target_base = target_cam

        self.publish_marker(target_base, BASE_FRAME)
        self.tui.coord_block("Target (base-frame):", *target_base, color=TUI.BLU)

        self.pending_target_cam = target_cam
        self.tui.prompt("Press  Y  to feed   ·   N  to cancel")

    # ── RViz marker helpers ────────────────────────────────────────────────────
    def publish_marker(self, pt, fid):
        m = Marker()
        m.header.frame_id = fid
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "target"; m.id = 0
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x = float(pt[0])
        m.pose.position.y = float(pt[1])
        m.pose.position.z = float(pt[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color.r = 1.0; m.color.a = 1.0
        m.lifetime.sec = 0
        self.marker_pub.publish(m)

    def delete_marker(self):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "target"; m.id = 0; m.action = Marker.DELETE
        self.marker_pub.publish(m)

    @staticmethod
    def _quat_to_matrix(q):
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1-2*y*y-2*z*z,   2*x*y-2*z*w,   2*x*z+2*y*w],
            [  2*x*y+2*z*w, 1-2*x*x-2*z*z,   2*y*z-2*x*w],
            [  2*x*z-2*y*w,   2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])

    # ── Phased joint stepping ─────────────────────────────────────────────────
    def move_to_point_cam(self, target_cam):
        if self.busy:
            return
        self.busy = True
        try:
            try:
                tf_ee_cam = self.tf_buffer.lookup_transform(
                    CAM_FRAME, EE_LINK, Time(), timeout=Duration(seconds=1.0))
            except Exception as e:
                self.tui.error(f"TF EE→cam failed: {e}")
                return

            start = np.array([
                tf_ee_cam.transform.translation.x,
                tf_ee_cam.transform.translation.y,
                tf_ee_cam.transform.translation.z,
            ])
            self.tui.separator()
            self.tui.coord_block("EE start (cam):", *start, color=TUI.GRY)
            self.tui.coord_block("Target   (cam):", *target_cam, color=TUI.CYN)

            self._perform_phased(start, target_cam)

            self.tui.separator()
            self.tui.success("Feeding sequence complete.")
            self.tui.separator()
            time.sleep(0.5)

        except Exception as e:
            self.tui.error(f"move_to_point_cam: {e}")
        finally:
            self.busy = False
            self._current_phase = ""

    def _perform_phased(self, startpoint, endpoint):
        """
        Smooth phased approach — same Y→X→Z logic as the module docstring but
        each phase is ONE trajectory command instead of N jerky steps.

        The total joint delta is computed analytically:
            delta_rad = (error_m / CART_DELTA_m) * STEP_RADS

        This is exactly the sum of all the incremental steps the old loop would
        have taken: the proportional controller (frac = clamp(|err|/CART_DELTA,1))
        converges in ceil(|err|/CART_DELTA) full steps plus one partial step.
        Summing those gives (|err| / CART_DELTA) * STEP_RADS in total, which is
        what we compute directly here.

        Duration is matched to the old total time:
            dur = max(MIN_DUR, |err| / CART_DELTA * TICK_DUR_S)
        so the arm moves at the same average speed but without interruptions.
        """
        MIN_DUR = 1.5   # floor (s) — gives the controller enough ramp time

        dy = endpoint[1] - startpoint[1]
        dz = endpoint[2] - startpoint[2]

        # Phase 2 uses camera-origin X, not EE-relative X (see below).
        # endpoint[0] = (u_click - CX) * Z / FX  — already measured from the
        # camera optical centre, which sits at X=0 in the camera frame.
        dx_cam = endpoint[0]

        # ── Phase 1: Y-axis → J2 ─────────────────────────────────────────────
        # Sign: dy>0 means EE is above target → decrease J2 → negative delta.
        self.tui.phase_header(1, "Y-axis  →  J2 (elbow pitch)",
                              f"error = {dy:+.3f} m")
        self._current_phase = "Y → J2"
        if abs(dy) > ERR_TOL_M:
            delta_J2 = -(dy / CART_DELTA_Y) * STEP_RADS
            dur_Y    = max(MIN_DUR, abs(dy) / CART_DELTA_Y * TICK_DUR_S)
            self._dispatch_smooth({1: delta_J2}, dur_Y, "Y-axis")
        self.tui.phase_done("Y-axis")

        # ── Phase 2: X-axis → J3  (camera as X origin) ───────────────────────
        # Instead of measuring how far the EE is from the target in 3D X, we
        # ask: "where is the target relative to the camera centre-line?"
        #   • endpoint[0] = (u_click − CX) × Z / FX
        #   • This is positive when the target is right of image centre,
        #     negative when left — direction is unambiguous from one number.
        #   • The camera centre (u = CX, pixel X = 0 in camera frame) is the
        #     natural "zero": if the target is centred, no X correction needed.
        # This avoids needing the EE's current camera-frame X from TF.
        px_offset = endpoint[0] * FX / endpoint[2]   # pixels from centre (for log)
        self.tui.phase_header(2, "X-axis  →  J3 (forearm rotation)",
                              f"target = {dx_cam:+.3f} m from cam-centre  "
                              f"({px_offset:+.0f} px)")
        self._current_phase = "X → J3"
        if abs(dx_cam) > ERR_TOL_M:
            delta_J3 = -(dx_cam / CART_DELTA_X) * STEP_RADS
            dur_X    = max(MIN_DUR, abs(dx_cam) / CART_DELTA_X * TICK_DUR_S)
            self._dispatch_smooth({2: delta_J3}, dur_X, "X-axis")
        self.tui.phase_done("X-axis")

        # ── Phase 3: Z-axis → J1 + J4 ────────────────────────────────────────
        # J1 and J4 move simultaneously in the same trajectory so the wrist
        # compensation happens continuously, keeping the fork level throughout.
        self.tui.phase_header(3, "Z-axis  →  J1 + J4 (depth approach)",
                              f"error = {dz:+.3f} m")
        self._current_phase = "Z → J1+J4"
        if dz > ERR_TOL_M:
            delta_J1 = (dz / CART_DELTA_Z) * STEP_RADS * 2.0
            delta_J4 = (dz / CART_DELTA_Z) * STEP_RADS * 2.0
            dur_Z    = max(MIN_DUR, dz / CART_DELTA_Z * TICK_DUR_S)
            self._dispatch_smooth({0: delta_J1, 3: delta_J4}, dur_Z, "Z-axis")
        self.tui.phase_done("Z-axis")

        self.tui.info("All three phases converged.")

    def _wait_with_bar(self, label: str, duration: float):
        """Sleep for duration while updating a live \\r progress bar every 80 ms."""
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            self.tui.moving(label, elapsed, duration)
            if elapsed >= duration + 0.15:
                break
            time.sleep(0.08)

    # ── Trajectory dispatch ───────────────────────────────────────────────────
    def _dispatch_step(self, joint_idx: int, delta_rad: float):
        """
        Send a two-point JointTrajectory: current → current+delta in TICK_DUR_S.
        Two points are required (not one) so the controller generates a
        velocity-limited trapezoid profile instead of a step command.
        """
        with self.lock:
            current_pos = [self.joints[n] for n in ARM_JOINT_NAMES]
        target_pos          = list(current_pos)
        target_pos[joint_idx] += delta_rad

        pt0 = JointTrajectoryPoint()
        pt0.positions       = current_pos
        pt0.time_from_start = DurationMsg(sec=0, nanosec=0)

        pt1 = JointTrajectoryPoint()
        pt1.positions       = target_pos
        pt1.time_from_start = DurationMsg(sec=0, nanosec=int(TICK_DUR_S * 1e9))

        traj               = JointTrajectory()
        traj.header.stamp  = self.get_clock().now().to_msg()
        traj.joint_names   = ARM_JOINT_NAMES
        traj.points        = [pt0, pt1]

        goal          = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self._traj_client.send_goal_async(goal)

    def _dispatch_smooth(self, deltas: dict, duration_s: float, label: str = "moving"):
        """
        Send ONE trajectory that moves several joints simultaneously, then
        BLOCK until the bridge action server reports completion (succeed/abort).
        Shows a live progress bar while waiting.

        Blocking is critical: without it, Phase 2 and Phase 3 can execute
        concurrently (the bridge runs concurrent goals with ReentrantCallbackGroup),
        causing two control loops to fight over the arm simultaneously.
        """
        with self.lock:
            current_pos = [self.joints[n] for n in ARM_JOINT_NAMES]
        target_pos = list(current_pos)
        for idx, delta in deltas.items():
            target_pos[idx] += delta

        pt0 = JointTrajectoryPoint()
        pt0.positions       = list(current_pos)
        pt0.time_from_start = DurationMsg(sec=0, nanosec=0)

        pt1 = JointTrajectoryPoint()
        pt1.positions       = target_pos
        dur_sec             = int(duration_s)
        dur_ns              = int((duration_s - dur_sec) * 1e9)
        pt1.time_from_start = DurationMsg(sec=dur_sec, nanosec=dur_ns)

        traj              = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = ARM_JOINT_NAMES
        traj.points       = [pt0, pt1]

        goal            = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        done_event = threading.Event()

        def _goal_resp(future):
            gh = future.result()
            if not gh.accepted:
                done_event.set()
                return
            gh.get_result_async().add_done_callback(lambda _: done_event.set())

        self._traj_client.send_goal_async(goal).add_done_callback(_goal_resp)

        # Block here, updating the progress bar every 80 ms, until bridge is done
        t0 = time.time()
        while not done_event.wait(timeout=0.08):
            self.tui.moving(label, time.time() - t0, duration_s)
        print()  # end the \r line


# ══════════════════════════════════════════════════════════════════════════════
#  OpenCV UI loop
# ══════════════════════════════════════════════════════════════════════════════
def run_ui(node: ClickPointer, tui: TUI):
    # ASCII-only window name: Qt encodes the title as a C-string internally;
    # non-ASCII characters (e.g. the middle-dot U+00B7) can cause namedWindow
    # to return a null handler, which makes setMouseCallback crash.
    winname = "Click-to-Feed | Kinova Jaco2"

    # Redirect stderr for the window lifetime so Qt font/platform warnings
    # don't pollute the terminal dashboard.  Our TUI uses stdout, so nothing
    # useful is lost.  We restore stderr before printing any error or exiting.
    _saved_stderr = os.dup(2)
    _devnull      = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)

    try:
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                node.on_click(x, y)

        cv2.setMouseCallback(winname, mouse_cb, node)

        frame_count = 0

        while rclpy.ok():
            frame = node.get_frame_data()

            if frame is not None:
                disp = frame.copy()
                disp = _draw_hud(disp, node, frame_count)
                cv2.imshow(winname, disp)
                frame_count += 1

            # 30 ms poll → ~30 fps; waitKey drives the OpenCV event loop
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                threading.Thread(target=node.on_space, daemon=True).start()
            elif key in (ord("y"), ord("Y")):
                node.confirm_feed()
            elif key in (ord("n"), ord("N")):
                node.cancel_feed()

    finally:
        os.dup2(_saved_stderr, 2)   # restore stderr before any terminal output
        os.close(_saved_stderr)
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    tui = TUI()
    tui.banner()

    rclpy.init(args=args)
    node = ClickPointer(tui)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        run_ui(node, tui)
    finally:
        tui.info("Cleaning up ROS2 node ...")
        node.destroy_node()
        rclpy.shutdown()
        tui.success("Shutdown complete.")


if __name__ == "__main__":
    main()
