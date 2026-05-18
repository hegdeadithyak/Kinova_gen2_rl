#!/usr/bin/env python3
"""
eye_confirm.py — Eye-gaze confirmation for click_pointer.py.

Publishes to /eye_confirm (std_msgs/Bool):
  True  →  confirm feed   (same as pressing Y in click_pointer)
  False →  cancel feed    (same as pressing N)

Run standalone:
  python3 eye_confirm.py [--cam <device_index>]
"""

import glob
import os
import subprocess
import sys
import threading
import time

# ── cv2 must be imported before Qt so its bundled Qt plugins load first,
#    then we override the plugin path back to the system Qt5 plugins.
sys.path.insert(0, "/home/amma/Eye-Sign-Detection_v1/Eye-Sign-Detection/ultralytics")

import numpy as np
import cv2                          # cv2 pip package sets QT_QPA_PLATFORM_PLUGIN_PATH to its own plugins
from collections import deque

# Override cv2's Qt plugin path with system Qt5 so PyQt5 loads correctly.
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"
if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ["QT_QPA_PLATFORM"] = "xcb"

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QFrame, QListWidget,
                              QGraphicsDropShadowEffect, QSizePolicy, QShortcut)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QLinearGradient, QKeySequence

import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = os.path.expanduser("~/.cache/mediapipe/face_landmarker.task")
_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")

def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        print(f"Downloading face_landmarker model → {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")
    return _MODEL_PATH

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool

# ── Constants ─────────────────────────────────────────────────
CLASS_NAMES   = ["Straight", "Top", "Right", "Left", "Close"]
CLASS_SYMBOLS = ["–", "↑", "→", "←", "↓"]

VIDEO_W, VIDEO_H = 640, 480

C_BG       = "#0b0e14"
C_PANEL    = "#161b22"
C_ACCENT   = "#58a6ff"
C_SUCCESS  = "#3fb950"
C_WARN     = "#d29922"
C_DANGER   = "#f85149"
C_TEXT     = "#c9d1d9"
C_TEXT_DIM = "#8b949e"

COMMANDS = {
    (3, 1): "Start Feeding",
    (1, 2): "Stop Feeding",
    (2, 4): "Call Nurse",
    (1, 0): "Home",
    (3, 2): "Settings",
    (2, 3): "Precheck",
    (4, 0): "Stop",
}


# ── ROS2 publisher node ───────────────────────────────────────

class EyeConfirmNode(Node):
    def __init__(self):
        super().__init__('eye_confirm')
        self._pub = self.create_publisher(Bool, '/eye_confirm', 10)

    def confirm(self):
        self._pub.publish(Bool(data=True))
        self.get_logger().info('[eye] confirm → /eye_confirm True')

    def cancel(self):
        self._pub.publish(Bool(data=False))
        self.get_logger().info('[eye] cancel  → /eye_confirm False')


# ── Custom widgets ────────────────────────────────────────────

class GlassPanel(QFrame):
    def __init__(self, parent=None, glow=False):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {C_PANEL}cc;
                border-radius: 12px;
                border: 1px solid #30363d;
            }}
        """)
        if glow:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(20)
            shadow.setColor(QColor(88, 166, 255, 40))
            shadow.setOffset(0, 0)
            self.setGraphicsEffect(shadow)


class SignalMeter(QWidget):
    def __init__(self, label, color, parent=None):
        super().__init__(parent)
        self.label = label
        self.color = QColor(color)
        self.value = 0.0
        self.setMinimumHeight(40)

    def set_value(self, val):
        self.value = max(0.0, min(1.0, val))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QColor(C_TEXT_DIM))
        p.setFont(QFont("Inter", 9))
        p.drawText(0, 15, self.label)
        p.setBrush(QColor("#21262d"))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 22, self.width(), 6, 3, 3)
        if self.value > 0:
            grad = QLinearGradient(0, 0, self.width() * self.value, 0)
            grad.setColorAt(0, self.color.darker(150))
            grad.setColorAt(1, self.color)
            p.setBrush(grad)
            p.drawRoundedRect(0, 22, int(self.width() * self.value), 6, 3, 3)


class PrecisionGauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.distance = 0.0
        self.setMinimumSize(100, 100)
        self.setMaximumSize(120, 120)

    def set_distance(self, d):
        self.distance = d
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        center = self.rect().center()
        radius = min(self.width(), self.height()) // 2 - 10
        p.setPen(QPen(QColor("#21262d"), 8))
        p.drawEllipse(center, radius, radius)
        color = QColor(C_SUCCESS)
        if self.distance < 0.3 or self.distance > 0.6:
            color = QColor(C_WARN)
        if self.distance == 0:
            color = QColor(C_TEXT_DIM)
        p.setPen(QPen(color, 8, Qt.SolidLine, Qt.RoundCap))
        span = min(int(self.distance * 360), 360)
        p.drawArc(center.x()-radius, center.y()-radius, radius*2, radius*2,
                  90*16, -span*16)
        p.setPen(QColor(C_TEXT))
        p.setFont(QFont("Monospace", 11, QFont.Bold))
        text = f"{self.distance:.2f}m" if self.distance > 0 else "--m"
        p.drawText(self.rect(), Qt.AlignCenter, text)


# ── Inference worker ──────────────────────────────────────────

class EyeWaveWorker(QThread):
    frame_ready       = pyqtSignal(np.ndarray, list, str, float, list)
    command_triggered = pyqtSignal(str)
    error_occurred    = pyqtSignal(str)

    # FaceMesh landmark indices
    _L_OUTER, _L_INNER = 33,  133   # left eye corners (outer=temporal, inner=nasal)
    _L_TOP,   _L_BOT   = 159, 145
    _L_IRIS             = 468
    _R_INNER, _R_OUTER = 362, 263   # right eye corners
    _R_TOP,   _R_BOT   = 386, 374
    _R_IRIS             = 473

    _EAR_THRESH = 0.18   # below → Close
    _H_THRESH   = 0.10   # |dx| above → Left / Right  (iris moves ~10% of eye width)
    _V_THRESH   = 0.12   # dy below (iris up) → Top

    def __init__(self, ros_node: "EyeConfirmNode", cam_index: int = 3):
        super().__init__()
        self.cam_index       = cam_index
        self.running         = True
        self.state_buffer    = deque(maxlen=15)
        self.confirmed_state = -1
        self.sequence        = []
        self.last_seq_t      = time.time()
        self.conf_levels     = [0.0] * 5

    # ── gaze classification ───────────────────────────────────

    def _classify(self, lm, w, h):
        """Return (cls, conf, dets) from FaceMesh landmarks."""
        def p(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        # EAR for both eyes — eye aspect ratio
        def ear(top, bot, outer, inner):
            return (np.linalg.norm(p(top) - p(bot)) /
                    (np.linalg.norm(p(outer) - p(inner)) + 1e-6))

        avg_ear = (ear(self._L_TOP, self._L_BOT, self._L_OUTER, self._L_INNER) +
                   ear(self._R_TOP, self._R_BOT, self._R_OUTER, self._R_INNER)) / 2

        if avg_ear < self._EAR_THRESH:
            cls  = 4   # Close
            conf = float(np.clip(1.0 - avg_ear / self._EAR_THRESH, 0, 1))
        else:
            # iris offset from eye centre, normalised by eye width/height.
            # Both eyes naturally give dx < 0 when looking RIGHT (irises shift to
            # smaller x in the image), so no flip needed — just average directly.
            def offset(iris_idx, outer_idx, inner_idx, top_idx, bot_idx):
                iris = p(iris_idx)
                cx   = (p(outer_idx)[0] + p(inner_idx)[0]) / 2
                cy   = (p(top_idx)[1]   + p(bot_idx)[1])   / 2
                ew   = abs(p(outer_idx)[0] - p(inner_idx)[0]) + 1e-6
                eh   = abs(p(top_idx)[1]   - p(bot_idx)[1])   + 1e-6
                return (iris[0] - cx) / ew, (iris[1] - cy) / eh

            ldx, ldy = offset(self._L_IRIS, self._L_OUTER, self._L_INNER, self._L_TOP, self._L_BOT)
            rdx, rdy = offset(self._R_IRIS, self._R_OUTER, self._R_INNER, self._R_TOP, self._R_BOT)
            dx = (ldx + rdx) / 2
            dy = (ldy + rdy) / 2

            # dx < 0 → irises shifted left in image → person looking to THEIR right
            # dx > 0 → irises shifted right in image → person looking to THEIR left
            if   dx < -self._H_THRESH: cls = 2; conf = float(np.clip(-dx / 0.35, 0, 1)) # Right
            elif dx >  self._H_THRESH: cls = 3; conf = float(np.clip(dx  / 0.35, 0, 1)) # Left
            elif dy < -self._V_THRESH: cls = 1; conf = float(np.clip(-dy / 0.35, 0, 1)) # Top
            else:                      cls = 0; conf = 0.85                               # Straight

        # bounding box spanning both eye regions for visualisation
        eye_pts = np.array([p(self._L_OUTER), p(self._L_INNER), p(self._L_TOP), p(self._L_BOT),
                            p(self._R_OUTER), p(self._R_INNER), p(self._R_TOP), p(self._R_BOT)])
        x1, y1 = (eye_pts.min(axis=0) - 12).tolist()
        x2, y2 = (eye_pts.max(axis=0) + 12).tolist()
        return cls, conf, [[x1, y1, x2, y2, conf, cls]]

    def _process_state(self, s):
        self.state_buffer.append(s)
        if len(self.state_buffer) >= 5:
            valid = [x for x in self.state_buffer if x != -1]
            if valid:
                counts = np.bincount(valid, minlength=5)
                new_s = int(counts.argmax()) if counts.max() >= 4 else -1
            else:
                new_s = -1
            if new_s != self.confirmed_state:
                if new_s != -1:
                    if not self.sequence or self.sequence[-1] != new_s:
                        self.sequence.append(new_s)
                        self.last_seq_t = time.time()
                        self._check_sequence()
                self.confirmed_state = new_s
        if self.sequence and (time.time() - self.last_seq_t > 2.5):
            self.sequence = []

    def _check_sequence(self):
        for length in [4, 3, 2]:
            if len(self.sequence) >= length:
                sub = tuple(self.sequence[-length:])
                if sub in COMMANDS:
                    self.command_triggered.emit(COMMANDS[sub])
                    self.sequence = []
                    return

    def _find_cam_index(self) -> int | None:
        """Return the first openable non-RealSense camera, or self.cam_index if set."""
        if self.cam_index != 3:  # user specified explicitly
            return self.cam_index
        # Read device names from sysfs and skip RealSense/depth cameras
        skip_keywords = ("realsense", "depth", "obsensor")
        candidates = []
        for path in sorted(glob.glob("/sys/class/video4linux/video*/name")):
            try:
                name = open(path).read().strip().lower()
                if any(k in name for k in skip_keywords):
                    continue
                idx = int(os.path.basename(os.path.dirname(path)).replace("video", ""))
                candidates.append((idx, name))
            except Exception:
                pass
        for idx, name in candidates:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.release()
                print(f"[eye] auto-selected /dev/video{idx} ({name})")
                return idx
            cap.release()
        return None

    def run(self):
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        detector = mp_vision.FaceLandmarker.create_from_options(options)

        cam_idx = 1
        if cam_idx is None:
            self.error_occurred.emit("No usable camera found. Try --cam <index>.")
            return
        print(f"[eye] opening /dev/video{cam_idx}")
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open /dev/video{cam_idx}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    continue

                h, w = frame.shape[:2]
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_img)

                dets          = []
                current_state = -1
                self.conf_levels = [0.0] * 5

                if result.face_landmarks:
                    lm = result.face_landmarks[0]   # list of NormalizedLandmark
                    cls, conf, dets = self._classify(lm, w, h)
                    current_state = cls
                    if 0 <= cls < 5:
                        self.conf_levels[cls] = conf

                self._process_state(current_state)
                state_name = CLASS_NAMES[self.confirmed_state] if self.confirmed_state != -1 else "IDLE"
                self.frame_ready.emit(frame, dets, state_name, 0.0, self.conf_levels)

            except Exception as e:
                print(f"[EyeWave] loop error: {e}")

        cap.release()
        detector.close()


# ── Main window ───────────────────────────────────────────────

class AegisUI(QMainWindow):
    pipeline_status = pyqtSignal(str)

    def __init__(self, ros_node: EyeConfirmNode, cam_index: int = 3):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle("EyeWave Aegis — click_pointer confirm")
        self.resize(1280, 800)
        self.setStyleSheet(
            f"background-color: {C_BG}; color: {C_TEXT}; font-family: 'Inter', sans-serif;"
        )
        self._zoom           = 1.0
        self._feeding_active = False
        self._scoop_proc     = None
        self._click_proc     = None
        self._init_ui()
        self.pipeline_status.connect(self._update_pipeline_status)

        QShortcut(QKeySequence(Qt.Key_Plus),  self).activated.connect(self._zoom_in)
        QShortcut(QKeySequence(Qt.Key_Equal), self).activated.connect(self._zoom_in)
        QShortcut(QKeySequence(Qt.Key_Minus), self).activated.connect(self._zoom_out)

        self.worker = EyeWaveWorker(ros_node, cam_index)
        self.worker.frame_ready.connect(self._update_data)
        self.worker.command_triggered.connect(self._on_command)
        self.worker.error_occurred.connect(lambda m: self.lbl_cmd.setText(m))
        self.worker.start()

    def _init_ui(self):
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        left_box = QVBoxLayout()
        vid_panel = GlassPanel(glow=True)
        vid_layout = QVBoxLayout(vid_panel)
        vid_layout.setContentsMargins(8, 8, 8, 8)
        self.lbl_vid = QLabel("INITIALIZING...")
        self.lbl_vid.setAlignment(Qt.AlignCenter)
        self.lbl_vid.setFixedSize(VIDEO_W, VIDEO_H)
        self.lbl_vid.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        vid_layout.addWidget(self.lbl_vid, alignment=Qt.AlignCenter)
        left_box.addWidget(vid_panel)

        tele_panel = GlassPanel()
        tele_lay = QHBoxLayout(tele_panel)
        self.gauge = PrecisionGauge()
        self.lbl_status = QLabel("SYSTEM: ACTIVE")
        self.lbl_status.setFont(QFont("Inter", 12, QFont.Bold))
        tele_lay.addWidget(self.gauge)
        tele_lay.addStretch()
        tele_lay.addWidget(self.lbl_status)
        left_box.addWidget(tele_panel)

        right_box = QVBoxLayout()

        cmd_panel = GlassPanel(glow=True)
        cmd_v = QVBoxLayout(cmd_panel)
        lbl_h = QLabel("PRIMARY OUTPUT")
        lbl_h.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 2px;")
        cmd_v.addWidget(lbl_h)
        self.lbl_cmd = QLabel("STANDING BY")
        self.lbl_cmd.setFont(QFont("Inter", 30, QFont.Bold))
        self.lbl_cmd.setStyleSheet(f"color: {C_ACCENT};")
        self.lbl_cmd.setAlignment(Qt.AlignCenter)
        self.lbl_cmd.setWordWrap(True)
        cmd_v.addWidget(self.lbl_cmd)
        self.lbl_pipeline = QLabel("PIPELINE: IDLE")
        self.lbl_pipeline.setFont(QFont("Inter", 11))
        self.lbl_pipeline.setStyleSheet(f"color: {C_TEXT_DIM};")
        self.lbl_pipeline.setAlignment(Qt.AlignCenter)
        cmd_v.addWidget(self.lbl_pipeline)
        right_box.addWidget(cmd_panel, 2)

        met_panel = GlassPanel()
        met_v = QVBoxLayout(met_panel)
        lbl_m = QLabel("SIGNAL ANALYTICS")
        lbl_m.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 2px;")
        met_v.addWidget(lbl_m)
        self.meters = [SignalMeter(f"{CLASS_SYMBOLS[i]}  {CLASS_NAMES[i]}", C_ACCENT)
                       for i in range(len(CLASS_NAMES))]
        for m in self.meters:
            met_v.addWidget(m)
        right_box.addWidget(met_panel, 3)

        intel_panel = GlassPanel()
        intel_v = QVBoxLayout(intel_panel)
        lbl_a = QLabel("RECENT ACTIVITY")
        lbl_a.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 2px;")
        intel_v.addWidget(lbl_a)
        self.log = QListWidget()
        self.log.setStyleSheet(
            f"border: none; background: transparent; color: {C_TEXT}; font-size: 12px;"
        )
        intel_v.addWidget(self.log)
        right_box.addWidget(intel_panel, 2)

        guide_panel = GlassPanel()
        guide_v = QVBoxLayout(guide_panel)
        lbl_g = QLabel("COMMAND GUIDE")
        lbl_g.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 2px;")
        guide_v.addWidget(lbl_g)
        for seq, name in COMMANDS.items():
            symbols = "  ".join(CLASS_SYMBOLS[s] for s in seq)
            special = "  <span style='color:#3fb950; font-size:10px;'>[confirm when active]</span>" \
                      if name == "Start Feeding" else ""
            row = QLabel(
                f"<span style='color:{C_ACCENT}; font-size:15px; font-weight:bold;'>{symbols}</span>"
                f"  <span style='color:{C_TEXT}; font-size:11px;'>{name}</span>{special}")
            row.setTextFormat(Qt.RichText)
            row.setContentsMargins(4, 2, 4, 2)
            guide_v.addWidget(row)
        right_box.addWidget(guide_panel, 3)

        layout.addLayout(left_box, 3)
        layout.addLayout(right_box, 2)
        self.setCentralWidget(central)

    def _update_data(self, frame, dets, state, dist, levels):
        h, w = frame.shape[:2]
        draw = frame.copy()
        for d in dets:
            x1, y1, x2, y2, conf, cls = d
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = (255, 166, 88)
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 1)
            tl = 12
            for px, py, dx, dy in [(x1,y1,1,0),(x1,y1,0,1),(x2,y1,-1,0),(x2,y1,0,1),
                                    (x1,y2,1,0),(x1,y2,0,-1),(x2,y2,-1,0),(x2,y2,0,-1)]:
                cv2.line(draw, (px, py), (px+dx*tl, py+dy*tl), color, 3)
            label = f"{CLASS_NAMES[cls] if 0 <= cls < 5 else cls} {conf:.2f}"
            cv2.putText(draw, label, (x1, max(y1-8, 14)), 0, 0.55, color, 2)

        # apply zoom: crop centre region of size (w/zoom, h/zoom) then scale to display
        if self._zoom > 1.0:
            cw, ch = int(w / self._zoom), int(h / self._zoom)
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            draw = draw[y0:y0+ch, x0:x0+cw]
            h, w = draw.shape[:2]

        qimg = QImage(draw.data, w, h, w * 3, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg).scaled(VIDEO_W, VIDEO_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_vid.setPixmap(pix)
        self.gauge.set_distance(dist)
        self.lbl_status.setText(f"STATE: {state}")
        for i, val in enumerate(levels):
            self.meters[i].set_value(val)

    def _on_command(self, cmd):
        if cmd == "Start Feeding":
            # Always confirm click_pointer if it's running standalone
            self.ros_node.confirm()
            self.lbl_pipeline.setText("PIPELINE: CONFIRMED → FEEDING")
            self.lbl_pipeline.setStyleSheet(f"color: {C_SUCCESS};")
            if not self._feeding_active:
                self._feeding_active = True
                threading.Thread(target=self._run_feed_pipeline, daemon=True).start()
        elif cmd == "Stop Feeding":
            self.ros_node.cancel()
            self._stop_pipeline()

        self.lbl_cmd.setText(cmd.upper())
        self.lbl_cmd.setStyleSheet(f"color: {C_SUCCESS};")
        self.log.insertItem(0, f"[{time.strftime('%H:%M:%S')}]  {cmd}")
        if self.log.count() > 50:
            self.log.takeItem(self.log.count() - 1)
        QTimer.singleShot(2500, lambda: self.lbl_cmd.setStyleSheet(f"color: {C_ACCENT};"))

    def _update_pipeline_status(self, msg: str):
        self.lbl_pipeline.setText(f"PIPELINE: {msg}")
        msg_up = msg.upper()
        if any(w in msg_up for w in ("FEED", "CONFIRM", "AWAIT")):
            color = C_SUCCESS
        elif any(w in msg_up for w in ("FAIL", "ERROR", "STOP")):
            color = C_DANGER
        else:
            color = C_WARN
        self.lbl_pipeline.setStyleSheet(f"color: {color};")

    def _run_feed_pipeline(self):
        _scoop = "/home/amma/kinova_ws/rl_v2-master/run_scoop.py"
        _click = "/home/amma/kinova_ws/rl_v2-master/src/click_pointer/scripts/click_pointer.py"

        self.pipeline_status.emit("SCOOP RUNNING")
        try:
            self._scoop_proc = subprocess.Popen(
                ["python3", _scoop],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ret = self._scoop_proc.wait()
        except Exception as e:
            self.pipeline_status.emit(f"SCOOP ERROR: {e}")
            self._feeding_active = False
            return

        if not self._feeding_active:
            return
        if ret != 0:
            self.pipeline_status.emit("SCOOP FAILED")
            self._feeding_active = False
            return

        self.pipeline_status.emit("AWAITING MOUTH — CONFIRM WITH EYES")
        try:
            self._click_proc = subprocess.Popen(
                ["python3", _click, "--backend", "joints"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            self.pipeline_status.emit(f"CLICK_POINTER ERROR: {e}")
            self._feeding_active = False

    def _stop_pipeline(self):
        self._feeding_active = False
        for proc in [self._click_proc, self._scoop_proc]:
            if proc and proc.poll() is None:
                proc.terminate()
        self._click_proc = None
        self._scoop_proc = None
        self.pipeline_status.emit("STOPPED")

    def _zoom_in(self):
        self._zoom = min(self._zoom + 0.25, 4.0)

    def _zoom_out(self):
        self._zoom = max(self._zoom - 0.25, 1.0)

    def closeEvent(self, e):
        self._stop_pipeline()
        self.worker.running = False
        self.worker.wait(3000)
        e.accept()


# ── Entry point ───────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="DroidCam video device index")
    args = parser.parse_args()

    rclpy.init()
    ros_node = EyeConfirmNode()
    executor = MultiThreadedExecutor()
    executor.add_node(ros_node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = AegisUI(ros_node, cam_index=args.cam)
    win.show()
    ret = app.exec_()

    executor.shutdown()
    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(ret)


if __name__ == "__main__":
    main()
