#!/usr/bin/env python3
"""Closed-loop picking: pick position → detect food → arm to food → scoop → repeat.

Eye gaze (laptop webcam) triggers the pick:
  Look LEFT then look UP  →  confirm pick
"""
import glob
import os
import sys
import math
import threading
import time
from collections import deque

sys.path.insert(0, "/home/amma/Eye-Sign-Detection_v1/Eye-Sign-Detection/ultralytics")
import numpy as np
import cv2
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"
if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"]  = "qt.*=false"
os.environ["OPENCV_LOG_LEVEL"]  = "ERROR"

import mediapipe as mp
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = os.path.expanduser("~/.cache/mediapipe/face_landmarker.task")
_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")

def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        print(f"Downloading face_landmarker → {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QPushButton, QHBoxLayout, QLabel, QFrame,
                              QListWidget, QGraphicsDropShadowEffect, QShortcut)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import (QImage, QPixmap, QFont, QColor, QPainter,
                         QLinearGradient, QKeySequence)

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient

from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PointStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as DurationMsg
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs  # noqa

# ── Constants ──────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]
NJ     = len(JOINT_NAMES)
TWO_PI = 2.0 * math.pi

ACTION_TOPIC      = '/arm_controller/follow_joint_trajectory'
JOINT_STATE_TOPIC = '/j2s6s200_driver/out/joint_state'
DUR_STEP          = 5.0

YOLO_WEIGHTS = '/home/amma/coco_food/runs/yolo11n_cafg-3/weights/best.pt'

PICK_RAD = [
    2.877301345958657, 3.828006445591176, 1.74634511479214,
    -9.59300878156525, 2.1700887984759727, 5.23718724373796,
]

SCOOP_SEQUENCE = [
    [2.473036155645718,   3.860704204822951,  1.1286894816046955,
     7.973092578992713,   3.928636916857019,  5.490991821944691],
    [2.5326994861789816,  4.036655263302851,  1.2444419141722265,
     1.6899869003297199,  3.9285500978054824, 5.4910413567409675],
    [2.2621428257673246,  3.981920911315107,  1.1588610993844757,
     -4.4203089183591215, 4.598469068963242,  5.372604191478527],
    [2.69807965934403,    4.241848895816258,  1.2520215369507008,
     1.2014285336160841,  4.778086501315855,  5.36437821950349],
]
SCOOP_LABELS = ['pre_scoop', 'after_pre_scoop', 'scoopv0', 'feed']

# Gaze
GAZE_STRAIGHT, GAZE_TOP, GAZE_RIGHT, GAZE_LEFT, GAZE_CLOSE = 0, 1, 2, 3, 4
GAZE_NAMES   = ["Straight", "Top", "Right", "Left", "Close"]
GAZE_SYMBOLS = ["--", "UP", "RIGHT", "LEFT", "CLOSE"]
PICK_SEQUENCE = (GAZE_LEFT, GAZE_TOP)

_L_OUTER, _L_INNER = 33,  133
_L_TOP,   _L_BOT   = 159, 145
_L_IRIS             = 468
_R_OUTER, _R_INNER = 263, 362
_R_TOP,   _R_BOT   = 386, 374
_R_IRIS             = 473
_EAR_THRESH = 0.18
_H_THRESH   = 0.10
_V_THRESH   = 0.12

VIDEO_W, VIDEO_H = 480, 360

C_BG       = "#0b0e14"
C_PANEL    = "#161b22"
C_ACCENT   = "#58a6ff"
C_SUCCESS  = "#3fb950"
C_WARN     = "#d29922"
C_DANGER   = "#f85149"
C_TEXT     = "#c9d1d9"
C_TEXT_DIM = "#8b949e"

_ITEM_COLOR = (0, 220, 255)   # cyan for food target


# ── Helpers ────────────────────────────────────────────────────────────────────

def _auto_webcam_index():
    skip = ("realsense", "depth", "obsensor", "dummy")
    for path in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        try:
            name = open(path).read().strip().lower()
            if any(k in name for k in skip):
                continue
            idx = int(os.path.basename(os.path.dirname(path)).replace("video", ""))
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f'[Eye] auto-selected /dev/video{idx} ({name})')
                    return idx
            else:
                cap.release()
        except Exception:
            pass
    return 1


def _classify_gaze(lm, w, h):
    def p(idx): return np.array([lm[idx].x * w, lm[idx].y * h])
    def ear(t, b, o, i): return (np.linalg.norm(p(t)-p(b)) /
                                  (np.linalg.norm(p(o)-p(i)) + 1e-6))
    avg_ear = (ear(_L_TOP,_L_BOT,_L_OUTER,_L_INNER) +
               ear(_R_TOP,_R_BOT,_R_OUTER,_R_INNER)) / 2
    if avg_ear < _EAR_THRESH:
        return GAZE_CLOSE, float(np.clip(1.0 - avg_ear / _EAR_THRESH, 0, 1))
    def off(iris, o, i, t, b):
        iris_p = p(iris)
        cx = (p(o)[0]+p(i)[0])/2; cy = (p(t)[1]+p(b)[1])/2
        ew = abs(p(o)[0]-p(i)[0])+1e-6; eh = abs(p(t)[1]-p(b)[1])+1e-6
        return (iris_p[0]-cx)/ew, (iris_p[1]-cy)/eh
    ldx, ldy = off(_L_IRIS,_L_OUTER,_L_INNER,_L_TOP,_L_BOT)
    rdx, rdy = off(_R_IRIS,_R_OUTER,_R_INNER,_R_TOP,_R_BOT)
    dx = (ldx+rdx)/2; dy = (ldy+rdy)/2
    if   dx < -_H_THRESH: return GAZE_RIGHT,    float(np.clip(-dx/0.35,0,1))
    elif dx >  _H_THRESH: return GAZE_LEFT,     float(np.clip(dx/0.35,0,1))
    elif dy < -_V_THRESH: return GAZE_TOP,      float(np.clip(-dy/0.35,0,1))
    else:                 return GAZE_STRAIGHT, 0.85


def _frame_to_pixmap(frame, w, h):
    fh, fw = frame.shape[:2]
    qimg = QImage(frame.data, fw, fh, fw*3, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ── Qt widgets ─────────────────────────────────────────────────────────────────

class GlassPanel(QFrame):
    def __init__(self, parent=None, glow=False):
        super().__init__(parent)
        self.setStyleSheet(f"""QFrame {{
            background-color: {C_PANEL}cc; border-radius: 12px;
            border: 1px solid #30363d; }}""")
        if glow:
            s = QGraphicsDropShadowEffect(self)
            s.setBlurRadius(20); s.setColor(QColor(88,166,255,40)); s.setOffset(0,0)
            self.setGraphicsEffect(s)


class SignalMeter(QWidget):
    def __init__(self, label, color, parent=None):
        super().__init__(parent)
        self.label = label; self.color = QColor(color); self.value = 0.0
        self.setMinimumHeight(36)
    def set_value(self, v): self.value = max(0.,min(1.,v)); self.update()
    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QColor(C_TEXT_DIM)); p.setFont(QFont("Inter",9))
        p.drawText(0,14,self.label)
        p.setBrush(QColor("#21262d")); p.setPen(Qt.NoPen)
        p.drawRoundedRect(0,20,self.width(),5,2,2)
        if self.value > 0:
            g = QLinearGradient(0,0,self.width()*self.value,0)
            g.setColorAt(0,self.color.darker(150)); g.setColorAt(1,self.color)
            p.setBrush(g); p.drawRoundedRect(0,20,int(self.width()*self.value),5,2,2)


# ── QThread workers ────────────────────────────────────────────────────────────

class EyeWorker(QThread):
    frame_ready    = pyqtSignal(np.ndarray, str, list)
    pick_confirmed = pyqtSignal()

    def __init__(self, cam_idx):
        super().__init__()
        self.cam_idx     = cam_idx
        self.running     = True
        self._buf        = deque(maxlen=15)
        self._conf       = -1
        self._sequence   = []
        self._last_seq_t = time.time()

    def _process(self, cls):
        self._buf.append(cls)
        if len(self._buf) >= 5:
            valid  = [x for x in self._buf if x != -1]
            counts = np.bincount(valid, minlength=5) if valid else np.zeros(5)
            new_s  = int(counts.argmax()) if counts.max() >= 4 else -1
            if new_s != self._conf:
                if new_s != -1 and (not self._sequence or self._sequence[-1] != new_s):
                    self._sequence.append(new_s)
                    self._last_seq_t = time.time()
                    self._check()
                self._conf = new_s
        if self._sequence and (time.time()-self._last_seq_t > 2.5):
            self._sequence = []

    def _check(self):
        if len(self._sequence) >= 2 and tuple(self._sequence[-2:]) == PICK_SEQUENCE:
            self.pick_confirmed.emit()
            self._sequence = []

    def run(self):
        cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            self.frame_ready.emit(np.zeros((480,640,3),dtype=np.uint8),'NO CAMERA',[0.]*5)
            return
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
            num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
        detector = mp_vision.FaceLandmarker.create_from_options(opts)
        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            h, w = frame.shape[:2]
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_img)
            disp = frame.copy(); gaze = 'IDLE'; levels = [0.]*5
            if result.face_landmarks:
                lm = result.face_landmarks[0]
                cls, conf = _classify_gaze(lm, w, h)
                gaze = GAZE_NAMES[cls]; levels[cls] = conf
                self._process(cls)
                eye_pts = [(int(lm[i].x*w), int(lm[i].y*h))
                           for i in [_L_OUTER,_L_INNER,_L_TOP,_L_BOT,
                                     _R_OUTER,_R_INNER,_R_TOP,_R_BOT]]
                pts = np.array(eye_pts)
                pad = 14
                x1,y1 = max(0,pts[:,0].min()-pad), max(0,pts[:,1].min()-pad)
                x2,y2 = min(w,pts[:,0].max()+pad), min(h,pts[:,1].max()+pad)
                col = (0,255,100) if cls==GAZE_STRAIGHT else (0,200,255)
                cv2.rectangle(disp,(x1,y1),(x2,y2),col,2)
                cv2.putText(disp,GAZE_SYMBOLS[cls],(x1,y1-8),
                            cv2.FONT_HERSHEY_DUPLEX,0.9,col,2,cv2.LINE_AA)
            col2 = (0,255,0) if gaze=='Straight' else (0,200,255)
            cv2.putText(disp,gaze,(10,28),cv2.FONT_HERSHEY_DUPLEX,0.8,col2,1)
            seq = ' > '.join(GAZE_NAMES[s] for s in self._sequence)
            cv2.putText(disp,seq,(10,54),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
            self.frame_ready.emit(disp, gaze, levels)
        cap.release(); detector.close()


class DepthWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, node):
        super().__init__()
        self._node = node; self.running = True

    def run(self):
        while self.running:
            depth, enc = self._node.get_depth_frame()
            if depth is not None:
                d = depth.astype(np.float32)
                if enc == '16UC1': d /= 1000.0
                d = np.clip(d, 0.1, 2.0)
                norm    = ((d - 0.1) / 1.9 * 255).astype(np.uint8)
                colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
                cv2.putText(colored, 'DEPTH', (8,24),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)
                self.frame_ready.emit(colored)
            self.msleep(66)


class FoodWorker(QThread):
    """Runs YOLO on RealSense color frames, emits annotated frame + target centroid pixel."""
    frame_ready = pyqtSignal(np.ndarray)   # annotated frame
    food_found  = pyqtSignal(int, int)     # centroid u, v (pixels) of target item

    def __init__(self, node):
        super().__init__()
        self._node   = node
        self.running = True
        self._model  = None

    def run(self):
        print('[FoodWorker] Loading YOLO model...')
        try:
            sys.path.insert(0, '/home/amma/coco_food')
            import food_cafg  # noqa — must import before YOLO loads CAFG weights
            from ultralytics import YOLO
            self._model = YOLO(YOLO_WEIGHTS)
            print('[FoodWorker] YOLO ready')
        except Exception as e:
            print(f'[FoodWorker] YOLO load failed: {e}')

        while self.running:
            frame, _, _, _ = self._node.get_display_data()
            if frame is None or self._model is None:
                self.msleep(100)
                continue

            h, w = frame.shape[:2]
            results = self._model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), verbose=False)
            disp    = frame.copy()

            target_set = False
            for r in results:
                if r.boxes is None: continue
                for i, box in enumerate(r.boxes):
                    x1,y1,x2,y2 = (int(v) for v in box.xyxy[0].tolist())
                    conf = float(box.conf)
                    is_target = (i == 0)
                    col = _ITEM_COLOR if is_target else (180,180,180)
                    cv2.rectangle(disp,(x1,y1),(x2,y2),col,2 if is_target else 1)
                    label = f'TARGET {conf:.2f}' if is_target else f'{conf:.2f}'
                    cv2.putText(disp,label,(x1,y1-8),
                                cv2.FONT_HERSHEY_DUPLEX,0.7,col,2 if is_target else 1,
                                cv2.LINE_AA)
                    if is_target and not target_set:
                        cu, cv_ = (x1+x2)//2, (y1+y2)//2
                        cv2.circle(disp,(cu,cv_),8,_ITEM_COLOR,-1)
                        cv2.circle(disp,(cu,cv_),11,(255,255,255),2)
                        self.food_found.emit(cu, cv_)
                        target_set = True

            if not target_set:
                cv2.putText(disp,'No food detected',(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,200),2)

            self.frame_ready.emit(disp)
            self.msleep(200)   # ~5 fps — YOLO on CPU is slow


# ── Main UI ────────────────────────────────────────────────────────────────────

class PickUI(QMainWindow):
    status_update = pyqtSignal(str)

    def __init__(self, node, eye_event, start_event, stop):
        super().__init__()
        self._node        = node
        self._eye_event   = eye_event
        self._start_event = start_event
        self._stop        = stop
        self._started     = False
        self.setWindowTitle("PickLoop — Kinova Picking System")
        self.resize(1400, 820)
        self.setStyleSheet(
            f"background-color:{C_BG}; color:{C_TEXT}; font-family:'Inter',sans-serif;")
        self._init_ui()
        self.status_update.connect(self._on_status)

        cam_idx = _auto_webcam_index()
        self._eye_worker = EyeWorker(cam_idx)
        self._eye_worker.frame_ready.connect(self._on_eye_frame)
        self._eye_worker.pick_confirmed.connect(self._on_pick_confirmed)
        self._eye_worker.start()

        self._food_worker = FoodWorker(node)
        self._food_worker.frame_ready.connect(self._on_food_frame)
        self._food_worker.food_found.connect(node.set_food_target)
        self._food_worker.start()

        self._depth_worker = DepthWorker(node)
        self._depth_worker.frame_ready.connect(self._on_depth_frame)
        self._depth_worker.start()

    def _init_ui(self):
        central = QWidget()
        root    = QHBoxLayout(central)
        root.setContentsMargins(16,16,16,16); root.setSpacing(16)

        # Left: eye camera
        left = QVBoxLayout()
        eye_panel = GlassPanel(glow=True)
        eye_v = QVBoxLayout(eye_panel); eye_v.setContentsMargins(8,8,8,8)
        QLabel("EYE CAMERA", styleSheet=f"color:{C_TEXT_DIM}; font-size:10px; letter-spacing:2px;")
        lbl_eh = QLabel("EYE CAMERA"); lbl_eh.setStyleSheet(
            f"color:{C_TEXT_DIM}; font-size:10px; letter-spacing:2px;")
        eye_v.addWidget(lbl_eh)
        self.lbl_eye = QLabel("INITIALIZING..."); self.lbl_eye.setAlignment(Qt.AlignCenter)
        self.lbl_eye.setFixedSize(VIDEO_W, VIDEO_H)
        eye_v.addWidget(self.lbl_eye, alignment=Qt.AlignCenter)
        left.addWidget(eye_panel)

        meter_panel = GlassPanel()
        meter_v = QVBoxLayout(meter_panel)
        lbl_m = QLabel("GAZE SIGNALS"); lbl_m.setStyleSheet(
            f"color:{C_TEXT_DIM}; font-size:10px; letter-spacing:2px;")
        meter_v.addWidget(lbl_m)
        self._meters = [SignalMeter(f"{GAZE_SYMBOLS[i]}  {GAZE_NAMES[i]}", C_ACCENT)
                        for i in range(5)]
        for m in self._meters: meter_v.addWidget(m)
        left.addWidget(meter_panel)

        guide = GlassPanel()
        gv = QVBoxLayout(guide)
        lbl_g = QLabel("COMMAND"); lbl_g.setStyleSheet(
            f"color:{C_TEXT_DIM}; font-size:10px; letter-spacing:2px;")
        gv.addWidget(lbl_g)
        row = QLabel(f"<span style='color:{C_ACCENT}; font-size:18px;'>LEFT  UP</span>"
                     f"  <span style='color:{C_TEXT};'>Confirm Pick</span>")
        row.setTextFormat(Qt.RichText); gv.addWidget(row)
        left.addWidget(guide)
        root.addLayout(left, 3)

        # Centre: food camera
        mid = QVBoxLayout()
        food_panel = GlassPanel(glow=True)
        food_v = QVBoxLayout(food_panel); food_v.setContentsMargins(8,8,8,8)
        lbl_fh = QLabel("FOOD CAMERA (RealSense + YOLO)"); lbl_fh.setStyleSheet(
            f"color:{C_TEXT_DIM}; font-size:10px; letter-spacing:2px;")
        food_v.addWidget(lbl_fh)
        self.lbl_food = QLabel("Loading YOLO model..."); self.lbl_food.setAlignment(Qt.AlignCenter)
        self.lbl_food.setFixedSize(VIDEO_W, VIDEO_H)
        food_v.addWidget(self.lbl_food, alignment=Qt.AlignCenter)
        mid.addWidget(food_panel)

        depth_panel = GlassPanel()
        depth_v     = QVBoxLayout(depth_panel); depth_v.setContentsMargins(8,8,8,8)
        lbl_dh = QLabel("DEPTH CAMERA (RealSense)"); lbl_dh.setStyleSheet(
            f"color:{C_TEXT_DIM}; font-size:10px; letter-spacing:2px;")
        depth_v.addWidget(lbl_dh)
        self.lbl_depth = QLabel("Waiting for depth...")
        self.lbl_depth.setAlignment(Qt.AlignCenter)
        self.lbl_depth.setFixedSize(VIDEO_W, VIDEO_H // 2)
        depth_v.addWidget(self.lbl_depth, alignment=Qt.AlignCenter)
        mid.addWidget(depth_panel)
        root.addLayout(mid, 3)

        # Right: status + log
        right = QVBoxLayout()
        cmd_panel = GlassPanel(glow=True)
        cmd_v = QVBoxLayout(cmd_panel)
        lbl_ch = QLabel("PICK STATUS"); lbl_ch.setStyleSheet(
            f"color:{C_TEXT_DIM}; font-size:10px; letter-spacing:2px;")
        cmd_v.addWidget(lbl_ch)
        self.lbl_status = QLabel("INITIALISING")
        self.lbl_status.setFont(QFont("Inter", 22, QFont.Bold))
        self.lbl_status.setStyleSheet(f"color:{C_ACCENT};")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        cmd_v.addWidget(self.lbl_status)
        self.lbl_gaze = QLabel("GAZE: —")
        self.lbl_gaze.setFont(QFont("Inter",13))
        self.lbl_gaze.setStyleSheet(f"color:{C_TEXT_DIM};")
        self.lbl_gaze.setAlignment(Qt.AlignCenter)
        cmd_v.addWidget(self.lbl_gaze)

        self.btn_start = QPushButton("▶  START PICK LOOP")
        self.btn_start.setFont(QFont("Inter",13,QFont.Bold))
        self.btn_start.setStyleSheet(f"""
            QPushButton {{ background-color:{C_SUCCESS}; color:#0b0e14;
                           border-radius:8px; padding:10px; }}
            QPushButton:hover {{ background-color:#56d364; }}
        """)
        self.btn_start.clicked.connect(self._on_start_clicked)
        cmd_v.addWidget(self.btn_start)
        right.addWidget(cmd_panel, 2)

        log_panel = GlassPanel()
        log_v = QVBoxLayout(log_panel)
        lbl_lh = QLabel("ACTIVITY LOG"); lbl_lh.setStyleSheet(
            f"color:{C_TEXT_DIM}; font-size:10px; letter-spacing:2px;")
        log_v.addWidget(lbl_lh)
        self._log = QListWidget()
        self._log.setStyleSheet(
            f"border:none; background:transparent; color:{C_TEXT}; font-size:12px;")
        log_v.addWidget(self._log)
        right.addWidget(log_panel, 3)
        root.addLayout(right, 2)
        self.setCentralWidget(central)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_eye_frame(self, frame, gaze, levels):
        self.lbl_eye.setPixmap(_frame_to_pixmap(frame, VIDEO_W, VIDEO_H))
        self.lbl_gaze.setText(f"GAZE: {gaze}")
        for i, v in enumerate(levels): self._meters[i].set_value(v)

    def _on_food_frame(self, frame):
        self.lbl_food.setPixmap(_frame_to_pixmap(frame, VIDEO_W, VIDEO_H))

    def _on_depth_frame(self, frame):
        self.lbl_depth.setPixmap(_frame_to_pixmap(frame, VIDEO_W, VIDEO_H // 2))

    def _on_pick_confirmed(self):
        if not self._started:
            self._on_start_clicked(); return
        self._eye_event.set()
        self._log_event('Eye sequence → CONFIRM PICK')
        self.lbl_status.setStyleSheet(f"color:{C_SUCCESS};")
        QTimer.singleShot(1500, lambda: self.lbl_status.setStyleSheet(f"color:{C_ACCENT};"))

    def _on_start_clicked(self):
        if not self._started:
            self._started = True
            self.btn_start.setText("■  RUNNING")
            self.btn_start.setStyleSheet(f"""
                QPushButton {{ background-color:{C_DANGER}; color:#fff;
                               border-radius:8px; padding:10px; }}
                QPushButton:hover {{ background-color:#ff6b6b; }}
            """)
            self.btn_start.clicked.disconnect()
            self.btn_start.clicked.connect(self._on_stop_clicked)
            self._start_event.set()
            self._log_event('Pick loop STARTED')

    def _on_stop_clicked(self):
        self._stop.set(); self._eye_event.set(); self._start_event.set()
        self.btn_start.setText("STOPPED"); self.btn_start.setEnabled(False)
        self._log_event('Pick loop STOPPED')

    def _on_status(self, msg):
        self.lbl_status.setText(msg.upper())
        self._log_event(msg)
        col = C_SUCCESS if 'LOOK' in msg.upper() else C_ACCENT
        self.lbl_status.setStyleSheet(f"color:{col};")

    def _log_event(self, msg):
        self._log.insertItem(0, f"[{time.strftime('%H:%M:%S')}]  {msg}")
        if self._log.count() > 60: self._log.takeItem(self._log.count()-1)

    def set_status(self, msg): self.status_update.emit(msg)

    def closeEvent(self, e):
        self._stop.set(); self._eye_event.set()
        self._eye_worker.running   = False
        self._food_worker.running  = False
        self._depth_worker.running = False
        self._eye_worker.wait(2000)
        self._food_worker.wait(2000)
        self._depth_worker.wait(2000)
        e.accept()


# ── ROS Node ───────────────────────────────────────────────────────────────────

class PickLoop(Node):
    def __init__(self):
        super().__init__('pick_loop')
        cb = ReentrantCallbackGroup()
        self.declare_parameter('cam_offset_x', -0.14)
        self.declare_parameter('cam_offset_y', -0.21)
        self.declare_parameter('cam_offset_z', -0.41)  # -0.15 - 0.26

        self._traj_client = ActionClient(
            self, FollowJointTrajectory, ACTION_TOPIC, callback_group=cb)
        self._latest_q = None
        self.create_subscription(JointState, JOINT_STATE_TOPIC,
                                 self._js_cb, 10, callback_group=cb)

        self._bridge       = CvBridge()
        self._lock         = threading.Lock()
        self._K            = None
        self._color_frame  = None
        self._depth_frame  = None
        self._depth_enc    = '16UC1'
        self._food_target  = None   # (u, v) pixel of target food item

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._goal_pub    = self.create_publisher(PointStamped, '/goal_point', 10)

        self.create_subscription(CameraInfo,
            '/camera/camera/color/camera_info', self._info_cb, 1, callback_group=cb)
        self.create_subscription(Image,
            '/camera/camera/color/image_raw', self._color_cb, 1, callback_group=cb)
        self.create_subscription(Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self._depth_cb, 1, callback_group=cb)

    def _js_cb(self, msg):
        try: idx = [msg.name.index(j) for j in JOINT_NAMES]
        except ValueError: return
        self._latest_q = [msg.position[i] for i in idx]

    def _info_cb(self, msg):
        with self._lock:
            if self._K is None:
                self._K = np.array(msg.k, dtype=np.float64).reshape(3,3)

    def _color_cb(self, msg):
        with self._lock:
            self._color_frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _depth_cb(self, msg):
        with self._lock:
            self._depth_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self._depth_enc   = msg.encoding

    def get_display_data(self):
        with self._lock:
            return self._color_frame, None, None, None

    def get_depth_frame(self):
        with self._lock:
            return self._depth_frame, self._depth_enc

    def set_food_target(self, u, v):
        with self._lock:
            self._food_target = (u, v)

    def _wait_js(self, timeout=5.0):
        t0 = time.monotonic()
        while self._latest_q is None and (time.monotonic()-t0) < timeout:
            time.sleep(0.05)
        return self._latest_q is not None

    def move_joint_space(self, target_rad, duration_s, label):
        self.get_logger().info(f'[{label}] moving...')
        if not self._traj_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Trajectory server unavailable'); return False
        positions = list(target_rad)
        if self._wait_js():
            positions = [t - round((t-c)/TWO_PI)*TWO_PI
                         for c, t in zip(self._latest_q, target_rad)]
        pt = JointTrajectoryPoint()
        pt.positions = positions; pt.velocities = [0.]*NJ
        sec = int(duration_s)
        pt.time_from_start = DurationMsg(sec=sec, nanosec=int((duration_s-sec)*1e9))
        traj = JointTrajectory(); traj.joint_names = JOINT_NAMES; traj.points = [pt]
        goal = FollowJointTrajectory.Goal(); goal.trajectory = traj
        done, ok = threading.Event(), [False]
        def _gc(fut):
            gh = fut.result()
            if not gh.accepted: done.set(); return
            def _rc(r):
                ok[0] = (r.result().result.error_code ==
                         FollowJointTrajectory.Result.SUCCESSFUL)
                done.set()
            gh.get_result_async().add_done_callback(_rc)
        self._traj_client.send_goal_async(goal).add_done_callback(_gc)
        done.wait(timeout=duration_s+15.0)
        self.get_logger().info(f'[{label}] {"done" if ok[0] else "FAILED"}')
        return ok[0]

    def publish_food_goal(self):
        with self._lock:
            ftgt  = self._food_target
            depth = self._depth_frame
            K     = self._K
            enc   = self._depth_enc

        if ftgt is None or depth is None or K is None:
            self.get_logger().warn('No food target / camera data')
            return False

        u, v = ftgt; h, w = depth.shape[:2]
        patch = depth[max(0,v-4):min(h,v+5), max(0,u-4):min(w,u+5)].astype(np.float32)
        valid = patch[patch>0]
        if len(valid) < 4:
            self.get_logger().warn('Bad depth at food target')
            return False
        depth_m = float(np.median(valid))/1000. if enc=='16UC1' else float(np.median(valid))
        if not (0.1 < depth_m < 5.0): return False

        fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
        pt = PointStamped()
        pt.header.frame_id = 'camera_color_optical_frame'
        pt.header.stamp    = rclpy.time.Time().to_msg()
        pt.point.x = (u-cx)*depth_m/fx + self.get_parameter('cam_offset_x').value
        pt.point.y = (v-cy)*depth_m/fy + self.get_parameter('cam_offset_y').value
        pt.point.z = depth_m            + self.get_parameter('cam_offset_z').value

        try:
            pt_root = self._tf_buffer.transform(pt,'root',timeout=Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'TF: {e}'); return False

        self._goal_pub.publish(pt_root)
        self.get_logger().info(
            f'Food → ({pt_root.point.x:.3f},{pt_root.point.y:.3f},{pt_root.point.z:.3f})')
        return True


# ── Loop thread ────────────────────────────────────────────────────────────────

def run_loop(node: PickLoop, ui: PickUI,
             eye_event: threading.Event, start_event: threading.Event,
             stop: threading.Event):
    ui.set_status('Press START to begin')
    start_event.wait()
    if stop.is_set(): return

    cycle = 0
    while rclpy.ok() and not stop.is_set():
        cycle += 1
        node.get_logger().info(f'══════ CYCLE {cycle} ══════')

        # Move to pick position
        ui.set_status('Moving to pick position...')
        if not node.move_joint_space(PICK_RAD, DUR_STEP, 'TO_PICK'):
            ui.set_status('FAILED: pick position'); stop.set(); return

        # Wait for eye confirm → arm to food
        ui.set_status('Look LEFT then UP to pick')
        while rclpy.ok() and not stop.is_set():
            eye_event.clear(); eye_event.wait()
            if stop.is_set(): return
            ui.set_status('Moving arm to food...')
            if node.publish_food_goal():
                break
            ui.set_status('No food detected — look LEFT then UP again')

        # Wait for arm_mover_node to reach the food
        time.sleep(8.0)
        if stop.is_set(): return

        # Scoop sequence
        for label, pos in zip(SCOOP_LABELS, SCOOP_SEQUENCE):
            if stop.is_set(): return
            ui.set_status(f'Scooping: {label}')
            if not node.move_joint_space(pos, DUR_STEP, label):
                ui.set_status(f'FAILED: {label}'); stop.set(); return

        # Wait for next eye confirm before next cycle
        ui.set_status('Look LEFT then UP for next cycle')
        eye_event.clear(); eye_event.wait()
        if stop.is_set(): return

        node.get_logger().info(f'Cycle {cycle} complete')


# ── Shutdown ───────────────────────────────────────────────────────────────────

def _shutdown(node, eye_event, start_event, stop, ui=None):
    print('\n\033[96m' + '═'*50)
    print('  Kinova PickLoop — shutting down cleanly')
    print('  Thank you for using the picking system  ')
    print('═'*50 + '\033[0m')
    stop.set(); eye_event.set(); start_event.set()
    if ui:
        try: ui.set_status('Shutting down...')
        except Exception: pass
    try: node.destroy_node()
    except Exception: pass
    try: rclpy.shutdown()
    except Exception: pass
    print('\033[92m  Goodbye!\033[0m\n')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import signal
    rclpy.init(args=sys.argv)
    node = PickLoop()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    eye_event   = threading.Event()
    start_event = threading.Event()
    stop        = threading.Event()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ui = PickUI(node, eye_event, start_event, stop)
    ui.show()

    def _sigint(sig, frame):
        _shutdown(node, eye_event, start_event, stop, ui)
        app.quit()
    signal.signal(signal.SIGINT, _sigint)
    timer = QTimer(); timer.timeout.connect(lambda: None); timer.start(200)

    threading.Thread(
        target=run_loop, args=(node, ui, eye_event, start_event, stop),
        daemon=True).start()

    ret = app.exec_()
    _shutdown(node, eye_event, start_event, stop)
    sys.exit(ret)


if __name__ == '__main__':
    main()
