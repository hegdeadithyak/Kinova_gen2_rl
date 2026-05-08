import math
import cv2
import numpy as np

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

EYE_CONN = [
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),
    (154,155),(155,133),(133,173),(173,157),(157,158),(158,159),
    (159,160),(160,161),(161,246),(246,33),
    (263,249),(249,390),(390,373),(373,374),(374,380),(380,381),
    (381,382),(382,362),(362,398),(398,384),(384,385),(385,386),
    (386,387),(387,388),(388,466),(466,263),
]

EYEBROW_CONN = [
    (46,53),(53,52),(52,65),(65,55),(55,70),(70,63),(63,105),(105,66),(66,107),
    (276,283),(283,282),(282,295),(295,285),(285,300),(300,293),(293,334),(334,296),(296,336),
]

NOSE_CONN = [
    (168,6),(6,197),(197,195),(195,5),(5,4),(4,1),(1,19),(19,94),(94,2),
]


def build_face_mesh():
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
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

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
            all_faces = [LandmarkList([Landmark(lm.x, lm.y, lm.z) for lm in face])
                         for face in result.face_landmarks]
            return type("R", (), {"multi_face_landmarks": all_faces})()

    return Wrapper()


def draw_face_overlay(disp: np.ndarray, lms, hw, frame_count: int) -> np.ndarray:
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

    OVAL_C = (30,  210, 255)
    EYE_C  = (50,  255, 160)
    BROW_C = (255, 60,  220)
    NOSE_C = (50,  190, 255)
    LIP_C  = (0,   120, 255)

    # Glow pass: thick strokes on black canvas, blurred, then blended over frame
    glow = np.zeros_like(disp)
    seg(glow, FACE_OVAL_CONN, OVAL_C, 4)
    seg(glow, EYE_CONN,       EYE_C,  3)
    seg(glow, EYEBROW_CONN,   BROW_C, 3)
    seg(glow, NOSE_CONN,      NOSE_C, 2)
    seg(glow, LIP_CONN,       LIP_C,  3)
    glow = cv2.GaussianBlur(glow, (17, 17), 0)
    cv2.addWeighted(disp, 1.0, glow, 0.70, 0, disp)

    seg(disp, FACE_OVAL_CONN, (60,  235, 255), 1)
    seg(disp, EYE_CONN,       (80,  255, 180), 1)
    seg(disp, EYEBROW_CONN,   (255, 80,  240), 1)
    seg(disp, NOSE_CONN,      (80,  210, 255), 1)
    seg(disp, LIP_CONN,       (0,   180, 255), 1)

    for conn in (FACE_OVAL_CONN, EYE_CONN, EYEBROW_CONN):
        seen = set()
        for a, b in conn:
            for i in (a, b):
                if i < n and i not in seen:
                    cv2.circle(disp, px(i), 1, (220, 255, 255), -1, cv2.LINE_AA)
                    seen.add(i)

    def draw_iris(center_i, edge_i, max_i):
        if max_i >= n:
            return
        c, e = px(center_i), px(edge_i)
        r = max(3, int(math.hypot(c[0] - e[0], c[1] - e[1])))
        cv2.circle(disp, c, r + 5, (20, 50, 50), 1, cv2.LINE_AA)
        cv2.circle(disp, c, r, (220, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(disp, c, max(2, r // 3), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(disp, (c[0] - r//5, c[1] - r//5), max(1, r//6), (255, 255, 255), -1, cv2.LINE_AA)

    draw_iris(468, 469, 472)
    draw_iris(473, 474, 477)

    oval_px = [px(a) for a, _ in FACE_OVAL_CONN if a < n]
    if oval_px:
        fy0 = min(p[1] for p in oval_px); fy1 = max(p[1] for p in oval_px)
        fx0 = min(p[0] for p in oval_px); fx1 = max(p[0] for p in oval_px)
        period = 80
        t = (frame_count % period) / period
        sy = int(fy0 + (fy1 - fy0) * t)
        alpha = max(0.0, math.sin(t * math.pi))
        if alpha > 0.05:
            scan_col = (int(60 * alpha), int(220 * alpha), int(60 * alpha))
            cv2.line(disp, (fx0, sy), (fx1, sy), scan_col, 1, cv2.LINE_AA)

    mc_xs = [lms[i].x * mw for i in MOUTH_CENTER_IDS if i < n]
    mc_ys = [lms[i].y * mh for i in MOUTH_CENTER_IDS if i < n]
    if mc_xs:
        mx, my = int(np.mean(mc_xs)), int(np.mean(mc_ys))
        for r, b in [(28, 20), (21, 50), (15, 90)]:
            cv2.circle(disp, (mx, my), r, (0, b, b // 2), 1, cv2.LINE_AA)
        pr = int(11 + 4 * math.sin(frame_count * 0.10))
        cv2.circle(disp, (mx, my), pr, (0, 255, 180), 2, cv2.LINE_AA)
        cv2.circle(disp, (mx, my), 5, (50, 255, 120), -1, cv2.LINE_AA)
        cv2.putText(disp, "MOUTH", (mx - 22, my - 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 230, 180), 1, cv2.LINE_AA)

    return disp
