# click_pointer — context for Claude

## Algorithm

### Step 1: Pixel → Camera-frame 3-D target

Standard pinhole back-projection:

    X_cam = (u − cx) · Z / fx
    Y_cam = (v − cy) · Z / fy
    Z_cam = Z_mm · 0.001

Depth is already aligned to the colour frame by the ROS driver — no rectification needed.

`CAM_Y_OFFSET = 0.09 m` is subtracted from Y_cam because the camera is mounted ~9 cm above the EE tip.
Camera Y points downward, so subtracting moves the aim-point down toward the EE.

### Step 2: Confirmation gate

Target is stored but NOT executed immediately. Operator presses Y/N to prevent accidental motion
from mis-aimed clicks at background clutter.

### Step 3: Phased proportional joint stepping

Error decomposed into three camera-frame axes, each corrected by the joint(s) whose primary coupling
is to that axis at the arm's nominal pose:

- **Phase 1 — Y-axis (vertical) → J2 (elbow pitch)**
  Cleared first because vertical error is largest at meal time and must be settled before X
  to avoid shoulder singularities when the arm is outstretched.

- **Phase 2 — X-axis (horizontal) → J3 (forearm rotation)**
  Uses camera-origin X, not EE-relative X: `endpoint[0]` is already `(u_click − CX) × Z / FX`,
  which is zero when the target is centred in the image — no TF lookup needed for the EE's X.

- **Phase 3 — Z-axis (depth) → J1 (base rotation) + J4 (wrist pitch)**
  J1 and J4 move simultaneously so wrist compensation is continuous, keeping the fork level.
  Only forward approach (`dz > 0`) is automated; retract is done by reset.

### Control law

    delta_rad = (error_m / CART_DELTA) * STEP_RADS
    duration  = max(MIN_DUR, |error| / CART_DELTA * TICK_DUR_S)

CART_DELTA constants are tuned empirically so one full STEP_RADS on J2/J3/J1 moves the EE
by CART_DELTA metres in the corresponding camera axis.

### Why open-loop position tracking?

Live TF queries inside the control loop add ≥100 ms latency per step. At TICK_DUR_S = 0.6 s
that dominates settling time and halves throughput. The total joint delta is computed analytically
from the initial error instead. Valid because the arm operates in a narrow meal-time workspace
and only needs ~5 cm accuracy (ERR_TOL_M).

---

## Design notes

### Neon face overlay (`face_overlay.py`)

Glow effect: draw coloured strokes on a black canvas, GaussianBlur it (kernel 17×17, ~8 px spread),
then `addWeighted` over the camera frame. Sharp crisp lines drawn on top for readability.
This simulates neon tube light diffusion using pure CPU OpenCV — no per-line alpha blending.

Iris radius = distance from iris centre landmark to one edge landmark, encoding visible iris diameter.

Animated scan line: sweeps top-to-bottom across the face bounding box every 80 frames.
Alpha fades via `sin(t×π)` to avoid hard appearance/disappearance at the boundaries.

### `_dispatch_smooth` — why it blocks

Blocking until the bridge action server reports completion is critical. Without it, Phase 2 and
Phase 3 execute concurrently (bridge runs concurrent goals with `ReentrantCallbackGroup`), causing
two control loops to fight over the arm simultaneously.

### Qt / stderr suppression

`QT_LOGGING_RULES=qt.*=false` and `OPENCV_LOG_LEVEL=ERROR` suppress Qt/OpenCV noise before any
window is created. `QT_QPA_FONTDIR` is pointed at a real system font dir to stop cv2 pip-package
font warnings on every Qt draw call. During the UI loop stderr is also redirected to `/dev/null`
since the TUI uses only stdout.

### Trajectory two-point requirement

Trajectories are sent as current → target (two points, not one) so the controller generates a
velocity-limited trapezoid profile rather than a step discontinuity.

### HUD colour semantics

- Cyan/teal = informational / idle
- Green = confirmed / complete
- Amber/yellow = awaiting operator decision
- Blue = active motion (FEEDING)
