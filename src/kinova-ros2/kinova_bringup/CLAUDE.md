# kinova_bringup — context for Claude

## kinova_trajectory_bridge.py

### Two-phase controller

The control loop has two phases:

- **TRACK** (`t < t_end`): `v = v_ff + KP*err` — no integral. Tracking lag during a ramp is
  expected and would wind up the integrator.
- **SETTLE** (`t >= t_end`): `v = KP*err + KI*∫err` — integral kicks in only on a static target
  to reject gravity load. Anti-windup clamp prevents integrator saturation.

### 100 Hz heartbeat

The Kinova DSP has a watchdog that halts the arm if it stops receiving velocity commands.
The 100 Hz timer publishes whatever velocity is currently commanded (zero when idle) to
keep the watchdog fed regardless of whether a trajectory is running.

### 2π angle wrap

MoveIt plans in a different 2π cycle than the driver reports (e.g. planner uses [-π, π],
driver reports [0, 2π]). On each new goal, per-joint integer-2π offsets are computed so
the first waypoint aligns with the measured position, then applied to all waypoints.

### Synthetic t=0 waypoint

Single-waypoint goals (just a target, no start) would cause the controller to step
instantaneously from wherever the arm is to the target. Inserting a synthetic t=0 waypoint
from the measured pose makes the interpolator generate a smooth trajectory instead.

---

## kinova_sdk_node.py

### Angle convention

SDK returns 0–360 degrees. URDF/MoveIt expects −π..+π radians. `_sdk_deg_to_urdf_rad` wraps
via `atan2(sin, cos)` — without this, joints near 180–360° publish values off by up to 2π,
causing MoveIt to plan a full extra revolution.

### JOINT_DIRECTION

Some joints have opposite sign between the SDK and URDF. `JOINT_DIRECTION[i]` is multiplied
on read (SDK→URDF) and write (URDF→SDK). If a joint moves the wrong way, flip its entry.

### Finger tips

Finger tip joints are not independently readable from the SDK. They are mirrored at 50% of
the corresponding finger joint angle. This is a known approximation; exact values would
require a custom firmware query.

### Planning scene sync at t=12s

`move_group` starts at approximately t=8s. The planning scene push is deferred to t=12s
to give MoveIt 4 seconds to become ready. Without this, MoveIt shows the URDF default pose
until `CurrentStateMonitor` catches up, which can cause IK to solve from a wrong start.

---

## demo_feed_planner.py

### Strategy

1. IK is seeded from the **current** joint state so MoveIt never picks a far-away kinematic
   configuration (elbow-flip, wrist-flip, etc.).
2. **RRTstar** with `PathLengthOptimizationObjective` is tried first (anytime, near-optimal).
   The remaining time after finding a first solution is used for OMPL's internal simplifier.
3. **RRTConnect + simplify** is used as a fallback when RRTstar fails.
4. Return motion uses the same strategy.

### Approach standoff

The approach pose backs off `APPROACH_STANDOFF` metres along the base-frame Z axis before
the final feed move. If the arm approaches from a different direction, adjust the axis in
`_feed_cb` where `approach_pose.position.z` is modified.
