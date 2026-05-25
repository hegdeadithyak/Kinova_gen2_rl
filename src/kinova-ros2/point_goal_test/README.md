# point_goal_test

Click-to-feed demo for the Kinova Gen2 assistive feeding robot.

A RealSense depth frame is shown in an OpenCV window. Click any pixel to move the arm end-effector (spoon) to the corresponding 3D point while keeping the spoon orientation locked throughout the entire path.

## Architecture

```
RealSense camera
       │
point_selector_node  (Python)
  — click pixel → 3D point via depth + TF
       │  /goal_point  (geometry_msgs/PointStamped)
       ▼
feeding_safe_mover  (C++)
  1. Lock current end-effector orientation
  2. Set OrientationConstraint as MoveIt 2 path constraint
  3. Plan with RRTConnectkConfigDefault
  4. Validate wrist joints (j5, j6) at every waypoint
  5. Execute only if planning and validation both pass
       │
MoveIt 2 move_group → FollowJointTrajectory → robot
```

## Build

```bash
cd ~/Kinova_gen2_rl
colcon build --packages-select point_goal_test --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Run (C++ safety mover)

```bash
ros2 launch point_goal_test feeding_safe.launch.py
```

Optional launch arguments:

```bash
ros2 launch point_goal_test feeding_safe.launch.py \
  offset_x:=0.0 offset_y:=0.0 offset_z:=0.05
```

## Parameter override for j2n6s300

If your robot uses the j2n6s300 model instead of j2s6s200, override the link and joint names:

```bash
ros2 run point_goal_test feeding_safe_mover --ros-args \
  -p ee_link:=j2n6s300_end_effector \
  -p joint_5:=j2n6s300_joint_5 \
  -p joint_6:=j2n6s300_joint_6
```

## All configurable parameters

| Parameter | Default | Description |
|---|---|---|
| `base_frame` | `root` | Planning reference frame |
| `ee_link` | `j2s6s200_end_effector` | End-effector link name |
| `joint_5` | `j2s6s200_joint_5` | Wrist roll joint for safety check |
| `joint_6` | `j2s6s200_joint_6` | Wrist pitch joint for safety check |
| `offset_x/y/z` | `0.0` | Goal-point offsets in metres |
| `ori_tol_x` | `0.05` | Orientation path constraint tolerance x (rad) |
| `ori_tol_y` | `0.05` | Orientation path constraint tolerance y (rad) |
| `ori_tol_z` | `0.20` | Orientation path constraint tolerance z (rad) |
| `wrist_tol` | `0.08` | Max wrist deviation per joint (rad); limit is ×1.5 |
| `planning_time` | `10.0` | OMPL planning timeout (s) |
| `velocity_scale` | `0.15` | Max velocity scaling (0–1) |
| `acceleration_scale` | `0.15` | Max acceleration scaling (0–1) |
| `enable_joint_constraints` | `false` | Also add hard joint constraints for j5/j6 (can cause planning failures; keep off unless needed) |

## Legacy Python mover

The original `arm_mover_node` (pymoveit2) is still available:

```bash
ros2 launch point_goal_test point_goal.launch.py
```

## Safety design notes

- The `OrientationConstraint` is registered as a **path** constraint via `setPathConstraints()`, not a goal constraint. This forces every OMPL sample to respect the orientation tolerance on every waypoint, not only at the final pose.
- `enforce_constrained_state_space: true` in `ompl_planning.yaml` makes OMPL project samples onto the constraint manifold (faster convergence than rejection sampling).
- A position-only planning fallback is intentionally absent. If planning fails with the orientation constraint active, the arm does not move.
- Every planned trajectory is validated before execution: if any waypoint moves `joint_5` or `joint_6` more than `wrist_tol × 1.5` from their starting positions the trajectory is discarded.
