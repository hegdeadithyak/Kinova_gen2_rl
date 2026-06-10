#!/usr/bin/env bash
# Bring up the full hardware stack and run the mouth-targeting bbox planner.
#
#   Terminal A : arm + TF + trajectory action server (kinova bridge + MoveIt)
#   Terminal B : RealSense with aligned depth
#   Terminal C : bbox_planner.py (mouth target)
#
# All three are started from this one script: the two launches run in the
# background (logging to /tmp), and the planner runs in the foreground.
# Ctrl-C (or the planner exiting) tears everything down cleanly.
#
# Usage:
#   ./run_bbox_planner.sh                 # default standoff 0.25 m, source=mouth
#   ./run_bbox_planner.sh --desired-z 0.20 --source mouth
#   ./run_bbox_planner.sh --source drag   # any extra args pass through to the planner

ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="$HOME/Kinova_gen2_rl/install/setup.bash"
WORKDIR="$HOME/Kinova_gen2_rl"
LOG_DIR="/tmp/bbox_planner_run"
mkdir -p "$LOG_DIR"

# Planner args: default if none given, else pass everything through.
if [ "$#" -eq 0 ]; then
    PLANNER_ARGS=(--source mouth --desired-z 0.25)
else
    PLANNER_ARGS=("$@")
fi

# ROS setup scripts reference unbound vars internally, so source with nounset off.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1090
source "$WS_SETUP"

PIDS=()
cleanup() {
    echo
    echo "[run] shutting down hardware stack ..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    # give launches a moment, then hard-kill any stragglers
    sleep 2
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
    echo "[run] done."
}
trap cleanup EXIT INT TERM

# helper: block until a condition command succeeds (defined early, used below)
wait_for() {  # description, command...
    local desc="$1"; shift
    local t=0
    until "$@" >/dev/null 2>&1; do
        sleep 1; t=$((t + 1))
        if [ $((t % 5)) -eq 0 ]; then echo "[run]   still waiting for $desc (${t}s) ..."; fi
        if [ "$t" -ge 90 ]; then
            echo "[run] ERROR: timed out waiting for $desc. Check $LOG_DIR/*.log"
            exit 1
        fi
    done
    echo "[run]   ✓ $desc"
}

# ── Terminal 0 : Kinova hardware driver (connects to the physical arm) ────────
echo "[run] launching Kinova driver (log: $LOG_DIR/driver.log) ..."
ros2 launch kinova_bringup kinova_robot_launch.py \
    > "$LOG_DIR/driver.log" 2>&1 &
PIDS+=($!)
wait_for "Kinova driver (joint feedback)" \
    ros2 topic info /j2s6s200_driver/out/joint_state

# ── Terminal A : arm + TF + trajectory action server ─────────────────────────
echo "[run] launching MoveIt + TF + trajectory bridge (log: $LOG_DIR/arm.log) ..."
ros2 launch j2s6s200_moveit_config real_moveit_launch.py \
    > "$LOG_DIR/arm.log" 2>&1 &
PIDS+=($!)

# ── Terminal B : RealSense with aligned depth ────────────────────────────────
echo "[run] launching RealSense (log: $LOG_DIR/camera.log) ..."
ros2 launch realsense2_camera rs_launch.py \
    enable_color:=true enable_depth:=true align_depth.enable:=true \
    rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30 \
    > "$LOG_DIR/camera.log" 2>&1 &
PIDS+=($!)

# ── Wait for the rest of the stack to come up ────────────────────────────────
echo "[run] waiting for camera, joint states and the trajectory action ..."
wait_for "aligned depth image" \
    ros2 topic info /camera/camera/aligned_depth_to_color/image_raw
wait_for "color image" \
    ros2 topic info /camera/camera/color/image_raw
wait_for "joint states" \
    ros2 topic info /joint_states
wait_for "trajectory action server" \
    bash -c "ros2 action list | grep -q /arm_controller/follow_joint_trajectory"

# ── Terminal C : the planner (foreground) ────────────────────────────────────
echo "[run] starting bbox_planner.py ${PLANNER_ARGS[*]}"
echo "[run] (sit in view, then press ENTER in the window to servo; q to quit)"
cd "$WORKDIR" || exit 1
python3 bbox_planner.py "${PLANNER_ARGS[@]}"
