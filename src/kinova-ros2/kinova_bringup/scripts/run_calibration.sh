#!/bin/bash
# Wrapper to launch eye-in-hand calibration with all workspaces sourced.
set -e

WS_ROOT="$(cd "$(dirname "$0")/../../../../../.." && pwd)"
WS_MOVEIT="$WS_ROOT/kinova_ws/rl_v2-master/ws_moveit"
WS_MAIN="$WS_ROOT/kinova_ws/rl_v2-master"

echo "[calibration] Sourcing ROS2 Humble..."
source /opt/ros/humble/setup.bash

echo "[calibration] Sourcing main workspace: $WS_MAIN"
source "$WS_MAIN/install/setup.bash"

echo "[calibration] Sourcing moveit_calibration overlay: $WS_MOVEIT"
source "$WS_MOVEIT/install/setup.bash"

echo "[calibration] Verifying plugin..."
python3 -c "
from ament_index_python.packages import get_resource
content, _ = get_resource('rviz_common__pluginlib__plugin', 'moveit_calibration_gui')
print('[calibration] Plugin found:', content)
" || { echo "[calibration] ERROR: moveit_calibration_gui plugin not found. Rebuild ws_moveit first."; exit 1; }

echo "[calibration] Launching..."
ros2 launch kinova_bringup handeye_calibration.launch.py "$@"
