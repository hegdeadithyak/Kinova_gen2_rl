#!/usr/bin/env bash
# =============================================================================
#  setup_kinova_feeding.sh
#  Place this file inside:  kinova_ws/src/kinova-ros2/kinova_bringup/
#  Run once from anywhere:  bash setup_kinova_feeding.sh
#
#  What this script does (everything EXCEPT launching):
#   1. Detects your ROS 2 distro & workspace root
#   2. Installs system / ROS dependencies
#   3. Writes the MoveIt+Gazebo launch file  (moveit_gazebo_launch.py)
#   4. Copies / creates the j2n6s300.srdf (if missing)
#   5. Patches CMakeLists.txt to install the new launch file
#   6. Writes the feeding_world.sdf  (if it doesn't exist yet)
#   7. Builds the workspace with colcon
#   8. Sources the install overlay
#
#  After this script finishes, run:
#   source ~/kinova_ws/install/setup.bash
#   ros2 launch kinova_bringup moveit_gazebo_launch.py
# =============================================================================

set -euo pipefail

###############################################################################
# 0.  Colour helpers
###############################################################################
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

###############################################################################
# 1.  Locate this script → derive package dir & workspace root
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Expected layout: <ws>/src/kinova-ros2/kinova_bringup/setup_kinova_feeding.sh
PACKAGE_DIR="$SCRIPT_DIR"
WS_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # four levels up = workspace

info "Package dir : $PACKAGE_DIR"
info "Workspace   : $WS_ROOT"

###############################################################################
# 2.  Detect ROS 2 distro
###############################################################################
if [[ -z "${ROS_DISTRO:-}" ]]; then
    for d in humble iron jazzy rolling; do
        if [[ -d "/opt/ros/$d" ]]; then ROS_DISTRO="$d"; break; fi
    done
fi
[[ -z "${ROS_DISTRO:-}" ]] && die "Could not detect ROS 2 distro. Source your ROS setup first."
info "ROS 2 distro: $ROS_DISTRO"
# source "/opt/ros/$ROS_DISTRO/setup.bash"

###############################################################################
# 3.  Install dependencies
###############################################################################
info "Installing ROS 2 and system dependencies …"
sudo apt-get update -qq

PKGS=(
    "ros-${ROS_DISTRO}-moveit"
    "ros-${ROS_DISTRO}-moveit-ros-move-group"
    "ros-${ROS_DISTRO}-moveit-simple-controller-manager"
    "ros-${ROS_DISTRO}-ompl"
    "ros-${ROS_DISTRO}-moveit-planners-ompl"
    "ros-${ROS_DISTRO}-ros-gz-bridge"
    "ros-${ROS_DISTRO}-ros-gz-sim"
    "ros-${ROS_DISTRO}-gz-ros2-control"
    "ros-${ROS_DISTRO}-robot-state-publisher"
    "ros-${ROS_DISTRO}-joint-state-publisher"
    "ros-${ROS_DISTRO}-rviz2"
    "ros-${ROS_DISTRO}-xacro"
    "python3-xacro"
    "python3-colcon-common-extensions"
)

for pkg in "${PKGS[@]}"; do
    if dpkg -s "$pkg" &>/dev/null; then
        info "  already installed: $pkg"
    else
        info "  installing: $pkg"
        sudo apt-get install -y "$pkg" || warn "Could not install $pkg — skipping"
    fi
done
success "Dependencies done."

###############################################################################
# 4.  Write moveit_gazebo_launch.py
###############################################################################
LAUNCH_FILE="$PACKAGE_DIR/launch/moveit_gazebo_launch.py"
info "Writing $LAUNCH_FILE …"

cat > "$LAUNCH_FILE" << 'PYEOF'
#!/usr/bin/env python3
"""Launch Gazebo (feeding_world) + MoveIt2 for the j2n6s300 Kinova arm."""

import os
import pathlib
import yaml

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, LogInfo
from launch_ros.actions import Node
from ament_index_python.packages import (
    get_package_share_directory,
    get_packages_with_prefixes,
)
import xacro

# ── Constants ─────────────────────────────────────────────────────────────────
PACKAGE_NAME = 'kinova_bringup'
ROBOT_NAME   = 'j2n6s300'
WORLD_FILE   = os.path.expanduser(
    '~/kinova_ws/src/kinova-ros2/kinova_bringup/worlds/feeding_world.sdf'
)


def generate_launch_description():
    package_dir = get_package_share_directory(PACKAGE_NAME)

    def load_file(filename):
        return pathlib.Path(
            os.path.join(package_dir, 'moveit_resource', filename)
        ).read_text()

    def load_yaml(filename):
        return yaml.safe_load(load_file(filename))

    # ── URDF ──────────────────────────────────────────────────────────────────
    xacro_file = os.path.join(
        get_package_share_directory('kinova_description'),
        'urdf', f'{ROBOT_NAME}_standalone.xacro',
    )
    robot_description_content = xacro.process_file(xacro_file).toprettyxml(indent='  ')
    description  = {'robot_description': robot_description_content}
    sim_time     = {'use_sim_time': True}

    # ── 1. Gazebo ──────────────────────────────────────────────────────────────
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', WORLD_FILE],
        output='screen',
    )

    # ── 2. ROS ↔ Gazebo bridge ─────────────────────────────────────────────────
    gz_ros_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_ros_bridge',
        output='screen',
        parameters=[sim_time],
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            f'/{ROBOT_NAME}/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/camera_mouth/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_mouth/depth/image_rect_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        ],
    )

    # ── 3. Robot State Publisher ───────────────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[description, sim_time],
        remappings=[('/joint_states', f'/{ROBOT_NAME}/joint_states')],
    )

    nodes = [gazebo, gz_ros_bridge, robot_state_publisher]

    # ── 4. MoveIt 2 ────────────────────────────────────────────────────────────
    if 'moveit' in get_packages_with_prefixes():
        description_semantic     = {'robot_description_semantic':   load_file(f'{ROBOT_NAME}.srdf')}
        description_kinematics   = {'robot_description_kinematics': load_yaml('kinematics.yaml')}
        description_joint_limits = {'robot_description_planning':   load_yaml('joint_limits.yaml')}

        ompl_pipeline = {
            'move_group': {
                'planning_plugin': 'ompl_interface/OMPLPlanner',
                'request_adapters': (
                    'default_planner_request_adapters/AddTimeOptimalParameterization '
                    'default_planner_request_adapters/FixWorkspaceBounds '
                    'default_planner_request_adapters/FixStartStateBounds '
                    'default_planner_request_adapters/FixStartStateCollision '
                    'default_planner_request_adapters/FixStartStatePathConstraints'
                ),
                'start_state_max_bounds_error': 0.1,
            }
        }
        ompl_pipeline['move_group'].update(load_yaml('ompl_planning.yaml'))

        moveit_controllers = {
            'moveit_controller_manager':
                'moveit_simple_controller_manager/MoveItSimpleControllerManager',
            'moveit_simple_controller_manager': load_yaml('controllers.yaml'),
        }

        move_group = Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[
                description, description_semantic, description_kinematics,
                moveit_controllers, ompl_pipeline, description_joint_limits,
                sim_time,
            ],
            remappings=[('/joint_states', f'/{ROBOT_NAME}/joint_states')],
        )

        rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(package_dir, 'moveit_resource', 'visualization.rviz')],
            parameters=[
                description, description_semantic,
                description_kinematics, description_joint_limits, sim_time,
            ],
        )

        # Delay 3 s so Gazebo clock is already ticking
        nodes.append(TimerAction(period=3.0, actions=[move_group, rviz]))
    else:
        nodes.append(LogInfo(msg='"moveit" is not installed.'))

    return LaunchDescription(nodes)
PYEOF

chmod +x "$LAUNCH_FILE"
success "Launch file written."

###############################################################################
# 5.  Create j2n6s300.srdf (if missing)
###############################################################################
SRDF="$PACKAGE_DIR/moveit_resource/j2n6s300.srdf"
if [[ ! -f "$SRDF" ]]; then
    info "Creating minimal j2n6s300.srdf …"
    cat > "$SRDF" << 'SRDFEOF'
<?xml version="1.0" ?>
<robot name="j2n6s300">
  <!-- END-EFFECTOR -->
  <end_effector name="eef" parent_link="j2n6s300_end_effector" group="arm"/>

  <!-- GROUPS -->
  <group name="arm">
    <chain base_link="j2n6s300_link_base" tip_link="j2n6s300_end_effector"/>
  </group>
  <group name="gripper">
    <joint name="j2n6s300_joint_finger_1"/>
    <joint name="j2n6s300_joint_finger_2"/>
    <joint name="j2n6s300_joint_finger_3"/>
  </group>

  <!-- VIRTUAL JOINTS -->
  <virtual_joint name="virtual_joint" type="fixed"
                 parent_frame="world" child_link="j2n6s300_link_base"/>

  <!-- DISABLE COLLISIONS (self-collision pairs that are always in contact) -->
  <disable_collisions link1="j2n6s300_link_base"    link2="j2n6s300_link_1"    reason="Adjacent"/>
  <disable_collisions link1="j2n6s300_link_1"       link2="j2n6s300_link_2"    reason="Adjacent"/>
  <disable_collisions link1="j2n6s300_link_2"       link2="j2n6s300_link_3"    reason="Adjacent"/>
  <disable_collisions link1="j2n6s300_link_3"       link2="j2n6s300_link_4"    reason="Adjacent"/>
  <disable_collisions link1="j2n6s300_link_4"       link2="j2n6s300_link_5"    reason="Adjacent"/>
  <disable_collisions link1="j2n6s300_link_5"       link2="j2n6s300_link_6"    reason="Adjacent"/>
  <disable_collisions link1="j2n6s300_link_6"       link2="j2n6s300_end_effector" reason="Adjacent"/>
  <disable_collisions link1="j2n6s300_link_finger_1"  link2="j2n6s300_link_6"  reason="Adjacent"/>
  <disable_collisions link1="j2n6s300_link_finger_2"  link2="j2n6s300_link_6"  reason="Adjacent"/>
  <disable_collisions link1="j2n6s300_link_finger_3"  link2="j2n6s300_link_6"  reason="Adjacent"/>
</robot>
SRDFEOF
    success "SRDF created."
else
    info "SRDF already exists — skipping."
fi

###############################################################################
# 6.  Create worlds/ directory and write feeding_world.sdf (if missing)
###############################################################################
WORLDS_DIR="$PACKAGE_DIR/worlds"
mkdir -p "$WORLDS_DIR"
WORLD_SDF="$WORLDS_DIR/feeding_world.sdf"

if [[ ! -f "$WORLD_SDF" ]]; then
    info "Writing feeding_world.sdf …"
    cat > "$WORLD_SDF" << 'SDFEOF'
<?xml version="1.0" ?>
<sdf version="1.9">
<world name="feeding_world">

  <physics name="default_physics" default="true" type="ode">
    <max_step_size>0.005</max_step_size>
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>200</real_time_update_rate>
    <ode>
      <solver><type>quick</type><iters>50</iters><sor>1.3</sor></solver>
      <constraints>
        <cfm>0.0</cfm><erp>0.2</erp>
        <contact_max_correcting_vel>100</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>

  <plugin filename="gz-sim-physics-system"        name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-sensors-system"         name="gz::sim::systems::Sensors">
    <render_engine>ogre2</render_engine>
  </plugin>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-user-commands-system"   name="gz::sim::systems::UserCommands"/>

  <light type="directional" name="ceiling_main">
    <cast_shadows>false</cast_shadows>
    <pose>0.5 0 3.0 0 0 0</pose>
    <diffuse>0.85 0.85 0.85 1</diffuse>
    <specular>0.20 0.20 0.20 1</specular>
    <direction>0.0 0.0 -1.0</direction>
    <intensity>1.0</intensity>
  </light>
  <light type="point" name="fill_light">
    <pose>0.05 -0.5 1.8 0 0 0</pose>
    <diffuse>0.60 0.60 0.60 1</diffuse>
    <specular>0.05 0.05 0.05 1</specular>
    <attenuation>
      <range>6</range><constant>0.5</constant>
      <linear>0.02</linear><quadratic>0.001</quadratic>
    </attenuation>
    <cast_shadows>false</cast_shadows>
  </light>

  <model name="ground_plane">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
        <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
      </collision>
      <visual name="visual">
        <geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
        <material>
          <ambient>0.40 0.40 0.40 1</ambient>
          <diffuse>0.40 0.40 0.40 1</diffuse>
        </material>
      </visual>
    </link>
  </model>

  <model name="robot_pedestal">
    <static>true</static>
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <pose>0 0 0.375 0 0 0</pose>
      <collision name="collision">
        <geometry><box><size>0.20 0.20 0.75</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>0.20 0.20 0.75</size></box></geometry>
        <material>
          <ambient>0.22 0.22 0.22 1</ambient><diffuse>0.22 0.22 0.22 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
        </material>
      </visual>
    </link>
  </model>

  <!-- Robot j2n6s300 — base at pedestal top z=0.75 -->
  <include>
    <uri>file:///home/amma/RL-based-Motionplanner-for-arm/src/kinova_j2n6s300_ign/kinova_j2n6s300</uri>
    <name>j2n6s300</name>
    <pose>0 0 0.75 0 0 0</pose>
  </include>

  <model name="table">
    <static>true</static>
    <pose>0.60 0 0 0 0 0</pose>
    <link name="tabletop">
      <pose>0 0 0.735 0 0 0</pose>
      <collision name="collision">
        <geometry><box><size>0.90 0.70 0.030</size></box></geometry>
        <surface><friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction></surface>
      </collision>
      <visual name="visual">
        <geometry><box><size>0.90 0.70 0.030</size></box></geometry>
        <material>
          <ambient>0.55 0.38 0.18 1</ambient><diffuse>0.55 0.38 0.18 1</diffuse>
          <specular>0.05 0.03 0.01 1</specular>
        </material>
      </visual>
    </link>
    <link name="leg_fl"><pose> 0.40  0.30 0.365 0 0 0</pose>
      <collision name="c"><geometry><box><size>0.05 0.05 0.73</size></box></geometry></collision>
      <visual    name="v"><geometry><box><size>0.05 0.05 0.73</size></box></geometry>
        <material><ambient>0.40 0.25 0.10 1</ambient><diffuse>0.40 0.25 0.10 1</diffuse></material>
      </visual></link>
    <link name="leg_fr"><pose> 0.40 -0.30 0.365 0 0 0</pose>
      <collision name="c"><geometry><box><size>0.05 0.05 0.73</size></box></geometry></collision>
      <visual    name="v"><geometry><box><size>0.05 0.05 0.73</size></box></geometry>
        <material><ambient>0.40 0.25 0.10 1</ambient><diffuse>0.40 0.25 0.10 1</diffuse></material>
      </visual></link>
    <link name="leg_bl"><pose>-0.40  0.30 0.365 0 0 0</pose>
      <collision name="c"><geometry><box><size>0.05 0.05 0.73</size></box></geometry></collision>
      <visual    name="v"><geometry><box><size>0.05 0.05 0.73</size></box></geometry>
        <material><ambient>0.40 0.25 0.10 1</ambient><diffuse>0.40 0.25 0.10 1</diffuse></material>
      </visual></link>
    <link name="leg_br"><pose>-0.40 -0.30 0.365 0 0 0</pose>
      <collision name="c"><geometry><box><size>0.05 0.05 0.73</size></box></geometry></collision>
      <visual    name="v"><geometry><box><size>0.05 0.05 0.73</size></box></geometry>
        <material><ambient>0.40 0.25 0.10 1</ambient><diffuse>0.40 0.25 0.10 1</diffuse></material>
      </visual></link>
  </model>

  <model name="plate">
    <static>true</static>
    <pose>0.50 0 0.756 0 0 0</pose>
    <link name="link">
      <collision name="collision">
        <geometry><cylinder><radius>0.130</radius><length>0.012</length></cylinder></geometry>
        <surface><friction><ode><mu>0.6</mu><mu2>0.6</mu2></ode></friction></surface>
      </collision>
      <visual name="visual">
        <geometry><cylinder><radius>0.130</radius><length>0.012</length></cylinder></geometry>
        <material>
          <ambient>0.95 0.95 0.95 1</ambient><diffuse>0.95 0.95 0.95 1</diffuse>
          <specular>0.30 0.30 0.30 1</specular>
        </material>
      </visual>
    </link>
  </model>

  <model name="food_item">
    <pose>0.50 0 0.784 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>0.040</mass>
        <inertia><ixx>8e-6</ixx><ixy>0</ixy><ixz>0</ixz>
                 <iyy>8e-6</iyy><iyz>0</iyz><izz>8e-6</izz></inertia>
      </inertial>
      <collision name="collision">
        <geometry><sphere><radius>0.022</radius></sphere></geometry>
        <surface>
          <friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
          <contact><ode><kp>1e5</kp><kd>1.0</kd></ode></contact>
        </surface>
      </collision>
      <visual name="visual">
        <geometry><sphere><radius>0.022</radius></sphere></geometry>
        <material>
          <ambient>0.95 0.50 0.05 1</ambient><diffuse>0.95 0.50 0.05 1</diffuse>
          <specular>0.10 0.05 0.00 1</specular>
        </material>
      </visual>
    </link>
  </model>

  <model name="patient">
    <static>true</static>
    <pose>0.90 0 0 0 0 3.14159265</pose>
    <link name="link">
      <visual name="torso">
        <pose>0 0 0.90 0 0 0</pose>
        <geometry><box><size>0.28 0.22 0.44</size></box></geometry>
        <material><ambient>0.25 0.40 0.65 1</ambient><diffuse>0.25 0.40 0.65 1</diffuse></material>
      </visual>
      <visual name="neck">
        <pose>0 0 1.12 0 0 0</pose>
        <geometry><cylinder><radius>0.030</radius><length>0.06</length></cylinder></geometry>
        <material><ambient>0.88 0.72 0.58 1</ambient><diffuse>0.88 0.72 0.58 1</diffuse></material>
      </visual>
      <visual name="head">
        <pose>0 0 1.15 0 0 0</pose>
        <geometry><sphere><radius>0.100</radius></sphere></geometry>
        <material><ambient>0.88 0.72 0.58 1</ambient><diffuse>0.88 0.72 0.58 1</diffuse></material>
      </visual>
      <visual name="mouth_target">
        <pose>-0.05 0 1.095 0 0 0</pose>
        <geometry><sphere><radius>0.018</radius></sphere></geometry>
        <material><ambient>0.85 0.15 0.10 1</ambient><diffuse>0.85 0.15 0.10 1</diffuse></material>
      </visual>
    </link>
  </model>

  <model name="camera_stand">
    <static>true</static>
    <pose>0.05 -0.55 0 0 0 0</pose>
    <link name="pole">
      <pose>0 0 0.65 0 0 0</pose>
      <visual name="vis">
        <geometry><cylinder><radius>0.018</radius><length>1.30</length></cylinder></geometry>
        <material><ambient>0.3 0.3 0.3 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material>
      </visual>
    </link>
  </model>

  <model name="mouth_camera_mount">
    <static>true</static>
    <pose>0.05 -0.45 1.30 0 0.22 0.51</pose>
    <link name="camera_link">
      <visual name="housing">
        <geometry><box><size>0.035 0.085 0.025</size></box></geometry>
        <material><ambient>0.08 0.08 0.08 1</ambient><diffuse>0.08 0.08 0.08 1</diffuse></material>
      </visual>
      <sensor name="camera_mouth_color" type="camera">
        <always_on>true</always_on>
        <update_rate>15</update_rate>
        <visualize>false</visualize>
        <camera name="color">
          <horizontal_fov>1.047196</horizontal_fov>
          <image><width>640</width><height>480</height><format>R8G8B8</format></image>
          <clip><near>0.05</near><far>3.0</far></clip>
          <noise><type>gaussian</type><mean>0.0</mean><stddev>0.005</stddev></noise>
        </camera>
        <topic>/camera_mouth/color/image_raw</topic>
        <gz_frame_id>camera_mouth_color_optical_frame</gz_frame_id>
      </sensor>
      <sensor name="camera_mouth_depth" type="depth_camera">
        <always_on>true</always_on>
        <update_rate>15</update_rate>
        <visualize>false</visualize>
        <camera name="depth">
          <horizontal_fov>1.047196</horizontal_fov>
          <image><width>640</width><height>480</height><format>R_FLOAT32</format></image>
          <clip><near>0.05</near><far>3.0</far></clip>
          <noise><type>gaussian</type><mean>0.0</mean><stddev>0.003</stddev></noise>
        </camera>
        <topic>/camera_mouth/depth/image_rect_raw</topic>
        <gz_frame_id>camera_mouth_depth_optical_frame</gz_frame_id>
      </sensor>
    </link>
  </model>

</world>
</sdf>
SDFEOF
    success "feeding_world.sdf written to $WORLD_SDF"
else
    info "feeding_world.sdf already exists — skipping."
fi

###############################################################################
# 7.  Patch CMakeLists.txt to install launch + worlds + moveit_resource
###############################################################################
CMAKE="$PACKAGE_DIR/CMakeLists.txt"
if [[ -f "$CMAKE" ]]; then
    info "Checking CMakeLists.txt for install rules …"

    # Add worlds install rule if not already present
    if ! grep -q "worlds" "$CMAKE"; then
        info "  Adding worlds/ install rule …"
        cat >> "$CMAKE" << 'CMAKEEOF'

# ── Auto-added by setup_kinova_feeding.sh ──
install(DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(DIRECTORY moveit_resource/
  DESTINATION share/${PROJECT_NAME}/moveit_resource
)
CMAKEEOF
        success "CMakeLists.txt patched."
    else
        info "  CMakeLists.txt already has worlds/ rule — skipping."
    fi
else
    warn "CMakeLists.txt not found at $CMAKE — skipping patch."
fi

###############################################################################
# 8.  Build workspace with colcon
###############################################################################
info "Building workspace at $WS_ROOT …"
cd "$WS_ROOT"

# rosdep install first (best-effort)
if command -v rosdep &>/dev/null; then
    info "Running rosdep install …"
    rosdep install --from-paths src --ignore-src -r -y || warn "rosdep had warnings — continuing."
fi

colcon build --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    --packages-select kinova_msgs kinova_description kinova_driver kinova_demo kinova_bringup \
    2>&1 | tee "$WS_ROOT/colcon_build.log"

BUILD_STATUS=${PIPESTATUS[0]}
if [[ $BUILD_STATUS -ne 0 ]]; then
    die "colcon build failed (exit $BUILD_STATUS). Check $WS_ROOT/colcon_build.log for details."
fi
success "Build complete."

###############################################################################
# 9.  Source the install overlay  (for the current shell via eval)
###############################################################################
SETUP_BASH="$WS_ROOT/install/setup.bash"
if [[ -f "$SETUP_BASH" ]]; then
    info "Sourcing $SETUP_BASH …"
    # Write a helper so the *parent* shell can source it easily
    echo "source $SETUP_BASH" > "$WS_ROOT/source_ws.sh"
    source "$SETUP_BASH"
    success "Workspace sourced."
fi

###############################################################################
# 10.  Final instructions
###############################################################################
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!  Everything is built and ready.${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo "  In EVERY new terminal, run:"
echo -e "    ${CYAN}source $SETUP_BASH${NC}"
echo ""
echo "  Then launch with:"
echo -e "    ${CYAN}ros2 launch kinova_bringup moveit_gazebo_launch.py${NC}"
echo ""
echo "  NOTE: If your Gazebo robot model URI differs, edit line ~60 of:"
echo "    $WORLDS_DIR/feeding_world.sdf"
echo "  and change the <uri> inside the <include> block."
echo ""