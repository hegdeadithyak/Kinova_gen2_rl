#!/usr/bin/env python3
"""
Launch file for Kinova j2n6s300 — Ignition Gazebo (Fortress) + MoveIt2

ROOT CAUSE FIX (this iteration):
  The gz_ros_bridge arguments had a trailing ']' on every gz message type,
  e.g. 'gz.msgs.Clock]' instead of 'gz.msgs.Clock'.
  This caused ALL three bridges (clock, camera x2) to fail at startup with:
    "Failed to create a bridge for topic [/clock (gz.msgs.Clock])"
  Without /clock bridged, joint_states are published with timestamp 0.000000.
  current_state_monitor then ALWAYS fails with:
    "Didn't receive robot state with recent timestamp within 1s.
     Requested time X, but latest received state has time 0.000000."
  This is why:
    - Interactive marker drag was silently ignored (IK couldn't get robot state)
    - Plan+Execute always aborted ("couldn't receive full joint state within 1s")
  Fix: remove the trailing ']' — the '[' prefix is the gz→ROS direction flag,
  no closing bracket is needed or valid.

All previous fixes are retained:
  FIX 1  current_state_monitor_wait_time = 10 s
  FIX 2  kinematics wrapped under robot_description_kinematics
  FIX 3  publish_robot_description / semantic = True
  FIX 4  planning_scene_monitor_options all True
  FIX 5  use_sim_time via parameters= on rviz2
  FIX 6  execution_duration_monitoring = False
  FIX 7  gripper removed from kinematics.yaml (not a serial chain)
  DART-safe inertia patching, legacy Gazebo Classic blocks stripped,
  finger_trajectory_controller spawned + activated,
  MoveIt/RViz delayed 35 s for controllers to be fully active.
"""

import os
import re
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

# ─────────────────────────────── CONFIG ───────────────────────────────
PACKAGE_NAME          = 'kinova_bringup'
ROBOT_NAME            = 'j2n6s300'
WORLD_FILE            = os.path.expanduser(
    '~/kinova_ws/src/kinova-ros2/kinova_bringup/worlds/feeding_scene.sdf'
)
ROS2_CONTROLLERS_YAML = os.path.expanduser(
    '~/kinova_ws/src/kinova-ros2/kinova_bringup/moveit_resource/ros2_controllers.yaml'
)
CONTROLLER_MANAGER_NS = '/controller_manager'

# Single source of truth — applied to EVERY node
SIM_TIME = {'use_sim_time': True}
# ──────────────────────────────────────────────────────────────────────


def _patch_urdf(raw_xml: str) -> str:
    """
    Make the URDF safe for DART / Ignition Fortress:
      1. Remove every legacy Gazebo Classic <gazebo> block
      2. Force all mass values to 0.01 kg
      3. Force safe diagonal inertias (catches scientific notation)
      4. Zero off-diagonal inertias
      5. Inject ros2_control hardware block + gz_ros2_control plugin
    """
    patched = re.sub(
        r'<gazebo[^>]*>.*?</gazebo>', '',
        raw_xml, flags=re.DOTALL | re.IGNORECASE,
    )
    patched = re.sub(r'<mass\s+value="[^"]*"\s*/>', '<mass value="0.01"/>', patched)
    patched = re.sub(r'\bixx="[^"]*"', 'ixx="0.0001"', patched)
    patched = re.sub(r'\biyy="[^"]*"', 'iyy="0.0001"', patched)
    patched = re.sub(r'\bizz="[^"]*"', 'izz="0.0001"', patched)
    patched = re.sub(r'\bixy="[^"]*"', 'ixy="0.0"', patched)
    patched = re.sub(r'\bixz="[^"]*"', 'ixz="0.0"', patched)
    patched = re.sub(r'\biyz="[^"]*"', 'iyz="0.0"', patched)

    arm_joints    = [f'{ROBOT_NAME}_joint_{i}' for i in range(1, 7)]
    finger_joints = [f'{ROBOT_NAME}_joint_finger_{i}' for i in range(1, 4)]

    INITIAL_POSITIONS = {
        f'{ROBOT_NAME}_joint_1':        4.71,
        f'{ROBOT_NAME}_joint_2':        2.71,
        f'{ROBOT_NAME}_joint_3':        1.57,
        f'{ROBOT_NAME}_joint_4':        4.71,
        f'{ROBOT_NAME}_joint_5':        0.0,
        f'{ROBOT_NAME}_joint_6':        3.14,
        f'{ROBOT_NAME}_joint_finger_1': 0.0,
        f'{ROBOT_NAME}_joint_finger_2': 0.0,
        f'{ROBOT_NAME}_joint_finger_3': 0.0,
    }

    def _joint_block(name):
        v = INITIAL_POSITIONS.get(name, 0.0)
        return f"""
        <joint name="{name}">
            <command_interface name="position">
                <param name="min">-6.28</param>
                <param name="max">6.28</param>
                <param name="initial_value">{v}</param>
            </command_interface>
            <state_interface name="position">
                <param name="initial_value">{v}</param>
            </state_interface>
            <state_interface name="velocity">
                <param name="initial_value">0.0</param>
            </state_interface>
        </joint>"""

    joint_blocks = ''.join(_joint_block(j) for j in arm_joints + finger_joints)

    ros2_control_block = f"""
  <ros2_control name="GazeboSystem" type="system">
    <hardware>
      <plugin>gz_ros2_control/GazeboSimSystem</plugin>
    </hardware>{joint_blocks}
  </ros2_control>
"""
    gazebo_plugin_block = f"""
  <gazebo>
    <plugin filename="gz_ros2_control-system"
            name="gz_ros2_control::GazeboSimROS2ControlPlugin">
      <parameters>{ROS2_CONTROLLERS_YAML}</parameters>
      <ros>
        <remapping>~/robot_description:=robot_description</remapping>
      </ros>
    </plugin>
  </gazebo>
"""
    insertion = ros2_control_block + gazebo_plugin_block
    patched = (
        patched.replace('</robot>', insertion + '</robot>', 1)
        if '</robot>' in patched else patched + insertion
    )

    debug_path = '/tmp/patched_robot.urdf'
    with open(debug_path, 'w') as fh:
        fh.write(patched)
    print(f'[DEBUG] Patched URDF saved → {debug_path}')
    return patched


def generate_launch_description():
    pkg_dir = get_package_share_directory(PACKAGE_NAME)

    def read_file(name):
        return pathlib.Path(pkg_dir, 'moveit_resource', name).read_text()

    def read_yaml(name):
        return yaml.safe_load(read_file(name))

    def moveit_res(name):
        return str(pathlib.Path(pkg_dir, 'moveit_resource', name))

    # ── Robot description ──────────────────────────────────────────────
    xacro_file = os.path.join(
        get_package_share_directory('kinova_description'),
        'urdf', f'{ROBOT_NAME}_standalone.xacro',
    )
    robot_description_content = _patch_urdf(
        xacro.process_file(xacro_file).toprettyxml(indent='  ')
    )
    description = {'robot_description': robot_description_content}

    # Shared MoveIt parameter blocks — used identically by move_group AND rviz2
    description_semantic   = {'robot_description_semantic':  read_file(f'{ROBOT_NAME}.srdf')}
    description_kinematics = {'robot_description_kinematics': read_yaml('kinematics.yaml')}
    description_planning   = {'robot_description_planning':  read_yaml('joint_limits.yaml')}

    # ── Ignition Gazebo ────────────────────────────────────────────────
    mesh_path = os.path.expanduser('~/kinova_ws/src/kinova-ros2')

    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', WORLD_FILE],
        additional_env={
            'IGN_GAZEBO_SYSTEM_PLUGIN_PATH':
                '/opt/ros/humble/lib:'
                + os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', ''),
            'IGN_GAZEBO_RESOURCE_PATH':
                mesh_path + ':'
                + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', ''),
        },
        output='screen',
    )

    # ── Robot State Publisher ──────────────────────────────────────────
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[description, SIM_TIME],
    )

    # ── Spawn robot ────────────────────────────────────────────────────
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name=f'spawn_{ROBOT_NAME}',
        output='screen',
        parameters=[SIM_TIME],
        arguments=[
            '-name',  ROBOT_NAME,
            '-topic', 'robot_description',
            '-x', '0.0', '-y', '0.0', '-z', '0.80',
        ],
    )

    # ── ROS ↔ Gazebo bridge ────────────────────────────────────────────
    # CRITICAL: NO trailing ']' after gz message type names.
    # The '[' prefix means gz→ROS direction. A closing ']' is invalid
    # and causes the bridge to silently fail, leaving /clock unbridged.
    # Without /clock, all joint_states have timestamp 0 → everything breaks.
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_ros_bridge',
        output='screen',
        parameters=[SIM_TIME],
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/camera_mouth/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_mouth/depth/image_rect_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        ],
    )

    # ── Controller spawners ────────────────────────────────────────────
    def _spawner(controller_name):
        args = [
            controller_name,
            '--controller-manager', CONTROLLER_MANAGER_NS,
            '--controller-manager-timeout', '60',
        ]
        if 'trajectory' in controller_name:
            args.append('--activate')
        return Node(
            package='controller_manager',
            executable='spawner',
            output='screen',
            parameters=[SIM_TIME],
            arguments=args,
        )

    controller_spawners = TimerAction(
        period=10.0,
        actions=[
            _spawner('joint_state_broadcaster'),
            _spawner('joint_trajectory_controller'),
            _spawner('finger_trajectory_controller'),
        ],
    )

    nodes = [gazebo, rsp, spawn_robot, gz_bridge, controller_spawners]

    # ── MoveIt 2 ───────────────────────────────────────────────────────
    if 'moveit' in get_packages_with_prefixes():

        ompl_config = {'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': (
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/FixStartStatePathConstraints'
            ),
            'start_state_max_bounds_error': 0.1,
        }}
        ompl_config['move_group'].update(read_yaml('ompl_planning.yaml'))

        move_group = Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[
                description,
                SIM_TIME,
                description_semantic,
                description_kinematics,
                description_planning,
                ompl_config,
                {
                    'moveit_simple_controller_manager': read_yaml('controllers.yaml'),
                    'moveit_controller_manager':
                        'moveit_simple_controller_manager/MoveItSimpleControllerManager',
                },
                {
                    'current_state_monitor_wait_time': 10.0,
                    'trajectory_execution.execution_duration_monitoring': False,
                    'trajectory_execution.allowed_execution_duration_scaling': 2.0,
                    'trajectory_execution.allowed_goal_duration_margin':       1.0,
                    'trajectory_execution.allowed_start_tolerance':            0.05,
                    'publish_robot_description':          True,
                    'publish_robot_description_semantic': True,
                    'planning_scene_monitor_options': {
                        'publish_planning_scene':     True,
                        'publish_geometry_updates':   True,
                        'publish_state_updates':      True,
                        'publish_transforms_updates': True,
                    },
                    'plan_with_sensing': False,
                },
            ],
        )

        rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', moveit_res('visualization.rviz')],
            parameters=[
                description,
                SIM_TIME,
                description_semantic,
                description_kinematics,
                description_planning,
            ],
        )

        nodes.append(TimerAction(period=35.0, actions=[move_group, rviz]))

    else:
        nodes.append(
            LogInfo(msg='"moveit" package not found — skipping move_group + rviz2.')
        )

    return LaunchDescription(nodes)