#!/usr/bin/env python3
"""
real_moveit_launch.py — Kinova j2s6s200 real arm via USB + MoveIt2

Package layout assumed (matches tree):
  j2s6s200_moveit_config/
    config/
      j2s6s200.srdf
      joint_limits.yaml
      kinematics.yaml
      ompl_planning.yaml
      moveit_controllers.yaml
    urdf/
      j2s6s200.urdf.xacro
    scripts/
      trajectory_executor.py
"""

import os
import re
import pathlib
import yaml
import tempfile
import subprocess

from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo
from launch_ros.actions import Node
from ament_index_python.packages import (
    get_package_share_directory,
    get_packages_with_prefixes,
)
# from launch_ros.actions import Node

# ── Package / path constants ──────────────────────────────────────────
PACKAGE_NAME = 'j2s6s200_moveit_config'
ROBOT_NAME   = 'j2s6s200'
NO_SIM       = {'use_sim_time': False}
KINOVA_SRC   = '/home/amma/kinova_ws/rl_v2-master/src/kinova-ros2'
SDK_LIB_DIR  = f'{KINOVA_SRC}/kinova_driver/lib/x86_64-linux-gnu'


# ── URDF helpers ──────────────────────────────────────────────────────

def _patch_urdf(raw_xml: str) -> str:
    """
    Remove Gazebo-only blocks and resolve package:// URIs to file:// so
    the URDF loads without ROS package resolution at runtime.
    Also writes a copy to /tmp for debugging.
    """
    # Strip all <gazebo>...</gazebo> blocks (plugins crash MoveIt2 parser)
    patched = re.sub(
        r'<gazebo[^>]*>.*?</gazebo>', '',
        raw_xml, flags=re.DOTALL | re.IGNORECASE
    )
    # Resolve package:// mesh URIs to absolute file:// paths
    patched = patched.replace(
        'package://kinova_description',
        f'file://{KINOVA_SRC}/kinova_description'
    )
    with open('/tmp/patched_real_robot.urdf', 'w') as f:
        f.write(patched)
    return patched


# ── Controller YAML ───────────────────────────────────────────────────

def _write_controllers_yaml() -> str:
    """
    Write MoveIt2 controller config in the correct ROS2 rcl params format.
    Returns the path to the temporary file.
    """
    config = {
        'move_group': {
            'ros__parameters': {
                'moveit_controller_manager':
                    'moveit_simple_controller_manager'
                    '/MoveItSimpleControllerManager',
                'moveit_simple_controller_manager': {
                    'controller_names': ['joint_trajectory_controller'],
                    'joint_trajectory_controller': {
                        'type':      'FollowJointTrajectory',
                        'action_ns': 'follow_joint_trajectory',
                        'default':   True,
                        'joints': [
                            f'{ROBOT_NAME}_joint_1',
                            f'{ROBOT_NAME}_joint_2',
                            f'{ROBOT_NAME}_joint_3',
                            f'{ROBOT_NAME}_joint_4',
                            f'{ROBOT_NAME}_joint_5',
                            f'{ROBOT_NAME}_joint_6',
                        ],
                    },
                },
            },
        },
    }
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False,
        prefix='kinova_controllers_'
    )
    yaml.dump(config, tmp, default_flow_style=False)
    tmp.flush()
    print(f'[DEBUG] Controllers yaml → {tmp.name}')
    return tmp.name


# ── Launch description ────────────────────────────────────────────────

def generate_launch_description():

    # Locate this package's share directory
    pkg_dir  = get_package_share_directory(PACKAGE_NAME)
    cfg_dir  = pathlib.Path(pkg_dir, 'config')
    urdf_dir = pathlib.Path(pkg_dir, 'urdf')

    # Convenience lambdas
    cfg_text = lambda n: (cfg_dir / n).read_text()
    cfg_yaml = lambda n: yaml.safe_load(cfg_text(n))
    cfg_path = lambda n: str(cfg_dir / n)

    # ── URDF ─────────────────────────────────────────────────────────
    xacro_file = str(urdf_dir / f'{ROBOT_NAME}.urdf.xacro')
    result = subprocess.run(
        ['xacro', xacro_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f'xacro failed:\n{result.stderr}')

    robot_desc = _patch_urdf(result.stdout)

    description            = {'robot_description': robot_desc}
    description_semantic   = {
        'robot_description_semantic': cfg_text(f'{ROBOT_NAME}.srdf')
    }
    description_kinematics = {
        'robot_description_kinematics': cfg_yaml('kinematics.yaml')
    }
    description_planning   = {
        'robot_description_planning': cfg_yaml('joint_limits.yaml')
    }

    controllers_yaml = _write_controllers_yaml()

    static_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_base',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'j2s6s200_link_base'],
            parameters=[NO_SIM],
        )
    

    # ── 1. Hardware bridge (trajectory executor + SDK node) ───────────
    hw_node = Node(
        package='j2s6s200_moveit_config',
        executable='trajectory_executor',
        name='kinova_hw_bridge',
        output='screen',
        additional_env={
            'LD_LIBRARY_PATH': (
                SDK_LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
            )
        },
        parameters=[NO_SIM],
    )

    # ── 2. Robot state publisher ──────────────────────────────────────
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[description, NO_SIM],
    )
    nodes = [hw_node, rsp]
    # nodes.append(static_tf)  

    # ── 3. MoveIt2 (move_group + RViz2) ──────────────────────────────
    if 'moveit_ros_move_group' in get_packages_with_prefixes():

        ompl_yaml = cfg_yaml('ompl_planning.yaml')
        ompl = {
            'move_group': {
                'planning_plugin': 'ompl_interface/OMPLPlanner',
                'request_adapters': (
                    'default_planner_request_adapters'
                    '/AddTimeOptimalParameterization '
                    'default_planner_request_adapters/FixWorkspaceBounds '
                    'default_planner_request_adapters/FixStartStateBounds '
                    'default_planner_request_adapters/FixStartStateCollision '
                    'default_planner_request_adapters'
                    '/FixStartStatePathConstraints'
                ),
                'start_state_max_bounds_error': 0.1,
            }
        }
        ompl['move_group'].update(ompl_yaml)

        move_group = Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[
                description,
                NO_SIM,
                description_semantic,
                description_kinematics,
                description_planning,
                ompl,
                controllers_yaml,   # loaded from temp file (correct rcl format)
                {
                    'current_state_monitor_wait_time': 10.0,
                    'trajectory_execution.execution_duration_monitoring': False,
                    'trajectory_execution.allowed_execution_duration_scaling': 3.0,
                    'trajectory_execution.allowed_goal_duration_margin': 2.0,
                    'trajectory_execution.allowed_start_tolerance': 0.1,
                    'publish_robot_description': True,
                    'publish_robot_description_semantic': True,
                    'planning_scene_monitor_options': {
                        'publish_planning_scene':        True,
                        'publish_geometry_updates':      True,
                        'publish_state_updates':         True,
                        'publish_transforms_updates':    True,
                    },
                    'plan_with_sensing': False,
                },
            ],
        )

        rviz_config = cfg_path('visualization.rviz') \
            if (cfg_dir / 'visualization.rviz').exists() \
            else ''

        rviz_args = ['-d', rviz_config] if rviz_config else []

        rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=rviz_args,
            parameters=[
                description,
                NO_SIM,
                description_semantic,
                description_kinematics,
            ],
        )
        # from launch_ros.actions import Node

         # add before hw_node and rsp
        # Delay MoveIt2 by 15 s so the hw bridge + RSP are fully up
        nodes.append(TimerAction(period=15.0, actions=[move_group, rviz]))

    else:
        nodes.append(LogInfo(
            msg='moveit_ros_move_group not found — running hw bridge only'
        ))

    return LaunchDescription(nodes)