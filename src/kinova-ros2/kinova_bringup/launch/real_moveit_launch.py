#!/usr/bin/env python3
"""
real_moveit_launch.py — Kinova j2n6s300 real arm via USB + MoveIt2
"""

import os, re, pathlib, yaml, tempfile, subprocess
from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, get_packages_with_prefixes

PACKAGE_NAME = 'kinova_bringup'
ROBOT_NAME   = 'j2n6s200'
NO_SIM       = {'use_sim_time': False}
KINOVA_SRC   = '/home/amma/Downloads/rl_v2-master-master/src/kinova-ros2'
SDK_LIB_DIR  = f'{KINOVA_SRC}/kinova_driver/lib/x86_64-linux-gnu'


def _patch_urdf(raw_xml):
    p = re.sub(r'<gazebo[^>]*>.*?</gazebo>', '', raw_xml, flags=re.DOTALL|re.IGNORECASE)
    p = p.replace('package://kinova_description', f'file://{KINOVA_SRC}/kinova_description')
    with open('/tmp/patched_real_robot.urdf', 'w') as f:
        f.write(p)
    return p


def _write_controllers_yaml():
    """
    Write controller config in correct ROS2 rcl params format.

    ROS2 params files CANNOT have sequences (lists) as keys.
    MoveIt2 Humble moveit_simple_controller_manager reads:
      - controller_names: [name1, name2, ...]
      - <name1>.type, <name1>.action_ns, <name1>.joints, ...
    """
    config = {
        'move_group': {
            'ros__parameters': {
                'moveit_controller_manager':
                    'moveit_simple_controller_manager/MoveItSimpleControllerManager',
                'moveit_simple_controller_manager': {
                    'controller_names': ['joint_trajectory_controller'],
                    'joint_trajectory_controller': {
                        'type': 'FollowJointTrajectory',
                        'action_ns': 'follow_joint_trajectory',
                        'default': True,
                        'joints': [
                            f'{ROBOT_NAME}_joint_1',
                            f'{ROBOT_NAME}_joint_2',
                            f'{ROBOT_NAME}_joint_3',
                            f'{ROBOT_NAME}_joint_4',
                            f'{ROBOT_NAME}_joint_5',
                            f'{ROBOT_NAME}_joint_6',
                        ]
                    }
                }
            }
        }
    }
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False, prefix='kinova_controllers_'
    )
    yaml.dump(config, tmp, default_flow_style=False)
    tmp.flush()
    print(f'[DEBUG] Controllers yaml → {tmp.name}')
    return tmp.name


def generate_launch_description():
    pkg_dir    = get_package_share_directory(PACKAGE_NAME)
    read_file  = lambda n: pathlib.Path(pkg_dir, 'moveit_resource', n).read_text()
    read_yaml  = lambda n: yaml.safe_load(read_file(n))
    moveit_res = lambda n: str(pathlib.Path(pkg_dir, 'moveit_resource', n))

    # ── URDF ──────────────────────────────────────────────────────────
    xacro_file = f'{KINOVA_SRC}/kinova_description/urdf/{ROBOT_NAME}_standalone.xacro'
    result = subprocess.run(['xacro', xacro_file], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'xacro failed:\n{result.stderr}')
    robot_desc = _patch_urdf(result.stdout)

    description            = {'robot_description': robot_desc}
    description_semantic   = {'robot_description_semantic':   read_file(f'{ROBOT_NAME}.srdf')}
    description_kinematics = {'robot_description_kinematics': read_yaml('kinematics.yaml')}
    description_planning   = {'robot_description_planning':   read_yaml('joint_limits.yaml')}

    controllers_yaml = _write_controllers_yaml()

    # ── 1. Hardware bridge ─────────────────────────────────────────────
    hw_node = Node(
        package='kinova_bringup',
        executable='trajectory_executor',
        name='kinova_hw_bridge',
        output='screen',
        additional_env={
            'LD_LIBRARY_PATH': SDK_LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        },
        parameters=[NO_SIM],
    )

    # ── 2. Robot state publisher ───────────────────────────────────────
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[description, NO_SIM],
    )

    nodes = [hw_node, rsp]

    # ── 3. MoveIt2 ────────────────────────────────────────────────────
    if 'moveit' in get_packages_with_prefixes():
        ompl_yaml = read_yaml('ompl_planning.yaml')
        ompl = {'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': (
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/FixStartStatePathConstraints'),
            'start_state_max_bounds_error': 0.1,
        }}
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
                # Controller config — loaded from file (correct rcl format)
                controllers_yaml,
                # Execution tuning
                {
                    'current_state_monitor_wait_time': 10.0,
                    'trajectory_execution.execution_duration_monitoring': False,
                    'trajectory_execution.allowed_execution_duration_scaling': 3.0,
                    'trajectory_execution.allowed_goal_duration_margin': 2.0,
                    'trajectory_execution.allowed_start_tolerance': 0.1,
                    'publish_robot_description': True,
                    'publish_robot_description_semantic': True,
                    'planning_scene_monitor_options': {
                        'publish_planning_scene': True,
                        'publish_geometry_updates': True,
                        'publish_state_updates': True,
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
            parameters=[description, NO_SIM, description_semantic, description_kinematics],
        )

        nodes.append(TimerAction(period=15.0, actions=[move_group, rviz]))

    else:
        nodes.append(LogInfo(msg='moveit not found — only hw bridge running'))

    return LaunchDescription(nodes)