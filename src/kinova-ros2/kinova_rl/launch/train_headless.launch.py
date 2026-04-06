"""
train_headless.launch.py
─────────────────────────
Starts the full Gazebo simulation headlessly + begins RL training.
Gazebo runs with -s (server only, no GUI) to save resources.

Usage:
  ros2 launch kinova_rl train_headless.launch.py
  ros2 launch kinova_rl train_headless.launch.py timesteps:=500000
  ros2 launch kinova_rl train_headless.launch.py resume:=/path/to/model
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

ROBOT_NAME    = 'j2n6s300'
BRINGUP_PKG   = 'kinova_bringup'
WORLD_FILE    = os.path.expanduser(
    '~/kinova_ws/src/kinova-ros2/kinova_bringup/worlds/feeding_scene.sdf'
)
ROS2_CONTROLLERS_YAML = os.path.expanduser(
    '~/kinova_ws/src/kinova-ros2/kinova_bringup/moveit_resource/ros2_controllers.yaml'
)

SIM_TIME = {'use_sim_time': True}


def generate_launch_description():
    timesteps_arg = DeclareLaunchArgument(
        'timesteps', default_value='1000000',
        description='Total training timesteps',
    )
    resume_arg = DeclareLaunchArgument(
        'resume', default_value='',
        description='Path to model checkpoint to resume training',
    )

    # Gazebo headless — -s flag = server only, no GUI rendering
    gazebo = ExecuteProcess(
        cmd=[
            'ign', 'gazebo', '-s', '-r', WORLD_FILE,
        ],
        additional_env={
            'IGN_GAZEBO_SYSTEM_PLUGIN_PATH':
                '/opt/ros/humble/lib:'
                + os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', ''),
            'IGN_GAZEBO_RESOURCE_PATH':
                os.path.expanduser('~/kinova_ws/src/kinova-ros2') + ':'
                + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', ''),
            # Suppress rendering output
            'LIBGL_ALWAYS_SOFTWARE': '1',
        },
        output='screen',
    )

    # Full bringup is handled by including the main launch file's nodes
    # directly. We skip rviz2 and move_group (not needed for headless RL).
    import pathlib
    import yaml
    import xacro

    pkg_dir  = get_package_share_directory(BRINGUP_PKG)
    xacro_file = os.path.join(
        get_package_share_directory('kinova_description'),
        'urdf', f'{ROBOT_NAME}_standalone.xacro',
    )

    # Import the patching function from the main bringup launch
    import sys
    sys.path.insert(0, os.path.expanduser(
        '~/kinova_ws/src/kinova-ros2/kinova_bringup/launch'
    ))
    try:
        from kinova_launch import _patch_urdf
        robot_description_content = _patch_urdf(
            xacro.process_file(xacro_file).toprettyxml(indent='  ')
        )
    except ImportError:
        robot_description_content = xacro.process_file(xacro_file).toprettyxml(
            indent='  '
        )

    description = {'robot_description': robot_description_content}

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[description, SIM_TIME],
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name=f'spawn_{ROBOT_NAME}',
        output='screen',
        parameters=[SIM_TIME],
        arguments=[
            '-name', ROBOT_NAME,
            '-topic', 'robot_description',
            '-x', '0.0', '-y', '0.0', '-z', '0.80',
        ],
    )

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_ros_bridge',
        output='screen',
        parameters=[SIM_TIME],
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
    )

    def _spawner(name):
        args = [name, '--controller-manager', '/controller_manager',
                '--controller-manager-timeout', '60']
        if 'trajectory' in name:
            args.append('--activate')
        return Node(
            package='controller_manager', executable='spawner',
            output='screen', parameters=[SIM_TIME], arguments=args,
        )

    controller_spawners = TimerAction(period=10.0, actions=[
        _spawner('joint_state_broadcaster'),
        _spawner('joint_trajectory_controller'),
        _spawner('finger_trajectory_controller'),
    ])

    # RL training starts after 30 s (controllers + robot fully up)
    train_cmd = [
        'ros2', 'run', 'kinova_rl', 'train', '--',
        '--headless',
        '--timesteps', LaunchConfiguration('timesteps'),
    ]

    train_node = TimerAction(
        period=30.0,
        actions=[
            ExecuteProcess(
                cmd=train_cmd,
                output='screen',
            )
        ],
    )

    return LaunchDescription([
        timesteps_arg,
        resume_arg,
        gazebo,
        rsp,
        spawn_robot,
        gz_bridge,
        controller_spawners,
        train_node,
    ])
