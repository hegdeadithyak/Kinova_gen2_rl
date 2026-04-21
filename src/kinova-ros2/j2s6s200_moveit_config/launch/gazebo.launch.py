import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro

def generate_launch_description():
    # 1. Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
    )

    # 2. Get URDF via xacro
    robot_description_path = os.path.join(
        get_package_share_directory('j2s6s200_moveit_config'), 'urdf', 'j2s6s200.urdf.xacro')
    robot_description_config = xacro.process_file(robot_description_path)
    robot_urdf = robot_description_config.toxml()

    # 3. Spawn Robot in Gazebo
    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description', '-entity', 'j2s6s200'],
                        output='screen')

    # 4. Load Controllers (Spawners)
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen'
    )

    load_arm_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'arm_controller'],
        output='screen'
    )

    return LaunchDescription([
        # We need a robot_state_publisher here to publish the URDF to the /robot_description topic
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_urdf, 'use_sim_time': True}]
        ),
        gazebo,
        spawn_entity,
        load_joint_state_broadcaster,
        load_arm_controller,
    ])