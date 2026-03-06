#!/usr/bin/env python3
"""
Standalone RViz2 launcher for Kinova j2n6s300 + MoveIt2.
Run this AFTER the main bringup launch is fully up.
Passes all required MoveIt parameters so kinematics and
interactive markers work correctly.
"""
import os
import pathlib
import yaml
import xacro

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

PACKAGE_NAME = 'kinova_bringup'
ROBOT_NAME   = 'j2n6s300'

def generate_launch_description():
    pkg_dir = get_package_share_directory(PACKAGE_NAME)

    def read_file(name):
        return pathlib.Path(pkg_dir, 'moveit_resource', name).read_text()

    def read_yaml(name):
        return yaml.safe_load(read_file(name))

    def moveit_res(name):
        return str(pathlib.Path(pkg_dir, 'moveit_resource', name))

    # Generate robot_description from xacro at runtime (same as main launch)
    xacro_file = os.path.join(
        get_package_share_directory('kinova_description'),
        'urdf', f'{ROBOT_NAME}_standalone.xacro',
    )
    robot_description_content = xacro.process_file(xacro_file).toprettyxml(indent='  ')
    description   = {'robot_description': robot_description_content}
    # CRITICAL: must be wrapped under this exact key or IK solver is not found
    kinematics    = {'robot_description_kinematics': read_yaml('kinematics.yaml')}

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', moveit_res('visualization.rviz')],
        parameters=[
            description,
            {'use_sim_time': True},
            {'robot_description_semantic':  read_file(f'{ROBOT_NAME}.srdf')},
            kinematics,
            {'robot_description_planning':  read_yaml('joint_limits.yaml')},
        ],
    )

    return LaunchDescription([rviz])