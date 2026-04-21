#!/usr/bin/env python

# Copyright 1996-2021 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch Webots Universal Robot simulation with MoveIt2."""

import os
import pathlib
import yaml
from launch.actions import LogInfo
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, get_packages_with_prefixes
import xacro
KINOVA_SRC   = '/home/amma/kinova_ws/rl_v2-master/src/kinova-ros2'

SDK_LIB_DIR  = f'{KINOVA_SRC}/kinova_driver/lib/x86_64-linux-gnu'

PACKAGE_NAME = 'j2s6s200_moveit_config'


def generate_launch_description():
    launch_description_nodes = []
    package_dir = get_package_share_directory(PACKAGE_NAME)

    def load_file(filename):
        return pathlib.Path(os.path.join(package_dir, 'config', filename)).read_text()

    def load_yaml(filename):
        return yaml.safe_load(load_file(filename))

    # Check if moveit is installed
    if 'moveit' in get_packages_with_prefixes():
        # Configuration
        xacro_file = os.path.join(get_package_share_directory('kinova_description'), 'urdf', 'j2s6s200_standalone.xacro')
        doc = xacro.process_file(xacro_file)
        description = {'robot_description': doc.toprettyxml(indent='  ')}

        description_semantic = {'robot_description_semantic': load_file('j2s6s200.srdf')}
        description_kinematics = {'robot_description_kinematics': load_yaml('kinematics.yaml')}
        description_joint_limits = {'robot_description_planning': load_yaml('joint_limits.yaml')}
        sim_time = {'use_sim_time': False}

        # Rviz node
        rviz_config_file = os.path.join(package_dir, 'config', 'visualization.rviz')

        launch_description_nodes.append(
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config_file],
                parameters=[
                    description,
                    description_semantic,
                    description_kinematics,
                    description_joint_limits,
                    sim_time
                ],
            )
        )

        # Planning Configuration
        ompl_planning_pipeline_config = {
            "move_group": {
                "planning_plugin": "ompl_interface/OMPLPlanner",
                "request_adapters": """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
                "start_state_max_bounds_error": 0.1,
            }
        }
        # MoveIt2 node
        ompl_planning_yaml = load_yaml('ompl_planning.yaml')
        ompl_planning_pipeline_config["move_group"].update(ompl_planning_yaml)

        moveit_controllers = {
            "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
            "moveit_simple_controller_manager": {
                "controller_names": ["arm_controller"],
                "arm_controller": {
                    "type": "FollowJointTrajectory",
                    "action_ns": "follow_joint_trajectory",   # NOT the full path
                    "default": True,
                    "joints": [
                        "j2s6s200_joint_1",
                        "j2s6s200_joint_2",
                        "j2s6s200_joint_3",
                        "j2s6s200_joint_4",
                        "j2s6s200_joint_5",
                        "j2s6s200_joint_6",
                    ],
                },
            },
        }

        launch_description_nodes.append(
            Node(
                package='moveit_ros_move_group',
                executable='move_group',
                output='screen',
                parameters=[
                    description,
                    description_semantic,
                    description_kinematics,
                    moveit_controllers,
                    ompl_planning_pipeline_config,
                    description_joint_limits,
                    sim_time
                ],
                # remappings=[('/joint_states', '/j2s6s200_driver/out/joint_state')],
            )
        )
        # launch_description_nodes.append(
        #     Node(
        #         package='j2s6s200_moveit_config',
        #         executable='trajectory_executor',
        #         name='kinova_hw_bridge',
        #         output='screen',
        #         additional_env={
        #             'LD_LIBRARY_PATH': (
        #                 SDK_LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        #             )
        #         },
        #         # parameters=[NO_SIM],
        #     )
        # )
        launch_description_nodes.append(
            Node(
                package='j2s6s200_moveit_config',
                executable='joint_state_relay',
                output='screen'
            )
        )
        launch_description_nodes.append(
            Node(
                package='j2s6s200_moveit_config',
                executable='scoop_action',
                name='scoop_action',
                output='screen',
            )
        )
        # prescoop_node = Node(
        #     package='kinova_bringup',       # Where you saved the python script
        #     executable='go_to_prescoop', # The python script above
        #     name='prescoop_planner',
        #     parameters=[description,
        #             description_semantic,
        #             description_kinematics,
        #             moveit_controllers,
        #             ompl_planning_pipeline_config,
        #             description_joint_limits,
        #             sim_time],     # Injecting MoveIt configs here is mandatory
        #     output='screen'
        # )
        # launch_description_nodes.append(prescoop_node)
    else:
        launch_description_nodes.append(LogInfo(msg='"moveit" package is not installed, \
                                                please install it in order to run this demo.'))

    return LaunchDescription(launch_description_nodes)