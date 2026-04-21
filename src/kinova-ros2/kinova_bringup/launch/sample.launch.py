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

"""Launch Kinova j2s6s200 arm + MoveIt2 + mouth-triggered feeding planner."""

import os
import pathlib
import yaml
from launch.actions import LogInfo, DeclareLaunchArgument
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory, get_packages_with_prefixes
import xacro
KINOVA_SRC   = '/home/amma/kinova_ws/rl_v2-master/src/kinova-ros2'

SDK_LIB_DIR  = f'{KINOVA_SRC}/kinova_driver/lib/x86_64-linux-gnu'

PACKAGE_NAME = 'j2s6s200_moveit_config'


def generate_launch_description():
    launch_description_nodes = []
    package_dir = get_package_share_directory(PACKAGE_NAME)

    # ── Camera TF arguments ───────────────────────────────────────────────────
    # Hand-eye calibration (gripper ↔ camera).  The correct rigid transform is
    # what "AX = XB" hand-eye calibration solves for; without proper data we
    # pick a physically reasonable default and let the user tune with CLI args.
    #
    # Parent frame: j2s6s200_end_effector  (NOT link_6).
    #   link_6 → end_effector has rotation (roll=180°, yaw=90°) and z=-0.16 m,
    #   meaning link_6's +Z points AWAY from the tool.  Using end_effector as
    #   the parent gives us a clean frame whose +Z is the tool (spoon) axis.
    #
    # Camera optical frame convention:  +X right, +Y down, +Z forward (out of
    # lens).  For the lens to look along the tool axis (end_effector +Z), we
    # need camera +Z ≡ end_effector +Z.  With roll=pitch=yaw=0 below this is
    # already satisfied, but the image "up" direction depends on how the
    # camera is physically rotated around its own Z axis — tune cam_yaw if
    # the image appears upside-down or sideways.
    #
    # cam_x / cam_y / cam_z are measured from end_effector origin (in
    # end_effector frame, metres).  Defaults assume the camera sits ~5 cm
    # above the spoon mount on the +Y side of the gripper.
    #
    # Override at launch time, e.g.:
    #   ros2 launch kinova_bringup sample.launch.py \
    #       cam_x:=0.0 cam_y:=0.05 cam_z:=0.0 cam_yaw:=0.0
    cam_args = [
        DeclareLaunchArgument('cam_x',     default_value='0.0'),
        DeclareLaunchArgument('cam_y',     default_value='0.05'),
        DeclareLaunchArgument('cam_z',     default_value='0.0'),
        DeclareLaunchArgument('cam_roll',  default_value='0.0'),
        DeclareLaunchArgument('cam_pitch', default_value='0.0'),
        DeclareLaunchArgument('cam_yaw',   default_value='0.0'),
    ]
    launch_description_nodes.extend(cam_args)

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

        # ── Static TF: world → root (identity) ───────────────────────────────
        # The SRDF declares a virtual joint world→root, but nothing broadcasts
        # it. Without this, `world` is a dangling frame and TF lookups from
        # world to any arm link fail with "two or more unconnected trees".
        launch_description_nodes.append(
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='world_to_root',
                output='screen',
                arguments=[
                    '--x', '0', '--y', '0', '--z', '0',
                    '--roll', '0', '--pitch', '0', '--yaw', '0',
                    '--frame-id', 'world',
                    '--child-frame-id', 'root',
                ],
            )
        )

        # ── Static TF: j2s6s200_link_6 → camera_color_optical_frame ──────────
        # Publishes the camera's fixed pose so mouth points can be transformed
        # into the robot world frame.  Adjust the cam_* launch args to match
        # your physical camera mounting.
        launch_description_nodes.append(
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='camera_tf',
                output='screen',
                arguments=[
                    '--x',           LaunchConfiguration('cam_x'),
                    '--y',           LaunchConfiguration('cam_y'),
                    '--z',           LaunchConfiguration('cam_z'),
                    '--roll',        LaunchConfiguration('cam_roll'),
                    '--pitch',       LaunchConfiguration('cam_pitch'),
                    '--yaw',         LaunchConfiguration('cam_yaw'),
                    '--frame-id',    'j2s6s200_end_effector',
                    '--child-frame-id', 'camera_color_optical_frame',
                ],
            )
        )

        # ── Mouth tracker (RealSense + MediaPipe face landmark) ───────────────
        launch_description_nodes.append(
            Node(
                package='mouth_tracking',
                executable='mouth_tracker',
                name='mouth_tracker',
                output='screen',
            )
        )

        # ── Feeding planner — call /feed_trigger to activate ─────────────────
        # On trigger: reads /mouth_3d_point, transforms to world frame,
        # computes EE pose 10 cm back (= 5 cm gap + 5 cm spoon), flat wrist,
        # plans Cartesian path, executes via arm_controller.
        launch_description_nodes.append(
            Node(
                package='kinova_bringup',
                executable='mouth_feeding_planner',
                name='mouth_feeding_planner',
                output='screen',
            )
        )

    else:
        launch_description_nodes.append(LogInfo(msg='"moveit" package is not installed, \
                                                please install it in order to run this demo.'))

    return LaunchDescription(launch_description_nodes)