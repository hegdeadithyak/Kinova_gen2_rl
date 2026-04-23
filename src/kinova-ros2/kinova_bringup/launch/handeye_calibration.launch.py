#!/usr/bin/env python3
"""Launch the hand-eye calibration session for the Kinova j2s6s200 (eye-in-hand).

Bring up:
  - MoveIt2 move_group
  - RealSense camera driver
  - world -> root static TF
  - RViz with the HandEyeCalibration panel pre-configured

After calibration, update sample.launch.py cam_x/y/z/roll/pitch/yaw with the
transform reported in the RViz panel (sensor -> end-effector).
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List

import yaml
import xacro

from ament_index_python.packages import get_package_share_directory, get_packages_with_prefixes
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PACKAGE_NAME = "j2s6s200_moveit_config"
KINOVA_DESC_PKG = "kinova_description"
KINOVA_BRINGUP_PKG = "kinova_bringup"
REALSENSE_PKG = "realsense2_camera"


def _load_text(package_name: str, relative_path: str) -> str:
    base = get_package_share_directory(package_name)
    return pathlib.Path(os.path.join(base, relative_path)).read_text()


def _load_yaml(package_name: str, relative_path: str):
    return yaml.safe_load(_load_text(package_name, relative_path))


def _moveit_node_params() -> List[Dict]:
    package_dir = get_package_share_directory(PACKAGE_NAME)

    xacro_file = os.path.join(
        get_package_share_directory(KINOVA_DESC_PKG),
        "urdf",
        "j2s6s200_standalone.xacro",
    )
    robot_doc = xacro.process_file(xacro_file)

    robot_description = {"robot_description": robot_doc.toprettyxml(indent="  ")}
    robot_semantic = {"robot_description_semantic": _load_text(PACKAGE_NAME, "config/j2s6s200.srdf")}
    robot_kinematics = {"robot_description_kinematics": _load_yaml(PACKAGE_NAME, "config/kinematics.yaml")}
    robot_joint_limits = {"robot_description_planning": _load_yaml(PACKAGE_NAME, "config/joint_limits.yaml")}
    sim_time = {"use_sim_time": False}

    ompl_planning = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": (
                "default_planner_request_adapters/AddTimeOptimalParameterization "
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ),
            "start_state_max_bounds_error": 0.1,
        }
    }
    ompl_planning["move_group"].update(_load_yaml(PACKAGE_NAME, "config/ompl_planning.yaml"))

    moveit_controllers = {
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {
            "controller_names": ["arm_controller"],
            "arm_controller": {
                "type": "FollowJointTrajectory",
                "action_ns": "follow_joint_trajectory",
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

    return [
        robot_description,
        robot_semantic,
        robot_kinematics,
        robot_joint_limits,
        moveit_controllers,
        ompl_planning,
        sim_time,
    ]


def _launch_setup(context, *args, **kwargs):
    nodes = []

    bringup_dir = get_package_share_directory(KINOVA_BRINGUP_PKG)
    rviz_config_file = os.path.join(bringup_dir, "moveit_resource", "handeye_calibration.rviz")

    nodes.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config_file],
            parameters=_moveit_node_params(),
        )
    )

    nodes.append(
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            name="move_group",
            output="screen",
            parameters=_moveit_node_params(),
        )
    )

    nodes.append(
        Node(
            package=PACKAGE_NAME,
            executable="joint_state_relay",
            name="joint_state_relay",
            output="screen",
        )
    )

    # world -> root static TF (SRDF virtual joint)
    nodes.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="world_to_root",
            output="screen",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--roll", "0", "--pitch", "0", "--yaw", "0",
                "--frame-id", "world",
                "--child-frame-id", "root",
            ],
        )
    )

    # RealSense driver
    try:
        rs_share = get_package_share_directory(REALSENSE_PKG)
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([rs_share, "/launch/rs_launch.py"]),
                launch_arguments={
                    "camera_name": "camera",
                    "enable_color": "true",
                    "enable_depth": "true",
                    "align_depth.enable": "true",
                    "depth_module.depth_profile": "640x480x30",
                    "rgb_camera.color_profile": "640x480x30",
                    "publish_tf": "true",
                    "tf_publish_rate": "0.0",
                }.items(),
            )
        )
    except Exception:
        nodes.append(
            LogInfo(msg="WARNING: realsense2_camera not found. Start the camera driver separately.")
        )

    return nodes


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_launch_setup)])
