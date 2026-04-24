#!/usr/bin/env python3
"""Launch the Kinova feeding stack.

This brings up:
- MoveIt2 + arm controller integration
- RealSense camera driver
- TF2 static frames for robot/camera alignment
- mouth_tracker (publishes /mouth_3d_point)
- mouth_feeding_planner (waits for /feed_trigger)

The planner itself performs the trigger-driven motion sequence:
1) read the latest mouth point
2) transform it into the base/world frame with TF2
3) solve IK for approach/feed waypoints
4) execute joint trajectories directly
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


def _robot_urdf_xml() -> str:
    """Process the j2s6s200 standalone xacro and return the URDF XML string."""
    xacro_file = os.path.join(
        get_package_share_directory(KINOVA_DESC_PKG),
        "urdf",
        "j2s6s200_standalone.xacro",
    )
    return xacro.process_file(xacro_file).toprettyxml(indent="  ")


def _moveit_node_params() -> List[Dict]:
    """Build MoveIt-related parameter dictionaries."""
    package_dir = get_package_share_directory(PACKAGE_NAME)

    robot_description = {"robot_description": _robot_urdf_xml()}
    robot_semantic = {"robot_description_semantic": _load_text(PACKAGE_NAME, "config/j2s6s200.srdf")}
    robot_kinematics = {"robot_description_kinematics": _load_yaml(PACKAGE_NAME, "config/kinematics.yaml")}
    robot_joint_limits = {"robot_description_planning": _load_yaml(PACKAGE_NAME, "config/joint_limits.yaml")}
    sim_time = {"use_sim_time": False}

    _ompl_yaml = _load_yaml(PACKAGE_NAME, "config/ompl_planning.yaml")
    # Flatten: the YAML has a nested "move_group:" section for request_adapters;
    # bring those keys up to the same level as planner_configs / arm / gripper.
    _ompl_flat = {"planning_plugin": _ompl_yaml.get("planning_plugin",
                                                      "ompl_interface/OMPLPlanner")}
    _ompl_flat.update(_ompl_yaml.get("move_group", {}))   # request_adapters, start_state_max_bounds_error
    for _k in ("planner_configs", "arm", "gripper"):
        if _k in _ompl_yaml:
            _ompl_flat[_k] = _ompl_yaml[_k]

    # Add BFMT (Bidirectional Fast Marching Tree) — shortest/near-optimal paths.
    # Not in ompl_planning.yaml so we can keep that file clean.
    _ompl_flat.setdefault("planner_configs", {})["BFMTkConfigDefault"] = {
        "type": "geometric::BFMT",
        "num_samples": 1000,
        "radius_multiplier": 1.1,
        "balanced": 1,
        "optimality": 1,
        "heuristics": 1,
        "nearest_k": 1,
    }
    _ompl_flat.setdefault("arm", {}).setdefault("planner_configs", []).append(
        "BFMTkConfigDefault"
    )

    ompl_planning = {"move_group": _ompl_flat}

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

    if "moveit" not in get_packages_with_prefixes():
        return [
            LogInfo(
                msg=(
                    'WARNING: the "moveit" package is not installed. '
                    "Install MoveIt2 before launching the feeding stack."
                )
            )
        ]

    # Optional RViz launch.
    if LaunchConfiguration("use_rviz").perform(context).lower() == "true":
        bringup_dir = get_package_share_directory(KINOVA_BRINGUP_PKG)
        rviz_config_file = os.path.join(bringup_dir, "moveit_resource", "visualization.rviz")
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

    # robot_state_publisher: reads /joint_states and publishes the full j2s6s200
    # TF chain (root → j2s6s200_link_base → … → j2s6s200_end_effector).
    # Required so that the static EE→camera TF below connects into one tree.
    # /joint_states is published by kinova_sdk_node (run separately).
    nodes.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[
                {"robot_description": _robot_urdf_xml()},
                {"use_sim_time": False},
            ],
        )
    )

    # MoveIt2 move_group is required because the feeding planner calls /compute_ik.
    nodes.append(
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            name="move_group",
            output="screen",
            parameters=_moveit_node_params(),
        )
    )

    # Controller/relay helpers used by the existing Kinova setup.
    nodes.append(
        Node(
            package=PACKAGE_NAME,
            executable="joint_state_relay",
            name="joint_state_relay",
            output="screen",
        )
    )

    nodes.append(
        Node(
            package=PACKAGE_NAME,
            executable="scoop_action",
            name="scoop_action",
            output="screen",
        )
    )

    # World/root TF bridge. The SRDF uses a virtual world->root relation.
    nodes.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="world_to_root",
            output="screen",
            arguments=[
                "--x", "0",
                "--y", "0",
                "--z", "0",
                "--roll", "0",
                "--pitch", "0",
                "--yaw", "0",
                "--frame-id", "world",
                "--child-frame-id", "root",
            ],
        )
    )

    # EE → camera_color_optical_frame static TF.
    # Values are the INVERSE of the HandEye panel output (panel shows sensor→EE;
    # we need EE→sensor here).  Rerun convert_calibration.py after any re-calibration.
    nodes.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="ee_to_camera_optical",
            output="screen",
            arguments=[
                "--x", LaunchConfiguration("cam_x"),
                "--y", LaunchConfiguration("cam_y"),
                "--z", LaunchConfiguration("cam_z"),
                "--roll", LaunchConfiguration("cam_roll"),
                "--pitch", LaunchConfiguration("cam_pitch"),
                "--yaw", LaunchConfiguration("cam_yaw"),
                "--frame-id", "j2s6s200_end_effector",
                "--child-frame-id", "camera_color_optical_frame",
            ],
        )
    )

    # RealSense driver.
    # publish_tf is False because we provide our own EE→camera_color_optical_frame
    # above; letting the driver also publish camera_color_frame→camera_color_optical_frame
    # would create a TF conflict (two parents for the same frame).
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
                    "pointcloud.enable": "true",
                    "depth_module.profile": "640x480x30",
                    "rgb_camera.profile": "640x480x30",
                    "publish_tf": "false",
                }.items(),
            )
        )
    except Exception:
        nodes.append(
            LogInfo(
                msg=(
                    "WARNING: realsense2_camera is not installed. "
                    "Install it with: sudo apt install ros-humble-realsense2-camera"
                )
            )
        )

    # Mouth tracker: must publish /mouth_3d_point for the feeding planner.
    nodes.append(
        Node(
            package="mouth_tracking",
            executable="mouth_tracker",
            name="mouth_tracker",
            output="screen",
        )
    )

    # Demo-based feeding planner: record a good pose with remote control,
    # then /feed_trigger corrects it for the current mouth position.
    #   ros2 service call /record_feed_pose std_srvs/srv/Trigger
    #   ros2 service call /feed_trigger     std_srvs/srv/Trigger
    nodes.append(
        Node(
            package=KINOVA_BRINGUP_PKG,
            executable="demo_feed_planner",
            name="demo_feed_planner",
            output="screen",
        )
    )

    return nodes


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "cam_x",
            default_value="-0.04934",
            description="EE -> camera_color_optical_frame translation X in meters",
        ),
        DeclareLaunchArgument(
            "cam_y",
            default_value="-0.12927",
            description="EE -> camera_color_optical_frame translation Y in meters",
        ),
        DeclareLaunchArgument(
            "cam_z",
            default_value="0.14613",
            description="EE -> camera_color_optical_frame translation Z in meters",
        ),
        DeclareLaunchArgument(
            "cam_roll",
            default_value="0.13855",
            description="EE -> camera_color_optical_frame roll in radians",
        ),
        DeclareLaunchArgument(
            "cam_pitch",
            default_value="-0.49254",
            description="EE -> camera_color_optical_frame pitch in radians",
        ),
        DeclareLaunchArgument(
            "cam_yaw",
            default_value="0.23797",
            description="EE -> camera_color_optical_frame yaw in radians",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Launch RViz with the MoveIt config",
        ),
    ]

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=_launch_setup)])
