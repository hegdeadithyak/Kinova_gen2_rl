#!/usr/bin/env python3
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


PACKAGE_NAME      = "j2s6s200_moveit_config"
KINOVA_DESC_PKG   = "kinova_description"
KINOVA_BRINGUP_PKG = "kinova_bringup"
REALSENSE_PKG     = "realsense2_camera"


def _load_text(package_name: str, relative_path: str) -> str:
    base = get_package_share_directory(package_name)
    return pathlib.Path(os.path.join(base, relative_path)).read_text()


def _load_yaml(package_name: str, relative_path: str):
    return yaml.safe_load(_load_text(package_name, relative_path))


def _robot_urdf_xml() -> str:
    xacro_file = os.path.join(
        get_package_share_directory(KINOVA_DESC_PKG), "urdf", "j2s6s200_standalone.xacro")
    return xacro.process_file(xacro_file).toprettyxml(indent="  ")


def _moveit_node_params() -> List[Dict]:
    robot_description      = {"robot_description": _robot_urdf_xml()}
    robot_semantic         = {"robot_description_semantic": _load_text(PACKAGE_NAME, "config/j2s6s200.srdf")}
    robot_kinematics       = {"robot_description_kinematics": _load_yaml(PACKAGE_NAME, "config/kinematics.yaml")}
    robot_joint_limits     = {"robot_description_planning": _load_yaml(PACKAGE_NAME, "config/joint_limits.yaml")}
    sim_time               = {"use_sim_time": False}

    _ompl_yaml = _load_yaml(PACKAGE_NAME, "config/ompl_planning.yaml")
    # The YAML has a nested "move_group:" key for request_adapters — flatten it to the top level.
    _ompl_flat = {"planning_plugin": _ompl_yaml.get("planning_plugin", "ompl_interface/OMPLPlanner")}
    _ompl_flat.update(_ompl_yaml.get("move_group", {}))
    for _k in ("planner_configs", "arm", "gripper"):
        if _k in _ompl_yaml:
            _ompl_flat[_k] = _ompl_yaml[_k]

    _ompl_flat.setdefault("planner_configs", {})["BFMTkConfigDefault"] = {
        "type": "geometric::BFMT",
        "num_samples": 1000,
        "radius_multiplier": 1.1,
        "balanced": 1,
        "optimality": 1,
        "heuristics": 1,
        "nearest_k": 1,
    }
    _ompl_flat.setdefault("arm", {}).setdefault("planner_configs", []).append("BFMTkConfigDefault")

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
                    "j2s6s200_joint_1", "j2s6s200_joint_2", "j2s6s200_joint_3",
                    "j2s6s200_joint_4", "j2s6s200_joint_5", "j2s6s200_joint_6",
                ],
            },
        },
    }

    chomp_config = {
        "planning_pipelines": ["chomp"],
        "planning_plugin": "chomp_interface/CHOMPPlanner",
        "chomp": {
            "planning_time_limit": 5.0,
            "max_iterations": 200,
            "smoothness_cost_weight": 0.1,
            "obstacle_cost_weight": 1.0,
            "learning_rate": 0.01,
            "ridge_factor": 0.001,
        }
    }

    return [
        robot_description, robot_semantic, robot_kinematics,
        robot_joint_limits, moveit_controllers, chomp_config, sim_time,
    ]


def _launch_setup(context, *args, **kwargs):
    nodes = []

    if "moveit" not in get_packages_with_prefixes():
        return [LogInfo(msg='WARNING: "moveit" not installed. Install MoveIt2 before launching.')]

    if LaunchConfiguration("use_rviz").perform(context).lower() == "true":
        bringup_dir      = get_package_share_directory(KINOVA_BRINGUP_PKG)
        rviz_config_file = os.path.join(bringup_dir, "moveit_resource", "visualization.rviz")
        nodes.append(Node(
            package="rviz2", executable="rviz2", name="rviz2", output="screen",
            arguments=["-d", rviz_config_file],
            parameters=_moveit_node_params()))

    nodes.append(Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": _robot_urdf_xml()}, {"use_sim_time": False}]))

    nodes.append(Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=_moveit_node_params()))

    nodes.append(Node(
        package=PACKAGE_NAME, executable="joint_state_relay",
        name="joint_state_relay", output="screen"))

    nodes.append(Node(
        package=PACKAGE_NAME, executable="scoop_action",
        name="scoop_action", output="screen"))

    nodes.append(Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_root",
        output="screen",
        arguments=["--x", "0", "--y", "0", "--z", "0",
                   "--roll", "0", "--pitch", "0", "--yaw", "0",
                   "--frame-id", "world", "--child-frame-id", "root"]))

    # EE → camera TF. Values are the INVERSE of HandEye panel output (panel shows sensor→EE).
    # Rerun convert_calibration.py after any re-calibration.
    nodes.append(Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="ee_to_camera_optical",
        output="screen",
        arguments=[
            "--x",     LaunchConfiguration("cam_x"),
            "--y",     LaunchConfiguration("cam_y"),
            "--z",     LaunchConfiguration("cam_z"),
            "--roll",  LaunchConfiguration("cam_roll"),
            "--pitch", LaunchConfiguration("cam_pitch"),
            "--yaw",   LaunchConfiguration("cam_yaw"),
            "--frame-id", "j2s6s200_end_effector",
            "--child-frame-id", "camera_color_optical_frame"]))

    try:
        rs_share = get_package_share_directory(REALSENSE_PKG)
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([rs_share, "/launch/rs_launch.py"]),
            launch_arguments={
                "camera_name":           "camera",
                "enable_color":          "true",
                "enable_depth":          "true",
                "align_depth.enable":    "true",
                "pointcloud.enable":     "true",
                "depth_module.profile":  "640x480x30",
                "rgb_camera.profile":    "640x480x30",
                "publish_tf":            "false",  # we own the EE→camera TF; driver publishing it causes a conflict
            }.items()))
    except Exception:
        nodes.append(LogInfo(msg=(
            "WARNING: realsense2_camera not installed. "
            "Install with: sudo apt install ros-humble-realsense2-camera")))

    nodes.append(Node(
        package="mouth_tracking", executable="mouth_tracker",
        name="mouth_tracker", output="screen"))

    nodes.append(Node(
        package=KINOVA_BRINGUP_PKG, executable="demo_feed_planner",
        name="demo_feed_planner", output="screen"))

    return nodes


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("cam_x",     default_value="0.0779"),
        DeclareLaunchArgument("cam_y",     default_value="0.0660"),
        DeclareLaunchArgument("cam_z",     default_value="-0.0587"),
        DeclareLaunchArgument("cam_roll",  default_value="0.2824"),
        DeclareLaunchArgument("cam_pitch", default_value="0.1040"),
        DeclareLaunchArgument("cam_yaw",   default_value="2.7041"),
        DeclareLaunchArgument("use_rviz",  default_value="true"),
    ]
    return LaunchDescription(declared_arguments + [OpaqueFunction(function=_launch_setup)])
