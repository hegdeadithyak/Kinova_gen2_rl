#!/usr/bin/env python3
"""Launch the click-to-move demo.

Mirrors sample.launch.py exactly, replacing demo_feed_planner with
point_selector_node + arm_mover_node.

Calibration values (EE → camera_color_optical_frame) can be overridden
as launch args, same as sample.launch.py.
"""
from __future__ import annotations

import os
import pathlib

import yaml
import xacro

from ament_index_python.packages import get_package_share_directory, get_packages_with_prefixes
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


MOVEIT_PKG      = "j2s6s200_moveit_config"
DESC_PKG        = "kinova_description"
BRINGUP_PKG     = "kinova_bringup"
REALSENSE_PKG   = "realsense2_camera"


def _load_text(pkg, rel):
    return pathlib.Path(os.path.join(get_package_share_directory(pkg), rel)).read_text()

def _load_yaml(pkg, rel):
    return yaml.safe_load(_load_text(pkg, rel))

def _robot_urdf_xml():
    xacro_file = os.path.join(
        get_package_share_directory(DESC_PKG), "urdf", "j2s6s200_standalone.xacro")
    return xacro.process_file(xacro_file).toprettyxml(indent="  ")

def _moveit_params():
    ompl_yaml = _load_yaml(MOVEIT_PKG, "config/ompl_planning.yaml")
    ompl_flat = {"planning_plugin": ompl_yaml.get("planning_plugin", "ompl_interface/OMPLPlanner")}
    ompl_flat.update(ompl_yaml.get("move_group", {}))
    for k in ("planner_configs", "arm", "gripper"):
        if k in ompl_yaml:
            ompl_flat[k] = ompl_yaml[k]
    ompl_flat.setdefault("planner_configs", {})["BFMTkConfigDefault"] = {
        "type": "geometric::BFMT", "num_samples": 1000,
        "radius_multiplier": 1.1, "balanced": 1,
        "optimality": 1, "heuristics": 1, "nearest_k": 1,
    }
    ompl_flat.setdefault("arm", {}).setdefault("planner_configs", []).append("BFMTkConfigDefault")

    return [
        {"robot_description": _robot_urdf_xml()},
        {"robot_description_semantic": _load_text(MOVEIT_PKG, "config/j2s6s200.srdf")},
        {"robot_description_kinematics": _load_yaml(MOVEIT_PKG, "config/kinematics.yaml")},
        {"robot_description_planning": _load_yaml(MOVEIT_PKG, "config/joint_limits.yaml")},
        {"moveit_controller_manager":
            "moveit_simple_controller_manager/MoveItSimpleControllerManager",
         "moveit_simple_controller_manager": {
             "controller_names": ["arm_controller"],
             "arm_controller": {
                 "type": "FollowJointTrajectory",
                 "action_ns": "follow_joint_trajectory",
                 "default": True,
                 "joints": [f"j2s6s200_joint_{i}" for i in range(1, 7)],
             },
         }},
        # In multi-pipeline mode each pipeline is configured under its own name
        {"planning_pipelines": ["ompl", "pilz_industrial_motion_planner"],
         "default_planning_pipeline": "ompl"},
        {"ompl": ompl_flat},
        {"pilz_industrial_motion_planner": {
            "planning_plugin": "pilz_industrial_motion_planner/CommandPlanner",
            "request_adapters": " ".join([
                "default_planning_request_adapters/ResolveConstraintFrames",
                "default_planning_request_adapters/ValidateWorkspaceBounds",
                "default_planning_request_adapters/CheckStartStateBounds",
                "default_planning_request_adapters/CheckStartStateCollision",
            ]),
            "start_state_max_bounds_error": 0.1,
        }},
        {"use_sim_time": False},
    ]


def _launch_setup(context, *args, **kwargs):
    if "moveit" not in get_packages_with_prefixes():
        return [LogInfo(msg='WARNING: "moveit" not installed.')]

    params = _moveit_params()
    nodes = []

    nodes.append(Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        name="robot_state_publisher", output="screen",
        parameters=[{"robot_description": _robot_urdf_xml()}, {"use_sim_time": False}]))

    nodes.append(Node(
        package="moveit_ros_move_group", executable="move_group",
        name="move_group", output="screen", parameters=params))

    nodes.append(Node(
        package=MOVEIT_PKG, executable="joint_state_relay",
        name="joint_state_relay", output="screen"))

    nodes.append(Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="world_to_root", output="screen",
        arguments=["--x", "0", "--y", "0", "--z", "0",
                   "--roll", "0", "--pitch", "0", "--yaw", "0",
                   "--frame-id", "world", "--child-frame-id", "root"]))

    # EE → camera_color_optical_frame hand-eye TF
    nodes.append(Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="ee_to_camera_optical", output="screen",
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
                "camera_name":        "camera",
                "enable_color":       "true",
                "enable_depth":       "true",
                "align_depth.enable": "true",
                "pointcloud.enable":  "false",
                "depth_module.depth_profile": "640x480x30",
                "rgb_camera.color_profile":   "640x480x30",
                "publish_tf":         "false",
            }.items()))
    except Exception:
        nodes.append(LogInfo(msg="WARNING: realsense2_camera not found"))

    nodes.append(Node(
        package="point_goal_test", executable="point_selector_node",
        name="point_selector_node", output="screen",
        parameters=[{
            'cam_offset_x': float(context.perform_substitution(LaunchConfiguration("cam_offset_x"))),
            'cam_offset_y': float(context.perform_substitution(LaunchConfiguration("cam_offset_y"))),
        }]))

    nodes.append(Node(
        package="point_goal_test", executable="arm_mover_node",
        name="arm_mover_node", output="screen",
        parameters=[{
            'offset_x': float(context.perform_substitution(LaunchConfiguration("offset_x"))),
            'offset_y': float(context.perform_substitution(LaunchConfiguration("offset_y"))),
            'offset_z': float(context.perform_substitution(LaunchConfiguration("offset_z"))),
        }]))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("cam_x",     default_value="0.077"),
        DeclareLaunchArgument("cam_y",     default_value="0.073"),
        DeclareLaunchArgument("cam_z",     default_value="-0.0878"),
        DeclareLaunchArgument("cam_roll",  default_value="3.0079"),
        DeclareLaunchArgument("cam_pitch", default_value="2.9800"),
        DeclareLaunchArgument("cam_yaw",   default_value="-1.0334"),
        DeclareLaunchArgument("offset_x",     default_value="0.0"),
        DeclareLaunchArgument("offset_y",     default_value="0.0"),
        DeclareLaunchArgument("offset_z",     default_value="0.0"),
        DeclareLaunchArgument("cam_offset_x", default_value="-0.14"),
        DeclareLaunchArgument("cam_offset_y", default_value="-0.14"),
        OpaqueFunction(function=_launch_setup),
    ])
