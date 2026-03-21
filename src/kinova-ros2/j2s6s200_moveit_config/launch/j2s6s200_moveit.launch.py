import os
from pathlib import Path

import yaml
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ── helpers ──────────────────────────────────────────────────────────────────

def load_yaml(pkg, *parts):
    path = os.path.join(get_package_share_directory(pkg), *parts)
    with open(path) as f:
        return yaml.safe_load(f)


def pkg(name):
    return get_package_share_directory(name)


# ── entry point ───────────────────────────────────────────────────────────────

def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="false",
        description="Use simulation (Gazebo) clock"
    )
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}

    # ── 1. Robot description ─────────────────────────────────────────────────
    xacro_file = os.path.join(
        pkg("j2s6s200_moveit_config"), "urdf", "j2s6s200.urdf.xacro"
    )
    robot_description_raw = xacro.process_file(xacro_file).toxml()
    robot_description = {"robot_description": robot_description_raw}

    # ── 2. SRDF ──────────────────────────────────────────────────────────────
    srdf_path = os.path.join(
        pkg("j2s6s200_moveit_config"), "config", "j2s6s200.srdf"
    )
    robot_description_semantic = {
        "robot_description_semantic": Path(srdf_path).read_text()
    }

    # ── 3. Kinematics ────────────────────────────────────────────────────────
    kinematics = {
        "robot_description_kinematics":
            load_yaml("j2s6s200_moveit_config", "config", "kinematics.yaml")
    }

    # ── 4. Joint limits ──────────────────────────────────────────────────────
    joint_limits = {
        "robot_description_planning":
            load_yaml("j2s6s200_moveit_config", "config", "joint_limits.yaml")
    }

    # ── 5. Planning pipeline (OMPL) ──────────────────────────────────────────
    ompl_yaml = load_yaml("j2s6s200_moveit_config", "config", "ompl_planning.yaml")
    planning_pipeline = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl_yaml,
    }

    # ── 6. Controllers ───────────────────────────────────────────────────────
    moveit_controllers = load_yaml(
        "j2s6s200_moveit_config", "config", "moveit_controllers.yaml"
    )

    trajectory_execution = {
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    # ── 7. Planning scene monitor ────────────────────────────────────────────
    psm_params = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
    }

    # ── Common param bundle ──────────────────────────────────────────────────
    common = {
        **use_sim_time,
        **robot_description,
        **robot_description_semantic,
        **kinematics,
        **joint_limits,
    }

    # ════════════════════════════════════════════════════════════════════════
    # Nodes
    # ════════════════════════════════════════════════════════════════════════

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            common,
            planning_pipeline,
            trajectory_execution,
            moveit_controllers,
            psm_params,
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[common],
    )

    # ros2_control node — uncomment if you have hardware / Gazebo
    # ros2_control_node = Node(
    #     package="controller_manager",
    #     executable="ros2_control_node",
    #     parameters=[
    #         robot_description,
    #         os.path.join(pkg("j2s6s200_moveit_config"), "config", "ros2_controllers.yaml"),
    #     ],
    #     output="screen",
    # )

    rviz_config = os.path.join(
        pkg("j2s6s200_moveit_config"), "config", "moveit.rviz"
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", rviz_config] if os.path.exists(rviz_config) else [],
        parameters=[common, planning_pipeline, kinematics],
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_state_publisher,
        move_group,
        rviz,
        # ros2_control_node,
    ])
