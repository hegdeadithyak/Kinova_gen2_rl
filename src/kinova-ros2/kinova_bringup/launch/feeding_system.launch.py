"""
feeding_system.launch.py
========================
Launches:
  1. mouth_detector_node  — MediaPipe face mesh + /feed_trigger service
  2. feeding_executor_node — Consumes /mouth_pose, commands Kinova arm via MoveIt2

Usage:
  ros2 launch <your_package> feeding_system.launch.py

Then trigger a feed:
  ros2 service call /feed_trigger std_srvs/srv/Trigger {}

Optionally view the annotated debug camera feed:
  ros2 run rqt_image_view rqt_image_view /mouth_detection/image
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # ── Launch arguments ─────────────────────────────────────────────
    approach_offset_arg = DeclareLaunchArgument(
        'approach_offset_z', default_value='0.05',
        description='Distance (m) to stop in front of the mouth')

    planning_group_arg = DeclareLaunchArgument(
        'planning_group', default_value='arm',
        description='MoveIt2 planning group name for your robot')

    vel_scale_arg = DeclareLaunchArgument(
        'vel_scale', default_value='0.15',
        description='Velocity scaling (0-1), keep low for patient safety')

    # ── Mouth Detector Node ──────────────────────────────────────────
    mouth_detector_node = Node(
        package='kinova_bringup',           # ← replace with your package name
        executable='mouth_detector_node',
        name='mouth_detector_node',
        output='screen',
        parameters=[{
            'approach_offset_z':         LaunchConfiguration('approach_offset_z'),
            'min_detection_confidence':  0.6,
            'min_tracking_confidence':   0.5,
            'camera_frame':              'camera_color_optical_frame',
        }],
        remappings=[
            # Remap if your camera topics differ:
            # ('/camera/color/image_raw', '/realsense/color/image_raw'),
        ],
    )

    # ── Feeding Executor Node ────────────────────────────────────────
    feeding_executor_node = Node(
        package='kinova_bringup',           # ← replace with your package name
        executable='feeding_executor_node',
        name='feeding_executor_node',
        output='screen',
        parameters=[{
            'planning_group':    LaunchConfiguration('planning_group'),
            'end_effector_link': 'tool_frame',    # Kinova default EE frame
            'robot_base_frame':  'base_link',
            'planning_time':     10.0,
            'vel_scale':         LaunchConfiguration('vel_scale'),
            'acc_scale':         0.1,
        }],
    )

    return LaunchDescription([
        approach_offset_arg,
        planning_group_arg,
        vel_scale_arg,
        mouth_detector_node,
        feeding_executor_node,
    ])