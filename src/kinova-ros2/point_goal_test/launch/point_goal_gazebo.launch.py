#!/usr/bin/env python3
import os
import re
import pathlib
import yaml
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, LogInfo, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# --- Configuration ---
ROBOT_NAME = 'j2s6s200'
WORLD_FILE_NAME = 'point_goal_scene.sdf'
PKG_POINT_GOAL = 'point_goal_test'
PKG_DESCRIPTION = 'kinova_description'
PKG_MOVEIT_CONFIG = 'j2s6s200_moveit_config'

SIM_TIME = {'use_sim_time': True}

# Robot spawn position in the world (matching pedestal in point_goal_scene.sdf)
SPAWN_X = '0.40'
SPAWN_Y = '-0.50'
SPAWN_Z = '0.85'

def _load_text(pkg, rel):
    return pathlib.Path(os.path.join(get_package_share_directory(pkg), rel)).read_text()

def _load_yaml(pkg, rel):
    return yaml.safe_load(_load_text(pkg, rel))

def _patch_urdf(raw_xml, controllers_yaml):
    """Add ros2_control and gazebo plugins to the URDF."""
    p = re.sub(r'<gazebo[^>]*>.*?</gazebo>', '', raw_xml, flags=re.DOTALL | re.IGNORECASE)
    
    # Initialize joint positions
    INIT = {
        f'{ROBOT_NAME}_joint_1': 2.31,
        f'{ROBOT_NAME}_joint_2': 4.09,
        f'{ROBOT_NAME}_joint_3': 1.20,
        f'{ROBOT_NAME}_joint_4': 1.65,
        f'{ROBOT_NAME}_joint_5': 4.64,
        f'{ROBOT_NAME}_joint_6': -5.93,
    }

    def jblock(name):
        v = INIT.get(name, 0.0)
        return f"""
        <joint name="{name}">
            <command_interface name="position">
                <param name="min">-6.28</param><param name="max">6.28</param>
                <param name="initial_value">{v}</param>
            </command_interface>
            <state_interface name="position"><param name="initial_value">{v}</param></state_interface>
            <state_interface name="velocity"><param name="initial_value">0.0</param></state_interface>
        </joint>"""

    arm_joints = [f'{ROBOT_NAME}_joint_{i}' for i in range(1, 7)]
    finger_joints = [f'{ROBOT_NAME}_joint_finger_{i}' for i in range(1, 3)]
    
    blocks = ''.join(jblock(j) for j in arm_joints + finger_joints)
    
    insertion = f"""
  <ros2_control name="GazeboSimSystem" type="system">
    <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>{blocks}
  </ros2_control>
  <gazebo>
    <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
      <parameters>{controllers_yaml}</parameters>
      <ros><remapping>~/robot_description:=robot_description</remapping></ros>
    </plugin>
  </gazebo>
  
  <!-- Arm-mounted Camera -->
  <gazebo reference="{ROBOT_NAME}_end_effector">
    <sensor name="arm_camera" type="rgbd_camera">
      <pose>0.05 0 0.05 0 0 0</pose>
      <always_on>true</always_on>
      <update_rate>15</update_rate>
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image><width>640</width><height>480</height></image>
        <clip><near>0.1</near><far>5</far></clip>
      </camera>
      <topic>arm_camera</topic>
      <gz_frame_id>camera_color_optical_frame</gz_frame_id>
    </sensor>
  </gazebo>
"""
    p = p.replace('</robot>', insertion + '</robot>', 1)
    return p

def generate_launch_description():
    # --- Paths ---
    pkg_point_goal = get_package_share_directory(PKG_POINT_GOAL)
    pkg_description = get_package_share_directory(PKG_DESCRIPTION)
    pkg_moveit_config = get_package_share_directory(PKG_MOVEIT_CONFIG)
    
    world_file = os.path.join(pkg_point_goal, 'worlds', WORLD_FILE_NAME)
    controllers_yaml = os.path.join(pkg_point_goal, 'config', 'gz_controllers.yaml')
    
    # --- Robot Description ---
    xacro_file = os.path.join(pkg_description, 'urdf', f'{ROBOT_NAME}_standalone.xacro')
    raw_urdf = xacro.process_file(xacro_file).toprettyxml(indent='  ')
    patched_urdf = _patch_urdf(raw_urdf, controllers_yaml)
    
    robot_description = {'robot_description': patched_urdf}
    
    # --- MoveIt 2 Parameters ---
    srdf = _load_text(PKG_MOVEIT_CONFIG, 'config/j2s6s200.srdf')
    kinematics = _load_yaml(PKG_MOVEIT_CONFIG, 'config/kinematics.yaml')
    joint_limits = _load_yaml(PKG_MOVEIT_CONFIG, 'config/joint_limits.yaml')
    
    ompl_yaml = _load_yaml(PKG_MOVEIT_CONFIG, "config/ompl_planning.yaml")
    ompl_params = {"planning_plugin": "ompl_interface/OMPLPlanner"}
    ompl_params.update(ompl_yaml.get("move_group", {}))
    
    moveit_params = [
        robot_description,
        {'robot_description_semantic': srdf},
        {'robot_description_kinematics': kinematics},
        {'robot_description_planning': joint_limits},
        {'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
         'moveit_simple_controller_manager': {
             'controller_names': ['arm_controller'],
             'arm_controller': {
                 'type': 'FollowJointTrajectory',
                 'action_ns': 'arm_controller/follow_joint_trajectory',
                 'default': True,
                 'joints': [f'{ROBOT_NAME}_joint_{i}' for i in range(1, 7)],
             }
         }},
        {'planning_pipelines': ['ompl', 'pilz_industrial_motion_planner'],
         'default_planning_pipeline': 'ompl'},
        {'ompl': ompl_params},
        {'pilz_industrial_motion_planner': {
            'planning_plugin': 'pilz_industrial_motion_planner/CommandPlanner',
            'request_adapters': ' '.join([
                'default_planning_request_adapters/ResolveConstraintFrames',
                'default_planning_request_adapters/ValidateWorkspaceBounds',
                'default_planning_request_adapters/CheckStartStateBounds',
                'default_planning_request_adapters/CheckStartStateCollision',
            ]),
            'start_state_max_bounds_error': 0.1,
        }},
        SIM_TIME
    ]

    # Define resource paths for Gazebo
    pkg_share_path = os.path.dirname(get_package_share_directory(PKG_DESCRIPTION))
    
    # 1. Gazebo
    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', world_file],
        additional_env={
            'IGN_GAZEBO_SYSTEM_PLUGIN_PATH':
                '/opt/ros/humble/lib:' + os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', ''),
            'IGN_GAZEBO_RESOURCE_PATH':
                pkg_share_path + ':' + '/opt/ros/humble/share:' + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', ''),
            'GZ_SIM_RESOURCE_PATH':
                pkg_share_path + ':' + '/opt/ros/humble/share:' + os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
        },
        output='screen',
    )
    
    # 2. Robot State Publisher
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, SIM_TIME]
    )
    
    # 3. Spawn Robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-name', ROBOT_NAME,
            '-topic', 'robot_description',
            '-world', WORLD_FILE_NAME.split('.')[0], # point_goal_scene
            '-x', SPAWN_X,
            '-y', SPAWN_Y,
            '-z', SPAWN_Z,
        ],
        parameters=[SIM_TIME]
    )
    
    # 4. Bridge Gazebo <-> ROS 2
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/arm_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/arm_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/arm_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        remappings=[
            ('/arm_camera/image', '/camera/camera/color/image_raw'),
            ('/arm_camera/depth_image', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('/arm_camera/camera_info', '/camera/camera/color/camera_info'),
        ],
        parameters=[SIM_TIME]
    )
    
    # 5. Controller Spawners
    def spawner(name):
        return Node(
            package='controller_manager',
            executable='spawner',
            arguments=[name, '--controller-manager', '/controller_manager'],
            parameters=[SIM_TIME],
            output='screen'
        )

    load_controllers = TimerAction(
        period=5.0,
        actions=[
            spawner('joint_state_broadcaster'),
            spawner('arm_controller'),
        ]
    )
    
    # 6. Static TF for camera (end_effector -> camera_color_optical_frame)
    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=['0.05', '0', '0.05', '0', '0', '0', f'{ROBOT_NAME}_end_effector', 'camera_color_optical_frame'],
        parameters=[SIM_TIME]
    )
    
    # 7. World to Root TF
    world_to_root = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_root',
        arguments=['--x', SPAWN_X, '--y', SPAWN_Y, '--z', SPAWN_Z,
                   '--frame-id', 'world', '--child-frame-id', 'root'],
        parameters=[SIM_TIME]
    )

    # 8. MoveIt 2 Move Group
    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=moveit_params
    )
    
    # 9. RViz 2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[robot_description, SIM_TIME]
    )

    # 10. Point Selector & Arm Mover
    point_selector = Node(
        package=PKG_POINT_GOAL,
        executable='point_selector_node',
        output='screen',
        parameters=[SIM_TIME]
    )
    
    arm_mover = Node(
        package=PKG_POINT_GOAL,
        executable='arm_mover_node',
        output='screen',
        parameters=[SIM_TIME]
    )

    return LaunchDescription([
        gazebo,
        rsp,
        spawn_robot,
        bridge,
        load_controllers,
        camera_tf,
        world_to_root,
        TimerAction(period=20.0, actions=[move_group, rviz]),
        TimerAction(period=25.0, actions=[point_selector, arm_mover]),
    ])
