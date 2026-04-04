#!/usr/bin/env python3
"""
kinova_launch.py  — Kinova j2n6s200 + Ignition Gazebo (Fortress) + MoveIt2

PLANNING PIPELINES
──────────────────
  ompl     → default, all motion planning (OMPL)
  elliptic → EllipticArc delivery planner (food-to-mouth arc)

In RViz: MotionPlanning → Planning tab → Pipeline dropdown → select pipeline
In Python:
  move_group.set_planning_pipeline_id("elliptic")
  move_group.set_planner_id("EllipticArc")
"""

import os, re, pathlib, yaml
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, LogInfo
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, get_packages_with_prefixes
import xacro

PACKAGE_NAME          = 'kinova_bringup'
ROBOT_NAME            = 'j2n6s200'
WORLD_FILE            = os.path.expanduser(
    '~/kinova_ws/src/kinova-ros2/kinova_bringup/worlds/feeding_scene.sdf')
ROS2_CONTROLLERS_YAML = os.path.expanduser(
    '~/kinova_ws/src/kinova-ros2/kinova_bringup/moveit_resource/ros2_controllers.yaml')
CONTROLLER_MANAGER_NS = '/controller_manager'
SIM_TIME = {'use_sim_time': True}

ARM_SPAWN_X =  '0.40'
ARM_SPAWN_Y = '-0.50'
ARM_SPAWN_Z =  '0.85'


def _patch_urdf(raw_xml):
    p = re.sub(r'<gazebo[^>]*>.*?</gazebo>', '', raw_xml,
               flags=re.DOTALL | re.IGNORECASE)
    p = re.sub(r'<mass\s+value="[^"]*"\s*/>', '<mass value="0.01"/>', p)
    for attr in ['ixx','iyy','izz']:
        p = re.sub(rf'\b{attr}="[^"]*"', f'{attr}="0.0001"', p)
    for attr in ['ixy','ixz','iyz']:
        p = re.sub(rf'\b{attr}="[^"]*"', f'{attr}="0.0"', p)

    arm_j    = [f'{ROBOT_NAME}_joint_{i}' for i in range(1, 7)]
    finger_j = [f'{ROBOT_NAME}_joint_finger_{i}' for i in range(1, 4)]
    INIT = {f'{ROBOT_NAME}_joint_1': 4.71, f'{ROBOT_NAME}_joint_2': 2.71,
            f'{ROBOT_NAME}_joint_3': 1.57, f'{ROBOT_NAME}_joint_4': 4.71,
            f'{ROBOT_NAME}_joint_5': 0.0,  f'{ROBOT_NAME}_joint_6': 3.14,
            f'{ROBOT_NAME}_joint_finger_1': 0.0,
            f'{ROBOT_NAME}_joint_finger_2': 0.0,
            f'{ROBOT_NAME}_joint_finger_3': 0.0}

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

    blocks = ''.join(jblock(j) for j in arm_j + finger_j)
    insertion = f"""
        <ros2_control name="GazeboSystem" type="system">
            <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>{blocks}
        </ros2_control>
        <gazebo>
            <plugin filename="gz_ros2_control-system"
                    name="gz_ros2_control::GazeboSimROS2ControlPlugin">
            <parameters>{ROS2_CONTROLLERS_YAML}</parameters>
            <ros><remapping>~/robot_description:=robot_description</remapping></ros>
            </plugin>
        </gazebo>
        """
    p = p.replace('</robot>', insertion + '</robot>', 1)
    with open('/tmp/patched_robot.urdf', 'w') as f:
        f.write(p)
    print('[DEBUG] Patched URDF → /tmp/patched_robot.urdf')
    return p


def generate_launch_description():
    pkg_dir    = get_package_share_directory(PACKAGE_NAME)
    read_file  = lambda n: pathlib.Path(pkg_dir, 'moveit_resource', n).read_text()
    read_yaml  = lambda n: yaml.safe_load(read_file(n))
    moveit_res = lambda n: str(pathlib.Path(pkg_dir, 'moveit_resource', n))

    xacro_file = os.path.join(get_package_share_directory('kinova_description'),
                               'urdf', f'{ROBOT_NAME}_standalone.xacro')
    robot_desc = _patch_urdf(xacro.process_file(xacro_file).toprettyxml(indent='  '))

    description            = {'robot_description': robot_desc}
    description_semantic   = {'robot_description_semantic':   read_file(f'{ROBOT_NAME}.srdf')}
    description_kinematics = {'robot_description_kinematics': read_yaml('kinematics.yaml')}
    description_planning   = {'robot_description_planning':   read_yaml('joint_limits.yaml')}

    mesh_path = os.path.expanduser('~/kinova_ws/src/kinova-ros2')

    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', WORLD_FILE],
        additional_env={
            'IGN_GAZEBO_SYSTEM_PLUGIN_PATH':
                '/opt/ros/humble/lib:' + os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', ''),
            'IGN_GAZEBO_RESOURCE_PATH':
                mesh_path + ':' + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', ''),
        },
        output='screen',
    )

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               output='screen', parameters=[description, SIM_TIME])

    spawn_robot = Node(
        package='ros_gz_sim', executable='create',
        name=f'spawn_{ROBOT_NAME}', output='screen',
        parameters=[SIM_TIME],
        arguments=[
            '-name',  ROBOT_NAME,
            '-topic', 'robot_description',
            '-x', ARM_SPAWN_X,
            '-y', ARM_SPAWN_Y,
            '-z', ARM_SPAWN_Z,
        ],
    )

    gz_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='gz_ros_bridge', output='screen',
        parameters=[SIM_TIME],
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/camera_mouth/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_mouth/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_mouth/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/camera_mouth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        remappings=[
            ('/camera_mouth/image',       '/camera_mouth/color/image_raw'),
            ('/camera_mouth/depth_image', '/camera_mouth/depth/image_rect_raw'),
        ],
    )

    camera_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='camera_mouth_tf', output='screen',
        parameters=[SIM_TIME],
        arguments=['0.15', '0', '1.30', '0', '0.409', '0',
                   'world', 'camera_mouth_optical_frame'],
    )

    depth_proc = Node(
        package='depth_image_proc', executable='point_cloud_xyz_node',
        name='camera_mouth_pointcloud', output='screen',
        parameters=[SIM_TIME],
        remappings=[
            ('image_rect',  '/camera_mouth/depth/image_rect_raw'),
            ('camera_info', '/camera_mouth/camera_info'),
            ('points',      '/camera_mouth/points_filtered'),
        ],
    )

    finger_tip_pub = Node(
        package='joint_state_publisher', executable='joint_state_publisher',
        name='finger_tip_state_publisher', output='screen',
        parameters=[SIM_TIME, {
            'rate': 50,
            'j2n6s200_joint_finger_tip_1': 0.0,
            'j2n6s200_joint_finger_tip_2': 0.0,
            'j2n6s200_joint_finger_tip_3': 0.0,
        }],
    )

    def _spawner(name):
        args = [name, '--controller-manager', CONTROLLER_MANAGER_NS,
                '--controller-manager-timeout', '60']
        if 'trajectory' in name:
            args.append('--activate')
        return Node(package='controller_manager', executable='spawner',
                    output='screen', parameters=[SIM_TIME], arguments=args)

    controller_spawners = TimerAction(period=10.0, actions=[
        _spawner('joint_state_broadcaster'),
        _spawner('joint_trajectory_controller'),
        _spawner('finger_trajectory_controller'),
    ])

    nodes = [gazebo, rsp, spawn_robot, gz_bridge,
             camera_tf, depth_proc, finger_tip_pub, controller_spawners]

    # ── MoveIt2 ───────────────────────────────────────────────────────────────
    if 'moveit' in get_packages_with_prefixes():

        # ── Pipeline 1: OMPL (default, approach + general motion) ────────────
        ompl_pipeline = {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': (
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/FixStartStatePathConstraints'),
            'start_state_max_bounds_error': 0.1,
        }
        ompl_pipeline.update(read_yaml('ompl_planning.yaml'))

        # ── Pipeline 2: EllipticArc (food-to-mouth delivery) ─────────────────
        elliptic_pipeline = {
            'planning_plugin': 'elliptic_planner/EllipticPlannerManager',
            'request_adapters': (
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds'),
            'lift_height': 0.10,   # arc apex height above midpoint (m)
            'n_waypoints': 40,     # IK waypoints along arc
            'total_time':  4.0,    # delivery duration (s)
        }

        # ── move_group parameters ─────────────────────────────────────────────
        # planning_pipelines declares both pipelines.
        # default_planning_pipeline sets which one RViz opens with.
        # Each pipeline's config is nested under its name key.
        move_group_params = [
            description,
            SIM_TIME,
            description_semantic,
            description_kinematics,
            description_planning,
            {
                # Declare both pipelines
                'planning_pipelines':        ['ompl', 'elliptic'],
                'default_planning_pipeline': 'ompl',

                # OMPL pipeline config
                'ompl': ompl_pipeline,

                # Elliptic pipeline config
                'elliptic': elliptic_pipeline,
            },
            {
                'moveit_simple_controller_manager': read_yaml('controllers.yaml'),
                'moveit_controller_manager':
                    'moveit_simple_controller_manager/MoveItSimpleControllerManager',
            },
            {
                'current_state_monitor_wait_time': 10.0,
                'trajectory_execution.execution_duration_monitoring': False,
                'trajectory_execution.allowed_execution_duration_scaling': 2.0,
                'trajectory_execution.allowed_goal_duration_margin': 1.0,
                'trajectory_execution.allowed_start_tolerance': 0.05,
                'publish_robot_description': True,
                'publish_robot_description_semantic': True,
                'planning_scene_monitor_options': {
                    'publish_planning_scene': True,
                    'publish_geometry_updates': True,
                    'publish_state_updates': True,
                    'publish_transforms_updates': True,
                },
                'plan_with_sensing': False,
            },
        ]

        move_group = Node(
            package='moveit_ros_move_group', executable='move_group',
            output='screen',
            parameters=move_group_params,
        )

        rviz = Node(
            package='rviz2', executable='rviz2', name='rviz2', output='screen',
            arguments=['-d', moveit_res('visualization.rviz')],
            parameters=[description, SIM_TIME, description_semantic,
                        description_kinematics, description_planning],
        )

        nodes.append(TimerAction(period=35.0, actions=[move_group, rviz]))

    else:
        nodes.append(LogInfo(msg='moveit not found — skipping move_group + rviz2.'))

    return LaunchDescription(nodes)