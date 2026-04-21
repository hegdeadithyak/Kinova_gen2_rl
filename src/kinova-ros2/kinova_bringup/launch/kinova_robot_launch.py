import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
import xacro
import yaml

configurable_parameters = [
    {'name': 'use_urdf',              'default': "true"},
    {'name': 'kinova_robotType',      'default': "j2s6s200"},
    {'name': 'kinova_robotName',      'default': "left"},
    {'name': 'kinova_robotSerial',    'default': "not_set"},
    {'name': 'use_jaco_v1_fingers',   'default': "true"},
    {'name': 'feedback_publish_rate', 'default': "0.1"},
    {'name': 'tolerance',             'default': "2.0"},
]

def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(p['name'], default_value=p['default']) for p in parameters]

def set_configurable_parameters(parameters):
    return {p['name']: LaunchConfiguration(p['name']) for p in parameters}

def yaml_to_dict(path_to_yaml):
    with open(path_to_yaml) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def launch_setup(context, *args, **kwargs):
    _config_file = os.path.join(
        get_package_share_directory('kinova_bringup'),
        'launch/config', 'robot_parameters.yaml'
    )
    params_from_file = yaml_to_dict(_config_file)
    robot_type = LaunchConfiguration("kinova_robotType").perform(context)

    kinova_driver = Node(
        package='kinova_driver',
        name=robot_type + '_driver',
        executable='kinova_arm_driver',
        parameters=[set_configurable_parameters(configurable_parameters), params_from_file],
        output='screen',
    )

    kinova_tf_updater = Node(
        package='kinova_driver',
        name=robot_type + '_tf_updater',
        executable='kinova_tf_updater',
        parameters=[{'base_frame': 'root'},
                    set_configurable_parameters(configurable_parameters),
                    params_from_file],
        remappings=[(robot_type + '_tf_updater/in/joint_angles',
                     robot_type + '_driver/out/joint_angles')],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration("use_urdf")),
    )

    xacro_file = os.path.join(
        get_package_share_directory('kinova_description'),
        'urdf', robot_type + '_standalone.xacro'
    )
    robot_desc = xacro.process_file(xacro_file).toprettyxml(indent='  ')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        remappings=[('joint_states', robot_type + '_driver/out/joint_state')],
        output='screen',
        parameters=[{'robot_description': robot_desc}],
        condition=IfCondition(LaunchConfiguration("use_urdf")),
    )

    # ── Trajectory bridge — converts action goals → joint velocity commands ──
    trajectory_bridge = Node(
        package='kinova_bringup',          # change to wherever you install the script
        executable='kinova_trajectory_bridge',
        name='kinova_trajectory_bridge',
        output='screen',
    )

    return [kinova_driver, kinova_tf_updater, robot_state_publisher, trajectory_bridge]

def generate_launch_description():
    return LaunchDescription(
        declare_configurable_parameters(configurable_parameters) +
        [OpaqueFunction(function=launch_setup)]
    )