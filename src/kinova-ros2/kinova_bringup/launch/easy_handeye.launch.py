from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    marker_id_arg = DeclareLaunchArgument('marker_id', default_value='0')
    marker_size_arg = DeclareLaunchArgument('marker_size', default_value='0.0594')

    # world -> root static TF (required for robot TF tree to connect)
    world_to_root = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_root',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'world', '--child-frame-id', 'root'],
    )

    # RealSense camera driver with TF publishing enabled
    try:
        rs_share = get_package_share_directory('realsense2_camera')
        camera_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([rs_share, '/launch/rs_launch.py']),
            launch_arguments={
                'camera_name': 'camera',
                'enable_color': 'true',
                'enable_depth': 'false',
                'publish_tf': 'true',
                'tf_publish_rate': '0.0',
                'rgb_camera.color_profile': '640x480x30',
            }.items(),
        )
    except Exception:
        camera_launch = LogInfo(msg='WARNING: realsense2_camera not found — start camera driver manually with publish_tf:=true')

    # ArUco detector using cv2.aruco DICT_5X5_250 — publishes TF: camera_color_optical_frame -> aruco_marker_frame
    aruco_node = Node(
        package='kinova_bringup',
        executable='aruco_tf_publisher.py',
        name='aruco_tf_publisher',
        parameters=[{
            'marker_id': LaunchConfiguration('marker_id'),
            'marker_size': LaunchConfiguration('marker_size'),
            'camera_frame': 'camera_color_optical_frame',
            'marker_frame': 'aruco_marker_frame',
        }],
    )

    # easy_handeye2 server — collects samples and computes calibration
    handeye_server = Node(
        package='easy_handeye2',
        executable='handeye_server',
        name='handeye_server',
        parameters=[{
            'name': 'kinova_eye_in_hand',
            'calibration_type': 'eye_in_hand',
            'tracking_base_frame': 'camera_color_optical_frame',
            'tracking_marker_frame': 'aruco_marker_frame',
            'robot_base_frame': 'root',
            'robot_effector_frame': 'j2s6s200_end_effector',
        }],
    )

    # rqt GUI for taking samples / computing / saving
    handeye_gui = Node(
        package='easy_handeye2',
        executable='rqt_calibrator.py',
        name='handeye_rqt_calibrator',
        parameters=[{
            'name': 'kinova_eye_in_hand',
            'calibration_type': 'eye_in_hand',
            'tracking_base_frame': 'camera_color_optical_frame',
            'tracking_marker_frame': 'aruco_marker_frame',
            'robot_base_frame': 'root',
            'robot_effector_frame': 'j2s6s200_end_effector',
        }],
    )

    return LaunchDescription([
        marker_id_arg,
        marker_size_arg,
        world_to_root,
        camera_launch,
        aruco_node,
        handeye_server,
        handeye_gui,
    ])
