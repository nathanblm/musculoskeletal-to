from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        DeclareLaunchArgument('image_width', default_value='1280'),
        DeclareLaunchArgument('image_height', default_value='720'),
        DeclareLaunchArgument('framerate', default_value='30.0'),
        DeclareLaunchArgument('pixel_format', default_value='mjpeg2rgb'),
    ]

    camera = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='webcam',
        namespace='camera',
        output='screen',
        parameters=[{
            'video_device': ParameterValue(
                LaunchConfiguration('video_device'), value_type=str),
            'image_width': ParameterValue(
                LaunchConfiguration('image_width'), value_type=int),
            'image_height': ParameterValue(
                LaunchConfiguration('image_height'), value_type=int),
            'framerate': ParameterValue(
                LaunchConfiguration('framerate'), value_type=float),
            'pixel_format': ParameterValue(
                LaunchConfiguration('pixel_format'), value_type=str),
            'io_method': 'mmap',
            'camera_name': 'webcam',
            'frame_id': 'camera',
        }],
    )

    return LaunchDescription(arguments + [camera])
