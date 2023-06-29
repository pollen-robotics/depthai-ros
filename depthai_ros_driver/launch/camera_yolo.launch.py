import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.conditions import IfCondition


def launch_setup(context, *args, **kwargs):
    name = LaunchConfiguration('name').perform(context)
    depthai_prefix = get_package_share_directory("depthai_ros_driver")
    # rviz_config = os.path.join(depthai_prefix, "config", "rviz", "rgbd.rviz")

    params_file= LaunchConfiguration("params_file")

    return [
        # Node(
        #         condition=IfCondition(LaunchConfiguration("use_rviz")),
        #         package="rviz2",
        #         executable="rviz2",
        #         name="rviz2",
        #         output="log",
        #         arguments=["-d", rviz_config],
        #     ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(depthai_prefix, 'launch', 'camera.launch.py')),
            launch_arguments={"name": name,
                              "params_file": params_file}.items()),
        Node(
            package="depthai_ros_driver",
            executable="obj_pub.py",
        ),
        ComposableNodeContainer(
            name=name+"_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                    ComposableNode(
                        package="depth_image_proc",
                        plugin="depth_image_proc::ConvertMetricNode",
                        name="convert_metric_node",
                        remappings=[('image_raw', name+'/stereo/image_raw'),
                                            ('camera_info', name+'/stereo/camera_info'),
                                            ('image', name+'/stereo/converted_depth')]
                    ),
                    ComposableNode(
                        package='depth_image_proc',
                        plugin='depth_image_proc::PointCloudXyzrgbNode',
                        name='point_cloud_xyzrgb_node',
                        remappings=[('depth_registered/image_rect', name+'/stereo/converted_depth'),
                                    ('rgb/image_rect_color', name+'/rgb/image_raw'),
                                    ('rgb/camera_info', name+'/rgb/camera_info'),
                                    ('points', name+'/points')],
                    ),
                    # ComposableNode(
                    #     package="depthai_ros_driver",
                    #     plugin="depthai_ros_driver::Camera",
                    #     name=name,
                    #     parameters=[params_file],
                    # ),
            ],
            output="screen",
        ),
    ]


def generate_launch_description():
    depthai_prefix = get_package_share_directory("depthai_ros_driver")
    declared_arguments = [
        DeclareLaunchArgument("name", default_value="oak"),
        DeclareLaunchArgument("params_file", default_value=os.path.join(depthai_prefix, 'config', 'camera_yolo.yaml')),
        # DeclareLaunchArgument("use_rviz", default_value="True"),

    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
