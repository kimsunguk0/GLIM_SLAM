from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    map_path = LaunchConfiguration("map_path")
    points_topic = LaunchConfiguration("points_topic")
    fix_topic = LaunchConfiguration("fix_topic")
    ekf_config = LaunchConfiguration("ekf_config")
    output_frame = LaunchConfiguration("output_frame")

    return LaunchDescription([
        DeclareLaunchArgument("map_path", description="PCD map path"),
        DeclareLaunchArgument("points_topic", default_value="/front/lidar_point"),
        DeclareLaunchArgument("fix_topic", default_value="/fix"),
        DeclareLaunchArgument("output_frame", default_value="utm_enu"),
        DeclareLaunchArgument(
            "ekf_config",
            default_value=PathJoinSubstitution([FindPackageShare("glim_ros"), "config", "ekf_vgicp_gnss.yaml"]),
        ),
        DeclareLaunchArgument("initial_x", default_value="0.0"),
        DeclareLaunchArgument("initial_y", default_value="0.0"),
        DeclareLaunchArgument("initial_z", default_value="0.0"),
        DeclareLaunchArgument("initial_yaw_deg", default_value="0.0"),
        DeclareLaunchArgument("map_to_output_x", default_value="0.0"),
        DeclareLaunchArgument("map_to_output_y", default_value="0.0"),
        DeclareLaunchArgument("map_to_output_z", default_value="0.0"),
        DeclareLaunchArgument("map_to_output_yaw_deg", default_value="0.0"),
        Node(
            package="glim_ros",
            executable="vgicp_map_localizer_node",
            name="vgicp_map_localizer",
            output="screen",
            parameters=[{
                "map_path": map_path,
                "points_topic": points_topic,
                "odom_topic": "/vgicp/odom_enu",
                "output_frame_id": output_frame,
                "child_frame_id": "base_link",
                "require_initial_pose": False,
                "initial_x": ParameterValue(LaunchConfiguration("initial_x"), value_type=float),
                "initial_y": ParameterValue(LaunchConfiguration("initial_y"), value_type=float),
                "initial_z": ParameterValue(LaunchConfiguration("initial_z"), value_type=float),
                "initial_yaw_deg": ParameterValue(LaunchConfiguration("initial_yaw_deg"), value_type=float),
                "map_to_output_x": ParameterValue(LaunchConfiguration("map_to_output_x"), value_type=float),
                "map_to_output_y": ParameterValue(LaunchConfiguration("map_to_output_y"), value_type=float),
                "map_to_output_z": ParameterValue(LaunchConfiguration("map_to_output_z"), value_type=float),
                "map_to_output_yaw_deg": ParameterValue(LaunchConfiguration("map_to_output_yaw_deg"), value_type=float),
            }],
        ),
        Node(
            package="glim_ros",
            executable="navsatfix_to_odom.py",
            name="navsatfix_to_odom",
            output="screen",
            parameters=[{
                "fix_topic": fix_topic,
                "odom_topic": "/gnss/odom_enu",
                "frame_id": output_frame,
                "child_frame_id": "gnss",
                "min_status": 0,
                "use_z": False,
            }],
        ),
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_filter_node",
            output="screen",
            parameters=[ekf_config],
            remappings=[("odometry/filtered", "/ekf/odom_enu")],
        ),
    ])
