#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.actions import LogInfo
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    team_color_arg = DeclareLaunchArgument('team_color', default_value='blue')
    player_number_arg = DeclareLaunchArgument('player_number', default_value='1')
    role_arg = DeclareLaunchArgument('role', default_value='striker')
    camera_type_arg = DeclareLaunchArgument('camera_type', default_value='usb')
    debug_arg = DeclareLaunchArgument('debug', default_value='true')
    model_path_arg = DeclareLaunchArgument('model_path', default_value='/ros2_ws/src/perception/resources/models/robocup_yolov8.engine')

    team_color = LaunchConfiguration('team_color')
    player_number = LaunchConfiguration('player_number')
    role = LaunchConfiguration('role')
    camera_type = LaunchConfiguration('camera_type')
    debug = LaunchConfiguration('debug')
    model_path = LaunchConfiguration('model_path')

    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('perception'), 'launch', 'perception.launch.py'])
        ),
        launch_arguments={
            'camera_type': camera_type,
            'debug': debug,
            'model_path': model_path,
            'confidence_threshold': '0.6',
            'device': 'cuda'
        }.items()
    )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('roboime_navigation'), 'launch', 'navigation.launch.py'])
        ),
        launch_arguments={
            'field_length': '9.0',
            'field_width': '6.0',
            'robot_id': player_number,
            'team_side': 'left',
            'use_global_localization': 'true',
            'enable_team_communication': 'false',
            'debug_mode': debug
        }.items()
    )

    behavior_node = Node(
        package='roboime_behavior',
        executable='behavior_node',
        name='soccer_behavior',
        output='screen',
        parameters=[{
            'team_color': team_color,
            'player_number': player_number,
            'role': role,
            'decision_frequency': 10.0,
            'max_ball_distance': 3.0,
            'goal_approach_distance': 1.5
        }]
    )

    motion_node = Node(
        package='motion',
        executable='walking_controller.py',
        name='walking_controller',
        output='screen'
    )

    startup_log = LogInfo(msg=['RoboIME Bringup iniciado'])

    return LaunchDescription([
        team_color_arg,
        player_number_arg,
        role_arg,
        camera_type_arg,
        debug_arg,
        model_path_arg,
        startup_log,
        perception_launch,
        navigation_launch,
        behavior_node,
        motion_node,
    ])


