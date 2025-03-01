#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Obter o diretório do pacote
    pkg_dir = get_package_share_directory('roboime_vision')
    
    # Arquivo de configuração
    config_file = os.path.join(pkg_dir, 'config', 'vision_params.yaml')
    
    # Argumentos de lançamento
    use_ball_detector = LaunchConfiguration('use_ball_detector', default='true')
    use_field_detector = LaunchConfiguration('use_field_detector', default='true')
    use_line_detector = LaunchConfiguration('use_line_detector', default='false')
    use_goal_detector = LaunchConfiguration('use_goal_detector', default='false')
    use_obstacle_detector = LaunchConfiguration('use_obstacle_detector', default='false')
    use_yoeo_detector = LaunchConfiguration('use_yoeo_detector', default='false')
    
    # Declarar argumentos
    declare_use_ball_detector = DeclareLaunchArgument(
        'use_ball_detector',
        default_value='true',
        description='Habilitar detector de bola'
    )
    
    declare_use_field_detector = DeclareLaunchArgument(
        'use_field_detector',
        default_value='true',
        description='Habilitar detector de campo'
    )
    
    declare_use_line_detector = DeclareLaunchArgument(
        'use_line_detector',
        default_value='false',
        description='Habilitar detector de linhas'
    )
    
    declare_use_goal_detector = DeclareLaunchArgument(
        'use_goal_detector',
        default_value='false',
        description='Habilitar detector de gols'
    )
    
    declare_use_obstacle_detector = DeclareLaunchArgument(
        'use_obstacle_detector',
        default_value='false',
        description='Habilitar detector de obstáculos'
    )
    
    declare_use_yoeo_detector = DeclareLaunchArgument(
        'use_yoeo_detector',
        default_value='false',
        description='Habilitar detector YOEO'
    )
    
    # Nós a serem lançados
    nodes = []
    
    # Pipeline principal de visão
    vision_pipeline_node = Node(
        package='roboime_vision',
        executable='vision_pipeline.py',
        name='vision_pipeline',
        parameters=[config_file],
        output='screen'
    )
    nodes.append(vision_pipeline_node)
    
    # Detector de bola (condicional)
    ball_detector_node = Node(
        package='roboime_vision',
        executable='ball_detector.py',
        name='ball_detector',
        parameters=[config_file],
        output='screen',
        condition=LaunchConfiguration('use_ball_detector')
    )
    nodes.append(ball_detector_node)
    
    # Detector de campo (condicional)
    field_detector_node = Node(
        package='roboime_vision',
        executable='field_detector.py',
        name='field_detector',
        parameters=[config_file],
        output='screen',
        condition=LaunchConfiguration('use_field_detector')
    )
    nodes.append(field_detector_node)
    
    # Detector de linhas (condicional)
    line_detector_node = Node(
        package='roboime_vision',
        executable='line_detector.py',
        name='line_detector',
        parameters=[config_file],
        output='screen',
        condition=LaunchConfiguration('use_line_detector')
    )
    nodes.append(line_detector_node)
    
    # Detector de gols (condicional)
    goal_detector_node = Node(
        package='roboime_vision',
        executable='goal_detector.py',
        name='goal_detector',
        parameters=[config_file],
        output='screen',
        condition=LaunchConfiguration('use_goal_detector')
    )
    nodes.append(goal_detector_node)
    
    # Detector de obstáculos (condicional)
    obstacle_detector_node = Node(
        package='roboime_vision',
        executable='obstacle_detector.py',
        name='obstacle_detector',
        parameters=[config_file],
        output='screen',
        condition=LaunchConfiguration('use_obstacle_detector')
    )
    nodes.append(obstacle_detector_node)
    
    # Detector YOEO (condicional)
    yoeo_detector_node = Node(
        package='roboime_vision',
        executable='yoeo_detector',
        name='yoeo_detector',
        parameters=[config_file],
        output='screen',
        condition=LaunchConfiguration('use_yoeo_detector')
    )
    nodes.append(yoeo_detector_node)
    
    # Criar e retornar a descrição de lançamento
    return LaunchDescription([
        declare_use_ball_detector,
        declare_use_field_detector,
        declare_use_line_detector,
        declare_use_goal_detector,
        declare_use_obstacle_detector,
        declare_use_yoeo_detector,
        *nodes
    ]) 