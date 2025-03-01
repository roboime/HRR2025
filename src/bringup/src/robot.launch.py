#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """Gera a descrição de inicialização para o robô de futebol completo."""
    
    # Argumentos de inicialização
    team_color_arg = DeclareLaunchArgument(
        'team_color',
        default_value='blue',
        description='Cor da equipe (blue ou red)'
    )
    
    player_number_arg = DeclareLaunchArgument(
        'player_number',
        default_value='1',
        description='Número do jogador (1-11)'
    )
    
    role_arg = DeclareLaunchArgument(
        'role',
        default_value='striker',
        description='Função do robô (striker, defender, goalkeeper)'
    )
    
    # Configurações
    team_color = LaunchConfiguration('team_color')
    player_number = LaunchConfiguration('player_number')
    role = LaunchConfiguration('role')
    
    # Nó de percepção (detector de bola)
    ball_detector_node = Node(
        package='roboime_perception',
        executable='ball_detector',
        name='ball_detector',
        parameters=[{
            'ball_color_lower': [0, 120, 70],  # HSV para laranja (bola)
            'ball_color_upper': [10, 255, 255],
            'min_ball_radius': 10,
            'debug_image': True
        }],
        remappings=[
            ('camera/image_raw', '/camera/image_raw'),
            ('camera/camera_info', '/camera/camera_info')
        ]
    )
    
    # Nó de movimento (controlador de caminhada)
    walking_controller_node = Node(
        package='roboime_motion',
        executable='walking_controller',
        name='walking_controller',
        parameters=[{
            'step_height': 0.04,
            'step_length': 0.06,
            'lateral_width': 0.05,
            'walk_frequency': 1.0,
            'max_linear_velocity': 0.2,
            'max_angular_velocity': 0.5
        }]
    )
    
    # Nó de comportamento (comportamento de futebol)
    soccer_behavior_node = Node(
        package='roboime_behavior',
        executable='soccer_behavior',
        name='soccer_behavior',
        parameters=[{
            'team_color': team_color,
            'player_number': player_number,
            'role': role
        }]
    )
    
    # Nó de navegação (localização)
    localization_node = Node(
        package='roboime_navigation',
        executable='localization',
        name='localization',
        parameters=[{
            'field_length': 9.0,
            'field_width': 6.0,
            'use_field_lines': True
        }]
    )
    
    # Criar a descrição de inicialização
    return LaunchDescription([
        # Argumentos
        team_color_arg,
        player_number_arg,
        role_arg,
        
        # Nós
        ball_detector_node,
        walking_controller_node,
        soccer_behavior_node,
        localization_node
    ]) 