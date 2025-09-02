#!/usr/bin/env python3
"""
Launch file principal do RoboIME HSL2025
Sistema completo de robô humanóide para futebol

Autor: RoboIME Team
Versão: 2.0.0 (YOLOv8 Unificado)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """Gera a descrição de inicialização para o robô de futebol RoboIME HSL2025."""
    
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
    
    camera_type_arg = DeclareLaunchArgument(
        'camera_type',
        default_value='csi',
        description='Tipo de câmera (csi para IMX219, usb para C930)'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='true',
        description='Habilitar modo debug com visualizações'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='resources/models/robocup_yolov8.pt',
        description='Caminho para modelo YOLOv8 customizado'
    )
    
    # Obter valores dos argumentos
    team_color = LaunchConfiguration('team_color')
    player_number = LaunchConfiguration('player_number')
    role = LaunchConfiguration('role')
    camera_type = LaunchConfiguration('camera_type')
    debug = LaunchConfiguration('debug')
    model_path = LaunchConfiguration('model_path')
    
    # 👁️ Sistema de Percepção (YOLOv8 Unificado)
    perception_launch = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('perception'),
            'launch',
            'perception.launch.py'
        ]),
        launch_arguments={
            'camera_type': camera_type,
            'debug': debug,
            'model_path': model_path,
            'confidence_threshold': '0.6',
            'device': 'cuda'
        }.items()
    )
    
    # 🤖 Nó de Comportamento (Soccer Behavior)
    behavior_node = Node(
        package='behavior',
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
        }],
        remappings=[
            ('/ball_detection', '/perception/ball_detection'),
            ('/robot_detections', '/perception/robot_detections'),
            ('/goal_detections', '/perception/goal_detections'),
            ('/field_detection', '/perception/field_detection'),
        ]
    )
    
    # 🚶 Nó de Movimento (Walking Controller)
    motion_node = Node(
        package='motion',
        executable='walking_controller',
        name='walking_controller',
        output='screen',
        parameters=[{
            'step_height': 0.04,
            'step_length': 0.06,
            'lateral_width': 0.05,
            'walk_frequency': 1.0,
            'max_linear_velocity': 0.2,
            'max_angular_velocity': 0.5,
            'balance_enable': True,
            'fall_protection': True
        }]
    )
    
    # 🧭 Sistema de Navegação (C++) + Planejamento (Python)
    navigation_launch = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('roboime_navigation'),
            'launch',
            'navigation.launch.py'
        ]),
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
    
    # Log de inicialização
    startup_log = LogInfo(
        msg=[
            "🤖 RoboIME HSL2025 - Sistema Iniciado\n",
            "🧠 Percepção: YOLOv8 Simplificado (7 classes otimizadas)\n",
            "📷 Câmera: ", camera_type, "\n",
            "🎯 Time: ", team_color, " | Jogador: ", player_number, " | Função: ", role, "\n",
            "🧠 Modelo: ", model_path, "\n",
            "🐛 Debug: ", debug
        ]
    )
    
    # Criar a descrição de inicialização
    return LaunchDescription([
        # Argumentos de launch
        team_color_arg,
        player_number_arg,
        role_arg,
        camera_type_arg,
        debug_arg,
        model_path_arg,
        
        # Log inicial
        startup_log,
        
        # Sistemas principais
        perception_launch,      # Sistema de percepção YOLOv8 unificado
        navigation_launch,      # Sistema de navegação completo
        
        # Nós individuais
        behavior_node,         # Comportamento de futebol
        motion_node,           # Controlador de movimento
    ]) 