#!/usr/bin/env python3
"""
Launch file para o sistema de navegação RoboIME HSL2025
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Argumentos de lançamento
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(
            get_package_share_directory('roboime_navigation'),
            'config', 'navigation_config.yaml'
        ),
        description='Arquivo de configuração do sistema de navegação'
    )
    
    field_length_arg = DeclareLaunchArgument(
        'field_length', default_value='9.0',
        description='Comprimento do campo (metros)'
    )
    
    field_width_arg = DeclareLaunchArgument(
        'field_width', default_value='6.0',
        description='Largura do campo (metros)'
    )
    
    robot_id_arg = DeclareLaunchArgument(
        'robot_id', default_value='1',
        description='ID único do robô (1-6)'
    )
    
    team_side_arg = DeclareLaunchArgument(
        'team_side', default_value='left',
        description='Lado do time (left/right)'
    )
    
    enable_safety_arg = DeclareLaunchArgument(
        'enable_safety', default_value='true',
        description='Habilitar verificações de segurança'
    )
    
    min_localization_confidence_arg = DeclareLaunchArgument(
        'min_localization_confidence', default_value='0.3',
        description='Confiança mínima para navegação'
    )
    
    min_landmarks_for_navigation_arg = DeclareLaunchArgument(
        'min_landmarks_for_navigation', default_value='1',
        description='Número mínimo de landmarks para navegação'
    )
    
    use_global_localization_arg = DeclareLaunchArgument(
        'use_global_localization', default_value='true',
        description='Usar localização global inicial'
    )
    
    enable_team_communication_arg = DeclareLaunchArgument(
        'enable_team_communication', default_value='false',
        description='Habilitar comunicação entre robôs'
    )
    
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode', default_value='false',
        description='Modo debug com publicações extras'
    )
    
    # Log de informações
    info_log = LogInfo(
        msg=[
            "🚀 Iniciando Sistema de Navegação Avançado RoboIME HSL2025\n",
            "=================================================================\n",
            "🧮 Algoritmos: Particle Filter (MCL) + Extended Kalman Filter\n",
            "🎮 Coordenador: Fusão inteligente com múltiplos modos\n", 
            "🎯 Landmarks: penalty_mark, goal_post, center_circle, field_corner, area_corner\n",
            "📍 Localização: Triangulação por landmarks + fusão sensorial\n",
            "🗺️ Planejamento: A* global + Campos potenciais local\n",
            "🏆 Sistema: Otimizado para RoboCup Humanoid League\n",
            "=================================================================\n"
        ]
    )
    
    # =============================================================================
    # SISTEMA DE LOCALIZAÇÃO C++ AVANÇADO
    # =============================================================================
    
    # Particle Filter (Monte Carlo Localization)
    particle_filter_node = Node(
        package='roboime_navigation',
        executable='particle_filter_node',
        name='particle_filter_node',
        output='screen',
        parameters=[
            {
                'field_length': LaunchConfiguration('field_length'),
                'field_width': LaunchConfiguration('field_width'),
                'num_particles': 500,  # Aumentado para melhor precisão
                'robot_id': LaunchConfiguration('robot_id'),
                'team_side': LaunchConfiguration('team_side'),
                'publish_rate': 20.0,
                'enable_tf_broadcast': False,  # TF será publicado pelo coordinator
                'resample_threshold': 0.5,
                'motion_noise_std': 0.05,
                'measurement_noise_std': 0.1
            }
        ],
        remappings=[
            ('odometry', 'odometry'),
            ('imu/data', 'imu/data'),
            ('perception/landmarks', '/perception/localization_landmarks'),
            ('localization/pose', 'mcl/pose'),
            ('localization/pose_with_covariance', 'mcl/pose_with_covariance'),
            ('localization/confidence', 'mcl/confidence'),
            ('localization/status', 'mcl/status')
        ]
    )
    
    # Extended Kalman Filter
    ekf_localization_node = Node(
        package='roboime_navigation',
        executable='ekf_localization_node',
        name='ekf_localization_node',
        output='screen',
        parameters=[
            {
                'field_length': LaunchConfiguration('field_length'),
                'field_width': LaunchConfiguration('field_width'),
                'publish_rate': 20.0,
                'enable_tf_broadcast': False,  # TF será publicado pelo coordinator
                'process_noise_std': 0.1,
                'measurement_noise_std': 0.15,
                'imu_noise_std': 0.02,
                'innovation_threshold': 9.0
            }
        ],
        remappings=[
            ('odometry', 'odometry'),
            ('imu/data', 'imu/data'),
            ('perception/landmarks', '/perception/localization_landmarks'),
            ('ekf/pose', 'ekf/pose'),
            ('ekf/pose_with_covariance', 'ekf/pose_with_covariance'),
            ('ekf/confidence', 'ekf/confidence'),
            ('ekf/status', 'ekf/status')
        ]
    )
    
    # Navigation Coordinator (Fusão Inteligente)
    navigation_coordinator_node = Node(
        package='roboime_navigation',
        executable='navigation_coordinator_node',
        name='navigation_coordinator_node',
        output='screen',
        parameters=[
            {
                'field_length': LaunchConfiguration('field_length'),
                'field_width': LaunchConfiguration('field_width'),
                'robot_id': LaunchConfiguration('robot_id'),
                'team_name': 'RoboIME',
                'team_side': LaunchConfiguration('team_side'),
                'publish_rate': 20.0,
                'enable_tf_broadcast': True,  # TF principal
                'enable_team_communication': LaunchConfiguration('enable_team_communication'),
                'confidence_threshold': LaunchConfiguration('min_localization_confidence'),
                'use_global_localization': LaunchConfiguration('use_global_localization')
            }
        ],
        remappings=[
            ('odometry', 'odometry'),
            ('imu/data', 'imu/data'),
            ('perception/landmarks', '/perception/localization_landmarks'),
            # Poses dos algoritmos individuais
            ('particle_filter/pose', 'mcl/pose'),
            ('ekf/pose', 'ekf/pose'),
            # Pose final fusionada
            ('robot_pose', 'robot_pose'),
            ('robot_pose_with_covariance', 'robot_pose_with_covariance'),
            ('localization_confidence', 'localization_confidence'),
            ('localization_status', 'localization_status'),
            ('localization_mode', 'localization_mode'),
            # Comunicação do time
            ('team/robot_info', 'team/robot_info'),
            ('team/broadcast', 'team/broadcast'),
            ('game_controller/state', 'game_controller/state')
        ]
    )
    
    # =============================================================================
    # SISTEMA DE PLANEJAMENTO (PYTHON)
    # =============================================================================
    
    # Planejador de trajetória
    path_planner_node = Node(
        package='roboime_navigation',
        executable='path_planner',
        name='path_planner_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'field_length': LaunchConfiguration('field_length'),
                'field_width': LaunchConfiguration('field_width'),
                'max_linear_velocity': 0.5,
                'max_angular_velocity': 1.2,
                'use_potential_fields': True,
                'attractive_gain': 1.2,
                'repulsive_gain': 2.5,
                'obstacle_threshold': 1.2,
                'planning_frequency': 5.0,
                'control_frequency': 20.0,
                'goal_tolerance': 0.2
            }
        ],
        remappings=[
            ('robot_pose', 'robot_pose'),  # Usa pose fusionada
            ('localization/confidence', 'localization_confidence'),
            ('navigation_goal', 'navigation_goal'),
            ('/perception/robot_detections', '/perception/robot_detections'),
            ('nav_cmd_vel', 'nav_cmd_vel'),
            ('planned_path', 'planned_path'),
            ('planner_status', 'planner_status')
        ]
    )
    
    # =============================================================================
    # COORDENADOR DE NAVEGAÇÃO (PYTHON)
    # =============================================================================
    
    # Coordenador principal (interface com comportamento)
    navigation_manager_node = Node(
        package='roboime_navigation',
        executable='navigation_manager',
        name='navigation_manager_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'enable_safety_checks': LaunchConfiguration('enable_safety'),
                'localization_timeout': 5.0,
                'planning_timeout': 2.0,
                'navigation_timeout': 30.0,
                'min_localization_confidence': LaunchConfiguration('min_localization_confidence'),
                'required_landmarks_for_navigation': LaunchConfiguration('min_landmarks_for_navigation')
            }
        ],
        remappings=[
            ('robot_pose', 'robot_pose'),  # Usa pose fusionada
            ('behavior/navigation_request', 'behavior/navigation_request'),
            ('behavior/cancel_navigation', 'behavior/cancel_navigation'),
            ('navigation_goal', 'navigation_goal'),
            ('nav_cmd_vel', 'nav_cmd_vel'),
            ('cmd_vel', 'cmd_vel'),
            ('navigation_status', 'navigation/status'),
            ('emergency_stop', 'navigation/emergency_stop'),
            ('localization_confidence', 'localization_confidence'),
            # Monitoramento da percepção
            ('/perception/localization_landmarks', '/perception/localization_landmarks'),
            ('/perception/goal_detections', '/perception/goal_detections')
        ]
    )
    
    # =============================================================================
    # RETORNO
    # =============================================================================
    
    nodes = [
        # Logs e argumentos
        info_log,
        
        # Sistema de localização C++ avançado
        particle_filter_node,
        ekf_localization_node,
        navigation_coordinator_node,
        
        # Sistema de planejamento Python
        path_planner_node,
        
        # Coordenador de navegação Python
        navigation_manager_node,
    ]
    
    # Adicionar nós de debug se habilitado
    debug_condition = PythonExpression(['"', LaunchConfiguration('debug_mode'), '" == "true"'])
    
    return LaunchDescription([
        # Argumentos
        config_file_arg,
        field_length_arg,
        field_width_arg,
        robot_id_arg,
        team_side_arg,
        enable_safety_arg,
        min_localization_confidence_arg,
        min_landmarks_for_navigation_arg,
        use_global_localization_arg,
        enable_team_communication_arg,
        debug_mode_arg,
        
        # Nós principais
        *nodes
    ]) 