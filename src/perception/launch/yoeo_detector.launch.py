#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """Gera a descrição de lançamento para o detector YOEO."""
    
    # Obter diretório do pacote
    pkg_dir = get_package_share_directory('perception')
    
    # Caminho para o arquivo de configuração
    config_file = os.path.join(pkg_dir, 'config', 'vision_params.yaml')
    
    # Declarar argumentos de lançamento
    model_path = LaunchConfiguration('model_path')
    input_width = LaunchConfiguration('input_width')
    input_height = LaunchConfiguration('input_height')
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    iou_threshold = LaunchConfiguration('iou_threshold')
    debug_image = LaunchConfiguration('debug_image')
    enable_ball_detection = LaunchConfiguration('enable_ball_detection')
    enable_goal_detection = LaunchConfiguration('enable_goal_detection')
    enable_robot_detection = LaunchConfiguration('enable_robot_detection')
    enable_referee_detection = LaunchConfiguration('enable_referee_detection')
    use_tensorrt = LaunchConfiguration('use_tensorrt')
    
    # Definir argumentos de lançamento
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='resource/models/yoeo_model.h5',
        description='Caminho para o modelo YOEO'
    )
    
    declare_input_width = DeclareLaunchArgument(
        'input_width',
        default_value='416',
        description='Largura da entrada do modelo'
    )
    
    declare_input_height = DeclareLaunchArgument(
        'input_height',
        default_value='416',
        description='Altura da entrada do modelo'
    )
    
    declare_confidence_threshold = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Limiar de confiança para detecções'
    )
    
    declare_iou_threshold = DeclareLaunchArgument(
        'iou_threshold',
        default_value='0.45',
        description='Limiar de IoU para non-maximum suppression'
    )
    
    declare_debug_image = DeclareLaunchArgument(
        'debug_image',
        default_value='True',
        description='Habilitar imagem de debug'
    )
    
    declare_enable_ball_detection = DeclareLaunchArgument(
        'enable_ball_detection',
        default_value='True',
        description='Habilitar detecção de bola'
    )
    
    declare_enable_goal_detection = DeclareLaunchArgument(
        'enable_goal_detection',
        default_value='True',
        description='Habilitar detecção de gol'
    )
    
    declare_enable_robot_detection = DeclareLaunchArgument(
        'enable_robot_detection',
        default_value='True',
        description='Habilitar detecção de robôs'
    )
    
    declare_enable_referee_detection = DeclareLaunchArgument(
        'enable_referee_detection',
        default_value='True',
        description='Habilitar detecção de árbitro'
    )
    
    declare_use_tensorrt = DeclareLaunchArgument(
        'use_tensorrt',
        default_value='False',
        description='Usar modelo otimizado com TensorRT'
    )
    
    # Nó do detector YOEO
    yoeo_detector_node = Node(
        package='perception',
        executable='yoeo_detector',
        name='yoeo_detector',
        parameters=[
            {
                'model_path': model_path,
                'input_width': input_width,
                'input_height': input_height,
                'confidence_threshold': confidence_threshold,
                'iou_threshold': iou_threshold,
                'debug_image': debug_image,
                'enable_ball_detection': enable_ball_detection,
                'enable_goal_detection': enable_goal_detection,
                'enable_robot_detection': enable_robot_detection,
                'enable_referee_detection': enable_referee_detection,
                'use_tensorrt': use_tensorrt
            }
        ],
        output='screen'
    )
    
    # Retornar descrição de lançamento
    return LaunchDescription([
        declare_model_path,
        declare_input_width,
        declare_input_height,
        declare_confidence_threshold,
        declare_iou_threshold,
        declare_debug_image,
        declare_enable_ball_detection,
        declare_enable_goal_detection,
        declare_enable_robot_detection,
        declare_enable_referee_detection,
        declare_use_tensorrt,
        yoeo_detector_node
    ]) 