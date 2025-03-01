#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    """Generate launch description for YOEO detector."""
    
    # Diretório de recursos do pacote
    package_share_dir = get_package_share_directory('perception')
    
    # Argumentos do lançamento
    use_tensorrt = LaunchConfiguration('use_tensorrt')
    model_path = LaunchConfiguration('model_path')
    config_file = LaunchConfiguration('config_file')
    camera_image_topic = LaunchConfiguration('camera_image_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    
    # Definir argumentos com valores padrão
    declare_use_tensorrt = DeclareLaunchArgument(
        'use_tensorrt',
        default_value='False',
        description='Flag para usar TensorRT para inferência otimizada'
    )
    
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(package_share_dir, 'resource', 'models', 'yoeo_model.h5'),
        description='Caminho para o arquivo do modelo YOEO'
    )
    
    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(package_share_dir, 'config', 'vision_params.yaml'),
        description='Caminho para o arquivo de configuração'
    )
    
    declare_camera_image_topic = DeclareLaunchArgument(
        'camera_image_topic',
        default_value='/camera/image_raw',
        description='Tópico para a imagem da câmera'
    )
    
    declare_camera_info_topic = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/camera_info',
        description='Tópico para informações da câmera'
    )
    
    # Nó YOEO Detector
    yoeo_detector_node = Node(
        package='perception',
        executable='yoeo_detector',
        name='yoeo_detector',
        output='screen',
        parameters=[
            config_file,
            {
                'use_tensorrt': use_tensorrt,
                'model_path': model_path
            }
        ],
        remappings=[
            ('camera/image_raw', camera_image_topic),
            ('camera/camera_info', camera_info_topic)
        ]
    )
    
    return LaunchDescription([
        declare_use_tensorrt,
        declare_model_path,
        declare_config_file,
        declare_camera_image_topic,
        declare_camera_info_topic,
        yoeo_detector_node
    ]) 