#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch file simplificado para Sistema de Percepção RoboIME HSL2025
YOLOv8 Unificado - Jetson Orin Nano Super
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """Gera descrição do launch para percepção YOLOv8 unificada"""
    
    # Argumentos de launch
    camera_type_arg = DeclareLaunchArgument(
        'camera_type',
        default_value='csi',
        description='Tipo de câmera: csi (IMX219) ou usb (C930)'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('perception'),
            'config',
            'perception_config.yaml'
        ]),
        description='Arquivo de configuração YAML'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='true',
        description='Habilitar visualizações de debug'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('perception'),
            'resources/models/robocup_yolov8.pt'
        ]),
        description='Caminho para modelo YOLOv8 customizado'
    )
    
    confidence_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.6',
        description='Threshold de confiança para detecções (0.0-1.0)'
    )
    
    # Nó da câmera CSI (IMX219)
    csi_camera_node = Node(
        package='perception',
        executable='csi_camera_node',
        name='csi_camera_node',
        output='screen',
        parameters=[{
            'camera_id': 0,
            'width': 1280,
            'height': 720,
            'fps': 30,
            'flip_method': 2,  # Rotação 180 graus se necessário
            'config_file': LaunchConfiguration('config_file')
        }],
        condition=lambda context: LaunchConfiguration('camera_type').perform(context) == 'csi'
    )
    
    # Nó da câmera USB (C930)
    usb_camera_node = Node(
        package='perception',
        executable='usb_camera_node',
        name='usb_camera_node',
        output='screen',
        parameters=[{
            'device_id': 0,
            'width': 1280,
            'height': 720,
            'fps': 30,
            'config_file': LaunchConfiguration('config_file')
        }],
        condition=lambda context: LaunchConfiguration('camera_type').perform(context) == 'usb'
    )
    
    # Nó YOLOv8 Unificado (principal)
    yolov8_detector_node = Node(
        package='perception',
        executable='yolov8_unified_detector',
        name='yolov8_unified_detector',
        output='screen',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'model_path': LaunchConfiguration('model_path'),
            'device': 'cuda',
            'publish_debug': LaunchConfiguration('debug'),
            'iou_threshold': 0.45,
            'max_detections': 300
        }],
        remappings=[
            ('camera/image_raw', '/camera/image_raw'),
        ]
    )
    
    # Logging de inicialização
    startup_log = LogInfo(
        msg=[
            'Iniciando Sistema de Percepção RoboIME HSL2025\n',
            'Câmera: ', LaunchConfiguration('camera_type'), '\n',
            'Debug: ', LaunchConfiguration('debug'), '\n',
            'Modelo: ', LaunchConfiguration('model_path')
        ]
    )
    
    return LaunchDescription([
        # Argumentos
        camera_type_arg,
        config_file_arg,
        debug_arg,
        model_path_arg,
        confidence_arg,
        
        # Log
        startup_log,
        
        # Nós de câmera (condicionais)
        csi_camera_node,
        usb_camera_node,
        
        # Detector principal
        yolov8_detector_node,
    ]) 