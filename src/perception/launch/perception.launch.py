#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch file simplificado para Sistema de Percepção RoboIME HSL2025
YOLOv8 Unificado - Jetson Orin Nano Super
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition

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

    # Permitir especificar device_path quando camera_type=usb
    device_path_arg = DeclareLaunchArgument(
        'device_path',
        default_value='/dev/video0',
        description='Caminho do dispositivo da câmera USB (ex: /dev/video0 ou índice)'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='true',
        description='Habilitar visualizações de debug (publica /yolov8_detector/debug_image_3d)'
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

    imgsz_arg = DeclareLaunchArgument(
        'imgsz',
        default_value='640',
        description='Tamanho de entrada da imagem para o YOLOv8'
    )

    turbo_mode_arg = DeclareLaunchArgument(
        'turbo_mode',
        default_value='false',
        description='Ativa modo turbo (usa turbo_imgsz)'
    )

    turbo_imgsz_arg = DeclareLaunchArgument(
        'turbo_imgsz',
        default_value='512',
        description='Tamanho de imagem quando turbo_mode=true (ex: 512 ou 384)'
    )
    
    def create_camera_nodes(context):
        cam = LaunchConfiguration('camera_type').perform(context)
        nodes = []
        if cam == 'usb':
            nodes.append(Node(
                package='perception',
                executable='usb_camera',
                name='usb_camera',
                output='screen',
                parameters=[{
                    'device_path': LaunchConfiguration('device_path'),
                    'width': 640,
                    'height': 480,
                    'fps': 30.0,
                    'fourcc': 'MJPG',
                    'zoom': 100,
                    'config_file': LaunchConfiguration('config_file')
                }]
            ))
        else:
            nodes.append(Node(
                package='perception',
                executable='csi_camera',
                name='csi_camera',
                output='screen',
                parameters=[{
                    'device_id': 0,
                    'width': 1280,
                    'height': 720,
                    'framerate': 30.0,
                    'flip_method': 2,
                    'config_file': LaunchConfiguration('config_file')
                }]
            ))
        return nodes
    camera_nodes = OpaqueFunction(function=create_camera_nodes)
    
    # Nó YOLOv8 Unificado (principal)
    yolov8_detector_node = Node(
        package='perception',
        executable='yolov8_detector',
        name='yolov8_detector',
        output='screen',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'model_path': LaunchConfiguration('model_path'),
            'device': 'cuda',
            'publish_debug': LaunchConfiguration('debug'),
            'iou_threshold': 0.45,
            'max_detections': 300,
            'imgsz': LaunchConfiguration('imgsz'),
            'turbo_mode': LaunchConfiguration('turbo_mode'),
            'turbo_imgsz': LaunchConfiguration('turbo_imgsz')
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
    # Abrir janela automaticamente quando debug=true
    debug_view = Node(
        package='image_view',
        executable='image_view',
        name='debug_view',
        output='screen',
        remappings=[('image', '/yolov8_detector/debug_image_3d')],
        condition=IfCondition(LaunchConfiguration('debug'))
    )
    # Viewer removido: quando debug=true, apenas publica a imagem em /yolov8_detector/debug_image_3d
    
    return LaunchDescription([
        # Argumentos
        camera_type_arg,
        config_file_arg,
        device_path_arg,
        debug_arg,
        model_path_arg,
        confidence_arg,
        imgsz_arg,
        turbo_mode_arg,
        turbo_imgsz_arg,
        
        # Log
        startup_log,
        
        # Nós de câmera (criados dinamicamente)
        camera_nodes,
        
        # Detector principal
        yolov8_detector_node,
        debug_view,
    ]) 