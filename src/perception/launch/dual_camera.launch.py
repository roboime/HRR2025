#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch file para Sistema de Percepção RoboIME HSL2025 com suporte dual câmeras
CSI IMX219 ou USB Logitech C930 - Jetson Orin Nano Super
YOLOv8 Unificado para todos os elementos do futebol robótico
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.substitutions import FindPackageShare
import os
import yaml

def load_camera_config(context, *args, **kwargs):
    """Carrega a configuração da câmera do arquivo YAML"""
    config_file = LaunchConfiguration('config_file').perform(context)
    
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config.get('camera_type', 'csi')
    except:
        return 'csi'  # Default

def create_camera_nodes(context, *args, **kwargs):
    """Cria os nós de câmera apropriados baseados na configuração"""
    config_file = LaunchConfiguration('config_file').perform(context)
    camera_type = LaunchConfiguration('camera_type').perform(context)
    debug = LaunchConfiguration('debug').perform(context)
    
    nodes = []
    
    # Parâmetros comuns
    common_params = {
        'config_file': config_file,
        'debug': debug == 'true'
    }
    
    if camera_type == 'usb':
        # Nó da câmera USB C930
        usb_camera_node = Node(
            package='perception',
            executable='usb_camera',
            name='usb_camera',
            output='screen',
            parameters=[config_file, common_params, {
                'device_path': LaunchConfiguration('device_path')
            }],
            remappings=[
                ('/camera/image_raw', LaunchConfiguration('camera_topic')),
                ('/camera/camera_info', LaunchConfiguration('camera_info_topic'))
            ]
        )
        nodes.append(usb_camera_node)
        
        # Log info
        nodes.append(LogInfo(
            msg="🎥 Inicializando com câmera USB Logitech C930"
        ))
        
    else:  # camera_type == 'csi' ou default
        # Nó da câmera CSI IMX219
        csi_camera_node = Node(
            package='perception',
            executable='csi_camera_node',
            name='csi_camera_node',
            output='screen',
            parameters=[config_file, common_params],
            remappings=[
                ('/camera/image_raw', LaunchConfiguration('camera_topic')),
                ('/camera/camera_info', LaunchConfiguration('camera_info_topic'))
            ]
        )
        nodes.append(csi_camera_node)
        
        # Log info
        nodes.append(LogInfo(
            msg="🎥 Inicializando com câmera CSI IMX219"
        ))
    
    return nodes

def generate_launch_description():
    """Gera a descrição de lançamento com suporte dual para câmeras"""
    
    # Argumentos de lançamento
    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('perception'),
            'config',
            'perception_config.yaml'
        ]),
        description='Caminho para o arquivo de configuração'
    )
    
    declare_camera_type = DeclareLaunchArgument(
        'camera_type',
        default_value='csi',
        description='Tipo de câmera: "csi" para IMX219 ou "usb" para C930'
    )

    declare_device_path = DeclareLaunchArgument(
        'device_path',
        default_value='/dev/video0',
        description='Caminho do dispositivo da câmera USB (ex: /dev/video0 ou índice)'
    )
    
    declare_use_debug = DeclareLaunchArgument(
        'debug',
        default_value='true',
        description='Habilitar imagens de debug'
    )
    
    declare_camera_topic = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Tópico da câmera'
    )
    
    declare_camera_info_topic = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/camera_info',
        description='Tópico de informações da câmera'
    )
    
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('perception'),
            'resources/models/robocup_yolov8.pt'
        ]),
        description='Caminho para o modelo YOLOv8 customizado'
    )
    
    declare_confidence = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.6',
        description='Threshold de confiança YOLOv8'
    )
    
    declare_device = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Dispositivo para YOLOv8 (cuda/cpu)'
    )
    
    declare_iou_threshold = DeclareLaunchArgument(
        'iou_threshold',
        default_value='0.45',
        description='Threshold de IoU para Non-Maximum Suppression'
    )
    
    declare_max_detections = DeclareLaunchArgument(
        'max_detections',
        default_value='300',
        description='Máximo número de detecções por frame'
    )
    
    # Função para criar nós de câmera dinamicamente
    camera_nodes = OpaqueFunction(function=create_camera_nodes)
    
    # Nó YOLOv8 Unificado (principal)
    yolov8_unified_detector = Node(
        package='perception',
        executable='yolov8_unified_detector',
        name='yolov8_unified_detector',
        output='screen',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'model_path': LaunchConfiguration('model_path'),
            'device': LaunchConfiguration('device'),
            'publish_debug': LaunchConfiguration('debug'),
            'iou_threshold': LaunchConfiguration('iou_threshold'),
            'max_detections': LaunchConfiguration('max_detections')
        }],
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('camera_topic')),
            ('/debug_image', '/perception/debug_image'),
            ('/ball_detection', '/perception/ball_detection'),
            ('/robot_detections', '/perception/robot_detections'),
            ('/goal_detections', '/perception/goal_detections'),
            ('/field_detection', '/perception/field_detection'),
            ('/line_detection', '/perception/line_detection'),
            ('/unified_detections', '/perception/unified_detections')
        ]
    )
    
    # Log informativo
    info_log = LogInfo(
        msg=[
            "🚀 Sistema de Percepção RoboIME HSL2025 Iniciado\n",
            "🎯 Detector: YOLOv8 Simplificado (7 classes otimizadas)\n",
            "🎥 Câmera: ", LaunchConfiguration('camera_type'), "\n",
            "🔧 Config: ", LaunchConfiguration('config_file'), "\n",
            "🧠 Modelo: ", LaunchConfiguration('model_path'), "\n",
            "⚡ Device: ", LaunchConfiguration('device')
        ]
    )
    
    return LaunchDescription([
        # Argumentos
        declare_config_file,
        declare_camera_type,
        declare_device_path,
        declare_use_debug,
        declare_camera_topic,
        declare_camera_info_topic,
        declare_model_path,
        declare_confidence,
        declare_device,
        declare_iou_threshold,
        declare_max_detections,
        
        # Log inicial
        info_log,
        
        # Nós de câmera (dinâmicos)
        camera_nodes,
        
        # Detector YOLOv8 Unificado
        yolov8_unified_detector,
    ])


if __name__ == '__main__':
    generate_launch_description() 