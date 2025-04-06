#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """Gera a descrição de inicialização para a câmera IMX219."""
    
    # Obter diretório do pacote
    pkg_dir = get_package_share_directory('perception')
    
    # Declarar argumentos de inicialização
    camera_mode = DeclareLaunchArgument(
        'camera_mode',
        default_value='6',  # Modo identificado pelo nvgstcapture para 1280x720@120fps
        description='Modo da câmera IMX219 (0=3280x2464@21fps, 1=1920x1080@60fps, 2=1280x720@120fps, 3=1280x720@60fps, 4=1920x1080@30fps, 5=1640x1232@30fps, 6=1280x720@120fps)'
    )
    
    camera_fps = DeclareLaunchArgument(
        'camera_fps',
        default_value='120',
        description='Taxa de quadros da câmera'
    )
    
    exposure_time = DeclareLaunchArgument(
        'exposure_time',
        default_value='13333',  # em microssegundos
        description='Tempo de exposição da câmera'
    )
    
    gain = DeclareLaunchArgument(
        'gain',
        default_value='1.0',
        description='Ganho da câmera'
    )
    
    awb_mode = DeclareLaunchArgument(
        'awb_mode',
        default_value='1',  # 0=off, 1=auto
        description='Modo de white balance automático'
    )
    
    enable_hdr = DeclareLaunchArgument(
        'enable_hdr',
        default_value='false',
        description='Habilitar HDR'
    )
    
    enable_cuda = DeclareLaunchArgument(
        'enable_cuda',
        default_value='true',
        description='Habilitar processamento CUDA'
    )
    
    enable_display = DeclareLaunchArgument(
        'enable_display',
        default_value='false',
        description='Habilitar exibição da imagem da câmera'
    )
    
    # Criar nó da câmera
    camera_node = Node(
        package='perception',
        executable='jetson_camera_node.py',
        name='imx219_camera_node',
        output='screen',
        parameters=[{
            'camera_mode': LaunchConfiguration('camera_mode'),
            'camera_fps': LaunchConfiguration('camera_fps'),
            'exposure_time': LaunchConfiguration('exposure_time'),
            'gain': LaunchConfiguration('gain'),
            'awb_mode': LaunchConfiguration('awb_mode'),
            'brightness': 0,
            'saturation': 1.0,
            'enable_hdr': LaunchConfiguration('enable_hdr'),
            'enable_cuda': LaunchConfiguration('enable_cuda'),
            'enable_display': LaunchConfiguration('enable_display'),
            'flip_method': 0
        }]
    )
    
    # Retornar descrição de inicialização
    return LaunchDescription([
        camera_mode,
        camera_fps,
        exposure_time,
        gain,
        awb_mode,
        enable_hdr,
        enable_cuda,
        enable_display,
        camera_node
    ])