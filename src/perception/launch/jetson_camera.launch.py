#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """Gera a descrição de inicialização para a câmera da Jetson Nano."""
    
    # Obter diretório do pacote
    pkg_dir = get_package_share_directory('perception')
    
    # Declarar argumentos de inicialização
    camera_type = DeclareLaunchArgument(
        'camera_type',
        default_value='csi',
        description='Tipo de câmera (csi ou usb)'
    )
    
    camera_index = DeclareLaunchArgument(
        'camera_index',
        default_value='0',
        description='Índice da câmera'
    )
    
    camera_width = DeclareLaunchArgument(
        'camera_width',
        default_value='640',
        description='Largura da imagem da câmera'
    )
    
    camera_height = DeclareLaunchArgument(
        'camera_height',
        default_value='480',
        description='Altura da imagem da câmera'
    )
    
    camera_fps = DeclareLaunchArgument(
        'camera_fps',
        default_value='30',
        description='Taxa de quadros da câmera'
    )
    
    display_width = DeclareLaunchArgument(
        'display_width',
        default_value='640',
        description='Largura da janela de exibição'
    )
    
    display_height = DeclareLaunchArgument(
        'display_height',
        default_value='480',
        description='Altura da janela de exibição'
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
        name='jetson_camera_node',
        output='screen',
        parameters=[{
            'camera_type': LaunchConfiguration('camera_type'),
            'camera_index': LaunchConfiguration('camera_index'),
            'camera_width': LaunchConfiguration('camera_width'),
            'camera_height': LaunchConfiguration('camera_height'),
            'camera_fps': LaunchConfiguration('camera_fps'),
            'display_width': LaunchConfiguration('display_width'),
            'display_height': LaunchConfiguration('display_height'),
            'enable_display': LaunchConfiguration('enable_display')
        }]
    )
    
    # Retornar descrição de inicialização
    return LaunchDescription([
        camera_type,
        camera_index,
        camera_width,
        camera_height,
        camera_fps,
        display_width,
        display_height,
        enable_display,
        camera_node
    ])