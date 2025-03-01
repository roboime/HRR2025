#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    """Gera a descrição de inicialização para o sistema de visão na Jetson Nano."""
    
    # Obter diretório do pacote
    pkg_dir = get_package_share_directory('perception')
    
    # Arquivo de configuração
    config_file = os.path.join(pkg_dir, 'config', 'vision_params.yaml')
    
    # Declarar argumentos de inicialização para a câmera
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
    
    # Declarar argumentos para os detectores
    enable_ball_detector = DeclareLaunchArgument(
        'enable_ball_detector',
        default_value='true',
        description='Habilitar detector de bola'
    )
    
    enable_field_detector = DeclareLaunchArgument(
        'enable_field_detector',
        default_value='true',
        description='Habilitar detector de campo'
    )
    
    enable_line_detector = DeclareLaunchArgument(
        'enable_line_detector',
        default_value='true',
        description='Habilitar detector de linhas'
    )
    
    enable_goal_detector = DeclareLaunchArgument(
        'enable_goal_detector',
        default_value='true',
        description='Habilitar detector de gol'
    )
    
    enable_obstacle_detector = DeclareLaunchArgument(
        'enable_obstacle_detector',
        default_value='true',
        description='Habilitar detector de obstáculos'
    )
    
    # Incluir inicialização da câmera da Jetson
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'jetson_camera.launch.py')
        ),
        launch_arguments={
            'camera_type': LaunchConfiguration('camera_type'),
            'camera_index': LaunchConfiguration('camera_index'),
            'camera_width': LaunchConfiguration('camera_width'),
            'camera_height': LaunchConfiguration('camera_height'),
            'camera_fps': LaunchConfiguration('camera_fps'),
            'enable_display': 'false'
        }.items()
    )
    
    # Incluir inicialização do sistema de visão
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'vision.launch.py')
        ),
        launch_arguments={
            'enable_ball_detector': LaunchConfiguration('enable_ball_detector'),
            'enable_field_detector': LaunchConfiguration('enable_field_detector'),
            'enable_line_detector': LaunchConfiguration('enable_line_detector'),
            'enable_goal_detector': LaunchConfiguration('enable_goal_detector'),
            'enable_obstacle_detector': LaunchConfiguration('enable_obstacle_detector')
        }.items()
    )
    
    # Retornar descrição de inicialização
    return LaunchDescription([
        camera_type,
        camera_index,
        camera_width,
        camera_height,
        camera_fps,
        enable_ball_detector,
        enable_field_detector,
        enable_line_detector,
        enable_goal_detector,
        enable_obstacle_detector,
        camera_launch,
        vision_launch
    ]) 