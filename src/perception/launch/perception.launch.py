#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Arquivo de lançamento simplificado para o sistema de percepção da RoboIME.

Este arquivo permite iniciar o sistema de percepção com diferentes configurações,
através de uma interface simplificada.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory

import os
import yaml


def generate_launch_description():
    """Gera a descrição de lançamento para o sistema de percepção."""
    
    # Diretório de recursos do pacote
    pkg_dir = get_package_share_directory('perception')
    config_path = os.path.join(pkg_dir, 'config', 'perception_config.yaml')
    
    # --- Argumentos Simplificados ---
    
    # Modos de operação
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='unified',
        description='Modo de operação: "unified" (YOEO + Tradicional), "yoeo" (apenas YOEO) ou "traditional" (apenas detectores tradicionais)'
    )
    
    # Câmera
    camera_src_arg = DeclareLaunchArgument(
        'camera_src',
        default_value='default',
        description='Fonte da câmera: "default", "usb", "csi" ou "simulation"'
    )
    
    # Debug
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='true',
        description='Habilitar informações de debug e visualização'
    )
    
    # Caminho para configuração
    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_path,
        description='Caminho para o arquivo de configuração'
    )
    
    # --- Configurações ---
    mode = LaunchConfiguration('mode')
    camera_src = LaunchConfiguration('camera_src')
    debug = LaunchConfiguration('debug')
    config_file = LaunchConfiguration('config_file')
    
    # Função para configurar o lançamento com base nos argumentos
    def configure_launch(context):
        # Obter valores dos argumentos
        mode_value = context.launch_configurations['mode']
        camera_src_value = context.launch_configurations['camera_src']
        debug_value = context.launch_configurations['debug']
        config_file_value = context.launch_configurations['config_file']
        
        # Carregar configuração
        with open(config_file_value, 'r') as f:
            config = yaml.safe_load(f)
        
        # Configurar parâmetros com base no modo
        use_yoeo = True if mode_value in ['unified', 'yoeo'] else False
        use_traditional = True if mode_value in ['unified', 'traditional'] else False
        
        # Nó da câmera
        camera_node = None
        if camera_src_value == 'csi':
            camera_node = Node(
                package='perception',
                executable='jetson_camera_node.py',
                name='camera',
                output='screen',
                parameters=[{
                    'camera_mode': 2,  # 0=3280x2464, 1=1920x1080, 2=1280x720
                    'camera_fps': config['camera']['fps'],
                    'flip_method': 0,
                    'exposure_time': 13333,  # Otimizado para iluminação indoor
                    'gain': 1.0,
                    'awb_mode': 1,  # Auto white balance
                    'brightness': 0,
                    'saturation': 1.0,
                    'enable_cuda': True,
                    'enable_hdr': False,  # Habilitar se necessário
                    'enable_display': debug_value == 'true'
                }]
            )
        
        # Nó do pipeline de visão
        pipeline_node = Node(
            package='perception',
            executable='vision_pipeline.py',
            name='vision_pipeline',
            output='screen',
            parameters=[{
                'camera_topic': config['camera']['topic'],
                'camera_info_topic': config['camera']['info_topic'],
                'debug_image': debug_value == 'true',
                'processing_fps': config['pipeline']['processing_fps'],
                'use_yoeo': use_yoeo,
                'use_traditional': use_traditional,
                'detector_ball': config['pipeline']['detector_ball'] if mode_value == 'unified' else 'yoeo' if mode_value == 'yoeo' else 'traditional',
                'detector_field': config['pipeline']['detector_field'] if mode_value == 'unified' else 'yoeo' if mode_value == 'yoeo' else 'traditional',
                'detector_lines': config['pipeline']['detector_lines'] if mode_value == 'unified' else 'yoeo' if mode_value == 'yoeo' else 'traditional',
                'detector_goals': config['pipeline']['detector_goals'] if mode_value == 'unified' else 'yoeo' if mode_value == 'yoeo' else 'traditional',
                'detector_robots': config['pipeline']['detector_robots'] if mode_value == 'unified' else 'yoeo' if mode_value == 'yoeo' else 'traditional',
                'yoeo_model_path': config['yoeo']['model_path'],
                'yoeo_confidence_threshold': config['yoeo']['confidence_threshold'],
                'use_tensorrt': config['yoeo']['use_tensorrt'],
                'enable_ball_detection': config['traditional']['ball']['enabled'],
                'enable_field_detection': config['traditional']['field']['enabled'],
                'enable_line_detection': config['traditional']['lines']['enabled'],
                'enable_goal_detection': config['traditional']['goals']['enabled'],
                'enable_obstacle_detection': config['traditional']['robots']['enabled'],
            }]
        )
        
        # Visualizador de debug (só é iniciado quando debug=true)
        debug_node = Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='image_view',
            arguments=['/vision/debug_image'],
            condition=IfCondition(debug)
        )
        
        nodes = []
        if camera_node:
            nodes.append(camera_node)
        nodes.append(pipeline_node)
        nodes.append(debug_node)
        
        return nodes
    
    return LaunchDescription([
        # Argumentos
        mode_arg,
        camera_src_arg,
        debug_arg,
        config_arg,
        
        # Configuração dinâmica
        OpaqueFunction(function=configure_launch)
    ]) 