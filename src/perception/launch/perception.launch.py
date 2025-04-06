#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Arquivo de lançamento simplificado para o sistema de percepção da RoboIME.

Este arquivo permite iniciar o sistema de percepção com diferentes configurações,
através de uma interface simplificada.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

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
        
        # Configurar os parâmetros como argumentos para o processo
        pipeline_params = [
            '--ros-args',
            '-p', f"camera_topic:={config['camera']['topic']}",
            '-p', f"camera_info_topic:={config['camera']['info_topic']}",
            '-p', f"debug_image:={debug_value == 'true'}",
            '-p', f"processing_fps:={config['pipeline']['processing_fps']}",
            '-p', f"use_yoeo:={use_yoeo}",
            '-p', f"use_traditional:={use_traditional}",
            '-p', f"detector_ball:={config['pipeline']['detector_ball'] if mode_value == 'unified' else 'yoeo' if mode_value == 'yoeo' else 'traditional'}",
            '-p', f"detector_field:={config['pipeline']['detector_field'] if mode_value == 'unified' else 'yoeo' if mode_value == 'yoeo' else 'traditional'}",
            '-p', f"detector_lines:={config['pipeline']['detector_lines'] if mode_value == 'unified' else 'yoeo' if mode_value == 'yoeo' else 'traditional'}",
            '-p', f"detector_goals:={config['pipeline']['detector_goals'] if mode_value == 'unified' else 'yoeo' if mode_value == 'yoeo' else 'traditional'}",
            '-p', f"detector_robots:={config['pipeline']['detector_robots'] if mode_value == 'unified' else 'yoeo' if mode_value == 'yoeo' else 'traditional'}",
            '-p', f"yoeo_model_path:={config['yoeo']['model_path']}",
            '-p', f"yoeo_confidence_threshold:={config['yoeo']['confidence_threshold']}",
            '-p', f"use_tensorrt:={config['yoeo']['use_tensorrt']}",
            '-p', f"enable_ball_detection:={config['traditional']['ball']['enabled']}",
            '-p', f"enable_field_detection:={config['traditional']['field']['enabled']}",
            '-p', f"enable_line_detection:={config['traditional']['lines']['enabled']}",
            '-p', f"enable_goal_detection:={config['traditional']['goals']['enabled']}",
            '-p', f"enable_obstacle_detection:={config['traditional']['robots']['enabled']}",
        ]
        
        # Nó da câmera (usando ExecuteProcess em vez de Node)
        camera_process = None
        if camera_src_value == 'csi':
            camera_params = [
                '--ros-args',
                '-p', f"camera_mode:=6",
                '-p', f"camera_fps:={config['camera']['fps']}",
                '-p', f"flip_method:=0",
                '-p', f"exposure_time:=13333",
                '-p', f"gain:=1.0",
                '-p', f"awb_mode:=1",
                '-p', f"brightness:=0",
                '-p', f"saturation:=1.0",
                '-p', f"enable_cuda:=True",
                '-p', f"enable_hdr:=False",
                '-p', f"enable_display:={debug_value == 'true'}"
            ]
            camera_process = ExecuteProcess(
                cmd=['/ros2_ws/src/perception/scripts/jetson_camera_wrapper.sh'] + camera_params,
                name='camera',
                output='screen'
            )
        
        # Nó do pipeline de visão (usando ExecuteProcess em vez de Node)
        pipeline_process = ExecuteProcess(
            cmd=['/ros2_ws/src/perception/scripts/vision_pipeline_wrapper.sh'] + pipeline_params,
            name='vision_pipeline',
            output='screen'
        )
        
        # Visualizador de debug (verificar se o pacote existe antes de adicionar)
        debug_node = None
        try:
            if debug_value == 'true':
                # Verificar se o pacote rqt_image_view está disponível
                get_package_share_directory('rqt_image_view')
                debug_node = Node(
                    package='rqt_image_view',
                    node_executable='rqt_image_view',
                    name='image_view',
                    arguments=['/vision/debug_image']
                )
        except PackageNotFoundError:
            # Adicionar um aviso em vez de falhar
            debug_node = LogInfo(
                msg="[AVISO] Pacote 'rqt_image_view' não encontrado. Visualização não disponível."
            )
        
        processes = []
        if camera_process:
            processes.append(camera_process)
        processes.append(pipeline_process)
        if debug_node:
            processes.append(debug_node)
        
        return processes
    
    return LaunchDescription([
        # Argumentos
        mode_arg,
        camera_src_arg,
        debug_arg,
        config_arg,
        
        # Configuração dinâmica
        OpaqueFunction(function=configure_launch)
    ]) 