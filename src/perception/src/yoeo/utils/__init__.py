#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitários para o sistema YOEO.

Este módulo contém funções e classes auxiliares para o sistema YOEO,
incluindo:

- Funções para pré-processamento de imagens
- Funções para pós-processamento de detecções e segmentações
- Utilitários para visualização de resultados
- Funções para avaliação de desempenho
- Utilitários para conversão de formatos de dados
- Funções para calibração de câmera

Estes utilitários são usados pelos componentes principais do sistema YOEO
para facilitar o processamento de imagens e a manipulação de dados.
"""

from .data_utils import (
    load_image, load_mask, load_annotations,
    normalize_image, prepare_dataset, YOEODataGenerator,
    visualize_batch
)

from .loss_utils import (
    detection_loss, segmentation_loss, YOEOLoss,
    detection_metrics, segmentation_metrics,
    iou_metric, dice_coefficient
)

# Versão do módulo de utilitários
__version__ = '0.1.0'

# Exportação das funções e classes principais (a serem implementadas)
__all__ = [
    # Pré-processamento
    # 'preprocess_image',
    # 'normalize_image',
    
    # Pós-processamento
    # 'filter_detections',
    # 'apply_nms',
    
    # Visualização
    # 'draw_detections',
    # 'draw_segmentation',
    
    # Avaliação
    # 'calculate_map',
    # 'calculate_iou',
    
    # Conversão
    # 'detection_to_ros_msg',
    # 'segmentation_to_ros_msg',
    
    # Calibração
    # 'calibrate_camera',
    # 'undistort_image',
] 