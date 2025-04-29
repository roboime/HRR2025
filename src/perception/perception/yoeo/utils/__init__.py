#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitários para o sistema YOLOv4-Tiny.

Este módulo fornece funções para auxiliar o funcionamento do modelo YOLOv4-Tiny
para detecção de objetos em imagens de robô futebol, incluindo:

- Processamento de imagens (pré-processamento, aumento de dados)
- Pós-processamento de detecções (NMS, conversão de coordenadas)
- Avaliação de desempenho (mAP, recall, precision)
- Geradores de dados para treinamento

"""

import numpy as np
from .data_utils import (
    load_image,
    load_annotations,
    normalize_image,
    prepare_dataset,
    create_augmentation_pipeline
)
from .loss_utils import (
    detection_loss,
    detection_metrics,
    create_detection_loss
)

# Versão do módulo de utilitários
__version__ = '0.2.0'

# Exportação das funções e classes principais
__all__ = [
    # Data utils
    'load_image',
    'load_annotations',
    'normalize_image',
    'prepare_dataset',
    'create_augmentation_pipeline',
    # Loss utils
    'detection_loss',
    'detection_metrics',
    'create_detection_loss'
] 