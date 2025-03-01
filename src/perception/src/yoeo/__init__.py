#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementação do modelo YOEO (You Only Encode Once) para percepção visual em robôs de futebol.

O YOEO é uma arquitetura neural híbrida que combina detecção de objetos e segmentação
semântica em um único modelo eficiente, otimizado para execução em tempo real em
plataformas com recursos limitados como o Jetson Nano.

Este módulo contém:
- Implementação da arquitetura do modelo YOEO
- Manipulador para carregar e processar o modelo
- Detector ROS para integração com o sistema robótico
- Componentes modulares para tarefas específicas de detecção e segmentação

O modelo YOEO é capaz de detectar objetos como bolas, gols, robôs e árbitros,
além de segmentar o campo e as linhas em uma única passagem pela rede neural.
"""

# Importação das classes principais
from .yoeo_model import YOEOModel
from .yoeo_handler import YOEOHandler, DetectionType, SegmentationType
from .yoeo_detector import YOEODetector

# Versão do módulo YOEO
__version__ = '0.1.0'

# Exportação das classes principais
__all__ = [
    'YOEOModel',
    'YOEOHandler',
    'YOEODetector',
    'DetectionType',
    'SegmentationType',
] 