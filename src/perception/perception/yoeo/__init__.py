#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementação do modelo YOLOv4-Tiny para percepção visual em robôs de futebol.

O YOLOv4-Tiny é uma arquitetura neural eficiente para detecção de objetos,
otimizada para execução em tempo real em plataformas com recursos limitados
como o Jetson Nano.

Este módulo contém:
- Implementação da arquitetura do modelo YOLOv4-Tiny
- Manipulador para carregar e processar o modelo
- Detector ROS para integração com o sistema robótico
- Componentes modulares para tarefas específicas de detecção

O modelo YOLOv4-Tiny é capaz de detectar objetos como bolas, gols e robôs
em tempo real.
"""

# Importação das classes principais
from .yoeo_model import YOEOModel
from .yoeo_handler import YOEOHandler, DetectionType
from .yoeo_detector import YOEODetector

# Versão do módulo
__version__ = '0.3.0'

# Exportação das classes principais
__all__ = [
    'YOEOModel',
    'YOEOHandler',
    'YOEODetector',
    'DetectionType',
] 