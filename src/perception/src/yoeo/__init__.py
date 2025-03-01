"""
Módulo YOEO (You Only Encode Once) para detecção e segmentação em tempo real.

Este módulo implementa uma arquitetura neural unificada para detecção de objetos
e segmentação semântica, otimizada para a Jetson Nano e adaptada do sistema
de visão dos Hamburg Bit-Bots para o projeto RoboIME.

Classes principais:
- YOEOModel: Implementação do modelo neural
- YOEOHandler: Gerenciamento de inferência e processamento
- YOEODetector: Nó ROS para integração com o sistema
"""

from src.perception.src.yoeo.yoeo_model import YOEOModel
from src.perception.src.yoeo.yoeo_handler import YOEOHandler
from src.perception.src.yoeo.yoeo_detector import YOEODetector

__all__ = [
    'YOEOModel',
    'YOEOHandler',
    'YOEODetector',
] 