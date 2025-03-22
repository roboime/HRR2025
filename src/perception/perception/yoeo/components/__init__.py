#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Componentes do sistema YOEO para detecção e segmentação.

Este módulo contém componentes modulares que processam as saídas do modelo YOEO
para tarefas específicas de detecção e segmentação, incluindo:

- BallComponent: Detecção e rastreamento de bola
- FieldComponent: Segmentação do campo de jogo
- LineComponent: Segmentação e detecção de linhas do campo
- GoalComponent: Detecção de gols e postes
- RobotComponent: Detecção de robôs (aliados e adversários)
- RefereeComponent: Detecção de árbitros

Cada componente pode ser usado independentemente ou em conjunto com outros
componentes, dependendo das necessidades da aplicação.
"""

# Importação dos componentes
from .ball_component import BallDetectionComponent
from .field_component import FieldSegmentationComponent
from .line_component import LineSegmentationComponent
from .goal_component import GoalDetectionComponent
from .robot_component import RobotDetectionComponent
from .referee_component import RefereeDetectionComponent

# Exportação das classes principais
__all__ = [
    'BallDetectionComponent',
    'FieldSegmentationComponent',
    'LineSegmentationComponent',
    'GoalDetectionComponent',
    'RobotDetectionComponent',
    'RefereeDetectionComponent',
] 