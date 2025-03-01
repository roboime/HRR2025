"""
Componentes modulares para o sistema YOEO.

Cada componente é responsável por uma funcionalidade específica de detecção 
ou segmentação, seguindo uma arquitetura modular inspirada nos Bit-Bots.
"""

from src.perception.src.yoeo.components.ball_component import BallDetectionComponent
from src.perception.src.yoeo.components.field_component import FieldSegmentationComponent
from src.perception.src.yoeo.components.line_component import LineSegmentationComponent
from src.perception.src.yoeo.components.goal_component import GoalDetectionComponent
from src.perception.src.yoeo.components.robot_component import RobotDetectionComponent
from src.perception.src.yoeo.components.referee_component import RefereeDetectionComponent

__all__ = [
    'BallDetectionComponent',
    'FieldSegmentationComponent',
    'LineSegmentationComponent',
    'GoalDetectionComponent',
    'RobotDetectionComponent',
    'RefereeDetectionComponent',
] 