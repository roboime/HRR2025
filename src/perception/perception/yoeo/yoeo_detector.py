#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nó ROS para detecção e segmentação usando o modelo YOEO.

Este nó implementa a interface ROS para o sistema YOEO, recebendo imagens
da câmera, processando-as com o modelo YOEO e publicando os resultados
de detecção e segmentação.
"""

import os
import cv2
import numpy as np
import yaml
import time
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseWithCovariance, Pose
from std_msgs.msg import Header

from .yoeo_handler import YOEOHandler, DetectionType, SegmentationType
from .components.ball_component import BallDetectionComponent
from .components.field_component import FieldSegmentationComponent
from .components.line_component import LineSegmentationComponent
from .components.goal_component import GoalDetectionComponent
from .components.robot_component import RobotDetectionComponent
from .components.referee_component import RefereeDetectionComponent

# Opcionalmente, importe os detectores tradicionais para fallback
try:
    from src.perception.src.ball_detector import BallDetector
    from src.perception.src.field_detector import FieldDetector
    from src.perception.src.line_detector import LineDetector
    from src.perception.src.goal_detector import GoalDetector
    from src.perception.src.obstacle_detector import ObstacleDetector
    TRADITIONAL_DETECTORS_AVAILABLE = True
except ImportError:
    TRADITIONAL_DETECTORS_AVAILABLE = False
    print("Detectores tradicionais não disponíveis para fallback")

class YOEODetector(Node):
    """
    Nó ROS para detecção e segmentação usando o modelo YOEO.
    
    Este nó recebe imagens da câmera, processa-as com o modelo YOEO
    e publica os resultados de detecção e segmentação.
    """
    
    def __init__(self):
        """Inicializa o nó detector YOEO."""
        super().__init__('yoeo_detector')
        
        # Declarar parâmetros
        self.declare_parameter('model_path', 'src/perception/resource/models/yoeo_model.h5')
        self.declare_parameter('config_file', 'src/perception/config/vision_params.yaml')
        self.declare_parameter('input_width', 416)
        self.declare_parameter('input_height', 416)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('use_tensorrt', False)
        self.declare_parameter('debug_image', True)
        
        # Parâmetros para habilitar/desabilitar componentes
        self.declare_parameter('enable_ball_detection', True)
        self.declare_parameter('enable_field_segmentation', True)
        self.declare_parameter('enable_line_segmentation', True)
        self.declare_parameter('enable_goal_detection', True)
        self.declare_parameter('enable_robot_detection', True)
        self.declare_parameter('enable_referee_detection', False)
        
        # Parâmetro para fallback para detectores tradicionais
        self.declare_parameter('fallback_to_traditional', True)
        
        # Obter parâmetros
        self.model_path = self.get_parameter('model_path').value
        self.config_file = self.get_parameter('config_file').value
        self.input_width = self.get_parameter('input_width').value
        self.input_height = self.get_parameter('input_height').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.use_tensorrt = self.get_parameter('use_tensorrt').value
        self.debug_image = self.get_parameter('debug_image').value
        
        # Flags para componentes
        self.enable_ball_detection = self.get_parameter('enable_ball_detection').value
        self.enable_field_segmentation = self.get_parameter('enable_field_segmentation').value
        self.enable_line_segmentation = self.get_parameter('enable_line_segmentation').value
        self.enable_goal_detection = self.get_parameter('enable_goal_detection').value
        self.enable_robot_detection = self.get_parameter('enable_robot_detection').value
        self.enable_referee_detection = self.get_parameter('enable_referee_detection').value
        
        self.fallback_to_traditional = self.get_parameter('fallback_to_traditional').value and TRADITIONAL_DETECTORS_AVAILABLE
        
        # Carregar configuração
        self.config = self._load_config()
        
        # Inicializar o manipulador YOEO
        self.yoeo_handler = YOEOHandler(
            model_path=self.model_path,
            input_width=self.input_width,
            input_height=self.input_height,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            use_tensorrt=self.use_tensorrt
        )
        
        # Inicializar componentes
        self._init_components()
        
        # Inicializar bridge para conversão de imagens
        self.bridge = CvBridge()
        
        # Informações da câmera
        self.camera_info = None
        
        # Subscribers
        self.create_subscription(
            Image,
            self.get_parameter_or('camera_image_topic', '/camera/image_raw').value,
            self.image_callback,
            10
        )
        
        self.create_subscription(
            CameraInfo,
            self.get_parameter_or('camera_info_topic', '/camera/camera_info').value,
            self.camera_info_callback,
            10
        )
        
        # Publishers
        self._init_publishers()
        
        # Estatísticas de desempenho
        self.frame_count = 0
        self.total_time = 0
        self.fps = 0
        
        self.get_logger().info('Nó YOEO Detector inicializado')
    
    def _load_config(self):
        """Carrega o arquivo de configuração."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            self.get_logger().info(f'Configuração carregada de {self.config_file}')
            return config
        except Exception as e:
            self.get_logger().error(f'Erro ao carregar configuração: {e}')
            return {}
    
    def _init_components(self):
        """Inicializa os componentes de detecção e segmentação."""
        self.components = {}
        
        # Componente de detecção de bola
        if self.enable_ball_detection:
            self.components['ball'] = BallDetectionComponent(
                yoeo_handler=self.yoeo_handler,
                ball_diameter=self.config.get('ball', {}).get('diameter', 0.043)
            )
            self.get_logger().info('Componente de detecção de bola inicializado')
        
        # Componente de segmentação de campo
        if self.enable_field_segmentation:
            self.components['field'] = FieldSegmentationComponent(
                yoeo_handler=self.yoeo_handler,
                min_field_area_ratio=self.config.get('field', {}).get('min_area_ratio', 0.1)
            )
            self.get_logger().info('Componente de segmentação de campo inicializado')
        
        # Componente de segmentação de linhas
        if self.enable_line_segmentation:
            self.components['line'] = LineSegmentationComponent(
                yoeo_handler=self.yoeo_handler,
                field_component=self.components.get('field', None)
            )
            self.get_logger().info('Componente de segmentação de linhas inicializado')
        
        # Componente de detecção de gol
        if self.enable_goal_detection:
            self.components['goal'] = GoalDetectionComponent(
                yoeo_handler=self.yoeo_handler,
                goal_height=self.config.get('goal', {}).get('height', 0.18)
            )
            self.get_logger().info('Componente de detecção de gol inicializado')
        
        # Componente de detecção de robô
        if self.enable_robot_detection:
            self.components['robot'] = RobotDetectionComponent(
                yoeo_handler=self.yoeo_handler,
                robot_height=self.config.get('robot', {}).get('height', 0.15)
            )
            self.get_logger().info('Componente de detecção de robô inicializado')
        
        # Componente de detecção de árbitro
        if self.enable_referee_detection:
            self.components['referee'] = RefereeDetectionComponent(
                yoeo_handler=self.yoeo_handler
            )
            self.get_logger().info('Componente de detecção de árbitro inicializado')
        
        # Configurar detectores tradicionais para fallback se solicitado
        if self.fallback_to_traditional:
            self.setup_traditional_detectors()
    
    def setup_traditional_detectors(self):
        """Configura detectores tradicionais para fallback."""
        if not TRADITIONAL_DETECTORS_AVAILABLE:
            self.get_logger().warn('Detectores tradicionais não disponíveis para fallback')
            return
        
        # Ball detector
        if self.enable_ball_detection and 'ball' in self.components:
            traditional_ball_detector = BallDetector()
            self.components['ball'].set_fallback_detector(traditional_ball_detector)
            self.get_logger().info('Detector tradicional de bola configurado para fallback')
        
        # Field detector
        if self.enable_field_segmentation and 'field' in self.components:
            traditional_field_detector = FieldDetector()
            self.components['field'].set_fallback_detector(traditional_field_detector)
            self.get_logger().info('Detector tradicional de campo configurado para fallback')
        
        # Line detector
        if self.enable_line_segmentation and 'line' in self.components:
            traditional_line_detector = LineDetector()
            self.components['line'].set_fallback_detector(traditional_line_detector)
            self.get_logger().info('Detector tradicional de linhas configurado para fallback')
        
        # Goal detector
        if self.enable_goal_detection and 'goal' in self.components:
            traditional_goal_detector = GoalDetector()
            self.components['goal'].set_fallback_detector(traditional_goal_detector)
            self.get_logger().info('Detector tradicional de gols configurado para fallback')
        
        # Robot detector (usando ObstacleDetector)
        if self.enable_robot_detection and 'robot' in self.components:
            traditional_robot_detector = ObstacleDetector()
            self.components['robot'].set_fallback_detector(traditional_robot_detector)
            self.get_logger().info('Detector tradicional de robôs configurado para fallback')
    
    def _init_publishers(self):
        """Inicializa os publishers para os resultados de detecção e segmentação."""
        self.publishers = {}
        
        # Publishers para detecções
        if self.enable_ball_detection:
            self.publishers['ball'] = self.create_publisher(
                Detection2DArray,
                'vision/ball/detections',
                10
            )
        
        if self.enable_goal_detection:
            self.publishers['goal'] = self.create_publisher(
                Detection2DArray,
                'vision/goal/detections',
                10
            )
        
        if self.enable_robot_detection:
            self.publishers['robot'] = self.create_publisher(
                Detection2DArray,
                'vision/robot/detections',
                10
            )
        
        if self.enable_referee_detection:
            self.publishers['referee'] = self.create_publisher(
                Detection2DArray,
                'vision/referee/detections',
                10
            )
        
        # Publishers para segmentações
        if self.enable_field_segmentation:
            self.publishers['field_mask'] = self.create_publisher(
                Image,
                'vision/field/mask',
                10
            )
        
        if self.enable_line_segmentation:
            self.publishers['line_mask'] = self.create_publisher(
                Image,
                'vision/line/mask',
                10
            )
        
        # Publisher para imagem de debug
        if self.debug_image:
            self.publishers['debug_image'] = self.create_publisher(
                Image,
                'vision/debug/image',
                10
            )
    
    def camera_info_callback(self, msg):
        """Callback para receber informações da câmera."""
        self.camera_info = msg
        
        # Atualizar informações da câmera nos componentes
        for component_name, component in self.components.items():
            if hasattr(component, 'set_camera_info'):
                component.set_camera_info(self.camera_info)
    
    def image_callback(self, msg):
        """Callback para processar imagens da câmera."""
        start_time = time.time()
        
        try:
            # Converter mensagem ROS para imagem OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'Erro na conversão da imagem: {e}')
            return
        
        # Processar a imagem com o modelo YOEO
        results = self._process_image(cv_image)
        
        # Publicar resultados
        self._publish_results(results, msg.header)
        
        # Calcular FPS
        self.frame_count += 1
        elapsed_time = time.time() - start_time
        self.total_time += elapsed_time
        
        if self.frame_count % 10 == 0:
            self.fps = 10 / self.total_time
            self.total_time = 0
            self.get_logger().info(f'FPS: {self.fps:.2f}')
    
    def _process_image(self, image):
        """
        Processa a imagem com o modelo YOEO e componentes.
        
        Args:
            image: Imagem OpenCV (BGR)
            
        Returns:
            Dicionário com os resultados de cada componente
        """
        results = {}
        debug_image = image.copy() if self.debug_image else None
        
        # Processar cada componente
        for component_name, component in self.components.items():
            try:
                component_result = component.process(image)
                results[component_name] = component_result
                
                # Adicionar visualizações à imagem de debug
                if self.debug_image and hasattr(component, 'draw_detections'):
                    component.draw_detections(debug_image, component_result)
            except Exception as e:
                self.get_logger().error(f'Erro no processamento do componente {component_name}: {e}')
        
        # Adicionar FPS à imagem de debug
        if self.debug_image:
            cv2.putText(
                debug_image,
                f'FPS: {self.fps:.2f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            results['debug_image'] = debug_image
        
        return results
    
    def _publish_results(self, results, header):
        """
        Publica os resultados de detecção e segmentação.
        
        Args:
            results: Dicionário com os resultados de cada componente
            header: Cabeçalho da mensagem original
        """
        # Publicar detecções de bola
        if 'ball' in results and 'ball' in self.publishers:
            ball_detections = self._create_detection_msg(
                results['ball'],
                header,
                'ball'
            )
            self.publishers['ball'].publish(ball_detections)
        
        # Publicar detecções de gol
        if 'goal' in results and 'goal' in self.publishers:
            goal_detections = self._create_detection_msg(
                results['goal'],
                header,
                'goal'
            )
            self.publishers['goal'].publish(goal_detections)
        
        # Publicar detecções de robô
        if 'robot' in results and 'robot' in self.publishers:
            robot_detections = self._create_detection_msg(
                results['robot'],
                header,
                'robot'
            )
            self.publishers['robot'].publish(robot_detections)
        
        # Publicar detecções de árbitro
        if 'referee' in results and 'referee' in self.publishers:
            referee_detections = self._create_detection_msg(
                results['referee'],
                header,
                'referee'
            )
            self.publishers['referee'].publish(referee_detections)
        
        # Publicar máscara de campo
        if 'field' in results and 'field_mask' in self.publishers:
            field_mask = results['field'].get('mask')
            if field_mask is not None:
                try:
                    mask_msg = self.bridge.cv2_to_imgmsg(field_mask, "mono8")
                    mask_msg.header = header
                    self.publishers['field_mask'].publish(mask_msg)
                except CvBridgeError as e:
                    self.get_logger().error(f'Erro na conversão da máscara de campo: {e}')
        
        # Publicar máscara de linha
        if 'line' in results and 'line_mask' in self.publishers:
            line_mask = results['line'].get('mask')
            if line_mask is not None:
                try:
                    mask_msg = self.bridge.cv2_to_imgmsg(line_mask, "mono8")
                    mask_msg.header = header
                    self.publishers['line_mask'].publish(mask_msg)
                except CvBridgeError as e:
                    self.get_logger().error(f'Erro na conversão da máscara de linha: {e}')
        
        # Publicar imagem de debug
        if 'debug_image' in results and 'debug_image' in self.publishers:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(results['debug_image'], "bgr8")
                debug_msg.header = header
                self.publishers['debug_image'].publish(debug_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Erro na conversão da imagem de debug: {e}')
    
    def _create_detection_msg(self, detections, header, detection_type):
        """
        Cria uma mensagem Detection2DArray a partir das detecções.
        
        Args:
            detections: Lista de detecções
            header: Cabeçalho da mensagem
            detection_type: Tipo de detecção ('ball', 'goal', 'robot', 'referee')
            
        Returns:
            Mensagem Detection2DArray
        """
        detection_array = Detection2DArray()
        detection_array.header = header
        
        if not detections:
            return detection_array
        
        for detection in detections:
            det_msg = Detection2D()
            det_msg.header = header
            
            # Definir centro da detecção
            if 'center' in detection:
                center_x, center_y = detection['center']
                det_msg.bbox.center.position.x = float(center_x)
                det_msg.bbox.center.position.y = float(center_y)
            
            # Definir tamanho da caixa delimitadora
            if 'bbox' in detection:
                x, y, w, h = detection['bbox']
                det_msg.bbox.size_x = float(w)
                det_msg.bbox.size_y = float(h)
            
            # Adicionar hipótese de objeto
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = detection_type
            hypothesis.hypothesis.score = float(detection.get('confidence', 1.0))
            
            # Adicionar pose 3D se disponível
            if 'position_3d' in detection and detection['position_3d'] is not None:
                x, y, z = detection['position_3d']
                hypothesis.pose.pose.position.x = float(x)
                hypothesis.pose.pose.position.y = float(y)
                hypothesis.pose.pose.position.z = float(z)
            
            det_msg.results.append(hypothesis)
            detection_array.detections.append(det_msg)
        
        return detection_array


def main(args=None):
    """Função principal para iniciar o nó ROS."""
    rclpy.init(args=args)
    node = YOEODetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 