#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nó ROS para detecção usando o modelo YOLO.

Este nó implementa a interface ROS para o sistema YOLOv5, recebendo imagens
da câmera, processando-as com o modelo YOLO e publicando os resultados
de detecção.
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

from .yoeo_handler import YOEOHandler, DetectionType
from .components.ball_component import BallDetectionComponent
from .components.goal_component import GoalDetectionComponent
from .components.robot_component import RobotDetectionComponent

# Opcionalmente, importe os detectores tradicionais para fallback
try:
    from src.perception.src.ball_detector import BallDetector
    from src.perception.src.goal_detector import GoalDetector
    from src.perception.src.obstacle_detector import ObstacleDetector
    TRADITIONAL_DETECTORS_AVAILABLE = True
except ImportError:
    TRADITIONAL_DETECTORS_AVAILABLE = False
    print("Detectores tradicionais não disponíveis para fallback")

class YOEODetector(Node):
    """
    Nó ROS para detecção usando o modelo YOLO.
    
    Este nó recebe imagens da câmera, processa-as com o modelo YOLO
    e publica os resultados de detecção.
    """
    
    def __init__(self):
        """Inicializa o nó detector YOLO."""
        super().__init__('yolo_detector')
        
        # Declarar parâmetros
        self.declare_parameter('model_path', 'src/perception/resource/models/yolov5s.pt')
        self.declare_parameter('config_file', 'src/perception/config/vision_params.yaml')
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 640)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('use_tensorrt', False)
        self.declare_parameter('debug_image', True)
        
        # Parâmetros para habilitar/desabilitar componentes
        self.declare_parameter('enable_ball_detection', True)
        self.declare_parameter('enable_goal_detection', True)
        self.declare_parameter('enable_robot_detection', True)
        
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
        self.enable_goal_detection = self.get_parameter('enable_goal_detection').value
        self.enable_robot_detection = self.get_parameter('enable_robot_detection').value
        
        self.fallback_to_traditional = self.get_parameter('fallback_to_traditional').value and TRADITIONAL_DETECTORS_AVAILABLE
        
        # Carregar configuração
        self.config = self._load_config()
        
        # Inicializar o manipulador YOLO
        self.yoeo_handler = YOEOHandler(
            config={
                "model_path": self.model_path,
                "input_width": self.input_width,
                "input_height": self.input_height,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "use_tensorrt": self.use_tensorrt
            }
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
        
        self.get_logger().info('Nó YOLO Detector inicializado')
    
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
        """Inicializa os componentes de detecção."""
        self.components = {}
        
        # Componente de detecção de bola
        if self.enable_ball_detection:
            self.components['ball'] = BallDetectionComponent(
                yoeo_handler=self.yoeo_handler,
                ball_diameter=self.config.get('ball', {}).get('diameter', 0.043)
            )
            self.get_logger().info('Componente de detecção de bola inicializado')
        
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
        
        # Configurar detectores tradicionais para fallback, se disponíveis
        if self.fallback_to_traditional:
            self.setup_traditional_detectors()
    
    def setup_traditional_detectors(self):
        """Configura detectores tradicionais para fallback."""
        try:
            if 'ball' in self.components and TRADITIONAL_DETECTORS_AVAILABLE:
                ball_detector = BallDetector()
                self.components['ball'].set_fallback_detector(ball_detector)
                self.get_logger().info('Detector tradicional de bola configurado como fallback')
                
            if 'goal' in self.components and TRADITIONAL_DETECTORS_AVAILABLE:
                goal_detector = GoalDetector()
                self.components['goal'].set_fallback_detector(goal_detector)
                self.get_logger().info('Detector tradicional de gol configurado como fallback')
                
            if 'robot' in self.components and TRADITIONAL_DETECTORS_AVAILABLE:
                robot_detector = ObstacleDetector()
                self.components['robot'].set_fallback_detector(robot_detector)
                self.get_logger().info('Detector tradicional de robô configurado como fallback')
        except Exception as e:
            self.get_logger().error(f'Erro ao configurar detectores tradicionais: {e}')
    
    def _init_publishers(self):
        """Inicializa os publishers ROS."""
        # Publishers para resultados de detecção
        self.detection_publishers = {}
        
        if self.enable_ball_detection:
            self.detection_publishers[DetectionType.BALL] = self.create_publisher(
                Detection2DArray, 
                'yolo/ball_detections', 
                10
            )
        
        if self.enable_goal_detection:
            self.detection_publishers[DetectionType.GOAL] = self.create_publisher(
                Detection2DArray, 
                'yolo/goal_detections', 
                10
            )
        
        if self.enable_robot_detection:
            self.detection_publishers[DetectionType.ROBOT] = self.create_publisher(
                Detection2DArray, 
                'yolo/robot_detections', 
                10
            )
        
        # Publisher para imagem de debug
        if self.debug_image:
            self.debug_image_publisher = self.create_publisher(
                Image, 
                'yolo/debug_image', 
                10
            )
    
    def camera_info_callback(self, msg):
        """Processa mensagens de informação da câmera."""
        self.camera_info = msg
        self.get_logger().debug('Recebidas informações da câmera')
    
    def image_callback(self, msg):
        """Processa imagens recebidas da câmera."""
        try:
            # Converter a mensagem ROS para formato OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Iniciar tempo para cálculo de FPS
            start_time = time.time()
            
            # Processar a imagem com o modelo YOLO
            results = self._process_image(cv_image)
            
            # Calcular FPS
            self.frame_count += 1
            elapsed_time = time.time() - start_time
            self.total_time += elapsed_time
            self.fps = self.frame_count / self.total_time
            
            if self.frame_count % 30 == 0:
                self.get_logger().info(f'FPS: {self.fps:.2f}')
            
            # Publicar resultados
            self._publish_results(results, msg.header)
            
        except CvBridgeError as e:
            self.get_logger().error(f'Erro na conversão da imagem: {e}')
        except Exception as e:
            self.get_logger().error(f'Erro no processamento: {e}')
    
    def _process_image(self, image):
        """
        Processa a imagem com o modelo YOLO.
        
        Args:
            image: Imagem no formato OpenCV (BGR)
        
        Returns:
            Dicionário com os resultados para cada tipo de detecção
        """
        results = {}
        debug_image = image.copy() if self.debug_image else None
        
        try:
            # Processar a imagem com YOLOv5
            detections = self.yoeo_handler.detect(image)
            
            # Processar cada tipo de detecção com seu componente específico
            if self.enable_ball_detection and 'ball' in self.components:
                ball_results = self.components['ball'].process(image, detections)
                results[DetectionType.BALL] = ball_results
                
                # Desenhar resultados na imagem de debug
                if debug_image is not None and ball_results:
                    for box, conf, cls in zip(ball_results['boxes'], ball_results['confidences'], ball_results['classes']):
                        x1, y1, x2, y2 = box
                        cv2.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(debug_image, f'Bola: {conf:.2f}', (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if self.enable_goal_detection and 'goal' in self.components:
                goal_results = self.components['goal'].process(image, detections)
                results[DetectionType.GOAL] = goal_results
                
                # Desenhar resultados na imagem de debug
                if debug_image is not None and goal_results:
                    for box, conf, cls in zip(goal_results['boxes'], goal_results['confidences'], goal_results['classes']):
                        x1, y1, x2, y2 = box
                        cv2.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(debug_image, f'Gol: {conf:.2f}', (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            if self.enable_robot_detection and 'robot' in self.components:
                robot_results = self.components['robot'].process(image, detections)
                results[DetectionType.ROBOT] = robot_results
                
                # Desenhar resultados na imagem de debug
                if debug_image is not None and robot_results:
                    for box, conf, cls in zip(robot_results['boxes'], robot_results['confidences'], robot_results['classes']):
                        x1, y1, x2, y2 = box
                        cv2.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(debug_image, f'Robô: {conf:.2f}', (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Adicionar informações de FPS à imagem de debug
            if debug_image is not None:
                cv2.putText(debug_image, f'FPS: {self.fps:.2f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
        except Exception as e:
            self.get_logger().error(f'Erro no processamento da imagem: {e}')
        
        # Salvar imagem de debug para publicação
        if debug_image is not None:
            results['debug_image'] = debug_image
        
        return results
    
    def _publish_results(self, results, header):
        """
        Publica os resultados de detecção nos tópicos ROS.
        
        Args:
            results: Dicionário com os resultados para cada tipo de detecção
            header: Header da mensagem original
        """
        # Publicar detecções de bola
        if DetectionType.BALL in results and DetectionType.BALL in self.detection_publishers:
            ball_msg = self._create_detection_msg(results[DetectionType.BALL], header, DetectionType.BALL)
            self.detection_publishers[DetectionType.BALL].publish(ball_msg)
        
        # Publicar detecções de gol
        if DetectionType.GOAL in results and DetectionType.GOAL in self.detection_publishers:
            goal_msg = self._create_detection_msg(results[DetectionType.GOAL], header, DetectionType.GOAL)
            self.detection_publishers[DetectionType.GOAL].publish(goal_msg)
        
        # Publicar detecções de robô
        if DetectionType.ROBOT in results and DetectionType.ROBOT in self.detection_publishers:
            robot_msg = self._create_detection_msg(results[DetectionType.ROBOT], header, DetectionType.ROBOT)
            self.detection_publishers[DetectionType.ROBOT].publish(robot_msg)
        
        # Publicar imagem de debug
        if self.debug_image and 'debug_image' in results:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(results['debug_image'], encoding='bgr8')
                debug_msg.header = header
                self.debug_image_publisher.publish(debug_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Erro ao converter imagem de debug: {e}')
    
    def _create_detection_msg(self, detections, header, detection_type):
        """
        Cria uma mensagem de detecção para publicação no ROS.
        
        Args:
            detections: Resultados de detecção
            header: Header da mensagem original
            detection_type: Tipo de detecção (BALL, GOAL, ROBOT)
        
        Returns:
            Mensagem de detecção formatada
        """
        detection_array = Detection2DArray()
        detection_array.header = header
        
        if not detections:
            return detection_array
        
        boxes = detections.get('boxes', [])
        confidences = detections.get('confidences', [])
        classes = detections.get('classes', [])
        
        for i, (box, confidence, cls) in enumerate(zip(boxes, confidences, classes)):
            detection = Detection2D()
            detection.header = header
            
            # ID da detecção
            detection.id = f"{detection_type.name.lower()}_{i}"
            
            # Posição central e tamanho da caixa
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Definir a posição central da detecção
            detection.bbox.center.position.x = float(center_x)
            detection.bbox.center.position.y = float(center_y)
            detection.bbox.center.theta = 0.0  # Orientação padrão
            
            # Definir as dimensões da caixa
            detection.bbox.size_x = float(width)
            detection.bbox.size_y = float(height)
            
            # Adicionar hipótese de objeto
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(cls)  # ID da classe
            hypothesis.score = float(confidence)
            
            # Pose do objeto (preenchida opcionalmente pelo componente específico)
            pose = PoseWithCovariance()
            pose.pose = Pose()
            
            # Se o componente forneceu uma estimativa de pose, usar
            if 'poses' in detections and i < len(detections['poses']):
                pose.pose = detections['poses'][i]
            
            hypothesis.pose = pose
            detection.results.append(hypothesis)
            
            detection_array.detections.append(detection)
        
        return detection_array


def main(args=None):
    """Função principal para iniciar o nó ROS."""
    rclpy.init(args=args)
    yolo_detector = YOEODetector()
    
    try:
        rclpy.spin(yolo_detector)
    except KeyboardInterrupt:
        pass
    finally:
        yolo_detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 