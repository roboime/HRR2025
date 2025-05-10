#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nó ROS para detecção usando o modelo YOLO da Ultralytics.

Este nó implementa a interface ROS para o sistema YOLO da Ultralytics, recebendo imagens
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
    Nó ROS para detecção usando o modelo YOLO da Ultralytics.
    
    Este nó recebe imagens da câmera, processa-as com o modelo YOLO
    e publica os resultados de detecção.
    """
    
    def __init__(self):
        """Inicializa o nó detector YOLO."""
        super().__init__('yolo_detector')
        
        # Declarar parâmetros
        self.declare_parameter('model_path', 'src/perception/resource/models/yolov5n.pt')
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
        
        self.get_logger().info('Nó YOLO Detector (Ultralytics) inicializado')
    
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
        # Publishers para detecções
        self.detection_publishers = {}
        
        if self.enable_ball_detection:
            self.detection_publishers[DetectionType.BALL] = self.create_publisher(
                Detection2DArray,
                'vision/ball_detections',
                10
            )
        
        if self.enable_goal_detection:
            self.detection_publishers[DetectionType.GOAL] = self.create_publisher(
                Detection2DArray,
                'vision/goal_detections',
                10
            )
        
        if self.enable_robot_detection:
            self.detection_publishers[DetectionType.ROBOT] = self.create_publisher(
                Detection2DArray,
                'vision/robot_detections',
                10
            )
        
        # Publisher para imagem de debug
        if self.debug_image:
            self.debug_publisher = self.create_publisher(
                Image,
                'vision/debug_image',
                10
            )
    
    def camera_info_callback(self, msg):
        """Callback para informações da câmera."""
        self.camera_info = msg
        
        # Atualizar informações da câmera nos componentes
        for _, component in self.components.items():
            if hasattr(component, 'set_camera_info'):
                component.set_camera_info(msg)
    
    def image_callback(self, msg):
        """Callback para imagens da câmera."""
        try:
            # Converter a mensagem ROS para imagem OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Processar a imagem
            start_time = time.time()
            results = self._process_image(cv_image)
            processing_time = time.time() - start_time
            
            # Atualizar estatísticas
            self.frame_count += 1
            self.total_time += processing_time
            self.fps = self.frame_count / self.total_time
            
            # Publicar resultados
            self._publish_results(results, msg.header)
            
        except CvBridgeError as e:
            self.get_logger().error(f'Erro ao converter imagem: {e}')
        except Exception as e:
            self.get_logger().error(f'Erro ao processar imagem: {e}')
    
    def _process_image(self, image):
        """
        Processa a imagem com o modelo YOLO e componentes.
        
        Args:
            image: Imagem BGR do OpenCV
            
        Returns:
            Dicionário com resultados de detecção
        """
        # Obter detecções do modelo YOLO
        detection_types = []
        
        if self.enable_ball_detection:
            detection_types.append(DetectionType.BALL)
        
        if self.enable_goal_detection:
            detection_types.append(DetectionType.GOAL)
        
        if self.enable_robot_detection:
            detection_types.append(DetectionType.ROBOT)
        
        results = self.yoeo_handler.get_detections(image, detection_types=detection_types)
        
        # Processar resultados com componentes
        processed_results = {
            'ball_detections': [],
            'goal_detections': [],
            'robot_detections': [],
            'debug_image': None
        }
        
        # Processar bolas
        if 'ball' in self.components:
            processed_results['ball_detections'] = self.components['ball'].process(image)
        
        # Processar gols
        if 'goal' in self.components:
            processed_results['goal_detections'] = self.components['goal'].process(image)
        
        # Processar robôs
        if 'robot' in self.components:
            processed_results['robot_detections'] = self.components['robot'].process(image)
        
        # Gerar imagem de debug
        if self.debug_image:
            debug_image = image.copy()
            
            # Desenhar detecções de cada componente
            if 'ball' in self.components and processed_results['ball_detections']:
                debug_image = self.components['ball'].draw_detections(
                    debug_image, processed_results['ball_detections']
                )
            
            if 'goal' in self.components and processed_results['goal_detections']:
                debug_image = self.components['goal'].draw_detections(
                    debug_image, processed_results['goal_detections']
                )
            
            if 'robot' in self.components and processed_results['robot_detections']:
                debug_image = self.components['robot'].draw_detections(
                    debug_image, processed_results['robot_detections']
                )
            
            # Adicionar informações de FPS
            cv2.putText(
                debug_image,
                f"FPS: {self.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            processed_results['debug_image'] = debug_image
        
        return processed_results
    
    def _publish_results(self, results, header):
        """
        Publica os resultados nos tópicos ROS.
        
        Args:
            results: Dicionário com resultados processados
            header: Cabeçalho da mensagem original
        """
        # Publicar detecções de bola
        if DetectionType.BALL in self.detection_publishers and results['ball_detections']:
            ball_msg = self._create_detection_msg(
                results['ball_detections'], header, DetectionType.BALL
            )
            self.detection_publishers[DetectionType.BALL].publish(ball_msg)
        
        # Publicar detecções de gol
        if DetectionType.GOAL in self.detection_publishers and results['goal_detections']:
            goal_msg = self._create_detection_msg(
                results['goal_detections'], header, DetectionType.GOAL
            )
            self.detection_publishers[DetectionType.GOAL].publish(goal_msg)
        
        # Publicar detecções de robô
        if DetectionType.ROBOT in self.detection_publishers and results['robot_detections']:
            robot_msg = self._create_detection_msg(
                results['robot_detections'], header, DetectionType.ROBOT
            )
            self.detection_publishers[DetectionType.ROBOT].publish(robot_msg)
        
        # Publicar imagem de debug
        if self.debug_image and results['debug_image'] is not None:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(results['debug_image'], "bgr8")
                debug_msg.header = header
                self.debug_publisher.publish(debug_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Erro ao converter imagem de debug: {e}')
    
    def _create_detection_msg(self, detections, header, detection_type):
        """
        Cria uma mensagem Detection2DArray para publicação.
        
        Args:
            detections: Lista de detecções
            header: Cabeçalho da mensagem
            detection_type: Tipo de detecção
            
        Returns:
            Mensagem Detection2DArray
        """
        msg = Detection2DArray()
        msg.header = header
        
        for detection in detections:
            det_msg = Detection2D()
            det_msg.header = header
            
            # Extrair coordenadas da caixa delimitadora
            x1, y1, x2, y2 = detection['bbox']
            
            # Calcular centro e tamanho
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Definir posição 2D
            det_msg.bbox.center.position.x = center_x
            det_msg.bbox.center.position.y = center_y
            det_msg.bbox.center.position.z = 0.0
            
            # Definir orientação (identidade)
            det_msg.bbox.center.orientation.w = 1.0
            
            # Definir tamanho
            det_msg.bbox.size_x = width
            det_msg.bbox.size_y = height
            
            # Adicionar hipótese de objeto
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = str(detection_type.value)
            hypothesis.score = float(detection['confidence'])
            
            # Adicionar posição 3D, se disponível
            if 'position_3d' in detection and detection['position_3d'] is not None:
                x, y, z = detection['position_3d']
                
                pose = PoseWithCovariance()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z
                pose.pose.orientation.w = 1.0
                
                hypothesis.pose = pose
            
            det_msg.results.append(hypothesis)
            msg.detections.append(det_msg)
        
        return msg

def main(args=None):
    """Função principal para iniciar o nó detector YOLO."""
    rclpy.init(args=args)
    
    try:
        node = YOEODetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Erro no nó detector YOLO: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main() 