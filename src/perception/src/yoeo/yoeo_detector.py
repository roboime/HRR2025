#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D, PoseArray, Pose

from src.perception.src.yoeo.yoeo_handler import YOEOHandler, DetectionType, SegmentationType
from src.perception.src.yoeo.components.ball_component import BallDetectionComponent
from src.perception.src.yoeo.components.field_component import FieldSegmentationComponent
from src.perception.src.yoeo.components.line_component import LineSegmentationComponent
from src.perception.src.yoeo.components.goal_component import GoalDetectionComponent
from src.perception.src.yoeo.components.robot_component import RobotDetectionComponent
from src.perception.src.yoeo.components.referee_component import RefereeDetectionComponent

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
    Nó ROS para detecção e segmentação usando YOEO.
    
    Este nó integra todos os componentes YOEO e fornece uma interface ROS
    para publicar detecções de bola, campo, linhas, gols, robôs e árbitros.
    Também suporta fallback para detectores tradicionais quando necessário.
    """
    
    def __init__(self):
        """Inicializa o nó YOEO Detector."""
        super().__init__('yoeo_detector')
        
        # Declarar parâmetros
        self.declare_parameter('model_path', 'resource/models/yoeo_model.h5')
        self.declare_parameter('input_width', 416)
        self.declare_parameter('input_height', 416)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('use_tensorrt', False)
        self.declare_parameter('debug_image', True)
        
        # Parâmetros para ativar/desativar componentes
        self.declare_parameter('enable_ball_detection', True)
        self.declare_parameter('enable_field_detection', True)
        self.declare_parameter('enable_line_detection', False)
        self.declare_parameter('enable_goal_detection', False)
        self.declare_parameter('enable_robot_detection', False)
        self.declare_parameter('enable_referee_detection', False)
        
        # Parâmetro para fallback para detectores tradicionais
        self.declare_parameter('fallback_to_traditional', True)
        
        # Obter parâmetros
        self.model_path = self.get_parameter('model_path').value
        self.input_width = self.get_parameter('input_width').value
        self.input_height = self.get_parameter('input_height').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.use_tensorrt = self.get_parameter('use_tensorrt').value
        self.debug_image = self.get_parameter('debug_image').value
        
        self.enable_ball = self.get_parameter('enable_ball_detection').value
        self.enable_field = self.get_parameter('enable_field_detection').value
        self.enable_line = self.get_parameter('enable_line_detection').value
        self.enable_goal = self.get_parameter('enable_goal_detection').value
        self.enable_robot = self.get_parameter('enable_robot_detection').value
        self.enable_referee = self.get_parameter('enable_referee_detection').value
        
        self.fallback_to_traditional = self.get_parameter('fallback_to_traditional').value and TRADITIONAL_DETECTORS_AVAILABLE
        
        # Inicializar bridge para conversão entre ROS e OpenCV
        self.cv_bridge = CvBridge()
        
        # Criar o manipulador YOEO
        self.yoeo_handler = YOEOHandler(
            model_path=self.model_path,
            input_width=self.input_width,
            input_height=self.input_height,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            use_tensorrt=self.use_tensorrt
        )
        
        # Inicializar componentes YOEO
        self.components = {}
        
        if self.enable_ball:
            self.components['ball'] = BallDetectionComponent(self.yoeo_handler)
        
        if self.enable_field:
            self.components['field'] = FieldSegmentationComponent(self.yoeo_handler)
        
        if self.enable_line:
            self.components['line'] = LineSegmentationComponent(self.yoeo_handler)
        
        if self.enable_goal:
            self.components['goal'] = GoalDetectionComponent(self.yoeo_handler)
        
        if self.enable_robot:
            self.components['robot'] = RobotDetectionComponent(self.yoeo_handler)
        
        if self.enable_referee:
            self.components['referee'] = RefereeDetectionComponent(self.yoeo_handler)
        
        # Configurar detectores tradicionais para fallback se solicitado
        if self.fallback_to_traditional:
            self.setup_traditional_detectors()
        
        # Publishers
        self.publishers = {}
        
        if self.enable_ball:
            self.publishers['ball'] = self.create_publisher(Pose2D, 'ball_position', 10)
        
        if self.enable_field:
            self.publishers['field_mask'] = self.create_publisher(Image, 'field_mask', 10)
            self.publishers['field_boundary'] = self.create_publisher(Image, 'field_boundary', 10)
        
        if self.enable_line:
            self.publishers['line'] = self.create_publisher(Image, 'lines_image', 10)
        
        if self.enable_goal:
            self.publishers['goal'] = self.create_publisher(PoseArray, 'goal_posts', 10)
        
        if self.enable_robot:
            self.publishers['robot'] = self.create_publisher(PoseArray, 'robots', 10)
        
        if self.enable_referee:
            self.publishers['referee'] = self.create_publisher(Pose, 'referee_position', 10)
        
        # Publisher para imagem de debug
        if self.debug_image:
            self.publishers['debug'] = self.create_publisher(Image, 'yoeo_detection_debug', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            'camera/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Variáveis
        self.camera_info = None
        
        self.get_logger().info('Nó detector YOEO iniciado')
    
    def setup_traditional_detectors(self):
        """Configura detectores tradicionais para fallback."""
        if not TRADITIONAL_DETECTORS_AVAILABLE:
            self.get_logger().warn('Detectores tradicionais não disponíveis para fallback')
            return
        
        # Ball detector
        if self.enable_ball and 'ball' in self.components:
            traditional_ball_detector = BallDetector()
            self.components['ball'].set_fallback_detector(traditional_ball_detector)
            self.get_logger().info('Detector tradicional de bola configurado para fallback')
        
        # Field detector
        if self.enable_field and 'field' in self.components:
            traditional_field_detector = FieldDetector()
            self.components['field'].set_fallback_detector(traditional_field_detector)
            self.get_logger().info('Detector tradicional de campo configurado para fallback')
        
        # Line detector
        if self.enable_line and 'line' in self.components:
            traditional_line_detector = LineDetector()
            self.components['line'].set_fallback_detector(traditional_line_detector)
            self.get_logger().info('Detector tradicional de linhas configurado para fallback')
        
        # Goal detector
        if self.enable_goal and 'goal' in self.components:
            traditional_goal_detector = GoalDetector()
            self.components['goal'].set_fallback_detector(traditional_goal_detector)
            self.get_logger().info('Detector tradicional de gols configurado para fallback')
        
        # Robot detector (usando ObstacleDetector)
        if self.enable_robot and 'robot' in self.components:
            traditional_robot_detector = ObstacleDetector()
            self.components['robot'].set_fallback_detector(traditional_robot_detector)
            self.get_logger().info('Detector tradicional de robôs configurado para fallback')
    
    def camera_info_callback(self, msg):
        """Callback para informações da câmera."""
        self.camera_info = msg
        
        # Atualizar informações da câmera nos componentes
        for component_name, component in self.components.items():
            if hasattr(component, 'set_camera_info'):
                component.set_camera_info(msg)
    
    def image_callback(self, msg):
        """Callback para processamento de imagem."""
        if self.camera_info is None:
            self.get_logger().warn('Informações da câmera ainda não recebidas')
            return
        
        try:
            # Converter ROS Image para OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Criar imagem de debug se necessário
            if self.debug_image:
                debug_image = cv_image.copy()
            
            # Processar imagem com cada componente ativo
            results = {}
            
            # Processar campo (geralmente primeiro, pois outros componentes podem usar a máscara)
            if 'field' in self.components:
                results['field'] = self.components['field'].process(cv_image)
                self.publish_field_results(results['field'], msg.header)
                
                if self.debug_image:
                    debug_image = self.components['field'].draw_segmentation(debug_image, results['field'])
            
            # Processar linhas
            if 'line' in self.components:
                results['line'] = self.components['line'].process(cv_image)
                self.publish_line_results(results['line'], msg.header)
                
                if self.debug_image:
                    debug_image = self.components['line'].draw_segmentation(debug_image, results['line'])
            
            # Processar bola
            if 'ball' in self.components:
                results['ball'] = self.components['ball'].process(cv_image)
                self.publish_ball_results(results['ball'], msg.header)
                
                if self.debug_image:
                    debug_image = self.components['ball'].draw_detections(debug_image, results['ball'])
            
            # Processar gols
            if 'goal' in self.components:
                results['goal'] = self.components['goal'].process(cv_image)
                self.publish_goal_results(results['goal'], msg.header)
                
                if self.debug_image:
                    debug_image = self.components['goal'].draw_detections(debug_image, results['goal'])
            
            # Processar robôs
            if 'robot' in self.components:
                results['robot'] = self.components['robot'].process(cv_image)
                self.publish_robot_results(results['robot'], msg.header)
                
                if self.debug_image:
                    debug_image = self.components['robot'].draw_detections(debug_image, results['robot'])
            
            # Processar árbitro
            if 'referee' in self.components:
                results['referee'] = self.components['referee'].process(cv_image)
                self.publish_referee_results(results['referee'], msg.header)
                
                if self.debug_image:
                    debug_image = self.components['referee'].draw_detections(debug_image, results['referee'])
            
            # Publicar imagem de debug
            if self.debug_image:
                debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, 'bgr8')
                debug_msg.header = msg.header
                self.publishers['debug'].publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f'Erro no processamento da imagem: {str(e)}')
    
    def publish_ball_results(self, ball_detections, header):
        """Publica os resultados da detecção de bola."""
        if not ball_detections:
            return
        
        # Publicar apenas a bola com maior confiança
        if ball_detections:
            # Ordenar por confiança (decrescente)
            sorted_balls = sorted(ball_detections, key=lambda x: x.get('confidence', 0), reverse=True)
            best_ball = sorted_balls[0]
            
            # Converter para mensagem ROS
            ball_msg = self.components['ball'].to_ros_message(best_ball)
            self.publishers['ball'].publish(ball_msg)
    
    def publish_field_results(self, field_result, header):
        """Publica os resultados da segmentação do campo."""
        if not field_result:
            return
        
        # Publicar máscara do campo
        if 'mask' in field_result and field_result['mask'] is not None:
            mask_msg = self.cv_bridge.cv2_to_imgmsg(field_result['mask'], 'mono8')
            mask_msg.header = header
            self.publishers['field_mask'].publish(mask_msg)
        
        # Publicar fronteira do campo
        if 'boundary' in field_result and field_result['boundary'] is not None:
            boundary_msg = self.cv_bridge.cv2_to_imgmsg(field_result['boundary'], 'mono8')
            boundary_msg.header = header
            self.publishers['field_boundary'].publish(boundary_msg)
    
    def publish_line_results(self, line_result, header):
        """Publica os resultados da segmentação de linhas."""
        if not line_result or 'mask' not in line_result or line_result['mask'] is None:
            return
        
        # Publicar máscara de linhas
        line_msg = self.cv_bridge.cv2_to_imgmsg(line_result['mask'], 'mono8')
        line_msg.header = header
        self.publishers['line'].publish(line_msg)
    
    def publish_goal_results(self, goal_detections, header):
        """Publica os resultados da detecção de gols."""
        if not goal_detections:
            return
        
        # Criar PoseArray para postes de gol
        pose_array = PoseArray()
        pose_array.header = header
        
        for goal in goal_detections:
            if 'position' in goal and goal['position'] is not None:
                pose = Pose()
                x, y, z = goal['position']
                pose.position.x = z  # Distância frontal
                pose.position.y = -x  # Distância lateral
                pose.position.z = 0.0
                pose_array.poses.append(pose)
        
        if pose_array.poses:
            self.publishers['goal'].publish(pose_array)
    
    def publish_robot_results(self, robot_detections, header):
        """Publica os resultados da detecção de robôs."""
        if not robot_detections:
            return
        
        # Criar PoseArray para robôs
        pose_array = PoseArray()
        pose_array.header = header
        
        for robot in robot_detections:
            if 'position' in robot and robot['position'] is not None:
                pose = Pose()
                x, y, z = robot['position']
                pose.position.x = z  # Distância frontal
                pose.position.y = -x  # Distância lateral
                pose.position.z = 0.0
                pose_array.poses.append(pose)
        
        if pose_array.poses:
            self.publishers['robot'].publish(pose_array)
    
    def publish_referee_results(self, referee_detection, header):
        """Publica os resultados da detecção de árbitro."""
        if not referee_detection or not referee_detection[0]:
            return
        
        # Usar apenas a detecção de árbitro mais confiável
        referee = referee_detection[0]
        
        if 'position' in referee and referee['position'] is not None:
            pose = Pose()
            x, y, z = referee['position']
            pose.position.x = z  # Distância frontal
            pose.position.y = -x  # Distância lateral
            pose.position.z = 0.0
            self.publishers['referee'].publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = YOEODetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main() 