#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D
import threading
import time

class VisionPipeline(Node):
    """
    Pipeline principal de visão computacional para o robô de futebol.
    
    Este nó coordena os diferentes detectores (bola, campo, linhas, gols, obstáculos)
    e gerencia o fluxo de processamento de imagens.
    """
    
    def __init__(self):
        super().__init__('vision_pipeline')
        
        # Parâmetros
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('debug_image', True)
        self.declare_parameter('enable_ball_detection', True)
        self.declare_parameter('enable_field_detection', True)
        self.declare_parameter('enable_line_detection', False)
        self.declare_parameter('enable_goal_detection', False)
        self.declare_parameter('enable_obstacle_detection', False)
        self.declare_parameter('processing_fps', 30.0)
        
        # Obter parâmetros
        self.camera_topic = self.get_parameter('camera_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.debug_image = self.get_parameter('debug_image').value
        self.enable_ball_detection = self.get_parameter('enable_ball_detection').value
        self.enable_field_detection = self.get_parameter('enable_field_detection').value
        self.enable_line_detection = self.get_parameter('enable_line_detection').value
        self.enable_goal_detection = self.get_parameter('enable_goal_detection').value
        self.enable_obstacle_detection = self.get_parameter('enable_obstacle_detection').value
        self.processing_fps = self.get_parameter('processing_fps').value
        
        # Publishers
        self.debug_image_pub = self.create_publisher(Image, 'vision_debug', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        # Variáveis
        self.cv_bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_image = None
        self.latest_image_time = None
        self.processing_thread = None
        self.processing_active = False
        
        # Iniciar thread de processamento
        self.start_processing_thread()
        
        self.get_logger().info('Pipeline de visão iniciado')
    
    def camera_info_callback(self, msg):
        """Callback para informações da câmera."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
    
    def image_callback(self, msg):
        """Callback para receber imagens da câmera."""
        # Apenas armazena a imagem mais recente para processamento na thread separada
        self.latest_image = msg
        self.latest_image_time = self.get_clock().now()
    
    def start_processing_thread(self):
        """Inicia a thread de processamento de imagens."""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            return
        
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_images(self):
        """Thread principal de processamento de imagens."""
        period = 1.0 / self.processing_fps
        
        while self.processing_active and rclpy.ok():
            start_time = time.time()
            
            # Processar a imagem mais recente
            if self.latest_image is not None and self.camera_matrix is not None:
                try:
                    # Converter ROS Image para OpenCV
                    cv_image = self.cv_bridge.imgmsg_to_cv2(self.latest_image, 'bgr8')
                    
                    # Criar imagem de debug
                    debug_image = cv_image.copy()
                    
                    # Executar detectores habilitados
                    results = {}
                    
                    if self.enable_field_detection:
                        # Implementar detecção de campo
                        # results['field'] = self.detect_field(cv_image, debug_image)
                        pass
                    
                    if self.enable_ball_detection:
                        # A detecção de bola já está implementada em outro nó
                        # Aqui podemos adicionar código para visualização
                        pass
                    
                    if self.enable_line_detection:
                        # Implementar detecção de linhas
                        # results['lines'] = self.detect_lines(cv_image, debug_image)
                        pass
                    
                    if self.enable_goal_detection:
                        # Implementar detecção de gols
                        # results['goals'] = self.detect_goals(cv_image, debug_image)
                        pass
                    
                    if self.enable_obstacle_detection:
                        # Implementar detecção de obstáculos
                        # results['obstacles'] = self.detect_obstacles(cv_image, debug_image)
                        pass
                    
                    # Publicar imagem de debug
                    if self.debug_image:
                        # Adicionar informações de timestamp e FPS
                        fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                        cv2.putText(debug_image, f"FPS: {fps:.1f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, 'bgr8')
                        debug_msg.header = self.latest_image.header
                        self.debug_image_pub.publish(debug_msg)
                
                except Exception as e:
                    self.get_logger().error(f'Erro no processamento da imagem: {str(e)}')
            
            # Controlar a taxa de processamento
            elapsed = time.time() - start_time
            if elapsed < period:
                time.sleep(period - elapsed)
    
    def detect_field(self, image, debug_image):
        """
        Detecta o campo na imagem.
        
        Args:
            image: Imagem OpenCV no formato BGR
            debug_image: Imagem para desenhar informações de debug
            
        Returns:
            dict: Resultados da detecção do campo
        """
        # Implementação será adicionada posteriormente
        return {}
    
    def detect_lines(self, image, debug_image):
        """
        Detecta as linhas do campo na imagem.
        
        Args:
            image: Imagem OpenCV no formato BGR
            debug_image: Imagem para desenhar informações de debug
            
        Returns:
            dict: Resultados da detecção de linhas
        """
        # Implementação será adicionada posteriormente
        return {}
    
    def detect_goals(self, image, debug_image):
        """
        Detecta os gols na imagem.
        
        Args:
            image: Imagem OpenCV no formato BGR
            debug_image: Imagem para desenhar informações de debug
            
        Returns:
            dict: Resultados da detecção de gols
        """
        # Implementação será adicionada posteriormente
        return {}
    
    def detect_obstacles(self, image, debug_image):
        """
        Detecta obstáculos na imagem (outros robôs, árbitros, etc).
        
        Args:
            image: Imagem OpenCV no formato BGR
            debug_image: Imagem para desenhar informações de debug
            
        Returns:
            dict: Resultados da detecção de obstáculos
        """
        # Implementação será adicionada posteriormente
        return {}

def main(args=None):
    rclpy.init(args=args)
    node = VisionPipeline()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
