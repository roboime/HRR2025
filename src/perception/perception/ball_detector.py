#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D
import math

class BallDetector(Node):
    """
    Nó para detecção da bola de futebol usando visão computacional.
    
    Este nó processa imagens da câmera para detectar a bola de futebol
    e publica sua posição relativa ao robô.
    """
    
    def __init__(self):
        super().__init__('ball_detector')
        
        # Parâmetros
        self.declare_parameter('ball_color_lower', [0, 120, 70])  # HSV para laranja (bola)
        self.declare_parameter('ball_color_upper', [10, 255, 255])
        self.declare_parameter('min_ball_radius', 10)
        self.declare_parameter('debug_image', True)
        
        # Obter parâmetros
        self.ball_color_lower = np.array(self.get_parameter('ball_color_lower').value)
        self.ball_color_upper = np.array(self.get_parameter('ball_color_upper').value)
        self.min_ball_radius = self.get_parameter('min_ball_radius').value
        self.debug_image = self.get_parameter('debug_image').value
        
        # Publishers
        self.ball_position_pub = self.create_publisher(Pose2D, 'ball_position', 10)
        self.debug_image_pub = self.create_publisher(Image, 'ball_detection_debug', 10)
        
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
        self.cv_bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.ball_diameter = 0.14  # Diâmetro da bola em metros (padrão RoboCup)
        
        self.get_logger().info('Nó detector de bola iniciado')
    
    def camera_info_callback(self, msg):
        """Callback para informações da câmera."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
    
    def image_callback(self, msg):
        """Callback para processamento de imagem."""
        if self.camera_matrix is None:
            self.get_logger().warn('Informações da câmera ainda não recebidas')
            return
        
        try:
            # Converter ROS Image para OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Detectar a bola
            ball_position, debug_image = self.detect_ball(cv_image)
            
            # Publicar a posição da bola se encontrada
            if ball_position is not None:
                ball_msg = Pose2D()
                ball_msg.x = ball_position[0]
                ball_msg.y = ball_position[1]
                ball_msg.theta = 0.0  # A bola não tem orientação
                self.ball_position_pub.publish(ball_msg)
            
            # Publicar imagem de debug se necessário
            if self.debug_image:
                debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, 'bgr8')
                self.debug_image_pub.publish(debug_msg)
                
        except Exception as e:
            self.get_logger().error(f'Erro no processamento da imagem: {str(e)}')
    
    def detect_ball(self, image):
        """
        Detecta a bola na imagem usando segmentação por cor e transformada de Hough.
        
        Args:
            image: Imagem OpenCV no formato BGR
            
        Returns:
            tuple: (posição_da_bola, imagem_debug)
                posição_da_bola: (x, y) em metros relativos ao robô, ou None se não encontrada
                imagem_debug: Imagem OpenCV com marcações de debug
        """
        # Criar cópia da imagem para debug
        debug_image = image.copy()
        
        # Converter para HSV para melhor segmentação de cor
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Criar máscara para a cor da bola
        mask = cv2.inRange(hsv, self.ball_color_lower, self.ball_color_upper)
        
        # Aplicar operações morfológicas para remover ruído
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Procurar o maior contorno circular
        ball_position = None
        max_radius = 0
        
        for contour in contours:
            # Calcular área e perímetro
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Verificar se o contorno é aproximadamente circular
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                
                if circularity > 0.7:  # Valor próximo de 1 indica círculo perfeito
                    # Encontrar o círculo que melhor se ajusta ao contorno
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    if radius > self.min_ball_radius and radius > max_radius:
                        max_radius = radius
                        
                        # Converter de pixels para coordenadas do mundo
                        # Implementação simplificada - assumindo que a câmera está calibrada
                        # e que conhecemos a distância focal
                        focal_length = self.camera_matrix[0, 0]  # Distância focal em pixels
                        distance = (self.ball_diameter * focal_length) / (2 * radius)
                        
                        # Calcular coordenadas X e Y no plano do chão
                        # Assumindo que a origem está no centro da imagem
                        center_x = image.shape[1] / 2
                        center_y = image.shape[0] / 2
                        
                        # Converter de pixels para metros
                        x_meters = ((x - center_x) / focal_length) * distance
                        y_meters = distance  # Distância frontal
                        
                        ball_position = (x_meters, y_meters)
                        
                        # Desenhar círculo na imagem de debug
                        cv2.circle(debug_image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                        cv2.putText(debug_image, f'Dist: {distance:.2f}m', (int(x), int(y) - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Desenhar informações na imagem de debug
        if ball_position is not None:
            cv2.putText(debug_image, f'Bola: ({ball_position[0]:.2f}, {ball_position[1]:.2f})',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(debug_image, 'Bola não encontrada', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return ball_position, debug_image

def main(args=None):
    rclpy.init(args=args)
    node = BallDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 