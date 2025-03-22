#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
import math

class GoalDetector(Node):
    """
    Nó para detecção dos gols de futebol usando visão computacional.
    
    Este nó processa imagens da câmera para detectar os postes do gol
    e publicar suas posições relativas ao robô.
    """
    
    def __init__(self):
        super().__init__('goal_detector')
        
        # Parâmetros
        self.declare_parameter('goal_color_lower', [200, 200, 200])  # BGR para branco (gols)
        self.declare_parameter('goal_color_upper', [255, 255, 255])
        self.declare_parameter('debug_image', True)
        self.declare_parameter('min_post_height', 50)  # Altura mínima em pixels
        self.declare_parameter('min_post_width', 10)   # Largura mínima em pixels
        self.declare_parameter('max_post_width', 50)   # Largura máxima em pixels
        self.declare_parameter('use_field_mask', True)
        
        # Obter parâmetros
        self.goal_color_lower = np.array(self.get_parameter('goal_color_lower').value)
        self.goal_color_upper = np.array(self.get_parameter('goal_color_upper').value)
        self.debug_image = self.get_parameter('debug_image').value
        self.min_post_height = self.get_parameter('min_post_height').value
        self.min_post_width = self.get_parameter('min_post_width').value
        self.max_post_width = self.get_parameter('max_post_width').value
        self.use_field_mask = self.get_parameter('use_field_mask').value
        
        # Publishers
        self.goal_posts_pub = self.create_publisher(PoseArray, 'goal_posts', 10)
        self.debug_image_pub = self.create_publisher(Image, 'goal_detection_debug', 10)
        
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
        
        # Subscriber para a máscara do campo (opcional)
        if self.use_field_mask:
            self.field_mask_sub = self.create_subscription(
                Image,
                'field_mask',
                self.field_mask_callback,
                10
            )
        
        # Variáveis
        self.cv_bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.field_mask = None
        self.goal_post_height = 0.8  # Altura real do poste do gol em metros (padrão RoboCup)
        
        self.get_logger().info('Nó detector de gols iniciado')
    
    def camera_info_callback(self, msg):
        """Callback para informações da câmera."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
    
    def field_mask_callback(self, msg):
        """Callback para receber a máscara do campo."""
        try:
            self.field_mask = self.cv_bridge.imgmsg_to_cv2(msg, 'mono8')
        except Exception as e:
            self.get_logger().error(f'Erro ao converter máscara do campo: {str(e)}')
    
    def image_callback(self, msg):
        """Callback para processamento de imagem."""
        if self.camera_matrix is None:
            self.get_logger().warn('Informações da câmera ainda não recebidas')
            return
        
        try:
            # Converter ROS Image para OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Detectar os postes do gol
            goal_posts, debug_image = self.detect_goal_posts(cv_image)
            
            # Publicar as posições dos postes do gol
            if goal_posts:
                pose_array = PoseArray()
                pose_array.header = msg.header
                
                for post in goal_posts:
                    pose = Pose()
                    pose.position.x = post['distance']
                    pose.position.y = post['lateral_distance']
                    pose.position.z = 0.0
                    pose_array.poses.append(pose)
                
                self.goal_posts_pub.publish(pose_array)
            
            # Publicar imagem de debug se necessário
            if self.debug_image:
                debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, 'bgr8')
                debug_msg.header = msg.header
                self.debug_image_pub.publish(debug_msg)
                
        except Exception as e:
            self.get_logger().error(f'Erro no processamento da imagem: {str(e)}')
    
    def detect_goal_posts(self, image):
        """
        Detecta os postes do gol na imagem.
        
        Args:
            image: Imagem OpenCV no formato BGR
            
        Returns:
            tuple: (postes_do_gol, imagem_debug)
                postes_do_gol: Lista de dicionários com informações dos postes
                imagem_debug: Imagem OpenCV com marcações de debug
        """
        # Criar cópia da imagem para debug
        debug_image = image.copy()
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold para destacar os postes brancos
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Se estiver usando a máscara do campo, aplicá-la
        if self.use_field_mask and self.field_mask is not None:
            # Redimensionar a máscara se necessário
            if binary.shape != self.field_mask.shape:
                self.field_mask = cv2.resize(self.field_mask, (binary.shape[1], binary.shape[0]))
            
            # Aplicar a máscara
            binary = cv2.bitwise_and(binary, self.field_mask)
        
        # Aplicar operações morfológicas para remover ruído
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lista para armazenar os postes detectados
        goal_posts = []
        
        # Processar cada contorno
        for contour in contours:
            # Obter o retângulo delimitador
            x, y, w, h = cv2.boundingRect(contour)
            
            # Verificar se o contorno tem as dimensões esperadas para um poste
            if h > self.min_post_height and self.min_post_width < w < self.max_post_width:
                # Calcular a razão altura/largura
                aspect_ratio = h / w
                
                # Postes devem ser mais altos que largos
                if aspect_ratio > 2.0:
                    # Calcular a posição 3D do poste
                    # Implementação simplificada - assumindo que a câmera está calibrada
                    focal_length = self.camera_matrix[0, 0]  # Distância focal em pixels
                    
                    # Estimar a distância usando a altura do poste
                    distance = (self.goal_post_height * focal_length) / h
                    
                    # Calcular a posição lateral
                    center_x = image.shape[1] / 2
                    lateral_distance = ((x + w/2) - center_x) / focal_length * distance
                    
                    # Adicionar o poste à lista
                    goal_posts.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'distance': distance,
                        'lateral_distance': lateral_distance
                    })
                    
                    # Desenhar retângulo na imagem de debug
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(debug_image, f'D: {distance:.2f}m', (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Adicionar informações na imagem de debug
        if goal_posts:
            cv2.putText(debug_image, f'Postes: {len(goal_posts)}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(debug_image, 'Postes não encontrados', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return goal_posts, debug_image

def main(args=None):
    rclpy.init(args=args)
    node = GoalDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 