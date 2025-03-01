#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
import math

class ObstacleDetector(Node):
    """
    Nó para detecção de obstáculos usando visão computacional.
    
    Este nó processa imagens da câmera para detectar obstáculos (outros robôs,
    árbitros, etc.) e publicar suas posições relativas ao robô.
    """
    
    def __init__(self):
        super().__init__('obstacle_detector')
        
        # Parâmetros
        self.declare_parameter('debug_image', True)
        self.declare_parameter('min_obstacle_height', 30)  # Altura mínima em pixels
        self.declare_parameter('min_obstacle_width', 20)   # Largura mínima em pixels
        self.declare_parameter('use_field_mask', True)
        self.declare_parameter('use_color_segmentation', False)
        self.declare_parameter('obstacle_color_lower', [0, 0, 0])  # HSV para preto (obstáculos)
        self.declare_parameter('obstacle_color_upper', [180, 255, 50])
        
        # Obter parâmetros
        self.debug_image = self.get_parameter('debug_image').value
        self.min_obstacle_height = self.get_parameter('min_obstacle_height').value
        self.min_obstacle_width = self.get_parameter('min_obstacle_width').value
        self.use_field_mask = self.get_parameter('use_field_mask').value
        self.use_color_segmentation = self.get_parameter('use_color_segmentation').value
        self.obstacle_color_lower = np.array(self.get_parameter('obstacle_color_lower').value)
        self.obstacle_color_upper = np.array(self.get_parameter('obstacle_color_upper').value)
        
        # Publishers
        self.obstacles_pub = self.create_publisher(PoseArray, 'obstacles', 10)
        self.debug_image_pub = self.create_publisher(Image, 'obstacle_detection_debug', 10)
        
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
        self.obstacle_height = 0.6  # Altura média de um obstáculo em metros
        
        self.get_logger().info('Nó detector de obstáculos iniciado')
    
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
        
        if self.use_field_mask and self.field_mask is None:
            self.get_logger().warn('Máscara do campo ainda não recebida')
            return
        
        try:
            # Converter ROS Image para OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Detectar os obstáculos
            obstacles, debug_image = self.detect_obstacles(cv_image)
            
            # Publicar as posições dos obstáculos
            if obstacles:
                pose_array = PoseArray()
                pose_array.header = msg.header
                
                for obstacle in obstacles:
                    pose = Pose()
                    pose.position.x = obstacle['distance']
                    pose.position.y = obstacle['lateral_distance']
                    pose.position.z = 0.0
                    pose_array.poses.append(pose)
                
                self.obstacles_pub.publish(pose_array)
            
            # Publicar imagem de debug se necessário
            if self.debug_image:
                debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, 'bgr8')
                debug_msg.header = msg.header
                self.debug_image_pub.publish(debug_msg)
                
        except Exception as e:
            self.get_logger().error(f'Erro no processamento da imagem: {str(e)}')
    
    def detect_obstacles(self, image):
        """
        Detecta obstáculos na imagem.
        
        Args:
            image: Imagem OpenCV no formato BGR
            
        Returns:
            tuple: (obstáculos, imagem_debug)
                obstáculos: Lista de dicionários com informações dos obstáculos
                imagem_debug: Imagem OpenCV com marcações de debug
        """
        # Criar cópia da imagem para debug
        debug_image = image.copy()
        
        # Método 1: Usando a máscara do campo para encontrar objetos que não são campo
        if self.use_field_mask and self.field_mask is not None:
            # Redimensionar a máscara se necessário
            if image.shape[:2] != self.field_mask.shape:
                self.field_mask = cv2.resize(self.field_mask, (image.shape[1], image.shape[0]))
            
            # Inverter a máscara do campo para obter objetos que não são campo
            not_field_mask = cv2.bitwise_not(self.field_mask)
            
            # Aplicar operações morfológicas para remover ruído
            kernel = np.ones((5, 5), np.uint8)
            not_field_mask = cv2.erode(not_field_mask, kernel, iterations=1)
            not_field_mask = cv2.dilate(not_field_mask, kernel, iterations=2)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(not_field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Método 2: Usando segmentação por cor
        elif self.use_color_segmentation:
            # Converter para HSV para melhor segmentação de cor
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Criar máscara para a cor dos obstáculos
            mask = cv2.inRange(hsv, self.obstacle_color_lower, self.obstacle_color_upper)
            
            # Aplicar operações morfológicas para remover ruído
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Método 3: Fallback para detecção de bordas
        else:
            # Converter para escala de cinza
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar blur para reduzir ruído
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detectar bordas com Canny
            edges = cv2.Canny(blurred, 50, 150)
            
            # Aplicar operações morfológicas para conectar bordas
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lista para armazenar os obstáculos detectados
        obstacles = []
        
        # Processar cada contorno
        for contour in contours:
            # Obter o retângulo delimitador
            x, y, w, h = cv2.boundingRect(contour)
            
            # Verificar se o contorno tem as dimensões mínimas para um obstáculo
            if h > self.min_obstacle_height and w > self.min_obstacle_width:
                # Calcular a posição 3D do obstáculo
                # Implementação simplificada - assumindo que a câmera está calibrada
                focal_length = self.camera_matrix[0, 0]  # Distância focal em pixels
                
                # Estimar a distância usando a altura do obstáculo
                # Assumindo que a base do obstáculo está no chão
                distance = (self.obstacle_height * focal_length) / h
                
                # Calcular a posição lateral
                center_x = image.shape[1] / 2
                lateral_distance = ((x + w/2) - center_x) / focal_length * distance
                
                # Adicionar o obstáculo à lista
                obstacles.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'distance': distance,
                    'lateral_distance': lateral_distance
                })
                
                # Desenhar retângulo na imagem de debug
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(debug_image, f'D: {distance:.2f}m', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Adicionar informações na imagem de debug
        if obstacles:
            cv2.putText(debug_image, f'Obstáculos: {len(obstacles)}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(debug_image, 'Obstáculos não encontrados', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return obstacles, debug_image

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 