#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import math

class FieldDetector(Node):
    """
    Nó para detecção do campo de futebol usando visão computacional.
    
    Este nó processa imagens da câmera para detectar o campo de futebol
    e sua fronteira.
    """
    
    def __init__(self):
        super().__init__('field_detector')
        
        # Parâmetros
        self.declare_parameter('field_color_lower', [35, 80, 25])  # HSV para verde (campo)
        self.declare_parameter('field_color_upper', [65, 255, 255])
        self.declare_parameter('debug_image', True)
        self.declare_parameter('min_field_area_ratio', 0.1)  # Área mínima do campo em relação à imagem
        
        # Obter parâmetros
        self.field_color_lower = np.array(self.get_parameter('field_color_lower').value)
        self.field_color_upper = np.array(self.get_parameter('field_color_upper').value)
        self.debug_image = self.get_parameter('debug_image').value
        self.min_field_area_ratio = self.get_parameter('min_field_area_ratio').value
        
        # Publishers
        self.field_mask_pub = self.create_publisher(Image, 'field_mask', 10)
        self.field_boundary_pub = self.create_publisher(Image, 'field_boundary', 10)
        self.debug_image_pub = self.create_publisher(Image, 'field_detection_debug', 10)
        
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
        
        self.get_logger().info('Nó detector de campo iniciado')
    
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
            
            # Detectar o campo
            field_mask, field_boundary, debug_image = self.detect_field(cv_image)
            
            # Publicar a máscara do campo
            field_mask_msg = self.cv_bridge.cv2_to_imgmsg(field_mask, 'mono8')
            field_mask_msg.header = msg.header
            self.field_mask_pub.publish(field_mask_msg)
            
            # Publicar a fronteira do campo
            field_boundary_msg = self.cv_bridge.cv2_to_imgmsg(field_boundary, 'mono8')
            field_boundary_msg.header = msg.header
            self.field_boundary_pub.publish(field_boundary_msg)
            
            # Publicar imagem de debug se necessário
            if self.debug_image:
                debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, 'bgr8')
                debug_msg.header = msg.header
                self.debug_image_pub.publish(debug_msg)
                
        except Exception as e:
            self.get_logger().error(f'Erro no processamento da imagem: {str(e)}')
    
    def detect_field(self, image):
        """
        Detecta o campo na imagem usando segmentação por cor.
        
        Args:
            image: Imagem OpenCV no formato BGR
            
        Returns:
            tuple: (máscara_do_campo, fronteira_do_campo, imagem_debug)
                máscara_do_campo: Imagem binária onde o campo é branco
                fronteira_do_campo: Imagem binária com a fronteira do campo
                imagem_debug: Imagem OpenCV com marcações de debug
        """
        # Criar cópia da imagem para debug
        debug_image = image.copy()
        
        # Converter para HSV para melhor segmentação de cor
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Criar máscara para a cor do campo
        mask = cv2.inRange(hsv, self.field_color_lower, self.field_color_upper)
        
        # Aplicar operações morfológicas para remover ruído
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Verificar se a área do campo é suficiente
        field_area_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        
        if field_area_ratio < self.min_field_area_ratio:
            self.get_logger().warn(f'Área do campo muito pequena: {field_area_ratio:.2f}')
            # Criar imagens vazias para retorno
            empty_mask = np.zeros_like(mask)
            return empty_mask, empty_mask, debug_image
        
        # Encontrar contornos do campo
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Encontrar o maior contorno (que deve ser o campo)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Criar máscara apenas com o maior contorno
            field_mask = np.zeros_like(mask)
            cv2.drawContours(field_mask, [largest_contour], 0, 255, -1)
            
            # Criar imagem da fronteira do campo
            field_boundary = np.zeros_like(mask)
            cv2.drawContours(field_boundary, [largest_contour], 0, 255, 2)
            
            # Desenhar contorno na imagem de debug
            cv2.drawContours(debug_image, [largest_contour], 0, (0, 255, 0), 2)
            
            # Adicionar informações na imagem de debug
            cv2.putText(debug_image, f'Campo: {field_area_ratio:.2f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Se não encontrou contornos, retornar máscaras vazias
            field_mask = np.zeros_like(mask)
            field_boundary = np.zeros_like(mask)
            
            # Adicionar informações na imagem de debug
            cv2.putText(debug_image, 'Campo não encontrado', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return field_mask, field_boundary, debug_image

def main(args=None):
    rclpy.init(args=args)
    node = FieldDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 