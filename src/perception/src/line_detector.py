#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import math

class LineDetector(Node):
    """
    Nó para detecção das linhas do campo de futebol usando visão computacional.
    
    Este nó processa imagens da câmera para detectar as linhas brancas do campo
    de futebol.
    """
    
    def __init__(self):
        super().__init__('line_detector')
        
        # Parâmetros
        self.declare_parameter('line_color_lower', [200, 200, 200])  # BGR para branco (linhas)
        self.declare_parameter('line_color_upper', [255, 255, 255])
        self.declare_parameter('debug_image', True)
        self.declare_parameter('canny_threshold1', 50)
        self.declare_parameter('canny_threshold2', 150)
        self.declare_parameter('hough_threshold', 50)
        self.declare_parameter('min_line_length', 30)
        self.declare_parameter('max_line_gap', 10)
        self.declare_parameter('use_field_mask', True)
        
        # Obter parâmetros
        self.line_color_lower = np.array(self.get_parameter('line_color_lower').value)
        self.line_color_upper = np.array(self.get_parameter('line_color_upper').value)
        self.debug_image = self.get_parameter('debug_image').value
        self.canny_threshold1 = self.get_parameter('canny_threshold1').value
        self.canny_threshold2 = self.get_parameter('canny_threshold2').value
        self.hough_threshold = self.get_parameter('hough_threshold').value
        self.min_line_length = self.get_parameter('min_line_length').value
        self.max_line_gap = self.get_parameter('max_line_gap').value
        self.use_field_mask = self.get_parameter('use_field_mask').value
        
        # Publishers
        self.lines_image_pub = self.create_publisher(Image, 'lines_image', 10)
        self.debug_image_pub = self.create_publisher(Image, 'line_detection_debug', 10)
        
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
        
        self.get_logger().info('Nó detector de linhas iniciado')
    
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
            
            # Detectar as linhas
            lines_image, debug_image = self.detect_lines(cv_image)
            
            # Publicar a imagem com as linhas
            lines_image_msg = self.cv_bridge.cv2_to_imgmsg(lines_image, 'mono8')
            lines_image_msg.header = msg.header
            self.lines_image_pub.publish(lines_image_msg)
            
            # Publicar imagem de debug se necessário
            if self.debug_image:
                debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, 'bgr8')
                debug_msg.header = msg.header
                self.debug_image_pub.publish(debug_msg)
                
        except Exception as e:
            self.get_logger().error(f'Erro no processamento da imagem: {str(e)}')
    
    def detect_lines(self, image):
        """
        Detecta as linhas do campo na imagem.
        
        Args:
            image: Imagem OpenCV no formato BGR
            
        Returns:
            tuple: (imagem_linhas, imagem_debug)
                imagem_linhas: Imagem binária com as linhas detectadas
                imagem_debug: Imagem OpenCV com marcações de debug
        """
        # Criar cópia da imagem para debug
        debug_image = image.copy()
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold para destacar as linhas brancas
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
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Detectar bordas com Canny
        edges = cv2.Canny(binary, self.canny_threshold1, self.canny_threshold2)
        
        # Detectar linhas com transformada de Hough
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold,
                               minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
        
        # Criar imagem para as linhas detectadas
        lines_image = np.zeros_like(gray)
        
        # Desenhar as linhas detectadas
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)
                cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Adicionar informações na imagem de debug
            cv2.putText(debug_image, f'Linhas: {len(lines)}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Adicionar informações na imagem de debug
            cv2.putText(debug_image, 'Linhas não encontradas', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return lines_image, debug_image

def main(args=None):
    rclpy.init(args=args)
    node = LineDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 