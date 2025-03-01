#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from geometry_msgs.msg import Pose2D
from src.perception.src.yoeo.yoeo_handler import DetectionType

class BallDetectionComponent:
    """
    Componente para detecção de bola usando YOEO.
    
    Este componente é responsável por:
    1. Extrair detecções de bola das saídas do YOEO
    2. Calcular a posição 3D da bola em relação ao robô
    3. Fornecer visualizações para debugging
    """
    
    def __init__(self, yoeo_handler, camera_info=None, ball_diameter=0.14):
        """
        Inicializa o componente de detecção de bola.
        
        Args:
            yoeo_handler: Instância de YOEOHandler para processar imagens
            camera_info: Informações da câmera para projeção 3D
            ball_diameter: Diâmetro da bola em metros (padrão da RoboCup)
        """
        self.yoeo_handler = yoeo_handler
        self.camera_info = camera_info
        self.ball_diameter = ball_diameter
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Se camera_info for fornecido, extrair matriz da câmera
        if camera_info is not None:
            self.set_camera_info(camera_info)
    
    def set_camera_info(self, camera_info):
        """
        Define as informações da câmera para cálculos de posição 3D.
        
        Args:
            camera_info: Mensagem CameraInfo do ROS
        """
        self.camera_matrix = np.array(camera_info.k).reshape(3, 3)
        self.dist_coeffs = np.array(camera_info.d)
    
    def process(self, image):
        """
        Processa a imagem e retorna as detecções de bola.
        
        Args:
            image: Imagem OpenCV no formato BGR
            
        Returns:
            Lista de dicionários contendo informações das bolas detectadas
        """
        # Obter detecções do YOEO
        results = self.yoeo_handler.get_detections(image)
        
        # Extrair apenas as detecções de bola
        ball_detections = results['detections'].get(DetectionType.BALL, [])
        
        # Se não houver detecções de bola, tentar com detector tradicional
        if not ball_detections and hasattr(self, 'fallback_detector'):
            # Este é um ponto onde podemos integrar o detector tradicional
            pass
        
        # Calcular posições 3D para cada bola detectada
        balls_with_positions = []
        for ball in ball_detections:
            # Extrair coordenadas da caixa delimitadora
            x1, y1, x2, y2 = ball['bbox']
            confidence = ball['confidence']
            
            # Calcular centro e raio
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            radius = ((x2 - x1) + (y2 - y1)) / 4  # Estimativa do raio
            
            # Calcular posição 3D se tivermos informações da câmera
            position = None
            if self.camera_matrix is not None:
                position = self._calculate_3d_position(center_x, center_y, radius)
            
            # Adicionar à lista de detecções
            balls_with_positions.append({
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'radius': radius,
                'confidence': confidence,
                'position': position
            })
        
        return balls_with_positions
    
    def _calculate_3d_position(self, x, y, radius):
        """
        Calcula a posição 3D da bola com base nas coordenadas da imagem.
        
        Args:
            x, y: Coordenadas do centro da bola na imagem
            radius: Raio da bola na imagem
            
        Returns:
            (x, y, z): Coordenadas 3D da bola em relação à câmera
        """
        if self.camera_matrix is None:
            return None
        
        # Distância focal em pixels
        focal_length = self.camera_matrix[0, 0]
        
        # Calcular distância usando a relação entre o tamanho real e o tamanho na imagem
        distance = (self.ball_diameter * focal_length) / (2 * radius)
        
        # Calcular coordenadas no plano da imagem em relação ao centro
        center_x = self.camera_matrix[0, 2]  # cx
        center_y = self.camera_matrix[1, 2]  # cy
        
        # Converter de pixels para coordenadas do mundo
        x_world = ((x - center_x) / focal_length) * distance
        y_world = ((y - center_y) / focal_length) * distance
        z_world = distance
        
        return (x_world, y_world, z_world)
    
    def to_ros_message(self, ball_detection):
        """
        Converte uma detecção de bola para mensagem ROS Pose2D.
        
        Args:
            ball_detection: Dicionário contendo informações da bola
            
        Returns:
            Mensagem Pose2D com a posição da bola
        """
        pose = Pose2D()
        
        if ball_detection['position'] is not None:
            x, y, z = ball_detection['position']
            pose.x = z  # Distância frontal (z no sistema da câmera)
            pose.y = -x  # Distância lateral (x no sistema da câmera)
            pose.theta = 0.0  # A bola não tem orientação
        
        return pose
    
    def draw_detections(self, image, ball_detections):
        """
        Desenha as detecções de bola na imagem.
        
        Args:
            image: Imagem OpenCV no formato BGR
            ball_detections: Lista de dicionários contendo informações das bolas
            
        Returns:
            Imagem com as detecções desenhadas
        """
        debug_image = image.copy()
        
        for ball in ball_detections:
            # Extrair informações
            x1, y1, x2, y2 = ball['bbox']
            center_x, center_y = ball['center']
            radius = ball['radius']
            confidence = ball['confidence']
            
            # Desenhar círculo
            cv2.circle(debug_image, (int(center_x), int(center_y)), int(radius), (0, 0, 255), 2)
            
            # Adicionar informações de confiança
            cv2.putText(debug_image, f'{confidence:.2f}', (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Se tivermos posição 3D, adicionar essa informação
            if ball['position'] is not None:
                x, y, z = ball['position']
                cv2.putText(debug_image, f'({x:.2f}, {y:.2f}, {z:.2f})m', 
                           (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Adicionar contador de bolas
        cv2.putText(debug_image, f'Bolas: {len(ball_detections)}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return debug_image
    
    def set_fallback_detector(self, detector):
        """
        Define um detector tradicional para ser usado como fallback.
        
        Args:
            detector: Instância de um detector tradicional
        """
        self.fallback_detector = detector 