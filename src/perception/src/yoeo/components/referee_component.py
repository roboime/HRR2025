#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from geometry_msgs.msg import Pose
from src.perception.src.yoeo.yoeo_handler import YOEOHandler, DetectionType

class RefereeDetectionComponent:
    """
    Componente para detecção de árbitro utilizando o YOEO.
    
    Este componente é responsável por detectar o árbitro na imagem,
    calcular sua posição 3D relativa ao robô, e fornecer visualizações
    para depuração.
    """
    
    def __init__(self, yoeo_handler, referee_width=0.5, referee_height=1.7):
        """
        Inicializa o componente de detecção de árbitro.
        
        Args:
            yoeo_handler: Instância do YOEOHandler para acesso ao modelo
            referee_width: Largura média do árbitro em metros
            referee_height: Altura média do árbitro em metros
        """
        self.yoeo_handler = yoeo_handler
        self.referee_width = referee_width
        self.referee_height = referee_height
        self.fallback_detector = None
        self.camera_info = None
    
    def set_camera_info(self, camera_info):
        """
        Define as informações da câmera para cálculos de posição.
        
        Args:
            camera_info: Informações da câmera (mensagem ROS CameraInfo)
        """
        self.camera_info = camera_info
    
    def set_fallback_detector(self, detector):
        """
        Define um detector tradicional para ser usado como fallback.
        
        Args:
            detector: Instância do detector tradicional
        """
        self.fallback_detector = detector
    
    def process(self, image):
        """
        Processa a imagem para detectar o árbitro.
        
        Args:
            image: Imagem BGR do OpenCV
            
        Returns:
            Lista de detecções de árbitro, cada uma contendo:
            {
                'bbox': [x, y, width, height],
                'confidence': confiança da detecção,
                'center': (x, y) centro da detecção,
                'position': (x, y, z) posição 3D relativa ao robô
            }
        """
        # Verificar se a imagem é válida
        if image is None or image.size == 0:
            return []
        
        # Tentar obter as detecções do YOEO
        try:
            # Obter detecções do YOEO
            detections = self.yoeo_handler.get_detections(image, DetectionType.REFEREE)
            
            # Processar detecções se encontradas
            if detections and len(detections) > 0:
                referees = []
                
                for detection in detections:
                    # Calcular centro da caixa delimitadora
                    x, y, w, h = detection.get('bbox', [0, 0, 0, 0])
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    # Calcular posição 3D
                    position = self._calculate_3d_position((center_x, center_y), w, h)
                    
                    # Adicionar informações de detecção
                    referee_info = {
                        'bbox': [x, y, w, h],
                        'confidence': detection.get('confidence', 0.0),
                        'center': (int(center_x), int(center_y)),
                        'position': position
                    }
                    
                    referees.append(referee_info)
                
                return referees
            elif self.fallback_detector is not None:
                # Usar detector tradicional como fallback
                return self.fallback_detector.detect(image)
            else:
                return []
        except Exception as e:
            print(f"Erro na detecção de árbitro: {e}")
            return []
    
    def _calculate_3d_position(self, center, width, height):
        """
        Calcula a posição 3D do árbitro a partir de suas coordenadas na imagem.
        
        Args:
            center: Coordenadas (x, y) do centro da caixa delimitadora
            width: Largura da caixa delimitadora
            height: Altura da caixa delimitadora
            
        Returns:
            Tupla (x, y, z) com a posição 3D do árbitro relativa ao robô (em metros)
            x: lateral (positivo à direita), y: vertical, z: frontal
        """
        if self.camera_info is None:
            return None
        
        try:
            # Extrair parâmetros intrínsecos da câmera
            fx = self.camera_info.k[0]  # Distância focal em x
            fy = self.camera_info.k[4]  # Distância focal em y
            cx = self.camera_info.k[2]  # Centro óptico x
            cy = self.camera_info.k[5]  # Centro óptico y
            
            # Calcular distância com base na altura para pessoas (geralmente mais confiável)
            distance_z = (self.referee_height * fy) / height
            
            # Calcular coordenadas 3D
            x_center, y_center = center
            x_3d = (x_center - cx) * distance_z / fx
            y_3d = (y_center - cy) * distance_z / fy
            
            return (x_3d, y_3d, distance_z)
        except Exception as e:
            print(f"Erro ao calcular posição 3D: {e}")
            return None
    
    def to_ros_message(self, referee_detection):
        """
        Converte detecção de árbitro para mensagem ROS.
        
        Args:
            referee_detection: Detecção de árbitro
            
        Returns:
            Mensagem Pose com a posição do árbitro
        """
        pose = Pose()
        
        if referee_detection and 'position' in referee_detection and referee_detection['position'] is not None:
            x, y, z = referee_detection['position']
            pose.position.x = z  # Distância frontal
            pose.position.y = -x  # Distância lateral (invertida para ROS)
            pose.position.z = 0.0  # Assume-se que o árbitro está no solo
            
            # A orientação poderia ser inferida com mais informações
            # Por enquanto, assume-se que não há orientação
        
        return pose
    
    def draw_detections(self, image, referee_detections):
        """
        Desenha as detecções de árbitro na imagem para visualização.
        
        Args:
            image: Imagem original
            referee_detections: Lista de detecções de árbitro
            
        Returns:
            Imagem com visualizações
        """
        if not referee_detections:
            return image
        
        # Criar cópia da imagem
        vis_image = image.copy()
        
        # Cor para árbitro (magenta)
        color = (255, 0, 255)
        
        for referee in referee_detections:
            # Extrair informações
            bbox = referee.get('bbox', [0, 0, 0, 0])
            confidence = referee.get('confidence', 0.0)
            position = referee.get('position', None)
            
            if bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                
                # Desenhar retângulo
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                
                # Texto de confiança
                conf_text = f"Árbitro: {confidence:.2f}"
                cv2.putText(vis_image, conf_text, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Adicionar informação de posição 3D se disponível
                if position is not None:
                    x_3d, y_3d, z_3d = position
                    pos_text = f"({x_3d:.2f}, {z_3d:.2f}m)"
                    cv2.putText(vis_image, pos_text, (x, y + h + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Adicionar contador de árbitros
        cv2.putText(vis_image, f"Árbitros: {len(referee_detections)}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image 