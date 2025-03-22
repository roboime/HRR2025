#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Componente para detecção de árbitro usando o modelo YOEO.

Este componente é responsável por detectar o árbitro no campo,
calcular sua posição 3D relativa à câmera e fornecer
visualizações para depuração.
"""

import cv2
import numpy as np
import math

from ..yoeo_handler import DetectionType


class RefereeDetectionComponent:
    """
    Componente para detecção de árbitro.
    
    Este componente processa a saída de detecção do modelo YOEO
    para identificar o árbitro, calcular sua posição 3D
    e fornecer informações úteis para estratégia.
    """
    
    def __init__(self, yoeo_handler, referee_height=1.7, confidence_threshold=0.5):
        """
        Inicializa o componente de detecção de árbitro.
        
        Args:
            yoeo_handler: Manipulador do modelo YOEO
            referee_height: Altura média do árbitro em metros
            confidence_threshold: Limiar de confiança para detecções
        """
        self.yoeo_handler = yoeo_handler
        self.referee_height = referee_height
        self.confidence_threshold = confidence_threshold
        self.fallback_detector = None
        self.camera_info = None
    
    def set_camera_info(self, camera_info):
        """
        Define as informações da câmera para cálculos de posição 3D.
        
        Args:
            camera_info: Mensagem CameraInfo do ROS
        """
        self.camera_info = camera_info
    
    def set_fallback_detector(self, detector):
        """
        Define um detector tradicional para fallback.
        
        Args:
            detector: Detector tradicional de árbitro
        """
        self.fallback_detector = detector
    
    def process(self, image):
        """
        Processa a imagem para detectar o árbitro.
        
        Args:
            image: Imagem BGR
            
        Returns:
            Lista de detecções de árbitro, cada uma contendo:
            - bbox: Caixa delimitadora (x, y, w, h)
            - confidence: Confiança da detecção
            - center: Centro da detecção (x, y)
            - position_3d: Posição 3D relativa à câmera (x, y, z) em metros
        """
        # Obter detecções do modelo YOEO
        detections = self.yoeo_handler.get_detections(
            image, detection_types=[DetectionType.REFEREE], segmentation_types=[]
        )
        
        # Verificar se há detecções de árbitro
        if DetectionType.REFEREE in detections['detections']:
            referee_detections = detections['detections'][DetectionType.REFEREE]
            
            # Filtrar por confiança
            referee_detections = [
                det for det in referee_detections 
                if det['confidence'] > self.confidence_threshold
            ]
            
            # Calcular posições 3D para cada detecção
            for detection in referee_detections:
                # Adicionar centro da detecção
                x, y, w, h = detection['bbox']
                detection['center'] = (int(x + w/2), int(y + h/2))
                
                # Calcular posição 3D
                detection['position_3d'] = self._calculate_3d_position(detection, image.shape)
            
            return referee_detections
        
        # Tentar usar o detector de fallback se disponível
        if self.fallback_detector is not None:
            try:
                return self.fallback_detector.detect_referee(image)
            except Exception as e:
                print(f"Erro no detector de fallback: {e}")
        
        # Se não houver detecções, retornar lista vazia
        return []
    
    def _calculate_3d_position(self, detection, image_shape):
        """
        Calcula a posição 3D do árbitro relativa à câmera.
        
        Args:
            detection: Detecção do árbitro
            image_shape: Forma da imagem (altura, largura, canais)
            
        Returns:
            Posição 3D (x, y, z) em metros, ou None se não for possível calcular
        """
        if self.camera_info is None:
            return None
        
        # Extrair parâmetros da câmera
        fx = self.camera_info.k[0]  # Distância focal x
        fy = self.camera_info.k[4]  # Distância focal y
        cx = self.camera_info.k[2]  # Centro óptico x
        cy = self.camera_info.k[5]  # Centro óptico y
        
        # Se os parâmetros da câmera não estiverem disponíveis, usar valores padrão
        if fx == 0 or fy == 0:
            fx = fy = 500.0  # Valor aproximado para câmeras comuns
            cx = image_shape[1] / 2
            cy = image_shape[0] / 2
        
        # Extrair informações da detecção
        x, y, w, h = detection['bbox']
        
        # Usar a altura da caixa delimitadora para estimar a distância
        # Baseado na relação: altura_real / altura_pixel = distância / distância_focal
        distance_z = (self.referee_height * fy) / h
        
        # Calcular coordenadas X e Y no mundo
        center_x = x + w/2
        center_y = y + h/2
        
        # Converter de coordenadas de pixel para coordenadas do mundo
        world_x = (center_x - cx) * distance_z / fx
        world_y = (center_y - cy) * distance_z / fy
        
        # Retornar posição 3D (x, y, z) em metros
        # x: lateral (positivo para a direita)
        # y: vertical (positivo para baixo)
        # z: profundidade (positivo para frente)
        return (world_x, world_y, distance_z)
    
    def draw_detections(self, image, detections):
        """
        Desenha as detecções de árbitro na imagem para visualização.
        
        Args:
            image: Imagem BGR original
            detections: Lista de detecções de árbitro
            
        Returns:
            Imagem com visualização das detecções
        """
        if not detections:
            return image
        
        # Criar cópia da imagem
        vis_image = image.copy()
        
        # Desenhar cada detecção
        for detection in detections:
            # Extrair informações da detecção
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            center = detection['center']
            
            # Desenhar caixa delimitadora
            cv2.rectangle(vis_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            
            # Desenhar centro
            cv2.circle(vis_image, center, 5, (255, 0, 0), -1)
            
            # Adicionar texto com confiança
            text = f"Árbitro: {confidence:.2f}"
            cv2.putText(vis_image, text, (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Adicionar informação de posição 3D se disponível
            if 'position_3d' in detection and detection['position_3d'] is not None:
                x3d, y3d, z3d = detection['position_3d']
                pos_text = f"X: {x3d:.2f}m Y: {y3d:.2f}m Z: {z3d:.2f}m"
                cv2.putText(vis_image, pos_text, (int(x), int(y + h) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return vis_image 