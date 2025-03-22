#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Componente para detecção de gols usando o modelo YOEO.

Este componente é responsável por detectar postes de gol,
calcular suas posições 3D relativas ao robô e fornecer
visualizações para depuração.
"""

import cv2
import numpy as np
import math

from ..yoeo_handler import DetectionType


class GoalDetectionComponent:
    """
    Componente para detecção de postes de gol.
    
    Este componente processa a saída de detecção do modelo YOEO
    para identificar postes de gol, calcular suas posições 3D
    e fornecer informações úteis para localização.
    """
    
    def __init__(self, yoeo_handler, goal_height=0.18, confidence_threshold=0.5):
        """
        Inicializa o componente de detecção de gol.
        
        Args:
            yoeo_handler: Manipulador do modelo YOEO
            goal_height: Altura real dos postes de gol em metros
            confidence_threshold: Limiar de confiança para detecções
        """
        self.yoeo_handler = yoeo_handler
        self.goal_height = goal_height
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
            detector: Detector tradicional de gol
        """
        self.fallback_detector = detector
    
    def process(self, image):
        """
        Processa a imagem para detectar postes de gol.
        
        Args:
            image: Imagem BGR
            
        Returns:
            Lista de detecções de postes de gol, cada uma contendo:
            - bbox: Caixa delimitadora (x, y, w, h)
            - confidence: Confiança da detecção
            - center: Centro da detecção (x, y)
            - position_3d: Posição 3D relativa à câmera (x, y, z) em metros
        """
        # Obter detecções do modelo YOEO
        detections = self.yoeo_handler.get_detections(
            image, detection_types=[DetectionType.GOAL], segmentation_types=[]
        )
        
        # Verificar se há detecções de gol
        if DetectionType.GOAL in detections['detections']:
            goal_detections = detections['detections'][DetectionType.GOAL]
            
            # Filtrar por confiança
            goal_detections = [
                det for det in goal_detections 
                if det['confidence'] > self.confidence_threshold
            ]
            
            # Calcular posições 3D para cada detecção
            for detection in goal_detections:
                # Adicionar centro da detecção
                x, y, w, h = detection['bbox']
                detection['center'] = (int(x + w/2), int(y + h/2))
                
                # Calcular posição 3D
                detection['position_3d'] = self._calculate_3d_position(detection, image.shape)
            
            return goal_detections
        
        # Tentar usar o detector de fallback se disponível
        if self.fallback_detector is not None:
            try:
                return self.fallback_detector.detect_goals(image)
            except Exception as e:
                print(f"Erro no detector de fallback: {e}")
        
        # Se não houver detecções, retornar lista vazia
        return []
    
    def _calculate_3d_position(self, detection, image_shape):
        """
        Calcula a posição 3D do poste de gol relativa à câmera.
        
        Args:
            detection: Detecção do poste de gol
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
        distance_z = (self.goal_height * fy) / h
        
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
        Desenha as detecções de postes de gol na imagem para visualização.
        
        Args:
            image: Imagem BGR original
            detections: Lista de detecções de postes de gol
            
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
            cv2.rectangle(vis_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            
            # Desenhar centro
            cv2.circle(vis_image, center, 5, (0, 0, 255), -1)
            
            # Adicionar texto com confiança
            text = f"Gol: {confidence:.2f}"
            cv2.putText(vis_image, text, (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Adicionar informação de posição 3D se disponível
            if 'position_3d' in detection and detection['position_3d'] is not None:
                x3d, y3d, z3d = detection['position_3d']
                pos_text = f"X: {x3d:.2f}m Y: {y3d:.2f}m Z: {z3d:.2f}m"
                cv2.putText(vis_image, pos_text, (int(x), int(y + h) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image 