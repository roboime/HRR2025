#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Componente para detecção de bola usando o modelo YOEO.

Este componente é responsável por detectar a bola no campo,
calcular sua posição 3D relativa à câmera e fornecer
visualizações para debugging.
"""

import cv2
import numpy as np
import math
from enum import Enum

from ..yoeo_handler import DetectionType


class BallDetectionComponent:
    """
    Componente para detecção de bola.
    
    Este componente processa as detecções do modelo YOEO para
    identificar a bola, calcular sua posição 3D e fornecer
    informações úteis para o sistema de visão.
    """
    
    def __init__(self, yoeo_handler, ball_diameter=0.043, confidence_threshold=0.5):
        """
        Inicializa o componente de detecção de bola.
        
        Args:
            yoeo_handler: Manipulador do modelo YOEO
            ball_diameter: Diâmetro da bola em metros (padrão: 0.043m para bola de futebol de robôs)
            confidence_threshold: Limiar de confiança para filtrar detecções
        """
        self.yoeo_handler = yoeo_handler
        self.ball_diameter = ball_diameter
        self.confidence_threshold = confidence_threshold
        self.fallback_detector = None
        self.camera_info = None
    
    def set_camera_info(self, camera_info):
        """
        Define as informações da câmera.
        
        Args:
            camera_info: Mensagem CameraInfo do ROS
        """
        self.camera_info = camera_info
    
    def set_fallback_detector(self, detector):
        """
        Define um detector tradicional para fallback.
        
        Args:
            detector: Detector tradicional de bola
        """
        self.fallback_detector = detector
    
    def process(self, image, field_mask=None):
        """
        Processa a imagem para detectar a bola.
        
        Args:
            image: Imagem BGR
            field_mask: Máscara opcional do campo para filtrar detecções
            
        Returns:
            Lista de detecções de bola, cada uma contendo:
            - bbox: [x1, y1, x2, y2] coordenadas do retângulo delimitador
            - confidence: Pontuação de confiança
            - center: (cx, cy) coordenadas do centro da bola
            - position_3d: (x, y, z) posição 3D em metros
        """
        # Obter detecções do modelo YOEO
        detections = self.yoeo_handler.get_detections(
            image, detection_types=[DetectionType.BALL], segmentation_types=[]
        )
        
        # Verificar se há detecções de bola
        ball_detections = []
        if DetectionType.BALL in detections['detections']:
            # Filtrar detecções por confiança
            for detection in detections['detections'][DetectionType.BALL]:
                if detection['confidence'] >= self.confidence_threshold:
                    # Extrair coordenadas do retângulo delimitador
                    x1, y1, x2, y2 = detection['bbox']
                    
                    # Verificar se a detecção está dentro do campo (se a máscara for fornecida)
                    if field_mask is not None:
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Verificar se o centro da bola está dentro do campo
                        if center_y < field_mask.shape[0] and center_x < field_mask.shape[1]:
                            if field_mask[center_y, center_x] == 0:  # Fora do campo
                                continue
                    
                    # Calcular o centro da bola
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    # Calcular a posição 3D da bola
                    position_3d = self._calculate_3d_position(detection['bbox'], image.shape)
                    
                    # Adicionar à lista de detecções
                    ball_detections.append({
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'center': center,
                        'position_3d': position_3d
                    })
        
        # Se não houver detecções e houver um detector de fallback, tentar usá-lo
        if not ball_detections and self.fallback_detector is not None:
            try:
                fallback_results = self.fallback_detector.detect_ball(image, field_mask)
                if fallback_results:
                    # Converter resultados do detector de fallback para o formato padrão
                    for result in fallback_results:
                        # Assumindo que o detector de fallback retorna (x, y, raio, confiança)
                        x, y, radius, confidence = result
                        
                        # Criar retângulo delimitador
                        x1 = max(0, x - radius)
                        y1 = max(0, y - radius)
                        x2 = min(image.shape[1], x + radius)
                        y2 = min(image.shape[0], y + radius)
                        
                        # Calcular posição 3D
                        position_3d = self._calculate_3d_position([x1, y1, x2, y2], image.shape)
                        
                        ball_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'center': (x, y),
                            'position_3d': position_3d
                        })
            except Exception as e:
                print(f"Erro no detector de fallback: {e}")
        
        # Ordenar detecções por confiança (maior primeiro)
        ball_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return ball_detections
    
    def _calculate_3d_position(self, bbox, image_shape):
        """
        Calcula a posição 3D da bola com base no tamanho do retângulo delimitador.
        
        Args:
            bbox: Retângulo delimitador [x1, y1, x2, y2]
            image_shape: Forma da imagem (altura, largura, canais)
            
        Returns:
            Tupla (x, y, z) representando a posição 3D em metros
        """
        # Se não houver informações da câmera, retornar estimativa aproximada
        if self.camera_info is None:
            # Estimativa simples baseada no tamanho da bola na imagem
            x1, y1, x2, y2 = bbox
            ball_width_px = x2 - x1
            ball_height_px = y2 - y1
            ball_size_px = (ball_width_px + ball_height_px) / 2
            
            # Estimar distância baseada no tamanho aparente
            # Quanto menor a bola na imagem, mais distante ela está
            # Esta é uma aproximação simples e deve ser calibrada
            focal_length_estimate = 500  # Valor aproximado para câmeras típicas
            distance = (self.ball_diameter * focal_length_estimate) / ball_size_px
            
            # Calcular coordenadas X e Y baseadas na posição na imagem
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Converter de coordenadas de pixel para coordenadas de mundo
            # Origem no centro da imagem
            image_center_x = image_shape[1] / 2
            image_center_y = image_shape[0] / 2
            
            # Fator de escala para converter pixels em metros na distância estimada
            scale_factor = distance / focal_length_estimate
            
            x = (center_x - image_center_x) * scale_factor
            y = (center_y - image_center_y) * scale_factor
            z = distance
            
            return (x, y, z)
        
        # Se houver informações da câmera, usar modelo de câmera pinhole
        else:
            # Extrair parâmetros da câmera
            fx = self.camera_info.k[0]  # Distância focal x
            fy = self.camera_info.k[4]  # Distância focal y
            cx = self.camera_info.k[2]  # Centro óptico x
            cy = self.camera_info.k[5]  # Centro óptico y
            
            # Calcular centro e tamanho da bola em pixels
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ball_size_px = ((x2 - x1) + (y2 - y1)) / 2
            
            # Calcular distância usando o modelo pinhole
            distance = (self.ball_diameter * fx) / ball_size_px
            
            # Calcular coordenadas 3D
            x = (center_x - cx) * distance / fx
            y = (center_y - cy) * distance / fy
            z = distance
            
            return (x, y, z)
    
    def draw_detections(self, image, detections):
        """
        Desenha as detecções de bola na imagem para visualização.
        
        Args:
            image: Imagem BGR original
            detections: Lista de detecções de bola
            
        Returns:
            Imagem com visualização das detecções
        """
        if not detections:
            return image
        
        # Criar cópia da imagem
        vis_image = image.copy()
        
        # Desenhar cada detecção
        for i, detection in enumerate(detections):
            # Extrair informações
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
            confidence = detection['confidence']
            center_x, center_y = [int(coord) for coord in detection['center']]
            x, y, z = detection['position_3d']
            
            # Desenhar retângulo delimitador
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
            
            # Desenhar centro
            cv2.circle(vis_image, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # Adicionar texto com confiança e posição 3D
            text_conf = f"Bola: {confidence:.2f}"
            text_pos = f"Pos: ({x:.2f}, {y:.2f}, {z:.2f})m"
            
            cv2.putText(vis_image, text_conf, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            cv2.putText(vis_image, text_pos, (x1, y1 + y2 - y1 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        return vis_image 