#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Componente para detecção de robôs usando o modelo YOEO.

Este componente é responsável por detectar robôs no campo,
calcular suas posições 3D relativas à câmera e fornecer
visualizações para depuração.
"""

import cv2
import numpy as np
import math

from ..yoeo_handler import DetectionType


class RobotDetectionComponent:
    """
    Componente para detecção de robôs.
    
    Este componente processa a saída de detecção do modelo YOEO
    para identificar robôs, calcular suas posições 3D
    e fornecer informações úteis para navegação e estratégia.
    """
    
    def __init__(self, yoeo_handler, robot_height=0.15, confidence_threshold=0.5):
        """
        Inicializa o componente de detecção de robô.
        
        Args:
            yoeo_handler: Manipulador do modelo YOEO
            robot_height: Altura média dos robôs em metros
            confidence_threshold: Limiar de confiança para detecções
        """
        self.yoeo_handler = yoeo_handler
        self.robot_height = robot_height
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
            detector: Detector tradicional de robôs
        """
        self.fallback_detector = detector
    
    def process(self, image):
        """
        Processa a imagem para detectar robôs.
        
        Args:
            image: Imagem BGR
            
        Returns:
            Lista de detecções de robôs, cada uma contendo:
            - bbox: Caixa delimitadora (x, y, w, h)
            - confidence: Confiança da detecção
            - center: Centro da detecção (x, y)
            - position_3d: Posição 3D relativa à câmera (x, y, z) em metros
            - team: Equipe do robô (se disponível)
        """
        # Obter detecções do modelo YOEO
        detections = self.yoeo_handler.get_detections(
            image, detection_types=[DetectionType.ROBOT], segmentation_types=[]
        )
        
        # Verificar se há detecções de robô
        if DetectionType.ROBOT in detections['detections']:
            robot_detections = detections['detections'][DetectionType.ROBOT]
            
            # Filtrar por confiança
            robot_detections = [
                det for det in robot_detections 
                if det['confidence'] > self.confidence_threshold
            ]
            
            # Calcular posições 3D e adicionar informações para cada detecção
            for detection in robot_detections:
                # Adicionar centro da detecção
                x, y, w, h = detection['bbox']
                detection['center'] = (int(x + w/2), int(y + h/2))
                
                # Calcular posição 3D
                detection['position_3d'] = self._calculate_3d_position(detection, image.shape)
                
                # Tentar identificar a equipe (simplificado)
                # Em uma implementação real, isso seria baseado em cores ou marcadores
                detection['team'] = self._identify_team(image, detection)
            
            return robot_detections
        
        # Tentar usar o detector de fallback se disponível
        if self.fallback_detector is not None:
            try:
                return self.fallback_detector.detect_robots(image)
            except Exception as e:
                print(f"Erro no detector de fallback: {e}")
        
        # Se não houver detecções, retornar lista vazia
        return []
    
    def _calculate_3d_position(self, detection, image_shape):
        """
        Calcula a posição 3D do robô relativa à câmera.
        
        Args:
            detection: Detecção do robô
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
        distance_z = (self.robot_height * fy) / h
        
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
    
    def _identify_team(self, image, detection):
        """
        Tenta identificar a equipe do robô com base em cores ou marcadores.
        
        Args:
            image: Imagem BGR original
            detection: Detecção do robô
            
        Returns:
            String indicando a equipe ('own', 'opponent', ou 'unknown')
        """
        # Implementação simplificada - em um sistema real, seria mais sofisticado
        # Extrair região do robô
        x, y, w, h = [int(v) for v in detection['bbox']]
        
        # Garantir que as coordenadas estejam dentro da imagem
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return 'unknown'
        
        # Extrair região de interesse
        roi = image[y:y+h, x:x+w]
        
        # Converter para HSV para melhor análise de cor
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Definir faixas de cor para as equipes (ajustar conforme necessário)
        # Exemplo: azul para equipe própria, amarelo para oponente
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        # Criar máscaras para cada cor
        mask_blue = cv2.inRange(hsv_roi, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
        
        # Contar pixels de cada cor
        blue_pixels = cv2.countNonZero(mask_blue)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        
        # Determinar equipe com base na cor predominante
        if blue_pixels > yellow_pixels and blue_pixels > roi.size * 0.1:
            return 'own'
        elif yellow_pixels > blue_pixels and yellow_pixels > roi.size * 0.1:
            return 'opponent'
        else:
            return 'unknown'
    
    def draw_detections(self, image, detections):
        """
        Desenha as detecções de robôs na imagem para visualização.
        
        Args:
            image: Imagem BGR original
            detections: Lista de detecções de robôs
            
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
            team = detection.get('team', 'unknown')
            
            # Definir cor com base na equipe
            if team == 'own':
                color = (255, 0, 0)  # Azul para equipe própria
            elif team == 'opponent':
                color = (0, 255, 255)  # Amarelo para oponente
            else:
                color = (0, 255, 0)  # Verde para desconhecido
            
            # Desenhar caixa delimitadora
            cv2.rectangle(vis_image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            
            # Desenhar centro
            cv2.circle(vis_image, center, 5, (0, 0, 255), -1)
            
            # Adicionar texto com confiança e equipe
            text = f"Robô ({team}): {confidence:.2f}"
            cv2.putText(vis_image, text, (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Adicionar informação de posição 3D se disponível
            if 'position_3d' in detection and detection['position_3d'] is not None:
                x3d, y3d, z3d = detection['position_3d']
                pos_text = f"X: {x3d:.2f}m Y: {y3d:.2f}m Z: {z3d:.2f}m"
                cv2.putText(vis_image, pos_text, (int(x), int(y + h) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_image 