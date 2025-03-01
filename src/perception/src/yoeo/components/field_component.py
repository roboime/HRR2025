#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Componente para segmentação do campo usando o modelo YOEO.

Este componente é responsável por extrair e processar a segmentação
do campo a partir das saídas do modelo YOEO.
"""

import cv2
import numpy as np
from enum import Enum

from ..yoeo_handler import SegmentationType


class FieldSegmentationComponent:
    """
    Componente para segmentação do campo.
    
    Este componente processa a saída de segmentação do modelo YOEO
    para extrair a máscara do campo, encontrar sua fronteira e
    fornecer informações úteis para localização.
    """
    
    def __init__(self, yoeo_handler, min_field_area_ratio=0.1):
        """
        Inicializa o componente de segmentação de campo.
        
        Args:
            yoeo_handler: Manipulador do modelo YOEO
            min_field_area_ratio: Razão mínima da área do campo em relação à imagem
        """
        self.yoeo_handler = yoeo_handler
        self.min_field_area_ratio = min_field_area_ratio
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
            detector: Detector tradicional de campo
        """
        self.fallback_detector = detector
    
    def process(self, image):
        """
        Processa a imagem para segmentar o campo.
        
        Args:
            image: Imagem BGR
            
        Returns:
            Dicionário com resultados da segmentação de campo:
            - mask: Máscara binária do campo
            - boundary: Imagem com a fronteira do campo
            - contours: Contornos do campo
        """
        # Obter segmentação do modelo YOEO
        segmentations = self.yoeo_handler.get_detections(
            image, detection_types=[], segmentation_types=[SegmentationType.FIELD]
        )
        
        # Verificar se a segmentação de campo está disponível
        if SegmentationType.FIELD in segmentations['segmentations']:
            field_mask = segmentations['segmentations'][SegmentationType.FIELD]
            
            # Processar a máscara de campo
            processed_mask, boundary, contours = self._process_mask(field_mask, image.shape[:2])
            
            # Verificar se a área do campo é suficiente
            if processed_mask is not None and self._check_field_area(processed_mask, image.shape[:2]):
                return {
                    'mask': processed_mask,
                    'boundary': boundary,
                    'contours': contours
                }
        
        # Tentar usar o detector de fallback se disponível
        if self.fallback_detector is not None:
            try:
                return self.fallback_detector.detect_field(image)
            except Exception as e:
                print(f"Erro no detector de fallback: {e}")
        
        # Se não houver segmentação de campo ou a área for muito pequena, retornar máscara vazia
        empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        return {
            'mask': empty_mask,
            'boundary': empty_mask.copy(),
            'contours': []
        }
    
    def _process_mask(self, mask, image_shape):
        """
        Processa a máscara de campo para melhorar a qualidade e extrair a fronteira.
        
        Args:
            mask: Máscara binária do campo
            image_shape: Forma da imagem original (altura, largura)
            
        Returns:
            Tupla (máscara processada, imagem da fronteira, contornos)
        """
        # Garantir que a máscara seja binária
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Aplicar operações morfológicas para melhorar a qualidade
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos do campo
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Criar imagem da fronteira
        boundary = np.zeros(image_shape, dtype=np.uint8)
        
        # Se houver contornos, desenhar o maior contorno como fronteira
        if contours:
            # Encontrar o maior contorno (assumindo que é o campo)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Desenhar o contorno na imagem da fronteira
            cv2.drawContours(boundary, [largest_contour], 0, 255, 2)
            
            # Manter apenas o maior contorno
            contours = [largest_contour]
        
        return processed_mask, boundary, contours
    
    def _check_field_area(self, mask, image_shape):
        """
        Verifica se a área do campo é suficiente.
        
        Args:
            mask: Máscara binária do campo
            image_shape: Forma da imagem original (altura, largura)
            
        Returns:
            True se a área do campo for suficiente, False caso contrário
        """
        # Calcular a área do campo (número de pixels brancos)
        field_area = cv2.countNonZero(mask)
        
        # Calcular a área total da imagem
        total_area = image_shape[0] * image_shape[1]
        
        # Verificar se a razão da área do campo é suficiente
        return field_area / total_area >= self.min_field_area_ratio
    
    def draw_segmentation(self, image, results):
        """
        Desenha a segmentação do campo na imagem para visualização.
        
        Args:
            image: Imagem BGR original
            results: Resultados da segmentação de campo
            
        Returns:
            Imagem com visualização da segmentação
        """
        if results is None or 'mask' not in results or results['mask'] is None:
            return image
        
        # Criar cópia da imagem
        vis_image = image.copy()
        
        # Desenhar máscara de campo com transparência
        field_overlay = np.zeros_like(image)
        field_overlay[results['mask'] > 0] = [0, 255, 0]  # Verde para campo
        
        # Combinar com a imagem original
        cv2.addWeighted(field_overlay, 0.3, vis_image, 1.0, 0, vis_image)
        
        # Desenhar fronteira do campo
        if 'boundary' in results and results['boundary'] is not None:
            # Encontrar pontos da fronteira
            boundary_points = np.where(results['boundary'] > 0)
            
            # Desenhar pontos da fronteira
            for y, x in zip(boundary_points[0], boundary_points[1]):
                cv2.circle(vis_image, (x, y), 1, (0, 0, 255), -1)
        
        # Desenhar contornos
        if 'contours' in results and results['contours']:
            cv2.drawContours(vis_image, results['contours'], -1, (255, 0, 0), 2)
        
        return vis_image
    
    # Alias para compatibilidade com a interface comum
    draw_detections = draw_segmentation 