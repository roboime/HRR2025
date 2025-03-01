#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Componente para segmentação de linhas do campo usando o modelo YOEO.

Este componente é responsável por extrair e processar a segmentação
de linhas do campo a partir das saídas do modelo YOEO.
"""

import cv2
import numpy as np
from enum import Enum

from ..yoeo_handler import SegmentationType


class LineSegmentationComponent:
    """
    Componente para segmentação de linhas do campo.
    
    Este componente processa a saída de segmentação do modelo YOEO
    para extrair as linhas do campo, aplicar pós-processamento e
    fornecer informações úteis para localização.
    """
    
    def __init__(self, yoeo_handler, field_component=None, min_line_area=100):
        """
        Inicializa o componente de segmentação de linhas.
        
        Args:
            yoeo_handler: Manipulador do modelo YOEO
            field_component: Componente de segmentação de campo (opcional)
            min_line_area: Área mínima para considerar um contorno como linha
        """
        self.yoeo_handler = yoeo_handler
        self.field_component = field_component
        self.min_line_area = min_line_area
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
            detector: Detector tradicional de linhas
        """
        self.fallback_detector = detector
    
    def process(self, image):
        """
        Processa a imagem para segmentar as linhas do campo.
        
        Args:
            image: Imagem BGR
            
        Returns:
            Dicionário com resultados da segmentação de linhas:
            - mask: Máscara binária das linhas
            - contours: Contornos das linhas
            - intersections: Pontos de interseção entre linhas
        """
        # Obter segmentação do modelo YOEO
        segmentations = self.yoeo_handler.get_detections(
            image, detection_types=[], segmentation_types=[SegmentationType.LINE]
        )
        
        # Verificar se a segmentação de linha está disponível
        if SegmentationType.LINE in segmentations['segmentations']:
            line_mask = segmentations['segmentations'][SegmentationType.LINE]
            
            # Processar a máscara de linha
            processed_mask, contours, intersections = self._process_line_mask(line_mask, image.shape[:2])
            
            return {
                'mask': processed_mask,
                'contours': contours,
                'intersections': intersections
            }
        
        # Tentar usar o detector de fallback se disponível
        if self.fallback_detector is not None:
            try:
                return self.fallback_detector.detect_lines(image)
            except Exception as e:
                print(f"Erro no detector de fallback: {e}")
        
        # Se não houver segmentação de linha e nem fallback, retornar máscara vazia
        return {
            'mask': np.zeros(image.shape[:2], dtype=np.uint8),
            'contours': [],
            'intersections': []
        }
    
    def _process_line_mask(self, line_mask, image_shape):
        """
        Processa a máscara de linha para melhorar a qualidade e extrair informações.
        
        Args:
            line_mask: Máscara binária das linhas
            image_shape: Forma da imagem original (altura, largura)
            
        Returns:
            Tupla (máscara processada, contornos, interseções)
        """
        # Garantir que a máscara seja binária
        _, binary_mask = cv2.threshold(line_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Se o componente de campo estiver disponível, usar a máscara de campo para limitar as linhas
        if self.field_component is not None:
            field_result = self.field_component.process(np.zeros(image_shape + (3,), dtype=np.uint8))
            if 'mask' in field_result and field_result['mask'] is not None:
                # Limitar as linhas à área do campo
                binary_mask = cv2.bitwise_and(binary_mask, field_result['mask'])
        
        # Aplicar operações morfológicas para melhorar a qualidade
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos das linhas
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos muito pequenos
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_line_area]
        
        # Encontrar interseções de linhas (simplificado)
        intersections = self._find_line_intersections(processed_mask)
        
        return processed_mask, contours, intersections
    
    def _find_line_intersections(self, line_mask):
        """
        Encontra pontos de interseção entre linhas.
        
        Args:
            line_mask: Máscara binária das linhas
            
        Returns:
            Lista de pontos de interseção (x, y)
        """
        # Implementação simplificada para encontrar interseções
        # Em uma implementação real, seria necessário um algoritmo mais robusto
        
        # Aplicar detector de cantos Harris
        corners = cv2.cornerHarris(line_mask.astype(np.float32), 5, 3, 0.04)
        
        # Dilatar para marcar os cantos
        corners = cv2.dilate(corners, None)
        
        # Limiar para pontos de interseção
        threshold = 0.01 * corners.max()
        corner_points = np.where(corners > threshold)
        
        # Converter para lista de pontos (x, y)
        intersections = [(x, y) for y, x in zip(corner_points[0], corner_points[1])]
        
        # Agrupar pontos próximos (simplificado)
        if intersections:
            # Converter para array numpy para facilitar o processamento
            points = np.array(intersections)
            
            # Usar K-means para agrupar pontos próximos
            if len(points) > 10:  # Se houver muitos pontos, reduzir usando K-means
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                k = min(10, len(points) // 2)  # Número de clusters
                _, _, centers = cv2.kmeans(points.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                intersections = [(int(x), int(y)) for x, y in centers]
        
        return intersections
    
    def draw_segmentation(self, image, results):
        """
        Desenha a segmentação de linhas na imagem para visualização.
        
        Args:
            image: Imagem BGR original
            results: Resultados da segmentação de linhas
            
        Returns:
            Imagem com visualização da segmentação
        """
        if results is None or 'mask' not in results or results['mask'] is None:
            return image
        
        # Criar cópia da imagem
        vis_image = image.copy()
        
        # Desenhar máscara de linha com transparência
        line_overlay = np.zeros_like(image)
        line_overlay[results['mask'] > 0] = [0, 255, 255]  # Amarelo para linhas
        
        # Combinar com a imagem original
        cv2.addWeighted(line_overlay, 0.5, vis_image, 1.0, 0, vis_image)
        
        # Desenhar contornos
        if 'contours' in results and results['contours']:
            cv2.drawContours(vis_image, results['contours'], -1, (0, 255, 0), 2)
        
        # Desenhar interseções
        if 'intersections' in results and results['intersections']:
            for point in results['intersections']:
                cv2.circle(vis_image, point, 5, (255, 0, 0), -1)
        
        return vis_image
    
    # Alias para compatibilidade com a interface comum
    draw_detections = draw_segmentation 