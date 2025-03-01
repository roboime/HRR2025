#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from src.perception.src.yoeo.yoeo_handler import SegmentationType

class FieldSegmentationComponent:
    """
    Componente para segmentação do campo usando YOEO.
    
    Este componente é responsável por:
    1. Extrair a máscara de segmentação do campo das saídas do YOEO
    2. Processar a máscara para melhorar a qualidade
    3. Detectar a fronteira do campo
    4. Fornecer visualizações para debugging
    """
    
    def __init__(self, yoeo_handler, min_field_area_ratio=0.1):
        """
        Inicializa o componente de segmentação do campo.
        
        Args:
            yoeo_handler: Instância de YOEOHandler para processar imagens
            min_field_area_ratio: Área mínima que o campo deve ocupar na imagem
        """
        self.yoeo_handler = yoeo_handler
        self.min_field_area_ratio = min_field_area_ratio
    
    def process(self, image):
        """
        Processa a imagem e retorna a segmentação do campo.
        
        Args:
            image: Imagem OpenCV no formato BGR
            
        Returns:
            Dicionário contendo 'mask', 'boundary' e 'contours'
        """
        # Obter detecções do YOEO
        results = self.yoeo_handler.get_detections(image)
        
        # Extrair apenas a segmentação do campo
        field_mask = results['segmentations'].get(SegmentationType.FIELD, None)
        
        # Se não houver segmentação do campo, tentar com detector tradicional
        if field_mask is None and hasattr(self, 'fallback_detector'):
            # Este é um ponto onde podemos integrar o detector tradicional
            pass
        
        # Se ainda não temos uma máscara, criar uma vazia
        if field_mask is None:
            field_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            return {
                'mask': field_mask,
                'boundary': np.zeros_like(field_mask),
                'contours': []
            }
        
        # Processar a máscara para melhorar a qualidade
        processed_mask = self._process_mask(field_mask)
        
        # Verificar se a área do campo é suficiente
        field_area_ratio = np.sum(processed_mask > 0) / (processed_mask.shape[0] * processed_mask.shape[1])
        
        if field_area_ratio < self.min_field_area_ratio:
            # Área do campo muito pequena, retornar resultado vazio
            return {
                'mask': processed_mask,
                'boundary': np.zeros_like(processed_mask),
                'contours': []
            }
        
        # Encontrar contornos e fronteira do campo
        contours, boundary = self._find_field_boundary(processed_mask)
        
        return {
            'mask': processed_mask,
            'boundary': boundary,
            'contours': contours
        }
    
    def _process_mask(self, mask):
        """
        Processa a máscara para melhorar a qualidade.
        
        Args:
            mask: Máscara binária do campo
            
        Returns:
            Máscara processada
        """
        # Garantir que a máscara seja binária
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Aplicar operações morfológicas para remover ruído
        kernel = np.ones((5, 5), np.uint8)
        processed = cv2.erode(binary_mask, kernel, iterations=1)
        processed = cv2.dilate(processed, kernel, iterations=2)
        
        return processed
    
    def _find_field_boundary(self, mask):
        """
        Encontra os contornos e a fronteira do campo.
        
        Args:
            mask: Máscara binária do campo
            
        Returns:
            (contornos, fronteira)
        """
        # Encontrar contornos na máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Criar imagem para a fronteira
        boundary = np.zeros_like(mask)
        
        # Se não houver contornos, retornar vazio
        if not contours:
            return [], boundary
        
        # Encontrar o maior contorno (que deve ser o campo)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Desenhar a fronteira
        cv2.drawContours(boundary, [largest_contour], 0, 255, 2)
        
        return [largest_contour], boundary
    
    def draw_segmentation(self, image, field_result):
        """
        Desenha a segmentação do campo na imagem.
        
        Args:
            image: Imagem OpenCV no formato BGR
            field_result: Resultado da segmentação do campo
            
        Returns:
            Imagem com a segmentação desenhada
        """
        debug_image = image.copy()
        
        # Criar sobreposição colorida
        overlay = np.zeros_like(debug_image)
        
        # Colorir a máscara do campo em verde semi-transparente
        mask = field_result['mask']
        overlay[mask > 0] = [0, 150, 0]  # Verde
        
        # Desenhar a fronteira em branco
        boundary = field_result['boundary']
        debug_image[boundary > 0] = [255, 255, 255]  # Branco
        
        # Combinar a imagem original com a sobreposição
        alpha = 0.4  # Opacidade da sobreposição
        cv2.addWeighted(overlay, alpha, debug_image, 1 - alpha, 0, debug_image)
        
        # Adicionar informações sobre a área do campo
        if mask.size > 0:
            field_area_ratio = np.sum(mask > 0) / mask.size
            cv2.putText(debug_image, f'Campo: {field_area_ratio:.2f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return debug_image
    
    def set_fallback_detector(self, detector):
        """
        Define um detector tradicional para ser usado como fallback.
        
        Args:
            detector: Instância de um detector tradicional
        """
        self.fallback_detector = detector 