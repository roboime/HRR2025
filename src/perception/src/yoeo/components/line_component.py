#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from src.perception.src.yoeo.yoeo_handler import YOEOHandler, SegmentationType

class LineSegmentationComponent:
    """
    Componente para segmentação de linhas do campo de futebol.
    
    Este componente é responsável por extrair a máscara de segmentação de linhas
    do campo a partir das saídas do YOEO, processar a máscara para melhorar a qualidade,
    extrair características como linhas e intersecções, e fornecer visualizações.
    """
    
    def __init__(self, yoeo_handler, min_line_length=15, max_line_gap=10):
        """
        Inicializa o componente de segmentação de linhas.
        
        Args:
            yoeo_handler: Instância do YOEOHandler para acesso ao modelo
            min_line_length: Comprimento mínimo de linha para detecção
            max_line_gap: Lacuna máxima entre segmentos de linha
        """
        self.yoeo_handler = yoeo_handler
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.fallback_detector = None
        self.camera_info = None

    def set_camera_info(self, camera_info):
        """
        Define as informações de câmera para cálculos de posição.
        
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
        Processa a imagem para extrair linhas do campo.
        
        Args:
            image: Imagem BGR do OpenCV
            
        Returns:
            Dicionário com resultados da segmentação de linhas: 
            {
                'mask': Máscara binária das linhas,
                'lines': Lista de linhas detectadas (coordenadas),
                'intersections': Lista de pontos de intersecção
            }
        """
        # Verificar se a imagem é válida
        if image is None or image.size == 0:
            return None
        
        # Tentar obter a segmentação do YOEO
        try:
            # Obter segmentação do YOEO
            segmentation = self.yoeo_handler.get_segmentation(image, SegmentationType.LINE)
            
            if segmentation is not None:
                # Processar a máscara de linhas
                processed_mask = self._process_mask(segmentation)
                
                # Extrair linhas da máscara processada
                lines, intersections = self._extract_line_features(processed_mask)
                
                return {
                    'mask': processed_mask,
                    'lines': lines,
                    'intersections': intersections
                }
            elif self.fallback_detector is not None:
                # Usar detector tradicional como fallback
                return self.fallback_detector.detect(image)
            else:
                # Retornar máscara vazia se não houver detecção
                return {
                    'mask': np.zeros(image.shape[:2], dtype=np.uint8),
                    'lines': [],
                    'intersections': []
                }
        except Exception as e:
            print(f"Erro na segmentação de linhas: {e}")
            return None
    
    def _process_mask(self, mask):
        """
        Processa a máscara de segmentação de linhas para melhorar a qualidade.
        
        Args:
            mask: Máscara binária das linhas
            
        Returns:
            Máscara processada
        """
        # Verificar se a máscara é válida
        if mask is None:
            return None
        
        # Aplicar operações morfológicas para melhorar a qualidade
        kernel = np.ones((3, 3), np.uint8)
        
        # Remover ruído com abertura morfológica
        mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Preencher pequenos buracos com fechamento morfológico
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
        
        # Adelgaçar as linhas para melhor detecção
        # mask_thinned = cv2.ximgproc.thinning(mask_closed)
        # Como cv2.ximgproc pode não estar disponível, usar erosão como alternativa
        mask_thinned = cv2.erode(mask_closed, kernel, iterations=1)
        
        return mask_thinned
    
    def _extract_line_features(self, mask):
        """
        Extrai características de linhas da máscara processada.
        
        Args:
            mask: Máscara binária processada
            
        Returns:
            Tupla (linhas, interseções)
        """
        # Verificar se a máscara é válida
        if mask is None:
            return [], []
        
        # Detectar linhas usando a transformada de Hough probabilística
        lines = cv2.HoughLinesP(
            mask, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30, 
            minLineLength=self.min_line_length, 
            maxLineGap=self.max_line_gap
        )
        
        # Converter para formato mais conveniente
        line_segments = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_segments.append(((x1, y1), (x2, y2)))
        
        # Encontrar interseções de linhas (simplificado)
        intersections = self._find_intersections(line_segments)
        
        return line_segments, intersections
    
    def _find_intersections(self, lines):
        """
        Encontra pontos de interseção entre as linhas.
        
        Args:
            lines: Lista de segmentos de linha
            
        Returns:
            Lista de pontos de interseção
        """
        intersections = []
        
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                # Obter linhas
                ((x1, y1), (x2, y2)) = lines[i]
                ((x3, y3), (x4, y4)) = lines[j]
                
                # Calcular determinante
                denominator = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
                
                # Verificar paralelismo (ou quase paralelismo)
                if denominator == 0:
                    continue
                
                # Calcular ponto de interseção
                ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
                ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator
                
                # Verificar se o ponto de interseção está nos segmentos de linha
                if 0 <= ua <= 1 and 0 <= ub <= 1:
                    x = int(x1 + ua * (x2 - x1))
                    y = int(y1 + ua * (y2 - y1))
                    intersections.append((x, y))
        
        return intersections
    
    def draw_segmentation(self, image, segmentation_result):
        """
        Desenha a segmentação e características de linhas na imagem.
        
        Args:
            image: Imagem original
            segmentation_result: Resultado da segmentação
            
        Returns:
            Imagem com visualização
        """
        if segmentation_result is None:
            return image
        
        # Criar cópia da imagem
        vis_image = image.copy()
        
        # Desenhar máscara como overlay
        if 'mask' in segmentation_result and segmentation_result['mask'] is not None:
            # Criar overlay
            line_overlay = np.zeros_like(vis_image)
            line_overlay[segmentation_result['mask'] > 0] = [255, 255, 0]  # Amarelo
            
            # Combinar overlay com a imagem original
            alpha = 0.3  # Transparência
            cv2.addWeighted(vis_image, 1, line_overlay, alpha, 0, vis_image)
        
        # Desenhar linhas detectadas
        if 'lines' in segmentation_result and segmentation_result['lines']:
            for (pt1, pt2) in segmentation_result['lines']:
                cv2.line(vis_image, pt1, pt2, (0, 0, 255), 2)  # Vermelho
        
        # Desenhar interseções
        if 'intersections' in segmentation_result and segmentation_result['intersections']:
            for pt in segmentation_result['intersections']:
                cv2.circle(vis_image, pt, 5, (0, 255, 0), -1)  # Verde
        
        # Adicionar texto informativo
        if 'lines' in segmentation_result:
            num_lines = len(segmentation_result['lines'])
            cv2.putText(vis_image, f"Linhas: {num_lines}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image 