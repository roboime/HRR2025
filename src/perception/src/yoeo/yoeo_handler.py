#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manipulador do modelo YOEO.

Este módulo fornece classes e funções para gerenciar o modelo YOEO,
processar suas saídas e fornecer uma interface unificada para os
componentes de detecção e segmentação.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from enum import Enum, auto
import logging
import time

from src.perception.src.yoeo.yoeo_model import YOEOModel

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('yoeo_handler')

class DetectionType(Enum):
    """Tipos de objetos que o modelo YOEO pode detectar."""
    BALL = auto()
    GOAL = auto()
    ROBOT = auto()
    REFEREE = auto()

class SegmentationType(Enum):
    """Tipos de segmentação que o modelo YOEO pode realizar."""
    FIELD = auto()
    LINE = auto()

class YOEOHandler:
    """
    Manipulador para o modelo YOEO.
    
    Esta classe é responsável por:
    1. Carregar e gerenciar o modelo YOEO
    2. Pré-processar imagens para entrada no modelo
    3. Processar as saídas do modelo para detecção e segmentação
    4. Fornecer uma interface unificada para os componentes
    """
    
    def __init__(self, model_path, input_size=(416, 416), confidence_threshold=0.5, iou_threshold=0.45):
        """
        Inicializa o manipulador YOEO.
        
        Args:
            model_path: Caminho para o modelo YOEO salvo
            input_size: Tamanho de entrada do modelo (largura, altura)
            confidence_threshold: Limiar de confiança para detecções
            iou_threshold: Limiar de IoU para supressão não-máxima
        """
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_mapping = {
            0: DetectionType.BALL,
            1: DetectionType.GOAL,
            2: DetectionType.ROBOT,
            3: DetectionType.REFEREE
        }
        self.segmentation_mapping = {
            0: None,  # Fundo (ignorado)
            1: SegmentationType.FIELD,
            2: SegmentationType.LINE
        }
        
        # Carregar o modelo
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo YOEO do caminho especificado."""
        try:
            print(f"Carregando modelo YOEO de {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            
            # Verificar se o modelo tem as saídas esperadas
            expected_outputs = 2  # Detecção e segmentação
            if len(self.model.outputs) != expected_outputs:
                print(f"Aviso: O modelo tem {len(self.model.outputs)} saídas, mas esperávamos {expected_outputs}")
            
            # Aquecer o modelo com uma inferência em dados fictícios
            dummy_input = np.zeros((1, *self.input_size, 3), dtype=np.float32)
            _ = self.model.predict(dummy_input)
            
            print("Modelo YOEO carregado com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar o modelo YOEO: {e}")
            raise
    
    def preprocess_image(self, image):
        """
        Pré-processa a imagem para entrada no modelo YOEO.
        
        Args:
            image: Imagem BGR do OpenCV
            
        Returns:
            Imagem pré-processada e fator de escala
        """
        # Salvar as dimensões originais
        original_height, original_width = image.shape[:2]
        
        # Redimensionar a imagem para o tamanho de entrada do modelo
        resized_image = cv2.resize(image, self.input_size)
        
        # Converter BGR para RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalizar para [0, 1]
        normalized_image = rgb_image.astype(np.float32) / 255.0
        
        # Adicionar dimensão de lote
        input_image = np.expand_dims(normalized_image, axis=0)
        
        # Calcular fatores de escala para converter coordenadas de volta para a imagem original
        scale_x = original_width / self.input_size[0]
        scale_y = original_height / self.input_size[1]
        
        return input_image, (scale_x, scale_y)
    
    def get_detections(self, image, detection_types=None, segmentation_types=None):
        """
        Obter as detecções e segmentações de uma imagem.
        
        Args:
            image: Imagem (numpy array BGR)
            detection_types: Lista de tipos de detecção para retornar (None para todos)
            segmentation_types: Lista de tipos de segmentação para retornar (None para todos)
            
        Returns:
            dict: Dicionário com detecções, segmentações e FPS
        """
        # Pré-processar a imagem
        input_image, scale_factors = self.preprocess_image(image)
        
        # A dimensão do batch já foi adicionada no método preprocess_image
        
        # Medir o tempo de inferência
        start_time = time.time()
        
        # Executar a inferência
        outputs = self.model.predict(input_image)
        
        # Calcular FPS
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        # Processar saídas
        detections = {}
        segmentations = {}
        
        # Verificar o formato das saídas (dicionário ou lista)
        if isinstance(outputs, dict):
            # Novo formato (dicionário)
            # Processar saída de detecção 
            # Usamos a detecção small pois tem melhor resolução para objetos pequenos
            detection_output = outputs["detection_small"]
            processed_detections = self._process_detection_output(detection_output, scale_factors, image.shape[:2])
            
            # Filtrar por tipos de detecção solicitados
            if detection_types is None:
                detections = processed_detections
            else:
                detections = {k: v for k, v in processed_detections.items() if k in detection_types}
            
            # Processar saída de segmentação
            segmentation_output = outputs["segmentation"]
            processed_segmentations = self._process_segmentation_output(segmentation_output, image.shape[:2])
            
            # Filtrar por tipos de segmentação solicitados
            if segmentation_types is None:
                segmentations = processed_segmentations
            else:
                segmentations = {k: v for k, v in processed_segmentations.items() if k in segmentation_types}
        else:
            # Formato antigo (lista)
            # Processar saída de detecção (assumindo formato YOLO)
            if len(outputs) >= 1:
                detection_output = outputs[0]
                processed_detections = self._process_detection_output(detection_output, scale_factors, image.shape[:2])
                
                # Filtrar por tipos de detecção solicitados
                if detection_types is None:
                    detections = processed_detections
                else:
                    detections = {k: v for k, v in processed_detections.items() if k in detection_types}
            
            # Processar saída de segmentação
            if len(outputs) >= 4:  # Agora são 4 saídas (3 detecções + 1 segmentação)
                segmentation_output = outputs[3]
                processed_segmentations = self._process_segmentation_output(segmentation_output, image.shape[:2])
                
                # Filtrar por tipos de segmentação solicitados
                if segmentation_types is None:
                    segmentations = processed_segmentations
                else:
                    segmentations = {k: v for k, v in processed_segmentations.items() if k in segmentation_types}
        
        # Retornar resultados
        return {
            "detections": detections,
            "segmentations": segmentations,
            "fps": fps
        }
    
    def _process_detection_output(self, detection_output, scale_factors, original_shape):
        """
        Processa a saída de detecção do modelo YOEO.
        
        Args:
            detection_output: Saída do modelo para detecção
            scale_factors: Fatores de escala (scale_x, scale_y)
            original_shape: Forma original da imagem (altura, largura)
            
        Returns:
            Dicionário com detecções por tipo
        """
        # Extrair fatores de escala
        scale_x, scale_y = scale_factors
        
        # Inicializar dicionário de detecções
        detections = {detection_type: [] for detection_type in DetectionType}
        
        # Processar cada detecção
        # Formato esperado: [batch, num_boxes, 5 + num_classes]
        # onde 5 = [x, y, w, h, confidence]
        detection_data = detection_output[0]  # Primeiro item do lote
        
        for box_data in detection_data:
            # Extrair confiança e verificar limiar
            confidence = box_data[4]
            if confidence < self.confidence_threshold:
                continue
            
            # Extrair probabilidades de classe
            class_probs = box_data[5:]
            class_id = np.argmax(class_probs)
            class_confidence = class_probs[class_id]
            
            # Verificar confiança da classe
            if class_confidence < self.confidence_threshold:
                continue
            
            # Mapear ID de classe para tipo de detecção
            if class_id not in self.class_mapping:
                continue
            
            detection_type = self.class_mapping[class_id]
            
            # Extrair coordenadas normalizadas (formato YOLO: x_center, y_center, width, height)
            x_center, y_center, width, height = box_data[:4]
            
            # Converter para coordenadas de pixel no formato [x1, y1, x2, y2]
            x1 = (x_center - width / 2) * self.input_size[0] * scale_x
            y1 = (y_center - height / 2) * self.input_size[1] * scale_y
            x2 = (x_center + width / 2) * self.input_size[0] * scale_x
            y2 = (y_center + height / 2) * self.input_size[1] * scale_y
            
            # Garantir que as coordenadas estejam dentro da imagem
            x1 = max(0, min(original_shape[1] - 1, x1))
            y1 = max(0, min(original_shape[0] - 1, y1))
            x2 = max(0, min(original_shape[1] - 1, x2))
            y2 = max(0, min(original_shape[0] - 1, y2))
            
            # Adicionar detecção à lista do tipo correspondente
            detections[detection_type].append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(class_confidence * confidence),
                'class_id': int(class_id)
            })
        
        # Aplicar supressão não-máxima para cada tipo de detecção
        for detection_type in detections:
            if detections[detection_type]:
                detections[detection_type] = self._apply_nms(detections[detection_type])
        
        return detections
    
    def _process_segmentation_output(self, segmentation_output, original_shape):
        """
        Processa a saída de segmentação do modelo YOEO.
        
        Args:
            segmentation_output: Saída do modelo para segmentação
            original_shape: Forma original da imagem (altura, largura)
            
        Returns:
            Dicionário com máscaras de segmentação por tipo
        """
        # Inicializar dicionário de segmentações
        segmentations = {}
        
        # Processar saída de segmentação
        # Formato esperado: [batch, height, width, num_classes]
        seg_data = segmentation_output[0]  # Primeiro item do lote
        
        # Obter mapa de classes (argmax ao longo do eixo de classes)
        class_map = np.argmax(seg_data, axis=-1)
        
        # Redimensionar para o tamanho original da imagem
        class_map_resized = cv2.resize(
            class_map.astype(np.uint8),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Criar máscaras para cada tipo de segmentação
        for class_id, seg_type in self.segmentation_mapping.items():
            if seg_type is None:  # Ignorar classe de fundo
                continue
            
            # Criar máscara binária para esta classe
            mask = np.zeros(original_shape[:2], dtype=np.uint8)
            mask[class_map_resized == class_id] = 255
            
            segmentations[seg_type] = mask
        
        return segmentations
    
    def _apply_nms(self, detections):
        """
        Aplica supressão não-máxima a um conjunto de detecções.
        
        Args:
            detections: Lista de detecções do mesmo tipo
            
        Returns:
            Lista de detecções após supressão não-máxima
        """
        # Extrair caixas delimitadoras e pontuações
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Converter de [x1, y1, x2, y2] para [y1, x1, y2, x2] (formato TensorFlow)
        boxes_tf = np.array([[box[1], box[0], box[3], box[2]] for box in boxes])
        
        # Aplicar NMS
        selected_indices = tf.image.non_max_suppression(
            boxes_tf, scores, max_output_size=100,
            iou_threshold=self.iou_threshold,
            score_threshold=self.confidence_threshold
        ).numpy()
        
        # Retornar detecções selecionadas
        return [detections[i] for i in selected_indices]
    
    def draw_results(self, image, results):
        """
        Desenha os resultados de detecção e segmentação na imagem.
        
        Args:
            image: Imagem BGR original
            results: Resultados de get_detections()
            
        Returns:
            Imagem com visualizações
        """
        # Criar cópia da imagem
        vis_image = image.copy()
        
        # Desenhar segmentações
        if 'segmentations' in results:
            # Criar sobreposição para segmentações
            overlay = np.zeros_like(vis_image)
            
            # Desenhar campo em verde
            if SegmentationType.FIELD in results['segmentations']:
                field_mask = results['segmentations'][SegmentationType.FIELD]
                overlay[field_mask > 0] = [0, 180, 0]  # Verde
            
            # Desenhar linhas em branco
            if SegmentationType.LINE in results['segmentations']:
                line_mask = results['segmentations'][SegmentationType.LINE]
                overlay[line_mask > 0] = [255, 255, 255]  # Branco
            
            # Combinar com a imagem original
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)
        
        # Desenhar detecções
        if 'detections' in results:
            # Cores para diferentes tipos de detecção
            colors = {
                DetectionType.BALL: (0, 165, 255),    # Laranja
                DetectionType.GOAL: (255, 255, 0),    # Ciano
                DetectionType.ROBOT: (0, 255, 0),     # Verde
                DetectionType.REFEREE: (255, 0, 255)  # Magenta
            }
            
            # Desenhar cada tipo de detecção
            for detection_type, detections in results['detections'].items():
                color = colors.get(detection_type, (255, 255, 255))
                
                for detection in detections:
                    # Extrair coordenadas e confiança
                    x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
                    confidence = detection['confidence']
                    
                    # Desenhar retângulo
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Adicionar rótulo
                    label = f"{detection_type.name}: {confidence:.2f}"
                    cv2.putText(vis_image, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Adicionar informações de FPS
        if 'fps' in results:
            fps_text = f"FPS: {results['fps']:.1f}"
            cv2.putText(vis_image, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_image 