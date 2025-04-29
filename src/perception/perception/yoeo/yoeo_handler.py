#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manipulador do modelo YOLOv4-Tiny para detecção de objetos.

Este módulo implementa um manipulador para o modelo YOLOv4-Tiny,
que gerencia o pré-processamento de imagens, inferência e pós-processamento
para detecção de objetos em contexto de futebol robótico.
"""

import time
import numpy as np
import tensorflow as tf
from enum import Enum, auto

from perception.yoeo.yoeo_model import YOEOModel


class DetectionType(Enum):
    """Enumeração dos tipos de detecção suportados pelo modelo."""
    BALL = 0
    GOAL = 1
    ROBOT = 2


class YOEOHandler:
    """
    Manipulador para o modelo YOLOv4-Tiny para detecção de objetos.
    
    Esta classe gerencia todas as operações relacionadas ao modelo YOLOv4-Tiny,
    incluindo carregamento do modelo, pré-processamento de imagens e inferência.
    """

    def __init__(self, config):
        """
        Inicializa o manipulador do modelo YOLOv4-Tiny.
        
        Args:
            config: Dicionário com configurações do modelo, incluindo:
                - model_path: Caminho para o arquivo do modelo
                - input_width: Largura da imagem de entrada para o modelo
                - input_height: Altura da imagem de entrada para o modelo
                - confidence_threshold: Limiar de confiança para filtrar detecções
                - iou_threshold: Limiar de IoU para non-maximum suppression
        """
        self.config = config
        self.model_path = config.get("model_path", "")
        self.input_width = config.get("input_width", 416)
        self.input_height = config.get("input_height", 416)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        
        # Carregar o modelo
        self.model = self._load_model()

    def _load_model(self):
        """
        Carrega o modelo YOLOv4-Tiny a partir do caminho especificado.
        
        Returns:
            Modelo carregado ou None em caso de erro
        """
        try:
            # Criar uma instância do modelo para detecção apenas
            yoeo_model = YOEOModel(
                input_shape=(self.input_height, self.input_width, 3),
                num_classes=len(DetectionType),
                detection_only=True
            )
            
            # Carregar pesos do modelo
            model = yoeo_model.load_weights(self.model_path)
            print(f"Modelo carregado com sucesso de: {self.model_path}")
            return model
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            print("Inicializando com um modelo vazio.")
            
            # Retornar um modelo vazio para evitar erros
            yoeo_model = YOEOModel(
                input_shape=(self.input_height, self.input_width, 3),
                num_classes=len(DetectionType),
                detection_only=True
            )
            return yoeo_model.build()

    def preprocess_image(self, image):
        """
        Pré-processa uma imagem para entrada no modelo.
        
        Args:
            image: Imagem numpy RGB
            
        Returns:
            Tensor preprocessado pronto para inferência
        """
        # Redimensionar a imagem para o tamanho de entrada do modelo
        input_image = tf.image.resize(image, (self.input_height, self.input_width))
        
        # Normalizar os valores dos pixels para [0, 1]
        input_image = input_image / 255.0
        
        # Adicionar dimensão de batch
        input_image = tf.expand_dims(input_image, 0)
        
        return input_image

    def process(self, image):
        """
        Processa uma imagem usando o modelo YOLOv4-Tiny.
        
        Args:
            image: Imagem numpy RGB
            
        Returns:
            Tupla contendo:
                - Lista de detecções, cada uma com (bbox, classe, confiança)
                - Tempo de inferência em segundos
        """
        if self.model is None:
            return [], 0.0
        
        # Pré-processar a imagem
        input_tensor = self.preprocess_image(image)
        
        # Medir o tempo de inferência
        start_time = time.time()
        
        # Realizar a inferência
        detections = self.model.predict(input_tensor)
        
        # Calcular o tempo de inferência
        inference_time = time.time() - start_time
        
        # Processar as saídas do YOLO para obter as detecções
        boxes, scores, classes = self._process_yolo_outputs(detections, image.shape[:2])
        
        # Filtrar por confiança
        mask = scores >= self.confidence_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_classes = classes[mask]
        
        # Converter para formato de detecções
        detection_results = []
        for box, score, class_id in zip(filtered_boxes, filtered_scores, filtered_classes):
            detection_results.append({
                'bbox': box,  # [x1, y1, x2, y2]
                'class': DetectionType(int(class_id)),
                'confidence': float(score)
            })
        
        return detection_results, inference_time

    def _process_yolo_outputs(self, yolo_outputs, original_shape):
        """
        Processa as saídas do modelo YOLO para obter caixas delimitadoras, 
        pontuações e classes.
        
        Args:
            yolo_outputs: Saídas do modelo YOLO
            original_shape: Forma da imagem original (altura, largura)
            
        Returns:
            Tupla com (boxes, scores, classes)
        """
        # Implementar o processamento das saídas do YOLO
        # Esta é uma implementação simplificada
        
        # Anchors para as duas escalas
        anchors = {
            'small': [[23, 27], [37, 58], [81, 82]],      # 26x26
            'large': [[81, 82], [135, 169], [344, 319]]   # 13x13
        }
        
        # Para simplificar, vamos assumir que temos boxes, scores e classes já processados
        # Em uma implementação real, você processaria as saídas do YOLO para obter esses valores
        
        # Converter coordenadas relativas para absolutas
        original_height, original_width = original_shape
        
        # Processar cada saída do YOLO (escalas diferentes)
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for output_idx, output in enumerate(yolo_outputs):
            # Extrair informações da forma da saída
            batch_size, grid_h, grid_w, num_anchors, box_attribs = output.shape
            
            # Selecionar os anchors apropriados para esta escala
            current_anchors = anchors['small'] if output_idx == 0 else anchors['large']
            
            # Extrair coordenadas, confiança e classes
            # Na prática, isso seria implementado com operações tensoriais
            for b in range(batch_size):
                for i in range(grid_h):
                    for j in range(grid_w):
                        for a in range(num_anchors):
                            # Obter confiança do objeto
                            objectness = output[b, i, j, a, 4]
                            
                            if objectness > self.confidence_threshold:
                                # Obter coordenadas x, y, w, h
                                x = (output[b, i, j, a, 0] + j) / grid_w
                                y = (output[b, i, j, a, 1] + i) / grid_h
                                w = np.exp(output[b, i, j, a, 2]) * current_anchors[a][0] / self.input_width
                                h = np.exp(output[b, i, j, a, 3]) * current_anchors[a][1] / self.input_height
                                
                                # Converter para coordenadas absolutas [x1, y1, x2, y2]
                                x1 = max(0, (x - w/2) * original_width)
                                y1 = max(0, (y - h/2) * original_height)
                                x2 = min(original_width, (x + w/2) * original_width)
                                y2 = min(original_height, (y + h/2) * original_height)
                                
                                # Obter as probabilidades de classe
                                class_probs = output[b, i, j, a, 5:]
                                class_id = np.argmax(class_probs)
                                confidence = objectness * class_probs[class_id]
                                
                                if confidence > self.confidence_threshold:
                                    all_boxes.append([x1, y1, x2, y2])
                                    all_scores.append(confidence)
                                    all_classes.append(class_id)
        
        # Converter para arrays numpy
        boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        classes = np.array(all_classes)
        
        # Aplicar non-maximum suppression
        if len(boxes) > 0:
            # Calcular índices de sobreposição
            indices = tf.image.non_max_suppression(
                boxes=boxes,
                scores=scores,
                max_output_size=100,
                iou_threshold=self.iou_threshold
            ).numpy()
            
            return boxes[indices], scores[indices], classes[indices]
        
        return np.array([]), np.array([]), np.array([]) 