#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manipulador do modelo YOLO da Ultralytics para detecção de objetos.

Este módulo implementa um manipulador para o modelo YOLO da Ultralytics,
que gerencia o pré-processamento de imagens, inferência e pós-processamento
para detecção de objetos em contexto de futebol robótico.
"""

import time
import numpy as np
import torch
import cv2
from enum import Enum, auto
import os

from perception.yoeo.yoeo_model import YOEOModel


class DetectionType(Enum):
    """Enumeração dos tipos de detecção suportados pelo modelo."""
    BALL = 0
    GOAL = 1
    ROBOT = 2


class YOEOHandler:
    """
    Manipulador para o modelo YOLO da Ultralytics para detecção de objetos.
    
    Esta classe gerencia todas as operações relacionadas ao modelo YOLO da Ultralytics,
    incluindo carregamento do modelo, pré-processamento de imagens e inferência.
    """

    def __init__(self, config):
        """
        Inicializa o manipulador do modelo YOLO da Ultralytics.
        
        Args:
            config: Dicionário com configurações do modelo, incluindo:
                - model_path: Caminho para o arquivo do modelo
                - input_width: Largura da imagem de entrada para o modelo
                - input_height: Altura da imagem de entrada para o modelo
                - confidence_threshold: Limiar de confiança para filtrar detecções
                - iou_threshold: Limiar de IoU para non-maximum suppression
        """
        print("DEBUG: Inicializando YOEOHandler")
        self.config = config
        self.model_path = config.get("model_path", "")
        self.input_width = config.get("input_width", 640)
        self.input_height = config.get("input_height", 640)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.use_tensorrt = config.get("use_tensorrt", True)
        
        print(f"DEBUG: Configuração do YOEOHandler:")
        print(f"DEBUG: - model_path: {self.model_path}")
        print(f"DEBUG: - input_width: {self.input_width}")
        print(f"DEBUG: - input_height: {self.input_height}")
        print(f"DEBUG: - confidence_threshold: {self.confidence_threshold}")
        print(f"DEBUG: - iou_threshold: {self.iou_threshold}")
        print(f"DEBUG: - use_tensorrt: {self.use_tensorrt}")
        
        # Verificar se o arquivo do modelo existe
        if not os.path.exists(self.model_path):
            print(f"DEBUG: ALERTA! Arquivo do modelo não encontrado: {self.model_path}")
            print(f"DEBUG: Diretório atual: {os.getcwd()}")
            print(f"DEBUG: Conteúdo do diretório resources/models (se existir):")
            models_dir = os.path.join("src", "perception", "resources", "models")
            if os.path.exists(models_dir):
                print(f"DEBUG: Listando {models_dir}:")
                for file in os.listdir(models_dir):
                    print(f"DEBUG:   - {file}")
            else:
                print(f"DEBUG: Diretório {models_dir} não existe")
        else:
            print(f"DEBUG: Arquivo do modelo encontrado: {self.model_path}")
        
        # Carregar o modelo
        print("DEBUG: Iniciando carregamento do modelo")
        self.model = self._load_model()
        print("DEBUG: Carregamento do modelo concluído")

    def _load_model(self):
        """
        Carrega o modelo YOLO da Ultralytics a partir do caminho especificado.
        
        Returns:
            Modelo carregado ou None em caso de erro
        """
        try:
            # Criar uma instância do modelo YOEOModel
            yoeo_model = YOEOModel(
                input_shape=(self.input_height, self.input_width, 3),
                num_classes=len(DetectionType),
                detection_only=True
            )
            
            # Carregar o modelo a partir do arquivo
            model = yoeo_model.load_weights(self.model_path)
            print(f"Modelo carregado com sucesso de: {self.model_path}")
            
            return model
            
        except Exception as e:
            print(f"DEBUG: Erro ao carregar o modelo: {e}")
            import traceback
            traceback_str = traceback.format_exc()
            print(f"DEBUG: Traceback do erro ao carregar modelo:\n{traceback_str}")
            print("DEBUG: Inicializando com um modelo vazio.")
            
            # Retornar um modelo vazio para evitar erros
            yoeo_model = YOEOModel(
                input_shape=(self.input_height, self.input_width, 3),
                num_classes=len(DetectionType),
                detection_only=True
            )
            return yoeo_model.build()

    def get_detections(self, image, detection_types=None, segmentation_types=None):
        """
        Obtém detecções a partir de uma imagem.
        
        Args:
            image: Imagem BGR
            detection_types: Lista de tipos de detecção para retornar
            segmentation_types: Lista de tipos de segmentação para retornar (não utilizado)
            
        Returns:
            Dicionário com detecções e segmentações (se houver)
        """
        # Se não for especificado tipos de detecção, usar todos
        if detection_types is None:
            detection_types = list(DetectionType)
        
        # Inicializar dicionário de resultados
        results = {
            'detections': {},
            'segmentations': {},
            'inference_time': 0.0
        }
        
        # Realizar inferência
        processed_results, inference_time = self.process(image)
        
        # Armazenar tempo de inferência
        results['inference_time'] = inference_time
        
        # Filtrar resultados por tipo
        for detection_type in detection_types:
            results['detections'][detection_type] = []
            
            for detection in processed_results:
                if detection['class'] == detection_type:
                    results['detections'][detection_type].append(detection)
        
        return results

    def process(self, image):
        """
        Processa uma imagem usando o modelo YOLO da Ultralytics.
        
        Args:
            image: Imagem numpy BGR (OpenCV)
            
        Returns:
            Tupla contendo:
                - Lista de detecções, cada uma com (bbox, classe, confiança)
                - Tempo de inferência em segundos
        """
        if self.model is None:
            return [], 0.0
        
        # Converter para RGB se necessário (o YOLO da Ultralytics espera RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Medir o tempo de inferência
        start_time = time.time()
        
        # Realizar a inferência com o modelo Ultralytics
        # O método 'predict' da Ultralytics já lida com o redimensionamento,
        # pré-processamento, e pós-processamento das detecções
        results = self.model.predict(
            source=rgb_image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=100,  # Máximo de detecções por imagem
            verbose=False
        )
        
        # Calcular o tempo de inferência
        inference_time = time.time() - start_time
        
        # Processar os resultados para nosso formato padrão
        detection_results = []
        
        # A Ultralytics retorna uma lista de resultados (um por imagem)
        # Como estamos processando apenas uma imagem, pegamos o primeiro resultado
        result = results[0]
        
        # Extrair as caixas (em pixels), classes e pontuações
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                # Obter coordenadas x1, y1, x2, y2 em formato absoluto (pixels)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Obter classe e confiança
                class_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Mapear índice de classe Ultralytics para nosso DetectionType
                # Assumindo que as classes são mapeadas assim: 0=Bola, 1=Gol, 2=Robô
                if class_id < len(DetectionType):
                    detection_type = DetectionType(class_id)
                else:
                    # Se tivermos alguma classe fora do nosso enum, ignorar
                    continue
                
                # Adicionar à lista de detecções
                detection_results.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class': detection_type,
                    'confidence': conf
                })
        
        return detection_results, inference_time 