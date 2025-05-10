#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manipulador do modelo YOLO para detecção de objetos.

Este módulo implementa um manipulador para o modelo YOLOv5,
que gerencia o pré-processamento de imagens, inferência e pós-processamento
para detecção de objetos em contexto de futebol robótico.
"""

import time
import numpy as np
import torch
import cv2
from enum import Enum, auto
import os
import sys

from perception.yoeo.yoeo_model import YOEOModel

# Verificar se o diretório do YOLOv5 foi definido
YOLOV5_PATH = os.environ.get('YOLOV5_PATH', '/opt/yolov5')
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)
    print(f"Adicionado {YOLOV5_PATH} ao sys.path")


class DetectionType(Enum):
    """Enumeração dos tipos de detecção suportados pelo modelo."""
    BALL = 0
    GOAL = 1
    ROBOT = 2


class YOEOHandler:
    """
    Manipulador para o modelo YOLOv5 para detecção de objetos.
    
    Esta classe gerencia todas as operações relacionadas ao modelo YOLOv5,
    incluindo carregamento do modelo, pré-processamento de imagens e inferência.
    """

    def __init__(self, config):
        """
        Inicializa o manipulador do modelo YOLOv5.
        
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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        print(f"DEBUG: Configuração do YOEOHandler:")
        print(f"DEBUG: - model_path: {self.model_path}")
        print(f"DEBUG: - input_width: {self.input_width}")
        print(f"DEBUG: - input_height: {self.input_height}")
        print(f"DEBUG: - confidence_threshold: {self.confidence_threshold}")
        print(f"DEBUG: - iou_threshold: {self.iou_threshold}")
        print(f"DEBUG: - use_tensorrt: {self.use_tensorrt}")
        print(f"DEBUG: - device: {self.device}")
        
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
        Carrega o modelo YOLOv5 a partir do caminho especificado.
        
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
            
            # Para o YOLOv5, precisamos configurar o modelo para inferência
            if hasattr(model, 'eval'):
                model.eval()
                
            # Configurar o modelo para utilizar o dispositivo correto
            if hasattr(model, 'to') and self.device:
                model.to(self.device)
                
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
    
    def detect(self, image):
        """
        Detecta objetos em uma imagem usando o modelo YOLOv5.
        
        Args:
            image: Imagem numpy BGR (OpenCV)
            
        Returns:
            Dicionário contendo:
                - boxes: Lista de caixas delimitadoras [x1, y1, x2, y2]
                - confidences: Lista de valores de confiança
                - classes: Lista de IDs de classe
        """
        # Realizar inferência e obter resultados processados
        processed_results, _ = self.process(image)
        
        # Organizar resultados em formato de lista
        boxes = []
        confidences = []
        classes = []
        
        for detection in processed_results:
            boxes.append(detection['bbox'])
            confidences.append(detection['confidence'])
            classes.append(detection['class'].value)
        
        return {
            'boxes': boxes,
            'confidences': confidences,
            'classes': classes
        }

    def process(self, image):
        """
        Processa uma imagem usando o modelo YOLOv5.
        
        Args:
            image: Imagem numpy BGR (OpenCV)
            
        Returns:
            Tupla contendo:
                - Lista de detecções, cada uma com (bbox, classe, confiança)
                - Tempo de inferência em segundos
        """
        if self.model is None:
            return [], 0.0
        
        # Pré-processar a imagem para o formato esperado pelo YOLOv5
        img = self._preprocess(image)
        
        # Medir o tempo de inferência
        start_time = time.time()
        
        try:
            # Verificar se estamos usando TensorRT
            if self.use_tensorrt and isinstance(self.model, object) and hasattr(self.model, 'engine'):
                # Processamento específico para TensorRT
                detections = self._process_tensorrt(img)
            else:
                # Inferência padrão com PyTorch
                with torch.no_grad():
                    # Realizar inferência
                    output = self.model(img)
                    
                    # Extrair detecções do output
                    detections = self._postprocess(output)
            
            # Calcular o tempo de inferência
            inference_time = time.time() - start_time
            
            return detections, inference_time
        
        except Exception as e:
            print(f"Erro durante a inferência: {e}")
            import traceback
            traceback.print_exc()
            return [], time.time() - start_time
    
    def _preprocess(self, image):
        """
        Pré-processa a imagem para inferência com o YOLOv5.
        
        Args:
            image: Imagem numpy BGR
            
        Returns:
            Tensor preprocessado
        """
        # Converter para RGB (YOLOv5 espera RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionar para o tamanho de entrada do modelo
        resized = cv2.resize(rgb_image, (self.input_width, self.input_height))
        
        # Converter para float, normalizar e reorganizar para formato NCHW
        img = resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # Adicionar dimensão de batch
        
        # Converter para tensor PyTorch
        img_tensor = torch.from_numpy(img).to(self.device)
        
        return img_tensor
    
    def _process_tensorrt(self, img):
        """
        Processa a imagem usando modelo TensorRT.
        
        Args:
            img: Tensor de imagem pré-processado
            
        Returns:
            Lista de detecções processadas
        """
        # Implementação específica para TensorRT
        # Pode precisar ser ajustada conforme sua implementação do TensorRT
        try:
            # Obter output do modelo TensorRT
            # Esta é uma implementação simplificada e deve ser ajustada conforme necessário
            output = self.model.context.execute_v2([img.cpu().numpy()])
            
            # Converter output para o formato correto e processar detecções
            # Note que o formato exato depende de como seu modelo TensorRT está configurado
            # Esta é apenas uma estrutura básica
            
            # Implementação exemplo - ajuste conforme sua configuração do TensorRT
            detections = []
            for i in range(len(output[0])):
                if output[1][i] > self.confidence_threshold:
                    x1, y1, x2, y2 = output[0][i][:4]
                    confidence = float(output[1][i])
                    class_id = int(output[2][i])
                    
                    # Converter para o formato de detecção esperado
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': DetectionType(class_id)
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Erro ao processar com TensorRT: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _postprocess(self, output):
        """
        Processa a saída do modelo YOLOv5 para obter detecções.
        
        Args:
            output: Saída do modelo YOLOv5
            
        Returns:
            Lista de detecções processadas
        """
        # Extrair detecções do output do YOLOv5
        # O formato exato pode variar dependendo da versão do YOLOv5
        try:
            # Se estamos usando a versão original do YOLOv5
            # output pode vir em diferentes formatos dependendo da versão
            
            # Verificar o tipo de output
            if isinstance(output, torch.Tensor):
                # Output é diretamente o tensor de previsão
                prediction = output
            elif isinstance(output, (tuple, list)):
                # Em algumas versões o output pode ser uma lista/tupla de tensores
                prediction = output[0]
            elif hasattr(output, 'pred'):
                # Em algumas versões o output é um objeto com atributo 'pred'
                prediction = output.pred[0]
            elif hasattr(output, 'xyxy'):
                # Em algumas versões o output é um objeto com métodos específicos
                # Converter para o formato xyxy (x1, y1, x2, y2)
                prediction = output.xyxy[0]
            else:
                print(f"Formato de output desconhecido: {type(output)}")
                return []
            
            # Converter para numpy array
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.cpu().numpy()
            
            # Processar detecções
            detections = []
            for det in prediction:
                # Formato esperado: [x1, y1, x2, y2, confidence, class_id]
                if len(det) >= 6:
                    x1, y1, x2, y2, confidence, class_id = det[:6]
                    
                    # Verificar se a confiança é maior que o limiar
                    if confidence > self.confidence_threshold:
                        # Verificar se a classe está dentro do range esperado
                        if 0 <= class_id < len(DetectionType):
                            detection = {
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(confidence),
                                'class': DetectionType(int(class_id))
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Erro ao processar detecções: {e}")
            import traceback
            traceback.print_exc()
            return [] 