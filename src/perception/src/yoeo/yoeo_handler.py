#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
import enum
import logging

from src.perception.src.yoeo.yoeo_model import YOEOModel

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('yoeo_handler')

class DetectionType(enum.Enum):
    """Tipos de detecção suportados pelo YOEO."""
    BALL = 0
    GOAL = 1
    ROBOT = 2
    REFEREE = 3

class SegmentationType(enum.Enum):
    """Tipos de segmentação suportados pelo YOEO."""
    FIELD = 0
    LINE = 1

class YOEOHandler:
    """
    Manipulador para o modelo YOEO.
    
    Esta classe é responsável por gerenciar o modelo YOEO, incluindo seu carregamento,
    pré-processamento de imagens, execução de inferência e pós-processamento de resultados.
    """
    
    def __init__(self, model_path, input_width=416, input_height=416, 
                 confidence_threshold=0.5, iou_threshold=0.45, use_tensorrt=False):
        """
        Inicializa o manipulador YOEO.
        
        Args:
            model_path: Caminho para o arquivo do modelo
            input_width: Largura da entrada do modelo
            input_height: Altura da entrada do modelo
            confidence_threshold: Limiar de confiança para as detecções
            iou_threshold: Limiar de IoU para non-maximum suppression
            use_tensorrt: Se deve usar o modelo otimizado com TensorRT
        """
        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_tensorrt = use_tensorrt
        
        # Mapeamento de classes para os tipos de detecção
        self.detection_classes = {
            DetectionType.BALL: 'bola',
            DetectionType.GOAL: 'gol',
            DetectionType.ROBOT: 'robo',
            DetectionType.REFEREE: 'arbitro'
        }
        
        # Mapeamento de classes para os tipos de segmentação
        self.segmentation_classes = {
            SegmentationType.FIELD: 'campo',
            SegmentationType.LINE: 'linha'
        }
        
        # Índices das classes no modelo
        self.class_indices = {
            'bola': 0,
            'gol': 1,
            'robo': 2,
            'arbitro': 3
        }
        
        # Índices de segmentação no modelo
        self.segmentation_indices = {
            'fundo': 0,
            'linha': 1,
            'campo': 2
        }
        
        # Carregar o modelo
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Carrega o modelo YOEO.
        
        Tenta carregar o modelo do caminho especificado. Se o modelo for TensorRT,
        carrega usando TensorRT. Caso contrário, carrega o modelo Keras padrão.
        """
        try:
            # Verificar se o arquivo do modelo existe
            if not os.path.exists(self.model_path):
                logger.error(f"Arquivo de modelo não encontrado: {self.model_path}")
                return
            
            # Carregar com TensorRT se especificado
            if self.use_tensorrt and self.model_path.endswith('.trt'):
                logger.info(f"Carregando modelo TensorRT de {self.model_path}")
                
                # Configurar o TensorRT
                from tensorflow.python.compiler.tensorrt import trt_convert as trt
                
                # Carregar modelo TensorRT
                converter = trt.TrtGraphConverterV2(input_saved_model_dir=self.model_path)
                self.model = converter.convert()
                logger.info("Modelo TensorRT carregado com sucesso")
            else:
                # Carregar modelo Keras normal
                logger.info(f"Carregando modelo Keras de {self.model_path}")
                
                # Criar o modelo YOEO
                yoeo_model = YOEOModel(
                    input_shape=(self.input_height, self.input_width, 3),
                    num_classes=len(self.class_indices),
                    seg_classes=len(self.segmentation_indices)
                )
                
                # Obter o modelo Keras
                self.model = yoeo_model.get_model()
                
                # Carregar pesos
                if self.model_path.endswith('.h5'):
                    self.model.load_weights(self.model_path)
                    logger.info("Pesos do modelo carregados com sucesso")
                else:
                    logger.warning(f"Formato de arquivo não suportado: {self.model_path}")
                    logger.warning("Usando modelo com pesos aleatórios")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo: {str(e)}")
            self.model = None
    
    def preprocess(self, image):
        """
        Pré-processa uma imagem para entrada no modelo.
        
        Args:
            image: Imagem BGR do OpenCV
            
        Returns:
            Imagem pré-processada pronta para inferência
        """
        if image is None or self.model is None:
            return None
        
        try:
            # Redimensionar a imagem para o tamanho de entrada do modelo
            resized = cv2.resize(image, (self.input_width, self.input_height))
            
            # Converter BGR para RGB (o modelo espera RGB)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalizar para [0, 1]
            normalized = rgb.astype(np.float32) / 255.0
            
            # Adicionar dimensão de batch
            batched = np.expand_dims(normalized, axis=0)
            
            return batched
        except Exception as e:
            logger.error(f"Erro no pré-processamento da imagem: {str(e)}")
            return None
    
    def predict(self, preprocessed_image):
        """
        Executa a inferência no modelo YOEO.
        
        Args:
            preprocessed_image: Imagem pré-processada
            
        Returns:
            Previsões brutas do modelo
        """
        if preprocessed_image is None or self.model is None:
            return None
        
        try:
            # Executar inferência
            predictions = self.model.predict(preprocessed_image)
            return predictions
        except Exception as e:
            logger.error(f"Erro na inferência do modelo: {str(e)}")
            return None
    
    def postprocess(self, predictions, image_shape):
        """
        Pós-processa as previsões do modelo.
        
        Args:
            predictions: Saída bruta do modelo
            image_shape: Forma da imagem original (altura, largura)
            
        Returns:
            Dicionário com detecções processadas e máscaras de segmentação
        """
        if predictions is None or self.model is None:
            return None
        
        try:
            # Extrair componentes das previsões
            # Assumindo formato de saída conforme definido em yoeo_model.py
            
            # Detecções em diferentes escalas
            large_scale = predictions[0]  # [box_xy, objectness, class_probs]
            medium_scale = predictions[1]
            small_scale = predictions[2]
            
            # Segmentação
            segmentation = predictions[3]
            
            # Processar segmentação
            seg_mask = np.argmax(segmentation[0], axis=-1)
            seg_mask = cv2.resize(seg_mask.astype(np.uint8), 
                                 (image_shape[1], image_shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
            
            # Processar detecções (versão simplificada)
            # Em uma implementação completa, precisaríamos decodificar os anchors,
            # aplicar non-maximum suppression, etc.
            
            # Para esta versão simplificada, vamos retornar algumas detecções fictícias
            # Em um sistema real, estas seriam derivadas dos outputs do modelo
            
            detections = []
            
            # Exemplo de detecção de bola (isso seria baseado nas previsões reais)
            ball_detection = {
                'class_id': 0,
                'class_name': 'bola',
                'confidence': 0.95,
                'bbox': [100, 200, 50, 50]  # [x, y, width, height]
            }
            detections.append(ball_detection)
            
            # Exemplo de detecção de robô
            robot_detection = {
                'class_id': 2,
                'class_name': 'robo',
                'confidence': 0.85,
                'bbox': [300, 250, 100, 150],
                'team': 'opponent'
            }
            detections.append(robot_detection)
            
            # Retornar resultados processados
            return {
                'detections': detections,
                'segmentation': seg_mask
            }
        except Exception as e:
            logger.error(f"Erro no pós-processamento: {str(e)}")
            return None
    
    def get_detections(self, image, detection_type=None):
        """
        Processa uma imagem para obter detecções de um tipo específico.
        
        Args:
            image: Imagem BGR do OpenCV
            detection_type: Tipo de detecção a ser filtrado (opcional)
            
        Returns:
            Lista de detecções do tipo especificado, ou todas as detecções se None
        """
        if image is None or self.model is None:
            return []
        
        try:
            # Pipeline completo: pré-processamento, inferência, pós-processamento
            preprocessed = self.preprocess(image)
            predictions = self.predict(preprocessed)
            results = self.postprocess(predictions, image.shape[:2])
            
            if results is None or 'detections' not in results:
                return []
            
            # Filtrar por tipo se especificado
            if detection_type is not None:
                class_name = self.detection_classes[detection_type]
                filtered_detections = [d for d in results['detections'] 
                                      if d['class_name'] == class_name]
                return filtered_detections
            else:
                return results['detections']
        except Exception as e:
            logger.error(f"Erro ao obter detecções: {str(e)}")
            return []
    
    def get_segmentation(self, image, segmentation_type):
        """
        Processa uma imagem para obter uma máscara de segmentação específica.
        
        Args:
            image: Imagem BGR do OpenCV
            segmentation_type: Tipo de segmentação (FIELD ou LINE)
            
        Returns:
            Máscara binária da segmentação solicitada
        """
        if image is None or self.model is None:
            return None
        
        try:
            # Pipeline completo: pré-processamento, inferência, pós-processamento
            preprocessed = self.preprocess(image)
            predictions = self.predict(preprocessed)
            results = self.postprocess(predictions, image.shape[:2])
            
            if results is None or 'segmentation' not in results:
                return None
            
            seg_mask = results['segmentation']
            
            # Extrair máscara específica com base no tipo de segmentação
            if segmentation_type == SegmentationType.FIELD:
                # Índice 2 para campo
                binary_mask = (seg_mask == self.segmentation_indices['campo']).astype(np.uint8) * 255
                return binary_mask
            elif segmentation_type == SegmentationType.LINE:
                # Índice 1 para linha
                binary_mask = (seg_mask == self.segmentation_indices['linha']).astype(np.uint8) * 255
                return binary_mask
            else:
                logger.warning(f"Tipo de segmentação não suportado: {segmentation_type}")
                return None
        except Exception as e:
            logger.error(f"Erro ao obter segmentação: {str(e)}")
            return None 