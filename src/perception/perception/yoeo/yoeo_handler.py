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
import os

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
        print("DEBUG: Inicializando YOEOHandler")
        self.config = config
        self.model_path = config.get("model_path", "")
        self.input_width = config.get("input_width", 224)
        self.input_height = config.get("input_height", 224)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.use_tensorrt = config.get("use_tensorrt", False)
        self.is_onnx_model = config.get("is_onnx_model", False)
        
        print(f"DEBUG: Configuração do YOEOHandler:")
        print(f"DEBUG: - model_path: {self.model_path}")
        print(f"DEBUG: - input_width: {self.input_width}")
        print(f"DEBUG: - input_height: {self.input_height}")
        print(f"DEBUG: - confidence_threshold: {self.confidence_threshold}")
        print(f"DEBUG: - iou_threshold: {self.iou_threshold}")
        print(f"DEBUG: - use_tensorrt: {self.use_tensorrt}")
        print(f"DEBUG: - is_onnx_model: {self.is_onnx_model}")
        
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
        Carrega o modelo YOLOv4-Tiny a partir do caminho especificado.
        
        Returns:
            Modelo carregado ou None em caso de erro
        """
        try:
            # Verificar se é um arquivo ONNX (.onnx)
            if self.is_onnx_model or self.model_path.lower().endswith('.onnx'):
                print("DEBUG: Arquivo ONNX (.onnx) detectado, configurando para carregar ONNX")
                self.is_onnx_model = True
                self.use_tensorrt = False  # Desativar TensorRT para ONNX
                
                try:
                    # Carregar modelo ONNX usando tensorflow-onnx ou onnxruntime
                    import onnxruntime as ort
                    print(f"DEBUG: Carregando modelo ONNX usando ONNX Runtime")
                    
                    # Configurar opções de sessão para otimização
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    
                    # Verificar os provedores disponíveis
                    providers = ort.get_available_providers()
                    print(f"DEBUG: Provedores ONNX disponíveis: {providers}")
                    
                    # Configurar provedores
                    provider_list = []
                    if 'CUDAExecutionProvider' in providers:
                        provider_list.append('CUDAExecutionProvider')
                    if 'CPUExecutionProvider' in providers:
                        provider_list.append('CPUExecutionProvider')
                    
                    # Criar sessão ONNX
                    onnx_session = ort.InferenceSession(
                        self.model_path, 
                        sess_options=sess_options,
                        providers=provider_list
                    )
                    print("DEBUG: Sessão ONNX criada com sucesso")
                    
                    # Obter informações do modelo
                    input_details = onnx_session.get_inputs()
                    output_details = onnx_session.get_outputs()
                    print(f"DEBUG: Detalhes de entrada: {input_details}")
                    print(f"DEBUG: Detalhes de saída: {output_details}")
                    
                    # Criar uma classe wrapper para compatibilidade com a interface
                    class ONNXModelWrapper:
                        def __init__(self, session, input_details, output_details):
                            self.session = session
                            self.input_name = input_details[0].name
                            self.output_names = [output.name for output in output_details]
                        
                        def predict(self, input_tensor):
                            # Converter tensor TF para numpy se necessário
                            if isinstance(input_tensor, tf.Tensor):
                                input_np = input_tensor.numpy()
                            else:
                                input_np = input_tensor
                            
                            # Executar inferência
                            outputs = self.session.run(
                                self.output_names, 
                                {self.input_name: input_np}
                            )
                            
                            return outputs
                    
                    # Criar e retornar o wrapper
                    model = ONNXModelWrapper(onnx_session, input_details, output_details)
                    print("DEBUG: Modelo ONNX carregado com sucesso")
                    return model
                    
                except ImportError:
                    print("DEBUG: ONNX Runtime não encontrado, tentando tensorflow-onnx")
                    try:
                        # Tentar usar tf2onnx
                        import tf2onnx
                        import tensorflow as tf
                        
                        # Converter ONNX para TensorFlow e carregar
                        print("DEBUG: Convertendo ONNX para TensorFlow com tf2onnx")
                        model_proto, _ = tf2onnx.convert.from_path(self.model_path)
                        
                        # Salvar em um diretório temporário
                        import tempfile
                        temp_dir = tempfile.mkdtemp()
                        saved_model_dir = os.path.join(temp_dir, 'saved_model')
                        tf2onnx.save.save_model(model_proto, saved_model_dir)
                        
                        # Carregar o modelo salvo
                        model = tf.saved_model.load(saved_model_dir)
                        print("DEBUG: Modelo ONNX convertido e carregado com tf2onnx")
                        return model
                        
                    except ImportError:
                        print("DEBUG: tf2onnx não encontrado, tentando abordagem padrão")
                        print("DEBUG: !! ATENÇÃO: É recomendado instalar onnxruntime-gpu ou tf2onnx")
                        print("DEBUG: Recorrendo ao carregamento padrão do modelo")
                
                except Exception as e:
                    print(f"DEBUG: Erro ao carregar modelo ONNX: {str(e)}")
                    print("DEBUG: Recorrendo ao carregamento normal do modelo")
            
            # Criar uma instância do modelo para detecção apenas
            print("DEBUG: Criando instância do YOEOModel")
            yoeo_model = YOEOModel(
                input_shape=(self.input_height, self.input_width, 3),
                num_classes=len(DetectionType),
                detection_only=True
            )
            print("DEBUG: YOEOModel criado com sucesso")
            
            # Carregar pesos do modelo
            print(f"DEBUG: Carregando pesos do modelo de: {self.model_path}")
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