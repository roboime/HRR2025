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
        self.input_width = config.get("input_width", 416)
        self.input_height = config.get("input_height", 416)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.use_tensorrt = config.get("use_tensorrt", False)
        
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
        Carrega o modelo YOLOv4-Tiny a partir do caminho especificado.
        
        Returns:
            Modelo carregado ou None em caso de erro
        """
        try:
            # Verificar se é um arquivo TensorRT (.trt)
            is_tensorrt_model = self.model_path.lower().endswith('.trt')
            
            if is_tensorrt_model:
                print("DEBUG: Arquivo TensorRT (.trt) detectado, carregando diretamente")
                try:
                    # Tentar carregar o modelo TensorRT
                    import tensorflow as tf
                    print(f"DEBUG: Carregando modelo TensorRT de: {self.model_path}")
                    
                    # Verificar se o arquivo existe
                    if not os.path.exists(self.model_path):
                        print(f"DEBUG: Arquivo TensorRT não encontrado: {self.model_path}")
                        raise FileNotFoundError(f"Arquivo TensorRT não encontrado: {self.model_path}")
                    
                    # Carregar como saved_model
                    try:
                        # Primeiro tentar carregar como SavedModel
                        model = tf.saved_model.load(self.model_path)
                        print("DEBUG: Modelo TensorRT carregado com sucesso como SavedModel")
                        return model
                    except Exception as e1:
                        print(f"DEBUG: Erro ao carregar como SavedModel: {e1}")
                        try:
                            # Tentar carregar diretamente como um modelo keras
                            model = tf.keras.models.load_model(self.model_path)
                            print("DEBUG: Modelo TensorRT carregado com sucesso como modelo Keras")
                            return model
                        except Exception as e2:
                            print(f"DEBUG: Erro ao carregar como modelo Keras: {e2}")
                            
                            # Como último recurso, tentar carregar o plano de execução diretamente
                            try:
                                import tensorrt as trt
                                import pycuda.driver as cuda
                                import pycuda.autoinit
                                
                                # Criar runtime TensorRT
                                TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                                runtime = trt.Runtime(TRT_LOGGER)
                                
                                # Carregar o modelo
                                with open(self.model_path, 'rb') as f:
                                    engine_data = f.read()
                                
                                # Deserializar o engine
                                engine = runtime.deserialize_cuda_engine(engine_data)
                                
                                # Criar um wrapper adequado para o TensorFlow
                                class TRTModel:
                                    def __init__(self, engine):
                                        self.engine = engine
                                        self.context = engine.create_execution_context()
                                        
                                    def predict(self, input_tensor):
                                        # Converter para numpy se necessário
                                        if isinstance(input_tensor, tf.Tensor):
                                            input_data = input_tensor.numpy()
                                        else:
                                            input_data = input_tensor
                                            
                                        # Dimensões de entrada/saída
                                        input_shape = (1, self.input_height, self.input_width, 3)
                                        
                                        # Alocar buffers
                                        host_inputs = [cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)]
                                        host_outputs = [cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)]
                                        
                                        # Criar buffers na GPU
                                        cuda_inputs = [cuda.mem_alloc(h_input.nbytes) for h_input in host_inputs]
                                        cuda_outputs = [cuda.mem_alloc(h_output.nbytes) for h_output in host_outputs]
                                        
                                        # Copiar dados para o host
                                        np.copyto(host_inputs[0], input_data.ravel())
                                        
                                        # Transferir para a GPU
                                        cuda.memcpy_htod(cuda_inputs[0], host_inputs[0])
                                        
                                        # Executar a inferência
                                        self.context.execute_v2(cuda_inputs + cuda_outputs)
                                        
                                        # Transferir resultados de volta
                                        for i in range(len(host_outputs)):
                                            cuda.memcpy_dtoh(host_outputs[i], cuda_outputs[i])
                                            
                                        # Remodelar para o formato esperado pelo processador YOLO
                                        output_shapes = [self.engine.get_binding_shape(i+1) for i in range(len(host_outputs))]
                                        outputs = [host_outputs[i].reshape(output_shapes[i]) for i in range(len(host_outputs))]
                                        
                                        return outputs
                                
                                # Criar e retornar o modelo
                                model = TRTModel(engine)
                                model.input_height = self.input_height
                                model.input_width = self.input_width
                                print("DEBUG: Modelo TensorRT carregado diretamente com sucesso")
                                return model
                            except Exception as e3:
                                print(f"DEBUG: Erro ao carregar modelo TensorRT diretamente: {e3}")
                                print("DEBUG: Recorrendo ao carregamento normal do modelo")
                
                except Exception as e:
                    print(f"DEBUG: Erro ao carregar modelo TensorRT: {e}")
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
            
            # Converter para TensorRT se solicitado e não for já um modelo TRT
            if self.use_tensorrt and not is_tensorrt_model:
                try:
                    print("DEBUG: Tentando converter modelo para TensorRT")
                    import tensorflow as tf
                    print(f"DEBUG: Versão do TensorFlow: {tf.__version__}")
                    print("DEBUG: Verificando disponibilidade do TensorRT")
                    tensorrt_available = hasattr(tf.python, 'compiler') and hasattr(tf.python.compiler, 'tensorrt')
                    print(f"DEBUG: TensorRT disponível: {tensorrt_available}")
                    
                    if tensorrt_available:
                        print("DEBUG: Iniciando conversão para TensorRT")
                        
                        # Método 1: Tentar usar o módulo tensorrt_converter do projeto
                        try:
                            print("DEBUG: Tentando usar o conversor customizado")
                            # Importar o conversor personalizado
                            import sys
                            import os
                            
                            # Adicionar o diretório do projeto ao caminho de importação
                            current_dir = os.path.dirname(os.path.abspath(__file__))
                            parent_dir = os.path.dirname(os.path.dirname(current_dir))
                            if parent_dir not in sys.path:
                                sys.path.append(parent_dir)
                            
                            # Tentar importar o conversor customizado
                            try:
                                from perception.yoeo.tensorrt_converter import convert_yolov4_tiny
                                print("DEBUG: Conversor customizado importado com sucesso")
                                
                                # Usar o conversor customizado
                                import tempfile
                                temp_dir = tempfile.mkdtemp()
                                temp_h5 = os.path.join(temp_dir, 'temp_model.h5')
                                
                                # Salvar o modelo primeiro
                                model.save(temp_h5)
                                print(f"DEBUG: Modelo salvo temporariamente em {temp_h5}")
                                
                                # Converter usando o método customizado
                                optimized_model_path = convert_yolov4_tiny(
                                    model_path=temp_h5,
                                    output_dir=temp_dir,
                                    input_shape=(1, self.input_height, self.input_width, 3),
                                    precision='FP16'
                                )
                                
                                # Verificar se a conversão foi bem-sucedida
                                if optimized_model_path:
                                    print(f"DEBUG: Modelo otimizado salvo em {optimized_model_path}")
                                    # Carregar o modelo otimizado
                                    trt_model = tf.saved_model.load(optimized_model_path)
                                    print("DEBUG: Modelo TensorRT carregado com sucesso")
                                    return trt_model
                                else:
                                    print("DEBUG: Falha na conversão customizada, tentando método alternativo")
                                    raise Exception("Falha na conversão customizada")
                                
                            except ImportError as ie:
                                print(f"DEBUG: Erro ao importar conversor customizado: {ie}")
                                print("DEBUG: Tentando método de conversão alternativo")
                                raise
                        
                        except Exception as e:
                            print(f"DEBUG: Erro ao usar conversor customizado: {e}")
                            print("DEBUG: Tentando método de conversão direto com TF-TRT")
                        
                        # Método 2: Usar a API direta do TensorFlow-TensorRT
                        print("DEBUG: Usando API TensorFlow-TensorRT direta")
                        from tensorflow.python.compiler.tensorrt import trt_convert as trt
                        import tempfile
                        
                        # Verificar problemas comuns de TensorRT
                        try:
                            # Testar se o TensorRT está funcionando corretamente
                            trt_version = trt.get_linked_tensorrt_version()
                            print(f"DEBUG: Versão do TensorRT: {trt_version}")
                        except Exception as e:
                            print(f"DEBUG: Erro ao verificar versão do TensorRT: {e}")
                            print("DEBUG: TensorRT pode não estar instalado corretamente")
                            return model  # Retornar modelo original se TensorRT não estiver disponível
                        
                        # Definir precisão (FP32, FP16, ou INT8)
                        precision_mode = 'FP16'  # Usar FP16 para melhor desempenho em Jetson
                        print(f"DEBUG: Usando precisão {precision_mode}")
                        
                        # Primeiro salvar o modelo em um diretório temporário
                        temp_dir = tempfile.mkdtemp()
                        saved_model_dir = os.path.join(temp_dir, 'saved_model')
                        print(f"DEBUG: Salvando modelo em {saved_model_dir}")
                        
                        # Definir assinatura de entrada para o modelo
                        input_shape = (1, self.input_height, self.input_width, 3)
                        
                        try:
                            # Salvar o modelo com sua assinatura
                            concrete_func = None
                            try:
                                # Obter função concreta para inferência
                                concrete_func = model.call.get_concrete_function(
                                    tf.TensorSpec(shape=input_shape, dtype=tf.float32)
                                )
                                print("DEBUG: Função concreta obtida com sucesso")
                            except Exception as e:
                                print(f"DEBUG: Erro ao obter função concreta: {e}")
                                print("DEBUG: Tentando método alternativo para obter assinatura")
                                
                                # Método alternativo
                                input_tensor = tf.random.normal(input_shape)
                                model._set_inputs(input_tensor)
                                print("DEBUG: Entradas do modelo definidas manualmente")
                            
                            # Salvar o modelo
                            if concrete_func:
                                tf.saved_model.save(
                                    model, 
                                    saved_model_dir, 
                                    signatures={'serving_default': concrete_func}
                                )
                            else:
                                tf.saved_model.save(model, saved_model_dir)
                                
                            print("DEBUG: Modelo salvo para conversão TensorRT")
                            
                            # Configurar parâmetros de conversão
                            conversion_params = trt.TrtConversionParams(
                                precision_mode=precision_mode,
                                max_workspace_size_bytes=8*1024*1024*1024,  # 8GB workspace
                                maximum_cached_engines=1
                            )
                            
                            # Criar conversor
                            converter = trt.TrtGraphConverterV2(
                                input_saved_model_dir=saved_model_dir,
                                conversion_params=conversion_params
                            )
                            
                            print("DEBUG: Convertendo modelo para TensorRT")
                            # Converter o modelo (sem calibração para INT8)
                            converter.convert()
                            
                            # Criar função de inferência otimizada
                            def input_fn():
                                # Gerar dados de entrada para a conversão
                                yield [tf.random.normal(input_shape)]
                            
                            # Compilar para gerar os motores TensorRT
                            print("DEBUG: Compilando motores TensorRT")
                            converter.build(input_fn=input_fn)
                            
                            # Salvar o modelo convertido
                            trt_saved_model_dir = os.path.join(temp_dir, 'trt_saved_model')
                            converter.save(trt_saved_model_dir)
                            print(f"DEBUG: Modelo TensorRT salvo em {trt_saved_model_dir}")
                            
                            # Carregar o modelo otimizado
                            print("DEBUG: Carregando modelo TensorRT otimizado")
                            trt_model = tf.saved_model.load(trt_saved_model_dir)
                            print("DEBUG: Conversão para TensorRT concluída com sucesso")
                            
                            # Retornar o modelo otimizado
                            return trt_model
                            
                        except Exception as e:
                            print(f"DEBUG: Erro durante conversão TensorRT: {e}")
                            print("DEBUG: Usando modelo original devido a falha na conversão")
                    else:
                        print("DEBUG: TensorRT não disponível, usando modelo original")
                except Exception as e:
                    print(f"DEBUG: Erro ao converter para TensorRT: {e}")
                    print("DEBUG: Continuando com o modelo original")
                    import traceback
                    traceback_str = traceback.format_exc()
                    print(f"DEBUG: Traceback completo de TensorRT:\n{traceback_str}")
            
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