#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementação do modelo YOLO para detecção de objetos.

Este módulo contém a implementação do modelo YOLOv5 adaptado
para detecção de objetos em contexto de futebol robótico.
Modificado para usar diretamente a API do YOLOv5 original sem depender do pacote Ultralytics.
"""

import torch
import os
import sys
import logging

# Caminho para a instalação do YOLOv5
# Isso deve ser ajustado para apontar para onde o código YOLOv5 foi clonado
YOLOV5_PATH = os.environ.get('YOLOV5_PATH', '/opt/yolov5')

# Adicionar o diretório do YOLOv5 ao path para podermos importar seus módulos
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)
    print(f"Adicionado {YOLOV5_PATH} ao sys.path")

# Configurar logger
logger = logging.getLogger(__name__)

class YOEOModel:
    """
    Implementação do modelo YOLO para detecção de objetos.
    
    Esta classe implementa o modelo YOLOv5 para detecção de objetos
    (bola, gol, robô) em contexto de futebol robótico.
    """
    
    def __init__(self, input_shape=(640, 640, 3), num_classes=3, detection_only=True, 
                 l2_regularization=0.0001, dropout_rate=0.1):
        """
        Inicializa o modelo YOLO.
        
        Args:
            input_shape: Forma de entrada para o modelo (altura, largura, canais)
            num_classes: Número de classes para detecção
            detection_only: Se True, remove completamente a parte de segmentação
            l2_regularization: Valor para regularização L2 dos pesos
            dropout_rate: Taxa de dropout para reduzir overfitting
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.detection_only = detection_only
        self.l2_reg = l2_regularization
        self.dropout_rate = dropout_rate
        self.model = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
    
    def build(self):
        """
        Constrói o modelo YOLO.
        
        Returns:
            Modelo YOLO
        """
        try:
            # Importar módulos do YOLOv5
            from models.experimental import attempt_load
            
            print("Inicializando um modelo YOLOv5")
            # Carregar um modelo pré-treinado (yolov5s.pt como padrão)
            weights_path = os.path.join(YOLOV5_PATH, 'yolov5s.pt')
            if os.path.exists(weights_path):
                print(f"Carregando modelo padrão de {weights_path}")
                self.model = attempt_load(weights_path, map_location=self.device)
                print("Modelo carregado com sucesso!")
            else:
                print(f"AVISO: Modelo padrão não encontrado em {weights_path}")
                print("Você precisará carregar pesos explicitamente com load_weights()")
                # Criar um modelo vazio para evitar erros
                from models.yolo import Model
                from models.common import AutoShape
                
                # Importar a configuração do modelo
                cfg_path = os.path.join(YOLOV5_PATH, 'models/yolov5s.yaml')
                if not os.path.exists(cfg_path):
                    print(f"ERRO: Arquivo de configuração não encontrado: {cfg_path}")
                    raise FileNotFoundError(f"Arquivo de configuração não encontrado: {cfg_path}")
                    
                self.model = Model(cfg_path, ch=3, nc=self.num_classes)
                self.model = AutoShape(self.model)
                print("Modelo vazio criado. Use load_weights() para carregar pesos.")
            
            return self.model
        except Exception as e:
            print(f"Erro ao construir o modelo: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_weights(self, weights_path):
        """
        Carrega os pesos do modelo a partir de um arquivo.
        
        Args:
            weights_path: Caminho para o arquivo de pesos
        """
        print(f"DEBUG: YOEOModel.load_weights() - Carregando pesos de: {weights_path}")
        try:
            if not os.path.exists(weights_path):
                print(f"DEBUG: YOEOModel.load_weights() - ERRO: Arquivo não existe: {weights_path}")
                raise FileNotFoundError(f"Arquivo de pesos não encontrado: {weights_path}")
            
            # Verificar se é um arquivo TensorRT (.engine ou .trt)
            if weights_path.lower().endswith(('.engine', '.trt')):
                print(f"DEBUG: YOEOModel.load_weights() - Arquivo TensorRT detectado")
                
                try:
                    # Tentar carregar modelo TensorRT
                    # Isso requer a implementação de um módulo TensorRT
                    # Pode ser necessário implementar um TRTModule personalizado
                    print(f"DEBUG: YOEOModel.load_weights() - Carregando modelo TensorRT")
                    
                    try:
                        # Verificar se temos o módulo TensorRT
                        from utils.trt_utils import TRTModule
                        self.model = TRTModule(weights_path)
                        print(f"DEBUG: YOEOModel.load_weights() - Modelo TensorRT carregado com sucesso")
                    except ImportError:
                        print(f"DEBUG: YOEOModel.load_weights() - Módulo TensorRT não encontrado.")
                        print(f"DEBUG: YOEOModel.load_weights() - Tentando carregamento alternativo")
                        
                        # Se não temos módulo TensorRT personalizado, tentamos usar o pycuda
                        import tensorrt as trt
                        import pycuda.driver as cuda
                        import pycuda.autoinit
                        
                        # Implementação básica de carregamento TensorRT
                        logger = trt.Logger(trt.Logger.WARNING)
                        runtime = trt.Runtime(logger)
                        
                        with open(weights_path, 'rb') as f:
                            engine_data = f.read()
                        
                        engine = runtime.deserialize_cuda_engine(engine_data)
                        context = engine.create_execution_context()
                        
                        # Criar um wrapper para o engine/context
                        class TRTWrapper:
                            def __init__(self, engine, context):
                                self.engine = engine
                                self.context = context
                        
                        self.model = TRTWrapper(engine, context)
                        print(f"DEBUG: YOEOModel.load_weights() - Modelo TensorRT carregado manualmente")
                    
                    return self.model
                except Exception as e:
                    print(f"DEBUG: YOEOModel.load_weights() - Erro ao carregar modelo TensorRT: {e}")
                    print(f"DEBUG: YOEOModel.load_weights() - Tentando carregamento padrão")
            
            # Se for arquivo ONNX
            elif weights_path.lower().endswith('.onnx'):
                print(f"DEBUG: YOEOModel.load_weights() - Arquivo ONNX detectado")
                
                try:
                    import onnxruntime as ort
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.model = ort.InferenceSession(weights_path, providers=providers)
                    print(f"DEBUG: YOEOModel.load_weights() - Modelo ONNX carregado com sucesso")
                    return self.model
                except Exception as e:
                    print(f"DEBUG: YOEOModel.load_weights() - Erro ao carregar modelo ONNX: {e}")
            
            # Carregar modelo PyTorch (.pt)
            else:
                # Importar módulos do YOLOv5
                from models.experimental import attempt_load
                
                print(f"DEBUG: YOEOModel.load_weights() - Carregando modelo PyTorch")
                self.model = attempt_load(weights_path, map_location=self.device)
                print(f"DEBUG: YOEOModel.load_weights() - Modelo carregado com sucesso")
                return self.model
                
        except Exception as e:
            print(f"DEBUG: YOEOModel.load_weights() - ERRO ao carregar pesos: {str(e)}")
            import traceback
            traceback_str = traceback.format_exc()
            print(f"DEBUG: YOEOModel.load_weights() - Traceback:\n{traceback_str}")
            raise
    
    def save(self, save_path):
        """
        Salva o modelo em um arquivo.
        
        Args:
            save_path: Caminho para salvar o modelo
        """
        if self.model is None:
            self.build()
        
        # Salvar o modelo
        torch.save(self.model.state_dict(), save_path)
        print(f"Modelo salvo em {save_path}")
        
    def summary(self):
        """
        Exibe um resumo da arquitetura do modelo.
        """
        if self.model is None:
            self.build()
        
        # Contar parâmetros
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Modelo YOLOv5:")
        print(f"Total de parâmetros: {num_params:,}")
        print(f"Parâmetros treináveis: {trainable_params:,}")
        print(f"Tamanho da entrada: {self.input_shape}")
        
        # Tentar exibir estrutura do modelo
        try:
            modules = self.model.modules()
            for i, m in enumerate(modules):
                if i == 0:  # Pular o módulo principal
                    continue
                print(f"{i}: {m.__class__.__name__}")
        except Exception as e:
            print(f"Não foi possível exibir a estrutura do modelo: {e}")


def train_model(data_yaml_path="data.yaml", epochs=200, imgsz=640, batch=16, workers=4):
    """
    Função para treinar um modelo YOLOv5.
    
    Args:
        data_yaml_path: Caminho para o arquivo YAML de configuração do dataset
        epochs: Número de épocas de treinamento
        imgsz: Tamanho das imagens para o treinamento
        batch: Tamanho do batch
        workers: Número de threads para carregamento de dados
        
    Returns:
        Modelo treinado
    """
    try:
        # Verificar se temos o YOLOv5 disponível
        if not os.path.exists(YOLOV5_PATH):
            print(f"ERRO: Diretório YOLOv5 não encontrado: {YOLOV5_PATH}")
            print("Por favor, clone o YOLOv5 e defina a variável YOLOV5_PATH")
            return None
        
        # Mude para o diretório do YOLOv5
        current_dir = os.getcwd()
        os.chdir(YOLOV5_PATH)
        
        # Configurar comando de treinamento
        cmd = [
            "python", "train.py",
            "--img", str(imgsz),
            "--batch", str(batch),
            "--epochs", str(epochs),
            "--data", data_yaml_path,
            "--weights", "yolov5s.pt",  # Modelo base
            "--workers", str(workers)
        ]
        
        # Executar treinamento
        import subprocess
        print(f"Executando: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Mostrar saída em tempo real
        for line in process.stdout:
            print(line, end='')
        
        # Esperar pela conclusão
        process.wait()
        
        # Voltar ao diretório original
        os.chdir(current_dir)
        
        # Carregar o modelo treinado
        best_weights = os.path.join(YOLOV5_PATH, "runs/train/exp/weights/best.pt")
        if os.path.exists(best_weights):
            print(f"Carregando melhor modelo: {best_weights}")
            model = YOEOModel(input_shape=(imgsz, imgsz, 3), num_classes=3)
            model.load_weights(best_weights)
            return model
        else:
            print(f"AVISO: Modelo treinado não encontrado em {best_weights}")
            return None
            
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Exemplo de uso
    model = YOEOModel(input_shape=(640, 640, 3), num_classes=3, detection_only=True)
    yolo = model.build()
    print("Modelo YOLOv5 para detecção construído com sucesso!")
    model.summary() 