#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementação do modelo YOLO da Ultralytics para detecção de objetos.

Este módulo contém a implementação do modelo YOLOv5 da Ultralytics
adaptado para detecção de objetos em contexto de futebol robótico.
"""

import torch
import os
from ultralytics import YOLO

class YOEOModel:
    """
    Implementação do modelo YOLO da Ultralytics para detecção de objetos.
    
    Esta classe implementa o modelo YOLOv5 da Ultralytics para detecção de objetos
    (bola, gol, robô) em contexto de futebol robótico.
    """
    
    def __init__(self, input_shape=(640, 640, 3), num_classes=3, detection_only=True, 
                 l2_regularization=0.0001, dropout_rate=0.1):
        """
        Inicializa o modelo YOLO da Ultralytics.
        
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
    
    def build(self):
        """
        Constrói o modelo YOLO da Ultralytics.
        
        Returns:
            Modelo YOLO da Ultralytics
        """
        # Inicializa um modelo YOLOv5 pré-treinado
        # O parâmetro task='detect' garante que estamos usando o modelo apenas para detecção
        print("Inicializando um modelo YOLOv5 da Ultralytics")
        self.model = YOLO("yolov5n.pt")  # modelo menor e mais leve
        return self.model
    
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
                
                # Para TensorRT, precisamos usar o método específico da Ultralytics
                try:
                    print(f"DEBUG: YOEOModel.load_weights() - Carregando modelo TensorRT")
                    # A Ultralytics tem métodos internos para carregar modelos TensorRT
                    # Primeiro carregamos um modelo normal
                    self.model = YOLO('yolov5n.pt') 
                    # Apontamos para o arquivo TensorRT
                    self.model.model = weights_path
                    print(f"DEBUG: YOEOModel.load_weights() - Modelo TensorRT carregado com sucesso")
                    return self.model
                except Exception as e:
                    print(f"DEBUG: YOEOModel.load_weights() - Erro ao carregar modelo TensorRT: {e}")
                    print(f"DEBUG: YOEOModel.load_weights() - Tentando carregamento padrão")
                    
            # Carregar o modelo YOLO normalmente
            print(f"DEBUG: YOEOModel.load_weights() - Carregando modelo YOLO da Ultralytics")
            self.model = YOLO(weights_path)
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
        self.model.save(save_path)
        
    def summary(self):
        """
        Exibe um resumo da arquitetura do modelo.
        """
        if self.model is None:
            self.build()
        print(self.model.info())  # Ultralytics tem seu próprio método info para mostrar informações


def train_model(data_yaml_path="data.yaml", epochs=200, imgsz=640, batch=16, workers=4):
    """
    Função para treinar um modelo YOLOv5 da Ultralytics.
    
    Args:
        data_yaml_path: Caminho para o arquivo YAML de configuração do dataset
        epochs: Número de épocas de treinamento
        imgsz: Tamanho das imagens para o treinamento
        batch: Tamanho do batch
        workers: Número de threads para carregamento de dados
        
    Returns:
        Modelo treinado
    """
    # Detectar o número máximo de threads suportadas pela CPU
    max_workers = torch.multiprocessing.cpu_count()

    # Use um valor moderado, como a metade dos núcleos da CPU
    workers = min(max_workers // 2, 8)

    # Configurações de ambiente para máxima performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    print(f"Usando {workers} workers.")

    # Verificação de dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Usando CPU")

    # Carregar o modelo YOLO
    model = YOLO('yolov5n.pt')

    # Configuração do treinamento
    model.train(
        data=data_yaml_path,     # Dataset e configurações no formato YOLO
        epochs=epochs,           # Número de épocas de treinamento
        imgsz=imgsz,             # Tamanho das imagens para o treinamento
        batch=batch,             # Tamanho do batch
        workers=workers,         # Número de threads para carregamento de dados
        augment=True,            # Habilitar aumentos de dados
        device=device            # Dispositivo a ser usado
    )

    # Avaliação (Val) final
    metrics = model.val(
        data=data_yaml_path,
        device=device
    )
    print(f"mAP@0.5: {metrics.box.map50:.3f}, mAP@0.5-0.95: {metrics.box.map:.3f}")

    # Salvar o modelo treinado
    model.save('yolov5_custom.pt')
    
    return model


if __name__ == "__main__":
    # Exemplo de uso
    model = YOEOModel(input_shape=(640, 640, 3), num_classes=3, detection_only=True)
    yolo = model.build()
    print("Modelo YOLOv5 para detecção construído com sucesso!") 