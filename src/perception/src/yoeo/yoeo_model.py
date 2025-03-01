#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2

class YOEOModel:
    """
    Implementação do modelo YOEO (You Only Encode Once) para detecção de objetos.
    
    Este modelo é baseado em uma arquitetura de rede neural convolucional que combina
    características de YOLO (You Only Look Once) com um codificador eficiente (MobileNetV2)
    para detecção de múltiplos objetos em tempo real, otimizado para a Jetson Nano.
    """
    
    def __init__(self, input_shape=(416, 416, 3), num_classes=4, backbone='mobilenetv2'):
        """
        Inicializa o modelo YOEO.
        
        Args:
            input_shape: Formato da imagem de entrada (altura, largura, canais)
            num_classes: Número de classes a serem detectadas (bola, gol, robôs, árbitro)
            backbone: Backbone da rede neural ('mobilenetv2' é recomendado para Jetson Nano)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone = backbone
        self.model = None
        self.anchors = np.array([
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # Anchors para escala grande
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],  # Anchors para escala média
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]   # Anchors para escala pequena
        ])
        self.classes = ['bola', 'gol', 'robo', 'arbitro']
        
        # Construir o modelo
        self._build_model()
    
    def _build_model(self):
        """Constrói a arquitetura do modelo YOEO."""
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Backbone (codificador)
        if self.backbone == 'mobilenetv2':
            # Usar MobileNetV2 como backbone (bom para dispositivos com recursos limitados como Jetson Nano)
            backbone = MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_tensor=inputs,
                alpha=0.75  # Fator de largura para reduzir a complexidade do modelo
            )
            
            # Extrair features de diferentes níveis
            # Características de baixo nível (detalhes finos)
            f1 = backbone.get_layer('block_6_expand_relu').output  # 52x52
            # Características de médio nível
            f2 = backbone.get_layer('block_13_expand_relu').output  # 26x26
            # Características de alto nível (semântica)
            f3 = backbone.get_layer('out_relu').output  # 13x13
        else:
            raise ValueError(f"Backbone '{self.backbone}' não suportado")
        
        # Congelar o backbone para treinamento inicial
        backbone.trainable = False
        
        # Neck (FPN - Feature Pyramid Network)
        # Combina características de diferentes níveis para melhorar a detecção
        
        # Caminho ascendente (de baixo para cima)
        x3 = layers.Conv2D(256, 1, padding='same', use_bias=False)(f3)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.LeakyReLU(alpha=0.1)(x3)
        
        x3_upsample = layers.UpSampling2D(2)(x3)
        x2 = layers.Conv2D(256, 1, padding='same', use_bias=False)(f2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.LeakyReLU(alpha=0.1)(x2)
        x2 = layers.Concatenate()([x2, x3_upsample])
        x2 = self._make_conv_block(x2, 256)
        
        x2_upsample = layers.UpSampling2D(2)(x2)
        x1 = layers.Conv2D(128, 1, padding='same', use_bias=False)(f1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.LeakyReLU(alpha=0.1)(x1)
        x1 = layers.Concatenate()([x1, x2_upsample])
        x1 = self._make_conv_block(x1, 128)
        
        # Caminho descendente (de cima para baixo)
        x1_downsample = layers.Conv2D(256, 3, strides=2, padding='same', use_bias=False)(x1)
        x1_downsample = layers.BatchNormalization()(x1_downsample)
        x1_downsample = layers.LeakyReLU(alpha=0.1)(x1_downsample)
        x2 = layers.Concatenate()([x1_downsample, x2])
        x2 = self._make_conv_block(x2, 256)
        
        x2_downsample = layers.Conv2D(512, 3, strides=2, padding='same', use_bias=False)(x2)
        x2_downsample = layers.BatchNormalization()(x2_downsample)
        x2_downsample = layers.LeakyReLU(alpha=0.1)(x2_downsample)
        x3 = layers.Concatenate()([x2_downsample, x3])
        x3 = self._make_conv_block(x3, 512)
        
        # Cabeças de detecção para diferentes escalas
        # Cada cabeça produz previsões para uma escala específica
        
        # Cabeça para objetos pequenos (escala grande)
        y1 = self._make_detection_head(x1, 128, self.num_classes)
        
        # Cabeça para objetos médios
        y2 = self._make_detection_head(x2, 256, self.num_classes)
        
        # Cabeça para objetos grandes (escala pequena)
        y3 = self._make_detection_head(x3, 512, self.num_classes)
        
        # Modelo final
        self.model = models.Model(inputs, [y1, y2, y3])
    
    def _make_conv_block(self, x, filters, num_blocks=1):
        """Cria um bloco de camadas convolucionais."""
        for _ in range(num_blocks):
            x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=0.1)(x)
            
            x = layers.Conv2D(filters * 2, 3, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=0.1)(x)
        return x
    
    def _make_detection_head(self, x, filters, num_classes):
        """Cria uma cabeça de detecção para uma escala específica."""
        x = self._make_conv_block(x, filters)
        
        # Saída final: para cada anchor, prevê [x, y, w, h, objectness, class1, class2, ...]
        num_anchors = 3  # Número de anchors por célula
        output_filters = num_anchors * (5 + num_classes)  # 5 = [x, y, w, h, objectness]
        
        x = layers.Conv2D(output_filters, 1, padding='same')(x)
        return x
    
    def load_weights(self, weights_path):
        """Carrega os pesos do modelo a partir de um arquivo."""
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"Pesos carregados de {weights_path}")
        else:
            print(f"Arquivo de pesos {weights_path} não encontrado")
    
    def save_weights(self, weights_path):
        """Salva os pesos do modelo em um arquivo."""
        self.model.save_weights(weights_path)
        print(f"Pesos salvos em {weights_path}")
    
    def predict(self, image):
        """
        Realiza a predição em uma imagem.
        
        Args:
            image: Imagem numpy no formato (altura, largura, canais)
            
        Returns:
            Lista de detecções no formato [x, y, w, h, confiança, classe]
        """
        # Pré-processamento da imagem
        image_resized = tf.image.resize(image, (self.input_shape[0], self.input_shape[1]))
        image_normalized = image_resized / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        # Realizar a predição
        predictions = self.model.predict(image_batch)
        
        # Pós-processamento (decodificar as previsões)
        detections = self._decode_predictions(predictions, image.shape)
        
        return detections
    
    def _decode_predictions(self, predictions, original_shape):
        """
        Decodifica as previsões do modelo.
        
        Args:
            predictions: Saída do modelo [y1, y2, y3]
            original_shape: Formato original da imagem (altura, largura, canais)
            
        Returns:
            Lista de detecções no formato [x, y, w, h, confiança, classe]
        """
        # Implementação simplificada da decodificação
        # Em uma implementação completa, seria necessário:
        # 1. Converter as previsões em coordenadas de caixa
        # 2. Aplicar sigmoid/exponencial conforme necessário
        # 3. Aplicar non-maximum suppression
        # 4. Converter para coordenadas da imagem original
        
        # Por simplicidade, retornamos uma lista vazia
        # A implementação completa seria mais complexa
        return []
    
    def compile(self, learning_rate=0.001):
        """Compila o modelo para treinamento."""
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        # Em uma implementação completa, seria necessário definir funções de perda personalizadas
        # para detecção de objetos (perda de localização, perda de confiança, perda de classificação)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',  # Placeholder, deve ser substituído por funções de perda adequadas
            metrics=['accuracy']  # Placeholder, métricas mais adequadas seriam mAP, recall, etc.
        )
    
    def summary(self):
        """Exibe um resumo do modelo."""
        return self.model.summary()
    
    def get_model(self):
        """Retorna o modelo Keras."""
        return self.model 