#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementação do modelo YOEO (You Only Encode Once) para detecção de objetos
e segmentação semântica em tempo real para a RoboIME.

Este módulo define a arquitetura neural do modelo YOEO, que combina detecção
de objetos e segmentação semântica em uma única rede neural eficiente.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, Add, 
    MaxPooling2D, UpSampling2D, Concatenate, Lambda
)
from tensorflow.keras.regularizers import l2

class YOEOModel:
    """
    Implementação do modelo YOEO (You Only Encode Once) que combina detecção
    de objetos e segmentação semântica em uma única rede neural.
    
    Esta classe implementa uma arquitetura baseada em YOLO com um backbone
    compartilhado e duas cabeças: uma para detecção de objetos e outra para
    segmentação semântica.
    """
    
    def __init__(self, 
                 input_shape=(416, 416, 3),
                 num_classes=3,
                 num_seg_classes=2,
                 anchors=None):
        """
        Inicializa o modelo YOEO.
        
        Args:
            input_shape (tuple): Formato da imagem de entrada (altura, largura, canais)
            num_classes (int): Número de classes para detecção de objetos
            num_seg_classes (int): Número de classes para segmentação semântica
            anchors (list): Lista de anchors para as diferentes escalas
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        
        # Anchors padrão se não forem fornecidos
        if anchors is None:
            self.anchors = {
                'large': [[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]],
                'medium': [[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]],
                'small': [[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]]
            }
        else:
            self.anchors = anchors
            
        self.model = None
    
    def _conv_block(self, x, filters, kernel_size, strides=1, use_bias=False):
        """
        Bloco convolucional básico: Conv2D + BatchNorm + LeakyReLU
        
        Args:
            x: Tensor de entrada
            filters: Número de filtros
            kernel_size: Tamanho do kernel
            strides: Stride da convolução
            use_bias: Se deve usar bias na camada convolucional
            
        Returns:
            Tensor de saída após aplicar o bloco convolucional
        """
        x = Conv2D(filters, kernel_size, strides=strides, padding='same',
                   use_bias=use_bias, kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x
    
    def _residual_block(self, x, filters):
        """
        Bloco residual com conexão de atalho
        
        Args:
            x: Tensor de entrada
            filters: Número de filtros
            
        Returns:
            Tensor de saída após aplicar o bloco residual
        """
        shortcut = x
        x = self._conv_block(x, filters // 2, 1)
        x = self._conv_block(x, filters, 3)
        x = Add()([shortcut, x])
        return x
    
    def _detection_head(self, inputs, num_anchors, num_classes):
        """
        Cabeça de detecção para uma escala específica
        
        Args:
            inputs: Tensor de entrada
            num_anchors: Número de anchors para esta escala
            num_classes: Número de classes para detecção
            
        Returns:
            Tensor de saída para detecção de objetos
        """
        x = self._conv_block(inputs, 256, 3)
        x = Conv2D(num_anchors * (5 + num_classes), 1, padding='same')(x)

        # Envolvemos a lógica de shape e reshape dentro de um Lambda
        def reshape_yolo(t):
            shape = tf.shape(t)
            b, h, w, c = shape[0], shape[1], shape[2], shape[3]
            # c deve ser == num_anchors*(5 + num_classes)
            return tf.reshape(t, [b, h, w, num_anchors, 5 + num_classes])

        x = Lambda(reshape_yolo)(x)
        return x
    
    def _segmentation_head(self, features, num_seg_classes):
        """
        Cabeça de segmentação que processa as features do backbone.
        
        Args:
            features: Lista de features do backbone em diferentes escalas
            num_seg_classes: Número de classes de segmentação
            
        Returns:
            Tensor de saída para segmentação
        """
        # Imprimir as formas das features para debug
        print("Formas das features:")
        for i, feat in enumerate(features):
            print(f"Feature {i}: {feat.shape}")
        
        # Começa no final (menor resolução)
        x = features[-1]  # (None, 13, 13, 1024)
        
        # Upsampling para 26x26
        x = self._conv_block(x, 512, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, features[-2]])  # Concatenar com feature de 26x26
        x = self._conv_block(x, 256, 3)
        
        # Upsampling para 52x52
        x = self._conv_block(x, 128, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, features[-3]])  # Concatenar com feature de 52x52
        x = self._conv_block(x, 128, 3)
        
        # Upsampling para 104x104
        x = self._conv_block(x, 64, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, features[-4]])  # Concatenar com feature de 104x104
        x = self._conv_block(x, 64, 3)
        
        # Upsampling para 208x208
        x = self._conv_block(x, 32, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, features[-5]])  # Concatenar com feature de 208x208
        x = self._conv_block(x, 32, 3)
        
        # Upsampling final para 416x416 (tamanho original)
        x = self._conv_block(x, 16, 1)
        x = UpSampling2D(2)(x)
        x = self._conv_block(x, 16, 3)
        
        # Camada final de segmentação
        x = Conv2D(num_seg_classes, 1, padding='same', activation='softmax')(x)
        
        return x

    def backbone(self, inputs):
        """
        Implementa o backbone da rede que extrai features em diferentes escalas.
        
        Args:
            inputs: Tensor de entrada
            
        Returns:
            Tupla de tensores de features (pequena, média, grande)
        """
        features = []
        
        # Bloco inicial
        x = self._conv_block(inputs, 32, 3)
        
        # Blocos downsampling
        x = self._conv_block(x, 64, 3, strides=2)
        features.append(x)  # 208x208
        
        x = self._conv_block(x, 128, 3, strides=2)
        for _ in range(2):
            x = self._residual_block(x, 128)
        features.append(x)  # 104x104
        
        x = self._conv_block(x, 256, 3, strides=2)
        for _ in range(8):
            x = self._residual_block(x, 256)
        features.append(x)  # 52x52
        
        x = self._conv_block(x, 512, 3, strides=2)
        for _ in range(8):
            x = self._residual_block(x, 512)
        features.append(x)  # 26x26
        
        x = self._conv_block(x, 1024, 3, strides=2)
        for _ in range(4):
            x = self._residual_block(x, 1024)
        features.append(x)  # 13x13
        
        # Processamento para gerar features para detecção em diferentes escalas
        # Feature grande (para objetos pequenos)
        x_large = self._conv_block(features[-1], 512, 1)
        x_large = self._conv_block(x_large, 1024, 3)
        x_large = self._conv_block(x_large, 512, 1)
        
        # Feature média (para objetos médios)
        x_medium = self._conv_block(x_large, 256, 1)
        x_medium = UpSampling2D(2)(x_medium)
        x_medium = Concatenate()([x_medium, features[-2]])
        x_medium = self._conv_block(x_medium, 256, 1)
        x_medium = self._conv_block(x_medium, 512, 3)
        x_medium = self._conv_block(x_medium, 256, 1)
        
        # Feature pequena (para objetos grandes)
        x_small = self._conv_block(x_medium, 128, 1)
        x_small = UpSampling2D(2)(x_small)
        x_small = Concatenate()([x_small, features[-3]])
        x_small = self._conv_block(x_small, 128, 1)
        x_small = self._conv_block(x_small, 256, 3)
        x_small = self._conv_block(x_small, 128, 1)
        
        return x_small, x_medium, x_large

    def detection_head(self, x, num_classes, name):
        """
        Implementa a cabeça de detecção para o modelo.
        
        Args:
            x: Tensor de entrada
            num_classes: Número de classes de detecção
            name: Nome da cabeça de detecção
            
        Returns:
            Tensor de saída para detecção
        """
        num_anchors = 3  # Usando 3 âncoras por escala
        
        # Aplicar convoluções e gerar saída
        x = self._conv_block(x, 256, 3)
        x = Conv2D(num_anchors * (5 + num_classes), 1, padding='same', name=name)(x)
        
        # Reshape para formato adequado [batch, grid_h, grid_w, anchors, 5+num_classes]
        def reshape_detection(t):
            shape = tf.shape(t)
            batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
            return tf.reshape(t, [batch_size, height, width, num_anchors, 5 + num_classes])
        
        x = Lambda(reshape_detection, name=f"{name}_reshape")(x)
        return x

    def segmentation_head(self, x_small, x_medium, x_large, num_seg_classes):
        """
        Implementa a cabeça de segmentação para o modelo.
        
        Args:
            x_small: Feature map pequena
            x_medium: Feature map média
            x_large: Feature map grande
            num_seg_classes: Número de classes de segmentação
            
        Returns:
            Tensor de saída para segmentação
        """
        # Começar com a feature map média
        x = self._conv_block(x_medium, 256, 3)
        
        # Upsample e concatenar com a feature map pequena
        x = self._conv_block(x, 128, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, x_small])
        
        # Continuar upsampling para alcançar a resolução de entrada
        for filters in [128, 64, 32]:
            x = self._conv_block(x, filters, 3)
            x = UpSampling2D(2)(x)
        
        # Camada final para segmentação
        x = Conv2D(num_seg_classes, 1, padding='same', activation='softmax', name='segmentation')(x)
        
        return x

    def build(self):
        """
        Constrói o modelo YOEO.
        
        Returns:
            Modelo compilado do YOEO com cabeças para detecção e segmentação.
        """
        # Camada de entrada
        inputs = Input(shape=self.input_shape, name='input')
        
        # Backbone e feature pyramid
        x_small, x_medium, x_large = self.backbone(inputs)
        
        # Cabeças de detecção para diferentes escalas
        detection_small = self.detection_head(x_small, self.num_classes, "detection_small")
        detection_medium = self.detection_head(x_medium, self.num_classes, "detection_medium")
        detection_large = self.detection_head(x_large, self.num_classes, "detection_large")
        
        # Cabeça de segmentação
        segmentation = self.segmentation_head(x_small, x_medium, x_large, self.num_seg_classes)
        
        # Criar e retornar o modelo usando um dicionário de saídas para facilitar o treinamento
        model = Model(
            inputs=inputs,
            outputs={
                "detection_small": detection_small,
                "detection_medium": detection_medium,
                "detection_large": detection_large,
                "segmentation": segmentation
            },
            name="yoeo"
        )
        
        self.model = model
        
        # Exibir resumo do modelo
        print("\nResumo do modelo YOEO:")
        print(f"- Entrada: {inputs.shape}")
        print(f"- Detecção (pequena): {detection_small.shape}")
        print(f"- Detecção (média): {detection_medium.shape}")
        print(f"- Detecção (grande): {detection_large.shape}")
        print(f"- Segmentação: {segmentation.shape}")
        
        return model
    
    def load_weights(self, weights_path):
        """
        Carrega pesos pré-treinados para o modelo
        
        Args:
            weights_path: Caminho para o arquivo de pesos
            
        Returns:
            True se os pesos foram carregados com sucesso, False caso contrário
        """
        if self.model is None:
            self.build()
            
        try:
            if weights_path.endswith('.h5'):
                self.model.load_weights(weights_path)
            else:
                self.model = tf.keras.models.load_model(weights_path)
            return True
        except Exception as e:
            print(f"Erro ao carregar pesos: {e}")
            return False
    
    def save(self, model_path):
        """
        Salva o modelo em um arquivo
        
        Args:
            model_path: Caminho para salvar o modelo
            
        Returns:
            True se o modelo foi salvo com sucesso, False caso contrário
        """
        if self.model is None:
            print("Modelo não foi construído ainda.")
            return False
            
        try:
            if model_path.endswith('.h5'):
                self.model.save_weights(model_path)
            else:
                self.model.save(model_path)
            return True
        except Exception as e:
            print(f"Erro ao salvar modelo: {e}")
            return False
    
    def summary(self):
        """
        Exibe um resumo da arquitetura do modelo
        """
        if self.model is None:
            self.build()
        
        return self.model.summary()


if __name__ == "__main__":
    # Exemplo de uso
    model = YOEOModel(input_shape=(416, 416, 3), num_classes=3, num_seg_classes=2)
    model.build()
    model.summary()
    
    # Verificar formato das saídas
    dummy_input = np.zeros((1, 416, 416, 3), dtype=np.float32)
    outputs = model.model.predict(dummy_input)
    
    print("\nFormatos das saídas:")
    print(f"Detecção (escala pequena): {outputs['detection_small'].shape}")
    print(f"Detecção (escala média): {outputs['detection_medium'].shape}")
    print(f"Detecção (escala grande): {outputs['detection_large'].shape}")
    print(f"Segmentação: {outputs['segmentation'].shape}") 