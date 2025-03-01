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
                 num_classes=4,
                 num_seg_classes=3,
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
        
        # Reshape para [batch, grid_h, grid_w, num_anchors, 5 + num_classes]
        shape = tf.shape(x)
        grid_h, grid_w = shape[1], shape[2]
        x = Lambda(lambda x: tf.reshape(x, [-1, grid_h, grid_w, num_anchors, 5 + num_classes]))(x)
        
        return x
    
    def _segmentation_head(self, features, num_classes):
        """
        Cabeça de segmentação semântica
        
        Args:
            features: Lista de features de diferentes níveis do backbone
            num_classes: Número de classes para segmentação
            
        Returns:
            Tensor de saída para segmentação semântica
        """
        # Usar features de diferentes níveis para segmentação
        x = features[-3]  # Feature map de resolução média
        
        # Upsample e concatenar com features de maior resolução
        x = self._conv_block(x, 128, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, features[-4]])
        
        x = self._conv_block(x, 64, 3)
        x = self._conv_block(x, 64, 3)
        
        # Upsample para resolução da imagem original
        x = self._conv_block(x, 32, 3)
        x = UpSampling2D(2)(x)
        x = self._conv_block(x, 32, 3)
        x = UpSampling2D(2)(x)
        
        # Camada final de segmentação
        x = Conv2D(num_classes, 1, padding='same', activation='softmax')(x)
        
        return x
    
    def build(self):
        """
        Constrói o modelo YOEO completo
        
        Returns:
            Modelo Keras compilado
        """
        # Input
        inputs = Input(shape=self.input_shape)
        
        # Backbone - Darknet-like
        # Bloco inicial
        x = self._conv_block(inputs, 32, 3)
        x = self._conv_block(x, 64, 3, strides=2)
        
        # Bloco 1
        x = self._residual_block(x, 64)
        x = self._conv_block(x, 128, 3, strides=2)
        
        # Bloco 2
        for _ in range(2):
            x = self._residual_block(x, 128)
        x = self._conv_block(x, 256, 3, strides=2)
        
        # Bloco 3
        features = [x]  # Armazenar para uso posterior
        for _ in range(8):
            x = self._residual_block(x, 256)
        features.append(x)
        x = self._conv_block(x, 512, 3, strides=2)
        
        # Bloco 4
        for _ in range(8):
            x = self._residual_block(x, 512)
        features.append(x)
        x = self._conv_block(x, 1024, 3, strides=2)
        
        # Bloco 5
        for _ in range(4):
            x = self._residual_block(x, 1024)
        features.append(x)
        
        # Cabeças de detecção para diferentes escalas
        # Escala grande (para objetos pequenos)
        x_large = self._conv_block(features[-1], 512, 1)
        x_large = self._conv_block(x_large, 1024, 3)
        x_large = self._conv_block(x_large, 512, 1)
        
        # Feature map para escala grande
        y_large = self._detection_head(x_large, len(self.anchors['large']), self.num_classes)
        
        # Escala média (para objetos médios)
        x_medium = self._conv_block(x_large, 256, 1)
        x_medium = UpSampling2D(2)(x_medium)
        x_medium = Concatenate()([x_medium, features[-2]])
        x_medium = self._conv_block(x_medium, 256, 1)
        x_medium = self._conv_block(x_medium, 512, 3)
        x_medium = self._conv_block(x_medium, 256, 1)
        
        # Feature map para escala média
        y_medium = self._detection_head(x_medium, len(self.anchors['medium']), self.num_classes)
        
        # Escala pequena (para objetos grandes)
        x_small = self._conv_block(x_medium, 128, 1)
        x_small = UpSampling2D(2)(x_small)
        x_small = Concatenate()([x_small, features[-3]])
        x_small = self._conv_block(x_small, 128, 1)
        x_small = self._conv_block(x_small, 256, 3)
        x_small = self._conv_block(x_small, 128, 1)
        
        # Feature map para escala pequena
        y_small = self._detection_head(x_small, len(self.anchors['small']), self.num_classes)
        
        # Cabeça de segmentação
        y_seg = self._segmentation_head(features, self.num_seg_classes)
        
        # Modelo completo
        self.model = Model(inputs, [y_small, y_medium, y_large, y_seg])
        
        return self.model
    
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
    model = YOEOModel(input_shape=(416, 416, 3), num_classes=4, num_seg_classes=3)
    model.build()
    model.summary()
    
    # Verificar formato das saídas
    dummy_input = np.zeros((1, 416, 416, 3), dtype=np.float32)
    outputs = model.model.predict(dummy_input)
    
    print("\nFormatos das saídas:")
    print(f"Detecção (escala pequena): {outputs[0].shape}")
    print(f"Detecção (escala média): {outputs[1].shape}")
    print(f"Detecção (escala grande): {outputs[2].shape}")
    print(f"Segmentação: {outputs[3].shape}") 