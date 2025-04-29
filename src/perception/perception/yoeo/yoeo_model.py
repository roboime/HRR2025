#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementação do modelo YOLOv4-Tiny para detecção de objetos.

Este módulo contém a implementação do modelo YOLOv4-Tiny
adaptado para detecção de objetos em contexto de futebol robótico.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization,
                                    LeakyReLU, MaxPool2D, ZeroPadding2D,
                                    Concatenate, Lambda, UpSampling2D)
from tensorflow.keras.models import Model


class YOEOModel:
    """
    Implementação do modelo YOLOv4-Tiny para detecção de objetos.
    
    Esta classe implementa o modelo YOLOv4-Tiny para detecção de objetos
    (bola, gol, robô) em contexto de futebol robótico.
    """
    
    def __init__(self, input_shape=(416, 416, 3), num_classes=3, detection_only=True):
        """
        Inicializa o modelo YOLOv4-Tiny.
        
        Args:
            input_shape: Forma de entrada para o modelo (altura, largura, canais)
            num_classes: Número de classes para detecção
            detection_only: Se True, remove completamente a parte de segmentação
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.detection_only = detection_only
        
        # Definir anchors para cada escala
        self.anchors = {
            'large': [[81, 82], [135, 169], [344, 319]],  # 13x13
            'small': [[23, 27], [37, 58], [81, 82]]      # 26x26
        }
        
    def _conv_block(self, x, filters, size, strides=1, batch_norm=True):
        """
        Bloco de convolução com normalização em lote e ativação.
        
        Args:
            x: Tensor de entrada
            filters: Número de filtros para a camada convolucional
            size: Tamanho do kernel
            strides: Tamanho do stride para a convolução
            batch_norm: Se deve aplicar normalização em lote
            
        Returns:
            Tensor resultante
        """
        padding = 'same'
        
        conv = Conv2D(filters=filters, kernel_size=size,
                      strides=strides, padding=padding,
                      use_bias=not batch_norm,
                      kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x)
        
        if batch_norm:
            conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.1)(conv)
            
        return conv
    
    def _detection_head(self, x, num_anchors, num_classes):
        """
        Cabeçalho de detecção YOLOv4-Tiny.
        
        Args:
            x: Tensor de entrada
            num_anchors: Número de anchors
            num_classes: Número de classes
            
        Returns:
            Tensor de saída com predições
        """
        # Número de parâmetros por bbox: 4 (x,y,w,h) + 1 (objectness) + num_classes
        num_filters = num_anchors * (5 + num_classes)
        
        x = self._conv_block(x, filters=num_filters, size=1, batch_norm=False)
        
        # Usar camada Reshape nativa do Keras em vez de Lambda
        # Isso evita problemas com KerasTensor
        input_shape = tf.keras.backend.int_shape(x)
        target_shape = (input_shape[1], input_shape[2], num_anchors, 5 + num_classes)
        
        # Reshape para [batch, grid_h, grid_w, num_anchors, box_params]
        # onde box_params = [x, y, w, h, objectness, ...class_probs]
        return tf.keras.layers.Reshape(target_shape)(x)
    
    def build(self):
        """
        Constrói o modelo YOLOv4-Tiny.
        
        Returns:
            Modelo Keras compilado
        """
        # Entrada
        inputs = Input(self.input_shape)
        
        # Backbone (CSPDarknet53-Tiny)
        x = self._conv_block(inputs, 32, 3)
        x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        
        x = self._conv_block(x, 64, 3)
        x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        
        x = self._conv_block(x, 128, 3)
        x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        
        x = self._conv_block(x, 256, 3)
        route_1 = x  # Salvar para concatenação posterior
        x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        
        x = self._conv_block(x, 512, 3)
        x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        
        x = self._conv_block(x, 1024, 3)
        
        # Detection branches
        # Branch 1 (13x13 grid)
        branch_1 = self._conv_block(x, 256, 1)
        branch_1 = self._conv_block(branch_1, 512, 3)
        branch_1_detection = self._detection_head(branch_1, len(self.anchors['large']), self.num_classes)
        
        # Branch 2 (26x26 grid)
        branch_2 = self._conv_block(branch_1, 128, 1)
        
        # Use uma camada de UpSampling2D em vez de Lambda com tf.image.resize
        # Isso evita problemas de KerasTensor sendo usado em uma função TensorFlow
        # A UpSampling2D é uma camada nativa do Keras que não causa esse problema
        route_1_shape = tf.keras.backend.int_shape(route_1)
        branch_1_shape = tf.keras.backend.int_shape(branch_2)
        
        # Calcular fator de upsampling 
        upsampling_factor = route_1_shape[1] // branch_1_shape[1]
        
        # Substituir Lambda por UpSampling2D
        branch_2 = tf.keras.layers.UpSampling2D(size=(upsampling_factor, upsampling_factor))(branch_2)
        
        branch_2 = Concatenate()([branch_2, route_1])
        branch_2 = self._conv_block(branch_2, 256, 3)
        branch_2_detection = self._detection_head(branch_2, len(self.anchors['small']), self.num_classes)
        
        # Adicionar nomes às camadas de saída
        branch_1_output = tf.keras.layers.Reshape(
            tf.keras.backend.int_shape(branch_1_detection)[1:],
            name="output_1"
        )(branch_1_detection)
        
        branch_2_output = tf.keras.layers.Reshape(
            tf.keras.backend.int_shape(branch_2_detection)[1:],
            name="output_2"
        )(branch_2_detection)
        
        # Se for somente detecção, remover qualquer parte de segmentação
        if self.detection_only:
            model = Model(inputs, [branch_2_output, branch_1_output])
        else:
            # Criar um output dummy de segmentação para manter compatibilidade
            # Usando camadas nativas do Keras em vez de Lambda
            zero_initializer = tf.keras.initializers.Zeros()
            dummy_segmentation = tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=1,
                padding='same',
                kernel_initializer=zero_initializer,
                bias_initializer=zero_initializer,
                trainable=False
            )(inputs)
            
            # Garantir que o tamanho seja correto usando Resize
            dummy_segmentation = tf.keras.layers.experimental.preprocessing.Resizing(
                self.input_shape[0], self.input_shape[1]
            )(dummy_segmentation)
            
            model = Model(inputs, [branch_2_output, branch_1_output, dummy_segmentation])
        
        return model
    
    def load_weights(self, weights_path):
        """
        Carrega os pesos do modelo a partir de um arquivo.
        
        Args:
            weights_path: Caminho para o arquivo de pesos
        """
        print(f"DEBUG: YOEOModel.load_weights() - Carregando pesos de: {weights_path}")
        try:
            import os
            if not os.path.exists(weights_path):
                print(f"DEBUG: YOEOModel.load_weights() - ERRO: Arquivo não existe: {weights_path}")
                raise FileNotFoundError(f"Arquivo de pesos não encontrado: {weights_path}")
                
            print(f"DEBUG: YOEOModel.load_weights() - Construindo modelo")
            model = self.build()
            print(f"DEBUG: YOEOModel.load_weights() - Modelo construído, carregando pesos")
            model.load_weights(weights_path)
            print(f"DEBUG: YOEOModel.load_weights() - Pesos carregados com sucesso")
            return model
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
        model = self.build()
        model.save(save_path)
        
    def summary(self):
        """
        Exibe um resumo da arquitetura do modelo.
        """
        model = self.build()
        model.summary()


if __name__ == "__main__":
    # Exemplo de uso
    model = YOEOModel(input_shape=(416, 416, 3), num_classes=3, detection_only=True)
    yolo = model.build()
    yolo.summary()
    print("Modelo YOLOv4-Tiny para detecção construído com sucesso!") 