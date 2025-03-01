#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from tensorflow.keras.layers import Concatenate, UpSampling2D, Reshape, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class YOEOModel:
    """
    Modelo YOEO (You Only Encode Once) para detecção de objetos e segmentação semântica.
    
    Esta classe define a arquitetura do modelo YOEO, que combina detecção de objetos
    e segmentação semântica em uma única arquitetura neural.
    """
    
    def __init__(self, input_shape=(416, 416, 3), num_classes=4, 
                 seg_classes=3, weight_decay=5e-4):
        """
        Inicializa o modelo YOEO.
        
        Args:
            input_shape: Forma da entrada (altura, largura, canais)
            num_classes: Número de classes para detecção de objetos 
                        (bola, gol, robô, árbitro)
            seg_classes: Número de classes para segmentação 
                        (fundo, linhas, campo)
            weight_decay: Regularização L2 para os pesos da rede
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.seg_classes = seg_classes
        self.weight_decay = weight_decay
        
        # Anchors para as diferentes escalas (podem ser ajustados)
        self.anchors = np.array([
            [[10, 13], [16, 30], [33, 23]],     # Anchors pequenos (detectar objetos pequenos)
            [[30, 61], [62, 45], [59, 119]],    # Anchors médios
            [[116, 90], [156, 198], [373, 326]]  # Anchors grandes
        ])
        
        # Construir o modelo
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Constrói a arquitetura do modelo YOEO.
        
        Returns:
            Modelo Keras
        """
        # Camada de entrada
        inputs = Input(shape=self.input_shape)
        
        # Backbone (extrator de características)
        # Bloco 1
        x = self._make_conv_block(inputs, 32, 3)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        
        # Bloco 2
        x = self._make_conv_block(x, 64, 3)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        
        # Bloco 3
        x = self._make_conv_block(x, 128, 3)
        x = self._make_conv_block(x, 64, 1)
        x = self._make_conv_block(x, 128, 3)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        
        # Bloco 4
        x = self._make_conv_block(x, 256, 3)
        x = self._make_conv_block(x, 128, 1)
        x = self._make_conv_block(x, 256, 3)
        x4 = x  # Feature map para detecção de objetos pequenos
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        
        # Bloco 5
        x = self._make_conv_block(x, 512, 3)
        x = self._make_conv_block(x, 256, 1)
        x = self._make_conv_block(x, 512, 3)
        x = self._make_conv_block(x, 256, 1)
        x = self._make_conv_block(x, 512, 3)
        x5 = x  # Feature map para detecção de objetos médios
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        
        # Bloco 6
        x = self._make_conv_block(x, 1024, 3)
        x = self._make_conv_block(x, 512, 1)
        x = self._make_conv_block(x, 1024, 3)
        x = self._make_conv_block(x, 512, 1)
        x = self._make_conv_block(x, 1024, 3)
        x6 = x  # Feature map para detecção de objetos grandes
        
        # Cabeças de detecção para diferentes escalas
        # Detecção em escala grande (objetos pequenos)
        large_box, large_obj, large_cls = self._make_detection_head(x6, 512, 
                                                                    self.num_classes, 
                                                                    self.anchors[2], 
                                                                    'large')
        
        # Upsample e concatenar para médio
        x = self._make_conv_block(x6, 256, 1)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate()([x, x5])
        x = self._make_conv_block(x, 256, 1)
        x = self._make_conv_block(x, 512, 3)
        x = self._make_conv_block(x, 256, 1)
        x = self._make_conv_block(x, 512, 3)
        medium_feature = self._make_conv_block(x, 256, 1)
        
        # Detecção em escala média (objetos médios)
        medium_box, medium_obj, medium_cls = self._make_detection_head(medium_feature, 
                                                                      256, 
                                                                      self.num_classes, 
                                                                      self.anchors[1], 
                                                                      'medium')
        
        # Upsample e concatenar para pequeno
        x = self._make_conv_block(medium_feature, 128, 1)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate()([x, x4])
        x = self._make_conv_block(x, 128, 1)
        x = self._make_conv_block(x, 256, 3)
        x = self._make_conv_block(x, 128, 1)
        x = self._make_conv_block(x, 256, 3)
        small_feature = self._make_conv_block(x, 128, 1)
        
        # Detecção em escala pequena (objetos grandes)
        small_box, small_obj, small_cls = self._make_detection_head(small_feature, 
                                                                   128, 
                                                                   self.num_classes, 
                                                                   self.anchors[0], 
                                                                   'small')
        
        # Segmentação semântica
        # Usar características da detecção pequena (alta resolução)
        seg = self._make_segmentation_head(small_feature, self.seg_classes)
        
        # Modelo final
        return Model(inputs=inputs, outputs=[
            # Outputs de detecção
            [large_box, large_obj, large_cls],
            [medium_box, medium_obj, medium_cls],
            [small_box, small_obj, small_cls],
            # Output de segmentação
            seg
        ])
    
    def _make_conv_block(self, x, filters, kernel_size, strides=1):
        """
        Cria um bloco convolucional com normalização em lote e ativação LeakyReLU.
        
        Args:
            x: Tensor de entrada
            filters: Número de filtros
            kernel_size: Tamanho do kernel
            strides: Stride da convolução
            
        Returns:
            Tensor de saída
        """
        x = Conv2D(filters, kernel_size, strides=strides, padding='same',
                   use_bias=False, kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x
    
    def _make_detection_head(self, x, filters, num_classes, anchors, name):
        """
        Cria uma cabeça de detecção para uma escala específica.
        
        Args:
            x: Tensor de entrada
            filters: Número de filtros
            num_classes: Número de classes
            anchors: Anchors para esta escala
            name: Nome para identificar a escala
            
        Returns:
            Tupla (box_output, objectness_output, class_output)
        """
        num_anchors = anchors.shape[0]
        
        # Camadas intermediárias
        x = self._make_conv_block(x, filters, 3)
        x = self._make_conv_block(x, filters*2, 3)
        x = self._make_conv_block(x, filters, 1)
        x = self._make_conv_block(x, filters*2, 3)
        
        # Camada de saída (caixa, objetividade, classes)
        output = Conv2D(num_anchors * (5 + num_classes), 1, padding='same',
                        kernel_regularizer=l2(self.weight_decay),
                        name=f'yolo_{name}')(x)
        
        # Reorganizar a saída para facilitar o processamento
        grid_shape = tf.shape(output)[1:3]
        output = Lambda(lambda x: tf.reshape(x, [-1, grid_shape[0], grid_shape[1], 
                                             num_anchors, 5 + num_classes]))(output)
        
        # Separar os componentes
        box_xy = Lambda(lambda x: tf.sigmoid(x[..., 0:2]), name=f'box_xy_{name}')(output)
        box_wh = Lambda(lambda x: tf.exp(x[..., 2:4]), name=f'box_wh_{name}')(output)
        objectness = Lambda(lambda x: tf.sigmoid(x[..., 4:5]), name=f'objectness_{name}')(output)
        class_probs = Lambda(lambda x: tf.sigmoid(x[..., 5:]), name=f'class_probs_{name}')(output)
        
        return box_xy, objectness, class_probs
    
    def _make_segmentation_head(self, x, num_classes):
        """
        Cria uma cabeça de segmentação.
        
        Args:
            x: Tensor de entrada
            num_classes: Número de classes de segmentação
            
        Returns:
            Tensor de saída
        """
        # Upsample para corresponder à resolução original
        x = self._make_conv_block(x, 64, 1)
        x = UpSampling2D(size=(2, 2))(x)
        x = self._make_conv_block(x, 64, 3)
        
        x = UpSampling2D(size=(2, 2))(x)
        x = self._make_conv_block(x, 32, 3)
        
        # Saída de segmentação com ativação softmax
        x = Conv2D(num_classes, 1, padding='same', 
                   kernel_regularizer=l2(self.weight_decay))(x)
        return Lambda(lambda x: tf.nn.softmax(x, axis=-1), name='segmentation')(x)
    
    def get_model(self):
        """
        Retorna o modelo Keras construído.
        
        Returns:
            Modelo Keras
        """
        return self.model
    
    def load_weights(self, weight_path):
        """
        Carrega pesos do modelo a partir de um arquivo.
        
        Args:
            weight_path: Caminho para o arquivo de pesos
        """
        if os.path.exists(weight_path):
            try:
                self.model.load_weights(weight_path)
                print(f"Pesos carregados de {weight_path}")
            except Exception as e:
                print(f"Erro ao carregar pesos: {e}")
                print("Usando pesos aleatórios inicializados")
        else:
            print(f"Arquivo de pesos {weight_path} não encontrado")
            print("Usando pesos aleatórios inicializados")
    
    def save_weights(self, weight_path):
        """
        Salva os pesos do modelo em um arquivo.
        
        Args:
            weight_path: Caminho para salvar os pesos
        """
        try:
            self.model.save_weights(weight_path)
            print(f"Pesos salvos em {weight_path}")
        except Exception as e:
            print(f"Erro ao salvar pesos: {e}")
    
    def predict(self, x):
        """
        Realiza a predição em uma imagem ou lote de imagens.
        
        Args:
            x: Imagem ou lote de imagens pré-processadas
            
        Returns:
            Previsões do modelo
        """
        return self.model.predict(x)
    
    def decode(self, predictions, image_shape):
        """
        Decodifica as previsões do modelo para o formato adequado.
        
        Args:
            predictions: Saída do modelo
            image_shape: Forma da imagem original
            
        Returns:
            Dicionário com detecções e segmentação processadas
        """
        # Esta é uma implementação simplificada
        # Em um sistema real, precisaria implementar a decodificação completa
        # dos anchors, non-max suppression, etc.
        
        # Extrair componentes
        detection_outputs = predictions[:3]  # Três saídas de detecção
        segmentation = predictions[3]  # Saída de segmentação
        
        # Implementação simplificada de decodificação
        # Em um sistema real, seria mais complexo
        
        # Retornar resultados processados
        return {
            'detections': detection_outputs,
            'segmentation': segmentation
        }
    
    def compile(self, learning_rate=0.001):
        """
        Compila o modelo para treinamento.
        
        Args:
            learning_rate: Taxa de aprendizado
        """
        # Otimizador
        optimizer = Adam(learning_rate=learning_rate)
        
        # Compilar modelo
        # Nota: Funções de perda reais seriam mais complexas para YOEO
        # Esta é apenas uma função placeholder
        self.model.compile(optimizer=optimizer, loss='mse')
    
    def summary(self):
        """
        Exibe um resumo da arquitetura do modelo.
        """
        return self.model.summary() 