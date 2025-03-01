#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Funções de perda para o modelo YOEO.

Este módulo fornece funções de perda personalizadas para treinamento
do modelo YOEO, incluindo perdas para detecção de objetos e segmentação.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def detection_loss(y_true, y_pred, num_classes=4, ignore_thresh=0.5):
    """
    Calcula a perda para a parte de detecção do modelo YOEO.
    
    Esta função implementa uma perda similar à do YOLOv3, com componentes para:
    - Erro de localização (coordenadas x, y, largura, altura)
    - Erro de confiança (objectness)
    - Erro de classificação
    
    Args:
        y_true: Tensor com os valores verdadeiros [batch, num_boxes, 5 + num_classes]
                Formato: [x, y, w, h, obj, class_1, class_2, ...]
        y_pred: Tensor com as previsões [batch, num_boxes, 5 + num_classes]
                Formato: [x, y, w, h, obj, class_1, class_2, ...]
        num_classes: Número de classes de detecção
        ignore_thresh: Limiar para ignorar previsões com baixa confiança
        
    Returns:
        Valor escalar da perda
    """
    # Extrair componentes
    # Coordenadas verdadeiras
    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]
    
    # Objectness verdadeiro
    true_obj = y_true[..., 4:5]
    
    # Classes verdadeiras (one-hot)
    true_class = y_true[..., 5:5+num_classes]
    
    # Coordenadas previstas
    pred_xy = y_pred[..., 0:2]
    pred_wh = y_pred[..., 2:4]
    
    # Objectness previsto
    pred_obj = y_pred[..., 4:5]
    
    # Classes previstas
    pred_class = y_pred[..., 5:5+num_classes]
    
    # Máscara para objetos reais (onde objectness > 0)
    object_mask = K.cast(true_obj > 0.5, K.dtype(true_obj))
    
    # Número de objetos (para normalização)
    num_objects = K.sum(object_mask)
    
    # Evitar divisão por zero
    num_objects = K.maximum(num_objects, 1)
    
    # Perda de coordenadas (x, y)
    xy_loss = object_mask * K.square(true_xy - pred_xy)
    xy_loss = K.sum(xy_loss) / num_objects
    
    # Perda de dimensões (largura, altura)
    # Usando erro quadrático na raiz quadrada para dar menos peso a erros em objetos grandes
    wh_loss = object_mask * K.square(K.sqrt(true_wh) - K.sqrt(K.maximum(pred_wh, 1e-10)))
    wh_loss = K.sum(wh_loss) / num_objects
    
    # Perda de objectness (confiança)
    obj_loss = K.binary_crossentropy(true_obj, pred_obj)
    obj_loss = K.sum(obj_loss) / num_objects
    
    # Perda de classificação (apenas para objetos reais)
    class_loss = object_mask * K.binary_crossentropy(true_class, pred_class)
    class_loss = K.sum(class_loss) / num_objects
    
    # Perda total
    total_loss = xy_loss + wh_loss + obj_loss + class_loss
    
    return total_loss


def segmentation_loss(y_true, y_pred):
    """
    Calcula a perda para a parte de segmentação do modelo YOEO.
    
    Utiliza entropia cruzada categórica para a segmentação semântica.
    
    Args:
        y_true: Tensor com os valores verdadeiros [batch, height, width, num_classes]
                Formato: one-hot encoding das classes de segmentação
        y_pred: Tensor com as previsões [batch, height, width, num_classes]
                Formato: probabilidades para cada classe de segmentação
        
    Returns:
        Valor escalar da perda
    """
    # Aplicar entropia cruzada categórica
    loss = K.categorical_crossentropy(y_true, y_pred)
    
    # Calcular média
    return K.mean(loss)


class YOEOLoss(tf.keras.losses.Loss):
    """
    Classe de perda personalizada para o modelo YOEO.
    
    Combina as perdas de detecção e segmentação com pesos configuráveis.
    """
    
    def __init__(self, num_classes=4, num_seg_classes=3, 
                 detection_weight=1.0, segmentation_weight=1.0, 
                 name="yoeo_loss", **kwargs):
        """
        Inicializa a função de perda YOEO.
        
        Args:
            num_classes: Número de classes de detecção
            num_seg_classes: Número de classes de segmentação
            detection_weight: Peso para a perda de detecção
            segmentation_weight: Peso para a perda de segmentação
            name: Nome da função de perda
        """
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.detection_weight = detection_weight
        self.segmentation_weight = segmentation_weight
    
    def call(self, y_true, y_pred):
        """
        Calcula a perda combinada.
        
        Args:
            y_true: Lista de tensores com valores verdadeiros [detecção, segmentação]
            y_pred: Lista de tensores com previsões [detecção, segmentação]
            
        Returns:
            Valor escalar da perda combinada
        """
        # Extrair componentes
        det_true, seg_true = y_true
        det_pred, seg_pred = y_pred
        
        # Calcular perdas individuais
        det_loss = detection_loss(det_true, det_pred, num_classes=self.num_classes)
        seg_loss = segmentation_loss(seg_true, seg_pred)
        
        # Combinar perdas com pesos
        total_loss = self.detection_weight * det_loss + self.segmentation_weight * seg_loss
        
        return total_loss
    
    def get_config(self):
        """Retorna a configuração da função de perda."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "num_seg_classes": self.num_seg_classes,
            "detection_weight": self.detection_weight,
            "segmentation_weight": self.segmentation_weight
        })
        return config


def detection_metrics():
    """
    Retorna métricas para avaliar a parte de detecção do modelo.
    
    Returns:
        Dicionário com métricas de detecção
    """
    # Implementar métricas personalizadas para detecção
    # Por exemplo: mAP (mean Average Precision)
    return {}


def segmentation_metrics():
    """
    Retorna métricas para avaliar a parte de segmentação do modelo.
    
    Returns:
        Dicionário com métricas de segmentação
    """
    # Implementar métricas personalizadas para segmentação
    # Por exemplo: IoU (Intersection over Union), Dice coefficient
    return {
        'iou': iou_metric,
        'dice': dice_coefficient
    }


def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Calcula o IoU (Intersection over Union) para segmentação.
    
    Args:
        y_true: Tensor com os valores verdadeiros
        y_pred: Tensor com as previsões
        smooth: Valor pequeno para evitar divisão por zero
        
    Returns:
        Valor escalar do IoU
    """
    # Converter previsões para máscaras binárias
    y_pred_mask = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
    y_true_mask = K.cast(K.argmax(y_true, axis=-1), K.floatx())
    
    # Calcular interseção e união
    intersection = K.sum(y_true_mask * y_pred_mask)
    union = K.sum(y_true_mask) + K.sum(y_pred_mask) - intersection
    
    # Calcular IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calcula o coeficiente Dice para segmentação.
    
    Args:
        y_true: Tensor com os valores verdadeiros
        y_pred: Tensor com as previsões
        smooth: Valor pequeno para evitar divisão por zero
        
    Returns:
        Valor escalar do coeficiente Dice
    """
    # Converter previsões para máscaras binárias
    y_pred_mask = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
    y_true_mask = K.cast(K.argmax(y_true, axis=-1), K.floatx())
    
    # Calcular interseção
    intersection = K.sum(y_true_mask * y_pred_mask)
    
    # Calcular coeficiente Dice
    dice = (2. * intersection + smooth) / (K.sum(y_true_mask) + K.sum(y_pred_mask) + smooth)
    
    return dice 