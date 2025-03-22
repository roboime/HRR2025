#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

def yoeo_loss(y_true, y_pred):
    """
    Função de perda combinada para detecção de objetos e segmentação.
    
    Args:
        y_true: Lista com ground truth [large_true, medium_true, small_true, seg_true]
        y_pred: Lista com previsões [large_pred, medium_pred, small_pred, seg_pred]
        
    Returns:
        Perda total
    """
    # Extrair componentes das previsões
    large_true, medium_true, small_true, seg_true = y_true
    large_pred, medium_pred, small_pred, seg_pred = y_pred
    
    # Perda de detecção para cada escala
    loss_large = detection_loss(large_true, large_pred)
    loss_medium = detection_loss(medium_true, medium_pred)
    loss_small = detection_loss(small_true, small_pred)
    
    # Perda de segmentação
    loss_seg = segmentation_loss(seg_true, seg_pred)
    
    # Combinar perdas (pode ajustar pesos)
    total_loss = loss_large + loss_medium + loss_small + loss_seg
    
    return total_loss


def detection_loss(y_true, y_pred):
    """
    Função de perda para detecção de objetos.
    
    Args:
        y_true: Ground truth para uma escala específica
        y_pred: Previsão para uma escala específica
        
    Returns:
        Perda de detecção
    """
    # Extrair componentes
    # Formato: [batch, grid_h, grid_w, num_anchors, 5 + num_classes]
    # 5 = [x, y, w, h, objectness]
    
    # Separar coordenadas, objectness e classes
    true_xy = y_true[..., 0:2]  # Centro da bounding box
    true_wh = y_true[..., 2:4]  # Largura e altura
    true_obj = y_true[..., 4:5]  # Objectness
    true_class = y_true[..., 5:]  # Classes
    
    pred_xy = y_pred[..., 0:2]  # Centro previsto
    pred_wh = y_pred[..., 2:4]  # Largura e altura previstas
    pred_obj = y_pred[..., 4:5]  # Objectness previsto
    pred_class = y_pred[..., 5:]  # Classes previstas
    
    # Máscara para células que têm objetos (objectness > 0)
    object_mask = tf.cast(true_obj > 0.5, tf.float32)
    noobject_mask = tf.cast(true_obj <= 0.5, tf.float32)
    
    # Perda de localização (MSE)
    # Só considerar células com objetos
    xy_loss = object_mask * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
    wh_loss = object_mask * tf.reduce_sum(tf.square(tf.sqrt(true_wh) - tf.sqrt(pred_wh)), axis=-1)
    
    # Perda de objectness (BCE)
    obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
    obj_loss = object_mask * obj_loss
    
    # Perda para células sem objetos (reduzir falsos positivos)
    noobj_loss = noobject_mask * tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
    
    # Perda de classificação (CCE)
    class_loss = object_mask * tf.keras.losses.categorical_crossentropy(true_class, pred_class)
    
    # Somar todas as perdas
    total_loss = tf.reduce_mean(xy_loss + wh_loss + obj_loss + noobj_loss * 0.5 + class_loss)
    
    return total_loss


def segmentation_loss(y_true, y_pred):
    """
    Função de perda para segmentação semântica.
    
    Args:
        y_true: Ground truth para segmentação
        y_pred: Previsão para segmentação
        
    Returns:
        Perda de segmentação
    """
    # Perda de entropia cruzada categórica para segmentação
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
    
    # Média sobre todos os pixels
    return tf.reduce_mean(loss)


class YOEOLoss(tf.keras.losses.Loss):
    """
    Classe de perda para YOEO que pode ser usada diretamente com model.compile()
    """
    
    def __init__(self, name="yoeo_loss"):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred):
        return yoeo_loss(y_true, y_pred)


def detection_metrics():
    """
    Retorna métricas para avaliar a detecção de objetos.
    
    Returns:
        Lista de métricas Keras
    """
    # Implementar métricas personalizadas para detecção
    # Por exemplo, precisão, recall, mAP
    pass


def segmentation_metrics():
    """
    Retorna métricas para avaliar a segmentação semântica.
    
    Returns:
        Lista de métricas Keras
    """
    # Implementar métricas personalizadas para segmentação
    # Por exemplo, IoU, dice coefficient
    pass 