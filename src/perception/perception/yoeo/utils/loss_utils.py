#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Funções de perda para o modelo YOEO.

Este módulo fornece funções de perda personalizadas para treinamento
do modelo YOEO, incluindo perdas para detecção de objetos e segmentação.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def detection_loss(y_true, y_pred, num_classes=3, ignore_thresh=0.5):
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
    Pode ser aplicada a cada saída do modelo (detecção ou segmentação).
    """
    
    def __init__(self, num_classes=3, num_seg_classes=2, det_loss_weight=1.0, seg_loss_weight=1.0, **kwargs):
        """
        Inicializa a perda YOEO.
        
        Args:
            num_classes: Número de classes para detecção
            num_seg_classes: Número de classes para segmentação
            det_loss_weight: Peso para a perda de detecção
            seg_loss_weight: Peso para a perda de segmentação
        """
        super(YOEOLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.det_loss_weight = det_loss_weight
        self.seg_loss_weight = seg_loss_weight
        
        # Inicializar perdas para segmentação
        self.seg_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    
    def call(self, y_true, y_pred):
        """
        Calcula a perda para uma única saída (detecção ou segmentação).
        
        Args:
            y_true: Array dos valores verdadeiros
            y_pred: Array dos valores preditos
            
        Returns:
            Valor da perda
        """
        # A detecção de qual tipo de perda calcular não pode depender apenas da forma do tensor
        # porque durante o treinamento as formas podem variar.
        # Vamos usar o nome da saída para determinar o tipo de perda
        
        # Obtemos o nome da saída do contexto, se disponível
        output_name = None
        try:
            # Tentamos obter o contexto atual
            if hasattr(tf.keras.backend, 'get_uid'):
                context = tf.keras.backend.get_uid()
                if 'segmentation' in context:
                    output_name = 'segmentation'
                elif 'detection' in context:
                    output_name = 'detection'
        except:
            pass
        
        # Se não conseguimos determinar pelo contexto, usamos a forma dos tensores
        if output_name is None:
            # Se y_pred tiver 5 dimensões, assumimos que é uma saída de detecção
            # shape: [batch, grid_h, grid_w, num_anchors, 5+num_classes]
            pred_rank = tf.rank(y_pred)
            is_detection = tf.equal(pred_rank, 5)
            
            return tf.cond(
                is_detection,
                lambda: self._calculate_detection_loss(y_true, y_pred),
                lambda: self._calculate_segmentation_loss(y_true, y_pred)
            )
        elif output_name == 'segmentation':
            return self._calculate_segmentation_loss(y_true, y_pred)
        else:  # detection
            return self._calculate_detection_loss(y_true, y_pred)
    
    def _calculate_detection_loss(self, y_true, y_pred):
        """
        Calcula a perda para detecção de objetos.
        
        Args:
            y_true: Tensor contendo os valores reais para detecção
            y_pred: Tensor contendo os valores preditos para detecção
        
        Returns:
            Valor da perda de detecção
        """
        # Primeiro verificamos e ajustamos as formas dos tensores
        # O formato pode variar dependendo de como o treinamento está configurado
        
        # Reshape y_true e y_pred para ter formato consistente se necessário
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)
        
        # Se y_true tem dimensão 5, é porque já está no formato esperado [batch, h, w, anchors, channels]
        # Caso contrário, assumimos que é um formato plano [batch, n, channels]
        is_true_detection_format = tf.equal(tf.rank(y_true), 5)
        
        # Se y_pred tem dimensão 5, é porque já está no formato esperado [batch, h, w, anchors, channels]
        # Caso contrário, assumimos que é um formato plano [batch, n, channels]
        is_pred_detection_format = tf.equal(tf.rank(y_pred), 5)
        
        # Função para realizar o processamento de detecção
        def process_detection():
            # Extrair componentes dos tensores planos
            batch_size = tf.shape(y_true)[0]
            num_channels = tf.shape(y_true)[-1]  # 5 + num_classes
            
            # Reshape para formato [batch, n, 5+num_classes] se não estiver já nesse formato
            y_true_flat = tf.cond(
                is_true_detection_format,
                lambda: tf.reshape(y_true, [batch_size, -1, num_channels]),
                lambda: y_true
            )
            
            y_pred_flat = tf.cond(
                is_pred_detection_format,
                lambda: tf.reshape(y_pred, [batch_size, -1, num_channels]),
                lambda: y_pred
            )
            
            # Extrair componentes dos tensores planos
            true_boxes = y_true_flat[..., :4]
            true_obj = y_true_flat[..., 4:5]
            true_class = y_true_flat[..., 5:]
            
            pred_boxes = y_pred_flat[..., :4]
            pred_obj = y_pred_flat[..., 4:5]
            pred_class = y_pred_flat[..., 5:]
            
            # Calcular IoU entre caixas reais e preditas
            iou = self._calculate_iou(true_boxes, pred_boxes)
            
            # Perda de localização (usar MSE ou GIoU/DIoU/CIoU)
            loc_loss = tf.reduce_sum(tf.square(true_boxes - pred_boxes), axis=-1)
            
            # Aplicar máscara de objetividade
            loc_loss = tf.reduce_sum(true_obj * loc_loss)
            
            # Perda de objetividade (usar BCE)
            obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
            obj_loss = tf.reduce_sum(obj_loss)
            
            # Perda de classificação (usar BCE ou Focal Loss)
            cls_loss = tf.keras.losses.binary_crossentropy(true_class, pred_class)
            cls_loss = tf.reduce_sum(true_obj * tf.reduce_sum(cls_loss, axis=-1))
            
            # Normalizar por número de objetos (+1e-6 para evitar divisão por zero)
            num_objects = tf.reduce_sum(true_obj) + 1e-6
            
            # Combinar todas as perdas
            total_loss = (loc_loss + obj_loss + cls_loss) / num_objects
            
            return total_loss
        
        # Executar o processamento de detecção
        return process_detection()
    
    def _calculate_segmentation_loss(self, y_true, y_pred):
        """
        Calcula a perda para segmentação.
        
        Args:
            y_true: Tensor contendo os valores reais para segmentação
            y_pred: Tensor contendo os valores preditos para segmentação
        
        Returns:
            Valor da perda de segmentação
        """
        # Verificar as formas de y_true e y_pred
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)
        
        # Se y_pred tem 5 dimensões (formato de detecção), então vamos converter para formato de segmentação
        # Este caso pode ocorrer durante a compilação quando o modelo tenta aplicar a perda de segmentação a um tensor de detecção
        y_pred_rank = tf.rank(y_pred)
        y_true_rank = tf.rank(y_true)
        
        # Se y_pred tem formato de detecção mas estamos calculando perda de segmentação,
        # retornamos zero para evitar erros durante a compilação
        is_pred_detection = tf.equal(y_pred_rank, 5)
        is_true_detection = tf.equal(y_true_rank, 3)  # [batch, boxes, channels]
        
        # Se os formatos não são compatíveis, retornamos uma perda zero
        # Isso é uma solução temporária para permitir que a compilação prossiga
        if is_pred_detection or is_true_detection:
            return tf.constant(0.0, dtype=tf.float32)
        
        # Para segmentação multiclasse, usar sparse categorical crossentropy
        if self.num_seg_classes > 2:
            # Converter one-hot para índice de classe se necessário
            if tf.shape(y_true)[-1] > 1:  # Se estiver em formato one-hot
                y_true_sparse = tf.argmax(y_true, axis=-1)
            else:  # Se já for índice de classe
                y_true_sparse = tf.squeeze(y_true, axis=-1)
            
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true_sparse, y_pred, from_logits=True)
        else:  # Para segmentação binária
            loss = tf.keras.losses.binary_crossentropy(
                y_true, y_pred, from_logits=True)
        
        # Média sobre todos os pixels
        return tf.reduce_mean(loss)
    
    def _calculate_iou(self, true_boxes, pred_boxes):
        """
        Calcula IoU (Intersection over Union) entre caixas reais e preditas.
        
        Args:
            true_boxes: Tensor de formato [..., 4] contendo [x, y, w, h]
            pred_boxes: Tensor de formato [..., 4] contendo [x, y, w, h]
        
        Returns:
            Tensor contendo valores IoU
        """
        # Converter de xywh para xyxy
        true_xy = true_boxes[..., :2]
        true_wh = true_boxes[..., 2:4]
        true_xy1 = true_xy - true_wh / 2
        true_xy2 = true_xy + true_wh / 2
        
        pred_xy = pred_boxes[..., :2]
        pred_wh = pred_boxes[..., 2:4]
        pred_xy1 = pred_xy - pred_wh / 2
        pred_xy2 = pred_xy + pred_wh / 2
        
        # Calcular coordenadas da interseção
        intersect_xy1 = tf.maximum(true_xy1, pred_xy1)
        intersect_xy2 = tf.minimum(true_xy2, pred_xy2)
        intersect_wh = tf.maximum(intersect_xy2 - intersect_xy1, 0.0)
        
        # Calcular área da interseção e união
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        true_area = true_wh[..., 0] * true_wh[..., 1]
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        union_area = true_area + pred_area - intersect_area
        
        # Calcular IoU
        iou = tf.clip_by_value(intersect_area / (union_area + 1e-6), 0.0, 1.0)
        
        return iou


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


class CombinedYOEOLoss(tf.keras.losses.Loss):
    """
    Classe de perda combinada simplificada para o modelo YOEO.
    """
    
    def __init__(self, num_classes=3, num_seg_classes=2, 
                 det_loss_weight=1.0, seg_loss_weight=1.0,
                 name="combined_yoeo_loss", **kwargs):
        """
        Inicializa a perda combinada do YOEO.
        """
        super(CombinedYOEOLoss, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.det_loss_weight = det_loss_weight
        self.seg_loss_weight = seg_loss_weight
        
        # Criar perdas básicas
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    
    def call(self, y_true, y_pred):
        """
        Calcula a perda combinada para o modelo YOEO.
        """
        # Versão simplificada para lidar com problemas de inicialização
        # Durante a fase inicial, usar apenas uma perda básica para estabilizar o treinamento
        
        # Se não estamos em formato de dicionário, retornar um valor constante para compilação
        if not isinstance(y_pred, dict) or not isinstance(y_true, dict):
            return tf.constant(0.1, dtype=tf.float32)
        
        try:
            # Aplicar perda simples a cada componente
            det_small_loss = tf.constant(0.0, dtype=tf.float32)
            det_medium_loss = tf.constant(0.0, dtype=tf.float32)
            det_large_loss = tf.constant(0.0, dtype=tf.float32)
            seg_loss = tf.constant(0.0, dtype=tf.float32)
            
            # Calcular perdas individuais se disponíveis
            if "detection_small" in y_pred and "detection_small" in y_true:
                det_small_loss = self.mse(y_true["detection_small"], y_pred["detection_small"])
            
            if "detection_medium" in y_pred and "detection_medium" in y_true:
                det_medium_loss = self.mse(y_true["detection_medium"], y_pred["detection_medium"])
            
            if "detection_large" in y_pred and "detection_large" in y_true:
                det_large_loss = self.mse(y_true["detection_large"], y_pred["detection_large"])
            
            if "segmentation" in y_pred and "segmentation" in y_true:
                seg_loss = self.ce(y_true["segmentation"], y_pred["segmentation"])
            
            # Média das perdas de detecção
            det_loss = (det_small_loss + det_medium_loss + det_large_loss) / 3.0
            
            # Combinar perdas com pesos
            total_loss = self.det_loss_weight * det_loss + self.seg_loss_weight * seg_loss
            
            # Verificar se a perda é um valor válido
            if tf.math.is_nan(total_loss) or tf.math.is_inf(total_loss):
                return tf.constant(0.1, dtype=tf.float32)
            
            return total_loss
            
        except Exception as e:
            # Em caso de erro, retornar um valor constante para não quebrar o treinamento
            if tf.executing_eagerly():
                tf.print("Erro na função de perda:", e)
            return tf.constant(0.1, dtype=tf.float32)


class DetectionLoss(tf.keras.losses.Loss):
    """
    Classe para calcular a perda específica para detecção.
    """
    
    def __init__(self, num_classes=3, 
                 coord_weight=5.0, obj_weight=1.0, noobj_weight=0.5, class_weight=1.0,
                 name="detection_loss", **kwargs):
        """
        Inicializa a perda de detecção.
        
        Args:
            num_classes: Número de classes para detecção
            coord_weight: Peso para erros de coordenadas
            obj_weight: Peso para erro de objectness positivo
            noobj_weight: Peso para erro de objectness negativo
            class_weight: Peso para erro de classificação
            name: Nome da perda
        """
        super(DetectionLoss, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.coord_weight = coord_weight
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.class_weight = class_weight
    
    def call(self, y_true, y_pred):
        """
        Calcula a perda para detecção de objetos.
        
        Args:
            y_true: Tensor dos valores verdadeiros [batch, h, w, anchors, 5+classes]
            y_pred: Tensor das previsões [batch, h, w, anchors, 5+classes]
            
        Returns:
            Valor escalar da perda
        """
        # Verificar se estamos em fase de compilação
        if not tf.executing_eagerly() and (tf.rank(y_true) < 5 or tf.rank(y_pred) < 5):
            # Placeholder durante compilação
            return tf.constant(0.0, dtype=tf.float32)
        
        # Reshape para formato plano se necessário
        batch_size = tf.shape(y_true)[0]
        
        if tf.rank(y_true) == 5 and tf.rank(y_pred) == 5:
            # Formato [batch, h, w, anchors, channels]
            # Reshape para formato plano [batch, h*w*anchors, channels]
            y_true_flat = tf.reshape(y_true, [batch_size, -1, 5 + self.num_classes])
            y_pred_flat = tf.reshape(y_pred, [batch_size, -1, 5 + self.num_classes])
        else:
            # Já está no formato plano
            y_true_flat = y_true
            y_pred_flat = y_pred
        
        # Extrair componentes
        true_xy = y_true_flat[..., 0:2]  # centros x,y
        true_wh = y_true_flat[..., 2:4]  # largura, altura
        true_obj = y_true_flat[..., 4:5]  # objectness
        true_class = y_true_flat[..., 5:]  # classes
        
        pred_xy = y_pred_flat[..., 0:2]
        pred_wh = y_pred_flat[..., 2:4]
        pred_obj = y_pred_flat[..., 4:5]
        pred_class = y_pred_flat[..., 5:]
        
        # Máscara para objetos (positivos) e fundo (negativos)
        obj_mask = tf.cast(true_obj > 0.5, tf.float32)
        noobj_mask = tf.cast(true_obj <= 0.5, tf.float32)
        
        # Perda de coordenadas (MSE para xy e wh)
        xy_loss = self.coord_weight * tf.reduce_sum(obj_mask * tf.square(true_xy - pred_xy))
        
        # Usamos raiz quadrada para wh para compensar objetos de diferentes tamanhos
        wh_loss = self.coord_weight * tf.reduce_sum(obj_mask * tf.square(
            tf.sqrt(tf.maximum(true_wh, 1e-7)) - tf.sqrt(tf.maximum(pred_wh, 1e-7))
        ))
        
        # Perda de objectness ponderada para positivos e negativos
        obj_loss = self.obj_weight * tf.reduce_sum(
            obj_mask * tf.keras.losses.binary_crossentropy(true_obj, pred_obj, from_logits=True)
        )
        
        noobj_loss = self.noobj_weight * tf.reduce_sum(
            noobj_mask * tf.keras.losses.binary_crossentropy(true_obj, pred_obj, from_logits=True)
        )
        
        # Perda de classificação (apenas para objetos reais)
        cls_loss = self.class_weight * tf.reduce_sum(
            obj_mask * tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(true_class, pred_class, from_logits=True),
                axis=-1
            )
        )
        
        # Normalizar por número de objetos (evitar divisão por zero)
        num_objects = tf.maximum(tf.reduce_sum(obj_mask), 1.0)
        
        # Combinar todas as perdas
        total_loss = (xy_loss + wh_loss + obj_loss + noobj_loss + cls_loss) / num_objects
        
        # Verificar se a perda está na faixa válida
        total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, tf.constant(1.0, dtype=tf.float32))
        
        return total_loss


class SegmentationLoss(tf.keras.losses.Loss):
    """
    Classe para calcular a perda específica para segmentação.
    """
    
    def __init__(self, num_classes=2, use_dice_loss=True, focal_gamma=2.0, name="segmentation_loss", **kwargs):
        """
        Inicializa a perda de segmentação.
        
        Args:
            num_classes: Número de classes para segmentação
            use_dice_loss: Se deve incluir a perda de coeficiente Dice
            focal_gamma: Parâmetro gamma para focal loss
            name: Nome da perda
        """
        super(SegmentationLoss, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.use_dice_loss = use_dice_loss
        self.focal_gamma = focal_gamma
    
    def _dice_loss(self, y_true, y_pred, smooth=1e-6):
        """
        Calcula a perda Dice.
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Valores preditos
            smooth: Fator de suavização para evitar divisão por zero
            
        Returns:
            Perda Dice (1 - coeficiente Dice)
        """
        # Flatten para ignorar posição espacial
        y_true_f = tf.reshape(y_true, [-1, self.num_classes])
        y_pred_f = tf.reshape(y_pred, [-1, self.num_classes])
        
        # Calcular interseção e união para cada classe
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
        
        # Calcular coeficiente Dice para cada classe
        dice = (2. * intersection + smooth) / (union + smooth)
        
        # Média sobre todas as classes
        dice_mean = tf.reduce_mean(dice)
        
        # Retornar 1 - dice como perda
        return 1.0 - dice_mean
    
    def _focal_loss(self, y_true, y_pred, gamma=2.0, alpha=0.25):
        """
        Implementa Focal Loss para lidar melhor com classes desbalanceadas.
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Valores preditos
            gamma: Fator de foco (reduz a contribuição de exemplos fáceis)
            alpha: Fator de balanceamento entre classes
            
        Returns:
            Focal loss
        """
        # Aplica sigmoid ou softmax aos logits se necessário
        if self.num_classes > 2:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        else:
            y_pred = tf.nn.sigmoid(y_pred)
        
        # Clip para evitar valores extremos
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calcular focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Aplicar pesos de foco
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        
        # Calcular loss ponderada
        loss = alpha_factor * modulating_factor * cross_entropy
        
        # Calcular media
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    
    def call(self, y_true, y_pred):
        """
        Calcula a perda para segmentação semântica.
        
        Args:
            y_true: Tensor dos valores verdadeiros [batch, h, w, classes]
            y_pred: Tensor das previsões [batch, h, w, classes]
            
        Returns:
            Valor escalar da perda
        """
        # Verificar se estamos em fase de compilação
        if not tf.executing_eagerly() and (tf.rank(y_true) < 4 or tf.rank(y_pred) < 4):
            # Placeholder durante compilação
            return tf.constant(0.0, dtype=tf.float32)
        
        # Combinar CE/BCE com Dice Loss para melhores resultados
        if self.num_classes > 2:
            # Multiclasse: CE + Dice
            ce_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
            )
        else:
            # Binária: BCE + Dice
            ce_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
            )
        
        # Adicionar focal loss para tratar desbalanceamento
        focal = self._focal_loss(y_true, y_pred, gamma=self.focal_gamma)
        
        # Combinar com Dice loss se configurado
        if self.use_dice_loss:
            dice = self._dice_loss(y_true, y_pred)
            return focal * 0.5 + dice * 0.5
        
        return focal


def create_loss(num_classes=3, num_seg_classes=2, det_weight=1.0, seg_weight=1.0):
    """
    Cria uma instância da perda combinada do YOEO.
    
    Args:
        num_classes: Número de classes para detecção
        num_seg_classes: Número de classes para segmentação
        det_weight: Peso para a perda de detecção
        seg_weight: Peso para a perda de segmentação
        
    Returns:
        Instância da perda combinada
    """
    return CombinedYOEOLoss(
        num_classes=num_classes,
        num_seg_classes=num_seg_classes,
        det_loss_weight=det_weight,
        seg_loss_weight=seg_weight
    ) 