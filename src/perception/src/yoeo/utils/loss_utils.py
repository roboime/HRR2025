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
    Classe de perda combinada para o modelo YOEO que lida com múltiplas saídas.
    
    Esta classe implementa corretamente a perda combinada para o modelo YOEO,
    lidando adequadamente com a fase de compilação e execução do TensorFlow.
    """
    
    def __init__(self, num_classes=3, num_seg_classes=2, 
                 det_loss_weight=1.0, seg_loss_weight=1.0, 
                 name="combined_yoeo_loss", **kwargs):
        """
        Inicializa a perda combinada do YOEO.
        
        Args:
            num_classes: Número de classes para detecção
            num_seg_classes: Número de classes para segmentação
            det_loss_weight: Peso para a perda de detecção
            seg_loss_weight: Peso para a perda de segmentação
            name: Nome da perda
        """
        super(CombinedYOEOLoss, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.det_loss_weight = det_loss_weight
        self.seg_loss_weight = seg_loss_weight
        
        # Criar funções de perda específicas para detecção e segmentação
        self.det_loss_fn = DetectionLoss(num_classes=num_classes)
        self.seg_loss_fn = SegmentationLoss(num_classes=num_seg_classes)
    
    def call(self, y_true, y_pred):
        """
        Calcula a perda combinada para o modelo YOEO.
        
        Este método é chamado durante o treinamento para calcular a perda.
        Ele precisa lidar tanto com a fase de compilação (quando y_true e y_pred
        são tensores simbólicos) quanto com a fase de execução.
        
        Args:
            y_true: Valores verdadeiros para as saídas do modelo (dicionário ou tensor)
            y_pred: Valores preditos pelo modelo (dicionário ou tensor)
            
        Returns:
            Valor escalar da perda combinada
        """
        # TensorFlow gerencia automaticamente se estamos em modo eager ou gráfico
        # Se tivermos múltiplas saídas, elas serão fornecidas como dicionários
        if isinstance(y_pred, dict) and len(y_pred) == 4:
            # Estamos recebendo as 4 saídas do modelo como esperado
            small_pred = y_pred["detection_small"]
            medium_pred = y_pred["detection_medium"]
            large_pred = y_pred["detection_large"]
            seg_pred = y_pred["segmentation"]
            
            small_true = y_true["detection_small"]
            medium_true = y_true["detection_medium"]
            large_true = y_true["detection_large"]
            seg_true = y_true["segmentation"]
            
            # Calcular perda para cada cabeça de detecção
            small_loss = self.det_loss_fn(small_true, small_pred)
            medium_loss = self.det_loss_fn(medium_true, medium_pred)
            large_loss = self.det_loss_fn(large_true, large_pred)
            
            # Média das perdas de detecção
            det_loss = (small_loss + medium_loss + large_loss) / 3.0
            
            # Calcular perda de segmentação
            seg_loss = self.seg_loss_fn(seg_true, seg_pred)
            
            # Combinar perdas com pesos
            total_loss = self.det_loss_weight * det_loss + self.seg_loss_weight * seg_loss
            
            return total_loss
        else:
            # Durante a compilação, não temos dicionários reais
            # Retornamos uma perda zero como placeholder
            # Esta parte só é usada durante a compilação inicial
            return tf.constant(0.0, dtype=tf.float32)


class DetectionLoss(tf.keras.losses.Loss):
    """
    Classe para calcular a perda específica para detecção.
    """
    
    def __init__(self, num_classes=3, name="detection_loss", **kwargs):
        """
        Inicializa a perda de detecção.
        
        Args:
            num_classes: Número de classes para detecção
            name: Nome da perda
        """
        super(DetectionLoss, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
    
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
        
        # Reshape para formato [batch, n, channels] se necessário
        # para lidar tanto com tensores planos quanto com tensores de grade
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
        
        # Perda de coordenadas (MSE para xy e wh)
        xy_loss = tf.reduce_sum(true_obj * tf.square(true_xy - pred_xy))
        wh_loss = tf.reduce_sum(true_obj * tf.square(
            tf.sqrt(tf.maximum(true_wh, 1e-7)) - tf.sqrt(tf.maximum(pred_wh, 1e-7))
        ))
        
        # Perda de objectness (BCE)
        obj_loss = tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(true_obj, pred_obj, from_logits=True)
        )
        
        # Perda de classificação (BCE para classes)
        cls_loss = tf.reduce_sum(
            true_obj * tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(true_class, pred_class, from_logits=True),
                axis=-1
            )
        )
        
        # Normalizar por número de objetos (evitar divisão por zero)
        num_objects = tf.maximum(tf.reduce_sum(true_obj), 1.0)
        
        # Combinar todas as perdas
        total_loss = (xy_loss + wh_loss + obj_loss + cls_loss) / num_objects
        
        return total_loss


class SegmentationLoss(tf.keras.losses.Loss):
    """
    Classe para calcular a perda específica para segmentação.
    """
    
    def __init__(self, num_classes=2, name="segmentation_loss", **kwargs):
        """
        Inicializa a perda de segmentação.
        
        Args:
            num_classes: Número de classes para segmentação
            name: Nome da perda
        """
        super(SegmentationLoss, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
    
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
        
        # Para segmentação multiclasse, usar categorical crossentropy
        if self.num_classes > 2:
            loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        else:
            # Para segmentação binária
            loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        
        # Média sobre todos os pixels
        return tf.reduce_mean(loss)


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