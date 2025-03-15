#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitários para processamento de dados para o modelo YOEO.

Este módulo fornece funções para carregar, processar e aumentar
dados para treinamento e avaliação do modelo YOEO.
"""

import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import albumentations as A
from pycocotools import mask as coco_mask


def load_image(image_path):
    """
    Carrega uma imagem do disco.
    
    Args:
        image_path: Caminho para a imagem
        
    Returns:
        Imagem no formato BGR
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    return img


def load_mask(mask_path):
    """
    Carrega uma máscara de segmentação do disco.
    
    Args:
        mask_path: Caminho para a máscara
        
    Returns:
        Máscara como array numpy
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Não foi possível carregar a máscara: {mask_path}")
    return mask


def load_annotations(annotation_path):
    """
    Carrega anotações de detecção de objetos de um arquivo JSON.
    
    Args:
        annotation_path: Caminho para o arquivo JSON
        
    Returns:
        Lista de anotações
    """
    try:
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        return annotations
    except Exception as e:
        raise ValueError(f"Erro ao carregar anotações de {annotation_path}: {e}")


def create_augmentation_pipeline(config):
    """
    Cria um pipeline de aumento de dados baseado na configuração.
    
    Args:
        config: Dicionário com configurações de aumento
        
    Returns:
        Pipeline de aumento do Albumentations
    """
    # Extrair parâmetros de configuração com valores padrão
    rotation_range = config.get('rotation_range', 10)
    width_shift = config.get('width_shift_range', 0.1)
    height_shift = config.get('height_shift_range', 0.1)
    brightness_range = config.get('brightness_range', [0.8, 1.2])
    horizontal_flip = config.get('horizontal_flip', True)
    
    # Garantir que os parâmetros não são None
    if rotation_range is None:
        rotation_range = 10
    if width_shift is None:
        width_shift = 0.1
    if height_shift is None:
        height_shift = 0.1
    if brightness_range is None:
        brightness_range = [0.8, 1.2]
    if horizontal_flip is None:
        horizontal_flip = True
        
    # Verificar se é uma lista para o brightness_range
    if isinstance(brightness_range, list) and len(brightness_range) >= 2:
        brightness_limit = brightness_range[1] - brightness_range[0]
    else:
        brightness_limit = 0.2
    
    # Criar pipeline de aumento
    transform = A.Compose([
        A.Rotate(limit=rotation_range, p=0.5),
        A.ShiftScaleRotate(shift_limit=max(width_shift, height_shift), scale_limit=0.1, rotate_limit=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=0.2, p=0.5),
        A.HorizontalFlip(p=0.5 if horizontal_flip else 0),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
        ], p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    return transform


def normalize_image(image):
    """
    Normaliza uma imagem para entrada no modelo.
    
    Args:
        image: Imagem BGR
        
    Returns:
        Imagem normalizada
    """
    # Converter BGR para RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalizar para [0, 1]
    normalized = rgb_image.astype(np.float32) / 255.0
    
    return normalized


def prepare_dataset(config):
    """
    Prepara os conjuntos de dados de treinamento, validação e teste.
    
    Args:
        config: Configuração do treinamento
        
    Returns:
        Geradores de dados para treinamento, validação e teste
    """
    # Extrair parâmetros da configuração
    batch_size = config['batch_size']
    input_size = (config['input_height'], config['input_width'])
    
    # Criar pipeline de aumento de dados
    augmentation = create_augmentation_pipeline(config)
    
    # Criar geradores de dados
    train_generator = YOEODataGenerator(
        data_dir=config['train_dir'],
        batch_size=batch_size,
        input_size=input_size,
        num_classes=len(config['classes']),
        num_seg_classes=len(config['segmentation_classes']),
        augmentation=augmentation,
        shuffle=True
    )
    
    val_generator = YOEODataGenerator(
        data_dir=config['val_dir'],
        batch_size=batch_size,
        input_size=input_size,
        num_classes=len(config['classes']),
        num_seg_classes=len(config['segmentation_classes']),
        augmentation=None,
        shuffle=False
    )
    
    # Criar gerador de teste se o diretório existir
    test_generator = None
    if 'test_dir' in config and os.path.exists(config['test_dir']):
        test_generator = YOEODataGenerator(
            data_dir=config['test_dir'],
            batch_size=1,  # Batch size 1 para teste
            input_size=input_size,
            num_classes=len(config['classes']),
            num_seg_classes=len(config['segmentation_classes']),
            augmentation=None,
            shuffle=False
        )
    
    return train_generator, val_generator, test_generator


class YOEODataGenerator(tf.keras.utils.Sequence):
    """
    Gerador de dados para o modelo YOEO.
    
    Este gerador carrega imagens e anotações no formato COCO,
    que contém tanto segmentação quanto bounding boxes.
    """
    
    def __init__(self, data_dir, batch_size=8, input_size=(416, 416), 
                 num_classes=3, num_seg_classes=2, augmentation=None, shuffle=True):
        """
        Inicializa o gerador de dados.
        
        Args:
            data_dir: Diretório contendo imagens e annotations.json
            batch_size: Tamanho do lote
            input_size: Tamanho de entrada do modelo (altura, largura)
            num_classes: Número de classes de detecção (bola, gol, robo)
            num_seg_classes: Número de classes de segmentação (linha, campo)
            augmentation: Pipeline de aumento de dados
            shuffle: Se deve embaralhar os dados a cada época
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_classes  # Deve ser 3
        self.num_seg_classes = num_seg_classes  # Deve ser 2
        self.augmentation = augmentation
        self.shuffle = shuffle
        
        # Classes válidas
        self.valid_classes = {'bola': 0, 'gol': 1, 'robo': 2}
        self.valid_seg_classes = {'linha': 0, 'campo': 1}
        
        # Carregar anotações COCO
        with open(os.path.join(data_dir, 'annotations.json'), 'r') as f:
            self.coco_data = json.load(f)
        
        # Criar índice de imagens
        self.image_ids = [img['id'] for img in self.coco_data['images']]
        self.image_info = {img['id']: img for img in self.coco_data['images']}
        
        # Criar mapeamento de categorias
        self.cat_mapping = {}
        for cat in self.coco_data['categories']:
            if cat['name'] in self.valid_classes:
                self.cat_mapping[cat['id']] = self.valid_classes[cat['name']]
        
        # Criar índice de anotações por imagem (apenas classes válidas)
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            # Verificar se é uma categoria válida
            if ann['category_id'] in self.cat_mapping:
                if img_id not in self.annotations:
                    self.annotations[img_id] = []
                self.annotations[img_id].append(ann)
        
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        super(YOEODataGenerator, self).__init__()
    
    def __len__(self):
        """Retorna o número de lotes por época."""
        return int(np.ceil(len(self.image_ids) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Retorna um batch de dados.
        
        Args:
            idx: Índice do batch
            
        Returns:
            Tupla (X, Y) onde X é um array Numpy de imagens de entrada
            e Y é uma lista de arrays Numpy [det_small, det_medium, det_large, seg]
        """
        # Calcular os índices das imagens para este batch
        indices = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Inicializar arrays para as imagens e saídas
        batch_images = np.zeros(
            (len(indices), self.input_size[0], self.input_size[1], 3),
            dtype=np.float32
        )
        
        # Inicializar arrays para detecção em diferentes escalas
        det_small_dims = (13, 13)
        det_medium_dims = (26, 26)
        det_large_dims = (52, 52)
        
        batch_det_small = np.zeros(
            (len(indices), det_small_dims[0], det_small_dims[1], 3, 5 + self.num_classes),
            dtype=np.float32
        )
        batch_det_medium = np.zeros(
            (len(indices), det_medium_dims[0], det_medium_dims[1], 3, 5 + self.num_classes),
            dtype=np.float32
        )
        batch_det_large = np.zeros(
            (len(indices), det_large_dims[0], det_large_dims[1], 3, 5 + self.num_classes),
            dtype=np.float32
        )
        
        # Inicializar array para segmentação
        seg_dims = (self.input_size[0], self.input_size[1])
        batch_segmentation = np.zeros(
            (len(indices), seg_dims[0], seg_dims[1], self.num_seg_classes),
            dtype=np.float32
        )
        
        # Processar cada imagem no batch
        for i, image_index in enumerate(indices):
            # Obter caminho da imagem
            img_id = self.image_ids[image_index]
            img_info = self.image_info[img_id]
            img_anns = self.annotations.get(img_id, [])
            
            # Carregar imagem
            img_path = os.path.join(self.data_dir, img_info['file_name'])
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Não foi possível carregar a imagem: {img_path}")
            
            # Criar máscara de segmentação com as mesmas dimensões da imagem
            # Isso é importante para garantir que a augmentação funcione corretamente
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            # Processar anotações
            boxes = []
            class_labels = []
            
            for ann in img_anns:
                # Processar segmentação
                if 'segmentation' in ann:
                    # Verificar se é uma classe de segmentação válida
                    class_name = next((cat['name'] for cat in self.coco_data['categories'] 
                                    if cat['id'] == ann['category_id']), None)
                    
                    if class_name in self.valid_seg_classes:
                        # Converter polígonos/RLE para máscara binária
                        if isinstance(ann['segmentation'], dict):  # RLE
                            seg_mask = self._rle_to_mask(ann['segmentation'], img_info['height'], img_info['width'])
                        else:  # Polígono
                            seg_mask = self._polygon_to_mask(ann['segmentation'], img_info['height'], img_info['width'])
                        
                        # Adicionar à máscara apropriada na dimensão original da imagem
                        seg_mask = cv2.resize(seg_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                        seg_idx = self.valid_seg_classes[class_name]
                        mask[seg_mask > 0] = seg_idx
                
                # Processar bounding box
                if 'bbox' in ann and ann['category_id'] in self.cat_mapping:
                    x, y, w, h = ann['bbox']
                    # Converter para formato YOLO (x_center, y_center, width, height)
                    x_center = (x + w/2) / img_info['width']
                    y_center = (y + h/2) / img_info['height']
                    width = w / img_info['width']
                    height = h / img_info['height']
                    
                    boxes.append([x_center, y_center, width, height])
                    class_labels.append(self.cat_mapping[ann['category_id']])
            
            # Aplicar aumento de dados se disponível
            if self.augmentation and boxes:
                augmented = self.augmentation(
                    image=image,
                    mask=mask,
                    bboxes=boxes,
                    class_labels=class_labels
                )
                image = augmented['image']
                mask = augmented['mask']
                boxes = augmented['bboxes']
                class_labels = augmented['class_labels']
            
            # Redimensionar imagem e máscara para as dimensões de entrada
            image = cv2.resize(image, self.input_size[::-1])
            mask = cv2.resize(mask, self.input_size[::-1], interpolation=cv2.INTER_NEAREST)
            
            # Normalizar imagem
            image = image.astype(np.float32) / 255.0
            
            # Preencher arrays do lote
            batch_images[i] = image
            
            # Processar anotações para detecção
            for j, (box, class_id) in enumerate(zip(boxes, class_labels)):
                if j >= 100:  # Limitar a 100 objetos por imagem
                    break
                
                # Verifica se class_id é válido
                class_id = int(class_id)  # Garante que class_id seja um inteiro
                if class_id >= self.num_classes:
                    print(f"AVISO: class_id {class_id} é maior que o número de classes ({self.num_classes}). Ignorando esta detecção.")
                    continue
                
                # Calcular tamanho do objeto para determinar em qual escala ele deve aparecer
                area = width * height
                
                # Determinar em qual escala o objeto deve aparecer
                scales = []
                if area < 0.04:  # Objetos pequenos
                    scales.append(0)
                if 0.03 < area < 0.16:  # Objetos médios
                    scales.append(1)
                if area > 0.12:  # Objetos grandes
                    scales.append(2)
                
                # Se o objeto não se encaixa em nenhuma escala, colocá-lo na escala mais apropriada
                if not scales:
                    if area < 0.1:
                        scales.append(0)
                    elif area < 0.2:
                        scales.append(1)
                    else:
                        scales.append(2)
                
                # Adicionar objeto a cada escala apropriada
                for scale in scales:
                    # Determinar as dimensões da grade para esta escala
                    grid_height, grid_width = (det_small_dims, det_medium_dims, det_large_dims)[scale]
                    
                    # Calcular a célula da grade onde o centro do objeto cai
                    grid_x = int(x_center * grid_width)
                    grid_y = int(y_center * grid_height)
                    
                    # Restringir para os limites da grade
                    grid_x = max(0, min(grid_x, grid_width - 1))
                    grid_y = max(0, min(grid_y, grid_height - 1))
                    
                    # Calcular deslocamento dentro da célula
                    x_offset = x_center * grid_width - grid_x
                    y_offset = y_center * grid_height - grid_y
                    
                    # Para cada âncora nesta escala
                    for anchor_idx in range(3):
                        # Selecionar o batch de saída correto
                        if scale == 0:
                            output_batch = batch_det_small
                        elif scale == 1:
                            output_batch = batch_det_medium
                        else:
                            output_batch = batch_det_large
                        
                        # Verificar se esta célula já tem um objeto
                        if output_batch[i, grid_y, grid_x, anchor_idx, 4] == 1.0:
                            continue
                        
                        # Preencher dados do objeto
                        output_batch[i, grid_y, grid_x, anchor_idx, 0] = x_offset
                        output_batch[i, grid_y, grid_x, anchor_idx, 1] = y_offset
                        output_batch[i, grid_y, grid_x, anchor_idx, 2] = width
                        output_batch[i, grid_y, grid_x, anchor_idx, 3] = height
                        output_batch[i, grid_y, grid_x, anchor_idx, 4] = 1.0  # objectness
                        output_batch[i, grid_y, grid_x, anchor_idx, 5 + class_id] = 1.0  # class
                        break
            
            # Processar máscara para segmentação (converter para one-hot)
            for c in range(self.num_seg_classes):
                batch_segmentation[i, :, :, c] = (mask == c).astype(np.float32)
        
        # Retornar batch de imagens e saídas como uma tupla
        # TensorFlow espera receber um dict para cada saída em vez de uma lista
        return batch_images, {
            "detection_small": batch_det_small,
            "detection_medium": batch_det_medium,
            "detection_large": batch_det_large,
            "segmentation": batch_segmentation
        }
    
    def on_epoch_end(self):
        """Chamado no final de cada época."""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _rle_to_mask(self, rle, height, width):
        """Converte RLE para máscara binária."""
        if isinstance(rle['counts'], list):
            rle = coco_mask.frPyObjects(rle, height, width)
        mask = coco_mask.decode(rle)
        return cv2.resize(mask, self.input_size[::-1], interpolation=cv2.INTER_NEAREST)
    
    def _polygon_to_mask(self, polygons, height, width):
        """Converte polígonos para máscara binária."""
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            pts = np.array(polygon).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return cv2.resize(mask, self.input_size[::-1], interpolation=cv2.INTER_NEAREST)


def visualize_batch(batch_images, batch_outputs, class_names, seg_class_names, num_samples=4):
    """
    Visualiza um lote de dados para debugging.
    
    Args:
        batch_images: Lote de imagens
        batch_outputs: Saídas do modelo (pode ser uma lista ou um dicionário)
        class_names: Nomes das classes de detecção
        seg_class_names: Nomes das classes de segmentação
        num_samples: Número de amostras a visualizar
        
    Returns:
        Lista de imagens visualizadas
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # Extrair segmentação e usar y_small para detecção
    if isinstance(batch_outputs, dict):
        # Formato de dicionário (antigo formato)
        batch_detection = batch_outputs['detection_small']  # Usar a escala pequena para visualização
        batch_segmentation = batch_outputs['segmentation']
    else:
        # Formato de lista (novo formato)
        batch_detection = batch_outputs[0]  # Primeira escala de detecção (y_small)
        batch_segmentation = batch_outputs[3]  # Segmentação
    
    num_samples = min(num_samples, len(batch_images))
    
    visualizations = []
    
    for i in range(num_samples):
        # Converter imagem de volta para BGR e [0, 255]
        image = batch_images[i].copy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Criar visualização da segmentação
        seg_map = np.argmax(batch_segmentation[i], axis=-1)
        
        # Criar mapa de cores para segmentação
        colors = plt.cm.get_cmap('tab10', len(seg_class_names))
        seg_vis = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        
        for c in range(len(seg_class_names)):
            mask = (seg_map == c)
            color = (np.array(colors(c)[:3]) * 255).astype(np.uint8)
            seg_vis[mask] = color
        
        # Sobrepor segmentação na imagem
        overlay = cv2.addWeighted(image, 0.7, seg_vis, 0.3, 0)
        
        # Desenhar caixas delimitadoras
        for j in range(100):  # Máximo de 100 objetos
            confidence = batch_detection[i, j, 4]
            if confidence > 0.5:  # Limiar de confiança
                # Extrair coordenadas e classe
                x_center, y_center, width, height = batch_detection[i, j, :4]
                class_probs = batch_detection[i, j, 5:]
                class_id = np.argmax(class_probs)
                
                # Converter para coordenadas de pixel
                x1 = int((x_center - width / 2) * image.shape[1])
                y1 = int((y_center - height / 2) * image.shape[0])
                x2 = int((x_center + width / 2) * image.shape[1])
                y2 = int((y_center + height / 2) * image.shape[0])
                
                # Desenhar retângulo e rótulo
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_names[class_id]}: {confidence:.2f}"
                cv2.putText(overlay, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        visualizations.append(overlay)
    
    return visualizations 