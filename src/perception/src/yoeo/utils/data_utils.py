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
    rotation_range = config['rotation_range']
    width_shift = config['width_shift_range']
    height_shift = config['height_shift_range']
    brightness_range = config['brightness_range']
    horizontal_flip = config['horizontal_flip']
    
    # Criar pipeline de aumento
    transform = A.Compose([
        A.Rotate(limit=rotation_range, p=0.5),
        A.ShiftScaleRotate(shift_limit=max(width_shift, height_shift), scale_limit=0.1, rotate_limit=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=brightness_range, contrast_limit=0.2, p=0.5),
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


class YOEODataGenerator(Sequence):
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
    
    def __len__(self):
        """Retorna o número de lotes por época."""
        return int(np.ceil(len(self.image_ids) / self.batch_size))
    
    def __getitem__(self, index):
        """
        Gera um lote de dados.
        
        Args:
            index: Índice do lote
            
        Returns:
            Tupla (X, Y) onde X são as imagens de entrada e Y são as saídas esperadas
        """
        # Gerar índices para este lote
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.image_ids[i] for i in batch_indexes]
        
        # Inicializar arrays para imagens e saídas
        batch_images = np.zeros((len(batch_ids), *self.input_size, 3), dtype=np.float32)
        batch_detection = np.zeros((len(batch_ids), 100, 5 + self.num_classes), dtype=np.float32)
        batch_segmentation = np.zeros((len(batch_ids), *self.input_size, self.num_seg_classes), dtype=np.float32)
        
        # Carregar e processar cada imagem no lote
        for i, img_id in enumerate(batch_ids):
            # Obter informações da imagem
            img_info = self.image_info[img_id]
            img_anns = self.annotations.get(img_id, [])
            
            # Carregar imagem
            img_path = os.path.join(self.data_dir, img_info['file_name'])
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Não foi possível carregar a imagem: {img_path}")
            
            # Criar máscara de segmentação
            mask = np.zeros((*self.input_size, self.num_seg_classes), dtype=np.float32)
            
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
                        
                        # Adicionar à máscara apropriada
                        seg_idx = self.valid_seg_classes[class_name]
                        mask[..., seg_idx] = np.logical_or(mask[..., seg_idx], seg_mask)
                
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
            
            # Redimensionar imagem
            image = cv2.resize(image, self.input_size[::-1])
            
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
            
            # Normalizar imagem
            image = image.astype(np.float32) / 255.0
            
            # Preencher arrays do lote
            batch_images[i] = image
            batch_segmentation[i] = mask
            
            # Preencher detecções
            for j, (box, class_id) in enumerate(zip(boxes, class_labels)):
                if j >= 100:  # Limitar a 100 objetos por imagem
                    break
                
                # Formato: [x_center, y_center, width, height, objectness, class_probs]
                batch_detection[i, j, :4] = box
                batch_detection[i, j, 4] = 1.0  # Objectness
                batch_detection[i, j, 5 + class_id] = 1.0  # Classe one-hot
        
        return batch_images, [batch_detection, batch_segmentation]
    
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
        batch_outputs: Lote de saídas [detecção, segmentação]
        class_names: Nomes das classes de detecção
        seg_class_names: Nomes das classes de segmentação
        num_samples: Número de amostras a visualizar
        
    Returns:
        Lista de imagens visualizadas
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    batch_detection, batch_segmentation = batch_outputs
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