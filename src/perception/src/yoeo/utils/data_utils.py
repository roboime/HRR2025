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
    width_shift = config.get('width_shift', 0.1)
    height_shift = config.get('height_shift', 0.1)
    brightness_range = config.get('brightness_range', 0.2)
    horizontal_flip = config.get('horizontal_flip', True)
    
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
    batch_size = config['training']['batch_size']
    input_size = (config['model']['input_height'], config['model']['input_width'])
    
    # Criar pipeline de aumento de dados
    augmentation = create_augmentation_pipeline(config['data_augmentation'])
    
    # Criar geradores de dados
    train_generator = YOEODataGenerator(
        config['dataset']['train_images'],
        config['dataset']['train_masks'],
        config['dataset']['train_annotations'],
        batch_size=batch_size,
        input_size=input_size,
        num_classes=len(config['model']['classes']),
        num_seg_classes=len(config['model']['segmentation_classes']),
        augmentation=augmentation,
        shuffle=True
    )
    
    val_generator = YOEODataGenerator(
        config['dataset']['val_images'],
        config['dataset']['val_masks'],
        config['dataset']['val_annotations'],
        batch_size=batch_size,
        input_size=input_size,
        num_classes=len(config['model']['classes']),
        num_seg_classes=len(config['model']['segmentation_classes']),
        augmentation=None,
        shuffle=False
    )
    
    # Criar gerador de teste se os caminhos estiverem definidos
    test_generator = None
    if all(key in config['dataset'] for key in ['test_images', 'test_masks', 'test_annotations']):
        test_generator = YOEODataGenerator(
            config['dataset']['test_images'],
            config['dataset']['test_masks'],
            config['dataset']['test_annotations'],
            batch_size=1,  # Batch size 1 para teste
            input_size=input_size,
            num_classes=len(config['model']['classes']),
            num_seg_classes=len(config['model']['segmentation_classes']),
            augmentation=None,
            shuffle=False
        )
    
    return train_generator, val_generator, test_generator


class YOEODataGenerator(Sequence):
    """
    Gerador de dados para o modelo YOEO.
    
    Este gerador carrega imagens, máscaras e anotações, aplica
    aumento de dados e gera lotes para treinamento e avaliação.
    """
    
    def __init__(self, image_dir, mask_dir, annotation_dir, batch_size=8,
                 input_size=(416, 416), num_classes=4, num_seg_classes=3,
                 augmentation=None, shuffle=True):
        """
        Inicializa o gerador de dados.
        
        Args:
            image_dir: Diretório contendo imagens
            mask_dir: Diretório contendo máscaras de segmentação
            annotation_dir: Diretório contendo anotações de detecção
            batch_size: Tamanho do lote
            input_size: Tamanho de entrada do modelo (altura, largura)
            num_classes: Número de classes de detecção
            num_seg_classes: Número de classes de segmentação
            augmentation: Pipeline de aumento de dados
            shuffle: Se deve embaralhar os dados a cada época
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.annotation_dir = annotation_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.augmentation = augmentation
        self.shuffle = shuffle
        
        # Listar arquivos de imagem
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.indexes = np.arange(len(self.image_files))
        
        # Embaralhar no início
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        """Retorna o número de lotes por época."""
        return int(np.ceil(len(self.image_files) / self.batch_size))
    
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
        batch_files = [self.image_files[i] for i in batch_indexes]
        
        # Inicializar arrays para imagens e saídas
        batch_images = np.zeros((len(batch_files), *self.input_size, 3), dtype=np.float32)
        batch_detection = np.zeros((len(batch_files), 100, 5 + self.num_classes), dtype=np.float32)
        batch_segmentation = np.zeros((len(batch_files), *self.input_size, self.num_seg_classes), dtype=np.float32)
        
        # Carregar e processar cada imagem no lote
        for i, image_file in enumerate(batch_files):
            # Construir caminhos para imagem, máscara e anotação
            image_path = os.path.join(self.image_dir, image_file)
            mask_path = os.path.join(self.mask_dir, os.path.splitext(image_file)[0] + '.png')
            annotation_path = os.path.join(self.annotation_dir, os.path.splitext(image_file)[0] + '.json')
            
            # Carregar imagem, máscara e anotações
            image = load_image(image_path)
            mask = load_mask(mask_path)
            annotations = load_annotations(annotation_path)
            
            # Extrair caixas delimitadoras e classes
            boxes = []
            class_labels = []
            for ann in annotations:
                # Formato YOLO: [x_center, y_center, width, height]
                x_center = ann['bbox'][0] + ann['bbox'][2] / 2
                y_center = ann['bbox'][1] + ann['bbox'][3] / 2
                width = ann['bbox'][2]
                height = ann['bbox'][3]
                
                # Normalizar para [0, 1]
                x_center /= image.shape[1]
                y_center /= image.shape[0]
                width /= image.shape[1]
                height /= image.shape[0]
                
                boxes.append([x_center, y_center, width, height])
                class_labels.append(ann['category_id'])
            
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
            
            # Redimensionar imagem e máscara
            image = cv2.resize(image, self.input_size[::-1])  # OpenCV usa (largura, altura)
            mask = cv2.resize(mask, self.input_size[::-1], interpolation=cv2.INTER_NEAREST)
            
            # Normalizar imagem
            image = normalize_image(image)
            
            # Converter máscara para one-hot encoding
            mask_one_hot = np.zeros((*self.input_size, self.num_seg_classes), dtype=np.float32)
            for c in range(self.num_seg_classes):
                mask_one_hot[..., c] = (mask == c).astype(np.float32)
            
            # Preencher arrays do lote
            batch_images[i] = image
            batch_segmentation[i] = mask_one_hot
            
            # Preencher detecções
            for j, (box, class_id) in enumerate(zip(boxes, class_labels)):
                if j >= 100:  # Limitar a 100 objetos por imagem
                    break
                
                # Formato: [x_center, y_center, width, height, objectness, class_probs]
                batch_detection[i, j, :4] = box
                batch_detection[i, j, 4] = 1.0  # Objectness
                batch_detection[i, j, 5 + class_id] = 1.0  # Classe one-hot
        
        # Retornar imagens e saídas esperadas
        return batch_images, [batch_detection, batch_segmentation]
    
    def on_epoch_end(self):
        """Chamado no final de cada época."""
        if self.shuffle:
            np.random.shuffle(self.indexes)


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