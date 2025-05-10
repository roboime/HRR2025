#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitários para processamento de dados do modelo YOLOv4-Tiny.

Este módulo fornece funções e classes para o carregamento, processamento
e preparação de dados para o modelo de detecção YOLOv4-Tiny.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

def load_image(image_path):
    """
    Carrega uma imagem a partir do caminho especificado.
    
    Args:
        image_path: Caminho para o arquivo de imagem
        
    Returns:
        Imagem carregada (array NumPy)
        
    Raises:
        IOError: Se a imagem não puder ser carregada
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(image_path):
            raise IOError(f"Arquivo de imagem não encontrado: {image_path}")
        
        # Carregar imagem com TensorFlow
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        
        # Converter para NumPy array
        image = image.numpy()
        
        # Verificar se a imagem foi carregada corretamente
        if image.size == 0:
            raise IOError(f"Imagem vazia carregada de: {image_path}")
            
        return image
    except Exception as e:
        raise IOError(f"Erro ao carregar a imagem {image_path}: {str(e)}")

def load_annotations(annotation_path):
    """
    Carrega anotações de detecção de objetos a partir de um arquivo JSON.
    
    Args:
        annotation_path: Caminho para o arquivo de anotações
        
    Returns:
        Dicionário contendo anotações de detecções
        
    Raises:
        IOError: Se o arquivo de anotações não puder ser carregado
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(annotation_path):
            raise IOError(f"Arquivo de anotações não encontrado: {annotation_path}")
        
        # Carregar JSON
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
            
        return annotations
    except Exception as e:
        raise IOError(f"Erro ao carregar anotações de {annotation_path}: {str(e)}")

def create_augmentation_pipeline(config):
    """
    Cria uma pipeline de augmentação de dados com base na configuração.
    
    Args:
        config: Dicionário com parâmetros de configuração para augmentação
            - flip_horizontal: Aplicar flip horizontal
            - flip_vertical: Aplicar flip vertical
            - rotation_range: Intervalo de rotação em graus
            - brightness_range: Intervalo de ajuste de brilho
            - zoom_range: Intervalo de zoom
            - random_crop: Aplicar cortes aleatórios
            - random_blur: Aplicar desfoque aleatório
            - mosaic: Aplicar técnica de mosaico
            - mixup: Aplicar técnica de mixup
            - contrast_range: Intervalo de ajuste de contraste
            - saturation_range: Intervalo de ajuste de saturação
            - hue_range: Intervalo de ajuste de matiz
            
    Returns:
        Função de augmentação que pode ser aplicada a imagens e caixas delimitadoras
    """
    # Valores padrão se não especificados
    flip_horizontal = config.get('horizontal_flip', True)
    flip_vertical = config.get('vertical_flip', False)
    rotation_range = config.get('rotation_range', 15)
    brightness_range = config.get('brightness_range', [0.8, 1.2])
    zoom_range = config.get('zoom_range', [0.8, 1.2]) if isinstance(config.get('zoom_range', 0.1), (list, tuple)) else [1.0-config.get('zoom_range', 0.1), 1.0+config.get('zoom_range', 0.1)]
    random_crop = config.get('random_crop', False)
    random_blur = config.get('random_blur', False)
    contrast_range = config.get('contrast_range', [0.8, 1.2])
    saturation_range = config.get('saturation_range', [0.8, 1.2])
    hue_range = config.get('hue_range', [-0.05, 0.05])
    width_shift_range = config.get('width_shift_range', 0.0)
    height_shift_range = config.get('height_shift_range', 0.0)
    
    def augment(image, boxes=None):
        """
        Aplica transformações de augmentação à imagem e caixas delimitadoras.
        
        Args:
            image: Imagem a ser augmentada (array NumPy)
            boxes: Caixas delimitadoras no formato [x_min, y_min, width, height]
            
        Returns:
            Imagem augmentada e caixas delimitadoras ajustadas
        """
        # Converter imagem para tensor
        img_tensor = tf.convert_to_tensor(image)
        h, w = img_tensor.shape[:2]
        
        # Se não houver caixas, criar um array vazio
        if boxes is None:
            boxes = np.zeros((0, 4))
        
        # Cópia das caixas originais para manipulação
        boxes_aug = boxes.copy()
        
        # Flip horizontal
        if flip_horizontal and tf.random.uniform(()) > 0.5:
            img_tensor = tf.image.flip_left_right(img_tensor)
            # Ajustar coordenadas das caixas
            if boxes_aug.shape[0] > 0:
                boxes_aug[:, 0] = w - boxes_aug[:, 0] - boxes_aug[:, 2]
        
        # Flip vertical
        if flip_vertical and tf.random.uniform(()) > 0.5:
            img_tensor = tf.image.flip_up_down(img_tensor)
            # Ajustar coordenadas das caixas
            if boxes_aug.shape[0] > 0:
                boxes_aug[:, 1] = h - boxes_aug[:, 1] - boxes_aug[:, 3]
        
        # Translação horizontal (com verificações de segurança)
        if width_shift_range > 0 and tf.random.uniform(()) > 0.5:
            # Limitar o valor do shift para evitar erros
            max_shift = w * width_shift_range
            shift_x = tf.random.uniform((), -max_shift, max_shift)
            shift_x = tf.cast(shift_x, tf.int32)
            
            # Garantir que os valores de offset e target são válidos
            offset_x = tf.maximum(0, -shift_x)
            target_w = w - tf.abs(shift_x)
            
            # Criar imagem temporária (paddida)
            temp_img = tf.image.pad_to_bounding_box(img_tensor, 0, 0, h, w)
            
            # Aplicar shift apenas se os valores forem válidos
            if target_w > 0 and target_w <= w and offset_x >= 0 and offset_x < w:
                try:
                    if shift_x >= 0:
                        img_tensor = tf.image.crop_to_bounding_box(temp_img, 0, offset_x, h, target_w)
                        img_tensor = tf.image.pad_to_bounding_box(img_tensor, 0, 0, h, w)
                    else:
                        img_tensor = tf.image.crop_to_bounding_box(temp_img, 0, 0, h, target_w)
                        img_tensor = tf.image.pad_to_bounding_box(img_tensor, 0, offset_x, h, w)
                        
                    # Ajustar coordenadas das caixas
                    if boxes_aug.shape[0] > 0:
                        boxes_aug[:, 0] = tf.clip_by_value(boxes_aug[:, 0] + tf.cast(shift_x, tf.float32) / w, 0, 1.0 - boxes_aug[:, 2])
                except:
                    # Em caso de erro, manter a imagem original
                    pass
        
        # Translação vertical (com verificações de segurança)
        if height_shift_range > 0 and tf.random.uniform(()) > 0.5:
            # Limitar o valor do shift para evitar erros
            max_shift = h * height_shift_range
            shift_y = tf.random.uniform((), -max_shift, max_shift)
            shift_y = tf.cast(shift_y, tf.int32)
            
            # Garantir que os valores de offset e target são válidos
            offset_y = tf.maximum(0, -shift_y)
            target_h = h - tf.abs(shift_y)
            
            # Criar imagem temporária (paddida)
            temp_img = tf.image.pad_to_bounding_box(img_tensor, 0, 0, h, w)
            
            # Aplicar shift apenas se os valores forem válidos
            if target_h > 0 and target_h <= h and offset_y >= 0 and offset_y < h:
                try:
                    if shift_y >= 0:
                        img_tensor = tf.image.crop_to_bounding_box(temp_img, offset_y, 0, target_h, w)
                        img_tensor = tf.image.pad_to_bounding_box(img_tensor, 0, 0, h, w)
                    else:
                        img_tensor = tf.image.crop_to_bounding_box(temp_img, 0, 0, target_h, w)
                        img_tensor = tf.image.pad_to_bounding_box(img_tensor, offset_y, 0, h, w)
                        
                    # Ajustar coordenadas das caixas
                    if boxes_aug.shape[0] > 0:
                        boxes_aug[:, 1] = tf.clip_by_value(boxes_aug[:, 1] + tf.cast(shift_y, tf.float32) / h, 0, 1.0 - boxes_aug[:, 3])
                except:
                    # Em caso de erro, manter a imagem original
                    pass
        
        # Ajuste de brilho
        if brightness_range and tf.random.uniform(()) > 0.3:
            factor = tf.random.uniform((), 
                                      brightness_range[0], 
                                      brightness_range[1])
            img_tensor = tf.image.adjust_brightness(img_tensor, factor - 1.0)
        
        # Ajuste de contraste
        if contrast_range and tf.random.uniform(()) > 0.3:
            factor = tf.random.uniform((), 
                                     contrast_range[0], 
                                     contrast_range[1])
            img_tensor = tf.image.adjust_contrast(img_tensor, factor)
        
        # Ajuste de saturação
        if saturation_range and tf.random.uniform(()) > 0.3:
            factor = tf.random.uniform((), 
                                     saturation_range[0], 
                                     saturation_range[1])
            img_tensor = tf.image.adjust_saturation(img_tensor, factor)
        
        # Ajuste de matiz
        if hue_range and tf.random.uniform(()) > 0.3:
            delta = tf.random.uniform((), 
                                     hue_range[0], 
                                     hue_range[1])
            img_tensor = tf.image.adjust_hue(img_tensor, delta)
        
        # Zoom / Escala (Implementação segura)
        if zoom_range and tf.random.uniform(()) > 0.5:
            scale = tf.random.uniform((), zoom_range[0], zoom_range[1])
            scale = tf.cast(scale, tf.float32)
            
            # Calcular novos tamanhos para o zoom
            new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
            new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
            
            # Evitar valores inválidos
            new_h = tf.maximum(1, tf.minimum(new_h, h * 2))
            new_w = tf.maximum(1, tf.minimum(new_w, w * 2))
            
            try:
                # Redimensionar imagem (equivalente ao zoom)
                img_tensor = tf.image.resize(img_tensor, [new_h, new_w])
                
                # Se for zoom in (scale > 1), cortar o excesso de imagem
                if scale > 1:
                    start_h = tf.maximum(0, (new_h - h) // 2)
                    start_w = tf.maximum(0, (new_w - w) // 2)
                    
                    # Garantir que o recorte está dentro dos limites
                    if start_h + h <= new_h and start_w + w <= new_w:
                        img_tensor = tf.image.crop_to_bounding_box(img_tensor, start_h, start_w, h, w)
                    else:
                        # Redimensionar de volta ao tamanho original se as dimensões não forem válidas
                        img_tensor = tf.image.resize(img_tensor, [h, w])
                        
                    # Ajustar caixas para o recorte
                    if boxes_aug.shape[0] > 0:
                        boxes_aug[:, 0] = boxes_aug[:, 0] * scale - tf.cast(start_w, tf.float32) / w
                        boxes_aug[:, 1] = boxes_aug[:, 1] * scale - tf.cast(start_h, tf.float32) / h
                        boxes_aug[:, 2] = boxes_aug[:, 2] * scale
                        boxes_aug[:, 3] = boxes_aug[:, 3] * scale
                # Se for zoom out (scale < 1), preencher com preto
                else:
                    pad_h = tf.maximum(0, (h - new_h) // 2)
                    pad_w = tf.maximum(0, (w - new_w) // 2)
                    
                    # Garantir que o padding está dentro dos limites
                    if pad_h + new_h <= h and pad_w + new_w <= w:
                        img_tensor = tf.image.pad_to_bounding_box(img_tensor, pad_h, pad_w, h, w)
                    else:
                        # Redimensionar de volta ao tamanho original se as dimensões não forem válidas
                        img_tensor = tf.image.resize(img_tensor, [h, w])
                        
                    # Ajustar caixas para o padding
                    if boxes_aug.shape[0] > 0:
                        boxes_aug[:, 0] = boxes_aug[:, 0] * scale + tf.cast(pad_w, tf.float32) / w
                        boxes_aug[:, 1] = boxes_aug[:, 1] * scale + tf.cast(pad_h, tf.float32) / h
                        boxes_aug[:, 2] = boxes_aug[:, 2] * scale
                        boxes_aug[:, 3] = boxes_aug[:, 3] * scale
            except:
                # Em caso de erro, redimensionar de volta ao tamanho original
                img_tensor = tf.image.resize(img_tensor, [h, w])
        
        # Aplicar desfoque gaussiano aleatório (simplificado para evitar erros)
        if random_blur and tf.random.uniform(()) > 0.7:
            try:
                # Método simplificado e mais robusto para desfoque
                img_tensor = tf.image.resize(img_tensor, [h//2, w//2])  # Diminuir resolução
                img_tensor = tf.image.resize(img_tensor, [h, w])  # Voltar à resolução original (causa desfoque)
            except:
                # Em caso de erro, manter a imagem original
                pass
                
        # Cortes aleatórios mantendo objetos (versão simplificada e segura)
        if random_crop and tf.random.uniform(()) > 0.7 and boxes_aug.shape[0] > 0:
            try:
                # Calcular limites do corte para garantir que objetos permaneçam visíveis
                min_x = tf.reduce_min(boxes_aug[:, 0])
                min_y = tf.reduce_min(boxes_aug[:, 1])
                max_x = tf.reduce_max(boxes_aug[:, 0] + boxes_aug[:, 2])
                max_y = tf.reduce_max(boxes_aug[:, 1] + boxes_aug[:, 3])
                
                # Converter para pixels
                min_x = tf.cast(min_x * tf.cast(w, tf.float32), tf.int32)
                min_y = tf.cast(min_y * tf.cast(h, tf.float32), tf.int32)
                max_x = tf.cast(max_x * tf.cast(w, tf.float32), tf.int32)
                max_y = tf.cast(max_y * tf.cast(h, tf.float32), tf.int32)
                
                # Calcular tamanho do corte (60-90% do tamanho original)
                crop_factor = tf.random.uniform((), 0.6, 0.9)
                crop_h = tf.cast(tf.cast(h, tf.float32) * crop_factor, tf.int32)
                crop_w = tf.cast(tf.cast(w, tf.float32) * crop_factor, tf.int32)
                
                # Garantir que o tamanho mínimo de corte é mantido
                crop_h = tf.maximum(crop_h, max_y - min_y)
                crop_w = tf.maximum(crop_w, max_x - min_x)
                
                # Limitar o tamanho para evitar cortes maiores que a imagem
                crop_h = tf.minimum(crop_h, h)
                crop_w = tf.minimum(crop_w, w)
                
                # Escolher um ponto de início para o corte que inclua os objetos
                # Garantir que não saímos dos limites da imagem
                start_x = tf.maximum(0, tf.minimum(min_x, w - crop_w))
                start_y = tf.maximum(0, tf.minimum(min_y, h - crop_h))
                
                # Verificar se o corte é válido
                if crop_w > 0 and crop_h > 0 and start_x + crop_w <= w and start_y + crop_h <= h:
                    # Aplicar o corte
                    cropped_img = tf.image.crop_to_bounding_box(img_tensor, start_y, start_x, crop_h, crop_w)
                    
                    # Redimensionar de volta ao tamanho original
                    img_tensor = tf.image.resize(cropped_img, [h, w])
                    
                    # Ajustar caixas para o corte
                    if boxes_aug.shape[0] > 0:
                        # Converter para coordenadas relativas ao corte
                        start_x_f = tf.cast(start_x, tf.float32) / tf.cast(w, tf.float32)
                        start_y_f = tf.cast(start_y, tf.float32) / tf.cast(h, tf.float32)
                        scale_x = tf.cast(w, tf.float32) / tf.cast(crop_w, tf.float32)
                        scale_y = tf.cast(h, tf.float32) / tf.cast(crop_h, tf.float32)
                        
                        # Ajustar coordenadas
                        boxes_aug[:, 0] = (boxes_aug[:, 0] - start_x_f) * scale_x
                        boxes_aug[:, 1] = (boxes_aug[:, 1] - start_y_f) * scale_y
                        boxes_aug[:, 2] = boxes_aug[:, 2] * scale_x
                        boxes_aug[:, 3] = boxes_aug[:, 3] * scale_y
                        
                        # Clipar valores para 0-1
                        boxes_aug = tf.clip_by_value(boxes_aug, 0, 1.0)
            except:
                # Em caso de erro, ignorar o corte
                pass
        
        # Normalizar imagem para evitar valores fora do intervalo 0-255
        img_tensor = tf.clip_by_value(img_tensor, 0, 255)
        
        # Converter de volta para NumPy
        augmented_image = img_tensor.numpy()
        
        # Clipar valores para 0-1
        boxes_aug = np.clip(boxes_aug, 0, 1.0)
        
        return augmented_image, boxes_aug
    
    return augment

def normalize_image(image):
    """
    Normaliza uma imagem para entrada no modelo YOLOv4-Tiny.
    
    Args:
        image: Imagem a ser normalizada (array NumPy)
        
    Returns:
        Imagem normalizada com valores entre 0 e 1
    """
    # Converter para float32 e normalizar para [0, 1]
    image = image.astype(np.float32) / 255.0
    return image

def prepare_dataset(config):
    """
    Prepara datasets de treinamento, validação e teste com base na configuração.
    
    Args:
        config: Dicionário com configurações para preparação do dataset
            - data_dir: Diretório contendo os dados
            - train_dir: Diretório específico para dados de treinamento
            - val_dir: Diretório específico para dados de validação
            - test_dir: Diretório específico para dados de teste
            - train_annotations: Caminho para anotações de treinamento
            - val_annotations: Caminho para anotações de validação (opcional)
            - test_annotations: Caminho para anotações de teste (opcional)
            - batch_size: Tamanho do batch para os geradores
            - input_shape: Forma de entrada para o modelo [height, width]
            - augmentation: Configuração para augmentação (apenas treinamento)
            
    Returns:
        Tupla contendo (train_generator, val_generator, test_generator)
        Qualquer um dos geradores pode ser None se não existir.
    """
    # Obter diretórios específicos ou usar data_dir como base
    data_dir = config.get('data_dir', '')
    train_dir = config.get('train_dir', os.path.join(data_dir, 'train'))
    val_dir = config.get('val_dir', os.path.join(data_dir, 'valid'))
    test_dir = config.get('test_dir', os.path.join(data_dir, 'test'))
    
    # Caminhos para arquivos de anotações
    train_annotations = config.get('train_annotations', '_annotations.coco.json')
    val_annotations = config.get('val_annotations', '_annotations.coco.json')
    test_annotations = config.get('test_annotations', '_annotations.coco.json')
    
    # Construir caminhos completos para os arquivos de anotações
    train_annot_path = os.path.join(train_dir, os.path.basename(train_annotations))
    val_annot_path = os.path.join(val_dir, os.path.basename(val_annotations))
    test_annot_path = os.path.join(test_dir, os.path.basename(test_annotations))
    
    # Outros parâmetros
    batch_size = config.get('batch_size', 8)
    input_shape = (
        config.get('input_height', 416),
        config.get('input_width', 416)
    )
    augmentation_config = config.get('augmentation', {})
    
    # Criar pipeline de augmentação
    augmentation_fn = create_augmentation_pipeline(augmentation_config)
    
    # Armazenar config global para acesso no gerador
    # Isso permite que o YOEODataGenerator acesse a configuração completa
    if augmentation_fn:
        augmentation_fn.__globals__['config'] = config
    
    # Inicializar geradores como None
    train_generator = None
    val_generator = None 
    test_generator = None
    
    # Dataset de treinamento
    if os.path.exists(train_annot_path):
        print(f"Usando anotações de treinamento: {train_annot_path}")
        train_generator = YOEODataGenerator(
            annotation_path=train_annot_path,
            batch_size=batch_size,
            input_shape=input_shape,
            shuffle=True,
            augmentation=augmentation_fn,
            data_dir=train_dir  # Usar o diretório de treinamento específico
        )
    else:
        print(f"AVISO: Arquivo de anotações de treinamento não encontrado: {train_annot_path}")
        print("Para verificar se os dados existem, execute: ")
        print(f"  ls -la {os.path.dirname(train_annot_path)}")
    
    # Dataset de validação
    if os.path.exists(val_annot_path):
        print(f"Usando anotações de validação: {val_annot_path}")
        val_generator = YOEODataGenerator(
            annotation_path=val_annot_path,
            batch_size=batch_size,
            input_shape=input_shape,
            shuffle=False,
            data_dir=val_dir  # Usar o diretório de validação específico
        )
    
    # Dataset de teste
    if os.path.exists(test_annot_path):
        print(f"Usando anotações de teste: {test_annot_path}")
        test_generator = YOEODataGenerator(
            annotation_path=test_annot_path,
            batch_size=batch_size,
            input_shape=input_shape,
            shuffle=False,
            data_dir=test_dir  # Usar o diretório de teste específico
        )
    
    # Verificar se temos pelo menos um gerador
    if not train_generator and not val_generator and not test_generator:
        print("ERRO: Nenhum dataset foi encontrado. Verifique os caminhos de anotações.")
        print(f"Diretórios procurados:")
        print(f"  - Treinamento: {train_dir}")
        print(f"  - Validação: {val_dir}")
        print(f"  - Teste: {test_dir}")
        
    # Retornar tupla de geradores
    return train_generator, val_generator, test_generator

class YOEODataGenerator(Sequence):
    """
    Gerador de dados para o modelo YOLOv4-Tiny.
    
    Esta classe carrega imagens e anotações no formato COCO 
    e os prepara para o treinamento do modelo.
    """
    
    def __init__(self, annotation_path, batch_size=8, input_shape=(416, 416),
                 shuffle=True, augmentation=None, data_dir=''):
        """
        Inicializa o gerador de dados.
        
        Args:
            annotation_path: Caminho para o arquivo de anotações (formato COCO)
            batch_size: Tamanho do batch
            input_shape: Dimensões da imagem de entrada (height, width)
            shuffle: Se deve embaralhar os dados a cada época
            augmentation: Função de augmentação a ser aplicada (opcional)
            data_dir: Diretório base para imagens (prefixo para caminhos relativos)
        """
        self.annotation_path = annotation_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.data_dir = data_dir
        
        # Configurações para técnicas avançadas de augmentação
        self.use_mosaic = False  # Será habilitado dinamicamente com base no config
        self.use_mixup = False   # Será habilitado dinamicamente com base no config
        self.mosaic_prob = 0.5   # Probabilidade de aplicar mosaic
        self.mixup_prob = 0.3    # Probabilidade de aplicar mixup
        self.mixup_alpha = 0.2   # Parâmetro alpha da distribuição beta para mixup
        
        # Carregar anotações
        self.annotations = load_annotations(annotation_path)
        
        # Extrair informações das anotações
        self.image_info = self.annotations['images']
        self.annotations_info = self.annotations['annotations']
        self.categories = {cat['id']: cat for cat in self.annotations['categories']}
        
        # Organizar anotações por imagem
        self.image_annotations = {}
        for ann in self.annotations_info:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(ann)
        
        # Criar lista de índices de imagens
        self.image_ids = sorted([img['id'] for img in self.image_info])
        self.indexes = np.arange(len(self.image_ids))
        
        # Verificar config de augmentação 
        if augmentation:
            # Verificar se há atributos para mosaic e mixup no config
            if 'mosaic' in augmentation.__globals__.get('config', {}).get('augmentation', {}):
                self.use_mosaic = augmentation.__globals__['config']['augmentation'].get('mosaic', False)
            
            if 'mixup' in augmentation.__globals__.get('config', {}).get('augmentation', {}):
                self.use_mixup = augmentation.__globals__['config']['augmentation'].get('mixup', False)
        
        # Embaralhar dados se necessário
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _load_image_and_boxes(self, image_id):
        """
        Carrega uma imagem e suas caixas delimitadoras.
        
        Args:
            image_id: ID da imagem a ser carregada
            
        Returns:
            image: Imagem carregada e redimensionada
            boxes: Array de caixas delimitadoras no formato [x, y, w, h]
            classes: Array de classes (category_ids)
        """
        # Encontrar informações da imagem
        image_info = next(img for img in self.image_info if img['id'] == image_id)
        
        # Carregar imagem
        file_name = image_info['file_name']
        
        # Determinar o tipo de caminho
        if 'train' in self.annotation_path.lower():
            img_path = os.path.join(os.path.dirname(self.annotation_path), file_name)
        elif 'valid' in self.annotation_path.lower() or 'val' in self.annotation_path.lower():
            img_path = os.path.join(os.path.dirname(self.annotation_path), file_name)
        elif 'test' in self.annotation_path.lower():
            img_path = os.path.join(os.path.dirname(self.annotation_path), file_name)
        else:
            # Caminho padrão
            img_path = os.path.join(self.data_dir, file_name)
        
        # Carregar imagem
        image = load_image(img_path)
        orig_height, orig_width = image.shape[:2]
        
        # Redimensionar imagem
        image = tf.image.resize(image, self.input_shape).numpy()
        
        # Obter anotações para esta imagem
        boxes = []
        classes = []
        
        anns = self.image_annotations.get(image_id, [])
        
        for ann in anns:
            # Extrair coordenadas da caixa delimitadora
            bbox = ann['bbox']  # [x, y, width, height] formato COCO
            
            # Normalizar coordenadas para o intervalo [0, 1]
            x = bbox[0] / orig_width
            y = bbox[1] / orig_height
            w = bbox[2] / orig_width
            h = bbox[3] / orig_height
            
            # Armazenar caixa e classe
            boxes.append([x, y, w, h])
            classes.append(ann['category_id'])
        
        # Converter para arrays NumPy
        boxes = np.array(boxes, dtype=np.float32)
        classes = np.array(classes, dtype=np.int32)
        
        return image, boxes, classes
    
    def _apply_mosaic(self, main_idx):
        """
        Aplica a técnica de mosaico combinando 4 imagens.
        
        Args:
            main_idx: Índice da imagem principal
            
        Returns:
            image: Imagem combinada em mosaico
            boxes: Array combinado de caixas delimitadoras
            classes: Array combinado de classes
        """
        # Selecionar 3 imagens adicionais aleatórias
        available_idxs = list(set(range(len(self.image_ids))) - {main_idx})
        if len(available_idxs) < 3:
            # Se não houver imagens suficientes, retornar imagem original
            image_id = self.image_ids[main_idx]
            return self._load_image_and_boxes(image_id)
        
        additional_idxs = np.random.choice(available_idxs, 3, replace=False)
        all_idxs = [main_idx] + list(additional_idxs)
        
        # Definir tamanho da imagem de saída
        h, w = self.input_shape
        
        # Criar imagem vazia para o mosaico
        mosaic_img = np.zeros((h, w, 3), dtype=np.float32)
        
        # Posição central do mosaico (com variação aleatória)
        center_x = int(w * (0.5 + np.random.uniform(-0.1, 0.1)))
        center_y = int(h * (0.5 + np.random.uniform(-0.1, 0.1)))
        
        # Lista para armazenar todas as caixas e classes
        all_boxes = []
        all_classes = []
        
        # Posições das 4 imagens no mosaico
        positions = [
            [0, 0, center_x, center_y],  # top-left
            [center_x, 0, w, center_y],  # top-right
            [0, center_y, center_x, h],  # bottom-left
            [center_x, center_y, w, h]   # bottom-right
        ]
        
        # Para cada uma das 4 imagens
        for i, idx in enumerate(all_idxs):
            # Carregar imagem e caixas
            image_id = self.image_ids[idx]
            img, boxes, classes = self._load_image_and_boxes(image_id)
            
            # Obter tamanho da imagem atual
            img_h, img_w = img.shape[:2]
            
            # Recortar para posição no mosaico
            x1, y1, x2, y2 = positions[i]
            
            # Calcular largura e altura da região
            region_w = x2 - x1
            region_h = y2 - y1
            
            # Redimensionar imagem para a região
            img_resized = tf.image.resize(img, (region_h, region_w)).numpy()
            
            # Colocar imagem redimensionada no mosaico
            mosaic_img[y1:y2, x1:x2] = img_resized
            
            # Se não houver caixas para esta imagem, continuar para a próxima
            if len(boxes) == 0:
                continue
                
            # Ajustar coordenadas das caixas para o mosaico
            # Converter de [x, y, w, h] para [x1, y1, x2, y2]
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0]  # x
            boxes_xyxy[:, 1] = boxes[:, 1]  # y
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
            
            # Escalar para a região no mosaico
            boxes_xyxy[:, 0] = boxes_xyxy[:, 0] * region_w + x1
            boxes_xyxy[:, 1] = boxes_xyxy[:, 1] * region_h + y1
            boxes_xyxy[:, 2] = boxes_xyxy[:, 2] * region_w + x1
            boxes_xyxy[:, 3] = boxes_xyxy[:, 3] * region_h + y1
            
            # Converter de volta para [x, y, w, h]
            boxes_mosaic = np.zeros_like(boxes)
            boxes_mosaic[:, 0] = boxes_xyxy[:, 0] / w  # normalizado pela largura do mosaico
            boxes_mosaic[:, 1] = boxes_xyxy[:, 1] / h  # normalizado pela altura do mosaico
            boxes_mosaic[:, 2] = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / w  # largura normalizada
            boxes_mosaic[:, 3] = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / h  # altura normalizada
            
            # Clipar valores para garantir que estão entre 0 e 1
            boxes_mosaic = np.clip(boxes_mosaic, 0, 1)
            
            # Adicionar apenas caixas válidas (largura e altura > 0)
            valid_indices = (boxes_mosaic[:, 2] > 0.005) & (boxes_mosaic[:, 3] > 0.005)
            all_boxes.append(boxes_mosaic[valid_indices])
            all_classes.append(classes[valid_indices])
        
        # Concatenar todas as caixas e classes
        if all_boxes:
            all_boxes = np.vstack(all_boxes)
            all_classes = np.concatenate(all_classes)
        else:
            all_boxes = np.zeros((0, 4), dtype=np.float32)
            all_classes = np.array([], dtype=np.int32)
        
        return mosaic_img, all_boxes, all_classes
    
    def _apply_mixup(self, img1, boxes1, classes1):
        """
        Aplica a técnica de mixup misturando duas imagens.
        
        Args:
            img1: Primeira imagem
            boxes1: Caixas da primeira imagem
            classes1: Classes da primeira imagem
            
        Returns:
            image: Imagem misturada
            boxes: Array combinado de caixas delimitadoras
            classes: Array combinado de classes
        """
        # Selecionar uma imagem aleatória
        idx2 = np.random.randint(0, len(self.image_ids))
        image_id2 = self.image_ids[idx2]
        img2, boxes2, classes2 = self._load_image_and_boxes(image_id2)
        
        # Gerar peso para mixup a partir de uma distribuição beta
        alpha = self.mixup_alpha
        lam = np.random.beta(alpha, alpha)
        
        # Combinar imagens
        mixed_img = lam * img1 + (1 - lam) * img2
        
        # Combinar caixas e classes (simplesmente concatenar)
        boxes = np.vstack([boxes1, boxes2]) if len(boxes2) > 0 else boxes1
        classes = np.concatenate([classes1, classes2]) if len(classes2) > 0 else classes1
        
        return mixed_img, boxes, classes
    
    def __len__(self):
        """Retorna o número de batches no gerador."""
        return int(np.ceil(len(self.image_ids) / self.batch_size))
    
    def __getitem__(self, index):
        """
        Retorna um batch de dados.
        
        Args:
            index: Índice do batch
            
        Returns:
            Tupla (batch_images, {"output_1": batch_targets, "output_2": batch_targets}) para treinamento
        """
        # Gerar índices para o batch atual
        batch_indexes = self.indexes[index * self.batch_size:
                                    (index + 1) * self.batch_size]
        
        # Inicializar arrays para imagens e alvos
        batch_images = np.zeros((len(batch_indexes), 
                                *self.input_shape, 3), 
                               dtype=np.float32)
        
        # Para simplicidade, vamos definir uma saída de detecção básica
        # Formato: [batch_size, max_boxes, (x, y, w, h, class_id)]
        max_boxes = 100  # Número máximo de caixas por imagem
        batch_detection_targets = np.zeros((len(batch_indexes), 
                                           max_boxes, 5), 
                                          dtype=np.float32)
        
        # Carregar e processar cada imagem no batch
        for i, idx in enumerate(batch_indexes):
            try:
                # Aplicar técnicas avançadas de augmentação
                image_id = self.image_ids[idx]
                
                # Técnica de Mosaic
                if self.use_mosaic and np.random.random() < self.mosaic_prob:
                    image, boxes, classes = self._apply_mosaic(idx)
                else:
                    # Carregamento normal
                    image, boxes, classes = self._load_image_and_boxes(image_id)
                
                # Técnica de Mixup (aplicada após mosaic)
                if self.use_mixup and np.random.random() < self.mixup_prob and len(boxes) > 0:
                    image, boxes, classes = self._apply_mixup(image, boxes, classes)
                
                # Aplicar augmentação padrão
                if self.augmentation and len(boxes) > 0:
                    image, boxes = self.augmentation(image, boxes)
                
                # Normalizar imagem
                image = normalize_image(image)
                
                # Armazenar imagem no batch
                batch_images[i] = image
                
                # Preencher alvos de detecção
                for j, (box, cls) in enumerate(zip(boxes, classes)):
                    if j < max_boxes:
                        # Formato: [x, y, w, h, class_id]
                        batch_detection_targets[i, j, :4] = box
                        batch_detection_targets[i, j, 4] = cls
                
            except Exception as e:
                print(f"Erro ao processar imagem {idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Manter arrays zerados para esta imagem
        
        # Retornar batch de imagens e alvos no formato de dicionário
        return batch_images, {"output_1": batch_detection_targets, "output_2": batch_detection_targets}
    
    def on_epoch_end(self):
        """Chamado no final de cada época para embaralhar os dados."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

def visualize_batch(batch_images, batch_outputs, class_names, num_samples=4):
    """
    Visualiza um batch de imagens e suas detecções.
    
    Args:
        batch_images: Lote de imagens normalizadas [batch_size, height, width, 3]
        batch_outputs: Saída do modelo para o lote
        class_names: Lista de nomes de classes
        num_samples: Número de amostras a visualizar
        
    Returns:
        Figura matplotlib com as visualizações
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Limitar o número de amostras
        num_samples = min(num_samples, batch_images.shape[0])
        
        # Criar figura
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]
        
        # Para cada amostra
        for i in range(num_samples):
            # Obter imagem
            img = batch_images[i].copy()
            
            # Desnormalizar imagem para visualização
            if img.max() <= 1.0:
                img = img * 255.0
            img = img.astype(np.uint8)
            
            # Exibir imagem
            axes[i].imshow(img)
            
            # Formato de saída depende do tipo de modelo (detecção)
            if isinstance(batch_outputs, np.ndarray):
                # Assumimos que a saída é [batch, num_boxes, 5+]
                # onde 5+ é [x, y, w, h, confidence, class_probs...]
                detections = batch_outputs[i]
                
                # Filtrar detecções por confiança
                confidence_threshold = 0.3
                valid_detections = detections[detections[:, 4] > confidence_threshold]
                
                # Exibir caixas
                for det in valid_detections:
                    x, y, w, h = det[:4]
                    confidence = det[4]
                    
                    # Converter para coordenadas absolutas
                    img_h, img_w = img.shape[:2]
                    x1 = int(x * img_w)
                    y1 = int(y * img_h)
                    width = int(w * img_w)
                    height = int(h * img_h)
                    
                    # Criar retângulo
                    rect = patches.Rectangle(
                        (x1, y1), width, height, 
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    axes[i].add_patch(rect)
                    
                    # Adicionar rótulo
                    class_id = int(det[5]) if len(det) > 5 else 0
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    axes[i].text(
                        x1, y1 - 5, 
                        f"{class_name}: {confidence:.2f}",
                        color='white', fontsize=10, 
                        bbox=dict(facecolor='red', alpha=0.5)
                    )
            
            axes[i].set_title(f"Amostra {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    except ImportError:
        print("matplotlib é necessário para visualizar os resultados.")
        return None 