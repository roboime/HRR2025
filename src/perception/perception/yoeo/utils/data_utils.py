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
            
    Returns:
        Função de augmentação que pode ser aplicada a imagens e caixas delimitadoras
    """
    # Valores padrão se não especificados
    flip_horizontal = config.get('flip_horizontal', True)
    flip_vertical = config.get('flip_vertical', False)
    rotation_range = config.get('rotation_range', 15)
    brightness_range = config.get('brightness_range', [0.8, 1.2])
    zoom_range = config.get('zoom_range', [0.8, 1.2])
    
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
        
        # Flip horizontal
        if flip_horizontal and tf.random.uniform(()) > 0.5:
            img_tensor = tf.image.flip_left_right(img_tensor)
            # Ajustar coordenadas das caixas
            if boxes.shape[0] > 0:
                boxes[:, 0] = w - boxes[:, 0] - boxes[:, 2]
        
        # Flip vertical
        if flip_vertical and tf.random.uniform(()) > 0.5:
            img_tensor = tf.image.flip_up_down(img_tensor)
            # Ajustar coordenadas das caixas
            if boxes.shape[0] > 0:
                boxes[:, 1] = h - boxes[:, 1] - boxes[:, 3]
        
        # Ajuste de brilho
        if brightness_range:
            factor = tf.random.uniform((), 
                                      brightness_range[0], 
                                      brightness_range[1])
            img_tensor = tf.image.adjust_brightness(img_tensor, factor - 1.0)
        
        # Normalizar imagem para evitar valores fora do intervalo 0-255
        img_tensor = tf.clip_by_value(img_tensor, 0, 255)
        
        # Converter de volta para NumPy
        augmented_image = img_tensor.numpy()
        
        return augmented_image, boxes
    
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
        
        # Embaralhar dados se necessário
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
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
        
        # Selecionar IDs de imagem para este batch
        batch_image_ids = [self.image_ids[i] for i in batch_indexes]
        
        # Inicializar arrays para imagens e alvos
        batch_images = np.zeros((len(batch_image_ids), 
                                *self.input_shape, 3), 
                               dtype=np.float32)
        
        # Para simplicidade, vamos definir uma saída de detecção básica
        # Formato: [batch_size, max_boxes, (x, y, w, h, class_id)]
        max_boxes = 100  # Número máximo de caixas por imagem
        batch_detection_targets = np.zeros((len(batch_image_ids), 
                                           max_boxes, 5), 
                                          dtype=np.float32)
        
        # Carregar e processar cada imagem no batch
        for i, image_id in enumerate(batch_image_ids):
            # Encontrar informações da imagem
            image_info = next(img for img in self.image_info if img['id'] == image_id)
            
            # Carregar imagem - verificar caminhos de anotações
            file_name = image_info['file_name']
            
            # Determinar o tipo de caminho (treinamento, validação ou teste)
            if 'train' in self.annotation_path.lower():
                img_path = os.path.join(os.path.dirname(self.annotation_path), file_name)
            elif 'valid' in self.annotation_path.lower() or 'val' in self.annotation_path.lower():
                img_path = os.path.join(os.path.dirname(self.annotation_path), file_name)
            elif 'test' in self.annotation_path.lower():
                img_path = os.path.join(os.path.dirname(self.annotation_path), file_name)
            else:
                # Caminho padrão
                img_path = os.path.join(self.data_dir, file_name)
            
            try:
                # Tentar carregar a imagem
                image = load_image(img_path)
                orig_height, orig_width = image.shape[:2]
                
                # Redimensionar imagem
                image = tf.image.resize(image, self.input_shape).numpy()
                
                # Aplicar augmentação se disponível
                boxes = []
                classes = []
                
                # Obter anotações para esta imagem
                anns = self.image_annotations.get(image_id, [])
                
                for j, ann in enumerate(anns):
                    if j >= max_boxes:
                        break
                        
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
                
                # Aplicar augmentação se especificada
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
                print(f"Erro ao processar imagem {img_path}: {str(e)}")
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