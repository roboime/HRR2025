#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import cv2
import json
import random
from datetime import datetime

from .yoeo_model import YOEOModel

class YOEODataGenerator(tf.keras.utils.Sequence):
    """
    Gerador de dados para treinamento do modelo YOEO.
    
    Este gerador carrega imagens e anotações de um diretório e as prepara
    para o treinamento do modelo YOEO.
    """
    
    def __init__(self, 
                 annotation_path, 
                 image_dir, 
                 batch_size=8, 
                 input_shape=(416, 416), 
                 shuffle=True,
                 augment=True,
                 class_names=['bola', 'gol', 'robo', 'arbitro']):
        """
        Inicializa o gerador de dados.
        
        Args:
            annotation_path: Caminho para o arquivo de anotações (formato COCO ou YOLO)
            image_dir: Diretório contendo as imagens
            batch_size: Tamanho do lote
            input_shape: Forma da entrada (altura, largura)
            shuffle: Se deve embaralhar os dados
            augment: Se deve aplicar aumento de dados
            class_names: Nomes das classes
        """
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.augment = augment
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Carregar anotações
        self.annotations = self._load_annotations()
        self.indices = np.arange(len(self.annotations))
        
        # Embaralhar dados
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Retorna o número de lotes por época."""
        return int(np.ceil(len(self.annotations) / self.batch_size))
    
    def __getitem__(self, index):
        """Gera um lote de dados."""
        # Selecionar índices para este lote
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Inicializar arrays para imagens e anotações
        batch_images = np.zeros((len(batch_indices), self.input_shape[0], self.input_shape[1], 3))
        
        # Inicializar arrays para as saídas (3 escalas diferentes)
        # Para cada escala: [batch_size, grid_h, grid_w, anchors, (5 + num_classes)]
        # 5 = [x, y, w, h, objectness]
        grid_shapes = [
            (self.input_shape[0] // 8, self.input_shape[1] // 8),    # Escala grande (52x52 para 416x416)
            (self.input_shape[0] // 16, self.input_shape[1] // 16),  # Escala média (26x26)
            (self.input_shape[0] // 32, self.input_shape[1] // 32)   # Escala pequena (13x13)
        ]
        
        batch_targets = [
            np.zeros((len(batch_indices), grid_shapes[0][0], grid_shapes[0][1], 3, 5 + self.num_classes)),
            np.zeros((len(batch_indices), grid_shapes[1][0], grid_shapes[1][1], 3, 5 + self.num_classes)),
            np.zeros((len(batch_indices), grid_shapes[2][0], grid_shapes[2][1], 3, 5 + self.num_classes))
        ]
        
        # Carregar imagens e anotações
        for i, idx in enumerate(batch_indices):
            # Carregar imagem
            img_path = os.path.join(self.image_dir, self.annotations[idx]['filename'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionar imagem
            img_resized = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            
            # Normalizar imagem
            img_normalized = img_resized / 255.0
            
            # Aplicar aumento de dados se necessário
            if self.augment:
                img_normalized = self._augment_image(img_normalized)
            
            # Armazenar imagem
            batch_images[i] = img_normalized
            
            # Processar anotações
            for box in self.annotations[idx]['boxes']:
                # Obter classe e coordenadas normalizadas
                class_id = box['class_id']
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                
                # Converter para coordenadas absolutas na imagem redimensionada
                x_abs = x * self.input_shape[1]
                y_abs = y * self.input_shape[0]
                w_abs = w * self.input_shape[1]
                h_abs = h * self.input_shape[0]
                
                # Determinar a melhor escala e anchor para esta caixa
                best_scale, best_anchor, best_iou = self._find_best_anchor(w_abs, h_abs)
                
                # Se o IoU for muito baixo, pular esta caixa
                if best_iou < 0.3:
                    continue
                
                # Calcular coordenadas da célula da grade
                grid_x = int(x_abs / (self.input_shape[1] / grid_shapes[best_scale][1]))
                grid_y = int(y_abs / (self.input_shape[0] / grid_shapes[best_scale][0]))
                
                # Calcular deslocamentos dentro da célula
                x_cell = x_abs / (self.input_shape[1] / grid_shapes[best_scale][1]) - grid_x
                y_cell = y_abs / (self.input_shape[0] / grid_shapes[best_scale][0]) - grid_y
                
                # Calcular largura e altura relativas ao anchor
                w_cell = np.log(w_abs / (self.anchors[best_scale][best_anchor][0] * self.input_shape[1]))
                h_cell = np.log(h_abs / (self.anchors[best_scale][best_anchor][1] * self.input_shape[0]))
                
                # Atribuir valores ao tensor de saída
                batch_targets[best_scale][i, grid_y, grid_x, best_anchor, 0] = x_cell
                batch_targets[best_scale][i, grid_y, grid_x, best_anchor, 1] = y_cell
                batch_targets[best_scale][i, grid_y, grid_x, best_anchor, 2] = w_cell
                batch_targets[best_scale][i, grid_y, grid_x, best_anchor, 3] = h_cell
                batch_targets[best_scale][i, grid_y, grid_x, best_anchor, 4] = 1.0  # objectness
                batch_targets[best_scale][i, grid_y, grid_x, best_anchor, 5 + class_id] = 1.0  # classe
        
        return batch_images, batch_targets
    
    def on_epoch_end(self):
        """Chamado no final de cada época."""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_annotations(self):
        """
        Carrega anotações do arquivo.
        Suporta formatos COCO e YOLO.
        """
        annotations = []
        
        # Verificar extensão do arquivo
        ext = os.path.splitext(self.annotation_path)[1].lower()
        
        if ext == '.json':  # Formato COCO
            with open(self.annotation_path, 'r') as f:
                coco_data = json.load(f)
            
            # Mapear categorias
            categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
            class_map = {cat_name: i for i, cat_name in enumerate(self.class_names)}
            
            # Processar imagens e anotações
            for img in coco_data['images']:
                img_id = img['id']
                img_file = img['file_name']
                img_width = img['width']
                img_height = img['height']
                
                # Encontrar anotações para esta imagem
                img_annotations = []
                for ann in coco_data['annotations']:
                    if ann['image_id'] == img_id:
                        cat_name = categories[ann['category_id']]
                        if cat_name in class_map:
                            class_id = class_map[cat_name]
                            
                            # Obter coordenadas da caixa
                            x, y, w, h = ann['bbox']
                            
                            # Normalizar coordenadas
                            x_center = (x + w/2) / img_width
                            y_center = (y + h/2) / img_height
                            w_norm = w / img_width
                            h_norm = h / img_height
                            
                            img_annotations.append({
                                'class_id': class_id,
                                'x': x_center,
                                'y': y_center,
                                'w': w_norm,
                                'h': h_norm
                            })
                
                annotations.append({
                    'filename': img_file,
                    'width': img_width,
                    'height': img_height,
                    'boxes': img_annotations
                })
                
        elif ext == '.txt':  # Formato YOLO
            # Ler arquivo de anotações
            with open(self.annotation_path, 'r') as f:
                lines = f.readlines()
            
            # Processar cada linha
            current_img = None
            current_boxes = []
            
            for line in lines:
                line = line.strip()
                if line.endswith('.jpg') or line.endswith('.png'):
                    # Nova imagem
                    if current_img is not None:
                        annotations.append({
                            'filename': current_img,
                            'boxes': current_boxes
                        })
                    
                    current_img = line
                    current_boxes = []
                else:
                    # Anotação de caixa
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        
                        current_boxes.append({
                            'class_id': class_id,
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h
                        })
            
            # Adicionar última imagem
            if current_img is not None:
                annotations.append({
                    'filename': current_img,
                    'boxes': current_boxes
                })
        
        else:
            raise ValueError(f"Formato de anotação não suportado: {ext}")
        
        return annotations
    
    def _find_best_anchor(self, width, height):
        """
        Encontra o melhor anchor e escala para uma caixa.
        
        Args:
            width: Largura da caixa em pixels
            height: Altura da caixa em pixels
            
        Returns:
            (melhor_escala, melhor_anchor, melhor_iou)
        """
        # Anchors para diferentes escalas
        self.anchors = np.array([
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # Escala grande (52x52)
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],  # Escala média (26x26)
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]   # Escala pequena (13x13)
        ])
        
        # Normalizar largura e altura
        w_norm = width / self.input_shape[1]
        h_norm = height / self.input_shape[0]
        
        # Calcular IoU para cada anchor
        best_iou = -1
        best_scale = 0
        best_anchor = 0
        
        for s in range(3):  # Para cada escala
            for a in range(3):  # Para cada anchor
                anchor_w, anchor_h = self.anchors[s][a]
                
                # Calcular área de interseção
                inter_w = min(w_norm, anchor_w)
                inter_h = min(h_norm, anchor_h)
                inter_area = inter_w * inter_h
                
                # Calcular área da união
                box_area = w_norm * h_norm
                anchor_area = anchor_w * anchor_h
                union_area = box_area + anchor_area - inter_area
                
                # Calcular IoU
                iou = inter_area / union_area
                
                if iou > best_iou:
                    best_iou = iou
                    best_scale = s
                    best_anchor = a
        
        return best_scale, best_anchor, best_iou
    
    def _augment_image(self, image):
        """
        Aplica aumento de dados a uma imagem.
        
        Args:
            image: Imagem normalizada
            
        Returns:
            Imagem aumentada
        """
        # Lista de possíveis aumentos
        augmentations = [
            lambda img: img,  # Identidade (sem aumento)
            lambda img: tf.image.flip_left_right(img),  # Espelhar horizontalmente
            lambda img: tf.image.random_brightness(img, 0.2),  # Ajustar brilho
            lambda img: tf.image.random_contrast(img, 0.8, 1.2),  # Ajustar contraste
            lambda img: tf.image.random_hue(img, 0.1),  # Ajustar matiz
            lambda img: tf.image.random_saturation(img, 0.8, 1.2)  # Ajustar saturação
        ]
        
        # Escolher aleatoriamente um aumento
        aug_func = random.choice(augmentations)
        
        # Aplicar aumento
        return aug_func(image)

def yoeo_loss(y_true, y_pred):
    """
    Função de perda personalizada para o modelo YOEO.
    
    Esta função combina:
    - Perda de localização (MSE para x, y, w, h)
    - Perda de confiança (BCE para objectness)
    - Perda de classificação (BCE para classes)
    
    Args:
        y_true: Tensor de verdade fundamental
        y_pred: Tensor de previsão
        
    Returns:
        Valor da perda
    """
    # Pesos para diferentes componentes da perda
    lambda_coord = 5.0  # Peso para erros de coordenadas
    lambda_noobj = 0.5  # Peso para células sem objetos
    
    # Extrair componentes
    # y_true/y_pred shape: [batch, grid_h, grid_w, anchors, (5 + num_classes)]
    # 5 = [x, y, w, h, objectness]
    
    # Coordenadas previstas
    pred_xy = y_pred[..., 0:2]
    pred_wh = y_pred[..., 2:4]
    pred_conf = y_pred[..., 4:5]
    pred_class = y_pred[..., 5:]
    
    # Coordenadas verdadeiras
    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]
    true_conf = y_true[..., 4:5]
    true_class = y_true[..., 5:]
    
    # Máscara para células com objetos
    object_mask = true_conf
    noobject_mask = 1.0 - object_mask
    
    # Perda de localização (apenas para células com objetos)
    xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask) * lambda_coord
    wh_loss = tf.reduce_sum(tf.square(true_wh - pred_wh) * object_mask) * lambda_coord
    
    # Perda de confiança
    conf_loss_obj = tf.reduce_sum(tf.square(true_conf - pred_conf) * object_mask)
    conf_loss_noobj = tf.reduce_sum(tf.square(true_conf - pred_conf) * noobject_mask) * lambda_noobj
    
    # Perda de classificação (apenas para células com objetos)
    class_loss = tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(true_class, pred_class) * object_mask
    )
    
    # Perda total
    total_loss = xy_loss + wh_loss + conf_loss_obj + conf_loss_noobj + class_loss
    
    return total_loss

def train_yoeo(
    train_annotation_path,
    train_image_dir,
    val_annotation_path=None,
    val_image_dir=None,
    output_dir='models',
    batch_size=8,
    epochs=100,
    learning_rate=0.001,
    input_shape=(416, 416),
    pretrained_weights=None,
    class_names=['bola', 'gol', 'robo', 'arbitro']
):
    """
    Treina o modelo YOEO.
    
    Args:
        train_annotation_path: Caminho para o arquivo de anotações de treinamento
        train_image_dir: Diretório contendo as imagens de treinamento
        val_annotation_path: Caminho para o arquivo de anotações de validação
        val_image_dir: Diretório contendo as imagens de validação
        output_dir: Diretório para salvar os modelos
        batch_size: Tamanho do lote
        epochs: Número de épocas
        learning_rate: Taxa de aprendizado
        input_shape: Forma da entrada (altura, largura)
        pretrained_weights: Caminho para pesos pré-treinados
        class_names: Nomes das classes
    """
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Criar geradores de dados
    train_generator = YOEODataGenerator(
        train_annotation_path,
        train_image_dir,
        batch_size=batch_size,
        input_shape=input_shape,
        shuffle=True,
        augment=True,
        class_names=class_names
    )
    
    if val_annotation_path and val_image_dir:
        val_generator = YOEODataGenerator(
            val_annotation_path,
            val_image_dir,
            batch_size=batch_size,
            input_shape=input_shape,
            shuffle=False,
            augment=False,
            class_names=class_names
        )
    else:
        val_generator = None
    
    # Criar modelo
    model = YOEOModel(
        input_shape=(input_shape[0], input_shape[1], 3),
        num_classes=len(class_names),
        backbone='mobilenetv2'
    ).get_model()
    
    # Carregar pesos pré-treinados se disponíveis
    if pretrained_weights and os.path.exists(pretrained_weights):
        model.load_weights(pretrained_weights)
        print(f"Pesos pré-treinados carregados de {pretrained_weights}")
    
    # Compilar modelo
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=yoeo_loss
    )
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(output_dir, 'logs', timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(output_dir, 'yoeo_model_epoch_{epoch:02d}_loss_{loss:.4f}.h5'),
            save_best_only=True,
            save_weights_only=False,
            monitor='val_loss' if val_generator else 'loss',
            mode='min'
        ),
        TensorBoard(log_dir=log_dir),
        ReduceLROnPlateau(
            monitor='val_loss' if val_generator else 'loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6
        ),
        EarlyStopping(
            monitor='val_loss' if val_generator else 'loss',
            patience=15,
            restore_best_weights=True
        )
    ]
    
    # Treinar modelo
    print("Iniciando treinamento do modelo YOEO...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Salvar modelo final
    final_model_path = os.path.join(output_dir, 'yoeo_model_final.h5')
    model.save(final_model_path)
    print(f"Modelo final salvo em {final_model_path}")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar modelo YOEO")
    parser.add_argument("--train_annotations", type=str, required=True,
                        help="Caminho para o arquivo de anotações de treinamento")
    parser.add_argument("--train_images", type=str, required=True,
                        help="Diretório contendo as imagens de treinamento")
    parser.add_argument("--val_annotations", type=str,
                        help="Caminho para o arquivo de anotações de validação")
    parser.add_argument("--val_images", type=str,
                        help="Diretório contendo as imagens de validação")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Diretório para salvar os modelos")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Tamanho do lote")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Número de épocas")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Taxa de aprendizado")
    parser.add_argument("--input_height", type=int, default=416,
                        help="Altura da entrada")
    parser.add_argument("--input_width", type=int, default=416,
                        help="Largura da entrada")
    parser.add_argument("--pretrained_weights", type=str,
                        help="Caminho para pesos pré-treinados")
    
    args = parser.parse_args()
    
    train_yoeo(
        train_annotation_path=args.train_annotations,
        train_image_dir=args.train_images,
        val_annotation_path=args.val_annotations,
        val_image_dir=args.val_images,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        input_shape=(args.input_height, args.input_width),
        pretrained_weights=args.pretrained_weights
    ) 