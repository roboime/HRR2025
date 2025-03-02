#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from yoeo_model import YOEOModel
from utils.data_utils import prepare_dataset
from utils.losses import yoeo_loss, YOEOLoss

# Configurar GPU (se disponível)
def setup_gpu():
    """Configura a GPU para treinamento."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs disponíveis: {len(gpus)}")
        except RuntimeError as e:
            print(f"Erro na configuração da GPU: {e}")
    else:
        print("Nenhuma GPU encontrada. Usando CPU.")

# Carregar configuração de treinamento
def load_config(config_path):
    """
    Carrega configurações de treinamento do arquivo YAML.
    
    Args:
        config_path: Caminho para o arquivo de configuração
        
    Returns:
        Dicionário com configurações
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Visualizar progresso do treinamento
def plot_training_history(history, output_dir):
    """
    Plota o histórico de treinamento (perdas e métricas).
    
    Args:
        history: Objeto history retornado por model.fit()
        output_dir: Diretório para salvar os gráficos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plotar perdas
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Treinamento')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend()
    
    # Plotar acurácia (se disponível)
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Treinamento')
        plt.plot(history.history['val_accuracy'], label='Validação')
        plt.title('Acurácia do Modelo')
        plt.ylabel('Acurácia')
        plt.xlabel('Época')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

# Exportar modelo para diferentes formatos
def export_model(model, config):
    """
    Exporta o modelo para diferentes formatos.
    
    Args:
        model: Modelo treinado
        config: Configuração de treinamento
    """
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar modelo no formato H5
    h5_path = os.path.join(output_dir, 'yoeo_model.h5')
    model.save_weights(h5_path)
    print(f"Modelo salvo em {h5_path}")
    
    # Salvar modelo no formato SavedModel
    saved_model_path = os.path.join(output_dir, 'yoeo_saved_model')
    tf.keras.models.save_model(model, saved_model_path)
    print(f"Modelo salvo em {saved_model_path}")
    
    # Exportar para TensorRT (se configurado)
    if config.get('export_tensorrt', False):
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            
            trt_path = os.path.join(output_dir, 'yoeo_tensorrt')
            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=saved_model_path,
                precision_mode='FP16'
            )
            converter.convert()
            converter.save(trt_path)
            print(f"Modelo TensorRT salvo em {trt_path}")
        except Exception as e:
            print(f"Erro ao exportar para TensorRT: {e}")

# Função principal de treinamento
def train_yoeo(config_path):
    """
    Treina o modelo YOEO com base nas configurações fornecidas.
    
    Args:
        config_path: Caminho para o arquivo de configuração
    """
    # Configurar GPU
    setup_gpu()
    
    # Carregar configuração
    config = load_config(config_path)
    print(f"Configuração carregada de {config_path}")
    
    # Criar diretórios de saída
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Preparar datasets
    train_generator, val_generator, test_generator = prepare_dataset(config)
    print(f"Datasets preparados: {len(train_generator)} batches de treinamento, {len(val_generator)} batches de validação")
    
    # Criar e compilar modelo
    input_shape = (config['input_height'], config['input_width'], 3)
    num_classes = len(config['classes'])
    segmentation_classes = len(config['segmentation_classes'])
    
    model = YOEOModel(
        input_shape=input_shape,
        num_classes=num_classes,
        num_seg_classes=segmentation_classes,
        anchors=None
    ).build()
    
    # Carregar pesos pré-treinados (se especificado)
    if config.get('pretrained_weights', None):
        pretrained_path = config['pretrained_weights']
        if os.path.exists(pretrained_path):
            model.load_weights(pretrained_path, by_name=True, skip_mismatch=True)
            print(f"Pesos pré-treinados carregados de {pretrained_path}")
        else:
            print(f"Arquivo de pesos pré-treinados não encontrado: {pretrained_path}")
    
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=YOEOLoss(),
        metrics=['accuracy']  # Métricas adicionais podem ser necessárias
    )
    
    # Mostrar resumo do modelo
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['checkpoint_dir'], 'yoeo_epoch_{epoch:02d}_loss_{val_loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=config.get('reduce_lr_patience', 5),
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config['log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    # Treinamento
    print("\nIniciando treinamento...\n")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Salvar histórico de treinamento
    plot_training_history(history, config['output_dir'])
    
    # Exportar modelo
    export_model(model, config)
    
    print("\nTreinamento concluído!\n")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar modelo YOEO")
    parser.add_argument('--config', type=str, default='C:/Users/Keller_/Desktop/RoboIME/HSL2025/src/perception/config/training_config.yaml',
                        help='Caminho para o arquivo de configuração de treinamento')
    args = parser.parse_args()
    
    train_yoeo(args.config) 