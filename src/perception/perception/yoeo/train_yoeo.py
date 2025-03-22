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
from utils.loss_utils import create_loss

# Configurar GPU (se disponível)
def setup_gpu():
    """Configura a GPU para treinamento, ou otimiza a CPU se nenhuma GPU estiver disponível."""
    # Verificar se TensorFlow está usando a aceleração de hardware correta
    print("\n--- Configuração de Hardware ---")
    
    # Verificar dispositivos disponíveis
    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')
    
    print(f"CPUs disponíveis: {len(cpus)}")
    print(f"GPUs disponíveis: {len(gpus)}")
    
    if gpus:
        try:
            # Configurar para usar a primeira GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # Permitir crescimento de memória conforme necessário
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Verificar se a GPU está configurada corretamente
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"GPUs físicas: {len(gpus)}, GPUs lógicas: {len(logical_gpus)}")
            
            # Verificar disponibilidade da GPU
            if tf.test.is_gpu_available():
                print("✓ TensorFlow está usando GPU")
            else:
                print("✗ TensorFlow não está usando GPU apesar de estar disponível")
            
            print(f"Dispositivos visíveis: {tf.config.get_visible_devices()}")
        except RuntimeError as e:
            print(f"Erro na configuração da GPU: {e}")
            print("Usando CPU como fallback.")
    else:
        print("\n⚠️ AVISO: Nenhuma GPU encontrada. O treinamento usará CPU, que será significativamente mais lento.")
        print("Verificar:")
        print("1. Se o TensorFlow-GPU está instalado corretamente (pip install tensorflow)")
        print("2. Se os drivers NVIDIA estão instalados")
        print("3. Se o CUDA Toolkit e cuDNN estão instalados e compatíveis com esta versão do TensorFlow")
        
        # Otimizar para CPU
        try:
            # Usar threads paralelos para melhorar desempenho da CPU
            tf.config.threading.set_intra_op_parallelism_threads(
                len(cpus) * 2  # Usar o dobro do número de CPUs físicas
            )
            tf.config.threading.set_inter_op_parallelism_threads(2)
            
            # Ativar otimizações específicas para CPU
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Usar otimizações oneDNN
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # Ativar XLA
            
            print("\n✓ Otimizações para CPU ativadas")
        except Exception as e:
            print(f"Erro ao ativar otimizações para CPU: {e}")

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
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend()
    
    # Plotar acurácia (se disponível)
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Treinamento')
        if 'val_accuracy' in history.history:
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
    # Imprimir versão do TensorFlow
    print(f"\nTensorFlow versão: {tf.__version__}")
    print(f"Modo Eager: {tf.executing_eagerly()}")
    
    # Verificar dispositivos disponíveis
    print("\nDispositivos TensorFlow disponíveis:")
    for device in tf.config.list_physical_devices():
        print(f"  {device.device_type}: {device.name}")
    
    # Configurar GPU/CPU
    setup_gpu()
    
    # Carregar configuração
    config = load_config(config_path)
    print(f"Configuração carregada de {config_path}")
    
    # Criar diretórios de saída
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Definir política de mixed precision para melhorar desempenho
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Usando mixed precision: float16/float32")
    except Exception as e:
        print(f"Não foi possível ativar mixed precision: {e}")
    
    # Preparar datasets
    train_generator, val_generator, test_generator = prepare_dataset(config)
    print(f"Datasets preparados: {len(train_generator)} batches de treinamento, {len(val_generator) if val_generator else 0} batches de validação")
    
    # Criar modelo
    input_shape = (config.get('input_height', 416), config.get('input_width', 416), 3)
    num_classes = len(config['classes'])
    segmentation_classes = len(config['segmentation_classes'])
    
    # Validação e impressão das configurações de classes
    print(f"\nClasses de detecção ({num_classes}): {config['classes']}")
    print(f"Classes de segmentação ({segmentation_classes}): {config['segmentation_classes']}")
    
    # Criar o modelo
    model = YOEOModel(
        input_shape=input_shape,
        num_classes=num_classes,
        num_seg_classes=segmentation_classes
    ).build()
    
    # Inicializar pesos aleatoriamente para evitar problemas com batch normalization
    # e garantir um bom ponto de partida
    dummy_data = np.random.random((1, input_shape[0], input_shape[1], input_shape[2]))
    _ = model(dummy_data)  # Executa forward pass para inicializar todas as variáveis
    
    print("Modelo inicializado com forward pass em dados aleatórios")
    
    # Carregar pesos pré-treinados (se especificado)
    if config.get('pretrained_weights'):
        pretrained_path = config['pretrained_weights']
        if os.path.exists(pretrained_path):
            print(f"Carregando pesos pré-treinados de {pretrained_path}")
            model.load_weights(pretrained_path, by_name=True, skip_mismatch=True)
        else:
            print(f"Arquivo de pesos pré-treinados não encontrado: {pretrained_path}")
    
    # Configurar otimizador com learning rate schedule
    initial_lr = config.get('learning_rate', 0.001)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=len(train_generator) * 5,  # a cada 5 épocas
        decay_rate=0.9,
        staircase=True
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule
    )
    
    # Criar a função de perda combinada
    loss = create_loss(
        num_classes=num_classes,
        num_seg_classes=segmentation_classes,
        det_weight=config.get('detection_loss_weight', 1.0),
        seg_weight=config.get('segmentation_loss_weight', 1.0)
    )
    
    # Compilar modelo com a perda combinada
    model.compile(
        optimizer=optimizer,
        loss=loss
    )
    
    # Configurar callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['checkpoint_dir'], 'yoeo_epoch_{epoch:02d}_loss_{loss:.4f}.weights.h5'),
            save_weights_only=True,
            save_best_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=config.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config['log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S")),
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Treinamento
    print("\nIniciando treinamento...\n")
    
    # Capturar possíveis erros durante o treinamento
    try:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=config.get('epochs', 100),
            callbacks=callbacks,
            verbose=1
        )
        
        # Salvar modelo e gerar visualizações
        # Garantir que o nome do arquivo termine com .weights.h5
        model_path = os.path.join(config['output_dir'], 'yoeo_final_model.weights.h5')
        
        try:
            model.save_weights(model_path)
            print(f"\nModelo final salvo em {model_path}")
        except Exception as e:
            print(f"Erro ao salvar modelo: {e}")
            # Tentar formato alternativo se falhar
            alt_path = os.path.join(config['output_dir'], 'yoeo_final_model.h5')
            try:
                model.save(alt_path)
                print(f"Modelo salvo no formato alternativo: {alt_path}")
            except Exception as e2:
                print(f"Erro ao salvar no formato alternativo: {e2}")
        
        # Plotar histórico de treinamento
        if history.history:
            plot_training_history(history, config['output_dir'])
        
        print("\nTreinamento concluído!\n")
        
        return model, history
        
    except Exception as training_err:
        print(f"\nErro durante o treinamento: {training_err}")
        import traceback
        traceback.print_exc()
        
        # Tentar salvar o modelo mesmo após erro
        try:
            recovery_path = os.path.join(config['output_dir'], 'yoeo_recovery.weights.h5')
            model.save_weights(recovery_path)
            print(f"Modelo salvo após erro em: {recovery_path}")
        except:
            print("Não foi possível salvar o modelo após o erro")
        
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar modelo YOEO")
    parser.add_argument('--config', type=str, default='C:/Users/Keller_/Desktop/RoboIME/HSL2025/src/perception/config/training_config.yaml',
                        help='Caminho para o arquivo de configuração de treinamento')
    args = parser.parse_args()
    
    train_yoeo(args.config) 