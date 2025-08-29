#!/usr/bin/env python3
"""
Script de Treinamento YOLOv8 para RoboCup Humanoid League
RoboIME HSL2025 - Jetson Orin Nano Super

Uso:
    python train_model.py
    python train_model.py --config configs/train_config.yaml
    python train_model.py --epochs 200 --batch 8
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Verificar ambiente de treinamento"""
    logger.info("🔍 Verificando ambiente...")
    
    # Verificar CUDA
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ CUDA disponível: {device} ({memory:.1f}GB)")
    else:
        logger.warning("⚠️ CUDA não disponível, usando CPU")
    
    # Verificar Ultralytics
    try:
        from ultralytics import __version__
        logger.info(f"✅ Ultralytics version: {__version__}")
    except ImportError:
        logger.error("❌ Ultralytics não instalado: pip install ultralytics")
        sys.exit(1)
    
    return torch.cuda.is_available()

def load_config(config_path):
    """Carregar configuração de treinamento"""
    if not os.path.exists(config_path):
        logger.error(f"❌ Arquivo de configuração não encontrado: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✅ Configuração carregada: {config_path}")
    return config

def setup_directories(config):
    """Criar diretórios necessários"""
    dirs_to_create = [
        config.get('project', 'logs'),
        'checkpoints',
        'metrics'
    ]
    
    for dir_name in dirs_to_create:
        os.makedirs(dir_name, exist_ok=True)
        logger.info(f"📁 Diretório criado/verificado: {dir_name}")

def train_model(config, args):
    """Executar treinamento do modelo"""
    logger.info("🚀 Iniciando treinamento YOLOv8...")
    
    # Carregar modelo base
    model_name = config.get('model', 'yolov8n.pt')
    logger.info(f"📦 Carregando modelo base: {model_name}")
    
    try:
        model = YOLO(model_name)
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        sys.exit(1)
    
    # Configurações de treinamento
    train_args = {
        'data': config.get('data', 'configs/robocup.yaml'),
        'epochs': args.epochs or config.get('epochs', 100),
        'batch': args.batch or config.get('batch', 16),
        'imgsz': config.get('imgsz', 640),
        'device': config.get('device', 0 if torch.cuda.is_available() else 'cpu'),
        'project': config.get('project', 'logs'),
        'name': args.name or config.get('name', 'robocup_v1'),
        'save_period': config.get('save_period', 10),
        'patience': config.get('patience', 30),
        'amp': config.get('amp', True),
        'plots': config.get('plots', True),
        'verbose': config.get('verbose', True)
    }
    
    # Log dos parâmetros
    logger.info("📋 Parâmetros de treinamento:")
    for key, value in train_args.items():
        logger.info(f"  {key}: {value}")
    
    # Executar treinamento
    try:
        results = model.train(**train_args)
        logger.info("✅ Treinamento concluído com sucesso!")
        
        # Salvar modelo otimizado
        model_path = f"{train_args['project']}/{train_args['name']}/weights/best.pt"
        if os.path.exists(model_path):
            # Exportar para ONNX
            logger.info("📤 Exportando modelo para ONNX...")
            model_best = YOLO(model_path)
            model_best.export(format='onnx', half=True)
            
            # Copiar para pasta de modelos
            target_path = "../models/yolov8/robocup_yolov8.pt"
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            import shutil
            shutil.copy2(model_path, target_path)
            logger.info(f"📋 Modelo copiado para: {target_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erro durante treinamento: {e}")
        sys.exit(1)

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Treinamento YOLOv8 para RoboCup')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Arquivo de configuração')
    parser.add_argument('--epochs', type=int, help='Número de épocas')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--name', type=str, help='Nome do experimento')
    parser.add_argument('--check-only', action='store_true',
                        help='Apenas verificar ambiente')
    
    args = parser.parse_args()
    
    # Verificar ambiente
    cuda_available = check_environment()
    
    if args.check_only:
        logger.info("✅ Verificação de ambiente concluída")
        return
    
    # Carregar configuração
    config = load_config(args.config)
    
    # Configurar diretórios
    setup_directories(config)
    
    # Verificar se dataset existe
    data_path = config.get('data', 'configs/robocup.yaml')
    if not os.path.exists(data_path):
        logger.error(f"❌ Dataset não encontrado: {data_path}")
        logger.info("💡 Certifique-se de que o dataset está configurado corretamente")
        sys.exit(1)
    
    # Executar treinamento
    results = train_model(config, args)
    
    logger.info("🎉 Processo concluído!")
    logger.info("📊 Para visualizar resultados:")
    logger.info(f"  - Logs: {config.get('project', 'logs')}")
    logger.info(f"  - Modelo: ../models/yolov8/robocup_yolov8.pt")

if __name__ == '__main__':
    main() 