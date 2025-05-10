#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para converter modelos YOLOv5 PyTorch para TensorRT na Jetson.

Este script facilita a conversão de modelos treinados em formato .pt (PyTorch)
para o formato otimizado TensorRT (.engine) para uso na Jetson Nano.
"""

import os
import sys
import argparse
import time
import subprocess
import logging
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Verificar se o diretório YOLOv5 está no path
YOLOV5_PATH = os.environ.get('YOLOV5_PATH', '/opt/yolov5')
if not os.path.exists(YOLOV5_PATH):
    logger.error(f"Diretório YOLOv5 não encontrado: {YOLOV5_PATH}")
    logger.info("Defina a variável de ambiente YOLOV5_PATH ou clone o YOLOv5 para /opt/yolov5")
    sys.exit(1)

if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)
    logger.info(f"Adicionado {YOLOV5_PATH} ao sys.path")

def parse_args():
    """Processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Converte modelos YOLOv5 para TensorRT")
    
    parser.add_argument(
        '--weights', 
        type=str, 
        required=True, 
        help='Caminho para o arquivo de pesos PyTorch (.pt)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Diretório de saída para o modelo TensorRT (padrão: mesmo do arquivo de entrada)'
    )
    parser.add_argument(
        '--size', 
        type=int, 
        default=640, 
        help='Tamanho da imagem de entrada (padrão: 640)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1, 
        help='Tamanho do batch (padrão: 1)'
    )
    parser.add_argument(
        '--precision', 
        type=str, 
        choices=['fp32', 'fp16', 'int8'], 
        default='fp16',
        help='Precisão do modelo TensorRT (padrão: fp16)'
    )
    parser.add_argument(
        '--device', 
        type=int, 
        default=0, 
        help='Dispositivo CUDA (padrão: 0)'
    )
    parser.add_argument(
        '--workspace', 
        type=int, 
        default=4, 
        help='Tamanho do workspace em GB (padrão: 4)'
    )
    
    return parser.parse_args()

def validate_input(args):
    """Valida os argumentos de entrada."""
    if not os.path.exists(args.weights):
        logger.error(f"Arquivo de pesos não encontrado: {args.weights}")
        return False
    
    if not args.weights.endswith('.pt'):
        logger.error(f"Formato de arquivo não suportado: {args.weights}. Apenas arquivos .pt são suportados.")
        return False
    
    return True

def convert_to_tensorrt(args):
    """Converte o modelo PyTorch para TensorRT."""
    weights_path = os.path.abspath(args.weights)
    output_dir = args.output if args.output else os.path.dirname(weights_path)
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Nome do arquivo de saída
    weights_name = os.path.basename(weights_path).replace('.pt', '')
    output_path = os.path.join(output_dir, f"{weights_name}_{args.size}_{args.precision}.engine")
    
    # Mudar para o diretório YOLOv5
    os.chdir(YOLOV5_PATH)
    logger.info(f"Mudando para o diretório YOLOv5: {YOLOV5_PATH}")
    
    # Comando de exportação
    cmd = [
        "python", "export.py",
        "--weights", weights_path,
        "--include", "engine",
        "--imgsz", str(args.size),
        "--batch-size", str(args.batch_size),
        "--device", str(args.device),
        "--workspace", str(args.workspace),
        "--half" if args.precision == "fp16" else "",
    ]
    
    # Remover opções vazias
    cmd = [opt for opt in cmd if opt]
    
    # Executar o comando
    logger.info(f"Executando comando: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Mostrar a saída em tempo real
        for line in process.stdout:
            print(line, end='')
        
        # Esperar pela conclusão
        process.wait()
        
        if process.returncode != 0:
            logger.error("Falha na execução do comando de exportação")
            return False
        
    except Exception as e:
        logger.error(f"Erro durante a execução do comando: {e}")
        return False
    
    # Calcular tempo de execução
    elapsed_time = time.time() - start_time
    logger.info(f"Conversão concluída em {elapsed_time:.2f} segundos")
    
    # Verificar se o arquivo foi gerado
    expected_output = os.path.join(YOLOV5_PATH, weights_name + ".engine")
    if not os.path.exists(expected_output):
        logger.error(f"Arquivo de saída não encontrado: {expected_output}")
        return False
    
    # Mover o arquivo para o diretório de saída
    try:
        os.rename(expected_output, output_path)
        logger.info(f"Modelo TensorRT salvo em: {output_path}")
    except Exception as e:
        logger.error(f"Erro ao mover o arquivo de saída: {e}")
        return False
    
    return True

def main():
    """Função principal."""
    args = parse_args()
    
    if not validate_input(args):
        sys.exit(1)
    
    logger.info(f"Iniciando conversão do modelo: {args.weights}")
    logger.info(f"Precisão: {args.precision}, Tamanho: {args.size}x{args.size}")
    
    if convert_to_tensorrt(args):
        logger.info("Conversão para TensorRT concluída com sucesso!")
    else:
        logger.error("Falha na conversão para TensorRT")
        sys.exit(1)
    
    logger.info("""
==========================================================
Dicas de Uso:
1. Certifique-se de que o modelo TensorRT seja utilizado no mesmo
   hardware onde foi gerado.
2. Para carregar o modelo, use:
   
   from perception.yoeo.yoeo_model import YOEOModel
   model = YOEOModel()
   model.load_weights('caminho/para/modelo.engine')
   
3. Para melhor desempenho, mantenha a resolução de entrada igual
   à usada durante a conversão.
==========================================================
""")

if __name__ == "__main__":
    main() 