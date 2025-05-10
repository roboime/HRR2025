#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para treinar o modelo YOLO da Ultralytics.

Este script treina um modelo YOLO para detecção de objetos no contexto de futebol
de robôs, incluindo detecção de bola, gol e robôs.
"""

import argparse
import os
import torch
from ultralytics import YOLO
import yaml
import shutil
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Treinar modelo YOLOv5 da Ultralytics')
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Caminho para o arquivo YAML do dataset')
    parser.add_argument('--model', type=str, default='yolov5n.pt',
                        help='Modelo pré-treinado para iniciar o treinamento')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Número de épocas de treinamento')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Tamanho do batch de treinamento')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Tamanho da imagem de treinamento')
    parser.add_argument('--workers', type=int, default=4,
                        help='Número de workers para carregar dados')
    parser.add_argument('--device', type=str, default='',
                        help='Dispositivo para treinamento (cuda ou cpu)')
    parser.add_argument('--output-dir', type=str, default='runs/train',
                        help='Diretório para salvar os resultados do treinamento')
    parser.add_argument('--name', type=str, default='roboime_yolov5',
                        help='Nome do experimento')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Verificar se o arquivo de dataset existe
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"O arquivo YAML do dataset não foi encontrado: {args.data}")
    
    # Criar diretório de saída se não existir
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detectar dispositivo disponível
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configurar número de workers
    if args.workers <= 0:
        # Detectar o número máximo de threads suportadas pela CPU
        max_workers = torch.multiprocessing.cpu_count()
        # Use um valor moderado, como a metade dos núcleos da CPU
        workers = min(max_workers // 2, 8)
    else:
        workers = args.workers
    
    print(f"Dispositivo: {device}")
    print(f"Workers: {workers}")
    
    # Configurações de ambiente para máxima performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    # Se estiver usando GPU, mostrar informações
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memória disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Carregar o modelo
    print(f"Carregando modelo base: {args.model}")
    model = YOLO(args.model)
    
    # Verificar número de classes
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    num_classes = len(data_config.get('names', []))
    print(f"Número de classes: {num_classes}")
    
    # Imprimir lista de classes para verificação
    class_names = data_config.get('names', [])
    print("Classes para detecção:")
    for i, name in enumerate(class_names):
        print(f" - {i}: {name}")
    
    # Configuração do treinamento
    print("\nIniciando treinamento...")
    start_time = time.time()
    
    # Iniciar o treinamento
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        workers=workers,
        device=device,
        patience=args.patience,
        project=args.output_dir,
        name=args.name,
        pretrained=True,
        amp=True,                # Precision mixed training
        augment=True,            # Usar augmentations
        cos_lr=True,             # Usar learning rate coseno
        close_mosaic=10          # Desativar mosaico nos últimos 10 epochs
    )
    
    # Tempo total de treinamento
    train_time = time.time() - start_time
    hours, remainder = divmod(train_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Treinamento completo em {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Realizar validação
    print("\nRealizando validação final...")
    val_results = model.val(data=args.data)
    
    # Exibir métricas
    print("\nMétricas finais:")
    print(f"mAP@0.5: {val_results.box.map50:.4f}")
    print(f"mAP@0.5-0.95: {val_results.box.map:.4f}")
    
    # Salvar o modelo final
    weights_path = output_dir / 'weights'
    weights_path.mkdir(exist_ok=True)
    final_model_path = weights_path / 'yolov5_roboime_final.pt'
    best_model_path = weights_path / 'yolov5_roboime_best.pt'

    # Copiar melhor modelo para local padrão
    if os.path.exists(output_dir / 'weights' / 'best.pt'):
        shutil.copy(output_dir / 'weights' / 'best.pt', best_model_path)
        print(f"Melhor modelo salvo em: {best_model_path}")
    
    # Copiar modelo final para local padrão
    if os.path.exists(output_dir / 'weights' / 'last.pt'):
        shutil.copy(output_dir / 'weights' / 'last.pt', final_model_path)
        print(f"Modelo final salvo em: {final_model_path}")
    
    print("\nTreinamento concluído!")
    
    # Exportar modelo para ONNX (compatibilidade)
    print("\nExportando modelo para ONNX...")
    try:
        onnx_path = weights_path / 'yolov5_roboime.onnx'
        if os.path.exists(output_dir / 'weights' / 'best.pt'):
            model = YOLO(output_dir / 'weights' / 'best.pt')
            model.export(format='onnx', imgsz=args.img_size, simplify=True, path=onnx_path)
            print(f"Modelo ONNX salvo em: {onnx_path}")
    except Exception as e:
        print(f"Erro ao exportar para ONNX: {str(e)}")


def create_data_yaml(output_path='data.yaml', train_path='datasets/train', val_path='datasets/val',
                    names=['bola', 'gol', 'robo']):
    """
    Cria um arquivo YAML de configuração para o dataset.
    
    Args:
        output_path: Caminho para salvar o arquivo YAML
        train_path: Caminho para as imagens de treinamento
        val_path: Caminho para as imagens de validação
        names: Lista de nomes das classes
    """
    data = {
        'path': '.',  # Raiz do projeto
        'train': train_path,  # Caminho para as imagens de treinamento
        'val': val_path,      # Caminho para as imagens de validação
        'test': '',           # Caminho para as imagens de teste (opcional)
        'names': names,       # Nomes das classes
        'nc': len(names)      # Número de classes
    }
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Salvar o arquivo YAML
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Arquivo de configuração do dataset criado em: {output_path}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Treinamento interrompido pelo usuário.")
    except Exception as e:
        print(f"Erro durante o treinamento: {str(e)}")
        import traceback
        traceback.print_exc()
