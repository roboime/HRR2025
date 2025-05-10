#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conversor de modelos YOLO da Ultralytics para TensorRT.

Este módulo fornece funções para converter modelos YOLO treinados
usando a biblioteca Ultralytics para TensorRT, otimizando a inferência
na plataforma Jetson.
"""

import os
import torch
import argparse
from ultralytics import YOLO
import tensorrt as trt
from pathlib import Path


def convert_to_tensorrt(model_path, output_path=None, imgsz=640, batch_size=1, 
                        precision="fp16", dynamic=False, device="cuda:0"):
    """
    Converte um modelo YOLO da Ultralytics para TensorRT.
    
    Args:
        model_path: Caminho para o modelo .pt treinado
        output_path: Caminho de saída para o modelo TensorRT (None para usar o mesmo nome com .engine)
        imgsz: Tamanho da imagem (altura, largura)
        batch_size: Tamanho do lote para o modelo
        precision: Precisão do modelo ("fp32", "fp16", ou "int8")
        dynamic: Se deve usar formas dinâmicas para entrada
        device: Dispositivo para conversão
        
    Returns:
        Caminho para o modelo TensorRT convertido
    """
    print(f"Convertendo modelo: {model_path}")
    
    # Verificar se o arquivo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    # Configurar caminho de saída
    if output_path is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(os.path.dirname(model_path), f"{model_name}_tensorrt.engine")
    
    # Se for uma lista (altura, largura), converter para tupla
    if isinstance(imgsz, list) and len(imgsz) == 2:
        imgsz = tuple(imgsz)
    # Se for único valor, fazer quadrado
    elif isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    else:
        raise ValueError("imgsz deve ser int ou lista/tupla [altura, largura]")
    
    try:
        print(f"Carregando modelo: {model_path}")
        # Carregar modelo
        model = YOLO(model_path)
        
        # Configurar o modelo para o dispositivo
        model.to(device)
        
        # Converter para TensorRT
        print(f"Convertendo para TensorRT com precisão {precision}...")
        
        # Exportar usando API do Ultralytics
        # A exportação para TensorRT é feita especificando format='engine'
        success = model.export(
            format='engine',          # Formato TensorRT
            imgsz=imgsz,              # Tamanho da imagem
            batch=batch_size,         # Tamanho do lote
            device=device,            # Dispositivo para conversão
            half=(precision == "fp16"),  # Usar precisão de meia-precisão
            simplify=True,            # Simplificar o modelo
            workspace=8,              # Tamanho de workspace em GB
            verbose=True,             # Saída detalhada
            dynamic=dynamic,          # Usar formas dinâmicas para entrada
            path=output_path          # Caminho de saída
        )
        
        if success:
            print(f"Modelo convertido com sucesso: {output_path}")
            return output_path
        else:
            raise RuntimeError("Falha na conversão do modelo para TensorRT")
    
    except Exception as e:
        print(f"Erro ao converter modelo para TensorRT: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def benchmark_tensorrt(model_path, imgsz=640, batch_size=1, num_runs=100, warmup=10, device="cuda:0"):
    """
    Avalia o desempenho de um modelo TensorRT.
    
    Args:
        model_path: Caminho para o modelo TensorRT
        imgsz: Tamanho da imagem (altura, largura)
        batch_size: Tamanho do lote para benchmarking
        num_runs: Número de execuções para média
        warmup: Número de execuções de aquecimento
        device: Dispositivo para benchmarking
        
    Returns:
        Tempo médio de inferência em ms
    """
    try:
        print(f"Carregando modelo TensorRT: {model_path}")
        model = YOLO(model_path)
        model.to(device)
        
        # Criar dados de teste
        # Ultralytics gerará uma imagem de teste para nós
        # Podemos usar profile() ou val() para benchmarking
        print(f"Executando benchmark com {num_runs} execuções (+ {warmup} de aquecimento)")
        
        # Aquecer
        dummy_input = torch.zeros((batch_size, 3, imgsz, imgsz), device=device)
        
        # Warmup
        for _ in range(warmup):
            model(dummy_input)
        
        # Tempo de infereência
        import time
        start_time = time.time()
        
        for _ in range(num_runs):
            model(dummy_input)
            
        end_time = time.time()
        avg_time = (end_time - start_time) * 1000 / num_runs  # ms
        
        print(f"Tempo médio de inferência: {avg_time:.2f} ms")
        print(f"FPS: {1000 / avg_time:.2f}")
        
        return avg_time
        
    except Exception as e:
        print(f"Erro ao realizar benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Configurar analisador de argumentos
    parser = argparse.ArgumentParser(description="Conversor de modelos YOLO para TensorRT")
    parser.add_argument("--model", type=str, required=True, help="Caminho para o modelo .pt treinado")
    parser.add_argument("--output", type=str, default=None, help="Caminho de saída para o modelo TensorRT")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem para o modelo")
    parser.add_argument("--batch-size", type=int, default=1, help="Tamanho do lote para o modelo")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"], help="Precisão do modelo")
    parser.add_argument("--dynamic", action="store_true", help="Usar formas dinâmicas para entrada")
    parser.add_argument("--benchmark", action="store_true", help="Realizar benchmark após a conversão")
    parser.add_argument("--num-runs", type=int, default=100, help="Número de execuções para benchmark")
    
    args = parser.parse_args()
    
    # Converter modelo
    output_path = convert_to_tensorrt(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        precision=args.precision,
        dynamic=args.dynamic
    )
    
    # Executar benchmark se solicitado
    if args.benchmark:
        benchmark_tensorrt(
            model_path=output_path,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            num_runs=args.num_runs
        ) 