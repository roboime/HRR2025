#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

class TensorRTConverter:
    """
    Utilitário para converter modelos TensorFlow para TensorRT.
    
    TensorRT é uma plataforma de otimização de inferência de alta performance da NVIDIA
    que pode acelerar significativamente a inferência em GPUs NVIDIA como a Jetson Nano.
    """
    
    def __init__(self, precision='FP16', max_workspace_size_bytes=8000000000):
        """
        Inicializa o conversor TensorRT.
        
        Args:
            precision: Precisão da conversão ('FP32', 'FP16', ou 'INT8')
            max_workspace_size_bytes: Tamanho máximo do workspace em bytes
        """
        self.precision = precision
        self.max_workspace_size_bytes = max_workspace_size_bytes
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TensorRTConverter")
        
        # Verificar se TensorRT está disponível
        self._check_tensorrt()
    
    def _check_tensorrt(self):
        """Verifica se o TensorRT está disponível no sistema."""
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            self.logger.info("TensorRT está disponível para uso.")
        except ImportError:
            self.logger.error("TensorRT não está disponível. Verifique se o TensorFlow-TensorRT está instalado corretamente.")
            raise ImportError("TensorRT não está disponível")
    
    def convert_saved_model(self, saved_model_dir, output_dir, input_shape=None):
        """
        Converte um modelo salvo do TensorFlow para TensorRT.
        
        Args:
            saved_model_dir: Diretório do modelo salvo
            output_dir: Diretório de saída para o modelo otimizado
            input_shape: Forma da entrada (opcional)
            
        Returns:
            Caminho para o modelo otimizado
        """
        self.logger.info(f"Convertendo modelo de {saved_model_dir} para TensorRT com precisão {self.precision}")
        
        # Criar diretório de saída se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar parâmetros de conversão
        conversion_params = trt.TrtConversionParams(
            precision_mode=self.precision,
            max_workspace_size_bytes=self.max_workspace_size_bytes,
            maximum_cached_engines=1
        )
        
        # Criar conversor
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=saved_model_dir,
            conversion_params=conversion_params
        )
        
        # Converter o modelo
        start_time = time.time()
        converter.convert()
        conversion_time = time.time() - start_time
        self.logger.info(f"Conversão concluída em {conversion_time:.2f} segundos")
        
        # Salvar o modelo otimizado
        output_saved_model_dir = os.path.join(output_dir, f"trt_{self.precision.lower()}")
        converter.save(output_saved_model_dir)
        self.logger.info(f"Modelo TensorRT salvo em {output_saved_model_dir}")
        
        return output_saved_model_dir
    
    def convert_keras_model(self, model, output_dir, model_name="yoeo_model"):
        """
        Converte um modelo Keras para TensorRT.
        
        Args:
            model: Modelo Keras
            output_dir: Diretório de saída
            model_name: Nome do modelo
            
        Returns:
            Caminho para o modelo otimizado
        """
        self.logger.info("Convertendo modelo Keras para TensorRT")
        
        # Criar diretório temporário para o modelo salvo
        temp_saved_model_dir = os.path.join(output_dir, "temp_saved_model")
        os.makedirs(temp_saved_model_dir, exist_ok=True)
        
        # Salvar o modelo Keras como SavedModel
        tf.saved_model.save(model, temp_saved_model_dir)
        self.logger.info(f"Modelo Keras salvo temporariamente em {temp_saved_model_dir}")
        
        # Converter o modelo salvo para TensorRT
        trt_model_dir = self.convert_saved_model(temp_saved_model_dir, output_dir)
        
        return trt_model_dir
    
    def benchmark_model(self, model_dir, input_shape, num_runs=100):
        """
        Realiza benchmark do modelo TensorRT.
        
        Args:
            model_dir: Diretório do modelo
            input_shape: Forma da entrada
            num_runs: Número de execuções para o benchmark
            
        Returns:
            Tempo médio de inferência em milissegundos
        """
        self.logger.info(f"Realizando benchmark do modelo em {model_dir}")
        
        # Carregar o modelo
        model = tf.saved_model.load(model_dir)
        infer = model.signatures["serving_default"]
        
        # Criar dados de entrada aleatórios
        input_tensor = tf.random.normal(input_shape)
        
        # Aquecer o modelo
        for _ in range(10):
            infer(input_tensor)
        
        # Medir o tempo de inferência
        start_time = time.time()
        for _ in range(num_runs):
            infer(input_tensor)
        end_time = time.time()
        
        # Calcular tempo médio
        avg_time_ms = (end_time - start_time) * 1000 / num_runs
        self.logger.info(f"Tempo médio de inferência: {avg_time_ms:.2f} ms")
        
        return avg_time_ms

def convert_yoeo_model(model_path, output_dir, input_shape=(1, 416, 416, 3), precision='FP16'):
    """
    Função de conveniência para converter o modelo YOEO para TensorRT.
    
    Args:
        model_path: Caminho para o modelo YOEO (.h5)
        output_dir: Diretório de saída
        input_shape: Forma da entrada
        precision: Precisão da conversão ('FP32', 'FP16', ou 'INT8')
        
    Returns:
        Caminho para o modelo otimizado
    """
    # Carregar o modelo
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Criar conversor
    converter = TensorRTConverter(precision=precision)
    
    # Converter o modelo
    trt_model_dir = converter.convert_keras_model(model, output_dir)
    
    # Realizar benchmark
    avg_time_ms = converter.benchmark_model(trt_model_dir, input_shape)
    
    print(f"Modelo YOEO convertido para TensorRT com precisão {precision}")
    print(f"Tempo médio de inferência: {avg_time_ms:.2f} ms")
    print(f"Modelo salvo em: {trt_model_dir}")
    
    return trt_model_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Converter modelo YOEO para TensorRT")
    parser.add_argument("--model_path", type=str, required=True, help="Caminho para o modelo YOEO (.h5)")
    parser.add_argument("--output_dir", type=str, required=True, help="Diretório de saída")
    parser.add_argument("--precision", type=str, default="FP16", choices=["FP32", "FP16", "INT8"], 
                        help="Precisão da conversão")
    parser.add_argument("--input_height", type=int, default=416, help="Altura da entrada")
    parser.add_argument("--input_width", type=int, default=416, help="Largura da entrada")
    
    args = parser.parse_args()
    
    input_shape = (1, args.input_height, args.input_width, 3)
    
    convert_yoeo_model(args.model_path, args.output_dir, input_shape, args.precision) 