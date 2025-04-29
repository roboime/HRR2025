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
    
    Esta versão foi adaptada especificamente para o modelo YOLOv4-Tiny.
    """
    
    def __init__(self, precision='FP16', max_workspace_size_bytes=8000000000, calibration_data=None):
        """
        Inicializa o conversor TensorRT.
        
        Args:
            precision: Precisão da conversão ('FP32', 'FP16', ou 'INT8')
            max_workspace_size_bytes: Tamanho máximo do workspace em bytes
            calibration_data: Dados para calibração INT8 (apenas para precision='INT8')
        """
        self.precision = precision
        self.max_workspace_size_bytes = max_workspace_size_bytes
        self.calibration_data = calibration_data
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TensorRTConverter")
        
        # Verificar se TensorRT está disponível
        self._check_tensorrt()
        
        # Verificar se INT8 foi solicitado e se os dados de calibração estão disponíveis
        if precision == 'INT8' and calibration_data is None:
            self.logger.warning("Precisão INT8 selecionada, mas nenhum dado de calibração fornecido. "
                               "Isto pode resultar em queda significativa de precisão.")
    
    def _check_tensorrt(self):
        """Verifica se o TensorRT está disponível no sistema."""
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            self.logger.info("TensorRT está disponível para uso.")
            
            # Verificar se a versão do TensorRT é compatível
            try:
                trt_version = trt.get_linked_tensorrt_version()
                self.logger.info(f"Versão do TensorRT: {trt_version}")
            except:
                self.logger.warning("Não foi possível determinar a versão do TensorRT.")
                
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
        
        # Configuração específica para INT8
        if self.precision == 'INT8' and self.calibration_data is not None:
            self.logger.info("Realizando calibração INT8 com os dados fornecidos")
            def calibration_input_fn():
                for data in self.calibration_data:
                    yield [tf.constant(data)]
            
            converter.convert(calibration_input_fn=calibration_input_fn)
        else:
            # Converter o modelo (FP32/FP16)
            converter.convert()
        
        # Medição de tempo
        start_time = time.time()
        conversion_time = time.time() - start_time
        self.logger.info(f"Conversão concluída em {conversion_time:.2f} segundos")
        
        # Salvar o modelo otimizado
        output_saved_model_dir = os.path.join(output_dir, f"yolov4_tiny_trt_{self.precision.lower()}")
        converter.save(output_saved_model_dir)
        self.logger.info(f"Modelo TensorRT salvo em {output_saved_model_dir}")
        
        return output_saved_model_dir
    
    def convert_keras_model(self, model, output_dir, model_name="yolov4_tiny"):
        """
        Converte um modelo Keras para TensorRT.
        
        Args:
            model: Modelo Keras
            output_dir: Diretório de saída
            model_name: Nome do modelo
            
        Returns:
            Caminho para o modelo otimizado
        """
        self.logger.info(f"Convertendo modelo Keras {model_name} para TensorRT")
        
        # Criar diretório temporário para o modelo salvo
        temp_saved_model_dir = os.path.join(output_dir, "temp_saved_model")
        os.makedirs(temp_saved_model_dir, exist_ok=True)
        
        # Salvar o modelo Keras como SavedModel
        tf.saved_model.save(model, temp_saved_model_dir)
        self.logger.info(f"Modelo Keras salvo temporariamente em {temp_saved_model_dir}")
        
        # Converter o modelo salvo para TensorRT
        trt_model_dir = self.convert_saved_model(temp_saved_model_dir, output_dir)
        
        # Opcional: Limpar diretório temporário após conversão bem-sucedida
        try:
            import shutil
            shutil.rmtree(temp_saved_model_dir)
            self.logger.info(f"Diretório temporário {temp_saved_model_dir} removido")
        except Exception as e:
            self.logger.warning(f"Não foi possível remover o diretório temporário: {e}")
        
        return trt_model_dir
    
    def benchmark_model(self, model_dir, input_shape, num_runs=100):
        """
        Realiza benchmark do modelo TensorRT.
        
        Args:
            model_dir: Diretório do modelo
            input_shape: Forma da entrada
            num_runs: Número de execuções para o benchmark
            
        Returns:
            Dicionário com estatísticas de inferência (média, min, max, etc.)
        """
        self.logger.info(f"Realizando benchmark do modelo em {model_dir}")
        
        # Carregar o modelo
        model = tf.saved_model.load(model_dir)
        infer = model.signatures["serving_default"]
        
        # Criar dados de entrada aleatórios
        input_tensor = tf.random.normal(input_shape)
        
        # Aquecer o modelo
        self.logger.info("Aquecendo modelo...")
        for _ in range(10):
            infer(input_tensor)
        
        # Medir o tempo de inferência
        self.logger.info(f"Executando {num_runs} inferências para benchmark...")
        inference_times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            infer(input_tensor)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # ms
        
        # Calcular estatísticas
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        std_dev = np.std(inference_times)
        
        stats = {
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "std_dev_ms": std_dev,
            "fps": 1000 / avg_time
        }
        
        self.logger.info(f"Tempo médio de inferência: {avg_time:.2f} ms (FPS: {1000/avg_time:.2f})")
        self.logger.info(f"Tempo mín/máx: {min_time:.2f}/{max_time:.2f} ms, Desvio padrão: {std_dev:.2f} ms")
        
        return stats

def convert_yolov4_tiny(model_path, output_dir, input_shape=(1, 416, 416, 3), precision='FP16', 
                       calibration_data=None, benchmark=True):
    """
    Função de conveniência para converter o modelo YOLOv4-Tiny para TensorRT.
    
    Args:
        model_path: Caminho para o modelo YOLOv4-Tiny (.h5)
        output_dir: Diretório de saída
        input_shape: Forma da entrada
        precision: Precisão da conversão ('FP32', 'FP16', ou 'INT8')
        calibration_data: Dados para calibração INT8 (opcional)
        benchmark: Se deve realizar benchmark após a conversão
        
    Returns:
        Caminho para o modelo otimizado
    """
    print(f"\n===============================================")
    print(f"Convertendo YOLOv4-Tiny para TensorRT ({precision})")
    print(f"===============================================\n")
    
    # Verificar se o arquivo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")
    
    # Carregar o modelo
    try:
        print(f"Carregando modelo de {model_path}...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Modelo carregado: {model.name}, Forma de entrada: {model.input_shape}")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        print("Tentando carregar apenas os pesos...")
        from yoeo.yoeo_model import YOEOModel
        
        input_shape_model = (input_shape[1], input_shape[2], input_shape[3])
        model = YOEOModel(input_shape=input_shape_model).build()
        model.load_weights(model_path)
        print("Pesos carregados com sucesso.")
    
    # Verificar estrutura do modelo
    print(f"\nResumo do modelo:")
    num_layers = len(model.layers)
    print(f"- Total de camadas: {num_layers}")
    print(f"- Entrada: {model.input_shape}")
    print(f"- Saída(s): {[out.shape for out in model.outputs]}")
    
    # Criar conversor
    converter = TensorRTConverter(precision=precision, calibration_data=calibration_data)
    
    # Converter o modelo
    try:
        trt_model_dir = converter.convert_keras_model(model, output_dir, model_name="yolov4_tiny")
        
        # Realizar benchmark se solicitado
        if benchmark:
            stats = converter.benchmark_model(trt_model_dir, input_shape)
            
            # Imprimir comparação de desempenho
            print("\n=== Comparação de Desempenho ===")
            print(f"Modelo: YOLOv4-Tiny")
            print(f"Precisão: {precision}")
            print(f"FPS médio: {stats['fps']:.2f}")
            print(f"Tempo de inferência: {stats['avg_time_ms']:.2f} ms (±{stats['std_dev_ms']:.2f} ms)")
        
        print(f"\nConversão concluída com sucesso!")
        print(f"Modelo TensorRT salvo em: {trt_model_dir}")
        return trt_model_dir
        
    except Exception as e:
        print(f"\nErro durante a conversão: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Converter modelo YOLOv4-Tiny para TensorRT")
    parser.add_argument("--model_path", type=str, required=True, help="Caminho para o modelo YOLOv4-Tiny (.h5)")
    parser.add_argument("--output_dir", type=str, required=True, help="Diretório de saída")
    parser.add_argument("--precision", type=str, default="FP16", choices=["FP32", "FP16", "INT8"], 
                        help="Precisão da conversão")
    parser.add_argument("--input_height", type=int, default=416, help="Altura da entrada")
    parser.add_argument("--input_width", type=int, default=416, help="Largura da entrada")
    parser.add_argument("--skip_benchmark", action="store_true", help="Pular o benchmark após conversão")
    parser.add_argument("--calibration_dataset", type=str, help="Diretório com imagens para calibração INT8 (apenas para --precision=INT8)")
    
    args = parser.parse_args()
    
    input_shape = (1, args.input_height, args.input_width, 3)
    
    # Carregar dados de calibração para INT8 se necessário
    calibration_data = None
    if args.precision == "INT8" and args.calibration_dataset:
        import cv2
        print(f"Carregando imagens para calibração INT8 de {args.calibration_dataset}...")
        
        calibration_data = []
        image_files = [f for f in os.listdir(args.calibration_dataset) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("Nenhuma imagem encontrada para calibração. Continuando sem calibração.")
        else:
            print(f"Encontradas {len(image_files)} imagens para calibração.")
            for i, img_file in enumerate(image_files[:20]):  # Limitar a 20 imagens para calibração
                img_path = os.path.join(args.calibration_dataset, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (args.input_width, args.input_height))
                img = img.astype(np.float32) / 255.0
                img = img.reshape(1, args.input_height, args.input_width, 3)
                calibration_data.append(img)
                
            print(f"Carregadas {len(calibration_data)} imagens para calibração INT8.")
    
    convert_yolov4_tiny(
        args.model_path, 
        args.output_dir, 
        input_shape, 
        args.precision,
        calibration_data,
        not args.skip_benchmark
    ) 