#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import platform
import random
import urllib.request
from pathlib import Path
import zipfile
import requests

# Configurações - ALTERE ESTAS VARIÁVEIS CONFORME SUA NECESSIDADE
DARKNET_DIR = "C:/Users/Keller_/Desktop/RoboIME/darknet"  # Diretório onde o Darknet será instalado
DATASET_DIR = "C:/Users/Keller_/Desktop/RoboIME/HSL2025/src/perception/resources/dataset_darknet"  # Diretório com seus dados
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")
CLASS_NAMES = ["bola", "gol", "robo"]  # Nomes das classes
NUM_CLASSES = len(CLASS_NAMES)
MAX_BATCHES = max(6000, NUM_CLASSES * 2000)  # Pelo menos 6000 ou classes*2000

def run_command(cmd, verbose=True):
    """Executa um comando no shell e retorna o resultado."""
    if verbose:
        print(f"Executando: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if verbose:
            print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar comando: {e}")
        print(f"Saída de erro: {e.stderr}")
        return None

def check_gpu():
    """Verifica se há GPU NVIDIA e retorna informações."""
    if platform.system() == 'Windows':
        cmd = 'nvidia-smi'
    else:
        cmd = 'nvidia-smi --query-gpu=name --format=csv,noheader'
    
    try:
        result = run_command(cmd, verbose=False)
        if result:
            print("GPU encontrada:")
            print(result)
            return True
        else:
            print("GPU NVIDIA não encontrada ou nvidia-smi não está instalado.")
            return False
    except:
        print("Erro ao verificar GPU.")
        return False

def get_gpu_arch():
    """Obtém o valor ARCH para a GPU."""
    # Arquiteturas comuns
    arch_types = {
        "Tesla V100": "-gencode arch=compute_70,code=[sm_70,compute_70]",
        "Tesla K80": "-gencode arch=compute_37,code=sm_37",
        "Tesla T4": "-gencode arch=compute_75,code=[sm_75,compute_75]",
        "GTX 1080": "-gencode arch=compute_61,code=sm_61",
        "GTX 1070": "-gencode arch=compute_61,code=sm_61",
        "GTX 1060": "-gencode arch=compute_61,code=sm_61",
        "RTX 2080": "-gencode arch=compute_75,code=[sm_75,compute_75]",
        "RTX 3080": "-gencode arch=compute_86,code=[sm_86,compute_86]",
        "RTX 3090": "-gencode arch=compute_86,code=[sm_86,compute_86]",
    }
    
    # Caso não encontre uma correspondência exata, usar um valor padrão
    return "-gencode arch=compute_61,code=sm_61"  # Valor padrão para GPUs modernas

def download_file(url, destination):
    """Baixa um arquivo da internet usando Python nativo."""
    print(f"Baixando {url} para {destination}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(destination, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Download concluído: {destination}")
        return True
    except Exception as e:
        print(f"Erro ao baixar arquivo: {e}")
        return False

def setup_darknet():
    """Configura o ambiente Darknet."""
    # Verificar se o diretório já existe
    if os.path.exists(DARKNET_DIR):
        print(f"Diretório {DARKNET_DIR} já existe. Deseja remover? (s/n)")
        if input().lower() == 's':
            shutil.rmtree(DARKNET_DIR)
        else:
            print("Usando diretório existente.")
            return
    
    # Criar diretório para o Darknet
    os.makedirs(DARKNET_DIR, exist_ok=True)
    
    # No Windows, em vez de clonar e compilar, baixamos a versão pré-compilada para Windows
    darknet_zip = os.path.join(DARKNET_DIR, "darknet.zip")
    
    # Baixando Darknet pré-compilado para Windows (AlexeyAB's versão)
    print("Baixando Darknet pré-compilado para Windows...")
    download_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/darknet_yolo_v4_pre.zip"
    if download_file(download_url, darknet_zip):
        # Extrair o arquivo zip
        with zipfile.ZipFile(darknet_zip, 'r') as zip_ref:
            zip_ref.extractall(DARKNET_DIR)
        
        # Limpar arquivos temporários
        os.remove(darknet_zip)
        print("Darknet extraído com sucesso!")
    else:
        print("Erro ao baixar Darknet. Por favor, baixe manualmente de:")
        print("https://github.com/AlexeyAB/darknet/releases")
        print(f"E extraia para: {DARKNET_DIR}")
        sys.exit(1)
    
    # Criar diretórios necessários
    os.makedirs(os.path.join(DARKNET_DIR, "cfg"), exist_ok=True)
    os.makedirs(os.path.join(DARKNET_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(DARKNET_DIR, "backup"), exist_ok=True)

def download_weights():
    """Baixa os pesos pré-treinados do YOLOv4-Tiny."""
    # Baixar os pesos do YOLOv4-Tiny
    yolov4_tiny_weights = os.path.join(DARKNET_DIR, "yolov4-tiny.weights")
    if not os.path.exists(yolov4_tiny_weights):
        download_file(
            "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
            yolov4_tiny_weights
        )
    
    # Baixar os pesos de convolução pré-treinados
    yolov4_tiny_conv = os.path.join(DARKNET_DIR, "yolov4-tiny.conv.29")
    if not os.path.exists(yolov4_tiny_conv):
        download_file(
            "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29",
            yolov4_tiny_conv
        )
    
    # Baixar o arquivo de configuração yolov4-tiny
    cfg_dir = os.path.join(DARKNET_DIR, "cfg")
    yolov4_tiny_cfg = os.path.join(cfg_dir, "yolov4-tiny.cfg")
    if not os.path.exists(yolov4_tiny_cfg):
        download_file(
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
            yolov4_tiny_cfg
        )

def prepare_dataset():
    """Prepara os arquivos de dataset para treinamento."""
    # Criar diretórios necessários
    obj_dir = os.path.join(DARKNET_DIR, "data", "obj")
    os.makedirs(obj_dir, exist_ok=True)
    
    # Criar arquivo de nomes (obj.names)
    obj_names_path = os.path.join(DARKNET_DIR, "data", "obj.names")
    with open(obj_names_path, 'w') as f:
        for class_name in CLASS_NAMES:
            f.write(f"{class_name}\n")
    
    # Copiar imagens e labels para data/obj
    print("Copiando imagens e labels de treinamento...")
    for img_file in os.listdir(TRAIN_DIR):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            shutil.copy(os.path.join(TRAIN_DIR, img_file), obj_dir)
            # Também copiar o arquivo .txt correspondente se existir
            txt_file = os.path.splitext(img_file)[0] + '.txt'
            if os.path.exists(os.path.join(TRAIN_DIR, txt_file)):
                shutil.copy(os.path.join(TRAIN_DIR, txt_file), obj_dir)
    
    print("Copiando imagens e labels de validação...")
    for img_file in os.listdir(VALID_DIR):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            shutil.copy(os.path.join(VALID_DIR, img_file), obj_dir)
            # Também copiar o arquivo .txt correspondente se existir
            txt_file = os.path.splitext(img_file)[0] + '.txt'
            if os.path.exists(os.path.join(VALID_DIR, txt_file)):
                shutil.copy(os.path.join(VALID_DIR, txt_file), obj_dir)
    
    # Criar arquivo obj.data
    obj_data_path = os.path.join(DARKNET_DIR, "data", "obj.data")
    with open(obj_data_path, 'w') as f:
        f.write(f"classes = {NUM_CLASSES}\n")
        f.write("train = data/train.txt\n")
        f.write("valid = data/valid.txt\n")
        f.write("names = data/obj.names\n")
        f.write("backup = backup/\n")
    
    # Criar arquivo train.txt
    print("Criando arquivos de lista de imagens...")
    train_txt_path = os.path.join(DARKNET_DIR, "data", "train.txt")
    with open(train_txt_path, 'w') as f:
        for img_file in os.listdir(TRAIN_DIR):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                f.write(f"data/obj/{img_file}\n")
    
    # Criar arquivo valid.txt
    valid_txt_path = os.path.join(DARKNET_DIR, "data", "valid.txt")
    with open(valid_txt_path, 'w') as f:
        for img_file in os.listdir(VALID_DIR):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                f.write(f"data/obj/{img_file}\n")
    
    print("Preparação de dataset concluída!")

def create_config():
    """Cria o arquivo de configuração para YOLOv4-Tiny personalizado."""
    steps1 = int(0.8 * MAX_BATCHES)
    steps2 = int(0.9 * MAX_BATCHES)
    steps_str = f"{steps1},{steps2}"
    num_filters = (NUM_CLASSES + 5) * 3
    
    # Vamos usar o arquivo yolov4-tiny.cfg baixado como base
    base_cfg_path = os.path.join(DARKNET_DIR, "cfg", "yolov4-tiny.cfg")
    if not os.path.exists(base_cfg_path):
        print(f"Arquivo base {base_cfg_path} não encontrado.")
        print("Tentando baixar yolov4-tiny.cfg...")
        download_file(
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
            base_cfg_path
        )
    
    # Ler o arquivo base
    try:
        with open(base_cfg_path, 'r') as f:
            cfg_content = f.read()
    except FileNotFoundError:
        print(f"Erro crítico: Não foi possível encontrar {base_cfg_path}")
        print("Verifique se você pode baixar manualmente de:")
        print("https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg")
        sys.exit(1)
    
    # Ajustar para nosso número de classes
    cfg_content = cfg_content.replace("classes=80", f"classes={NUM_CLASSES}")
    
    # Ajustar número de iterações (max_batches)
    cfg_content = cfg_content.replace("max_batches = 500500", f"max_batches = {MAX_BATCHES}")
    
    # Ajustar steps
    cfg_content = cfg_content.replace("steps=400000,450000", f"steps={steps_str}")
    
    # Ajustar filtros nas camadas YOLO
    # Encontrar e substituir as linhas de filtro antes das camadas [yolo]
    lines = cfg_content.split('\n')
    for i in range(len(lines)):
        if '[yolo]' in lines[i]:
            # Encontrar a linha de filtros imediatamente antes da camada yolo
            j = i - 1
            while j >= 0 and 'filters=' not in lines[j]:
                j -= 1
            if j >= 0:
                lines[j] = f"filters={num_filters}"
    
    # Reconstruir o conteúdo
    cfg_content = '\n'.join(lines)
    
    # Salvar o arquivo de configuração personalizado
    custom_cfg_path = os.path.join(DARKNET_DIR, "cfg", "custom-yolov4-tiny-detector.cfg")
    with open(custom_cfg_path, 'w') as f:
        f.write(cfg_content)
    
    print(f"Arquivo de configuração criado: {custom_cfg_path}")
    print(f"Número de classes: {NUM_CLASSES}")
    print(f"Max batches: {MAX_BATCHES}")
    print(f"Steps: {steps_str}")
    print(f"Número de filtros: {num_filters}")

def train_model():
    """Inicia o treinamento do modelo."""
    # Criar diretório de backup se não existir
    backup_dir = os.path.join(DARKNET_DIR, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Montar o comando de treinamento - adaptado para Windows
    darknet_exe = os.path.join(DARKNET_DIR, "build", "darknet", "x64", "darknet.exe")
    obj_data = os.path.join(DARKNET_DIR, "data", "obj.data")
    custom_cfg = os.path.join(DARKNET_DIR, "cfg", "custom-yolov4-tiny-detector.cfg")
    conv_weights = os.path.join(DARKNET_DIR, "yolov4-tiny.conv.29")
    
    # Verificar se os arquivos existem
    if not os.path.exists(darknet_exe):
        print(f"Erro: {darknet_exe} não encontrado.")
        print("Verifique se o Darknet foi baixado e extraído corretamente.")
        return
    
    # Mudar para o diretório Darknet
    original_dir = os.getcwd()
    os.chdir(DARKNET_DIR)
    
    # Executar o treinamento
    print("Iniciando treinamento...")
    train_cmd = f"darknet.exe detector train data/obj.data cfg/custom-yolov4-tiny-detector.cfg yolov4-tiny.conv.29 -dont_show -map"
    run_command(train_cmd)
    
    # Voltar para o diretório original
    os.chdir(original_dir)

def test_model():
    """Testa o modelo treinado em uma imagem."""
    # Copiar o arquivo obj.names para coco.names (hack para o detector)
    obj_names = os.path.join(DARKNET_DIR, "data", "obj.names")
    coco_names = os.path.join(DARKNET_DIR, "data", "coco.names")
    if os.path.exists(obj_names):
        shutil.copy(obj_names, coco_names)
    
    # Encontrar imagens de teste
    test_dir = os.path.join(DARKNET_DIR, "test")
    os.makedirs(test_dir, exist_ok=True)
    
    # Copiar algumas imagens de teste do diretório de teste
    print("Copiando imagens de teste...")
    for img_file in os.listdir(TEST_DIR):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            shutil.copy(os.path.join(TEST_DIR, img_file), test_dir)
    
    # Selecionar uma imagem aleatoriamente
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if test_images:
        img_path = os.path.join("test", random.choice(test_images))
        
        # Executar detecção
        darknet_exe = os.path.join(DARKNET_DIR, "build", "darknet", "x64", "darknet.exe")
        weights_path = os.path.join(DARKNET_DIR, "backup", "custom-yolov4-tiny-detector_best.weights")
        custom_cfg = os.path.join(DARKNET_DIR, "cfg", "custom-yolov4-tiny-detector.cfg")
        
        if os.path.exists(weights_path) and os.path.exists(darknet_exe):
            # Mudar para o diretório Darknet
            original_dir = os.getcwd()
            os.chdir(DARKNET_DIR)
            
            # Executar detecção
            detect_cmd = f"darknet.exe detect cfg/custom-yolov4-tiny-detector.cfg backup/custom-yolov4-tiny-detector_best.weights {img_path}"
            run_command(detect_cmd)
            
            print(f"Previsões salvas em: {os.path.join(DARKNET_DIR, 'predictions.jpg')}")
            
            # Voltar para o diretório original
            os.chdir(original_dir)
        else:
            print(f"Erro: Arquivo de pesos {weights_path} ou darknet.exe não encontrado.")
    else:
        print("Nenhuma imagem de teste encontrada.")

def check_directories():
    """Verifica se os diretórios de dataset existem."""
    dirs_to_check = [DATASET_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR]
    missing_dirs = []
    
    for d in dirs_to_check:
        if not os.path.exists(d):
            missing_dirs.append(d)
    
    if missing_dirs:
        print("AVISO: Os seguintes diretórios não existem:")
        for d in missing_dirs:
            print(f"  - {d}")
        
        print("\nPor favor, certifique-se de que seu dataset está organizado da seguinte forma:")
        print(f"{DATASET_DIR}/")
        print("  ├── train/   (imagens e arquivos .txt)")
        print("  ├── valid/   (imagens e arquivos .txt)")
        print("  └── test/    (imagens para teste)")
        
        create = input("\nDeseja criar esses diretórios agora? (s/n): ")
        if create.lower() == 's':
            for d in missing_dirs:
                os.makedirs(d, exist_ok=True)
            print("Diretórios criados. Por favor, adicione suas imagens e arquivos de anotação.")
        else:
            print("Por favor, organize seu dataset e execute novamente.")
            sys.exit(1)

def main():
    """Função principal."""
    print("==== Treinamento de YOLOv4-Tiny com Darknet ====")
    
    # Verificar diretórios de dataset
    check_directories()
    
    # Verificar GPU
    has_gpu = check_gpu()
    if not has_gpu:
        print("Aviso: GPU não detectada. O treinamento será muito mais lento na CPU.")
        proceed = input("Deseja continuar mesmo assim? (s/n): ")
        if proceed.lower() != 's':
            print("Treinamento cancelado.")
            return
    
    # Configurar Darknet
    print("\n1. Configurando ambiente Darknet...")
    setup_darknet()
    
    # Baixar pesos pré-treinados
    print("\n2. Baixando pesos pré-treinados...")
    download_weights()
    
    # Preparar dataset
    print("\n3. Preparando dataset...")
    prepare_dataset()
    
    # Criar arquivo de configuração
    print("\n4. Criando arquivo de configuração...")
    create_config()
    
    # Treinamento
    print("\n5. Iniciando treinamento...")
    print("Isto pode levar várias horas dependendo da sua GPU.")
    train_model()
    
    # Teste
    print("\n6. Testando modelo...")
    test_model()
    
    print("\nTreinamento concluído! Os pesos estão salvos em:")
    print(f"{os.path.join(DARKNET_DIR, 'backup')}")

if __name__ == "__main__":
    main()