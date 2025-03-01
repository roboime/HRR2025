"""
Módulo de recursos para o sistema de percepção.

Este módulo fornece acesso aos recursos necessários para o sistema de percepção,
incluindo modelos pré-treinados, arquivos de calibração, imagens de teste e rótulos.
"""

import os
from pathlib import Path

# Obtém o caminho absoluto para o diretório de recursos
RESOURCES_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Define caminhos para subdiretórios
MODELS_DIR = RESOURCES_DIR / "models"
CALIBRATION_DIR = RESOURCES_DIR / "calibration"
TEST_IMAGES_DIR = RESOURCES_DIR / "test_images"
LABELS_DIR = RESOURCES_DIR / "labels"

def get_model_path(model_name):
    """
    Retorna o caminho para um modelo específico.
    
    Args:
        model_name (str): Nome do arquivo do modelo.
        
    Returns:
        Path: Caminho completo para o arquivo do modelo.
    """
    return MODELS_DIR / model_name

def get_calibration_path(calibration_file):
    """
    Retorna o caminho para um arquivo de calibração específico.
    
    Args:
        calibration_file (str): Nome do arquivo de calibração.
        
    Returns:
        Path: Caminho completo para o arquivo de calibração.
    """
    return CALIBRATION_DIR / calibration_file

def get_test_image_path(image_name):
    """
    Retorna o caminho para uma imagem de teste específica.
    
    Args:
        image_name (str): Nome do arquivo de imagem.
        
    Returns:
        Path: Caminho completo para o arquivo de imagem.
    """
    return TEST_IMAGES_DIR / image_name

def get_label_path(label_file):
    """
    Retorna o caminho para um arquivo de rótulos específico.
    
    Args:
        label_file (str): Nome do arquivo de rótulos.
        
    Returns:
        Path: Caminho completo para o arquivo de rótulos.
    """
    return LABELS_DIR / label_file

def list_available_models():
    """
    Lista todos os modelos disponíveis no diretório de modelos.
    
    Returns:
        list: Lista de nomes de arquivos de modelos disponíveis.
    """
    return [f.name for f in MODELS_DIR.glob("*") if f.is_file()]

def list_available_calibrations():
    """
    Lista todos os arquivos de calibração disponíveis.
    
    Returns:
        list: Lista de nomes de arquivos de calibração disponíveis.
    """
    return [f.name for f in CALIBRATION_DIR.glob("*") if f.is_file()]

def list_available_test_images():
    """
    Lista todas as imagens de teste disponíveis.
    
    Returns:
        list: Lista de nomes de arquivos de imagens de teste disponíveis.
    """
    return [f.name for f in TEST_IMAGES_DIR.glob("*") if f.is_file()]

def list_available_labels():
    """
    Lista todos os arquivos de rótulos disponíveis.
    
    Returns:
        list: Lista de nomes de arquivos de rótulos disponíveis.
    """
    return [f.name for f in LABELS_DIR.glob("*") if f.is_file()] 