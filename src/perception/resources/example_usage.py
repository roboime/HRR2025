#!/usr/bin/env python3
"""
Exemplo de uso do módulo de recursos.

Este script demonstra como usar o módulo de recursos para acessar
modelos, arquivos de calibração, imagens de teste e rótulos.
"""

import os
import sys
from pathlib import Path

# Adiciona o diretório pai ao caminho de busca do Python
sys.path.append(str(Path(__file__).parent.parent.parent))

# Importa o módulo de recursos
from perception.resources import (
    get_model_path,
    get_calibration_path,
    get_test_image_path,
    get_label_path,
    list_available_models,
    list_available_calibrations,
    list_available_test_images,
    list_available_labels,
)


def main():
    """Função principal que demonstra o uso do módulo de recursos."""
    print("Demonstração do módulo de recursos do sistema de percepção\n")

    # Lista todos os recursos disponíveis
    print("Modelos disponíveis:")
    for model in list_available_models():
        print(f"  - {model}")
    print()

    print("Arquivos de calibração disponíveis:")
    for calibration in list_available_calibrations():
        print(f"  - {calibration}")
    print()

    print("Imagens de teste disponíveis:")
    for image in list_available_test_images():
        print(f"  - {image}")
    print()

    print("Arquivos de rótulos disponíveis:")
    for label in list_available_labels():
        print(f"  - {label}")
    print()

    # Demonstra como obter caminhos para recursos específicos
    if list_available_calibrations():
        calibration_file = list_available_calibrations()[0]
        calibration_path = get_calibration_path(calibration_file)
        print(f"Caminho para o arquivo de calibração '{calibration_file}':")
        print(f"  {calibration_path}")
        print(f"  Existe: {calibration_path.exists()}")
        print()

    if list_available_labels():
        label_file = list_available_labels()[0]
        label_path = get_label_path(label_file)
        print(f"Caminho para o arquivo de rótulos '{label_file}':")
        print(f"  {label_path}")
        print(f"  Existe: {label_path.exists()}")
        print()

        # Demonstra como ler um arquivo de rótulos
        print(f"Conteúdo do arquivo de rótulos '{label_file}':")
        with open(label_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        for i, class_name in enumerate(classes):
            print(f"  {i}: {class_name}")
        print()


if __name__ == "__main__":
    main() 