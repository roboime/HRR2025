#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nó ROS 2 para o detector YOEO.

Este script inicia o nó detector YOEO que recebe imagens da câmera,
processa-as com o modelo YOEO e publica os resultados de detecção
e segmentação.
"""

import os
import sys
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

# Adicionar diretório do pacote ao PYTHONPATH
package_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(package_dir)

# Importar módulos YOEO
from perception.src.yoeo.yoeo_detector import YOEODetector


def main(args=None):
    """Função principal para iniciar o nó detector YOEO."""
    # Inicializar ROS
    rclpy.init(args=args)
    
    try:
        # Criar nó detector
        detector_node = YOEODetector()
        
        # Iniciar processamento
        rclpy.spin(detector_node)
    except Exception as e:
        print(f"Erro no nó detector YOEO: {e}")
    finally:
        # Limpar recursos
        if 'detector_node' in locals():
            detector_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 