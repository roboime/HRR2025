#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nó ROS 2 para o detector YOEO.

Este script inicia o nó detector YOEO que recebe imagens da câmera,
processa-as com o modelo YOEO e publica os resultados de detecção
e segmentação.
"""

import rclpy
from perception.yoeo.yoeo_detector import YOEODetector

def main(args=None):
    """Função principal para iniciar o nó detector YOEO."""
    # Inicializar ROS
    rclpy.init(args=args)
    
    try:
        # Criar nó detector
        node = YOEODetector()
        
        # Iniciar processamento
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Erro no nó detector YOEO: {e}")
    finally:
        # Limpar recursos
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 