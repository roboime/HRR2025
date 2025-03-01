#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nó ROS 2 para visualização dos resultados do detector YOEO.

Este script inicia um nó que se inscreve nos tópicos de detecção e segmentação
do detector YOEO e exibe os resultados em uma janela de visualização.
"""

import os
import sys
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header

# Adicionar diretório do pacote ao PYTHONPATH
package_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(package_dir)


class YOEOVisualizer(Node):
    """
    Nó para visualização dos resultados do detector YOEO.
    
    Este nó se inscreve nos tópicos de detecção, segmentação e imagem de debug
    do detector YOEO e exibe os resultados em uma janela de visualização.
    """
    
    def __init__(self):
        """Inicializa o nó visualizador."""
        super().__init__('yoeo_visualizer')
        
        # Declarar parâmetros
        self.declare_parameter('detection_topic', '/yoeo/detections')
        self.declare_parameter('segmentation_topic', '/yoeo/segmentation')
        self.declare_parameter('debug_image_topic', '/yoeo/debug_image')
        self.declare_parameter('window_name', 'YOEO Detector')
        
        # Obter parâmetros
        self.detection_topic = self.get_parameter('detection_topic').value
        self.segmentation_topic = self.get_parameter('segmentation_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.window_name = self.get_parameter('window_name').value
        
        # Configurar QoS para mensagens de imagem
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Criar bridge para conversão entre ROS e OpenCV
        self.bridge = CvBridge()
        
        # Criar inscrições
        self.debug_image_sub = self.create_subscription(
            Image,
            self.debug_image_topic,
            self.debug_image_callback,
            qos_profile
        )
        
        # Variáveis para armazenar dados
        self.latest_debug_image = None
        
        # Criar janela de visualização
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        # Configurar timer para atualização da visualização
        self.timer = self.create_timer(0.033, self.update_visualization)  # ~30 FPS
        
        self.get_logger().info('Nó visualizador YOEO iniciado')
    
    def debug_image_callback(self, msg):
        """
        Callback para a imagem de debug.
        
        Args:
            msg: Mensagem de imagem ROS
        """
        try:
            self.latest_debug_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Erro ao converter imagem: {e}')
    
    def update_visualization(self):
        """Atualiza a visualização com os dados mais recentes."""
        # Verificar se temos uma imagem para exibir
        if self.latest_debug_image is not None:
            # Exibir imagem
            cv2.imshow(self.window_name, self.latest_debug_image)
            cv2.waitKey(1)
    
    def destroy_node(self):
        """Limpa recursos ao destruir o nó."""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Função principal para iniciar o nó visualizador YOEO."""
    # Inicializar ROS
    rclpy.init(args=args)
    
    try:
        # Criar nó visualizador
        visualizer_node = YOEOVisualizer()
        
        # Iniciar processamento
        rclpy.spin(visualizer_node)
    except Exception as e:
        print(f"Erro no nó visualizador YOEO: {e}")
    finally:
        # Limpar recursos
        if 'visualizer_node' in locals():
            visualizer_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 