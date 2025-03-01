#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import threading
import time

class JetsonCameraNode(Node):
    """
    Nó ROS 2 para a câmera CSI da Jetson Nano.
    Publica imagens e informações da câmera para o sistema de visão.
    """

    def __init__(self):
        super().__init__('jetson_camera_node')
        
        # Parâmetros da câmera
        self.declare_parameter('camera_type', 'csi')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('camera_fps', 30)
        self.declare_parameter('display_width', 640)
        self.declare_parameter('display_height', 480)
        self.declare_parameter('enable_display', True)
        
        # Obter parâmetros
        self.camera_type = self.get_parameter('camera_type').value
        self.camera_index = self.get_parameter('camera_index').value
        self.camera_width = self.get_parameter('camera_width').value
        self.camera_height = self.get_parameter('camera_height').value
        self.camera_fps = self.get_parameter('camera_fps').value
        self.display_width = self.get_parameter('display_width').value
        self.display_height = self.get_parameter('display_height').value
        self.enable_display = self.get_parameter('enable_display').value
        
        # Inicializar bridge para converter entre OpenCV e ROS
        self.bridge = CvBridge()
        
        # Publicadores
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        # Informações da câmera
        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.header.frame_id = 'camera'
        self.camera_info_msg.height = self.camera_height
        self.camera_info_msg.width = self.camera_width
        
        # Inicializar câmera
        self.init_camera()
        
        # Iniciar thread de captura
        self.is_running = True
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.get_logger().info('Jetson Camera Node iniciado')
        
    def init_camera(self):
        """Inicializa a câmera com base no tipo e parâmetros."""
        if self.camera_type == 'csi':
            # Configuração para câmera CSI da Jetson Nano
            gst_str = (
                f'nvarguscamerasrc sensor-id={self.camera_index} ! '
                f'video/x-raw(memory:NVMM), width=(int){self.camera_width}, height=(int){self.camera_height}, '
                f'format=(string)NV12, framerate=(fraction){self.camera_fps}/1 ! '
                f'nvvidconv flip-method=0 ! '
                f'video/x-raw, width=(int){self.camera_width}, height=(int){self.camera_height}, '
                f'format=(string)BGRx ! '
                f'videoconvert ! '
                f'video/x-raw, format=(string)BGR ! '
                f'appsink'
            )
            self.get_logger().info(f'Iniciando câmera CSI com pipeline: {gst_str}')
            self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        elif self.camera_type == 'usb':
            # Configuração para câmera USB
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            self.get_logger().info(f'Iniciando câmera USB com índice: {self.camera_index}')
        else:
            self.get_logger().error(f'Tipo de câmera não suportado: {self.camera_type}')
            raise ValueError(f'Tipo de câmera não suportado: {self.camera_type}')
            
        if not self.cap.isOpened():
            self.get_logger().error('Falha ao abrir a câmera')
            raise RuntimeError('Falha ao abrir a câmera')
            
        self.get_logger().info('Câmera inicializada com sucesso')
        
    def capture_loop(self):
        """Loop principal para captura e publicação de imagens."""
        while self.is_running and rclpy.ok():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().warn('Falha ao capturar frame')
                    time.sleep(0.1)
                    continue
                    
                # Publicar imagem
                timestamp = self.get_clock().now().to_msg()
                img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
                img_msg.header.stamp = timestamp
                img_msg.header.frame_id = 'camera'
                self.image_pub.publish(img_msg)
                
                # Publicar informações da câmera
                self.camera_info_msg.header.stamp = timestamp
                self.camera_info_pub.publish(self.camera_info_msg)
                
                # Exibir imagem se habilitado
                if self.enable_display:
                    display_frame = cv2.resize(frame, (self.display_width, self.display_height))
                    cv2.imshow('Jetson Camera', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            except CvBridgeError as e:
                self.get_logger().error(f'Erro no CvBridge: {e}')
            except Exception as e:
                self.get_logger().error(f'Erro na captura: {e}')
                
            # Controle de taxa de captura
            time.sleep(1.0 / self.camera_fps)
            
    def destroy_node(self):
        """Limpa recursos ao encerrar o nó."""
        self.is_running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            
        if self.enable_display:
            cv2.destroyAllWindows()
            
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = JetsonCameraNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 