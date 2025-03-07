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
import subprocess

class IMX219CameraNode(Node):
    """
    Nó ROS 2 otimizado para a câmera IMX219 na Jetson Nano.
    Suporta todas as capacidades da câmera, incluindo:
    - Resolução máxima de 3280x2464
    - HDR
    - Ajuste de exposição e ganho
    - Processamento CUDA
    """

    def __init__(self):
        super().__init__('imx219_camera_node')
        
        # Parâmetros da câmera
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_mode', 2),  # 0=3280x2464, 1=1920x1080, 2=1280x720
                ('camera_fps', 120),  # Ajustado para máximo FPS no modo HD
                ('flip_method', 0),
                ('exposure_time', 13333),  # em microssegundos
                ('gain', 1.0),
                ('awb_mode', 1),  # 0=off, 1=auto
                ('brightness', 0),
                ('saturation', 1.0),
                ('enable_cuda', True),
                ('enable_hdr', False),
                ('enable_display', False),
                ('enable_isp', False),
                ('enable_noise_reduction', False),
                ('enable_edge_enhancement', False)
            ]
        )
        
        # Configurações baseadas no modo da câmera
        self.camera_modes = {
            0: (3280, 2464, 21),  # Máxima resolução
            1: (1920, 1080, 60),  # Full HD
            2: (1280, 720, 120)   # HD
        }
        
        # Obter modo da câmera
        self.camera_mode = self.get_parameter('camera_mode').value
        self.width, self.height, self.max_fps = self.camera_modes[self.camera_mode]
        
        # Ajustar FPS baseado no modo
        requested_fps = min(self.get_parameter('camera_fps').value, self.max_fps)
        self.set_parameter(rclpy.Parameter('camera_fps', value=requested_fps))
        
        # Publishers
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)
        
        # Bridge
        self.bridge = CvBridge()
        
        # Inicializar câmera
        self.init_camera()
        
        # Thread de captura
        self.is_running = True
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Monitoramento de recursos
        self.create_timer(5.0, self.monitor_resources)
        
        self.get_logger().info(f'IMX219 Camera Node iniciado - Modo: {self.camera_mode} ({self.width}x{self.height} @ {requested_fps}fps)')

    def init_camera(self):
        """Inicializa a câmera com configurações otimizadas."""
        # Pipeline GStreamer otimizado para IMX219
        pipeline = (
            f"nvarguscamerasrc sensor-id=0 "
            f"exposuretimerange='{self.get_parameter('exposure_time').value} {self.get_parameter('exposure_time').value}' "
            f"gainrange='{self.get_parameter('gain').value} {self.get_parameter('gain').value}' "
            f"wbmode={self.get_parameter('awb_mode').value} "
            f"tnr-mode=2 ee-mode=2 "  # Redução de ruído temporal e aprimoramento de bordas
            f"saturation={self.get_parameter('saturation').value} "
            f"brightness={self.get_parameter('brightness').value} "
            f"ispdigitalgainrange='1 2' "  # Otimização do ganho digital
            f"! video/x-raw(memory:NVMM), "
            f"width=(int){self.width}, height=(int){self.height}, "
            f"format=(string)NV12, framerate=(fraction){self.get_parameter('camera_fps').value}/1 ! "
        )
        
        # Adicionar processamento ISP se habilitado
        if self.get_parameter('enable_isp').value:
            pipeline += (
                f"nvvidconv "
                f"interpolation-method=5 "  # Melhor qualidade de interpolação
                f"flip-method={self.get_parameter('flip_method').value} "
            )
        else:
            pipeline += f"nvvidconv flip-method={self.get_parameter('flip_method').value} "
        
        # Adicionar HDR se habilitado
        if self.get_parameter('enable_hdr').value:
            pipeline += "post-processing=1 "
        
        # Adicionar redução de ruído se habilitada
        if self.get_parameter('enable_noise_reduction').value:
            pipeline += "noise-reduction=1 "
        
        # Adicionar aprimoramento de bordas se habilitado
        if self.get_parameter('enable_edge_enhancement').value:
            pipeline += "edge-enhancement=1 "
        
        # Completar pipeline
        pipeline += (
            f"! video/x-raw, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! "
            f"appsink max-buffers=1 drop=True"
        )
        
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            self.get_logger().error('Falha ao abrir câmera!')
            raise RuntimeError('Falha ao abrir câmera!')
            
        self.get_logger().info('Câmera inicializada com sucesso')

    def capture_loop(self):
        """Loop principal de captura com processamento CUDA."""
        cuda_enabled = self.get_parameter('enable_cuda').value
        
        if cuda_enabled:
            # Inicializar contexto CUDA
            self.cuda_stream = cv2.cuda_Stream()
            self.cuda_upload = cv2.cuda_GpuMat()
            self.cuda_color = cv2.cuda_GpuMat()
            self.cuda_download = cv2.cuda_GpuMat()
        
        while self.is_running and rclpy.ok():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().warn('Falha ao capturar frame')
                    time.sleep(0.1)
                    continue
                
                if cuda_enabled:
                    # Processamento CUDA
                    self.cuda_upload.upload(frame)
                    
                    # Redução de ruído e aprimoramento
                    cv2.cuda.fastNlMeansDenoisingColored(
                        self.cuda_upload,
                        None,
                        h_luminance=3,
                        photo_render=1,
                        stream=self.cuda_stream
                    )
                    
                    # Ajuste de cor e contraste
                    cv2.cuda.gammaCorrection(
                        self.cuda_upload,
                        self.cuda_color,
                        stream=self.cuda_stream
                    )
                    
                    # Aprimoramento de bordas
                    if self.get_parameter('enable_edge_enhancement').value:
                        cv2.cuda.createSobelFilter(
                            cv2.CV_8UC3,
                            cv2.CV_16S,
                            1, 0
                        ).apply(
                            self.cuda_color,
                            self.cuda_download,
                            self.cuda_stream
                        )
                    
                    # Download do resultado
                    frame = self.cuda_color.download()
                
                # Publicar imagem
                timestamp = self.get_clock().now().to_msg()
                img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                img_msg.header.stamp = timestamp
                img_msg.header.frame_id = "camera_optical_frame"
                self.image_pub.publish(img_msg)
                
                # Publicar info da câmera
                self.publish_camera_info(img_msg.header)
                
                # Exibir se necessário
                if self.get_parameter('enable_display').value:
                    cv2.imshow('IMX219 Camera', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            except CvBridgeError as e:
                self.get_logger().error(f'Erro no CvBridge: {e}')
            except Exception as e:
                self.get_logger().error(f'Erro na captura: {e}')
                
            # Controle de taxa
            time.sleep(1.0 / self.get_parameter('camera_fps').value)

    def publish_camera_info(self, header):
        """Publica informações da câmera."""
        info_msg = CameraInfo()
        info_msg.header = header
        info_msg.height = self.height
        info_msg.width = self.width
        
        # Matriz K para IMX219
        focal_length = self.width * 0.8  # Aproximação baseada no FOV de 130°
        center_x = self.width / 2
        center_y = self.height / 2
        
        info_msg.k = [
            focal_length, 0, center_x,
            0, focal_length, center_y,
            0, 0, 1
        ]
        
        # Sem distorção por enquanto
        info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.camera_info_pub.publish(info_msg)

    def monitor_resources(self):
        """Monitora recursos do sistema."""
        try:
            # Temperatura
            temp = float(subprocess.check_output(['cat', '/sys/class/thermal/thermal_zone0/temp']).decode()) / 1000.0
            
            # Uso de GPU via tegrastats
            tegrastats = subprocess.check_output(['tegrastats', '--interval', '1', '--count', '1']).decode()
            
            self.get_logger().info(f'Temperatura: {temp:.1f}°C | Tegrastats: {tegrastats}')
            
            # Alertar se temperatura muito alta
            if temp > 80.0:
                self.get_logger().warn('Temperatura muito alta! Considere reduzir FPS ou resolução')
                
        except Exception as e:
            self.get_logger().error(f'Erro ao monitorar recursos: {e}')

    def destroy_node(self):
        """Limpa recursos ao encerrar."""
        self.is_running = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            
        if self.get_parameter('enable_display').value:
            cv2.destroyAllWindows()
            
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = IMX219CameraNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 