#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nó ROS 2 para câmera USB Logitech C930 - Sistema de Percepção YOLOv8 (7 classes)
Otimizado para Jetson Orin Nano Super com suporte GPU
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import threading
import time
import os
import yaml
import traceback
from typing import Optional, Tuple

class USB_C930_CameraNode(Node):
    """
    Nó ROS 2 para câmera USB Logitech C930 com otimizações para Jetson Orin Nano Super
    Suporte completo para aceleração GPU e configurações avançadas da C930
    """

    def __init__(self):
        super().__init__('usb_camera_node')
        self.get_logger().info(f"🎥 Inicializando Nó USB Camera C930 - OpenCV: {cv2.__version__}")
        
        # Bridge para conversão ROS<->OpenCV
        self.bridge = CvBridge()
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Threading e controle
        self.capture_thread: Optional[threading.Thread] = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.current_frame = None
        
        # Estatísticas de performance
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
        # Declarar parâmetros específicos da C930
        self._declare_parameters()
        
        # Configurar câmera
        self._setup_camera()
        
        # Publishers
        self._setup_publishers()
        
        # Inicializar captura
        self._start_capture()
        
        self.get_logger().info("✅ Nó USB Camera C930 inicializado com sucesso!")

    def _declare_parameters(self):
        """Declara parâmetros específicos da Logitech C930"""
        self.declare_parameters(
            namespace='',
            parameters=[
                # Configurações básicas
                ('device_path', '/dev/video0'),
                ('camera_name', 'logitech_c930'),
                ('display', False),
                ('enable_cuda', True),
                
                # Resolução e FPS (C930 otimizada)
                ('width', 1280),                    # C930 sweet spot
                ('height', 720),                    # 720p para performance
                ('fps', 30.0),                      # 30fps estável
                
                # Configurações avançadas C930
                ('auto_exposure', True),            # Auto exposição (recomendado)
                ('exposure', 156),                  # Exposição manual (se auto_exposure=False)
                ('brightness', 128),                # Brilho (0-255)
                ('contrast', 128),                  # Contraste (0-255)
                ('saturation', 128),                # Saturação (0-255)
                ('sharpness', 128),                 # Nitidez (0-255)
                ('gamma', 100),                     # Gamma (72-500)
                ('white_balance_auto', True),       # Auto white balance
                ('white_balance_temp', 4000),       # Temperatura de cor (se auto=False)
                ('gain', 64),                       # Ganho (0-255)
                ('power_line_frequency', 2),        # 0=disabled, 1=50Hz, 2=60Hz
                ('backlight_compensation', 0),      # Compensação de luz de fundo
                ('auto_focus', True),               # Auto foco (recomendado para C930)
                ('focus', 0),                       # Foco manual (se auto_focus=False)
                ('zoom', 100),                      # Zoom (100-400, 100=sem zoom)
                ('pan', 0),                         # Pan (-36000 to 36000)
                ('tilt', 0),                        # Tilt (-36000 to 36000)
                
                # Configurações de qualidade
                ('fourcc', 'MJPG'),                 # Codec (MJPG recomendado para C930)
                ('buffer_size', 1),                 # Buffer mínimo para latência baixa
                
                # Calibração
                ('calibration_file', ''),
                
                # Debug e monitoring
                ('log_fps', True),
                ('log_interval', 5.0),
            ]
        )

    def _setup_camera(self):
        """Configura a câmera USB C930 com parâmetros otimizados"""
        device_path = self.get_parameter('device_path').value
        
        try:
            # Inicializar VideoCapture
            self.cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"❌ Não foi possível abrir câmera em {device_path}")
            
            # Configurar FOURCC (codec)
            fourcc_str = self.get_parameter('fourcc').value
            if fourcc_str == 'MJPG':
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            elif fourcc_str == 'YUYV':
                fourcc = cv2.VideoWriter_fourcc(*'YUYV')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Default
            
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            
            # Configurar resolução e FPS
            width = self.get_parameter('width').value
            height = self.get_parameter('height').value
            fps = self.get_parameter('fps').value
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Buffer size para baixa latência
            buffer_size = self.get_parameter('buffer_size').value
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
            
            # Configurações específicas da C930
            self._configure_c930_settings()
            
            # Verificar configurações aplicadas
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.get_logger().info(f"📷 C930 Configurada: {actual_width}x{actual_height}@{actual_fps}fps")
            
            # Validar se conseguiu configurar adequadamente
            if actual_width != width or actual_height != height:
                self.get_logger().warn(f"⚠️  Resolução solicitada {width}x{height} != real {actual_width}x{actual_height}")
                
        except Exception as e:
            self.get_logger().error(f"❌ Erro ao configurar câmera C930: {e}")
            raise

    def _configure_c930_settings(self):
        """Configura parâmetros avançados específicos da Logitech C930"""
        try:
            # Auto Exposure
            auto_exposure = self.get_parameter('auto_exposure').value
            if auto_exposure:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = auto mode
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode
                exposure = self.get_parameter('exposure').value
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            
            # Controles de imagem
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.get_parameter('brightness').value)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.get_parameter('contrast').value)
            self.cap.set(cv2.CAP_PROP_SATURATION, self.get_parameter('saturation').value)
            self.cap.set(cv2.CAP_PROP_SHARPNESS, self.get_parameter('sharpness').value)
            self.cap.set(cv2.CAP_PROP_GAMMA, self.get_parameter('gamma').value)
            self.cap.set(cv2.CAP_PROP_GAIN, self.get_parameter('gain').value)
            
            # White Balance
            white_balance_auto = self.get_parameter('white_balance_auto').value
            if white_balance_auto:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                wb_temp = self.get_parameter('white_balance_temp').value
                self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temp)
            
            # Auto Focus (C930 feature)
            auto_focus = self.get_parameter('auto_focus').value
            if auto_focus:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            else:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                focus = self.get_parameter('focus').value
                self.cap.set(cv2.CAP_PROP_FOCUS, focus)
            
            # Zoom, Pan, Tilt (C930 mechanical features)
            self.cap.set(cv2.CAP_PROP_ZOOM, self.get_parameter('zoom').value)
            self.cap.set(cv2.CAP_PROP_PAN, self.get_parameter('pan').value)
            self.cap.set(cv2.CAP_PROP_TILT, self.get_parameter('tilt').value)
            
            # Power line frequency (para evitar flicker)
            power_freq = self.get_parameter('power_line_frequency').value
            self.cap.set(cv2.CAP_PROP_FPS, power_freq)  # Pode variar por driver
            
            # Backlight compensation
            backlight = self.get_parameter('backlight_compensation').value
            self.cap.set(cv2.CAP_PROP_BACKLIGHT, backlight)
            
            self.get_logger().info("✅ Configurações avançadas da C930 aplicadas")
            
        except Exception as e:
            self.get_logger().warn(f"⚠️  Erro ao aplicar algumas configurações da C930: {e}")

    def _setup_publishers(self):
        """Configura os publishers ROS2"""
        # Publisher da imagem
        self.image_pub = self.create_publisher(
            Image, 
            '/camera/image_raw', 
            10
        )
        
        # Publisher das informações da câmera
        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            '/camera/camera_info',
            10
        )
        
        # Timer para publicação das info da câmera
        self.camera_info_timer = self.create_timer(0.1, self._publish_camera_info)

    def _start_capture(self):
        """Inicia a thread de captura"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        """Loop principal de captura de frames"""
        while self.running and rclpy.ok():
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret and frame is not None:
                        # Aplicar aceleração GPU se habilitada
                        if self.get_parameter('enable_cuda').value:
                            frame = self._apply_gpu_processing(frame)
                        
                        # Atualizar frame atual thread-safe
                        with self.frame_lock:
                            self.current_frame = frame.copy()
                        
                        # Publicar frame
                        self._publish_frame(frame)
                        
                        # Atualizar estatísticas
                        self._update_fps_stats()
                        
                    else:
                        self.get_logger().warn("⚠️  Falha na captura de frame da C930")
                        time.sleep(0.01)
                else:
                    self.get_logger().error("❌ Câmera C930 não está disponível")
                    break
                    
            except Exception as e:
                self.get_logger().error(f"❌ Erro no loop de captura: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    def _apply_gpu_processing(self, frame):
        """Aplica processamento GPU se disponível"""
        try:
            # Converter para GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Operações GPU básicas (exemplo: blur para reduzir ruído)
            gpu_blurred = cv2.cuda.bilateralFilter(gpu_frame, -1, 50, 50)
            
            # Download de volta para CPU
            result = gpu_blurred.download()
            return result
            
        except Exception as e:
            # Se GPU falhar, retorna frame original
            return frame

    def _publish_frame(self, frame):
        """Publica o frame via ROS2"""
        try:
            # Converter frame para mensagem ROS
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera_link'
            
            # Publicar
            self.image_pub.publish(img_msg)
            
        except CvBridgeError as e:
            self.get_logger().error(f"❌ Erro na conversão CV->ROS: {e}")

    def _publish_camera_info(self):
        """Publica informações da câmera"""
        if self.cap is None:
            return
            
        try:
            camera_info = CameraInfo()
            camera_info.header.stamp = self.get_clock().now().to_msg()
            camera_info.header.frame_id = 'camera_link'
            
            # Dimensões
            camera_info.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            camera_info.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Matriz de calibração básica (substituir por calibração real se disponível)
            fx = float(camera_info.width)
            fy = float(camera_info.width)  # placeholder simples; idealmente usar fx/fy reais
            cx = float(camera_info.width) / 2.0
            cy = float(camera_info.height) / 2.0

            camera_info.k = [
                fx, 0.0, cx,
                0.0, fy, cy,
                0.0, 0.0, 1.0
            ]
            
            # Modelo de distorção
            camera_info.distortion_model = "plumb_bob"
            camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            
            # Publicar
            self.camera_info_pub.publish(camera_info)
            
        except Exception as e:
            self.get_logger().error(f"❌ Erro ao publicar camera_info: {e}")

    def _update_fps_stats(self):
        """Atualiza estatísticas de FPS"""
        self.frame_count += 1
        
        if self.get_parameter('log_fps').value:
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            
            log_interval = self.get_parameter('log_interval').value
            if elapsed >= log_interval:
                self.fps = self.frame_count / elapsed
                self.get_logger().info(f"📊 C930 Performance: {self.fps:.1f} FPS")
                
                # Reset counters
                self.frame_count = 0
                self.last_fps_time = current_time

    def get_current_frame(self):
        """Retorna o frame atual de forma thread-safe"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def destroy_node(self):
        """Limpa recursos ao destruir o nó"""
        self.get_logger().info("🔄 Finalizando nó USB Camera C930...")
        
        # Parar captura
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Liberar câmera
        if self.cap:
            self.cap.release()
        
        super().destroy_node()
        self.get_logger().info("✅ Nó USB Camera C930 finalizado")


def main(args=None):
    """Função principal"""
    rclpy.init(args=args)
    
    try:
        camera_node = USB_C930_CameraNode()
        rclpy.spin(camera_node)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        traceback.print_exc()
    finally:
        try:
            camera_node.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main() 