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
import os
import signal
import math
import atexit
import tempfile
import sys
import glob
import traceback
import yaml
import psutil

class IMX219CameraNode(Node):
    """
    Nó ROS 2 simplificado para a câmera IMX219 na Jetson Nano usando GStreamer e CUDA.
    """

    def __init__(self):
        super().__init__('jetson_camera_node')
        
        # Reduzir logging para o mínimo essencial
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        
        # Declarar parâmetros essenciais da câmera
        self.declare_parameters(
            namespace='',
            parameters=[
                ('device_id', 0),
                ('display', False),
                ('framerate', 30.0),
                ('enable_cuda', True),
                ('calibration_file', ''),
                ('flip_method', 0),
            ]
        )

        # Variáveis de estado
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.bridge = CvBridge()
        self.camera_info_msg = None
        self.camera_info_pub = None
        self.frame_count = 0
        self.last_fps_update = self.get_clock().now()
        self.current_fps = 0.0
        
        # Configuração e inicialização
        try:
            # MODO DE CÂMERA FIXO PARA MÁXIMO FPS
            # Modo 5: 1280x720 @ 120fps
            self.width = 1280
            self.height = 720
            self.camera_fps = 30.0
            self.max_fps = 120.0
            
            self.get_logger().info("INICIANDO NÓ DA CÂMERA CSI JETSON (MÍNIMO)")
            
            # Publishers
            self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
            
            # Inicializar câmera GStreamer
            if not self.init_gstreamer_camera():
                self.get_logger().fatal("FALHA CRÍTICA AO INICIALIZAR CÂMERA GSTREAMER!")
                raise RuntimeError("Não foi possível inicializar a câmera GStreamer.")
            else:
                self.get_logger().info(f"Câmera inicializada: {self.width}x{self.height}")
                
                # Carregar calibração se especificada
                self.load_camera_calibration()
                
                # Criar publisher de camera_info se calibração foi carregada
                if self.camera_info_msg:
                    self.camera_info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)
                
                # Iniciar thread de captura
                self.is_running = True
                self.capture_thread = threading.Thread(target=self.capture_loop)
                self.capture_thread.daemon = True
                self.capture_thread.start()
                
        except Exception as e:
            self.get_logger().fatal(f"FALHA NA INICIALIZAÇÃO: {str(e)}")
            traceback.print_exc()
            raise

    def init_gstreamer_camera(self):
        """Inicializa a câmera GStreamer com o pipeline CSI."""
        self.get_logger().info("Verificando ambiente e pré-requisitos...")

        # Construir pipeline GStreamer (SIMPLIFICADO AO MÁXIMO)
        pipeline_str = self._build_gstreamer_pipeline()
        if not pipeline_str:
            self.get_logger().error("Falha ao construir string do pipeline GStreamer.")
            return False

        self.get_logger().info(f"Pipeline: {pipeline_str}")

        # Tentar abrir a câmera
        try:
            self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                self.get_logger().error("FALHA: cv2.VideoCapture não conseguiu abrir o pipeline")
                self.get_logger().error("1. Câmera CSI mal conectada")
                self.get_logger().error("2. nvargus-daemon não está rodando no HOST")
                self.get_logger().error("3. Container sem acesso aos dispositivos NVIDIA")
                self.get_logger().error("4. Outro processo usando a câmera (pkill -f nvargus)")
                
                # Testar pipeline básico para diagnóstico
                basic_success = self.test_camera_pipeline(timeout=3)
                if basic_success:
                    self.get_logger().info("Teste básico PASSOU mas cv2.VideoCapture falhou")
                    self.get_logger().info("Isso sugere um problema na sintaxe do pipeline para OpenCV")
                
                return False

            # Verificar se conseguimos ler um frame
            self.get_logger().info("Tentando ler o primeiro frame...")
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.get_logger().error("Falha ao ler o primeiro frame!")
                self.cap.release()
                self.cap = None
                self.test_camera_pipeline(timeout=3)
                return False
            else:
                self.get_logger().info(f"Primeiro frame lido: {frame.shape}")
                del frame

            return True

        except Exception as e:
            self.get_logger().error(f"Exceção: {str(e)}")
            traceback.print_exc()
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def _build_gstreamer_pipeline(self):
        """Constrói um pipeline GStreamer CSI simplificado."""
        try:
            # PIPELINE ESSENCIAL - ULTRA MÍNIMO
            # Comparado com o teste bem-sucedido, com apenas o essencial
            device_id = self.get_parameter('device_id').value
            flip_method = self.get_parameter('flip_method').value
            
            # Em vez de um pipeline complexo, vamos usar o mais mínimo possível
            # Adicionamos parâmetros importantes para o appsink que podem estar faltando
            pipeline = (
                f"nvarguscamerasrc sensor-id={device_id} "
                f"! video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
                f"format=NV12, framerate=30/1 "
                f"! nvvidconv flip-method={flip_method} "
                f"! video/x-raw, format=BGRx "
                f"! videoconvert "
                f"! appsink name=sink max-buffers=1 drop=true sync=false emit-signals=true"
            )
            
            return pipeline
            
        except Exception as e:
            self.get_logger().error(f"Erro ao construir pipeline: {str(e)}")
            traceback.print_exc()
            return None

    def check_argus_socket(self):
        """Verifica o socket Argus."""
        socket_path = '/tmp/argus_socket'
        return os.path.exists(socket_path)

    def check_nvargus_daemon(self):
        """Verifica o daemon nvargus."""
        try:
            result = subprocess.run("pgrep -f nvargus-daemon", shell=True, 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.returncode == 0
        except:
            return False

    def check_gstreamer_plugin(self, plugin_name):
        """Verifica plugin GStreamer."""
        try:
            result = subprocess.run(['gst-inspect-1.0', plugin_name],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.returncode == 0
        except:
            return False

    def log_gstreamer_failure_hints(self):
        """Imprime dicas comuns de depuração para falhas do GStreamer."""
        self.get_logger().error("┌─────────────────────────────────────────────────────┐")
        self.get_logger().error("│       POSSÍVEIS CAUSAS DA FALHA GSTREAMER           │")
        self.get_logger().error("├─────────────────────────────────────────────────────┤")
        self.get_logger().error("│ 1. Câmera CSI mal conectada ou com problema físico  │")
        self.get_logger().error("│ 2. nvargus-daemon não está rodando no HOST          │")
        self.get_logger().error("│ 3. Container sem acesso aos dispositivos NVIDIA     │")
        self.get_logger().error("│ 4. Outro processo já está usando a câmera           │")
        self.get_logger().error("│    Tente: sudo pkill -f nvargus                     │")
        self.get_logger().error("└─────────────────────────────────────────────────────┘")

    def capture_loop(self):
        """Loop principal para capturar frames."""
        while rclpy.ok() and self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Processar e publicar frame
                    self.process_and_publish_frame(frame)
                    
                    # Calcular FPS (apenas logs ocasionais)
                    self.frame_count += 1
                    now = self.get_clock().now()
                    elapsed = (now.nanoseconds - self.last_fps_update.nanoseconds) / 1e9
                    if elapsed >= 3.0:  # Reduzir frequência do log de FPS
                        self.current_fps = self.frame_count / elapsed
                        self.get_logger().info(f"FPS: {self.current_fps:.1f}")
                        self.frame_count = 0
                        self.last_fps_update = now
                else:
                    # Apenas tentar novamente sem spam de logs
                    time.sleep(0.1)
            else:
                self.get_logger().error("VideoCapture não está aberto.")
                self.is_running = False
                break

            # Limpeza rápida
            if self.cap:
                self.cap.release()
                self.cap = None

    def process_and_publish_frame(self, frame):
        """Processa e publica um frame sem processamento adicional."""
        try:
            # Publicar diretamente sem processamento
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera_optical_frame"
            self.image_pub.publish(img_msg)
            
            # Publicar CameraInfo se disponível
            if self.camera_info_pub and self.camera_info_msg:
                self.camera_info_msg.header = img_msg.header
                self.camera_info_pub.publish(self.camera_info_msg)
            
            # Mostrar imagem se display estiver habilitado (sem texto adicional)
            if self.get_parameter('display').value:
                cv2.imshow('Camera CSI', frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    # Fechar display sem mensagens
                    cv2.destroyAllWindows()
                    
        except Exception as e:
            self.get_logger().error(f"Erro: {str(e)}")

    def test_camera_pipeline(self, timeout=3):
        """Testa um pipeline básico da câmera para diagnóstico."""
        try:
            # Pipeline de teste ULTRA MÍNIMO
            # Isso é o mínimo absoluto para verificar se a câmera responde
            test_cmd = "gst-launch-1.0 nvarguscamerasrc num-buffers=1 ! fakesink -v"
            process = subprocess.run(test_cmd, shell=True, timeout=timeout,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            success = process.returncode == 0
            if success:
                self.get_logger().info("Teste GStreamer básico: SUCESSO")
            else:
                self.get_logger().error("Teste GStreamer básico: FALHA")
            return success
        except Exception as e:
            self.get_logger().error(f"Erro no teste: {str(e)}")
            return False

    def load_camera_calibration(self):
        """Carrega o arquivo de calibração da câmera."""
        calibration_file = self.get_parameter('calibration_file').value
        if not calibration_file or not os.path.isfile(calibration_file):
            return
        
        try:
            with open(calibration_file, 'r') as f:
                calib_data = yaml.safe_load(f)
                
            if 'camera_matrix' in calib_data and 'distortion_coefficients' in calib_data:
                # Criar mensagem básica
                self.camera_info_msg = CameraInfo()
                self.camera_info_msg.width = self.width
                self.camera_info_msg.height = self.height
                self.camera_info_msg.k = [float(x) for x in calib_data['camera_matrix']['data']]
                self.camera_info_msg.d = [float(x) for x in calib_data['distortion_coefficients']['data']]
                self.camera_info_msg.distortion_model = 'plumb_bob'
        except Exception as e:
            self.camera_info_msg = None

    def destroy_node(self):
        """Limpa recursos ao encerrar."""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
        
        if self.get_parameter('display').value:
            cv2.destroyAllWindows()
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = IMX219CameraNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"FALHA: {str(e)}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
