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
        
        # Declarar parâmetros essenciais da câmera
        self.declare_parameters(
            namespace='',
            parameters=[
                ('device_id', 0), # Sensor ID para nvarguscamerasrc
                ('display', False), # Exibir imagem em janela (para debug)
                ('framerate', 30.0), # Framerate alvo (limitado pelo modo da câmera)
                ('enable_cuda', True), # Usar elementos CUDA no pipeline GStreamer
                ('calibration_file', ''), # Arquivo de calibração (opcional)
                ('flip_method', 0), # 0=none, 1=counterclockwise, 2=180, 3=clockwise, 4=horizontal, 5=upper-right-diag, 6=vertical
                # Removidos parâmetros menos essenciais para simplificar
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
            self.camera_fps = 30.0  # Podemos limitar a menos que 120fps se necessário
            self.max_fps = 120.0    # O máximo que o modo suporta
            
            self.get_logger().info("┌───────────────────────────────────────────────┐")
            self.get_logger().info("│ INICIANDO NÓ DA CÂMERA CSI JETSON (ULTRAFAST) │")
            self.get_logger().info("└───────────────────────────────────────────────┘")
            
            # Publishers
            self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
            
            # Inicializar câmera GStreamer
            if not self.init_gstreamer_camera():
                self.get_logger().fatal("⛔ FALHA CRÍTICA AO INICIALIZAR CÂMERA GSTREAMER!")
                raise RuntimeError("Não foi possível inicializar a câmera GStreamer.")
            else:
                self.get_logger().info(f"✅ Câmera inicializada: {self.width}x{self.height} @ {self.camera_fps}fps")
                
                # Carregar calibração se especificada
                self.load_camera_calibration()
                
                # Criar publisher de camera_info se calibração foi carregada
                if self.camera_info_msg:
                    self.camera_info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)
                    self.get_logger().info("✅ Parâmetros de calibração carregados e prontos para publicação")
                
                # Iniciar thread de captura
                self.is_running = True
                self.capture_thread = threading.Thread(target=self.capture_loop)
                self.capture_thread.daemon = True
                self.capture_thread.start()
                self.get_logger().info("✅ Thread de captura iniciada com sucesso")
                
        except Exception as e:
            self.get_logger().fatal(f"⛔ FALHA NA INICIALIZAÇÃO: {str(e)}")
            traceback.print_exc()
            raise

    def init_gstreamer_camera(self):
        """Inicializa a câmera GStreamer com o pipeline CSI."""
        self.get_logger().info("➡️ Verificando ambiente e pré-requisitos...")

        # Verificar se estamos em um container
        is_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
        if is_container:
             self.get_logger().info("ℹ️ Ambiente containerizado detectado")
             
             # Verificar socket Argus e permissões de dispositivos
             socket_ok = self.check_argus_socket()
             if not socket_ok:
                 self.get_logger().warn("⚠️ Socket Argus não encontrado - daemon nvargus talvez não esteja acessível")
                 
             nvargus_daemon_ok = self.check_nvargus_daemon()
             if not nvargus_daemon_ok:
                 self.get_logger().warn("⚠️ nvargus-daemon não confirmado - verifique se está rodando no HOST")

        # Verificar GStreamer e plugin nvarguscamerasrc
        if not self.check_gstreamer_plugin('nvarguscamerasrc'):
             self.get_logger().error("⛔ Plugin nvarguscamerasrc não encontrado! Não é possível continuar.")
             return False

        # Construir pipeline GStreamer (SIMPLIFICADO AO MÁXIMO)
        pipeline_str = self._build_gstreamer_pipeline()
        if not pipeline_str:
             self.get_logger().error("⛔ Falha ao construir string do pipeline GStreamer.")
             return False

        self.get_logger().info(f"➡️ Abrindo pipeline: {pipeline_str}")

        # Tentar abrir a câmera
        try:
            self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                self.get_logger().error("⛔ cv2.VideoCapture falhou ao abrir o pipeline GStreamer!")
                self.log_gstreamer_failure_hints()
                self.test_camera_pipeline(timeout=5)  # Teste básico para diagnóstico
                return False

            # Verificar se conseguimos ler um frame
            self.get_logger().info("➡️ Tentando ler o primeiro frame...")
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.get_logger().error("⛔ Falha ao ler o primeiro frame!")
                self.cap.release()
                self.cap = None
                self.test_camera_pipeline(timeout=5)
                return False
            else:
                self.get_logger().info(f"✅ Primeiro frame lido: {frame.shape}")
                del frame  # Liberar memória

            return True

        except Exception as e:
            self.get_logger().error(f"⛔ Exceção ao abrir VideoCapture: {str(e)}")
            traceback.print_exc()
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def _build_gstreamer_pipeline(self):
        """Constrói um pipeline GStreamer CSI simplificado."""
        try:
            # PIPELINE ULTRAMINALISTA
            # Usar elementos básicos: nvarguscamerasrc -> nvvidconv -> formato BGR para OpenCV
            device_id = self.get_parameter('device_id').value
            flip_method = self.get_parameter('flip_method').value
            
            # Calcular framerate como fração (simplificado)
            fps_num = int(self.camera_fps)
            fps_den = 1
            
            # PIPELINE MÍNIMO - Remoção de todas as opções não essenciais
            pipeline = (
                f"nvarguscamerasrc sensor-id={device_id} "
                f"! video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
                f"format=NV12, framerate={fps_num}/{fps_den} "
                f"! nvvidconv flip-method={flip_method} "
                f"! video/x-raw, format=BGRx "
                f"! videoconvert "
                f"! video/x-raw, format=BGR "
                f"! appsink drop=true"
            )
            
            self.get_logger().info("✅ Pipeline GStreamer ULTRAMINIMALISTA criado com sucesso")
            return pipeline
            
        except Exception as e:
            self.get_logger().error(f"⛔ Erro ao construir pipeline: {str(e)}")
            traceback.print_exc()
            return None

    def check_argus_socket(self):
        """Verifica o socket Argus em ambiente containerizado."""
        socket_path = '/tmp/argus_socket'
        if os.path.exists(socket_path):
            self.get_logger().info(f"✅ Socket Argus encontrado em {socket_path}")
            return True
        else:
            self.get_logger().warn(f"⚠️ Socket Argus não encontrado em {socket_path}")
            # Verificar diretórios alternativos
            alt_paths = ['/tmp/.argus_socket', '/var/nvidia/nvcam/camera-daemon-socket']
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    self.get_logger().info(f"✅ Socket Argus alternativo encontrado em {alt_path}")
                    try:
                        os.makedirs(os.path.dirname(socket_path), exist_ok=True)
                        os.symlink(alt_path, socket_path)
                        self.get_logger().info(f"✅ Link simbólico criado: {socket_path} -> {alt_path}")
                        return True
                    except Exception as e:
                        self.get_logger().warn(f"⚠️ Não foi possível criar link: {str(e)}")
            
            self.get_logger().error("⛔ Socket Argus NÃO encontrado em nenhum local conhecido")
            return False

    def check_nvargus_daemon(self):
        """Verifica o serviço nvargus-daemon (simplificado)."""
        try:
            # Tentar verificar via pgrep (mais provável de funcionar no container)
            result = subprocess.run("pgrep -f nvargus-daemon", shell=True, 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                self.get_logger().info("✅ Processo nvargus-daemon encontrado via pgrep")
                return True
            else:
                self.get_logger().warn("⚠️ Processo nvargus-daemon NÃO encontrado via pgrep")
                self.get_logger().warn("⚠️ O daemon precisa estar rodando no HOST se estiver em container")
                return False
        except Exception as e:
            self.get_logger().warn(f"⚠️ Erro ao verificar nvargus-daemon: {str(e)}")
            return False

    def check_gstreamer_plugin(self, plugin_name):
        """Verifica se um plugin GStreamer específico está disponível."""
        self.get_logger().info(f"➡️ Verificando plugin {plugin_name}...")
        try:
            result = subprocess.run(['gst-inspect-1.0', plugin_name],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                self.get_logger().info(f"✅ Plugin {plugin_name} encontrado")
                return True
            else:
                stderr = result.stderr.decode() if result.stderr else ""
                self.get_logger().error(f"⛔ Plugin {plugin_name} NÃO encontrado: {stderr}")
                return False
        except Exception as e:
            self.get_logger().error(f"⛔ Erro ao verificar plugin {plugin_name}: {str(e)}")
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
        """Loop principal para capturar frames da câmera."""
        self.get_logger().info("➡️ Iniciando loop de captura...")
        while rclpy.ok() and self.is_running:
            if self.cap and self.cap.isOpened():
                start_time = time.time()
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    # Processar e publicar frame
                    self.process_and_publish_frame(frame)

                    # Calcular FPS
                    self.frame_count += 1
                    now = self.get_clock().now()
                    elapsed = (now.nanoseconds - self.last_fps_update.nanoseconds) / 1e9
                    if elapsed >= 1.0:
                        self.current_fps = self.frame_count / elapsed
                        self.get_logger().info(f"📊 FPS: {self.current_fps:.1f}")
                        self.frame_count = 0
                        self.last_fps_update = now

                    # Pequena pausa para não sobrecarregar CPU
                    elapsed_proc = time.time() - start_time
                    target_period = 1.0 / self.camera_fps
                    sleep_time = max(0, target_period - elapsed_proc - 0.001)
                    if sleep_time > 0.001:
                        time.sleep(sleep_time)
                else:
                    self.get_logger().warn("⚠️ Falha ao ler frame. Tentando novamente...")
                    time.sleep(0.1)
            else:
                self.get_logger().error("⛔ VideoCapture não está aberto! Parando thread.")
                self.is_running = False
                break

        self.get_logger().info("➡️ Loop de captura encerrado")
        if self.cap:
            self.cap.release()
            self.cap = None

    def process_and_publish_frame(self, frame):
        """Processa e publica um frame (simplificado, sem CUDA)."""
        try:
            # Usar o frame diretamente sem processamento complexo
            processed_frame = frame
            
            # Publicar como mensagem ROS
            timestamp = self.get_clock().now().to_msg()
            img_msg = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
            img_msg.header.stamp = timestamp
            img_msg.header.frame_id = "camera_optical_frame"
            self.image_pub.publish(img_msg)
            
            # Publicar CameraInfo se disponível
            if self.camera_info_pub and self.camera_info_msg:
                self.camera_info_msg.header = img_msg.header
                self.camera_info_pub.publish(self.camera_info_msg)
            
            # Mostrar imagem se display estiver habilitado
            if self.get_parameter('display').value:
                try:
                    # Adicionar FPS e dimensões ao display
                    display_frame = processed_frame.copy()
                    h, w = display_frame.shape[:2]
                    info_text = f"FPS: {self.current_fps:.1f} | {w}x{h}"
                    
                    # Fundo semi-transparente para o texto
                    cv2.rectangle(display_frame, (10, 10), (350, 40), (0, 0, 0), -1)
                    cv2.putText(display_frame, info_text, (15, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    cv2.imshow('Camera CSI (Ultrasimplificada)', display_frame)
                    key = cv2.waitKey(1)
                    # Permitir fechar com 'q'
                    if key == ord('q'):
                        self.get_logger().info("➡️ Tecla 'q' pressionada, fechando display")
                        cv2.destroyAllWindows()
                except Exception as e:
                    self.get_logger().warn(f"⚠️ Erro no display: {str(e)}")
                    
        except Exception as e:
            self.get_logger().error(f"⛔ Erro ao processar frame: {str(e)}")

    def test_camera_pipeline(self, timeout=5):
        """Testa um pipeline básico da câmera para diagnóstico."""
        self.get_logger().info("➡️ Teste diagnóstico de pipeline mínimo...")
        
        # Pipeline de teste SIMPLIFICADO
        test_cmd = "gst-launch-1.0 nvarguscamerasrc num-buffers=1 ! fakesink -v"
        
        try:
            process = subprocess.run(test_cmd, shell=True, timeout=timeout,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if process.returncode == 0:
                self.get_logger().info("✅ Teste básico de GStreamer PASSOU!")
                return True
            else:
                stderr = process.stderr.decode() if process.stderr else ""
                self.get_logger().error(f"⛔ Teste básico de GStreamer FALHOU: {stderr[:200]}")
                
                # Sugerir soluções específicas
                if "nvargus-daemon" in stderr:
                    self.get_logger().error("⚠️ Problema com nvargus-daemon detectado")
                return False
        except subprocess.TimeoutExpired:
            self.get_logger().error(f"⛔ Timeout após {timeout}s no teste de pipeline")
            return False
        except Exception as e:
            self.get_logger().error(f"⛔ Erro ao testar pipeline: {str(e)}")
            return False

    def load_camera_calibration(self):
        """Carrega o arquivo de calibração da câmera, se especificado (simplificado)."""
        calibration_file = self.get_parameter('calibration_file').value
        if not calibration_file:
            return
            
        if not os.path.isfile(calibration_file):
            self.get_logger().warn(f"⚠️ Arquivo de calibração não encontrado: {calibration_file}")
            return
            
        try:
            with open(calibration_file, 'r') as f:
                calib_data = yaml.safe_load(f)
                
            # Verificar formato mínimo
            if not ('camera_matrix' in calib_data and 'distortion_coefficients' in calib_data):
                self.get_logger().error("⛔ Arquivo de calibração com formato inválido")
                return
                
            # Criar mensagem CameraInfo
            self.camera_info_msg = CameraInfo()
            self.camera_info_msg.width = self.width
            self.camera_info_msg.height = self.height
            
            # Preencher dados de calibração
            self.camera_info_msg.k = [float(x) for x in calib_data['camera_matrix']['data']]
            self.camera_info_msg.d = [float(x) for x in calib_data['distortion_coefficients']['data']]
            self.camera_info_msg.distortion_model = calib_data.get('distortion_model', 'plumb_bob')
            
            self.get_logger().info("✅ Calibração da câmera carregada com sucesso")
            
        except Exception as e:
            self.get_logger().error(f"⛔ Erro ao carregar calibração: {str(e)}")
            self.camera_info_msg = None

    def destroy_node(self):
        """Limpa recursos ao encerrar."""
        self.get_logger().info("➡️ Finalizando nó da câmera...")
        
        # Parar thread de captura
        self.is_running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            
        # Liberar recursos OpenCV
        if self.cap:
            self.cap.release()
            
        # Fechar janelas
        if self.get_parameter('display').value:
            cv2.destroyAllWindows()
            
        super().destroy_node()
        self.get_logger().info("✅ Nó da câmera finalizado com sucesso")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = IMX219CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n⛔ Interrompido pelo usuário (Ctrl+C)")
    except Exception as e:
        print(f"\n⛔ Erro fatal: {str(e)}")
        traceback.print_exc()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
