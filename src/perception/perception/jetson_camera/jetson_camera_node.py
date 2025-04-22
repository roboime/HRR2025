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
import traceback
import yaml

class IMX219CameraNode(Node):
    """
    Nó ROS 2 simplificado para a câmera IMX219 na Jetson Nano usando GStreamer e CUDA.
    """

    def __init__(self):
        super().__init__('jetson_camera_node')
        
        # Parâmetros
        self.declare_parameter('device_id', 0)
        self.declare_parameter('display', False)
        self.declare_parameter('framerate', 30.0)
        self.declare_parameter('enable_cuda', True)
        self.declare_parameter('camera_mode', 6)
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('bypass_opencv', False)
        
        self.device_id = self.get_parameter('device_id').value
        self.display_debug = self.get_parameter('display').value
        self.framerate = self.get_parameter('framerate').value
        self.enable_cuda = self.get_parameter('enable_cuda').value
        self.camera_mode = self.get_parameter('camera_mode').value
        
        # Variáveis de estado
        self.camera = None
        self.is_running = False
        self.capture_thread = None
        self.bridge = CvBridge()
        self.camera_info_msg = None
        self.camera_info_pub = None
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        
        # Configuração e inicialização
        try:
            # Configurar resoluções com base no modo da câmera
            # Modo 6: 1280x720 @ 120fps
            self.width = 1280
            self.height = 720
            self.camera_fps = 30.0
            self.min_exposure = 13000
            self.max_exposure = 683709000
            
            self.get_logger().info("INICIANDO NÓ DA CÂMERA CSI JETSON (MÍNIMO)")
            
            # Publishers
            self.image_pub = self.create_publisher(Image, 'image_raw', 10)
            
            # Verificar se estamos em ambiente containerizado
            is_container = os.path.exists('/.dockerenv')
            if is_container:
                self.get_logger().info("Detectado ambiente containerizado (Docker)")
            
            # Verificar e configurar socket Argus para comunicação com a câmera
            if not self.check_socket_permissions():
                self.get_logger().warn("Falha ao configurar socket Argus; tentando prosseguir mesmo assim")
            
            # Verificar daemon nvargus (que deve estar rodando no host)
            if not self.check_nvargus_daemon():
                self.get_logger().warn("Daemon nvargus não encontrado; tentando prosseguir mesmo assim")
            
            # Inicializar câmera GStreamer
            if not self.configure_camera():
                self.get_logger().fatal("FALHA AO INICIALIZAR CÂMERA GSTREAMER")
                self.test_camera_pipeline()
                self.log_gstreamer_failure_hints()
                raise RuntimeError("Não foi possível inicializar a câmera GStreamer.")
            
            self.get_logger().info(f"Câmera inicializada: {self.width}x{self.height}")
            
            # Carregar calibração se especificada
            self.load_camera_calibration()
            
            # Criar publisher de camera_info se calibração foi carregada
            if self.camera_info_msg:
                self.camera_info_pub = self.create_publisher(CameraInfo, 'camera_info', 10)
            
            # Iniciar thread de captura
            self.is_running = True
            self.capture_thread = threading.Thread(target=self.capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            # Log informações sobre a configuração
            self.get_logger().info(f"Nó de câmera inicializado com ID {self.device_id}")
            self.get_logger().info(f"Taxa de frames configurada para: {self.framerate}")
            self.get_logger().info(f"Modo de câmera: {self.camera_mode}")
            self.get_logger().info(f"CUDA habilitado: {self.enable_cuda}")
            self.get_logger().info(f"Debug display: {self.display_debug}")
            
        except Exception as e:
            self.get_logger().fatal(f"FALHA NA INICIALIZAÇÃO: {str(e)}")
            traceback.print_exc()
            raise

    def configure_camera(self):
        """Configura a câmera utilizando GStreamer e OpenCV."""
        try:
            # Verificar disponibilidade do plugin GStreamer
            try:
                self.get_logger().info("Verificando plugins GStreamer necessários...")
                
                # Verificar nvarguscamerasrc
                gst_check = subprocess.run(
                    ['gst-inspect-1.0', 'nvarguscamerasrc'], 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                if gst_check.returncode != 0:
                    self.get_logger().error("Plugin 'nvarguscamerasrc' não encontrado! Este plugin é essencial.")
                    return False
                
                # Verificar nvvidconv
                nv_check = subprocess.run(
                    ['gst-inspect-1.0', 'nvvidconv'], 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                if nv_check.returncode != 0:
                    self.get_logger().error("Plugin 'nvvidconv' não encontrado! Este plugin é essencial.")
                    return False
                
                self.get_logger().info("Plugins GStreamer necessários estão disponíveis.")
                
            except Exception as e:
                self.get_logger().error(f"Erro ao verificar plugins GStreamer: {str(e)}")
                return False
            
            # Teste básico do pipeline antes de tentar configurar
            test_result = self.test_camera_pipeline(timeout=5)
            if not test_result:
                self.get_logger().warn("Teste de pipeline falhou! Tentando configuração mesmo assim...")
            
            # Construir pipeline GStreamer
            pipeline = self._build_gstreamer_pipeline()
            self.get_logger().info(f"Iniciando câmera com pipeline: {pipeline}")
            
            # Tentar abrir a câmera
            self.camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.camera.isOpened():
                self.get_logger().error("Falha ao abrir a câmera com OpenCV/GStreamer")
                
                # Tentar pipeline alternativo ultra-simplificado
                self.get_logger().info("Tentando pipeline alternativo...")
                alt_pipeline = (
                    f"nvarguscamerasrc sensor-id={self.device_id} ! "
                    f"nvvidconv ! video/x-raw, format=BGRx ! "
                    f"videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink max-buffers=1 drop=true sync=false"
                )
                
                self.get_logger().info(f"Pipeline alternativo: {alt_pipeline}")
                self.camera = cv2.VideoCapture(alt_pipeline, cv2.CAP_GSTREAMER)
                
                if not self.camera.isOpened():
                    self.get_logger().error("Falha também com pipeline alternativo!")
                    return False
                else:
                    self.get_logger().info("Pipeline alternativo funcionou! Usando esta configuração.")
                    return True
                
            # Verificar se a câmera realmente funciona lendo um frame
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                self.get_logger().error("Câmera aberta, mas não conseguiu ler frame!")
                self.camera.release()
                self.camera = None
                return False
                
            self.get_logger().info(f"Câmera inicializada com sucesso! Shape do frame: {test_frame.shape}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Erro ao configurar câmera: {str(e)}")
            traceback.print_exc()
            return False

    def _build_gstreamer_pipeline(self):
        """Constrói o pipeline GStreamer para a câmera IMX219."""
        # Parâmetros básicos
        sensor_id = self.device_id
        framerate = int(self.framerate)
        
        # Pipeline básico para a câmera IMX219 (Raspberry Pi v2)
        # Versão simplificada e otimizada para ambiente containerizado
        pipeline = (
            f"nvarguscamerasrc sensor-id={sensor_id} "
            f"! video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate={framerate}/1 "
        )
        
        # Adicionar elementos de processamento
        if self.enable_cuda:
            # Versão com conversão CUDA simplificada e robusta para ambientes containerizados
            pipeline += (
                f"! nvvidconv ! "
                f"video/x-raw, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink max-buffers=1 drop=true sync=false"
            )
        else:
            # Fallback simples sem CUDA
            pipeline += (
                f"! nvvidconv ! "
                f"video/x-raw, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink max-buffers=1 drop=true sync=false"
            )
        
        self.get_logger().info(f"Pipeline GStreamer: {pipeline}")
        return pipeline

    def test_camera_pipeline(self, timeout=3):
        """Testa se a câmera está acessível usando gst-launch-1.0 diretamente."""
        try:
            self.get_logger().info("Executando teste básico de pipeline GStreamer...")

            # Teste com um pipeline simples para verificar se a câmera está acessível
            basic_cmd = f"gst-launch-1.0 nvarguscamerasrc num-buffers=10 ! fakesink -v"
            self.get_logger().info(f"Comando de teste: {basic_cmd}")
            
            process = subprocess.run(
                basic_cmd, 
                shell=True,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout = process.stdout.decode()
            stderr = process.stderr.decode()
            
            # Verificar se o pipeline executou com sucesso
            if process.returncode == 0:
                self.get_logger().info("Teste básico PASSOU: A câmera está acessível via GStreamer")
                
                # Teste 2: pipeline exatamente igual ao que estamos usando
                self.get_logger().info("Testando pipeline de integração...")
                integration_test_cmd = (
                    f"gst-launch-1.0 nvarguscamerasrc sensor-id={self.device_id} "
                    f"! video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
                    f"format=NV12, framerate={int(self.framerate)}/1 "
                    f"! nvvidconv "
                    f"! video/x-raw, format=BGRx "
                    f"! fakesink num-buffers=5 -v"
                )
                
                self.get_logger().info(f"Comando de integração: {integration_test_cmd}")
                
                integration_process = subprocess.run(
                    integration_test_cmd, 
                    shell=True,
                    timeout=timeout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                int_stdout = integration_process.stdout.decode()
                int_stderr = integration_process.stderr.decode()
                
                if integration_process.returncode == 0:
                    self.get_logger().info("Teste de integração PASSOU: Pipeline está funcionando")
                    return True
                else:
                    self.get_logger().error("Teste básico PASSOU, mas teste de integração FALHOU")
                    self.get_logger().info("Isso sugere problema nas configurações do pipeline")
                    self.get_logger().error(f"Stdout: {int_stdout[:200]}")
                    self.get_logger().error(f"Stderr: {int_stderr[:200]}")
                    
                    # Teste 3: Pipeline ultra-simplificado com nvvidconv
                    self.get_logger().info("Tentando pipeline ultra-simplificado...")
                    simple_test_cmd = (
                        f"gst-launch-1.0 nvarguscamerasrc ! "
                        f"nvvidconv ! fakesink -v"
                    )
                    
                    simple_process = subprocess.run(
                        simple_test_cmd, 
                        shell=True,
                        timeout=timeout,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    if simple_process.returncode == 0:
                        self.get_logger().info("PIPELINE ULTRA-SIMPLIFICADO FUNCIONOU!")
                        self.get_logger().info("Isso sugere que o problema está nos parâmetros/formatos")
                    else:
                        self.get_logger().error("Mesmo o pipeline ultra-simplificado falhou.")
                        self.get_logger().error("Isso sugere que o nvvidconv está com problemas no sistema")
                    
                    # Ver status do daemon nvargus
                    try:
                        nvargus_check = subprocess.run(
                            "ps aux | grep nvargus-daemon | grep -v grep", 
                            shell=True, 
                            stdout=subprocess.PIPE
                        )
                        if nvargus_check.returncode == 0:
                            self.get_logger().info(f"nvargus-daemon está rodando: {nvargus_check.stdout.decode()}")
                        else:
                            self.get_logger().error("ERRO CRÍTICO: nvargus-daemon não está rodando no host!")
                    except Exception:
                        pass
                        
                    return False
            else:
                self.get_logger().error("Teste básico FALHOU: A câmera pode não estar acessível")
                
                # Fornecer informações de diagnóstico
                self.get_logger().error(f"Código de retorno: {process.returncode}")
                if stdout:
                    self.get_logger().error(f"stdout: {stdout[:200]}...")
                if stderr:
                    self.get_logger().error(f"stderr: {stderr[:200]}...")
                    
                # Verificar CV2 info
                self.get_logger().info(f"OpenCV version: {cv2.__version__}")
                
                # Verificar dispositivos de vídeo disponíveis
                try:
                    video_devices = subprocess.run(
                        "ls -la /dev/video*", 
                        shell=True, 
                        stdout=subprocess.PIPE
                    )
                    if video_devices.returncode == 0:
                        self.get_logger().info(f"Dispositivos de vídeo disponíveis:\n{video_devices.stdout.decode()}")
                except Exception:
                    pass
                
                return False
                
        except subprocess.TimeoutExpired:
            self.get_logger().error(f"Timeout ao testar o pipeline (após {timeout}s)")
            return False
        except Exception as e:
            self.get_logger().error(f"Erro ao testar o pipeline: {str(e)}")
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
            if self.camera and self.camera.isOpened():
                try:
                    # Processar e publicar frame
                    self.process_and_publish_frame()
                    
                    # Calcular FPS (apenas logs ocasionais)
                    self.frame_count += 1
                    now = time.time()
                    elapsed = now - self.last_fps_update
                    if elapsed >= 3.0:  # Reduzir frequência do log de FPS
                        self.current_fps = self.frame_count / elapsed
                        self.get_logger().info(f"FPS: {self.current_fps:.1f}")
                        self.frame_count = 0
                        self.last_fps_update = now
                except Exception as e:
                    self.get_logger().error(f"Erro no loop de captura: {str(e)}")
            else:
                self.get_logger().error("Câmera não está aberta ou foi fechada.")
                break
            
            # Pequena pausa para não sobrecarregar CPU
            time.sleep(0.001)

        # Limpeza
        if self.camera:
            self.camera.release()
            self.camera = None

    def process_and_publish_frame(self):
        """Captura, processa e publica um frame da câmera."""
        if not self.camera or not self.camera.isOpened():
            self.get_logger().error("Câmera não está aberta para captura")
            return

        try:
            # Capturar frame da câmera
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                self.get_logger().warn("Não foi possível capturar frame da câmera")
                return
                
            # Processar imagem com CUDA (se disponível)
            # Na versão simplificada, apenas usamos o frame original
            processed_frame = frame
            
            # Adicionar informações de debug na imagem se solicitado
            if self.display_debug:
                self._add_debug_info(processed_frame)
            
            # Converter para mensagem ROS
            img_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera_frame"
            
            # Publicar mensagens
            self.image_pub.publish(img_msg)
            
            # Publicar camera_info se disponível
            if self.camera_info_pub and self.camera_info_msg:
                self.camera_info_msg.header = img_msg.header
                self.camera_info_pub.publish(self.camera_info_msg)
            
            # Exibir imagem se solicitado
            if self.display_debug:
                cv2.imshow("Câmera IMX219", processed_frame)
                cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f"Erro ao processar frame: {str(e)}")
            traceback.print_exc()

    def _add_debug_info(self, image):
        """Adiciona informações de debug à imagem."""
        if image is None:
            return

        # Adicionar informações básicas
        height, width = image.shape[:2]
        text_color = (0, 255, 0)  # Verde
        
        # Linha 1: FPS
        cv2.putText(
            image, 
            f"FPS: {self.current_fps:.1f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            text_color, 
            2
        )
        
        # Linha 2: Resolução
        cv2.putText(
            image, 
            f"Resolução: {width}x{height}", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            text_color, 
            2
        )
        
        # Linha 3: Status CUDA
        cuda_status = "Ativado" if self.enable_cuda else "Desativado"
        cv2.putText(
            image, 
            f"CUDA: {cuda_status}", 
            (10, 110), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            text_color, 
            2
        )

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
            self.get_logger().error(f"Erro ao carregar calibração: {str(e)}")
            self.camera_info_msg = None

    def destroy_node(self):
        """Limpa recursos ao encerrar."""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.camera:
            self.camera.release()
        
        if self.display_debug:
            cv2.destroyAllWindows()
        
        super().destroy_node()

    def check_nvargus_daemon(self):
        """Verifica se o daemon nvargus está rodando no sistema host."""
        try:
            self.get_logger().info("Verificando status do daemon nvargus...")
            
            # Verificar se o daemon está rodando
            daemon_check = subprocess.run(
                "ps aux | grep nvargus-daemon | grep -v grep", 
                shell=True, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if daemon_check.returncode == 0:
                daemon_info = daemon_check.stdout.decode()
                self.get_logger().info(f"Daemon nvargus está rodando: {daemon_info.strip()}")
                return True
            else:
                self.get_logger().error("ERRO CRÍTICO: Daemon nvargus não encontrado no sistema!")
                self.get_logger().error("O daemon nvargus-daemon precisa estar rodando no host para acessar a câmera CSI")
                self.get_logger().error("Isso geralmente ocorre no sistema host, não no container")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Erro ao verificar daemon nvargus: {str(e)}")
            return False
            
    def check_socket_permissions(self):
        """Verifica permissões do socket Argus."""
        try:
            socket_path = "/tmp/argus_socket"
            
            # Verificar se o diretório do socket existe
            if not os.path.exists(socket_path):
                self.get_logger().info(f"Criando diretório do socket: {socket_path}")
                os.makedirs(socket_path, exist_ok=True)
                
            # Verificar permissões
            subprocess.run(f"chmod 777 {socket_path}", shell=True)
            
            self.get_logger().info(f"Diretório do socket configurado: {socket_path}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Erro ao configurar socket Argus: {str(e)}")
            return False

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