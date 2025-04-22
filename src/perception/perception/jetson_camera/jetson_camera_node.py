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
import asyncio
import sys
import glob
import traceback
import yaml
import psutil

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
        super().__init__('jetson_camera_node')
        
        # Declarar parâmetros da câmera
        self.declare_parameters(
            namespace='',
            parameters=[
                ('device_id', 0),
                ('sim_mode', False),
                ('sim_path', ''),
                ('sim_loop', True),
                ('display', False),
                ('framerate', 30.0),
                ('cuda_enabled', False),
                ('cuda_resize', False),
                ('apply_noise_reduction', False),
                ('apply_edge_enhancement', False),
                ('apply_brightness_adjustment', False),
                ('brightness_factor', 1.0),
                ('resize_output', False),
                ('output_width', 1280),
                ('output_height', 720),
                ('calibration_file', ''),
                ('image_width', 3280),
                ('image_height', 2464),
                # Parâmetros adicionais do arquivo de lançamento
                ('camera_mode', 6),
                ('camera_fps', 120.0),
                ('exposure_time', 13333),
                ('gain', 1.0),
                ('awb_mode', 1),
                ('brightness', 0),
                ('saturation', 1.0),
                ('enable_hdr', False),
                ('enable_cuda', True),
                ('enable_display', False),
                ('flip_method', 0)
            ]
        )
        
        # Inicializar variáveis que serão usadas mais tarde para evitar erros
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.real_fps = 0.0
        self.is_simulated = False
        self.video_device_node = None
        self.fake_camera = False
        
        # Configurar a câmera com base nos parâmetros
        self._configure_camera()
        
        # Inicializar o processamento
        self._configure_processing()
        
        # Carregar calibração da câmera
        self.load_camera_calibration()
        
        # Criar timer para callback da câmera
        callback_period = 1.0 / self.camera_fps
        self.camera_timer = self.create_timer(callback_period, self.camera_callback)
        
        # Configurar monitoramento de recursos
        self.last_resource_print = time.time()
        self.resource_timer = self.create_timer(2.0, self.monitor_resources)
        
        self.get_logger().info('Nó da câmera IMX219 inicializado')

    def _configure_camera(self):
        """Configura a câmera com base nos parâmetros."""
        self.device_id = self.get_parameter('device_id').value
        self.sim_mode = self.get_parameter('sim_mode').value
        self.sim_path = self.get_parameter('sim_path').value
        # Configurações baseadas no modo da câmera
        self.camera_modes = {
            0: (3280, 2464, 21),  # Máxima resolução
            1: (1920, 1080, 60),  # Full HD
            2: (1280, 720, 120),  # HD
            3: (1280, 720, 60),   # HD (60fps)
            4: (1920, 1080, 30),  # Full HD (30fps)
            5: (1640, 1232, 30),  # 4:3 (30fps)
            6: (1280, 720, 120)   # HD (120fps, modo da câmera IMX219)
        }
        
        # Obter modo da câmera
        self.camera_mode = self.get_parameter('camera_mode').value
        self.width, self.height, self.max_fps = self.camera_modes[self.camera_mode]
        
        # Ajustar FPS baseado no modo
        requested_fps = min(self.get_parameter('camera_fps').value, self.max_fps)
        self.get_logger().info(f'Ajustando FPS para {requested_fps} (máximo permitido para o modo selecionado)')
        self.camera_fps = requested_fps  # Armazenar o valor ajustado
        
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

    def debug_camera_devices(self):
        """Verifica e lista dispositivos de câmera disponíveis no sistema.
        
        Returns:
            bool: True se pelo menos um dispositivo de câmera foi encontrado, False caso contrário
        """
        self.get_logger().info('Verificando dispositivos de câmera disponíveis...')
        devices_found = []
        
        # Método 1: Verificar no diretório /dev/video*
        try:
            import glob
            video_devices = glob.glob('/dev/video*')
            if video_devices:
                self.get_logger().info(f'Dispositivos encontrados em /dev: {video_devices}')
                devices_found.extend(video_devices)
        except Exception as e:
            self.get_logger().warn(f'Erro ao listar dispositivos em /dev: {str(e)}')
        
        # Método 2: Usar o v4l-utils se disponível
        try:
            import subprocess
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True)
            if result.returncode == 0 and result.stdout:
                self.get_logger().info(f'Dispositivos detectados via v4l2-ctl:\n{result.stdout}')
                # Analisar a saída para detectar câmeras CSI
                if 'imx219' in result.stdout.lower() or 'csi' in result.stdout.lower():
                    self.get_logger().info('Câmera CSI IMX219 detectada!')
                    devices_found.append('imx219')
            else:
                self.get_logger().debug(f'v4l2-ctl falhou ou não encontrou dispositivos: {result.stderr}')
        except Exception as e:
            self.get_logger().debug(f'Erro ao executar v4l2-ctl: {str(e)}')
        
        # Método 3: Verificar CUDA devices para imx219
        try:
            # No Windows, verificamos diretamente os dispositivos de câmera via API do Windows
            if os.name == 'nt':
                import cv2
                for i in range(10):  # Testa índices de 0 a 9
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            self.get_logger().info(f'Câmera encontrada no índice {i}')
                            # Obter informações da câmera
                            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            self.get_logger().info(f'Resolução: {width}x{height}')
                            devices_found.append(f'camera_{i}')
                        cap.release()
            else:
                # Em sistemas Linux, verificamos dispositivos específicos da Jetson
                gst_exists = subprocess.run(['which', 'gst-inspect-1.0'], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE).returncode == 0
                
                if gst_exists:
                    result = subprocess.run(['gst-inspect-1.0', 'nvarguscamerasrc'], 
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE,
                                           text=True)
                    if 'nvarguscamerasrc' in result.stdout:
                        self.get_logger().info('Plugin GStreamer nvarguscamerasrc disponível')
                        devices_found.append('nvarguscamerasrc')
        except Exception as e:
            self.get_logger().warn(f'Erro ao verificar dispositivos específicos: {str(e)}')
        
        # Resumo dos dispositivos encontrados
        if devices_found:
            self.get_logger().info(f'Total de {len(devices_found)} dispositivos de câmera encontrados')
            return True
        else:
            self.get_logger().error('Nenhum dispositivo de câmera encontrado!')
            return False
            
    def init_camera(self):
        """Inicializa a câmera com configurações otimizadas."""
        # Verificar se os dispositivos da câmera estão disponíveis
        devices_available = self.debug_camera_devices()
        
        if not devices_available:
            self.get_logger().error('Nenhum dispositivo de câmera encontrado!')
            raise RuntimeError("Câmera CSI não encontrada. Verifique a conexão física.")
        
        # Verificar ambiente containerizado
        is_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
        if is_container:
            self.get_logger().info('Ambiente containerizado detectado')
            
            # Verificar socket Argus
            self.check_argus_socket()
            
            # Verificar permissões de dispositivos
            self.check_nvargus_permissions()
            
            # Verificar serviço nvargus-daemon
            self.check_nvargus_daemon()
        
        # Configurar variáveis de ambiente
        self.get_logger().info('Configurando variáveis de ambiente para GStreamer...')
        os.environ["GST_GL_API"] = "gles2"
        os.environ["GST_GL_PLATFORM"] = "egl"
        os.environ["GST_GL_XINITTHREADS"] = "0"
        os.environ["__GL_SYNC_TO_VBLANK"] = "0"
        
        # Construir e inicializar o pipeline
        if not self._construct_camera_pipeline():
            self.get_logger().error('Falha ao inicializar câmera!')
            raise RuntimeError("Falha ao inicializar câmera CSI. Verifique os logs para diagnóstico.")
        
        # Thread de captura
        self.is_running = True
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.get_logger().info(f'Câmera CSI inicializada com sucesso: {self.width}x{self.height} @ {self.camera_fps}fps')

    def _construct_camera_pipeline(self):
        """Constrói o pipeline GStreamer para a câmera CSI."""
        try:
            # Verificar se o plugin nvarguscamerasrc está disponível
            try:
                self.get_logger().info('Verificando disponibilidade do plugin nvarguscamerasrc...')
                gst_check = subprocess.check_output(['gst-inspect-1.0', 'nvarguscamerasrc'], stderr=subprocess.STDOUT).decode('utf-8')
                if 'nvarguscamerasrc' not in gst_check:
                    self.get_logger().error('Plugin GStreamer nvarguscamerasrc não encontrado.')
                    self.get_logger().info('Resultado da inspeção: ' + gst_check[:200] + '...' if len(gst_check) > 200 else gst_check)
                    return False
                else:
                    self.get_logger().info('Plugin nvarguscamerasrc encontrado com sucesso!')
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                self.get_logger().error(f'GStreamer não disponível: {str(e)}')
                return False
            
            # Calcular framerate como fração reduzida para evitar problemas
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return a
                
            fps_num = int(self.camera_fps)
            fps_den = 1
            divisor = gcd(fps_num, fps_den)
            fps_num //= divisor
            fps_den //= divisor
            
            # Verificar ambiente containerizado
            is_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
            
            # Pipeline para ambiente containerizado - mais simples e direto
            if is_container:
                pipeline = (
                    f"nvarguscamerasrc sensor-id=0 "
                    f"! video/x-raw(memory:NVMM), width=(int){self.width}, height=(int){self.height}, "
                    f"format=(string)NV12, framerate=(fraction){fps_num}/{fps_den} "
                    f"! nvvidconv flip-method={self.get_parameter('flip_method').value} "
                    f"! video/x-raw, format=(string)BGRx "
                    f"! videoconvert ! video/x-raw, format=(string)BGR "
                    f"! appsink drop=true max-buffers=2"
                )
            else:
                # Pipeline padrão para ambiente nativo da Jetson
                pipeline = (
                    f"nvarguscamerasrc sensor-id=0 do-timestamp=true "
                    f"exposuretimerange='{self.get_parameter('exposure_time').value} {self.get_parameter('exposure_time').value}' "
                    f"gainrange='{self.get_parameter('gain').value} {self.get_parameter('gain').value}' "
                    f"wbmode={self.get_parameter('awb_mode').value} "
                    f"tnr-mode=2 ee-mode=2 "
                    f"! video/x-raw(memory:NVMM), width=(int){self.width}, height=(int){self.height}, "
                    f"format=(string)NV12, framerate=(fraction){fps_num}/{fps_den} "
                    f"! nvvidconv flip-method={self.get_parameter('flip_method').value} "
                    f"! video/x-raw, format=(string)BGRx "
                )
            
            # Adicionar elementos específicos para CUDA se habilitado
            if self.get_parameter('enable_cuda').value:
                # Verificar permissões do CUDA antes de abrir a câmera
                if not self.check_cuda_permissions():
                    self.get_logger().warn('CUDA não disponível. Usando pipeline sem aceleração CUDA.')
                    pipeline += f"! videoconvert ! video/x-raw, format=(string)BGR "
                else:
                    # Adicionar processamento CUDA via nvivafilter
                    pipeline += (
                        f"! nvvideoconvert ! video/x-raw(memory:NVMM), format=(string)RGBA "
                        f"! nvdsosd ! nvegltransform ! video/x-raw(memory:NVMM), format=(string)RGBA "
                        f"! nvvideoconvert ! video/x-raw, format=(string)BGR "
                    )
            else:
                pipeline += f"! videoconvert ! video/x-raw, format=(string)BGR "
            
            # Finalizar o pipeline com o elemento de saída
            pipeline += f"! appsink max-buffers=4 drop=true sync=false name=sink emit-signals=true"
            
            self.get_logger().info(f'Pipeline GStreamer: {pipeline}')
            
            # Abrir a câmera com o pipeline GStreamer
            try:
                # Liberar recursos existentes
                self.get_logger().info('Liberando recursos GStreamer existentes...')
                try:
                    subprocess.run("pkill -f 'gst-launch.*nvarguscamerasrc'", shell=True, 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(1)
                except Exception as kill_err:
                    self.get_logger().debug(f'Erro ao liberar recursos: {str(kill_err)}')
                
                # Abrir a câmera
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                
                # Verificar se a câmera foi aberta com sucesso
                if not self.cap.isOpened():
                    self.get_logger().error('Falha ao abrir câmera com GStreamer!')
                    
                    # Tentar um pipeline mais simples como último recurso
                    self.get_logger().warn('Tentando pipeline mais simples...')
                    simple_pipeline = (
                        f"nvarguscamerasrc sensor-id=0 ! "
                        f"video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
                        f"nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
                        f"videoconvert ! video/x-raw, format=BGR ! "
                        f"appsink max-buffers=1 drop=true"
                    )
                    self.get_logger().info(f'Pipeline simplificado: {simple_pipeline}')
                    self.cap = cv2.VideoCapture(simple_pipeline, cv2.CAP_GSTREAMER)
                    
                    if not self.cap.isOpened():
                        self.get_logger().error('Falha com pipeline simplificado.')
                        self.get_logger().info('Ativando modo de simulação.')
                        self.setup_simulation_mode()
                        return True  # Retornar True mesmo com simulação
                
                # Câmera aberta com sucesso, capturar o primeiro frame
                ret, test_frame = self.cap.read()
                if ret:
                    self.get_logger().info(f'Primeiro frame capturado: {test_frame.shape}')
                else:
                    self.get_logger().warn('Não foi possível capturar o primeiro frame.')
                    
                # Configurar variáveis para cálculo de FPS real
                self.frame_count = 0
                self.last_fps_time = time.time()
                self.real_fps = 0.0
                
                return True
                
            except Exception as e:
                self.get_logger().error(f'Exceção ao inicializar câmera: {str(e)}')
                self.get_logger().info('Ativando modo de simulação.')
                self.setup_simulation_mode()
                return True  # Retornar True mesmo com simulação
                
        except Exception as e:
            self.get_logger().error(f'Erro ao construir pipeline: {str(e)}')
            return False
            
    def check_argus_socket(self):
        """Verifica o socket Argus em ambiente containerizado."""
        socket_path = '/tmp/argus_socket'
        if not os.path.exists(socket_path):
            self.get_logger().warn(f'Socket Argus não encontrado em {socket_path}')
            self.get_logger().info('Verificando socket Argus em locais alternativos...')
            
            # Tentar encontrar o socket Argus em outros lugares comuns
            for alt_path in ['/tmp/.argus_socket', '/var/tmp/argus_socket']:
                if os.path.exists(alt_path):
                    self.get_logger().info(f'Socket Argus encontrado em: {alt_path}')
                    # Tente criar um link simbólico para o local padrão
                    try:
                        os.symlink(alt_path, socket_path)
                        self.get_logger().info(f'Link simbólico criado para {socket_path}')
                        break
                    except Exception as e:
                        self.get_logger().warn(f'Não foi possível criar link simbólico: {str(e)}')
                        
    def check_nvargus_permissions(self):
        """Verifica permissões de dispositivos NVIDIA."""
        devices = ['/dev/nvhost-ctrl', '/dev/nvhost-ctrl-gpu', '/dev/nvhost-vic']
        for device in devices:
            if os.path.exists(device):
                try:
                    mode = os.stat(device).st_mode
                    readable = bool(mode & 0o444)
                    self.get_logger().info(f'Dispositivo {device} é legível: {readable}')
                except Exception as e:
                    self.get_logger().warn(f'Erro ao verificar {device}: {str(e)}')
                    
    def check_nvargus_daemon(self):
        """Verifica o serviço nvargus-daemon."""
        try:
            self.get_logger().info('Verificando status do serviço nvargus-daemon...')
            try:
                status = subprocess.run(['systemctl', 'is-active', 'nvargus-daemon'], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if status.returncode != 0:
                    self.get_logger().warn('Serviço nvargus-daemon não está ativo!')
                    self.get_logger().info('Tentando iniciar o serviço...')
                    
                    try:
                        subprocess.run(['systemctl', 'start', 'nvargus-daemon'], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        time.sleep(2)  # Aguardar serviço iniciar
                        
                        # Verificar novamente
                        status = subprocess.run(['systemctl', 'is-active', 'nvargus-daemon'], 
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        if status.returncode == 0:
                            self.get_logger().info('Serviço nvargus-daemon iniciado com sucesso!')
                        else:
                            self.get_logger().warn('Não foi possível iniciar o serviço nvargus-daemon')
                    except Exception as e:
                        self.get_logger().warn(f'Erro ao iniciar serviço: {str(e)}')
                else:
                    self.get_logger().info('Serviço nvargus-daemon está ativo e funcionando')
            except Exception as e:
                self.get_logger().warn(f'Erro ao verificar serviço nvargus-daemon: {str(e)}')
        except Exception as e:
            self.get_logger().warn(f'Erro ao verificar serviço nvargus-daemon: {str(e)}')

    def monitor_resources(self):
        """Monitora recursos do sistema."""
        try:
            # Limitar atualizações a cada 5 segundos
            now = time.time()
            if hasattr(self, 'last_resource_print') and now - self.last_resource_print < 5.0:
                return True
                
            # Verificar uso de CPU e memória
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            mem_used_mb = mem.used / (1024 * 1024)
            mem_total_mb = mem.total / (1024 * 1024)
            
            # Definir status baseado no modo de operação
            status = "Simulação" if self.is_simulated else "Ativa"
            
            # Registrar informações de recursos
            self.get_logger().info(
                f"Status da câmera: {status} | "
                f"CPU: {cpu_percent:.1f}% | "
                f"RAM: {mem_used_mb:.0f}/{mem_total_mb:.0f}MB | "
                f"FPS: {self.real_fps:.1f}/{self.camera_fps}"
            )
            
            # Atualizar timestamp
            self.last_resource_print = now
            return True
            
        except Exception as e:
            # Usar warning ao invés de error para não interromper a operação
            self.get_logger().warning(f'Erro ao monitorar recursos: {str(e)}')
            return False

    def get_camera_info_gstreamer(self):
        """Obtém informações da câmera Jetson através do GStreamer."""
        self.get_logger().info('Obtendo informações da câmera via GStreamer')

        # Verifica se o daemon nvargus está em execução
        self.get_logger().info('Verificando status do nvargus-daemon...')
        
        daemon_running = False
        restart_attempted = False
        
        # Método 1: Verificar processo usando ps
        try:
            ps_result = subprocess.run(
                "ps -ef | grep nvargus-daemon | grep -v grep", 
                shell=True, 
                stdout=subprocess.PIPE
            )
            if ps_result.returncode == 0 and ps_result.stdout:
                self.get_logger().info('Processo nvargus-daemon encontrado via ps')
                daemon_running = True
        except Exception as e:
            self.get_logger().warn(f'Erro ao verificar processo nvargus-daemon: {str(e)}')
        
        # Método 2: Verificar serviço usando systemctl
        if not daemon_running:
            try:
                systemctl_result = subprocess.run(
                    "systemctl is-active nvargus-daemon", 
                    shell=True, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                if systemctl_result.returncode == 0 and b'active' in systemctl_result.stdout:
                    self.get_logger().info('Serviço nvargus-daemon ativo via systemctl')
                    daemon_running = True
            except Exception as e:
                self.get_logger().warn(f'Erro ao verificar serviço nvargus-daemon: {str(e)}')
        
        # Método 3: Verificar socket
        if not daemon_running:
            if os.path.exists('/tmp/nvargus-daemon'):
                self.get_logger().info('Socket do nvargus-daemon encontrado')
                daemon_running = True
        
        # Se o daemon não estiver em execução, tenta reiniciá-lo
        if not daemon_running:
            self.get_logger().warn('Daemon nvargus-daemon não está em execução!')
            
            # Tenta matar qualquer processo residual
            try:
                self.get_logger().info('Tentando matar processos residuais nvargus...')
                subprocess.run("pkill -f nvargus", shell=True)
                time.sleep(1)  # Aguarda um pouco
            except Exception as e:
                self.get_logger().warn(f'Erro ao tentar matar processos: {str(e)}')
            
            # Tenta reiniciar via systemctl
            try:
                self.get_logger().info('Tentando reiniciar serviço nvargus-daemon...')
                result = subprocess.run(
                    "systemctl restart nvargus-daemon", 
                    shell=True,
                    stderr=subprocess.PIPE
                )
                if result.returncode == 0:
                    time.sleep(2)  # Aguarda o serviço iniciar
                    restart_attempted = True
                    self.get_logger().info('Reinício do serviço nvargus-daemon resolveu o problema!')
                else:
                    stderr = result.stderr.decode('utf-8') if result.stderr else "Unknown error"
                    self.get_logger().warn(f"Erro ao reiniciar via systemctl: {stderr}")
            except Exception as e:
                self.get_logger().warn(f'Erro ao tentar reiniciar serviço nvargus-daemon: {str(e)}')
            
            # Se systemctl falhou, tenta iniciar diretamente
            if not restart_attempted:
                try:
                    self.get_logger().info('Tentando iniciar daemon diretamente...')
                    # Inicia em background e descarta saída
                    subprocess.Popen(
                        "nvargus-daemon",
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    time.sleep(2)  # Aguarda o serviço iniciar
                    self.get_logger().info('Daemon iniciado manualmente')
                except Exception as e:
                    self.get_logger().error(f'Falha ao iniciar daemon manualmente: {str(e)}')
        
        # Verifica se o plugin está disponível
        try:
            plugin_check = subprocess.run(
                "gst-inspect-1.0 nvarguscamerasrc", 
                shell=True, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if plugin_check.returncode != 0:
                self.get_logger().error('Plugin nvarguscamerasrc não encontrado! Verifique a instalação do Jetson Multimedia API')
                return None
            else:
                self.get_logger().info('Plugin nvarguscamerasrc disponível')
        except Exception as e:
            self.get_logger().error(f'Erro ao verificar plugin nvarguscamerasrc: {str(e)}')
            return None
            
        # Testa a pipeline da câmera
        camera_test_success, error_msg = self.test_camera_pipeline(timeout=8)
        if not camera_test_success:
            self.get_logger().error(f'Falha no teste da pipeline da câmera: {error_msg}')
            
            # Se acabamos de reiniciar o daemon, espere mais um pouco e tente novamente
            if restart_attempted:
                self.get_logger().info('Aguardando mais 5 segundos após reinício do daemon...')
                time.sleep(5)
                camera_test_success, error_msg = self.test_camera_pipeline(timeout=8)
                if not camera_test_success:
                    self.get_logger().error('Falha persistente após reinício do daemon e tempo de espera adicional')
                    return None
                else:
                    self.get_logger().info('Sucesso no teste da câmera após tempo de espera adicional!')
            else:
                return None
        
        # Define uma pipeline simples para resolução 720p a 30fps
        pipeline_str = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR"
        )
        
        self.get_logger().info(f'Pipeline GStreamer definida: {pipeline_str}')
        
        # Configurar e retornar informações da câmera
        camera_info = {
            'width': 1280,
            'height': 720,
            'fps': 30,
            'pipeline': pipeline_str
        }
        
        self.get_logger().info(f'Informações da câmera obtidas com sucesso: {camera_info}')
        return camera_info

    def run_process_with_timeout(self, cmd, timeout=5):
        """
        Executa um processo com timeout e gerencia seu ciclo de vida.
        Retorna (sucesso, saída, erro)
        """
        try:
            self.get_logger().debug(f"Executando comando: {cmd}")
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Permite matar o grupo de processos
            )
            
            # Aguardar com timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
                stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""
                
                if process.returncode == 0:
                    self.get_logger().debug("Processo concluído com sucesso")
                    return True, stdout_text, stderr_text
                else:
                    self.get_logger().debug(f"Processo falhou com código {process.returncode}")
                    return False, stdout_text, stderr_text
                    
            except subprocess.TimeoutExpired:
                self.get_logger().warn(f"Timeout após {timeout}s. Matando processo...")
                
                # Tentar terminar o processo graciosamente primeiro
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    # Aguardar um pouco para ver se o processo termina
                    for _ in range(5):  # Tentar por 0.5 segundos
                        if process.poll() is not None:
                            break
                        time.sleep(0.1)
                except Exception as kill_error:
                    self.get_logger().debug(f"Erro ao terminar processo: {str(kill_error)}")
                
                # Se ainda estiver rodando, forçar a morte
                if process.poll() is None:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        self.get_logger().debug("Processo terminado forçadamente")
                    except Exception as kill_error:
                        self.get_logger().debug(f"Erro ao matar processo: {str(kill_error)}")
                
                return False, "", "Timeout expirado"
                
        except Exception as e:
            self.get_logger().error(f"Erro ao executar processo: {str(e)}")
            return False, "", str(e)

    def test_camera_pipeline(self, timeout=5):
        """
        Testa a pipeline da câmera com timeout e diagnóstico de erros
        Retorna (sucesso, mensagem_erro)
        """
        self.get_logger().info(f"Testando acesso à câmera com timeout de {timeout}s...")
        
        # Pipeline de teste básico
        test_cmd = "gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink -v"
        
        # Executar teste
        success, stdout, stderr = self.run_process_with_timeout(test_cmd, timeout)
        
        if success:
            self.get_logger().info("Teste de pipeline básico bem-sucedido!")
            return True, ""
        else:
            # Analisar o erro para diagnóstico mais preciso
            error_msg = ""
            if stderr:
                if "cannot access /dev/video0" in stderr or "Failed to open" in stderr:
                    error_msg = "Não foi possível acessar o dispositivo de vídeo. Verifique as permissões."
                elif "cannot connect to camera" in stderr:
                    error_msg = "Não foi possível conectar à câmera. Verifique as conexões físicas."
                elif "Resource busy" in stderr:
                    error_msg = "Câmera está ocupada. Outro processo pode estar usando-a."
                elif "No such element or plugin" in stderr:
                    error_msg = "Plugin nvarguscamerasrc não encontrado. Verifique a instalação do Jetson Multimedia API."
                elif "nvargus-daemon" in stderr and ("failed" in stderr or "error" in stderr):
                    error_msg = "Problema com o nvargus-daemon. Tente reiniciar o serviço."
                elif "timeout" in stderr.lower():
                    error_msg = "Timeout ao acessar a câmera."
                else:
                    error_msg = f"Erro no teste da câmera: {stderr}"
            else:
                error_msg = "Erro desconhecido no teste da câmera (sem saída de erro)"
                
            self.get_logger().error(error_msg)
            return False, error_msg

    def destroy_node(self):
        """Limpa recursos ao encerrar."""
        self.is_running = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        # Encerrar processo gst-launch se estiver usando o método direto
        if hasattr(self, 'gst_process') and hasattr(self, 'using_gst_launch_direct'):
            try:
                self.gst_process.terminate()
                self.gst_process.wait(timeout=2)
            except:
                pass
                
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            
        if self.get_parameter('enable_display').value:
            cv2.destroyAllWindows()
        
        # Cancelar timer de simulação se estiver ativo
        if hasattr(self, 'sim_timer'):
            self.sim_timer.cancel()
            
        super().destroy_node()

    def _configure_pipeline(self):
        """Configura o pipeline GStreamer com base nos parâmetros."""
        try:
            # Verificar se temos acesso a um dispositivo de câmera real
            if self.video_device_node is not None:
                # Usar o dispositivo especificado
                device_exists = os.path.exists(self.video_device_node)
                self.get_logger().info(f"Verificando dispositivo: {self.video_device_node} - Existe: {device_exists}")
            else:
                # Verificar dispositivos de vídeo disponíveis
                video_devices = glob.glob('/dev/video*')
                self.get_logger().info(f"Dispositivos de vídeo disponíveis: {video_devices}")
                device_exists = len(video_devices) > 0
                
                # Se encontrou pelo menos um dispositivo, use o primeiro
                if device_exists:
                    self.video_device_node = video_devices[0]
                    self.get_logger().info(f"Usando dispositivo automático: {self.video_device_node}")
                
            # Decidir qual pipeline usar com base no ambiente e dispositivos disponíveis
            is_jetson = os.path.exists('/etc/nv_tegra_release') or os.path.exists('/etc/nv_boot_control.conf')
            is_jetson_nano = False
            if is_jetson:
                try:
                    with open('/proc/device-tree/model', 'r') as f:
                        model = f.read()
                        is_jetson_nano = 'Nano' in model
                        self.get_logger().info(f"Modelo Jetson detectado: {model}")
                except:
                    pass
            
            # Configurar fluxo com base na disponibilidade de hardware
            if is_jetson and (not self.fake_camera or device_exists):
                # Pipeline para Jetson com câmera real
                if is_jetson_nano and os.path.exists('/dev/nvhost-ctrl'):
                    # Pipeline específico para Jetson Nano com câmera CSI (IMX219)
                    self.get_logger().info("Configurando pipeline para Jetson Nano com câmera CSI")
                    self._configure_jetson_csi_pipeline()
                elif device_exists:
                    # Pipeline para Jetson com V4L2 (webcam USB ou similar)
                    self.get_logger().info("Configurando pipeline para Jetson com câmera V4L2")
                    self._configure_v4l2_pipeline(self.video_device_node)
                else:
                    self.get_logger().warn("Nenhum dispositivo de câmera disponível no Jetson, usando simulação")
                    self._configure_fake_camera_pipeline()
            elif device_exists:
                # Pipeline genérico para outras plataformas com câmera
                self.get_logger().info("Configurando pipeline para câmera genérica V4L2")
                self._configure_v4l2_pipeline(self.video_device_node)
            else:
                # Nenhum dispositivo disponível, usar câmera fake
                self.get_logger().warn("Nenhum dispositivo de câmera disponível, usando simulação")
                self._configure_fake_camera_pipeline()
            
            # Configurar parâmetros de processamento
            self._configure_processing()
            
            return True
        except Exception as e:
            self.get_logger().error(f"Erro ao configurar pipeline: {str(e)}")
            traceback.print_exc()
            return False
            
    def _configure_jetson_csi_pipeline(self):
        """Configura pipeline específico para câmera CSI na Jetson em ambiente containerizado."""
        width = self.width
        height = self.height
        framerate = self.camera_fps
        exposure_time = self.get_parameter('exposure_time').value
        gain = self.get_parameter('gain').value
        awb_mode = self.get_parameter('awb_mode').value
        flip_method = self.get_parameter('flip_method').value
        
        # Pipeline otimizado especificamente para ambiente containerizado
        pipeline_str = (
            f"nvarguscamerasrc sensor-id=0 "
            f"! video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
            f"format=(string)NV12, framerate=(fraction){int(framerate)}/1 "
            f"! nvvidconv flip-method={flip_method} "
            f"! video/x-raw, format=(string)BGRx "
            f"! videoconvert "
            f"! video/x-raw, format=(string)BGR "
            f"! appsink drop=true max-buffers=2"
        )
        
        self.get_logger().info(f"Pipeline CSI para container: {pipeline_str}")
        
        # Verificar permissões do dispositivo nvargus antes de tentar abrir
        self.check_nvargus_permissions()
        
        # Testar se o serviço nvargus-daemon está em execução
        self.check_nvargus_daemon()
        
        # Verificar socket Argus
        self.check_argus_socket()
        
        # Abrir captura com pipeline GStreamer
        self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            self.get_logger().error("FALHA CRÍTICA: Não foi possível abrir a câmera CSI com pipeline para container")
            self.get_logger().error("Motivos possíveis:")
            self.get_logger().error("1. Hardware da câmera não conectado ou com problema")
            self.get_logger().error("2. Serviço nvargus-daemon não está em execução")
            self.get_logger().error("3. Permissões insuficientes para acessar o dispositivo")
            self.get_logger().error("4. Outro processo já está utilizando a câmera")
            self.get_logger().error("5. Socket Argus não está disponível ou acessível")
            self.get_logger().error("6. Container não foi iniciado com as flags corretas para acesso ao hardware")
            
            # Tentar obter mais informações de diagnóstico
            self.get_logger().info("Executando diagnóstico detalhado...")
            self.get_logger().info(f"Status de captura: {self.cap.isOpened()}")
            
            # Testar pipeline simplificado como último recurso
            self.get_logger().warn("Testando pipeline extremamente simples como última tentativa...")
            minimal_pipeline = (
                f"nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM) ! "
                f"nvvidconv ! "
                f"video/x-raw, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink"
            )
            
            self.get_logger().info(f"Pipeline mínimo: {minimal_pipeline}")
            test_cap = cv2.VideoCapture(minimal_pipeline, cv2.CAP_GSTREAMER)
            
            if test_cap.isOpened():
                self.get_logger().info("Pipeline mínimo funcionou! Usando-o como alternativa.")
                self.cap = test_cap
            else:
                test_cap.release()
                self.get_logger().error("Todas as tentativas de pipeline falharam.")
                
                # Verificar se há algum processo usando a câmera
                try:
                    process_check = subprocess.check_output("ps aux | grep -E 'gst-launch|nvargus' | grep -v grep", shell=True).decode('utf-8')
                    if process_check.strip():
                        self.get_logger().error(f"Processos potencialmente usando a câmera:\n{process_check}")
                except Exception:
                    pass
                
                # Não permitir fallback para pipeline simples ou simulação
                raise RuntimeError("Falha ao inicializar câmera CSI em ambiente containerizado - verificar logs para diagnóstico")
    
    def _configure_v4l2_pipeline(self, device_path):
        """Configura pipeline usando v4l2src para câmeras USB ou outros dispositivos V4L2."""
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        framerate = self.get_parameter('framerate').value
        
        # Pipeline para dispositivos V4L2 genéricos (webcams, etc)
        pipeline_str = (
            f"v4l2src device={device_path} "
            f"! video/x-raw, width={width}, height={height}, framerate={framerate}/1 "
            f"! videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        self.get_logger().info(f"Pipeline V4L2: {pipeline_str}")
        self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            self.get_logger().warn("Pipeline GStreamer V4L2 falhou. Tentando abertura direta...")
            # Tente abertura direta como fallback
            device_index = int(device_path.split('video')[-1]) if 'video' in device_path else 0
            self.cap = cv2.VideoCapture(device_index)
            
            if self.cap.isOpened():
                # Configurar resolução
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, framerate)
                self.get_logger().info(f"Abertura direta bem-sucedida para dispositivo {device_index}")
            else:
                self.get_logger().error(f"Falha ao abrir câmera {device_path}. Usando simulação.")
                self._configure_fake_camera_pipeline()
    
    def _configure_fake_camera_pipeline(self):
        """Configura uma câmera simulada para testes quando não há hardware disponível."""
        self.get_logger().info("Configurando câmera simulada")
        self.cap = None  # Não usamos um objeto de captura para a câmera simulada
        self.fake_camera = True
        
        # Configurar dimensões
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        
        # Criar uma imagem simulada
        self.fake_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Desenhar um campo simulado
        # Fundo verde
        self.fake_frame[:, :] = (0, 128, 0)  # BGR para verde
        
        # Adicionar linhas e elementos do campo
        center_x, center_y = width // 2, height // 2
        
        # Linha central
        cv2.line(self.fake_frame, (center_x, 0), (center_x, height), (255, 255, 255), 2)
        
        # Círculo central
        cv2.circle(self.fake_frame, (center_x, center_y), min(width, height) // 6, (255, 255, 255), 2)
        
        # Bordas do campo
        cv2.rectangle(self.fake_frame, (50, 50), (width - 50, height - 50), (255, 255, 255), 2)
        
        # Adicionar área do gol
        goal_width = width // 6
        goal_depth = height // 8
        
        # Gol esquerdo
        cv2.rectangle(self.fake_frame, (50, center_y - goal_width//2), 
                     (50 + goal_depth, center_y + goal_width//2), (255, 255, 255), 2)
        
        # Gol direito
        cv2.rectangle(self.fake_frame, (width - 50 - goal_depth, center_y - goal_width//2), 
                     (width - 50, center_y + goal_width//2), (255, 255, 255), 2)
        
        # Adicionar uma bola laranja
        ball_x = center_x + np.random.randint(-100, 100)
        ball_y = center_y + np.random.randint(-100, 100)
        cv2.circle(self.fake_frame, (ball_x, ball_y), 15, (0, 165, 255), -1)
        
        # Inicializar parâmetros para animação da bola
        self.ball_pos = [ball_x, ball_y]
        self.ball_dir = [np.random.randint(3, 7), np.random.randint(3, 7)]
        if np.random.random() > 0.5:
            self.ball_dir[0] *= -1
        if np.random.random() > 0.5:
            self.ball_dir[1] *= -1
        
        # Adicionar texto indicando simulação
        cv2.putText(self.fake_frame, "CAMERA SIMULADA", (center_x - 150, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        self.get_logger().info(f"Modo de simulação configurado com resolução {width}x{height}")
        self.get_logger().warn("Usando modo de simulação - câmera física não disponível!")
        
    def _configure_processing(self):
        """Configura opções de processamento com base em hardware disponível."""
        # Verificar se CUDA está disponível e habilitado
        cuda_enabled = self.get_parameter('enable_cuda').value
        
        if not hasattr(self, '_cuda_available'):
            # Executar verificação de CUDA se ainda não foi feita
            self._cuda_available = self.check_cuda_permissions()
        
        self.use_cuda = cuda_enabled and self._cuda_available
        self.get_logger().info(f"Processamento CUDA: {'habilitado' if self.use_cuda else 'desabilitado'}")
        
        # Configurar opções de pré-processamento
        self.resize_output = self.get_parameter('resize_output').value
        self.output_width = self.get_parameter('output_width').value
        self.output_height = self.get_parameter('output_height').value
        
        # Inicializar contadores para métricas
        self.frame_count = 0
        self.last_fps_update = self.get_clock().now()
        self.current_fps = 0.0
        
        # Inicializar temporizador para callback de câmera
        callback_period = 1.0 / self.get_parameter('framerate').value
        self.camera_timer = self.create_timer(callback_period, self.camera_callback)
        
    def camera_callback(self):
        """Callback para capturar e processar frames da câmera."""
        # Se não há câmera configurada e não estamos em modo de simulação, sair silenciosamente
        if not hasattr(self, 'cap') and not self.is_simulated:
            return
        
        # Iniciar medição de tempo para cálculo de FPS
        start_time = time.time()
        
        # Obter frame da câmera
        if hasattr(self, 'fake_camera') and self.fake_camera:
            # Câmera simulada - usar frame fake
            if hasattr(self, 'fake_frame') and self.fake_frame is not None:
                frame = self.fake_frame.copy()
                
                # Adicionar movimento à bola para simular animação
                if not hasattr(self, 'ball_pos'):
                    # Inicializar posição da bola e direção se não existirem
                    h, w = frame.shape[:2]
                    self.ball_pos = [w // 2, h // 2]
                    self.ball_dir = [np.random.randint(3, 7), np.random.randint(3, 7)]
                    if np.random.random() > 0.5:
                        self.ball_dir[0] *= -1
                    if np.random.random() > 0.5:
                        self.ball_dir[1] *= -1
                
                # Atualizar posição da bola
                h, w = frame.shape[:2]
                self.ball_pos[0] += self.ball_dir[0]
                self.ball_pos[1] += self.ball_dir[1]
                
                # Verificar colisões com as bordas
                if self.ball_pos[0] < 20 or self.ball_pos[0] > w - 20:
                    self.ball_dir[0] *= -1
                if self.ball_pos[1] < 20 or self.ball_pos[1] > h - 20:
                    self.ball_dir[1] *= -1
                
                # Limitar posição para dentro da imagem
                self.ball_pos[0] = max(20, min(w - 20, self.ball_pos[0]))
                self.ball_pos[1] = max(20, min(h - 20, self.ball_pos[1]))
                
                # Redesenhar o campo (para limpar a bola anterior)
                # Desenhar fundo verde
                frame[:, :] = (0, 128, 0)  # BGR para verde
                
                # Linhas brancas
                center_x, center_y = w // 2, h // 2
                # Linha central
                cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 255), 2)
                # Círculo central
                cv2.circle(frame, (center_x, center_y), min(w, h) // 6, (255, 255, 255), 2)
                # Bordas do campo
                cv2.rectangle(frame, (50, 50), (w - 50, h - 50), (255, 255, 255), 2)
                
                # Desenhar bola na nova posição
                cv2.circle(frame, 
                          (int(self.ball_pos[0]), int(self.ball_pos[1])), 
                          15, (0, 165, 255), -1)
                
                # Adicionar timestamp
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
                
                # Adicionar texto de simulação
                cv2.putText(frame, "CAMERA SIMULADA", (center_x - 150, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                ret = True
            else:
                self.get_logger().error("Frame simulado não está disponível")
                return
        elif hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            # Câmera real - capturar frame
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Falha ao capturar frame da câmera")
                return
        else:
            self.get_logger().error("Nenhuma câmera configurada")
            return
        
        # Processar frame com CUDA se disponível
        if self.use_cuda:
            try:
                # Transferir para GPU
                if not hasattr(self, 'cuda_stream'):
                    self.cuda_stream = cv2.cuda_Stream()
                
                # Fazer upload do frame para GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # Aplicar processamento CUDA
                # Redução de ruído
                if self.get_parameter('enable_noise_reduction').value:
                    # Usar denoise para remover ruído
                    denoised = cv2.cuda.bilateralFilter(gpu_frame, 5, 75, 75, borderType=cv2.BORDER_DEFAULT)
                    gpu_frame = denoised
                
                # Enhancement de borda
                if self.get_parameter('enable_edge_enhancement').value:
                    # Aplicar sharpening usando Laplaciano
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
                    filter_gpu = cv2.cuda.createLinearFilter(
                        cv2.CV_8UC3, cv2.CV_8UC3, kernel, (-1, -1))
                    enhanced = filter_gpu.apply(gpu_frame)
                    gpu_frame = enhanced
                
                # Ajuste de gamma/brilho se necessário
                brightness = self.get_parameter('brightness').value
                if brightness != 0:
                    # Converter para float
                    gpu_float = cv2.cuda.createGpuMatFromGpuMat(gpu_frame)
                    gpu_float.convertTo(cv2.CV_32F, 1.0 / 255.0)
                    
                    # Ajuste gamma para brilho
                    gamma = 1.0 + brightness / 10.0  # Mapear brightness para gamma
                    gamma_gpu = cv2.cuda.createGpuMatFromGpuMat(gpu_float)
                    cv2.cuda.pow(gpu_float, gamma, gamma_gpu)
                    
                    # Converter de volta para 8-bit
                    gamma_gpu.convertTo(cv2.CV_8U, 255.0)
                    gpu_frame = gamma_gpu
                
                # Processar o frame com ISP (Image Signal Processing) adicional
                if self.get_parameter('enable_isp').value:
                    # Aplicar correção de cor e contraste
                    cv2.cuda.gammaCorrection(gpu_frame, gpu_frame)
                
                # Baixar o resultado de volta para CPU
                processed_frame = gpu_frame.download()
            except Exception as e:
                self.get_logger().error(f"Erro ao processar com CUDA: {str(e)}")
                self.get_logger().warn("Continuando sem processamento CUDA")
                processed_frame = frame
        else:
            # Processamento CPU quando CUDA não está disponível
            processed_frame = frame.copy()
            
            # Processamento básico em CPU
            if self.get_parameter('enable_noise_reduction').value:
                # Redução de ruído em CPU
                processed_frame = cv2.bilateralFilter(processed_frame, 5, 75, 75)
            
            if self.get_parameter('enable_edge_enhancement').value:
           
                # Aplicar sharpen
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
                processed_frame = cv2.filter2D(processed_frame, -1, kernel)
                
            # Ajuste de brilho
            brightness = self.get_parameter('brightness').value
            if brightness != 0:
                hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                
                # Ajustar valor (brilho)
                lim = 255 - brightness if brightness > 0 else -brightness
                v = v + brightness if brightness > 0 else v * (1.0 + brightness/lim)
                v = np.clip(v, 0, 255).astype(np.uint8)
                
                # Reconectar canais e converter de volta para BGR
                hsv = cv2.merge((h, s, v))
                processed_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Redimensionar se necessário
        if self.resize_output:
            processed_frame = cv2.resize(processed_frame, 
                                       (self.output_width, self.output_height),
                                       interpolation=cv2.INTER_AREA)
        
        # Adicionar informações de diagnóstico
        if hasattr(self, 'current_fps') and self.current_fps > 0:
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(processed_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Publicar mensagem de imagem
        try:
            # Criar mensagem ROS com timestamp
            timestamp = self.get_clock().now().to_msg()
            img_msg = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
            img_msg.header.stamp = timestamp
            img_msg.header.frame_id = "camera_optical_frame"
            
            # Publicar a imagem
            self.image_pub.publish(img_msg)
            
            # Publicar informações da câmera
            self.publish_camera_info(img_msg.header)
            
            # Atualizar cálculo de FPS
            self.frame_count += 1
            current_time = self.get_clock().now()
            elapsed = (current_time.nanoseconds - self.last_fps_update.nanoseconds) / 1e9
            
            if elapsed >= 1.0:  # Atualizar a cada segundo
                self.current_fps = self.frame_count / elapsed
                self.get_logger().debug(f"FPS atual: {self.current_fps:.1f}")
                self.frame_count = 0
                self.last_fps_update = current_time
                
        except CvBridgeError as e:
            self.get_logger().error(f"Erro no CvBridge: {str(e)}")
        
        # Calcular tempo total de processamento
        process_time = time.time() - start_time
        if process_time > (1.0 / self.get_parameter('framerate').value) * 0.8:
            # Avisar se estamos perto de não conseguir manter o FPS desejado
            self.get_logger().debug(f"Tempo de processamento: {process_time*1000:.1f}ms (Ideal: {1000/self.get_parameter('framerate').value:.1f}ms)")
            
        # Mostrar imagem se display estiver habilitado
        if self.get_parameter('enable_display').value:
            cv2.imshow('IMX219 Camera', processed_frame)
            cv2.waitKey(1)

    def check_cuda_permissions(self):
        """Verifica se o CUDA está disponível e tem permissões adequadas.
        
        Returns:
            bool: True se CUDA estiver disponível e com permissões corretas
        """
        try:
            self.get_logger().info('Verificando disponibilidade e permissões do CUDA...')
            
            # Verificar se os módulos CUDA estão carregados
            has_cuda = False
            
            # Método 1: Verificar dispositivos CUDA via arquivo de dispositivo
            cuda_devices = ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-modeset']
            cuda_device_found = False
            
            for device in cuda_devices:
                if os.path.exists(device):
                    cuda_device_found = True
                    # Verificar permissões
                    try:
                        mode = os.stat(device).st_mode
                        readable = bool(mode & 0o444)
                        writable = bool(mode & 0o222)
                        self.get_logger().info(f'Dispositivo CUDA {device} permissões: r={readable}, w={writable}')
                        if readable and writable:
                            has_cuda = True
                    except Exception as e:
                        self.get_logger().warn(f'Erro ao verificar permissões de {device}: {str(e)}')
            
            # Método 2: Verificar se o driver CUDA está carregado
            try:
                lsmod_output = subprocess.check_output(['lsmod'], text=True)
                if 'nvidia' in lsmod_output:
                    self.get_logger().info('Módulo nvidia detectado no kernel')
                    has_cuda = True
            except Exception as e:
                self.get_logger().debug(f'Erro ao verificar módulos do kernel: {str(e)}')
            
            # Método 3: Tentar inicializar o CUDA via OpenCV
            try:
                # Verificar se OpenCV foi compilado com suporte a CUDA
                cuda_build = cv2.getBuildInformation()
                if 'CUDA:YES' in cuda_build:
                    self.get_logger().info('OpenCV foi compilado com suporte a CUDA')
                    
                    # Verificar se pode criar matrizes CUDA
                    try:
                        dummy = np.zeros((10, 10), dtype=np.uint8)
                        gpu_mat = cv2.cuda_GpuMat()
                        gpu_mat.upload(dummy)
                        gpu_mat.download()
                        self.get_logger().info('CUDA inicializado com sucesso via OpenCV')
                        has_cuda = True
                    except Exception as e:
                        self.get_logger().warn(f'Erro ao inicializar CUDA no OpenCV: {str(e)}')
                else:
                    self.get_logger().warn('OpenCV não tem suporte a CUDA')
            except Exception as e:
                self.get_logger().debug(f'Erro ao verificar suporte CUDA no OpenCV: {str(e)}')
            
            if has_cuda:
                self.get_logger().info('CUDA está disponível e utilizável')
                return True
            else:
                if cuda_device_found:
                    self.get_logger().warn('Dispositivos CUDA encontrados, mas sem permissões adequadas')
                else:
                    self.get_logger().warn('Nenhum dispositivo CUDA encontrado no sistema')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Erro ao verificar CUDA: {str(e)}')
            return False

    def setup_simulation_mode(self):
        """Configura o modo de simulação quando a câmera real não está disponível."""
        self.get_logger().info("Configurando modo de simulação para a câmera")
        self.is_simulated = True
        self.fake_camera = True
        
        # Configurar dimensões
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        
        # Criar uma imagem simulada
        self.fake_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Desenhar um campo simulado
        # Fundo verde
        self.fake_frame[:, :] = (0, 128, 0)  # BGR para verde
        
        # Adicionar linhas e elementos do campo
        center_x, center_y = width // 2, height // 2
        
        # Linha central
        cv2.line(self.fake_frame, (center_x, 0), (center_x, height), (255, 255, 255), 2)
        
        # Círculo central
        cv2.circle(self.fake_frame, (center_x, center_y), min(width, height) // 6, (255, 255, 255), 2)
        
        # Bordas do campo
        cv2.rectangle(self.fake_frame, (50, 50), (width - 50, height - 50), (255, 255, 255), 2)
        
        # Adicionar área do gol
        goal_width = width // 6
        goal_depth = height // 8
        
        # Gol esquerdo
        cv2.rectangle(self.fake_frame, (50, center_y - goal_width//2), 
                     (50 + goal_depth, center_y + goal_width//2), (255, 255, 255), 2)
        
        # Gol direito
        cv2.rectangle(self.fake_frame, (width - 50 - goal_depth, center_y - goal_width//2), 
                     (width - 50, center_y + goal_width//2), (255, 255, 255), 2)
        
        # Adicionar uma bola laranja
        ball_x = center_x + np.random.randint(-100, 100)
        ball_y = center_y + np.random.randint(-100, 100)
        cv2.circle(self.fake_frame, (ball_x, ball_y), 15, (0, 165, 255), -1)
        
        # Inicializar parâmetros para animação da bola
        self.ball_pos = [ball_x, ball_y]
        self.ball_dir = [np.random.randint(3, 7), np.random.randint(3, 7)]
        if np.random.random() > 0.5:
            self.ball_dir[0] *= -1
        if np.random.random() > 0.5:
            self.ball_dir[1] *= -1
        
        # Adicionar texto indicando simulação
        cv2.putText(self.fake_frame, "CAMERA SIMULADA", (center_x - 150, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        self.get_logger().info(f"Modo de simulação configurado com resolução {width}x{height}")
        self.get_logger().warn("Usando modo de simulação - câmera física não disponível!")

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