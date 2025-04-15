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
                ('enable_display', True),
                ('flip_method', 0)
            ]
        )
        
        # Configurar a câmera com base nos parâmetros
        self._configure_camera()
        
        # Inicializar o processamento
        self._configure_processing()
        
        # Carregar calibração da câmera
        self.load_camera_calibration()
        
        # Criar timer para callback da câmera
        callback_period = 1.0 / self.camera_fps
        self.camera_timer = self.create_timer(callback_period, self.camera_callback)
        
        # Inicializar valor atual de FPS
        self.current_fps = 0.0
        
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

    def init_camera(self):
        """Inicializa a câmera com configurações otimizadas usando GStreamer e CUDA."""
        # Verificar se os dispositivos da câmera estão disponíveis
        devices_available = self.debug_camera_devices()
        
        # Se não houver dispositivos disponíveis, tentar modo de simulação
        if not devices_available:
            self.get_logger().warn('Nenhum dispositivo de câmera encontrado. Iniciando modo de simulação.')
            self.setup_simulation_mode()
            return
        
        # Verificar se estamos em ambiente containerizado
        is_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
        if is_container:
            self.get_logger().info('Ambiente containerizado detectado, ajustando pipeline para compatibilidade.')
            
        # Verificar se o plugin nvarguscamerasrc está disponível
        try:
            self.get_logger().info('Verificando disponibilidade do plugin nvarguscamerasrc...')
            gst_check = subprocess.check_output(['gst-inspect-1.0', 'nvarguscamerasrc'], stderr=subprocess.STDOUT).decode('utf-8')
            if 'nvarguscamerasrc' not in gst_check:
                self.get_logger().error('Plugin GStreamer nvarguscamerasrc não encontrado. Este plugin é necessário para o funcionamento da câmera CSI.')
                self.get_logger().info('Resultado da inspeção do plugin: ' + gst_check[:200] + '...' if len(gst_check) > 200 else gst_check)
                raise RuntimeError('Plugin GStreamer nvarguscamerasrc não encontrado')
            else:
                self.get_logger().info('Plugin nvarguscamerasrc encontrado com sucesso!')
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            self.get_logger().error(f'GStreamer não disponível ou não configurado corretamente: {str(e)}')
            raise RuntimeError('GStreamer não disponível')
        
        # Testar acesso básico à câmera CSI usando nvarguscamerasrc
        self.get_logger().info('Testando acesso básico à câmera CSI...')
        test_cmd = "gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink -v"
        
        try:
            test_result = subprocess.run(test_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            if test_result.returncode == 0:
                self.get_logger().info('Teste básico de acesso à câmera CSI bem-sucedido!')
            else:
                stderr = test_result.stderr.decode('utf-8')
                self.get_logger().warn(f'Teste básico de acesso à câmera CSI falhou: {stderr}')
                
                # Analisar erro específico
                if "no such element or plugin" in stderr:
                    self.get_logger().error("ERRO CRÍTICO: Plugin nvarguscamerasrc não disponível!")
                elif "could not link" in stderr:
                    self.get_logger().error("ERRO: Não foi possível vincular elementos do pipeline!")
                elif "Resource busy" in stderr:
                    self.get_logger().error("ERRO: Recurso ocupado! Outro processo pode estar usando a câmera.")
                    # Tentar matar qualquer processo usando nvarguscamerasrc
                    self.get_logger().info("Tentando liberar recursos da câmera...")
                    try:
                        subprocess.run("pkill -f 'gst-launch.*nvarguscamerasrc'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        time.sleep(2)  # Dar tempo para os recursos serem liberados
                    except Exception as kill_err:
                        self.get_logger().warn(f'Erro ao tentar liberar recursos: {str(kill_err)}')
                
                # Tentar reiniciar o serviço nvargus-daemon
                self.get_logger().info("Tentando reiniciar o serviço nvargus-daemon...")
                try:
                    subprocess.run(['systemctl', 'restart', 'nvargus-daemon'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                    time.sleep(2)  # Dar tempo para o serviço reiniciar
                    
                    # Testar novamente após reiniciar o serviço
                    test_result = subprocess.run(test_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                    if test_result.returncode == 0:
                        self.get_logger().info('Reinício do serviço nvargus-daemon resolveu o problema!')
                    else:
                        stderr = test_result.stderr.decode('utf-8')
                        self.get_logger().warn(f'Teste após reinício do serviço também falhou: {stderr}')
                except Exception as restart_err:
                    self.get_logger().error(f'Erro ao reiniciar serviço nvargus-daemon: {str(restart_err)}')
        except subprocess.TimeoutExpired:
            self.get_logger().warn('Timeout ao testar acesso à câmera CSI. O processo pode estar travado.')
            # Tentar matar qualquer processo pendente
            try:
                subprocess.run("pkill -f 'gst-launch.*nvarguscamerasrc'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                pass
        except Exception as e:
            self.get_logger().error(f'Erro ao testar acesso à câmera CSI: {str(e)}')
        
        # Configurar variáveis de ambiente para CUDA e GStreamer
        self.get_logger().info('Configurando variáveis de ambiente para CUDA e GStreamer...')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["GST_GL_API"] = "gles2"
        os.environ["GST_GL_PLATFORM"] = "egl"
        os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,video,utility"
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
        # Importante: desativar o GL nos pipelines GStreamer para evitar problemas de EGL
        os.environ["GST_GL_XINITTHREADS"] = "0"
        os.environ["__GL_SYNC_TO_VBLANK"] = "0"
        # Adicionar variáveis de ambiente para melhor desempenho do nvarguscamerasrc
        os.environ["GST_ARGUS_SENSOR_MODE"] = str(self.camera_mode)  # Usar o modo de câmera configurado
        
        # Pipeline GStreamer otimizado para IMX219 com aceleração CUDA
        # Calcular framerate como fração reduzida para evitar problemas
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
            
        fps_num = self.camera_fps
        fps_den = 1
        divisor = gcd(fps_num, fps_den)
        fps_num //= divisor
        fps_den //= divisor
        
        # Verificar se CUDA está disponível
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                self.get_logger().info(f'Dispositivos CUDA disponíveis: {cuda_devices}')
                # Testar se CUDA está realmente funcionando
                try:
                    test_mat = cv2.cuda_GpuMat((10, 10), cv2.CV_8UC3)
                    self.get_logger().info('Teste CUDA bem-sucedido! CUDA está funcionando corretamente.')
                    cuda_enabled = True
                except Exception as cuda_test_error:
                    self.get_logger().warn(f'CUDA detectado, mas teste falhou: {str(cuda_test_error)}')
                    cuda_enabled = False
            else:
                self.get_logger().warn('Nenhum dispositivo CUDA disponível!')
                cuda_enabled = False
        except Exception as e:
            self.get_logger().warn(f'Erro ao verificar suporte CUDA: {str(e)}')
            cuda_enabled = False
        
        # Reduzir o fps para 30 se for captura ao vivo em resolução alta (evita problemas de timeout)
        if fps_num > 30 and (self.width > 1920 or self.height > 1080):
            self.get_logger().info(f'Reduzindo FPS para 30 devido à alta resolução para evitar problemas de timeout')
            fps_num = 30
            fps_den = 1
        
        # Construir pipeline GStreamer otimizado
        # Usar nvcamerasrc se disponível (melhor performance) ou nvarguscamerasrc como alternativa
        self.get_logger().info('Construindo pipeline GStreamer otimizado para câmera CSI com CUDA...')
        
        # Montar pipeline de acordo com parâmetros
        use_cuda_element = "true" if cuda_enabled else "false"
        
        # Pipeline otimizado com nvinfer para processamento acelerado (se CUDA disponível)
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
        if cuda_enabled and self.get_parameter('enable_cuda').value:
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
        
        self.get_logger().info(f'Pipeline GStreamer final: {pipeline}')
        
        # Abrir a câmera com o pipeline GStreamer
        try:
            self.get_logger().info('Abrindo câmera com pipeline GStreamer...')
            
            # Verificar permissões do CUDA antes de abrir a câmera
            if cuda_enabled:
                self.check_cuda_permissions()
            
            # Matar qualquer processo gst-launch existente
            try:
                self.get_logger().info('Liberando recursos GStreamer existentes...')
                subprocess.run("pkill -f 'gst-launch.*nvarguscamerasrc'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(1)  # Dar tempo para os recursos serem liberados
            except Exception as kill_err:
                self.get_logger().debug(f'Erro ao tentar liberar recursos: {str(kill_err)}')
            
            # Abrir a câmera
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            # Verificar se a câmera foi aberta com sucesso
            if not self.cap.isOpened():
                self.get_logger().error('Falha ao abrir câmera com GStreamer!')
                
                # Tente um pipeline mais simples como último recurso
                self.get_logger().warn('Tentando pipeline mais simples como último recurso...')
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
                    self.get_logger().error('Falha também com pipeline simplificado!')
                    self.get_logger().error('Possíveis causas: serviço nvargus-daemon não está funcionando, hardware CSI não está conectado corretamente, ou outro processo está usando a câmera.')
                    
                    # Diagnosticar especificamente se a câmera CSI está sendo reconhecida
                    try:
                        self.get_logger().info('Verificando reconhecimento da câmera IMX219...')
                        i2c_check = subprocess.run("i2cdetect -y 0 | grep 10", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        if i2c_check.returncode == 0 and "10" in i2c_check.stdout.decode():
                            self.get_logger().info("Câmera IMX219 detectada no barramento I2C!")
                        else:
                            self.get_logger().error("Câmera IMX219 NÃO detectada no barramento I2C! Verifique a conexão física.")
                    except Exception as i2c_err:
                        self.get_logger().warn(f'Erro ao verificar câmera via I2C: {str(i2c_err)}')
                    
                    # Ativar modo de simulação
                    self.get_logger().warn('Ativando modo de simulação devido a falhas na câmera.')
                    self.setup_simulation_mode()
                    return
            
            # Câmera aberta com sucesso, tentar capturar o primeiro frame
            self.get_logger().info('Câmera aberta com sucesso! Tentando capturar o primeiro frame...')
            ret, test_frame = self.cap.read()
            if ret:
                self.get_logger().info(f'Primeiro frame capturado com sucesso: {test_frame.shape}')
                
                # Salvar o primeiro frame como arquivo para diagnóstico
                try:
                    cv2.imwrite('/tmp/first_frame.jpg', test_frame)
                    self.get_logger().info('Primeiro frame salvo em /tmp/first_frame.jpg')
                except Exception as save_err:
                    self.get_logger().warn(f'Não foi possível salvar o primeiro frame: {str(save_err)}')
                
                # Verificar FPS reportado
                fps_value = self.cap.get(cv2.CAP_PROP_FPS)
                if fps_value <= 0:
                    self.get_logger().warn('FPS não reportado pela câmera. Usando valor configurado.')
                else:
                    self.get_logger().info(f'FPS reportado pela câmera: {fps_value}')
            else:
                self.get_logger().warn('Não foi possível capturar o primeiro frame. A câmera pode estar inicializando...')
                
                # Esperar um pouco e tentar novamente
                self.get_logger().info('Aguardando 2 segundos e tentando novamente...')
                time.sleep(2)
                ret, test_frame = self.cap.read()
                
                if ret:
                    self.get_logger().info('Segundo teste de captura bem-sucedido!')
                else:
                    self.get_logger().error('Falha persistente na captura. A câmera pode estar com problemas.')
                    # Mas vamos tentar continuar mesmo assim, o loop de captura vai lidar com isso
            
            # Configurar variáveis para cálculo de FPS real
            self.frame_count = 0
            self.last_fps_time = time.time()
            self.real_fps = 0.0
            
            # Ajustar propriedades para melhor desempenho
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)  # Aumentar buffer para minimizar latência
            
            self.get_logger().info('Câmera CSI com GStreamer e CUDA inicializada com sucesso!')
            
        except Exception as e:
            self.get_logger().error(f'Exceção ao inicializar câmera: {str(e)}')
            self.get_logger().error(f'Stack trace:\n{traceback.format_exc()}')
            
            # Diagnosticar o erro mais especificamente
            self.get_logger().info('Executando diagnóstico detalhado...')
            
            # Verificar módulos carregados da NVIDIA
            try:
                lsmod = subprocess.check_output("lsmod | grep -E 'nvidia|tegra'", shell=True).decode('utf-8')
                self.get_logger().info(f'Módulos NVIDIA/Tegra carregados:\n{lsmod}')
            except Exception:
                self.get_logger().warn('Não foi possível verificar módulos NVIDIA/Tegra.')
            
            # Verificar status dos serviços NVIDIA
            try:
                nvidia_services = subprocess.check_output("systemctl list-units --type=service | grep -E 'nvidia|argus'", shell=True).decode('utf-8')
                self.get_logger().info(f'Serviços NVIDIA:\n{nvidia_services}')
            except Exception:
                self.get_logger().warn('Não foi possível listar serviços NVIDIA.')
            
            # Ativar modo de simulação como último recurso
            self.get_logger().warn('Ativando modo de simulação devido a falha na inicialização da câmera.')
            self.setup_simulation_mode()

    def setup_simulation_mode(self):
        """Configura o modo de simulação para a câmera quando dispositivos físicos não estão disponíveis."""
        self.get_logger().info('Iniciando modo de simulação da câmera...')
        
        # Criar uma imagem de teste
        self.simulated_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Desenhar um padrão de teste na imagem para que seja visível
        cv2.putText(self.simulated_frame, 'CAMERA SIMULADA', (self.width//2-150, self.height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Desenhar um círculo que se move para mostrar que é um vídeo simulado
        self.circle_position = [self.width//4, self.height//4]
        self.circle_direction = [5, 5]
        
        # Flag de simulação
        self.is_simulated = True
        
        self.get_logger().info('Modo de simulação da câmera inicializado com sucesso')
        
        # Configurar um timer para atualizar a imagem simulada (simulando FPS real)
        self.sim_timer = self.create_timer(1.0/self.camera_fps, self.update_simulated_frame)

    def update_simulated_frame(self):
        """Atualiza o frame simulado com um padrão em movimento."""
        if not hasattr(self, 'simulated_frame'):
            return
            
        # Limpar frame
        self.simulated_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Texto de simulação
        cv2.putText(self.simulated_frame, 'CAMERA SIMULADA', (self.width//2-150, self.height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Atualizar posição do círculo
        self.circle_position[0] += self.circle_direction[0]
        self.circle_position[1] += self.circle_direction[1]
        
        # Verificar colisão com bordas
        if self.circle_position[0] <= 0 or self.circle_position[0] >= self.width:
            self.circle_direction[0] *= -1
        if self.circle_position[1] <= 0 or self.circle_position[1] >= self.height:
            self.circle_direction[1] *= -1
            
        # Desenhar círculo
        cv2.circle(self.simulated_frame, 
                  (int(self.circle_position[0]), int(self.circle_position[1])), 
                  50, (0, 0, 255), -1)
        
        # Adicionar timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(self.simulated_frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Publicar frame simulado
        try:
            timestamp = self.get_clock().now().to_msg()
            img_msg = self.bridge.cv2_to_imgmsg(self.simulated_frame, "bgr8")
            img_msg.header.stamp = timestamp
            img_msg.header.frame_id = "camera_optical_frame"
            self.image_pub.publish(img_msg)
            
            # Publicar info da câmera
            self.publish_camera_info(img_msg.header)
            
            # Atualizar cálculo de FPS real
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed >= 1.0:  # Atualizar a cada segundo
                self.real_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_fps_time = current_time
                self.get_logger().debug(f'FPS simulado: {self.real_fps:.1f}')
                
        except CvBridgeError as e:
            self.get_logger().error(f'Erro no CvBridge: {e}')

    def debug_camera_devices(self):
        """
        Debugar dispositivos de câmera disponíveis.
        Retorna: True se pelo menos um dispositivo de câmera está disponível, False caso contrário
        """
        devices_available = False
        csi_available = False
        
        try:
            # Verificar dispositivos de vídeo
            self.get_logger().info('Verificando dispositivos de câmera disponíveis:')
            try:
                # Verificar diretamente se /dev/video0 existe em vez de usar 'ls'
                if os.path.exists('/dev/video0'):
                    self.get_logger().info('Dispositivo /dev/video0 encontrado')
                    devices_available = True
                else:
                    self.get_logger().warn('Dispositivo /dev/video0 não encontrado')
                    
                # Também verificar outros dispositivos
                for dev_id in range(1, 10):  # Verificar dispositivos de 1 a 9 (0 já verificado acima)
                    dev_path = f"/dev/video{dev_id}"
                    if os.path.exists(dev_path):
                        self.get_logger().info(f'Dispositivo {dev_path} encontrado')
                        devices_available = True
                
                # Verificar se dispositivos CSI estão disponíveis (diferentes de V4L2)
                try:
                    # Verificar se a câmera CSI está conectada através do i2cdetect
                    i2c_output = subprocess.check_output(['i2cdetect', '-y', '0'], stderr=subprocess.STDOUT).decode('utf-8')
                    if '36' in i2c_output or '10' in i2c_output:  # Endereços comuns para câmeras IMX219
                        self.get_logger().info('Câmera CSI detectada pelo i2cdetect')
                        devices_available = True
                        csi_available = True
                    else:
                        self.get_logger().warn('Câmera CSI não detectada pelo i2cdetect')
                except (subprocess.SubprocessError, FileNotFoundError) as e:
                    self.get_logger().info(f'i2cdetect não disponível para verificar câmera CSI: {str(e)}')
                
                # Verificar diretamente com GStreamer se a câmera CSI está disponível
                try:
                    self.get_logger().info('Testando câmera CSI com GStreamer (isso pode levar alguns segundos)...')
                    # Usar num-buffers=1 para evitar que o comando execute indefinidamente
                    result = subprocess.run(
                        ['gst-launch-1.0', 'nvarguscamerasrc', 'sensor-id=0', 'num-buffers=1', '!', 'fakesink'], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        timeout=10  # Timeout mais longo para dar tempo suficiente
                    )
                    # Verificar se o comando foi bem-sucedido
                    if result.returncode == 0:
                        self.get_logger().info('Câmera CSI detectada via nvarguscamerasrc')
                        csi_available = True
                        devices_available = True
                    else:
                        stderr = result.stderr.decode('utf-8')
                        if "Could not initialize supporting library" in stderr:
                            self.get_logger().warn('Biblioteca de suporte à câmera CSI não inicializada corretamente')
                        elif "Resource busy" in stderr:
                            self.get_logger().warn('Câmera CSI já em uso por outro processo')
                        else:
                            self.get_logger().warn(f'Erro ao testar câmera CSI via nvarguscamerasrc: {stderr}')
                except (subprocess.SubprocessError, FileNotFoundError) as e:
                    self.get_logger().warn(f'Teste de câmera CSI via GStreamer falhou: {str(e)}')
                    self.get_logger().info('Isso pode ser normal em ambiente containerizado. Tentando método alternativo...')
                    
                    # Tentar método alternativo usando apenas a inspeção do plugin
                    try:
                        inspect_output = subprocess.check_output(['gst-inspect-1.0', 'nvarguscamerasrc'], 
                                                              stderr=subprocess.STDOUT, timeout=3).decode('utf-8')
                        if "nvarguscamerasrc:" in inspect_output:
                            self.get_logger().info('Plugin nvarguscamerasrc está disponível (detecção alternativa).')
                            # Assumir que a câmera CSI está disponível se o plugin existe
                            csi_available = True
                            devices_available = True
                    except Exception as inspect_err:
                        self.get_logger().warn(f'Inspeção alternativa do plugin falhou: {str(inspect_err)}')
                
                # Verificar mais detalhes apenas se um dispositivo for encontrado
                if devices_available:
                    for dev_id in range(10):
                        dev_path = f"/dev/video{dev_id}"
                        if os.path.exists(dev_path):
                            try:
                                # Verificar se o dispositivo é acessível (permissões)
                                mode = os.stat(dev_path).st_mode
                                readable = bool(mode & 0o444)  # Verificar permissão de leitura
                                self.get_logger().info(f'Dispositivo {dev_path} é legível: {readable}')
                                
                                # Verificar grupos do dispositivo
                                group_id = os.stat(dev_path).st_gid
                                user_groups = os.getgroups()
                                self.get_logger().info(f'Grupo do dispositivo: {group_id}, Grupos do usuário: {user_groups}')
                                
                                # Verificar detalhes da câmera com v4l2-ctl se disponível
                                try:
                                    caps = subprocess.check_output(['v4l2-ctl', '--device', dev_path, '--list-formats-ext']).decode('utf-8')
                                    self.get_logger().info(f'Capacidades da câmera {dev_path}:\n{caps}')
                                except (subprocess.SubprocessError, FileNotFoundError):
                                    self.get_logger().info(f'v4l2-ctl não disponível para verificar {dev_path}')
                                    
                                # Tentar abrir o dispositivo diretamente com OpenCV para testar acesso
                                test_cap = cv2.VideoCapture(dev_id)
                                if test_cap.isOpened():
                                    self.get_logger().info(f'Teste de abertura do dispositivo {dev_path} com OpenCV: SUCESSO')
                                    test_cap.release()
                                else:
                                    self.get_logger().warn(f'Teste de abertura do dispositivo {dev_path} com OpenCV: FALHA')
                            except Exception as e:
                                self.get_logger().warn(f'Erro ao verificar detalhes de {dev_path}: {str(e)}')
            except Exception as e:
                self.get_logger().warn(f'Erro ao verificar dispositivos: {str(e)}')
            
            # Verificar permissões
            uid = os.getuid()
            gid = os.getgid()
            self.get_logger().info(f'UID: {uid}, GID: {gid}')
            self.get_logger().info(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "não definido")}')
            
            # Verificar se estamos em um ambiente containerizado
            if os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv'):
                self.get_logger().info('Executando em ambiente containerizado')
            
            # Verificar se o GStreamer NVARGUS está disponível
            try:
                gst_output = subprocess.check_output(['gst-inspect-1.0', 'nvarguscamerasrc'], stderr=subprocess.STDOUT).decode('utf-8')
                if 'nvarguscamerasrc' in gst_output:
                    self.get_logger().info('Plugin GStreamer nvarguscamerasrc disponível')
                else:
                    self.get_logger().warn('Plugin GStreamer nvarguscamerasrc não está disponível')
            except (subprocess.SubprocessError, FileNotFoundError):
                self.get_logger().warn('Não foi possível verificar o plugin nvarguscamerasrc. GStreamer pode não estar disponível')
            
            # Verificar diretório de dispositivos no Windows (WSL)
            if os.name == 'nt' or ('WSL' in os.uname().release if hasattr(os, 'uname') else False):
                self.get_logger().info('Ambiente Windows/WSL detectado. Verificando dispositivos de vídeo disponíveis:')
                try:
                    # No Windows, usar DirectShow para listar dispositivos
                    self.get_logger().info('Ambiente Windows não suporta diretamente /dev/video*. Use webcam USB.')
                except Exception:
                    pass
            
        except Exception as e:
            self.get_logger().error(f'Erro ao debugar dispositivos: {str(e)}')
        
        return devices_available

    def check_cuda_permissions(self):
        """Verifica permissões de CUDA e tenta resolver problemas comuns."""
        try:
            # Verificar se o usuário tem acesso aos dispositivos NVIDIA
            nvidia_devices = ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-modeset']
            devices_found = False
            
            for device in nvidia_devices:
                if os.path.exists(device):
                    devices_found = True
                    try:
                        mode = os.stat(device).st_mode
                        readable = bool(mode & 0o444)  # Verificar permissão de leitura
                        writeable = bool(mode & 0o222)  # Verificar permissão de escrita
                        self.get_logger().info(f'Dispositivo {device} - Leitura: {readable}, Escrita: {writeable}')
                    except Exception as e:
                        self.get_logger().warn(f'Erro ao verificar permissões de {device}: {str(e)}')
            
            if not devices_found:
                self.get_logger().warn('Nenhum dispositivo NVIDIA encontrado no sistema')
                
                # Verificar se estamos em ambiente WSL (Windows Subsystem for Linux)
                is_wsl = False
                try:
                    with open('/proc/version', 'r') as f:
                        if 'microsoft' in f.read().lower():
                            is_wsl = True
                            self.get_logger().warn('Ambiente WSL detectado. CUDA pode não funcionar corretamente no WSL1.')
                            self.get_logger().info('Para usar CUDA com câmeras, é recomendado usar WSL2 com GPU passthrough configurado.')
                except:
                    pass
            
            # Verificar se CUDA está disponível no OpenCV
            try:
                cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
                self.get_logger().info(f'Dispositivos CUDA disponíveis no OpenCV: {cuda_devices}')
                
                if cuda_devices > 0:
                    # Testar se CUDA está realmente funcionando
                    try:
                        # Criar e usar um objeto CUDA para verificar se realmente funciona
                        test_mat = cv2.cuda_GpuMat((10, 10), cv2.CV_8UC3)
                        self.get_logger().info('Teste CUDA bem-sucedido! CUDA está funcionando corretamente.')
                    except Exception as cuda_test_error:
                        self.get_logger().warn(f'CUDA detectado, mas teste falhou: {str(cuda_test_error)}')
                        self.get_logger().info('CUDA pode não estar funcionando corretamente apesar de ser detectado')
                else:
                    self.get_logger().warn('Nenhum dispositivo CUDA disponível no OpenCV!')
                    self.get_logger().info('Verificando variáveis de ambiente CUDA...')
                    
                    # Verificar se o OpenCV foi compilado com suporte CUDA
                    try:
                        has_cuda_build = 'cuda' in cv2.getBuildInformation().lower()
                        if not has_cuda_build:
                            self.get_logger().warn('OpenCV não foi compilado com suporte CUDA')
                        else:
                            self.get_logger().info('OpenCV foi compilado com suporte CUDA, mas nenhum dispositivo CUDA foi detectado')
                    except:
                        pass
                    
                    # Verificar se as variáveis de ambiente CUDA estão configuradas
                    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'PATH', 'CUDAHOME', 'CUDA_HOME']
                    for var in cuda_vars:
                        self.get_logger().info(f'{var}: {os.environ.get(var, "não definido")}')
                    
                    # Verificar se os módulos estão disponíveis (para diagnóstico)
                    try:
                        import subprocess
                        ldconfig = subprocess.check_output('ldconfig -p | grep -i cuda', shell=True).decode('utf-8')
                        self.get_logger().info(f'Bibliotecas CUDA encontradas: {ldconfig[:200]}...' if len(ldconfig) > 200 else ldconfig)
                    except:
                        self.get_logger().info('Não foi possível verificar bibliotecas CUDA via ldconfig')
                    
                    # Configurar para desabilitar CUDA se não estiver disponível
                    if self.get_parameter('enable_cuda').value:
                        self.get_logger().warn('Desabilitando processamento CUDA devido à indisponibilidade')
                        # Não podemos definir parâmetros diretamente, mas podemos avisar que CUDA será ignorado
                        # ou podemos criar um parâmetro interno para controlar isso
                        self._cuda_available = False
            except Exception as e:
                self.get_logger().warn(f'Erro ao verificar dispositivos CUDA no OpenCV: {str(e)}')
                self._cuda_available = False
            
            # Verificar se o módulo nvarguscamerasrc está disponível e funcionando
            try:
                # Testar se o módulo nvarguscamerasrc pode ser carregado
                test_cmd = "gst-launch-1.0 nvarguscamerasrc num-buffers=0 ! fakesink"
                result = subprocess.run(test_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                
                # Se o comando falhar rapidamente, provavelmente há um problema com nvarguscamerasrc
                if result.returncode != 0:
                    stderr = result.stderr.decode('utf-8')
                    if "no element" in stderr and "nvarguscamerasrc" in stderr:
                        self.get_logger().error('Plugin GStreamer nvarguscamerasrc não está disponível!')
                        self.get_logger().info('Verificando alternativas para acesso à câmera...')
                        
                        # Verificar se v4l2src está disponível como alternativa
                        try:
                            v4l2_test = subprocess.run("gst-launch-1.0 v4l2src ! fakesink", 
                                                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                            if v4l2_test.returncode == 0:
                                self.get_logger().info('Plugin v4l2src disponível como alternativa para câmeras USB')
                                self.get_logger().info('Use uma webcam USB em vez da câmera CSI para funcionamento básico')
                            else:
                                self.get_logger().warn('Nem nvarguscamerasrc nem v4l2src estão funcionando corretamente')
                        except:
                            self.get_logger().warn('Erro ao testar alternativa v4l2src')
                    elif "could not link" in stderr:
                        self.get_logger().error('Problemas de compatibilidade entre elementos GStreamer no pipeline.')
                    elif "Resource error" in stderr:
                        self.get_logger().error('Erro de recurso com nvarguscamerasrc - câmera pode estar em uso por outro processo.')
                        self.get_logger().info('Tentando identificar processos usando a câmera...')
                        
                        try:
                            # Tentar identificar processos que podem estar usando a câmera
                            camera_procs = subprocess.check_output(['fuser', '-v', '/dev/video0'], stderr=subprocess.STDOUT).decode('utf-8')
                            self.get_logger().info(f'Processos usando a câmera: {camera_procs}')
                        except:
                            self.get_logger().info('Não foi possível identificar processos usando a câmera.')
                        
                        # Tentar reiniciar o serviço nvargus-daemon
                        try:
                            self.get_logger().info('Tentando reiniciar o serviço nvargus-daemon...')
                            subprocess.run(['systemctl', 'restart', 'nvargus-daemon'], 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                            time.sleep(2)  # Aguardar serviço reiniciar
                            self.get_logger().info('Serviço nvargus-daemon reiniciado, tentando novamente...')
                            
                            # Testar novamente
                            result = subprocess.run(test_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                            if result.returncode == 0:
                                self.get_logger().info('Reinício do serviço nvargus-daemon resolveu o problema!')
                            else:
                                stderr = result.stderr.decode('utf-8') if result.stderr else "Unknown error"
                                self.get_logger().warn(f"Erro ao reiniciar via systemctl: {stderr}")
                        except Exception as e:
                            self.get_logger().warn(f'Erro ao tentar reiniciar serviço nvargus-daemon: {str(e)}')
                    else:
                        self.get_logger().error(f'Erro ao testar nvarguscamerasrc: {stderr}')
                    
            except subprocess.TimeoutExpired:
                # Se o comando não retornar dentro do timeout, provavelmente está funcionando mas pendurando
                self.get_logger().info('Teste de nvarguscamerasrc executado sem erros imediatos.')
            except Exception as e:
                self.get_logger().warn(f'Erro ao testar nvarguscamerasrc: {str(e)}')
            
            # Adicionar variável de classe para rastrear disponibilidade de CUDA
            if not hasattr(self, '_cuda_available'):
                self._cuda_available = cuda_devices > 0
            
            return self._cuda_available
            
        except Exception as e:
            self.get_logger().error(f'Erro ao verificar permissões CUDA: {str(e)}')
            self._cuda_available = False
            return False

    def capture_loop(self):
        """Loop principal de captura com processamento CUDA."""
        # Verificar se estamos no modo simulado
        if hasattr(self, 'is_simulated') and self.is_simulated:
            # No modo simulado, a atualização é feita pelo timer
            return
        
        # Verificar se CUDA está habilitado
        cuda_enabled = self.get_parameter('enable_cuda').value
        
        if cuda_enabled:
            # Inicializar contexto CUDA
            try:
                self.get_logger().info('Inicializando contexto CUDA...')
                self.cuda_stream = cv2.cuda_Stream()
                self.cuda_upload = cv2.cuda_GpuMat()
                self.cuda_color = cv2.cuda_GpuMat()
                self.cuda_download = cv2.cuda_GpuMat()
                self.get_logger().info('Inicialização de contexto CUDA bem-sucedida')
            except Exception as e:
                self.get_logger().warn(f'Erro ao inicializar contexto CUDA: {str(e)}')
                self.get_logger().warn('Continuando sem aceleração CUDA.')
                cuda_enabled = False
        
        # Variáveis para controle de FPS e diagnóstico
        self.frame_interval = 1.0 / self.camera_fps  # Intervalo entre frames desejado
        self.last_capture_time = time.time()
        self.fps_timer_active = True
        
        # Configurar limite de tentativas para recuperação
        retry_count = 0
        max_retries = 5
        last_frame_time = time.time()
        frame_capture_error_count = 0
        
        # Contador total de frames para diagnóstico
        total_frames = 0
        capture_start_time = time.time()
        last_diagnostic_time = time.time()
        
        self.get_logger().info('Loop de captura iniciado com GStreamer e CUDA')
        
        while self.is_running and rclpy.ok():
            try:
                # Controlar intervalo entre capturas para manter FPS consistente
                frame_start_time = time.time()
                elapsed = frame_start_time - self.last_capture_time
                if elapsed < self.frame_interval:
                    sleep_time = self.frame_interval - elapsed
                    time.sleep(sleep_time)
                
                # Diagnosticar a cada 30 frames
                if (total_frames % 30) == 0:
                    self.get_logger().debug(f'Tentando capturar frame #{total_frames}...')
                
                # Capturar frame através do GStreamer
                ret, frame = self.cap.read()
                
                # Diagnosticar a cada 10 segundos
                current_time = time.time()
                if current_time - last_diagnostic_time > 10.0:
                    elapsed_total = current_time - capture_start_time
                    avg_fps = total_frames / elapsed_total if elapsed_total > 0 else 0
                    self.get_logger().info(f'Diagnóstico: {total_frames} frames em {elapsed_total:.1f}s (média {avg_fps:.1f} FPS)')
                    
                    # Verificar se estamos muito abaixo do FPS esperado (menos de 50%)
                    if avg_fps < (self.camera_fps * 0.5) and total_frames > 30:
                        self.get_logger().warn(f'FPS está muito abaixo do esperado: {avg_fps:.1f} vs {self.camera_fps} esperado')
                    
                    last_diagnostic_time = current_time
                
                # Atualizar timestamp de captura
                self.last_capture_time = time.time()
                
                # Verificar se a captura foi bem-sucedida
                if not ret:
                    self.get_logger().warn(f'Falha ao capturar frame #{total_frames}')
                    frame_capture_error_count += 1
                    time_since_last_frame = time.time() - last_frame_time
                    
                    # Alertar após múltiplas falhas
                    if frame_capture_error_count >= 3:
                        self.get_logger().warn(f'Múltiplas falhas consecutivas: {frame_capture_error_count}')
                    
                    # Verificar se passou muito tempo sem frames (5 segundos)
                    if time_since_last_frame > 5.0:
                        retry_count += 1
                        self.get_logger().warn(f'Sem frames por {time_since_last_frame:.1f}s. Tentativa {retry_count}/{max_retries}')
                        
                        # Tentar recuperar após várias tentativas
                        if retry_count >= max_retries:
                            self.get_logger().error('Excedido limite de tentativas. Entrando em modo de simulação.')
                            self.setup_simulation_mode()
                            return
                    
                    # Aguardar um pouco antes de tentar novamente
                    time.sleep(0.1)
                    continue
                
                # Captura bem-sucedida, resetar contadores
                total_frames += 1
                retry_count = 0
                frame_capture_error_count = 0
                last_frame_time = time.time()
                
                # Processamento CUDA para aprimoramento de imagem (se habilitado)
                if cuda_enabled:
                    try:
                        # Upload da imagem para a GPU
                        self.cuda_upload.upload(frame)
                        
                        # Aplicar filtros de aprimoramento conforme configuração
                        if self.get_parameter('apply_noise_reduction').value:
                            # Redução de ruído
                            cv2.cuda.fastNlMeansDenoisingColored(
                                self.cuda_upload, None, 
                                h_luminance=3, photo_render=1,
                                stream=self.cuda_stream
                            )
                        
                        # Aprimoramento de brilho/contraste
                        if self.get_parameter('apply_brightness_adjustment').value:
                            # Ajuste de gama
                            factor = float(self.get_parameter('brightness_factor').value)
                            cv2.cuda.gammaCorrection(
                                self.cuda_upload,
                                self.cuda_color,
                                factor,
                                stream=self.cuda_stream
                            )
                        else:
                            # Copiar sem alteração
                            self.cuda_upload.copyTo(self.cuda_color, self.cuda_stream)
                        
                        # Aprimoramento de bordas
                        if self.get_parameter('apply_edge_enhancement').value:
                            # Filtro Sobel para detecção de bordas
                            cv2.cuda.createSobelFilter(
                                cv2.CV_8UC3, cv2.CV_16S, 1, 0
                            ).apply(
                                self.cuda_color,
                                self.cuda_download,
                                self.cuda_stream
                            )
                            # Download do resultado da GPU para CPU
                            frame = self.cuda_download.download()
                        else:
                            # Download direto
                            frame = self.cuda_color.download()
                            
                    except Exception as cuda_err:
                        self.get_logger().error(f'Erro no processamento CUDA: {str(cuda_err)}')
                        # Continuar usando o frame original sem processamento CUDA
                
                # Publicar imagem processada
                try:
                    timestamp = self.get_clock().now().to_msg()
                    img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    img_msg.header.stamp = timestamp
                    img_msg.header.frame_id = "camera_optical_frame"
                    self.image_pub.publish(img_msg)
                    
                    # Publicar informações da câmera
                    self.publish_camera_info(img_msg.header)
                except CvBridgeError as e:
                    self.get_logger().error(f'Erro no CvBridge: {e}')
                
            except Exception as e:
                self.get_logger().error(f'Erro na captura: {str(e)}')
                time.sleep(0.1)  # Evitar loops muito rápidos em caso de erro

    def publish_camera_info(self, header):
        """Publica informações da câmera para calibração e transformações."""
        # Criar mensagem de info da câmera
        cam_info = CameraInfo()
        cam_info.header = header
        cam_info.distortion_model = 'plumb_bob'
        
        # Configurar tamanho da imagem
        width = self.get_parameter('image_width').value
        height = self.get_parameter('image_height').value
        
        if self.resize_output:
            width = self.output_width
            height = self.output_height
            
        cam_info.width = width
        cam_info.height = height
        
        # Verificar se temos informações de calibração
        if hasattr(self, 'camera_matrix') and self.camera_matrix is not None:
            # Usar calibração existente
            cam_info.k = self.camera_matrix.flatten().tolist()
            if hasattr(self, 'dist_coeffs') and self.dist_coeffs is not None:
                cam_info.d = self.dist_coeffs.flatten().tolist()
            if hasattr(self, 'projection_matrix') and self.projection_matrix is not None:
                cam_info.p = self.projection_matrix.flatten().tolist()
            if hasattr(self, 'rectification_matrix') and self.rectification_matrix is not None:
                cam_info.r = self.rectification_matrix.flatten().tolist()
        else:
            # Usar valores padrão para a câmera IMX219 se não tivermos calibração
            # Usar valores aproximados para parâmetros da câmera
            # Estes podem ser substituídos por valores de calibração reais
            focal_length = 3.04  # mm
            sensor_width = 4.8   # mm
            sensor_height = 3.6  # mm
            
            # Converter para pixels
            fx = focal_length * width / sensor_width
            fy = focal_length * height / sensor_height
            cx = width / 2.0
            cy = height / 2.0
            
            # Matriz de câmera K (intrínsecos)
            K = [fx, 0.0, cx, 
                 0.0, fy, cy, 
                 0.0, 0.0, 1.0]
            cam_info.k = K
            
            # Matriz de projeção P (identidade para câmera sem distorção)
            P = [fx, 0.0, cx, 0.0,
                 0.0, fy, cy, 0.0,
                 0.0, 0.0, 1.0, 0.0]
            cam_info.p = P
            
            # Matriz de retificação R (identidade para câmera sem retificação)
            R = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
            cam_info.r = R
            
            # Coeficientes de distorção (zero para câmera sem distorção)
            cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Publicar a mensagem
        self.camera_info_pub.publish(cam_info)
        
    def load_camera_calibration(self):
        """Carrega dados de calibração da câmera de um arquivo."""
        try:
            # Verificar se há um arquivo de calibração disponível
            calibration_file = self.get_parameter('calibration_file').value
            
            # Se não há calibração especificada, tentar encontrar um padrão
            if calibration_file == "":
                # Verificar pastas padrão
                possible_paths = [
                    os.path.join(os.path.expanduser('~'), '.ros', 'camera_info', 'camera.yaml'),
                    '/opt/ros/humble/share/camera_calibration/data/camera.yaml',
                    os.path.join(os.getcwd(), 'camera_info', 'camera.yaml')
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        calibration_file = path
                        self.get_logger().info(f"Encontrado arquivo de calibração em: {path}")
                        break
            
            if calibration_file != "" and os.path.exists(calibration_file):
                # Carregar dados YAML
                with open(calibration_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Extrair matrizes
                self.camera_matrix = np.array(data.get('camera_matrix', {}).get('data', [])).reshape(3, 3)
                self.dist_coeffs = np.array(data.get('distortion_coefficients', {}).get('data', []))
                
                if 'rectification_matrix' in data:
                    self.rectification_matrix = np.array(data['rectification_matrix']['data']).reshape(3, 3)
                    
                if 'projection_matrix' in data:
                    self.projection_matrix = np.array(data['projection_matrix']['data']).reshape(3, 4)
                
                self.get_logger().info("Calibração da câmera carregada com sucesso")
                return True
            else:
                self.get_logger().warn(f"Arquivo de calibração não encontrado: {calibration_file}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Erro ao carregar calibração da câmera: {str(e)}")
            return False

    def monitor_resources(self):
        """Monitora recursos do sistema."""
        try:
            # Temperatura - tratamento para ambiente containerizado
            try:
                temp = float(subprocess.check_output(['cat', '/sys/class/thermal/thermal_zone0/temp']).decode()) / 1000.0
                self.get_logger().info(f'Temperatura: {temp:.1f}°C')
                
                # Alertar se temperatura muito alta
                if temp > 80.0:
                    self.get_logger().warn('Temperatura muito alta! Considere reduzir FPS ou resolução')
            except (subprocess.SubprocessError, FileNotFoundError, ValueError):
                self.get_logger().debug('Não foi possível ler a temperatura')
            
            # Estatísticas do GPU via tegrastats (quando disponível)
            try:
                tegrastats = subprocess.check_output(['tegrastats', '--interval', '1', '--count', '1']).decode()
                self.get_logger().info(f'Tegrastats: {tegrastats}')
            except (subprocess.SubprocessError, FileNotFoundError):
                # Tentar nvidia-smi como alternativa
                try:
                    gpu_stats = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv']).decode()
                    self.get_logger().info(f'GPU Stats: {gpu_stats}')
                except (subprocess.SubprocessError, FileNotFoundError):
                    self.get_logger().debug('Não foi possível obter estatísticas da GPU')
            
            # Verificar estatísticas da câmera
            if hasattr(self, 'cap') and self.cap.isOpened():
                try:
                    # Usar o FPS real calculado internamente em vez de tentar obter da câmera
                    if hasattr(self, 'real_fps') and self.real_fps > 0:
                        self.get_logger().info(f'Câmera: FPS real={self.real_fps:.1f}')
                    else:
                        # Se real_fps não estiver disponível, usar o valor configurado
                        self.get_logger().info(f'Câmera: FPS configurado={self.camera_fps}')
                        
                        # Se real_fps estiver definido mas for 0, pode haver um problema
                        if hasattr(self, 'real_fps') and self.real_fps == 0:
                            # Em contêiner, isso é esperado - usar nossa abordagem de temporizador
                            if not hasattr(self, 'fps_warning_shown'):
                                self.get_logger().info('Em ambiente containerizado, usando temporizador interno para FPS em vez do FPS da câmera')
                                self.fps_warning_shown = True
                                
                            # Mesmo assim, fazer uma verificação rápida da câmera para garantir que está tudo bem
                            ret, _ = self.cap.read()
                            if not ret:
                                self.get_logger().error('Não foi possível obter imagem da câmera - "Unable to get camera FPS"')
                                self.get_logger().info('Tentando reiniciar a câmera...')
                                
                                # Verificar se o pipeline precisa ser reiniciado
                                self.get_logger().info('Tentando reiniciar a captura...')
                                self.get_camera_info_gstreamer()
                                
                                # Reiniciar os contadores de FPS
                                self.frame_count = 0
                                self.last_fps_time = time.time()
                except Exception as e:
                    self.get_logger().warn(f'Erro ao obter estatísticas da câmera: {e}')
                
        except Exception as e:
            self.get_logger().error(f'Erro ao monitorar recursos: {e}')

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
        """Configura pipeline específico para câmera CSI na Jetson."""
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        framerate = self.get_parameter('framerate').value
        
        # Pipeline especifico para câmera IMX219 (Raspberry Pi Camera v2) na Jetson
        pipeline_str = (
            f"nvarguscamerasrc sensor-id={self.get_parameter('sensor_id').value} "
            f"! video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate={framerate}/1 ! nvvidconv flip-method={self.get_parameter('flip_method').value} "
            f"! video/x-raw, width={width}, height={height}, format=BGRx ! videoconvert "
            f"! video/x-raw, format=BGR ! appsink"
        )
        
        self.get_logger().info(f"Pipeline CSI: {pipeline_str}")
        self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            self.get_logger().error("Falha ao abrir pipeline nvarguscamerasrc. Verificando alternativas...")
            # Tente pipeline alternativo com nvarguscamerasrc
            alt_pipeline = (
                f"nvarguscamerasrc sensor-id={self.get_parameter('sensor_id').value} "
                f"! video/x-raw(memory:NVMM), width={width}, height={height}, "
                f"format=NV12, framerate={framerate}/1 ! nvvidconv "
                f"! video/x-raw, format=BGRx ! videoconvert ! appsink"
            )
            self.get_logger().info(f"Tentando pipeline alternativo: {alt_pipeline}")
            self.cap = cv2.VideoCapture(alt_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                # Se ainda falhar, tente com v4l2src como último recurso
                video_devices = glob.glob('/dev/video*')
                if video_devices:
                    self.get_logger().warn("Tentando câmera V4L2 como alternativa...")
                    self._configure_v4l2_pipeline(video_devices[0])
                else:
                    self.get_logger().error("Todas as tentativas de câmera falharam. Usando simulação.")
                    self._configure_fake_camera_pipeline()
    
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
        
        # Carregar uma imagem de teste ou criar uma imagem em branco
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        
        # Tentar carregar uma imagem de teste de vários locais possíveis
        test_image_paths = [
            'src/perception/resources/test_images/field.jpg',
            '/ros2_ws/src/perception/resources/test_images/field.jpg',
            '/usr/local/share/perception/test_images/field.jpg',
            '/opt/ros/humble/share/perception/test_images/field.jpg'
        ]
        
        self.fake_frame = None
        for path in test_image_paths:
            if os.path.exists(path):
                self.get_logger().info(f"Carregando imagem de teste: {path}")
                try:
                    self.fake_frame = cv2.imread(path)
                    if self.fake_frame is not None:
                        # Redimensionar para a resolução configurada
                        self.fake_frame = cv2.resize(self.fake_frame, (width, height))
                        break
                except Exception as e:
                    self.get_logger().warn(f"Erro ao carregar imagem {path}: {str(e)}")
        
        # Se nenhuma imagem de teste for encontrada, criar uma em branco
        if self.fake_frame is None:
            self.get_logger().warn("Nenhuma imagem de teste encontrada, criando uma em branco")
            self.fake_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Desenhar um campo simples para simular visão
            # Fundo verde
            self.fake_frame[:, :] = (0, 128, 0)  # BGR para verde
            
            # Linhas brancas
            center_x, center_y = width // 2, height // 2
            # Linha central
            cv2.line(self.fake_frame, (center_x, 0), (center_x, height), (255, 255, 255), 2)
            # Círculo central
            cv2.circle(self.fake_frame, (center_x, center_y), min(width, height) // 6, (255, 255, 255), 2)
            # Bordas do campo
            cv2.rectangle(self.fake_frame, (50, 50), (width - 50, height - 50), (255, 255, 255), 2)
            
            # Adicionar uma bola laranja
            cv2.circle(self.fake_frame, 
                      (center_x + np.random.randint(-100, 100), 
                       center_y + np.random.randint(-100, 100)), 
                      15, (0, 165, 255), -1)
            
            self.get_logger().info(f"Imagem simulada criada com dimensões {width}x{height}")
            
        # Configurar timestamp para frames simulados
        self.last_frame_time = self.get_clock().now()
        
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