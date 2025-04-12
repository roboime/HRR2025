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
            gst_check = subprocess.check_output(['gst-inspect-1.0', 'nvarguscamerasrc'], stderr=subprocess.STDOUT).decode('utf-8')
            if 'nvarguscamerasrc' not in gst_check:
                self.get_logger().error('Plugin GStreamer nvarguscamerasrc não encontrado. Este plugin é necessário para o funcionamento da câmera CSI.')
                raise RuntimeError('Plugin GStreamer nvarguscamerasrc não encontrado')
        except (subprocess.SubprocessError, FileNotFoundError):
            self.get_logger().error('GStreamer não disponível ou não configurado corretamente.')
            raise RuntimeError('GStreamer não disponível')
            
        # Em ambientes containerizados, verificar se gst-launch básico funciona com nvarguscamerasrc
        if is_container:
            gst_test_success = False
            try:
                self.get_logger().info('Testando funcionalidade básica de nvarguscamerasrc...')
                # Usar num-buffers=1 para garantir que o comando não bloqueia indefinidamente
                test_pipeline = "gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink"
                # Usar um timeout mais longo (10 segundos) para dar tempo ao comando completar
                subprocess.check_call(test_pipeline, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                self.get_logger().info('Teste básico de nvarguscamerasrc bem-sucedido.')
                gst_test_success = True
            except subprocess.TimeoutExpired:
                self.get_logger().error('Teste de nvarguscamerasrc falhou: timeout')
                self.get_logger().info('Tentando método alternativo com menor timeout...')
                
                # Tentar novamente com um comando diferente
                try:
                    # Verificar apenas se o plugin está disponível sem tentar capturar
                    inspect_cmd = "gst-inspect-1.0 nvarguscamerasrc"
                    subprocess.check_call(inspect_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                    self.get_logger().info('Plugin nvarguscamerasrc está disponível.')
                    gst_test_success = True  # Considerar sucesso se pelo menos o plugin existir
                except Exception as inspect_error:
                    self.get_logger().error(f'Verificação do plugin nvarguscamerasrc falhou: {str(inspect_error)}')
                    self.get_logger().error('Problema fundamental com o acesso à câmera CSI.')
                    raise RuntimeError('Falha no teste básico de nvarguscamerasrc')
            except subprocess.CalledProcessError as e:
                self.get_logger().error(f'Teste de nvarguscamerasrc falhou: {str(e)}')
                self.get_logger().error('Problema fundamental com o acesso à câmera CSI.')
                
                # Verificar o motivo específico da falha para dar mais informações ao usuário
                if hasattr(e, 'stderr') and e.stderr:
                    stderr = e.stderr.decode('utf-8') if isinstance(e.stderr, bytes) else str(e.stderr)
                    if "cannot connect to camera" in stderr:
                        self.get_logger().error('Problema de conexão com a câmera. Verifique se a câmera está conectada corretamente.')
                    elif "Resource busy" in stderr:
                        self.get_logger().error('Câmera está em uso por outro processo. Tente fechar outros aplicativos que possam estar usando a câmera.')
                
                raise RuntimeError('Falha no teste básico de nvarguscamerasrc')
                
            if gst_test_success:
                # Configurar variáveis de ambiente específicas para NVIDIA
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                if "DISPLAY" in os.environ:
                    self.get_logger().info(f'Variável DISPLAY encontrada: {os.environ["DISPLAY"]}')
                
                # Para ambientes containerizados, desabilitar EGL e configurar corretamente
                os.environ["GST_GL_API"] = "gles2"
                os.environ["GST_GL_PLATFORM"] = "egl"
                os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,video,utility"
                if "NVIDIA_VISIBLE_DEVICES" not in os.environ:
                    os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
                
                # Importante: desativar o GL nos pipelines GStreamer para evitar problemas de EGL
                os.environ["GST_GL_XINITTHREADS"] = "0"
                os.environ["__GL_SYNC_TO_VBLANK"] = "0"
                
                self.get_logger().info('Variáveis de ambiente configuradas para contêiner.')
                
                # Não usar V4L2, usar apenas GStreamer
                self.get_logger().info('Pulando testes com V4L2, usando diretamente GStreamer para câmera CSI')
        
        # Pipeline GStreamer otimizado para IMX219 com aceleração CUDA
        # Calcular framerate como fração reduzida para evitar problemas
        import math
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
            
        fps_num = self.camera_fps
        fps_den = 1
        divisor = gcd(fps_num, fps_den)
        fps_num //= divisor
        fps_den //= divisor
        
        # Detectar se temos acesso a EGL (necessário para alguns recursos NVMM)
        try:
            # Checar se a variável de ambiente DISPLAY está definida
            has_display = "DISPLAY" in os.environ and os.environ["DISPLAY"]
            # Em contêineres, mesmo que DISPLAY exista, pode não haver acesso real
            if is_container:
                has_display = False
                # Adicionar tentativa de testar com comando simples
                try:
                    subprocess.check_call(['xdpyinfo'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    has_display = True
                except (subprocess.SubprocessError, FileNotFoundError):
                    has_display = False
        except Exception:
            has_display = False
        
        self.get_logger().info(f'Ambiente com display X11: {"Sim" if has_display else "Não"}')
        
        # Pipeline principal com envio para appsink (para ROS) e opcionalmente para ximagesink (display)
        display_enabled = self.get_parameter('enable_display').value and has_display
        
        # Reduzir o fps para 30 se for captura ao vivo (evita problemas de timeout)
        if fps_num > 30:
            self.get_logger().info(f'Reduzindo FPS para 30 para evitar problemas de timeout')
            fps_num = 30
            fps_den = 1
        
        # Definir pipeline dependendo do ambiente
        if is_container:
            # Pipeline otimizado para ambiente containerizado com CUDA
            # Sem codificação/decodificação para melhor desempenho
            # Usando apenas elementos essenciais para melhorar a estabilidade
            pipeline = (
                f"nvarguscamerasrc sensor-id=0 do-timestamp=true " 
                f"! video/x-raw(memory:NVMM), width=(int){self.width}, height=(int){self.height}, "
                f"format=(string)NV12, framerate=(fraction){fps_num}/{fps_den} "
                f"! nvvidconv flip-method={self.get_parameter('flip_method').value} "
                f"! video/x-raw, format=(string)BGRx "
                f"! videoconvert ! video/x-raw, format=(string)BGR "
                f"! appsink max-buffers=4 drop=true sync=false name=sink emit-signals=true"
            )
            
            # Pipeline alternativo em caso de falha
            self.fallback_pipeline = (
                f"nvarguscamerasrc sensor-id=0 num-buffers=0 "
                f"! video/x-raw(memory:NVMM), width=1280, height=720, "
                f"format=(string)NV12, framerate=30/1 "
                f"! nvvidconv ! video/x-raw, format=(string)BGRx "
                f"! videoconvert ! video/x-raw, format=(string)BGR "
                f"! appsink max-buffers=2 drop=true sync=false"
            )
        else:
            # Pipeline completo para ambiente nativo com aceleração CUDA
            pipeline = (
                f"nvarguscamerasrc sensor-id=0 "
                f"exposuretimerange='{self.get_parameter('exposure_time').value} {self.get_parameter('exposure_time').value}' "
                f"gainrange='{self.get_parameter('gain').value} {self.get_parameter('gain').value}' "
                f"wbmode={self.get_parameter('awb_mode').value} "
                f"tnr-mode=2 ee-mode=2 "
                f"saturation={self.get_parameter('saturation').value} "
                f"brightness={self.get_parameter('brightness').value} "
                f"ispdigitalgainrange='1 2' "
                f"! video/x-raw(memory:NVMM), width=(int){self.width}, height=(int){self.height}, "
                f"format=(string)NV12, framerate=(fraction){fps_num}/{fps_den} "
                f"! nvvidconv flip-method={self.get_parameter('flip_method').value} "
                f"! video/x-raw, format=(string)BGRx "
                f"! videoconvert ! video/x-raw, format=(string)BGR "
                f"! appsink max-buffers=2 drop=true sync=false"
            )
        
        self.get_logger().info(f'Pipeline GStreamer: {pipeline}')
        
        # Definir variáveis de ambiente para CUDA se ainda não estiverem definidas
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Tentar abrir a câmera com o pipeline GStreamer
        try:
            # Verificar permissões para CUDA antes de tentar abrir a câmera
            self.check_cuda_permissions()
            
            # Tentar matar qualquer processo gst-launch existente que possa estar ocupando a câmera
            try:
                if is_container:
                    self.get_logger().info('Tentando liberar recursos da câmera...')
                    subprocess.run("pkill -f 'gst-launch.*nvarguscamerasrc'", shell=True, 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    # Dar tempo para os recursos serem liberados
                    time.sleep(1)
            except Exception as kill_err:
                self.get_logger().debug(f'Erro ao tentar liberar recursos: {str(kill_err)}')
            
            # Abrir a câmera com o pipeline GStreamer
            self.get_logger().info('Abrindo câmera com pipeline GStreamer...')
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                self.get_logger().warn('Falha na primeira tentativa com pipeline principal. Tentando pipeline alternativo...')
                
                # Tentar pipeline alternativo mais simples
                if hasattr(self, 'fallback_pipeline'):
                    self.cap = cv2.VideoCapture(self.fallback_pipeline, cv2.CAP_GSTREAMER)
                    
                    if self.cap.isOpened():
                        self.get_logger().info('Pipeline alternativo bem-sucedido!')
                    else:
                        raise RuntimeError('Falha também com pipeline alternativo')
                
                if is_container:
                    self.get_logger().error('Falha ao abrir câmera com GStreamer em ambiente containerizado.')
                    self.get_logger().error('A integração entre GStreamer e OpenCV pode ser problemática em contêineres.')
                    self.get_logger().info('Tentando método GST-LAUNCH integrado...')
                    
                    # Tentar com abordagem alternativa que usa gst-launch diretamente
                    # Esta abordagem contorna problemas de integração OpenCV-GStreamer em contêineres
                    try:
                        # Criar um pipe nomeado temporário
                        import tempfile
                        import atexit
                        
                        # Criar diretório temporário se não existir
                        temp_dir = "/tmp/camera_pipes"
                        if not os.path.exists(temp_dir):
                            os.makedirs(temp_dir)
                            
                        pipe_path = f"{temp_dir}/camera_pipe_{os.getpid()}"
                        if os.path.exists(pipe_path):
                            os.unlink(pipe_path)
                        os.mkfifo(pipe_path)
                        
                        # Registrar limpeza ao sair
                        def cleanup_pipe():
                            if os.path.exists(pipe_path):
                                os.unlink(pipe_path)
                        atexit.register(cleanup_pipe)
                        
                        # Construir comando gst-launch para escrever frames em formato bruto para o pipe
                        # Otimizado para usar CUDA diretamente
                        gst_cmd = (
                            f"gst-launch-1.0 nvarguscamerasrc sensor-id=0 "
                            f"! video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
                            f"format=NV12, framerate={fps_num}/{fps_den} "
                            f"! nvvidconv flip-method={self.get_parameter('flip_method').value} "
                            f"! video/x-raw, format=BGRx "
                            f"! videoconvert ! video/x-raw, format=BGR "
                            f"! filesink location={pipe_path} append=true buffer-size=16777216"
                        )
                        
                        # Executar gst-launch em background
                        self.gst_process = subprocess.Popen(
                            gst_cmd, 
                            shell=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE
                        )
                        
                        # Aguardar um pouco para o pipeline iniciar
                        time.sleep(2)
                        
                        # Verificar se o processo está rodando
                        if self.gst_process.poll() is not None:
                            stderr = self.gst_process.stderr.read().decode()
                            self.get_logger().error(f"Falha ao iniciar gst-launch: {stderr}")
                            raise RuntimeError(f"gst-launch falhou: {stderr}")
                            
                        # Criar VideoCapture para ler do pipe
                        self.cap = cv2.VideoCapture(pipe_path)
                        if not self.cap.isOpened():
                            self.get_logger().error("Falha ao abrir pipe para leitura de frames")
                            raise RuntimeError("Falha ao abrir pipe para leitura")
                            
                        self.get_logger().info("Câmera inicializada com sucesso via método GST-LAUNCH integrado")
                        
                        # Configurar variáveis para cálculo de FPS real
                        self.frame_count = 0
                        self.last_fps_time = time.time()
                        self.real_fps = 0.0
                        
                        # Flag que indica que estamos usando o método alternativo
                        self.using_gst_launch_direct = True
                        return
                        
                    except Exception as pipe_error:
                        self.get_logger().error(f"Falha com método GST-LAUNCH integrado: {str(pipe_error)}")
                        raise RuntimeError(f"Todos os métodos de acesso à câmera falharam: {str(pipe_error)}")
                else:
                    # Em ambiente não-containerizado, tentar solução padrão
                    self.get_logger().error('Falha ao abrir câmera com o pipeline GStreamer. Verifique conexões e drivers.')
                    raise RuntimeError('Falha ao abrir câmera com GStreamer')
            
            # Tentar capturar um frame logo após abrir a câmera para verificar se está funcionando
            ret, test_frame = self.cap.read()
            if ret:
                self.get_logger().info(f'Primeiro frame capturado com sucesso: {test_frame.shape}')
                
                # Testar acesso ao FPS
                fps_value = self.cap.get(cv2.CAP_PROP_FPS)
                if fps_value <= 0:
                    self.get_logger().warn('Não foi possível obter FPS da câmera. Usando valor configurado.')
                else:
                    self.get_logger().info(f'FPS reportado pela câmera: {fps_value}')
            else:
                self.get_logger().warn('Não foi possível capturar o primeiro frame. A câmera pode não estar funcionando corretamente.')
            
            # Configurar cálculo de FPS real
            self.frame_count = 0
            self.last_fps_time = time.time()
            self.real_fps = 0.0
            
            # Ajustar propriedades específicas (embora possa não ter efeito com GStreamer)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Minimizar latência
            
            self.get_logger().info('Câmera inicializada com sucesso usando GStreamer')
            
        except Exception as e:
            self.get_logger().error(f'Exceção ao inicializar câmera: {str(e)}')
            self.debug_camera_devices()
            raise RuntimeError(f'Falha ao abrir câmera: {str(e)}')

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
            for device in nvidia_devices:
                if os.path.exists(device):
                    try:
                        mode = os.stat(device).st_mode
                        readable = bool(mode & 0o444)  # Verificar permissão de leitura
                        writeable = bool(mode & 0o222)  # Verificar permissão de escrita
                        self.get_logger().info(f'Dispositivo {device} - Leitura: {readable}, Escrita: {writeable}')
                    except Exception as e:
                        self.get_logger().warn(f'Erro ao verificar permissões de {device}: {str(e)}')
            
            # Verificar se CUDA está disponível no OpenCV
            try:
                cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
                self.get_logger().info(f'Dispositivos CUDA disponíveis no OpenCV: {cuda_devices}')
                
                if cuda_devices == 0:
                    self.get_logger().warn('Nenhum dispositivo CUDA disponível no OpenCV!')
                    self.get_logger().info('Verificando variáveis de ambiente CUDA...')
                    
                    # Verificar se as variáveis de ambiente CUDA estão configuradas
                    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'PATH']
                    for var in cuda_vars:
                        self.get_logger().info(f'{var}: {os.environ.get(var, "não definido")}')
            except Exception as e:
                self.get_logger().warn(f'Erro ao verificar dispositivos CUDA no OpenCV: {str(e)}')
            
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
                        self.get_logger().info('Verifique se o Jetpack está instalado corretamente.')
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
                                self.get_logger().warn('Reinício do serviço nvargus-daemon não resolveu o problema.')
                        except Exception as e:
                            self.get_logger().warn(f'Erro ao tentar reiniciar serviço nvargus-daemon: {str(e)}')
                    else:
                        self.get_logger().error(f'Erro ao testar nvarguscamerasrc: {stderr}')
                        
            except subprocess.TimeoutExpired:
                # Se o comando não retornar dentro do timeout, provavelmente está funcionando mas pendurando
                self.get_logger().info('Teste de nvarguscamerasrc executado sem erros imediatos.')
            except Exception as e:
                self.get_logger().warn(f'Erro ao testar nvarguscamerasrc: {str(e)}')
            
        except Exception as e:
            self.get_logger().error(f'Erro ao verificar permissões CUDA: {str(e)}')

    def capture_loop(self):
        """Loop principal de captura com processamento CUDA."""
        # Verificar se estamos no modo simulado
        if hasattr(self, 'is_simulated') and self.is_simulated:
            # No modo simulado, a atualização é feita pelo timer
            return
            
        # Verificar se estamos usando o método direto gst-launch
        using_gst_direct = hasattr(self, 'using_gst_launch_direct') and self.using_gst_launch_direct
        
        cuda_enabled = self.get_parameter('enable_cuda').value
        
        if cuda_enabled:
            # Inicializar contexto CUDA
            try:
                self.cuda_stream = cv2.cuda_Stream()
                self.cuda_upload = cv2.cuda_GpuMat()
                self.cuda_color = cv2.cuda_GpuMat()
                self.cuda_download = cv2.cuda_GpuMat()
                self.get_logger().info('Inicialização de contexto CUDA bem-sucedida')
            except Exception as e:
                self.get_logger().warn(f'Erro ao inicializar contexto CUDA: {str(e)}')
                cuda_enabled = False
        
        # Tentar detectar problemas com CUDA em ambiente containerizado
        is_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
        if is_container and cuda_enabled:
            try:
                # Verificar se CUDA está realmente disponível no contêiner
                cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
                if not cuda_available:
                    self.get_logger().warn('CUDA está habilitado nas configurações, mas dispositivos CUDA não foram detectados!')
                    self.get_logger().warn('Desabilitando processamento CUDA.')
                    cuda_enabled = False
            except Exception as e:
                self.get_logger().warn(f'Erro ao verificar suporte CUDA: {str(e)}')
                self.get_logger().warn('Desabilitando processamento CUDA.')
                cuda_enabled = False
        
        # Variáveis para cálculo de FPS real
        fps_update_interval = 1.0  # segundos
        frame_times = []
        max_frame_times = 30  # Para média móvel
        retry_count = 0
        max_retries = 5
        last_frame_time = time.time()
        frame_capture_error_count = 0
        
        # Quando estamos em container, sempre tratar como modo "direto" para gerenciar melhor os frames
        # independentemente de estar usando SSH ou monitor direto
        self.get_logger().info('Usando modo containerizado otimizado para Jetson')
        
        # Em ambientes containerizados, melhor usar nosso próprio temporizador
        # para controlar o FPS em vez de depender da detecção do OpenCV
        self.frame_interval = 1.0 / self.camera_fps  # Intervalo entre frames desejado
        self.last_capture_time = time.time()
        
        # Flag para indicar que estamos usando nosso próprio timer para FPS
        self.fps_timer_active = True
        
        # Ignorar mensagens de aviso sobre "Unable to get camera FPS" porque 
        # estamos usando nosso próprio temporizador em contêiner
        self.get_logger().info('Em contêiner, usando temporizador interno para controle de FPS')
        
        while self.is_running and rclpy.ok():
            try:
                # Registrar tempo de início do frame
                frame_start_time = time.time()
                
                # Em ambiente containerizado, garantir o intervalo certo entre frames
                # para evitar o problema "Unable to get camera FPS"
                elapsed = frame_start_time - self.last_capture_time
                if elapsed < self.frame_interval:
                    sleep_time = self.frame_interval - elapsed
                    time.sleep(sleep_time)
                
                # Capturar frame
                ret, frame = self.cap.read()
                
                # Atualizar timestamp da última captura bem-sucedida
                current_time = time.time()
                self.last_capture_time = current_time
                
                if not ret:
                    frame_capture_error_count += 1
                    current_time = time.time()
                    time_since_last_frame = current_time - last_frame_time
                    
                    if frame_capture_error_count >= 3:  # Falhou 3 vezes seguidas
                        self.get_logger().warn(f'Múltiplas falhas ao capturar frame (count={frame_capture_error_count})')
                        
                    if time_since_last_frame > 5.0:  # Sem frames por 5 segundos
                        retry_count += 1
                        self.get_logger().warn(f'Falha ao capturar frame por {time_since_last_frame:.1f}s. Tentativa {retry_count}/{max_retries}')
                        
                        if retry_count >= max_retries:
                            self.get_logger().error('Máximo de tentativas excedido. Reiniciando a câmera...')
                            # Tentar reiniciar a câmera com método GST-LAUNCH direto
                            try:
                                self.get_logger().info('Reiniciando captura...')
                                # Fechar a câmera atual
                                if hasattr(self, 'cap') and self.cap.isOpened():
                                    self.cap.release()
                                
                                # Tenta reiniciar usando GStreamer diretamente com pipeline simples
                                self.get_camera_info_gstreamer()
                                
                                # Tentar reiniciar o serviço nvargus-daemon
                                try:
                                    self.get_logger().info("Tentando reiniciar serviço nvargus-daemon...")
                                    subprocess.run("systemctl restart nvargus-daemon", shell=True, 
                                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                                    time.sleep(2)  # Esperar o serviço reiniciar
                                except Exception as e:
                                    self.get_logger().debug(f"Erro ao reiniciar serviço: {str(e)}")
                                
                                # Usar pipeline mais simples para evitar problemas de FPS
                                fps_num = 30  # Usar FPS mais baixo para estabilidade
                                fps_den = 1
                                
                                # Pipeline extremamente simplificado para último recurso
                                test_pipeline = (
                                    f"nvarguscamerasrc sensor-id=0 num-buffers=0 "
                                    f"! video/x-raw(memory:NVMM), width=1280, height=720, "
                                    f"format=(string)NV12, framerate=(fraction){fps_num}/{fps_den} "
                                    f"! nvvidconv "
                                    f"! video/x-raw, format=(string)BGRx "
                                    f"! videoconvert "
                                    f"! video/x-raw, format=(string)BGR "
                                    f"! appsink max-buffers=2 drop=true sync=false"
                                )
                                
                                self.get_logger().info(f'Tentando pipeline simplificado: {test_pipeline}')
                                self.cap = cv2.VideoCapture(test_pipeline, cv2.CAP_GSTREAMER)
                                
                                if not self.cap.isOpened():
                                    raise RuntimeError("Não foi possível reiniciar a câmera")
                                
                                # Reiniciar contadores
                                self.frame_count = 0
                                self.last_fps_time = time.time()
                                retry_count = 0
                                frame_capture_error_count = 0
                                
                                self.get_logger().info('Câmera reiniciada com sucesso!')
                            except Exception as restart_error:
                                self.get_logger().error(f'Falha ao reiniciar câmera: {str(restart_error)}')
                                # Entrar em modo simulado como último recurso
                                self.get_logger().info('Entrando em modo de simulação após falhas na câmera real')
                                self.setup_simulation_mode()
                                return
                    
                    time.sleep(0.1)
                    continue
                
                # Reset do contador de tentativas e atualizar último frame bem-sucedido
                retry_count = 0
                frame_capture_error_count = 0
                last_frame_time = time.time()
                
                # Atualizar contagem para cálculo de FPS
                self.frame_count += 1
                current_time = time.time()
                frame_times.append(current_time - frame_start_time)
                if len(frame_times) > max_frame_times:
                    frame_times.pop(0)  # Manter apenas os frames mais recentes
                
                # Calcular FPS real a cada segundo
                elapsed = current_time - self.last_fps_time
                if elapsed >= fps_update_interval:
                    self.real_fps = self.frame_count / elapsed
                    
                    # Avisar se o FPS real está muito abaixo do esperado
                    if self.real_fps < (self.camera_fps * 0.5) and self.frame_count > 0:
                        self.get_logger().warn(f'FPS real ({self.real_fps:.1f}) está muito abaixo do configurado ({self.camera_fps})')
                    
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                if cuda_enabled:
                    # Processamento CUDA
                    try:
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
                            frame = self.cuda_download.download()
                        else:
                            # Download do resultado
                            frame = self.cuda_color.download()
                    except Exception as cuda_error:
                        self.get_logger().error(f'Erro no processamento CUDA: {str(cuda_error)}')
                        self.get_logger().warn('Continuando sem processamento CUDA.')
                        # Não modificar o frame, usar o original
                
                # Publicar imagem
                timestamp = self.get_clock().now().to_msg()
                img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                img_msg.header.stamp = timestamp
                img_msg.header.frame_id = "camera_optical_frame"
                self.image_pub.publish(img_msg)
                
                # Publicar info da câmera
                self.publish_camera_info(img_msg.header)
                
                # Não é necessário exibir aqui, já que o pipeline GStreamer tem ximagesink
                # se enable_display for verdadeiro
                
            except CvBridgeError as e:
                self.get_logger().error(f'Erro no CvBridge: {e}')
            except Exception as e:
                self.get_logger().error(f'Erro na captura: {e}')
                
            # Controle de taxa
            time.sleep(1.0 / self.camera_fps)

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
        """Tenta obter informações da câmera diretamente usando GStreamer"""
        try:
            self.get_logger().info('Tentando obter informações da câmera diretamente pelo GStreamer...')
            
            # Verificar e tentar reiniciar o nvargus-daemon
            try:
                # Verificar se o serviço está rodando
                daemon_check = "ps -ef | grep nvargus-daemon | grep -v grep"
                ps_result = subprocess.run(daemon_check, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                
                if ps_result.returncode != 0:
                    self.get_logger().warn("Serviço nvargus-daemon não está rodando. Tentando reiniciar...")
                    
                    # Primeiro, tentar matar qualquer processo nvargus existente
                    subprocess.run("pkill -f 'nvargus'", shell=True, 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(1)
                    
                    # Tentar iniciar o serviço nvargus-daemon
                    try:
                        subprocess.run("systemctl start nvargus-daemon", shell=True, 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                        self.get_logger().info("Serviço nvargus-daemon reiniciado.")
                        time.sleep(2)  # Dar tempo para o serviço inicializar
                    except Exception as e:
                        self.get_logger().warn(f"Erro ao reiniciar serviço: {str(e)}")
                else:
                    self.get_logger().info("Serviço nvargus-daemon está rodando.")
            except Exception as e:
                self.get_logger().debug(f"Erro ao verificar status de nvargus-daemon: {str(e)}")
            
            # Verificar se o plugin nvarguscamerasrc está disponível
            try:
                self.get_logger().info("Verificando disponibilidade do plugin nvarguscamerasrc...")
                inspect_cmd = "gst-inspect-1.0 nvarguscamerasrc"
                result = subprocess.run(inspect_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                
                if result.returncode == 0:
                    self.get_logger().info("Plugin nvarguscamerasrc está disponível.")
                    
                    # Tentar um teste simples para garantir que a câmera esteja acessível
                    self.get_logger().info("Testando acesso à câmera...")
                    
                    # Pipeline simples que não deve bloquear
                    test_cmd = "gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink"
                    test_result = subprocess.run(test_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                    
                    if test_result.returncode == 0:
                        self.get_logger().info("Teste básico com a câmera CSI bem-sucedido!")
                        
                        # Tentar pipeline com configurações específicas para garantir compatibilidade
                        validate_cmd = (
                            "gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! "
                            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
                            "fakesink"
                        )
                        
                        try:
                            self.get_logger().info("Testando configurações específicas...")
                            validate_result = subprocess.run(validate_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                            
                            if validate_result.returncode == 0:
                                self.get_logger().info("Teste de configurações específicas bem-sucedido!")
                            else:
                                stderr = validate_result.stderr.decode('utf-8')
                                self.get_logger().warn(f"Falha no teste de configurações específicas: {stderr}")
                        except subprocess.TimeoutExpired:
                            self.get_logger().warn("Timeout ao testar configurações específicas.")
                        except Exception as config_err:
                            self.get_logger().warn(f"Erro ao testar configurações: {str(config_err)}")
                    else:
                        stderr = test_result.stderr.decode('utf-8')
                        self.get_logger().warn(f"Falha no teste básico com a câmera: {stderr}")
                else:
                    self.get_logger().error("Plugin nvarguscamerasrc não está disponível!")
            except subprocess.TimeoutExpired:
                self.get_logger().warn("Timeout ao testar plugin nvarguscamerasrc.")
            except Exception as plugin_err:
                self.get_logger().warn(f"Erro ao verificar plugin nvarguscamerasrc: {str(plugin_err)}")
            
        except Exception as e:
            self.get_logger().error(f'Erro ao obter informações da câmera: {str(e)}')

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