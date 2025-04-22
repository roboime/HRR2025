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
                ('device_id', 0), # Embora usemos nvargus, manter para consistência
                ('display', False),
                ('framerate', 30.0),
                ('enable_cuda', True), # Habilitar CUDA por padrão
                ('calibration_file', ''),
                # Parâmetros específicos da câmera CSI IMX219 e GStreamer
                ('camera_mode', 6), # Modo HD 120fps
                ('flip_method', 0), # 0=none, 1=counterclockwise, 2=rotate180, 3=clockwise, 4=horizontal, 5=upper-right-diag, 6=vertical, 7=upper-left-diag
                # Parâmetros de controle (podem não ser todos suportados pelo nvarguscamerasrc diretamente)
                ('exposure_time', 13333), # Exemplo: ~1/75s em us
                ('gain', 1.0),
                ('awb_mode', 1), # 0=off, 1=auto, 2=incandescent, etc.
                ('saturation', 1.0),
                ('brightness', 0), # Pode não ter efeito direto no nvarguscamerasrc
                # Parâmetros de processamento
                ('enable_noise_reduction', False), # Exemplo de processamento CUDA
                ('enable_edge_enhancement', False), # Exemplo de processamento CUDA
            ]
        )

        # Variáveis de estado
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.bridge = CvBridge()
        self.camera_info_msg = None
        self.frame_count = 0
        self.last_fps_update = self.get_clock().now()
        self.current_fps = 0.0
        self._cuda_available = False # Será verificado depois

        # Configurar e inicializar
        self._configure_camera()
        self._configure_processing()
        self.load_camera_calibration() # Carrega se o arquivo for especificado

        # Timer para publicar CameraInfo (se carregado)
        self.camera_info_timer = self.create_timer(1.0, self.publish_camera_info)

        self.get_logger().info('Nó da câmera IMX219 (simplificado) inicializado.')

    def _configure_camera(self):
        """Configura a câmera e inicia o pipeline GStreamer."""
        self.get_logger().info('Configurando a câmera CSI...')

        # Mapeamento de modos da câmera IMX219
        # (Largura, Altura, FPS Máximo)
        self.camera_modes = {
            0: (3264, 2464, 21),
            1: (3264, 1848, 28),
            2: (1920, 1080, 30),
            3: (1640, 1232, 30), # Adicionado modo 4:3
            4: (1280, 720, 60),
            5: (1280, 720, 120) # Modo padrão no código original era 6? Revertendo para 5 se 6 não existir.
            # Original usava modo 6: (1280, 720, 120), parece ser o mesmo que 5. Usaremos 5.
        }

        self.camera_mode = self.get_parameter('camera_mode').value
        if self.camera_mode not in self.camera_modes:
             self.get_logger().warn(f"Modo de câmera {self.camera_mode} inválido. Usando modo 5 (1280x720@120).")
             self.camera_mode = 5 # Fallback para um modo conhecido

        self.width, self.height, self.max_fps = self.camera_modes[self.camera_mode]

        # Ajustar FPS solicitado ao máximo do modo
        requested_fps = self.get_parameter('framerate').value
        self.camera_fps = min(requested_fps, float(self.max_fps))
        if requested_fps > self.max_fps:
             self.get_logger().warn(f"FPS solicitado ({requested_fps}) maior que o máximo do modo ({self.max_fps}). Usando {self.camera_fps} FPS.")
        else:
             self.get_logger().info(f"Usando {self.camera_fps} FPS.")


        # Publishers
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        # CameraInfo será publicado por um timer separado após carregar calibração

        # Inicializar câmera GStreamer
        if not self.init_gstreamer_camera():
            # Se a inicialização falhar, o nó deve parar ou indicar erro claramente
            self.get_logger().fatal("FALHA CRÍTICA AO INICIALIZAR A CÂMERA GSTREAMER! O nó não funcionará.")
            # Considerar desligar o nó: rclpy.shutdown() ou levantar uma exceção mais específica
            raise RuntimeError("Não foi possível inicializar a câmera GStreamer.")
        else:
            self.get_logger().info(f'Câmera GStreamer inicializada: {self.width}x{self.height} @ {self.camera_fps}fps')
            # Iniciar thread de captura
            self.is_running = True
            self.capture_thread = threading.Thread(target=self.capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()

    def init_gstreamer_camera(self):
        """Inicializa a câmera GStreamer com o pipeline CSI."""
        self.get_logger().info("Verificando pré-requisitos para câmera CSI GStreamer...")

        # Verificar ambiente (é Jetson? Container?)
        is_jetson = os.path.exists('/etc/nv_tegra_release') or os.path.exists('/etc/nv_boot_control.conf')
        is_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
        if not is_jetson:
             self.get_logger().error("Este nó simplificado é projetado para Jetson com câmera CSI.")
             # return False # Comentado para permitir testes em outros ambientes se GStreamer estiver configurado

        if is_container:
             self.get_logger().info("Ambiente containerizado detectado.")
             # Verificações específicas do container (daemon, socket, permissões)
             self.check_argus_socket()
             self.check_nvargus_permissions()
             self.check_nvargus_daemon() # Pode não funcionar dentro do container, mas tenta

        # Verificar GStreamer e plugin nvarguscamerasrc
        if not self.check_gstreamer_plugin('nvarguscamerasrc'):
             return False

        # Construir o pipeline
        pipeline_str = self._build_gstreamer_pipeline()
        if not pipeline_str:
             self.get_logger().error("Falha ao construir a string do pipeline GStreamer.")
             return False

        self.get_logger().info(f"Tentando abrir pipeline GStreamer: {pipeline_str}")

        # Tentar abrir a câmera
        try:
            self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                self.get_logger().error("cv2.VideoCapture falhou ao abrir o pipeline GStreamer.")
                self.get_logger().error("Verifique:")
                self.get_logger().error("  - Conexão física da câmera CSI.")
                self.get_logger().error("  - Se o serviço nvargus-daemon está rodando (no host se em container).")
                self.get_logger().error("  - Permissões dos dispositivos /dev/nvhost-* e /dev/video*.")
                self.get_logger().error("  - Se outro processo está usando a câmera (use 'sudo pkill gst-launch' ou 'sudo pkill nvargus').")
                self.get_logger().error("  - Logs do sistema (dmesg, syslog) para erros relacionados à câmera ou GStreamer.")
                # Tentar diagnóstico adicional
                self.test_camera_pipeline(timeout=5) # Testa um pipeline básico
                return False

            # Verificar se conseguimos ler um frame (importante!)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.get_logger().error("Pipeline GStreamer aberto, mas falha ao ler o primeiro frame.")
                self.cap.release()
                self.cap = None
                # Tentar diagnóstico adicional
                self.test_camera_pipeline(timeout=5)
                return False
            else:
                self.get_logger().info(f"Primeiro frame lido com sucesso! Shape: {frame.shape}")
                # Liberar o frame de teste
                del frame

            self.get_logger().info("Pipeline GStreamer aberto e primeiro frame lido com sucesso.")
            return True

        except Exception as e:
            self.get_logger().error(f"Exceção ao tentar abrir VideoCapture GStreamer: {str(e)}")
            traceback.print_exc()
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def _build_gstreamer_pipeline(self):
        """Constrói a string do pipeline GStreamer CSI com base nos parâmetros."""
        try:
            # Calcular framerate como fração para GStreamer
            def gcd(a, b):
                while b: a, b = b, a % b
                return a
            fps_num = int(self.camera_fps * 100) # Usar precisão para evitar float
            fps_den = 100
            common = gcd(fps_num, fps_den)
            fps_num //= common
            fps_den //= common

            # Parâmetros de controle (alguns podem precisar ser configurados fora do pipeline)
            exposure = self.get_parameter('exposure_time').value
            gain = self.get_parameter('gain').value
            awb = self.get_parameter('awb_mode').value
            saturation = self.get_parameter('saturation').value
            flip = self.get_parameter('flip_method').value

            # Base do pipeline com nvarguscamerasrc
            # Nota: Controles como exposure/gain/awb podem precisar de 'nvarguscamerasrc ! tee name=t ... t. ! queue ! nvvidconv ... t. ! queue ! fakesink'
            # Ou configuração via v4l2-ctl ANTES de iniciar o pipeline. Simplificando aqui.
            pipeline = (
                f"nvarguscamerasrc sensor-id={self.get_parameter('device_id').value} "
                f"wbmode={awb} "
                # Adicionar controles de ganho/exposição se suportado diretamente (pode variar com versão L4T)
                # f"gainrange='{gain} {gain}' exposuretimerange='{exposure} {exposure}' " # Exemplo
                f"saturation={saturation} "
                f"! video/x-raw(memory:NVMM), width=(int){self.width}, height=(int){self.height}, "
                f"format=(string)NV12, framerate=(fraction){fps_num}/{fps_den} "
                f"! nvvidconv flip-method={flip} "
            )

            # Adicionar processamento CUDA se habilitado
            self._cuda_available = self.check_cuda_permissions() # Verificar/cachear status CUDA
            if self.get_parameter('enable_cuda').value and self._cuda_available:
                self.get_logger().info("Adicionando elementos CUDA ao pipeline.")
                # Conversão otimizada na GPU
                pipeline += f"! video/x-raw(memory:NVMM), format=(string)RGBA " # Formato intermediário para alguns filtros CUDA
                # Adicionar filtros CUDA opcionais aqui se necessário (ex: nvivafilter, nvdsosd)
                pipeline += f"! nvvideoconvert " # Converte de volta se necessário
                pipeline += f"! video/x-raw, format=(string)BGRx " # Formato comum para OpenCV
                pipeline += f"! videoconvert " # Conversão final para BGR
                pipeline += f"! video/x-raw, format=(string)BGR "

            else:
                if self.get_parameter('enable_cuda').value and not self._cuda_available:
                     self.get_logger().warn("CUDA solicitado, mas não disponível/permitido. Usando conversão CPU.")
                else:
                     self.get_logger().info("Usando conversão de vídeo via CPU (videoconvert).")
                # Conversão via CPU
                pipeline += f"! video/x-raw, format=(string)BGRx " # Saída comum do nvvidconv
                pipeline += f"! videoconvert " # Conversão principal na CPU
                pipeline += f"! video/x-raw, format=(string)BGR "

            # Finalizar com appsink
            pipeline += f"! appsink max-buffers=4 drop=true sync=false name=sink emit-signals=true"

            return pipeline

        except Exception as e:
            self.get_logger().error(f"Erro interno ao construir pipeline GStreamer: {str(e)}")
            traceback.print_exc()
            return None

    def check_gstreamer_plugin(self, plugin_name):
        """Verifica se um plugin GStreamer específico está disponível."""
        self.get_logger().info(f"Verificando disponibilidade do plugin GStreamer: {plugin_name}...")
        try:
            gst_check = subprocess.run(['gst-inspect-1.0', plugin_name],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.get_logger().info(f"Plugin '{plugin_name}' encontrado com sucesso!")
            return True
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            self.get_logger().error(f"Falha ao verificar plugin '{plugin_name}'. GStreamer pode não estar instalado corretamente ou o plugin está faltando.")
            self.get_logger().error(f"Erro: {str(e)}")
            if isinstance(e, subprocess.CalledProcessError):
                 stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
                 self.get_logger().error(f"Stderr: {stderr}")
            return False

    def check_argus_socket(self):
        """Verifica o socket Argus em ambiente containerizado."""
        socket_path = '/tmp/argus_socket'
        alt_paths = ['/tmp/.argus_socket', '/var/nvidia/nvcam/camera-daemon-socket']
        if not os.path.exists(socket_path):
            self.get_logger().warn(f'Socket Argus padrão não encontrado em {socket_path}. Verificando alternativos...')
            found = False
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    self.get_logger().info(f'Socket Argus encontrado em: {alt_path}. Tentando criar link...')
                    try:
                        os.symlink(alt_path, socket_path)
                        self.get_logger().info(f'Link simbólico criado: {socket_path} -> {alt_path}')
                        found = True
                        break
                    except Exception as e:
                        self.get_logger().warn(f'Não foi possível criar link simbólico para {socket_path}: {str(e)}')
            if not found:
                 self.get_logger().error(f'Socket Argus não encontrado em nenhum local esperado. nvarguscamerasrc provavelmente falhará.')

    def check_nvargus_permissions(self):
        """Verifica permissões de dispositivos NVIDIA necessários pelo Argus."""
        self.get_logger().info("Verificando permissões dos dispositivos nvhost...")
        # Lista pode variar um pouco com a versão do JetPack
        devices = glob.glob('/dev/nvhost-*') + glob.glob('/dev/nvmap') + glob.glob('/dev/nvidia*')
        if not devices:
             self.get_logger().warn("Nenhum dispositivo /dev/nvhost-* ou /dev/nvidia* encontrado. Isso é inesperado na Jetson.")
             return

        all_ok = True
        for device in devices:
            if os.path.exists(device):
                readable = os.access(device, os.R_OK)
                writable = os.access(device, os.W_OK)
                if not readable or not writable:
                    self.get_logger().warn(f'Permissões insuficientes para {device}: Read={readable}, Write={writable}. GStreamer pode falhar.')
                    all_ok = False
                else:
                    self.get_logger().debug(f'Permissões OK para {device}')
            else:
                 self.get_logger().debug(f"Dispositivo {device} não encontrado (pode ser normal).")

        if all_ok:
             self.get_logger().info("Permissões básicas dos dispositivos NVIDIA parecem OK.")


    def check_nvargus_daemon(self):
        """Tenta verificar o serviço nvargus-daemon (melhor esforço)."""
        # Nota: systemctl geralmente não funciona bem dentro de containers padrão.
        self.get_logger().info('Verificando status do serviço nvargus-daemon (melhor esforço)...')
        try:
            # Tenta via systemctl (pode falhar no container)
            status = subprocess.run(['systemctl', 'is-active', 'nvargus-daemon'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
            stdout = status.stdout.decode().strip() if status.stdout else ""
            if status.returncode == 0 and stdout == 'active':
                self.get_logger().info('Serviço nvargus-daemon está ativo (via systemctl).')
                return True
            else:
                self.get_logger().warn('Serviço nvargus-daemon não está ativo ou systemctl falhou.')
                # Tentar verificar via ps (mais provável de funcionar no container se o daemon foi iniciado de outra forma)
                try:
                     ps_check = subprocess.run(['pgrep', '-f', 'nvargus-daemon'], check=True, stdout=subprocess.PIPE)
                     self.get_logger().info(f"Processo nvargus-daemon encontrado (PID: {ps_check.stdout.decode().strip()}).")
                     return True
                except (subprocess.SubprocessError, FileNotFoundError):
                     self.get_logger().error("Processo nvargus-daemon não encontrado via pgrep.")
                     self.get_logger().error("O daemon PRECISA estar rodando (geralmente no host) para a câmera CSI funcionar.")
                     return False
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            self.get_logger().warn(f'Falha ao verificar nvargus-daemon (systemctl pode não estar disponível no container): {e}')
            # Não impede a continuação, mas registra o aviso.
            return False # Indica que não foi possível confirmar
        except Exception as e:
             self.get_logger().error(f"Erro inesperado ao verificar nvargus-daemon: {e}")
             return False

    def load_camera_calibration(self):
        """Carrega o arquivo de calibração da câmera, se especificado."""
        calibration_file = self.get_parameter('calibration_file').value
        if calibration_file and os.path.isfile(calibration_file):
            self.get_logger().info(f"Carregando arquivo de calibração: {calibration_file}")
            try:
                with open(calibration_file, 'r') as f:
                    calib_data = yaml.safe_load(f)
                    if 'camera_matrix' in calib_data and 'distortion_coefficients' in calib_data:
                        matrix = calib_data['camera_matrix']['data']
                        dist = calib_data['distortion_coefficients']['data']

                        self.camera_info_msg = CameraInfo()
                        self.camera_info_msg.width = calib_data.get('image_width', self.width)
                        self.camera_info_msg.height = calib_data.get('image_height', self.height)
                        self.camera_info_msg.k = [float(x) for x in matrix]
                        self.camera_info_msg.d = [float(x) for x in dist]
                        # Assumindo modelo Pinhole por padrão
                        self.camera_info_msg.distortion_model = calib_data.get('distortion_model', 'plumb_bob')

                        # P, R geralmente são para câmera estéreo, mas preenchemos se disponíveis
                        if 'projection_matrix' in calib_data:
                            self.camera_info_msg.p = [float(x) for x in calib_data['projection_matrix']['data']]
                        else: # Criar P a partir de K se não fornecido
                             self.camera_info_msg.p = [self.camera_info_msg.k[0], 0.0, self.camera_info_msg.k[2], 0.0,
                                                      0.0, self.camera_info_msg.k[4], self.camera_info_msg.k[5], 0.0,
                                                      0.0, 0.0, 1.0, 0.0]

                        if 'rectification_matrix' in calib_data: # R é identidade para mono
                             self.camera_info_msg.r = [float(x) for x in calib_data['rectification_matrix']['data']]
                        else:
                            self.camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] # Identidade

                        self.get_logger().info("Calibração da câmera carregada com sucesso.")
                    else:
                        self.get_logger().error("Arquivo de calibração YAML inválido. Faltando 'camera_matrix' ou 'distortion_coefficients'.")
                        self.camera_info_msg = None
            except (yaml.YAMLError, IOError, ValueError) as e:
                self.get_logger().error(f"Erro ao carregar ou processar arquivo de calibração: {str(e)}")
                self.camera_info_msg = None
        elif calibration_file:
            self.get_logger().warn(f"Arquivo de calibração especificado '{calibration_file}' não encontrado.")
            self.camera_info_msg = None
        else:
            self.get_logger().info("Nenhum arquivo de calibração especificado. CameraInfo não será publicado com dados de calibração.")
            self.camera_info_msg = None # Garantir que esteja None

    def publish_camera_info(self, header=None):
        """Publica a mensagem CameraInfo (se disponível)."""
        if self.camera_info_msg:
            # Atualizar o timestamp com o do frame correspondente, se fornecido
            if header:
                 self.camera_info_msg.header = header
            else:
                 # Se não houver header do frame, usar o tempo atual
                 self.camera_info_msg.header.stamp = self.get_clock().now().to_msg()
                 self.camera_info_msg.header.frame_id = "camera_optical_frame" # Ou nome apropriado

            self.camera_info_pub.publish(self.camera_info_msg)


    def _configure_processing(self):
        """Configura opções de processamento (ex: CUDA)."""
        # A disponibilidade do CUDA (_cuda_available) é verificada em _build_gstreamer_pipeline
        # Aqui apenas usamos a flag resultante
        self.use_cuda = self.get_parameter('enable_cuda').value and self._cuda_available
        self.get_logger().info(f"Processamento de frame via CUDA: {'Habilitado' if self.use_cuda else 'Desabilitado'}")


    def capture_loop(self):
        """Loop principal para capturar frames da câmera."""
        self.get_logger().info("Iniciando loop de captura de frames...")
        while rclpy.ok() and self.is_running:
            if self.cap and self.cap.isOpened():
                start_time = time.time()
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    # Processar e publicar
                    self.process_and_publish_frame(frame)

                    # Calcular FPS
                    self.frame_count += 1
                    now = self.get_clock().now()
                    elapsed = (now.nanoseconds - self.last_fps_update.nanoseconds) / 1e9
                    if elapsed >= 1.0:
                        self.current_fps = self.frame_count / elapsed
                        self.get_logger().debug(f"FPS: {self.current_fps:.1f}")
                        self.frame_count = 0
                        self.last_fps_update = now

                    # Pequena pausa para evitar uso excessivo de CPU se GStreamer não limitar FPS
                    elapsed_proc = time.time() - start_time
                    target_period = 1.0 / self.camera_fps
                    sleep_time = target_period - elapsed_proc
                    if sleep_time > 0.001: # Dormir apenas se houver tempo significativo
                         time.sleep(sleep_time)

                else:
                    self.get_logger().warn("Falha ao ler frame do pipeline GStreamer. Tentando novamente...")
                    time.sleep(0.1) # Esperar um pouco antes de tentar novamente
                    # Considerar reabrir o pipeline se falhar consistentemente?
                    # if not self.cap.isOpened(): self.init_gstreamer_camera() # Exemplo de reabertura
            else:
                self.get_logger().error("VideoCapture não está aberto no loop de captura. Parando a thread.")
                self.is_running = False
                break # Sair do loop se a câmera não estiver aberta

        self.get_logger().info("Loop de captura de frames encerrado.")
        # Garantir que a câmera seja liberada ao sair da thread
        if self.cap:
             self.cap.release()
             self.cap = None
             self.get_logger().info("Recursos da câmera GStreamer liberados.")

    def process_and_publish_frame(self, frame):
        """Processa um frame (opcionalmente com CUDA) e publica."""
        try:
            # Processamento opcional (ex: undistort, filtros)
            processed_frame = frame # Começa com o frame original

            # Aplicar Undistortion se a calibração foi carregada
            if self.camera_info_msg and self.camera_info_msg.k != [0.0]*9:
                 K = np.array(self.camera_info_msg.k).reshape((3,3))
                 D = np.array(self.camera_info_msg.d)
                 # Tentar usar a versão CUDA se disponível, senão CPU
                 if self.use_cuda:
                      try:
                           if not hasattr(self, 'gpu_frame'): self.gpu_frame = cv2.cuda_GpuMat()
                           if not hasattr(self, 'gpu_undistorted'): self.gpu_undistorted = cv2.cuda_GpuMat()
                           self.gpu_frame.upload(frame)
                           cv2.cuda.undistort(self.gpu_frame, K, D, self.gpu_undistorted)
                           processed_frame = self.gpu_undistorted.download()
                           self.get_logger().debug("Frame undistorted usando CUDA.")
                      except cv2.error as e:
                           self.get_logger().warn(f"Falha no undistort CUDA ({e}), usando CPU.")
                           processed_frame = cv2.undistort(frame, K, D)
                 else:
                      processed_frame = cv2.undistort(frame, K, D)
                      self.get_logger().debug("Frame undistorted usando CPU.")


            # Processamento Adicional (Exemplos CUDA/CPU)
            if self.use_cuda:
                # Coloque aqui filtros CUDA (denoise, edge enhancement) se habilitados
                # Exemplo: if self.get_parameter('enable_noise_reduction').value: ...
                # Precisa de upload/download ou operação direta em GpuMat
                pass # Adicionar lógica CUDA aqui se necessário
            else:
                 # Coloque aqui filtros CPU se habilitados
                 if self.get_parameter('enable_noise_reduction').value:
                      processed_frame = cv2.bilateralFilter(processed_frame, 5, 75, 75)
                      self.get_logger().debug("Aplicado bilateralFilter (CPU).")
                 if self.get_parameter('enable_edge_enhancement').value:
                      kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                      processed_frame = cv2.filter2D(processed_frame, -1, kernel)
                      self.get_logger().debug("Aplicado sharpen (CPU).")


            # Criar mensagem ROS
            timestamp = self.get_clock().now().to_msg()
            img_msg = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
            img_msg.header.stamp = timestamp
            img_msg.header.frame_id = "camera_optical_frame"

            # Publicar
            self.image_pub.publish(img_msg)

            # Publicar CameraInfo correspondente (usa o header da imagem)
            self.publish_camera_info(img_msg.header)

            # Mostrar imagem se display estiver habilitado
            if self.get_parameter('display').value:
                try:
                    # Adicionar FPS ao display
                    display_frame = processed_frame.copy()
                    fps_text = f"FPS: {self.current_fps:.1f}"
                    cv2.putText(display_frame, fps_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('IMX219 Camera (Simplificado)', display_frame)
                    cv2.waitKey(1)
                except Exception as e:
                     self.get_logger().warn(f"Erro ao exibir imagem: {e}")
                     # Desabilitar display para evitar mais erros
                     # self.set_parameters([rclpy.parameter.Parameter('display', rclpy.Parameter.Type.BOOL, False)])


        except CvBridgeError as e:
            self.get_logger().error(f"Erro no CvBridge: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"Erro inesperado no processamento/publicação do frame: {str(e)}")
            traceback.print_exc()

    def destroy_node(self):
        """Limpa recursos ao encerrar."""
        self.get_logger().info("Encerrando o nó da câmera...")
        self.is_running = False # Sinaliza para a thread parar
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.get_logger().info("Aguardando a thread de captura encerrar...")
            self.capture_thread.join(timeout=2.0) # Espera até 2 segundos
            if self.capture_thread.is_alive():
                 self.get_logger().warn("Thread de captura não encerrou graciosamente.")

        # A thread deve liberar self.cap, mas garantimos aqui
        if self.cap is not None:
            self.get_logger().info("Liberando VideoCapture explicitamente...")
            self.cap.release()
            self.cap = None

        if self.get_parameter('display').value:
             cv2.destroyAllWindows()
             self.get_logger().info("Janelas de display fechadas.")

        super().destroy_node()
        self.get_logger().info("Nó da câmera encerrado.")

    # --- Funções de Diagnóstico/Verificação (Mantidas) ---
    def run_process_with_timeout(self, cmd, timeout=5):
        """Executa um processo com timeout. Retorna (sucesso, stdout, stderr)"""
        try:
            self.get_logger().debug(f"Executando comando: {cmd}")
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=os.setsid if sys.platform != "win32" else None # Permite matar o grupo
            )
            stdout, stderr = process.communicate(timeout=timeout)
            stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
            stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""

            if process.returncode == 0:
                self.get_logger().debug(f"Comando '{cmd[:30]}...' sucesso.")
                return True, stdout_text, stderr_text
            else:
                self.get_logger().debug(f"Comando '{cmd[:30]}...' falhou (código {process.returncode}).")
                return False, stdout_text, stderr_text

        except subprocess.TimeoutExpired:
            self.get_logger().warn(f"Timeout ({timeout}s) para comando: {cmd[:30]}... Matando...")
            try:
                 if sys.platform != "win32":
                      os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                 else:
                      process.kill()
            except Exception as kill_error:
                self.get_logger().debug(f"Erro ao matar processo: {str(kill_error)}")
            return False, "", "Timeout expirado"
        except Exception as e:
            self.get_logger().error(f"Erro ao executar comando '{cmd[:30]}...': {str(e)}")
            return False, "", str(e)

    def test_camera_pipeline(self, timeout=5):
        """Testa a pipeline da câmera com timeout. Retorna (sucesso, mensagem_erro)"""
        self.get_logger().info(f"Testando acesso básico à câmera CSI com GStreamer (timeout {timeout}s)...")
        test_cmd = "gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink -v"
        success, stdout, stderr = self.run_process_with_timeout(test_cmd, timeout)

        if success:
            self.get_logger().info("Teste de pipeline GStreamer básico bem-sucedido!")
            return True, ""
        else:
            error_msg = f"Falha no teste básico do GStreamer. Stdout: {stdout[:100]}... Stderr: {stderr[:200]}..."
            self.get_logger().error(error_msg)
            # Tentar dar uma dica mais específica
            if "nvargus-daemon" in stderr or "socket" in stderr:
                 hint = "Verifique se o nvargus-daemon está rodando (no host) e o socket está acessível."
            elif "No such element or plugin 'nvarguscamerasrc'" in stderr:
                 hint = "Plugin nvarguscamerasrc não encontrado. Verifique a instalação da Jetson Multimedia API."
            elif "Failed to query video capabilities" in stderr or "Device '/dev/video0' failed" in stderr:
                 hint = "Problema ao acessar o dispositivo da câmera. Verifique conexões e permissões."
            else:
                 hint = "Verifique os logs do sistema (dmesg) e a instalação do GStreamer/JetPack."
            self.get_logger().error(f"Dica: {hint}")
            return False, error_msg + " " + hint

    def check_cuda_permissions(self):
        """Verifica se CUDA está minimamente funcional."""
        try:
            self.get_logger().info('Verificando disponibilidade e permissões básicas do CUDA...')
            has_cuda = False
            # 1. Verificar OpenCV build
            try:
                cuda_build = cv2.getBuildInformation()
                if 'CUDA:YES' in cuda_build:
                    self.get_logger().info('OpenCV compilado com suporte a CUDA.')
                    # 2. Tentar operação básica CUDA
                    try:
                        dummy = np.zeros((10, 10), dtype=np.uint8)
                        gpu_mat = cv2.cuda_GpuMat()
                        gpu_mat.upload(dummy)
                        gpu_mat.download() # Teste básico de upload/download
                        self.get_logger().info('Operação básica OpenCV CUDA bem-sucedida.')
                        has_cuda = True
                    except cv2.error as e:
                        self.get_logger().warn(f'Falha na operação básica OpenCV CUDA: {str(e)}')
                        if "CUDA driver version is insufficient" in str(e):
                             self.get_logger().error("ERRO: Versão do driver NVIDIA incompatível com a toolkit CUDA usada pelo OpenCV.")
                        elif "system has no CUDA-enabled device" in str(e):
                            self.get_logger().error("ERRO: Nenhum dispositivo CUDA encontrado pelo OpenCV.")
                        else:
                            self.get_logger().error("Verifique a instalação do driver NVIDIA, CUDA Toolkit e compatibilidade com OpenCV.")
                    except Exception as e:
                         self.get_logger().error(f"Erro inesperado ao testar OpenCV CUDA: {e}")
                else:
                    self.get_logger().warn('OpenCV NÃO foi compilado com suporte a CUDA.')

            except Exception as e:
                self.get_logger().error(f'Erro ao obter informações de build do OpenCV: {str(e)}')

            # 3. Verificar dispositivos NVIDIA (alternativa)
            if not has_cuda:
                 cuda_devices_found = False
                 devices = glob.glob('/dev/nvidia*')
                 if devices:
                      self.get_logger().info(f"Dispositivos /dev/nvidia* encontrados: {devices}")
                      cuda_devices_found = True
                 else:
                      self.get_logger().warn("Nenhum dispositivo /dev/nvidia* encontrado.")

                 if cuda_devices_found:
                      # Tentar rodar nvidia-smi (se disponível)
                      try:
                           smi = subprocess.run(['nvidia-smi'], timeout=3, check=True, capture_output=True)
                           self.get_logger().info("Comando 'nvidia-smi' executado com sucesso.")
                           # Poderia tentar analisar a saída, mas só a execução já ajuda
                           # has_cuda = True # Considerar True se nvidia-smi funcionou?
                      except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                           self.get_logger().warn(f"Comando 'nvidia-smi' falhou ou não encontrado: {e}")

            if has_cuda:
                self.get_logger().info('Verificação CUDA indica que deve estar funcional.')
                return True
            else:
                self.get_logger().error('Verificação CUDA falhou. Processamento CUDA será desabilitado.')
                return False

        except Exception as e:
            self.get_logger().error(f'Erro geral ao verificar CUDA: {str(e)}')
            return False


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = IMX219CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Ctrl+C detectado, encerrando...")
    except RuntimeError as e:
         # Captura o erro de inicialização da câmera
         if node:
              node.get_logger().fatal(f"Erro fatal durante a inicialização ou execução: {e}")
         else:
              print(f"Erro fatal antes da inicialização completa do nó: {e}", file=sys.stderr)
         # Garante que o rclpy seja finalizado mesmo com erro grave
         if rclpy.ok():
             rclpy.shutdown()
         sys.exit(1) # Sai com código de erro
    except Exception as e:
         # Captura outras exceções inesperadas
         if node:
              node.get_logger().fatal(f"Erro inesperado não tratado: {e}")
              traceback.print_exc()
         else:
              print(f"Erro inesperado antes da inicialização completa do nó: {e}", file=sys.stderr)
              traceback.print_exc()
         if rclpy.ok():
              rclpy.shutdown()
         sys.exit(1) # Sai com código de erro
    finally:
        # Limpeza final
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Recursos do nó liberados.")

if __name__ == '__main__':
    main()
