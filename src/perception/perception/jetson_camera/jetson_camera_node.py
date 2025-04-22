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
                ('device_id', 0), # sensor-id para nvarguscamerasrc
                ('display', False),
                ('framerate', 30.0),
                ('enable_cuda', True), # Habilitar CUDA por padrão
                ('calibration_file', ''),
                # Parâmetros específicos da câmera CSI IMX219 e GStreamer
                ('camera_mode', 5), # Modo 5 (1280x720@120) ou 2 (1920x1080@30) são comuns
                ('flip_method', 0), # 0=none, 1=counterclockwise, 2=rotate180, 3=clockwise, 4=horizontal, 5=upper-right-diag, 6=vertical, 7=upper-left-diag
                # Parâmetros de controle (alguns podem precisar ser configurados fora, ex: v4l2-ctl)
                ('exposure_time', 13333), # Exemplo: ~1/75s em us
                ('gain', 1.0),
                ('awb_mode', 1), # 0=off, 1=auto, 2=incandescent, etc.
                ('saturation', 1.0),
                # Parâmetros de processamento
                ('enable_noise_reduction', False), # Exemplo de processamento (CPU neste exemplo simplificado)
                ('enable_edge_enhancement', False), # Exemplo de processamento (CPU neste exemplo simplificado)
            ]
        )

        # Variáveis de estado
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.bridge = CvBridge()
        self.camera_info_msg = None
        self.camera_info_pub = None # Será criado se calibração for carregada
        self.image_pub = None
        self.width = 0
        self.height = 0
        self.camera_fps = 30.0
        self.frame_count = 0
        self.last_fps_update = self.get_clock().now()
        self.current_fps = 0.0
        self._cuda_available = False # Será verificado

        # Configurar e inicializar
        if self._configure_camera():
            self._configure_processing()
            self.load_camera_calibration() # Carrega se o arquivo for especificado

            # Criar publicadores
            self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
            if self.camera_info_msg: # Só cria publisher se a calibração foi carregada
                 self.camera_info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)
                 self.camera_info_timer = self.create_timer(1.0, self.publish_camera_info_callback) # Publica info periodicamente

            self.get_logger().info('Nó da câmera IMX219 (simplificado) inicializado e pronto.')

            # Iniciar thread de captura (somente se configuração foi bem sucedida)
            self.is_running = True
            self.capture_thread = threading.Thread(target=self.capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
        else:
            self.get_logger().fatal("Falha na configuração inicial da câmera. O nó não funcionará.")
            # Considerar lançar exceção ou desligar
            # raise RuntimeError("Falha na configuração da câmera")

    def _configure_camera(self):
        """Configura a câmera e inicia o pipeline GStreamer."""
        self.get_logger().info('Configurando a câmera CSI...')

        # Mapeamento de modos da câmera IMX219 (ajuste conforme necessário)
        # (Largura, Altura, FPS Máximo)
        self.camera_modes = {
            0: (3264, 2464, 21),
            1: (3264, 1848, 28),
            2: (1920, 1080, 30),
            3: (1640, 1232, 30),
            4: (1280, 720, 60),
            5: (1280, 720, 120)
        }

        self.camera_mode = self.get_parameter('camera_mode').value
        if self.camera_mode not in self.camera_modes:
             self.get_logger().warn(f"Modo de câmera {self.camera_mode} inválido. Usando modo 5 (1280x720@120).")
             self.camera_mode = 5

        self.width, self.height, self.max_fps = self.camera_modes[self.camera_mode]

        # Ajustar FPS solicitado ao máximo do modo
        requested_fps = self.get_parameter('framerate').value
        self.camera_fps = min(requested_fps, float(self.max_fps))
        if requested_fps > self.max_fps:
             self.get_logger().warn(f"FPS solicitado ({requested_fps}) maior que o máximo do modo ({self.max_fps}). Usando {self.camera_fps} FPS.")
        else:
             self.get_logger().info(f"Usando {self.camera_fps} FPS.")

        # Inicializar câmera GStreamer (único método agora)
        if not self.init_gstreamer_camera():
            self.get_logger().error("Falha ao inicializar a câmera GStreamer.")
            return False # Indica falha na configuração

        self.get_logger().info(f'Câmera GStreamer configurada: {self.width}x{self.height} @ {self.camera_fps}fps')
        return True # Indica sucesso na configuração

    def init_gstreamer_camera(self):
        """Inicializa a câmera GStreamer com o pipeline CSI."""
        self.get_logger().info("Verificando pré-requisitos para câmera CSI GStreamer...")

        # Verificar ambiente (é Jetson? Container?)
        is_jetson = os.path.exists('/etc/nv_tegra_release') or os.path.exists('/etc/nv_boot_control.conf')
        is_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
        if not is_jetson:
             self.get_logger().warn("AVISO: Este nó é otimizado para Jetson com câmera CSI.")

        if is_container:
             self.get_logger().info("Ambiente containerizado detectado.")
             # Verificações específicas do container (daemon, socket, permissões)
             self.check_argus_socket()
             self.check_nvargus_permissions()
             self.check_nvargus_daemon() # Pode não funcionar bem no container

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
            # Liberar câmera anterior, se houver
            if self.cap is not None and self.cap.isOpened():
                self.get_logger().info("Liberando câmera GStreamer anterior...")
                self.cap.release()
                self.cap = None
                time.sleep(0.5) # Pequena pausa

            self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)

            if not self.cap.isOpened():
                self.get_logger().error("cv2.VideoCapture FALHOU ao abrir o pipeline GStreamer.")
                self.log_gstreamer_failure_hints()
                self.test_camera_pipeline(timeout=5) # Testa um pipeline básico para diagnóstico
                return False

            # Verificar se conseguimos ler um frame (importante!)
            self.get_logger().info("Pipeline GStreamer aberto. Tentando ler o primeiro frame...")
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.get_logger().error("FALHA ao ler o primeiro frame do pipeline GStreamer.")
                self.cap.release()
                self.cap = None
                self.log_gstreamer_failure_hints()
                self.test_camera_pipeline(timeout=5)
                return False
            else:
                self.get_logger().info(f"Primeiro frame lido com sucesso! Shape: {frame.shape}")
                del frame # Liberar memória do frame de teste

            self.get_logger().info("Pipeline GStreamer aberto e teste de leitura bem-sucedido.")
            return True

        except Exception as e:
            self.get_logger().error(f"Exceção CRÍTICA ao tentar abrir VideoCapture GStreamer: {str(e)}")
            traceback.print_exc()
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def log_gstreamer_failure_hints(self):
        """Imprime dicas comuns de depuração para falhas do GStreamer."""
        self.get_logger().error("-------------------------------------------------")
        self.get_logger().error("Possíveis Causas da Falha GStreamer/Câmera CSI:")
        self.get_logger().error("  1. Conexão Física: Câmera CSI mal conectada?")
        self.get_logger().error("  2. nvargus-daemon: Serviço não está rodando? (Verifique no HOST se estiver em container)")
        self.get_logger().error("     -> Tente: sudo systemctl restart nvargus-daemon (no host)")
        self.get_logger().error("  3. Permissões: Container Docker tem acesso aos dispositivos? (/dev/nvhost*, /dev/video*)")
        self.get_logger().error("     -> Verifique as flags de --device ou --privileged no docker run.")
        self.get_logger().error("  4. Câmera Ocupada: Outro processo usando a câmera?")
        self.get_logger().error("     -> Tente: sudo pkill -f nvargus | sudo pkill -f gst-launch")
        self.get_logger().error("  5. Instalação: JetPack/GStreamer/Multimedia API instalados corretamente?")
        self.get_logger().error("  6. Logs do Sistema: Verifique 'dmesg' ou 'journalctl' por erros relacionados.")
        self.get_logger().error("-------------------------------------------------")


    def _build_gstreamer_pipeline(self):
        """Constrói a string do pipeline GStreamer CSI com base nos parâmetros."""
        try:
            # Calcular framerate como fração para GStreamer (mantido caso algum elemento futuro precise)
            def gcd(a, b):
                while b: a, b = b, a % b
                return a
            fps_num = int(self.camera_fps * 1000)
            fps_den = 1000
            common = gcd(fps_num, fps_den)
            fps_num //= common
            fps_den //= common

            # Parâmetros de controle
            sensor_id = self.get_parameter('device_id').value
            awb = self.get_parameter('awb_mode').value
            saturation = self.get_parameter('saturation').value
            flip = self.get_parameter('flip_method').value
            # camera_mode é lido do parâmetro no __init__

            # Base do pipeline com nvarguscamerasrc usando sensor-mode
            # Removemos width/height/framerate das caps pois sensor-mode define isso
            pipeline = (
                f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={self.camera_mode} "
                f"wbmode={awb} saturation={saturation} "
                f"! video/x-raw(memory:NVMM) " # Apenas o formato de memória é necessário aqui
                # Conversor NVIDIA com flip
                f"! nvvidconv flip-method={flip} "
            )

            # Adicionar elementos CUDA se habilitado E disponível
            self._cuda_available = self.check_cuda_permissions() # Verificar/cachear status CUDA
            if self.get_parameter('enable_cuda').value and self._cuda_available:
                 self.get_logger().info("Adicionando elementos CUDA ao pipeline GStreamer.")
                 # Conversão otimizada BGRx -> videoconvert -> BGR
                 pipeline += f"! video/x-raw, format=(string)BGRx " # Saída comum do nvvidconv
                 pipeline += f"! videoconvert " # Conversão final CPU para BGR (OpenCV espera BGR)
                 pipeline += f"! video/x-raw, format=(string)BGR "

            else:
                 if self.get_parameter('enable_cuda').value and not self._cuda_available:
                      self.get_logger().warn("CUDA solicitado, mas não disponível/permitido. Usando conversão CPU.")
                 else:
                      self.get_logger().info("CUDA desabilitado. Usando conversão de vídeo via CPU (videoconvert).")
                 # Conversão via CPU: nvvidconv -> BGRx -> videoconvert -> BGR
                 pipeline += f"! video/x-raw, format=(string)BGRx " # Saída comum do nvvidconv
                 pipeline += f"! videoconvert " # Conversão principal na CPU
                 pipeline += f"! video/x-raw, format=(string)BGR "

            # Finalizar com appsink para OpenCV ler
            pipeline += f"! appsink max-buffers=2 drop=true sync=false name=sink emit-signals=true"

            self.get_logger().info(f"Pipeline GStreamer Construído: {pipeline}") # Log do pipeline final
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
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=5)
            self.get_logger().info(f"Plugin '{plugin_name}' encontrado com sucesso!")
            return True
        except FileNotFoundError:
             self.get_logger().error("Comando 'gst-inspect-1.0' não encontrado. GStreamer não está instalado ou não está no PATH.")
             return False
        except subprocess.TimeoutExpired:
             self.get_logger().error(f"Timeout ao verificar plugin '{plugin_name}'.")
             return False
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Plugin GStreamer '{plugin_name}' NÃO encontrado.")
            stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
            self.get_logger().error(f"Erro gst-inspect-1.0: {stderr}")
            return False
        except Exception as e:
            self.get_logger().error(f"Erro inesperado ao verificar plugin '{plugin_name}': {e}")
            return False

    def check_argus_socket(self):
        """Verifica o socket Argus em ambiente containerizado."""
        self.get_logger().info("Verificando socket Argus...")
        # Caminho padrão pode variar com JetPack
        std_path = '/tmp/argus_socket'
        # Caminho alternativo/novo
        alt_path = '/var/nvidia/nvcam/camera-daemon-socket'

        if os.path.exists(std_path):
            self.get_logger().info(f"Socket Argus encontrado em {std_path}")
            return True
        elif os.path.exists(alt_path):
            self.get_logger().info(f"Socket Argus encontrado em {alt_path}. Verifique se é acessível.")
            # Não criar link, apenas informar
            return True
        else:
            self.get_logger().error(f"Socket Argus NÃO encontrado em {std_path} ou {alt_path}.")
            self.get_logger().error("Verifique se o serviço nvargus-daemon está rodando no HOST.")
            return False


    def check_nvargus_permissions(self):
        """Verifica permissões de dispositivos NVIDIA necessários pelo Argus."""
        self.get_logger().info("Verificando permissões dos dispositivos nvhost...")
        # Lista pode variar com a versão do JetPack
        # Adicionar /dev/nvmap e /dev/nvidia* que também podem ser relevantes
        devices = glob.glob('/dev/nvhost-*') + glob.glob('/dev/nvmap') + glob.glob('/dev/nvidia*')
        if not devices:
             self.get_logger().warn("Nenhum dispositivo /dev/nvhost-* ou /dev/nvidia* encontrado. Inesperado na Jetson.")
             return False

        all_ok = True
        for device in devices:
            if os.path.exists(device):
                # Verificar leitura E escrita, pois alguns drivers precisam de ambos
                readable = os.access(device, os.R_OK)
                writable = os.access(device, os.W_OK)
                if not readable or not writable:
                    self.get_logger().warn(f'Permissões INSUFICIENTES para {device}: Read={readable}, Write={writable}. GStreamer pode falhar.')
                    all_ok = False
                else:
                    self.get_logger().debug(f'Permissões OK para {device}')
            # Não reportar erro se um dispositivo específico não existir, pode ser normal
            # else:
            #      self.get_logger().debug(f"Dispositivo {device} não encontrado.")

        if all_ok:
             self.get_logger().info("Permissões básicas dos dispositivos NVIDIA parecem OK.")
             return True
        else:
             self.get_logger().error("Problemas de permissão detectados nos dispositivos NVIDIA. Verifique as flags do container ou grupos de usuário.")
             return False


    def check_nvargus_daemon(self):
        """Tenta verificar o serviço nvargus-daemon (melhor esforço, especialmente em container)."""
        self.get_logger().info('Verificando status do serviço nvargus-daemon (melhor esforço)...')
        daemon_confirmed = False
        try:
            # Tenta via systemctl (geralmente só funciona no host)
            status = subprocess.run(['systemctl', 'is-active', 'nvargus-daemon'],
                                  capture_output=True, text=True, timeout=2)
            if status.returncode == 0 and 'active' in status.stdout:
                self.get_logger().info('Serviço nvargus-daemon está ativo (via systemctl - provavelmente no host).')
                daemon_confirmed = True
            else:
                self.get_logger().debug('systemctl is-active nvargus-daemon falhou ou retornou inativo.')
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            self.get_logger().debug(f'systemctl não disponível ou demorou: {e}. Tentando pgrep.')
        except Exception as e:
             self.get_logger().warn(f'Erro ao executar systemctl: {e}')

        # Se systemctl falhou ou não confirmou, tenta pgrep (mais útil em container)
        if not daemon_confirmed:
             try:
                  # Procurar por processos com 'nvargus-daemon' no nome
                  pgrep_check = subprocess.run(['pgrep', '-af', 'nvargus-daemon'], check=True, capture_output=True, text=True, timeout=2)
                  self.get_logger().info(f"Processo(s) nvargus-daemon encontrado(s) via pgrep:\n{pgrep_check.stdout.strip()}")
                  daemon_confirmed = True
             except FileNotFoundError:
                  self.get_logger().warn("Comando 'pgrep' não encontrado.")
             except subprocess.TimeoutExpired:
                   self.get_logger().warn("Timeout ao executar pgrep.")
             except subprocess.CalledProcessError:
                  self.get_logger().error("Nenhum processo nvargus-daemon encontrado via pgrep.")
                  self.get_logger().error("O daemon PRECISA estar rodando (geralmente no host) para a câmera CSI funcionar.")
             except Exception as e:
                   self.get_logger().error(f"Erro inesperado ao executar pgrep: {e}")

        if not daemon_confirmed:
             self.get_logger().error("Não foi possível confirmar se o nvargus-daemon está rodando.")
             return False
        else:
             self.get_logger().info("Confirmação (via systemctl ou pgrep) de que o nvargus-daemon está rodando.")
             return True

    def load_camera_calibration(self):
        """Carrega o arquivo de calibração da câmera, se especificado."""
        calibration_file = self.get_parameter('calibration_file').value
        if calibration_file and os.path.isfile(calibration_file):
            self.get_logger().info(f"Carregando arquivo de calibração: {calibration_file}")
            try:
                with open(calibration_file, 'r') as f:
                    # Usar Loader=yaml.SafeLoader por segurança
                    calib_data = yaml.load(f, Loader=yaml.SafeLoader)
                    # Validar estrutura mínima
                    if not isinstance(calib_data, dict) or \
                       'camera_matrix' not in calib_data or \
                       'distortion_coefficients' not in calib_data or \
                       'data' not in calib_data['camera_matrix'] or \
                       'data' not in calib_data['distortion_coefficients']:
                        self.get_logger().error("Arquivo de calibração YAML inválido ou faltando campos obrigatórios (camera_matrix.data, distortion_coefficients.data).")
                        self.camera_info_msg = None
                        return

                    matrix = calib_data['camera_matrix']['data']
                    dist = calib_data['distortion_coefficients']['data']

                    # Validar tamanho dos dados
                    if len(matrix) != 9 or len(dist) < 4: # Pelo menos k1,k2,p1,p2
                        self.get_logger().error("Dados de matriz (9) ou distorção (>=4) inválidos no arquivo de calibração.")
                        self.camera_info_msg = None
                        return

                    self.camera_info_msg = CameraInfo()
                    # Usar a resolução real da câmera, não a do arquivo, a menos que não tenhamos ainda
                    self.camera_info_msg.width = calib_data.get('image_width', self.width if self.width else 0)
                    self.camera_info_msg.height = calib_data.get('image_height', self.height if self.height else 0)
                    # Garantir que a resolução seja válida
                    if self.camera_info_msg.width == 0 or self.camera_info_msg.height == 0:
                         self.get_logger().error("Largura/Altura da câmera inválida ao carregar calibração.")
                         self.camera_info_msg = None
                         return

                    self.camera_info_msg.k = [float(x) for x in matrix]
                    self.camera_info_msg.d = [float(x) for x in dist]
                    # Assumir modelo Pinhole (plumb_bob) se não especificado
                    self.camera_info_msg.distortion_model = calib_data.get('distortion_model', 'plumb_bob')

                    # P, R são para câmera estéreo, preencher com valores mono padrão se ausentes
                    if 'projection_matrix' in calib_data and 'data' in calib_data['projection_matrix'] and len(calib_data['projection_matrix']['data']) == 12:
                        self.camera_info_msg.p = [float(x) for x in calib_data['projection_matrix']['data']]
                    else: # Criar P a partir de K (assumindo Tx=Ty=Tz=0)
                         self.camera_info_msg.p = [self.camera_info_msg.k[0], 0.0, self.camera_info_msg.k[2], 0.0,
                                                   0.0, self.camera_info_msg.k[4], self.camera_info_msg.k[5], 0.0,
                                                   0.0, 0.0, 1.0, 0.0]

                    if 'rectification_matrix' in calib_data and 'data' in calib_data['rectification_matrix'] and len(calib_data['rectification_matrix']['data']) == 9:
                         self.camera_info_msg.r = [float(x) for x in calib_data['rectification_matrix']['data']]
                    else: # R é identidade para câmera mono
                        self.camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

                    self.get_logger().info("Calibração da câmera carregada com sucesso.")

            except (yaml.YAMLError, IOError, ValueError, TypeError, KeyError) as e:
                self.get_logger().error(f"Erro ao carregar ou processar arquivo de calibração '{calibration_file}': {str(e)}")
                self.camera_info_msg = None
        elif calibration_file:
            self.get_logger().warn(f"Arquivo de calibração especificado '{calibration_file}' não encontrado ou inválido.")
            self.camera_info_msg = None
        else:
            self.get_logger().info("Nenhum arquivo de calibração especificado.")
            self.camera_info_msg = None # Garantir que esteja None

    def publish_camera_info_callback(self):
        """Callback do Timer para publicar CameraInfo."""
        if self.camera_info_pub and self.camera_info_msg:
             # Usar tempo atual pois não temos header de frame correspondente neste timer
             self.camera_info_msg.header.stamp = self.get_clock().now().to_msg()
             self.camera_info_msg.header.frame_id = "camera_optical_frame" # Ou nome apropriado
             self.camera_info_pub.publish(self.camera_info_msg)

    def _configure_processing(self):
        """Configura opções de processamento (ex: CUDA)."""
        # A disponibilidade real do CUDA (_cuda_available) é verificada em _build_gstreamer_pipeline
        # A flag self.use_cuda é definida lá também. Aqui apenas logamos.
        # É importante que check_cuda_permissions seja chamado ANTES disto
        # para que self._cuda_available esteja definido.
        self._cuda_available = self.check_cuda_permissions()
        self.use_cuda = self.get_parameter('enable_cuda').value and self._cuda_available
        self.get_logger().info(f"Processamento de frame via CUDA: {'Habilitado' if self.use_cuda else 'Desabilitado'}")


    def capture_loop(self):
        """Loop principal para capturar frames da câmera GStreamer."""
        self.get_logger().info("Iniciando loop de captura de frames...")
        fail_count = 0
        max_fails = 10 # Número de falhas consecutivas antes de parar

        while rclpy.ok() and self.is_running:
            if self.cap and self.cap.isOpened():
                start_time = time.time()
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    fail_count = 0 # Resetar contador de falhas
                    # Processar e publicar
                    self.process_and_publish_frame(frame)

                    # Calcular FPS
                    self.frame_count += 1
                    now = self.get_clock().now()
                    elapsed_since_last_update = (now.nanoseconds - self.last_fps_update.nanoseconds) / 1e9
                    if elapsed_since_last_update >= 1.0:
                        self.current_fps = self.frame_count / elapsed_since_last_update
                        self.get_logger().debug(f"FPS: {self.current_fps:.1f}")
                        self.frame_count = 0
                        self.last_fps_update = now

                    # Adicionar pequena pausa para não sobrecarregar CPU se GStreamer não limitar FPS
                    elapsed_proc = time.time() - start_time
                    target_period = 1.0 / self.camera_fps if self.camera_fps > 0 else 0.033
                    sleep_time = max(0, target_period - elapsed_proc - 0.001) # Subtrai 1ms para overhead
                    if sleep_time > 0.0005: # Dormir apenas se for > 0.5ms
                         time.sleep(sleep_time)

                else:
                    fail_count += 1
                    self.get_logger().warn(f"Falha ao ler frame {fail_count}/{max_fails} do GStreamer. Ret={ret}, Frame is None={frame is None}")
                    if not self.cap.isOpened():
                         self.get_logger().error("VideoCapture não está mais aberto! Parando thread.")
                         self.is_running = False
                         break
                    if fail_count >= max_fails:
                         self.get_logger().error(f"Falha ao ler frame {max_fails} vezes consecutivas. Parando thread.")
                         self.is_running = False
                         # Considerar tentar reabrir a câmera aqui se desejado
                         break
                    time.sleep(0.1) # Esperar um pouco antes de tentar novamente

            else:
                # Isso não deveria acontecer se a inicialização foi bem sucedida
                self.get_logger().error("VideoCapture não está inicializado ou aberto no loop de captura. Parando thread.")
                self.is_running = False
                break # Sair do loop

        self.get_logger().info("Loop de captura de frames encerrado.")
        # Garantir que a câmera seja liberada ao sair da thread
        if self.cap and self.cap.isOpened():
             self.get_logger().info("Liberando recursos da câmera GStreamer na thread...")
             self.cap.release()
             self.cap = None


    def process_and_publish_frame(self, frame):
        """Processa um frame (opcionalmente com undistort/filtros) e publica."""
        try:
            processed_frame = frame # Começa com o frame original

            # Aplicar Undistortion se a calibração foi carregada
            # TODO: Implementar undistort via CUDA se self.use_cuda for True
            if self.camera_info_msg and hasattr(self.camera_info_msg, 'k') and self.camera_info_msg.k != [0.0]*9:
                 self.get_logger().debug("Aplicando undistortion (CPU)...")
                 K = np.array(self.camera_info_msg.k).reshape((3,3))
                 D = np.array(self.camera_info_msg.d)
                 # Otimização: Calcular mapas de undistort uma vez se a resolução não mudar
                 # if not hasattr(self, 'undistort_map1') or self.width != self._last_undistort_width:
                 #    self.undistort_map1, self.undistort_map2 = cv2.initUndistortRectifyMap(K, D, None, K, (self.width, self.height), cv2.CV_32FC1)
                 #    self._last_undistort_width = self.width
                 # processed_frame = cv2.remap(frame, self.undistort_map1, self.undistort_map2, cv2.INTER_LINEAR)
                 processed_frame = cv2.undistort(frame, K, D, None, K) # Simples por enquanto


            # Processamento Adicional (Exemplos CPU)
            # TODO: Implementar filtros via CUDA se self.use_cuda for True
            if not self.use_cuda: # Só aplica filtros CPU se CUDA não estiver fazendo
                 if self.get_parameter('enable_noise_reduction').value:
                      processed_frame = cv2.bilateralFilter(processed_frame, 5, 75, 75)
                      self.get_logger().debug("Aplicado bilateralFilter (CPU).")
                 if self.get_parameter('enable_edge_enhancement').value:
                      kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                      processed_frame = cv2.filter2D(processed_frame, -1, kernel)
                      self.get_logger().debug("Aplicado sharpen (CPU).")


            # Criar mensagem ROS
            timestamp = self.get_clock().now().to_msg()
            try:
                 img_msg = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
                 img_msg.header.stamp = timestamp
                 img_msg.header.frame_id = "camera_optical_frame"

                 # Publicar imagem (verificar se publisher existe)
                 if self.image_pub:
                      self.image_pub.publish(img_msg)
                 else:
                      self.get_logger().warn("Image publisher não inicializado!")

                 # CameraInfo é publicado por um timer separado

            except CvBridgeError as e:
                 self.get_logger().error(f"Erro no CvBridge: {str(e)}")


            # Mostrar imagem se display estiver habilitado
            if self.get_parameter('display').value:
                try:
                    # Adicionar FPS ao display
                    display_frame = processed_frame.copy()
                    fps_text = f"FPS: {self.current_fps:.1f}"
                    cv2.putText(display_frame, fps_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('IMX219 Camera (Simplificado)', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    # Permitir fechar a janela com 'q' ou ESC
                    if key == ord('q') or key == 27:
                         self.get_logger().info("Tecla 'q' ou ESC pressionada. Desabilitando display.")
                         # Desabilitar display via parâmetro
                         self.set_parameters([rclpy.parameter.Parameter('display', rclpy.Parameter.Type.BOOL, False)])
                         cv2.destroyAllWindows()

                except Exception as e:
                     self.get_logger().warn(f"Erro ao exibir imagem: {e}. Desabilitando display.")
                     # Desabilitar display se houver erro
                     self.set_parameters([rclpy.parameter.Parameter('display', rclpy.Parameter.Type.BOOL, False)])
                     try:
                          cv2.destroyAllWindows()
                     except: pass # Ignorar erros ao fechar


        except Exception as e:
            self.get_logger().error(f"Erro INESPERADO no processamento/publicação do frame: {str(e)}")
            traceback.print_exc()


    def destroy_node(self):
        """Limpa recursos ao encerrar."""
        self.get_logger().info("Encerrando o nó da câmera...")
        self.is_running = False # Sinaliza para a thread parar
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.get_logger().info("Aguardando a thread de captura encerrar (timeout 2s)...")
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                 self.get_logger().warn("Thread de captura não encerrou graciosamente.")

        # A thread deveria liberar self.cap, mas garantimos aqui
        if self.cap is not None and self.cap.isOpened():
            self.get_logger().info("Liberando VideoCapture explicitamente...")
            self.cap.release()
            self.cap = None

        # Parar timers
        if hasattr(self, 'camera_info_timer'):
             self.camera_info_timer.cancel()

        # Fechar janelas de display
        if self.get_parameter('display').value: # Verificar se ainda está ativo
             try:
                cv2.destroyAllWindows()
                self.get_logger().info("Janelas de display fechadas.")
             except: pass # Ignorar erros

        super().destroy_node()
        self.get_logger().info("Nó da câmera encerrado.")


    # --- Funções de Diagnóstico/Verificação (Mantidas e Refinadas) ---
    def run_process_with_timeout(self, cmd, timeout=5):
        """Executa um processo com timeout. Retorna (sucesso, stdout, stderr)"""
        try:
            # Usar shlex para segurança se o comando for complexo, mas aqui é simples
            self.get_logger().debug(f"Executando comando (timeout {timeout}s): {cmd}")
            process = subprocess.Popen(
                cmd, shell=True, # Shell=True é necessário para alguns comandos GStreamer
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=os.setsid if sys.platform != "win32" else None # Permite matar o grupo de processos
            )
            stdout, stderr = process.communicate(timeout=timeout)
            stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
            stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""

            if process.returncode == 0:
                self.get_logger().debug(f"Comando '{cmd[:40]}...' sucesso.")
                return True, stdout_text, stderr_text
            else:
                self.get_logger().warn(f"Comando '{cmd[:40]}...' falhou (código {process.returncode}).")
                self.get_logger().warn(f"Stderr: {stderr_text}")
                return False, stdout_text, stderr_text

        except subprocess.TimeoutExpired:
            self.get_logger().error(f"Timeout ({timeout}s) para comando: {cmd[:40]}... Matando processo...")
            try:
                 # Tentar matar o grupo de processos
                 if sys.platform != "win32" and hasattr(os, 'killpg'):
                      os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                 else: # Fallback para Windows ou se getpgid falhar
                      process.kill()
            except Exception as kill_error:
                self.get_logger().error(f"Erro ao matar processo {process.pid}: {kill_error}")
            return False, "", "Timeout expirado"
        except FileNotFoundError:
             self.get_logger().error(f"Comando não encontrado: {cmd.split()[0]}. Verifique a instalação e o PATH.")
             return False, "", "Comando não encontrado"
        except Exception as e:
            self.get_logger().error(f"Erro inesperado ao executar comando '{cmd[:40]}...': {str(e)}")
            return False, "", str(e)


    def test_camera_pipeline(self, timeout=5):
        """Testa a pipeline da câmera com timeout. Retorna (sucesso, mensagem_erro)"""
        self.get_logger().info(f"Testando acesso básico à câmera CSI com GStreamer (timeout {timeout}s)...")
        # Pipeline de teste mínimo que usa nvarguscamerasrc
        test_cmd = "gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink -v"
        success, stdout, stderr = self.run_process_with_timeout(test_cmd, timeout)

        if success:
            self.get_logger().info("Teste de pipeline GStreamer básico BEM-SUCEDIDO!")
            return True, ""
        else:
            # Analisar o erro para diagnóstico mais preciso
            error_msg = f"FALHA no teste básico do GStreamer. Erro: {stderr}" if stderr else "FALHA no teste básico do GStreamer (sem stderr)"
            self.get_logger().error(error_msg)
            hint = "Causa raiz desconhecida."
            if stderr:
                if "nvargus-daemon" in stderr or "socket" in stderr:
                     hint = "Verifique se o nvargus-daemon está rodando (no host) e o socket está acessível."
                elif "No such element or plugin 'nvarguscamerasrc'" in stderr:
                     hint = "Plugin nvarguscamerasrc não encontrado. Verifique a instalação da Jetson Multimedia API."
                elif "Failed to query video capabilities" in stderr or "Device '/dev/video0' failed" in stderr or "driver" in stderr:
                     hint = "Problema ao acessar o dispositivo/driver da câmera. Verifique conexões, permissões e logs do kernel (dmesg)."
                elif "timeout" in stderr.lower():
                     hint = "Timeout ao executar o teste GStreamer."
                elif "Cannot identify device" in stderr:
                     hint = "Dispositivo da câmera não identificado corretamente."
            self.get_logger().error(f"Dica: {hint}")
            return False, error_msg + " | " + hint


    def check_cuda_permissions(self):
        """Verifica se CUDA está minimamente funcional."""
        # Evitar verificações repetidas
        if hasattr(self, '_cuda_checked') and self._cuda_checked:
             return self._cuda_available

        self.get_logger().info('Verificando disponibilidade e funcionalidade básica do CUDA...')
        self._cuda_available = False # Assume False até ser provado True
        try:
            # 1. Verificar OpenCV build (rápido)
            cuda_build_info = cv2.getBuildInformation()
            if 'CUDA:' not in cuda_build_info or 'YES' not in cuda_build_info.split('CUDA:')[1].split('\n')[0]:
                 self.get_logger().warn('OpenCV NÃO foi compilado com suporte a CUDA. CUDA desabilitado para OpenCV.')
                 self._cuda_checked = True
                 return False

            self.get_logger().info('OpenCV compilado com suporte a CUDA.')

            # 2. Tentar operação básica CUDA via OpenCV (teste mais concreto)
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() == 0:
                    self.get_logger().error("Nenhum dispositivo CUDA encontrado pelo OpenCV.")
                    self._cuda_checked = True
                    return False

                self.get_logger().info(f"Dispositivos CUDA encontrados pelo OpenCV: {cv2.cuda.getCudaEnabledDeviceCount()}")
                # Teste simples de alocação/upload/download
                dummy = np.zeros((10, 10), dtype=np.uint8)
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(dummy)
                _ = gpu_mat.download() # Descarrega o resultado
                self.get_logger().info('Operação básica OpenCV CUDA (upload/download) bem-sucedida.')
                self._cuda_available = True # CUDA está funcional para OpenCV

            except cv2.error as e:
                self.get_logger().error(f'Falha na operação básica OpenCV CUDA: {str(e)}')
                if "CUDA driver version is insufficient" in str(e):
                     self.get_logger().error("ERRO CRÍTICO: Versão do driver NVIDIA incompatível com a toolkit CUDA usada pelo OpenCV.")
                elif "system has no CUDA-enabled device" in str(e):
                     self.get_logger().error("ERRO CRÍTICO: Nenhum dispositivo CUDA encontrado pelo OpenCV (apesar da compilação).")
                else:
                     self.get_logger().error("Verifique instalação do driver NVIDIA, CUDA Toolkit e compatibilidade com OpenCV.")
                self._cuda_available = False
            except Exception as e:
                 self.get_logger().error(f"Erro inesperado ao testar OpenCV CUDA: {e}")
                 self._cuda_available = False

            # 3. Verificar dispositivos /dev/nvidia* (informativo)
            devices = glob.glob('/dev/nvidia*')
            if devices:
                 self.get_logger().info(f"Dispositivos /dev/nvidia* encontrados: {devices}")
                 # Poderia verificar permissões aqui também se necessário
            else:
                 self.get_logger().warn("Nenhum dispositivo /dev/nvidia* encontrado (pode ser normal se CUDA ainda funcionar).")

            # 4. Tentar nvidia-smi (informativo, pode não estar instalado)
            try:
                smi = subprocess.run(['nvidia-smi'], timeout=3, check=True, capture_output=True, text=True)
                self.get_logger().info(f"Comando 'nvidia-smi' executado com sucesso:\n{smi.stdout[:200]}...")
            except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                self.get_logger().info(f"Comando 'nvidia-smi' falhou ou não encontrado (informativo): {e}")
            except Exception as e:
                 self.get_logger().warn(f"Erro ao executar nvidia-smi: {e}")


        except Exception as e:
            self.get_logger().error(f'Erro geral ao verificar CUDA: {str(e)}')
            self._cuda_available = False

        self._cuda_checked = True # Marcar que a verificação foi feita
        if self._cuda_available:
            self.get_logger().info('Verificação CUDA indica que está funcional para OpenCV.')
        else:
             self.get_logger().error('Verificação CUDA falhou ou indicou indisponibilidade. Processamento CUDA será desabilitado.')
        return self._cuda_available


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = IMX219CameraNode()
        # O spin só começa se a inicialização for bem-sucedida (sem exceção)
        if node and node.is_running: # Verifica se a thread de captura iniciou
            rclpy.spin(node)
        else:
            print("Falha na inicialização do nó. Encerrando.", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nCtrl+C detectado, encerrando...", file=sys.stderr)
    except RuntimeError as e:
         # Captura o erro de inicialização da câmera ou outros RuntimeErrors
         log_func = node.get_logger().fatal if node else lambda msg: print(f"FATAL: {msg}", file=sys.stderr)
         log_func(f"Erro fatal durante a inicialização ou execução: {e}")
         # Garante que o rclpy seja finalizado
         if rclpy.ok(): rclpy.shutdown()
         sys.exit(f"Erro Runtime: {e}") # Sai com código de erro e mensagem
    except Exception as e:
         # Captura outras exceções inesperadas
         log_func = node.get_logger().fatal if node else lambda msg: print(f"FATAL: {msg}", file=sys.stderr)
         log_func(f"Erro inesperado não tratado: {e}")
         traceback.print_exc()
         if rclpy.ok(): rclpy.shutdown()
         sys.exit(f"Erro Inesperado: {e}") # Sai com código de erro
    finally:
        # Limpeza final
        if node is not None:
            node.destroy_node() # Chama a lógica de limpeza do nó
        # Garantir que shutdown seja chamado se ainda estiver OK
        if rclpy.ok():
            print("Finalizando rclpy...", file=sys.stderr)
            rclpy.shutdown()
        print("Recursos do nó liberados.", file=sys.stderr)

if __name__ == '__main__':
    main()