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
        
        # Verificar se o plugin nvarguscamerasrc está disponível
        try:
            gst_check = subprocess.check_output(['gst-inspect-1.0', 'nvarguscamerasrc'], stderr=subprocess.STDOUT).decode('utf-8')
            if 'nvarguscamerasrc' not in gst_check:
                self.get_logger().error('Plugin GStreamer nvarguscamerasrc não encontrado. Este plugin é necessário para o funcionamento da câmera CSI.')
                raise RuntimeError('Plugin GStreamer nvarguscamerasrc não encontrado')
        except (subprocess.SubprocessError, FileNotFoundError):
            self.get_logger().error('GStreamer não disponível ou não configurado corretamente.')
            raise RuntimeError('GStreamer não disponível')
            
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
        
        # Pipeline principal com envio para appsink (para ROS) e opcionalmente para ximagesink (display)
        display_enabled = self.get_parameter('enable_display').value
        
        # Construir o pipeline base com nvarguscamerasrc
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
            f"format=(string)NV12, framerate=(fraction){fps_num}/{fps_den} ! "
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
            
        # Converter para BGRx para processamento
        pipeline += "! video/x-raw, format=(string)BGRx ! "
        
        # Usar tee para enviar para appsink e para ximagesink (se display estiver habilitado)
        if display_enabled:
            # Com display ativado, usar tee para dividir o fluxo
            pipeline += (
                "tee name=t ! queue ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! "
                "appsink name=appsink max-buffers=2 drop=true sync=false "
                "t. ! queue ! "
                "videoconvert ! ximagesink sync=false"
            )
        else:
            # Sem display, apenas enviar para appsink
            pipeline += (
                "videoconvert ! video/x-raw, format=(string)BGR ! "
                "appsink name=appsink max-buffers=2 drop=true sync=false"
            )
        
        self.get_logger().info(f'Pipeline GStreamer: {pipeline}')
        
        # Definir variáveis de ambiente para CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        try:
            # Abrir a câmera com o pipeline GStreamer e verificar se foi bem-sucedido
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                self.get_logger().error('Falha ao abrir câmera com o pipeline GStreamer. Verifique se nvarguscamerasrc está instalado e se a câmera CSI está conectada corretamente.')
                
                # Executar gst-launch diretamente para diagnóstico
                try:
                    test_pipeline = f"gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! fakesink"
                    subprocess.check_call(test_pipeline, shell=True, stderr=subprocess.STDOUT)
                    self.get_logger().info('Teste básico de nvarguscamerasrc bem-sucedido.')
                except subprocess.CalledProcessError as e:
                    self.get_logger().error(f'Teste de nvarguscamerasrc falhou: {e.output if hasattr(e, "output") else str(e)}')
                
                raise RuntimeError('Falha ao abrir câmera com GStreamer')
            
            # Configurar cálculo de FPS real
            self.frame_count = 0
            self.last_fps_time = time.time()
            self.real_fps = 0.0
            
            # Ajustar propriedades específicas (embora possa não ter efeito com GStreamer)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Minimizar latência
            
            self.get_logger().info('Câmera inicializada com sucesso usando GStreamer')
            
        except Exception as e:
            self.get_logger().error(f'Exceção ao inicializar câmera: {e}')
            self.debug_camera_devices()
            raise RuntimeError(f'Falha ao abrir câmera: {e}')

    def debug_camera_devices(self):
        """
        Debugar dispositivos de câmera disponíveis.
        Retorna: True se pelo menos um dispositivo de câmera está disponível, False caso contrário
        """
        devices_available = False
        
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
                                self.get_logger().warn(f'Erro ao verificar detalhes de {dev_path}: {e}')
            except Exception as e:
                self.get_logger().warn(f'Erro ao verificar dispositivos: {e}')
            
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
            self.get_logger().error(f'Erro ao debugar dispositivos: {e}')
        
        return devices_available

    def capture_loop(self):
        """Loop principal de captura com processamento CUDA."""
        # Verificar se estamos no modo simulado
        if hasattr(self, 'is_simulated') and self.is_simulated:
            # No modo simulado, a atualização é feita pelo timer
            return
            
        cuda_enabled = self.get_parameter('enable_cuda').value
        
        if cuda_enabled:
            # Inicializar contexto CUDA
            self.cuda_stream = cv2.cuda_Stream()
            self.cuda_upload = cv2.cuda_GpuMat()
            self.cuda_color = cv2.cuda_GpuMat()
        
        # Variáveis para cálculo de FPS real
        fps_update_interval = 1.0  # segundos
        frame_times = []
        max_frame_times = 30  # Para média móvel
        
        while self.is_running and rclpy.ok():
            try:
                # Registrar tempo de início do frame
                frame_start_time = time.time()
                
                # Capturar frame
                ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().warn('Falha ao capturar frame')
                    time.sleep(0.1)
                    continue
                
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
                    self.frame_count = 0
                    self.last_fps_time = current_time
                    
                    # Calcular tempo médio de processamento
                    if frame_times:
                        avg_frame_time = sum(frame_times) / len(frame_times)
                        self.get_logger().debug(f'Tempo médio de processamento: {avg_frame_time*1000:.2f}ms, FPS real: {self.real_fps:.1f}')
                
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
                    # Usar o FPS real calculado em vez de tentar obter da câmera
                    if hasattr(self, 'real_fps'):
                        self.get_logger().info(f'Câmera: FPS real={self.real_fps:.1f}')
                    else:
                        self.get_logger().info(f'Câmera: FPS configurado={self.camera_fps}')
                except Exception as e:
                    self.get_logger().warn(f'Erro ao obter estatísticas da câmera: {e}')
                
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