#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline unificado de visão computacional para o robô de futebol.

Este nó coordena tanto o detector YOEO quanto os detectores tradicionais,
permitindo a alternância entre eles ou uso em conjunto.
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D, PoseArray, Pose
from vision_msgs.msg import Detection2DArray
import threading
import time
import os
import importlib.util
import sys

# Importar componentes YOEO se disponíveis
try:
    from .yoeo.yoeo_detector import YOEOHandler
    from .yoeo.yoeo_model import YOEOModel
    YOEO_AVAILABLE = True
except ImportError:
    try:
        # Tentativa alternativa se o import relativo falhar
        from perception.yoeo.yoeo_detector import YOEOHandler
        from perception.yoeo.yoeo_model import YOEOModel
        YOEO_AVAILABLE = True
    except ImportError:
        YOEO_AVAILABLE = False
        print("AVISO: YOEO não está disponível. Usando apenas detectores tradicionais.")

# Importar detectores tradicionais
try:
    from .ball_detector import BallDetector
    from .field_detector import FieldDetector
    from .line_detector import LineDetector
    from .goal_detector import GoalDetector
    from .obstacle_detector import ObstacleDetector
    TRADITIONAL_DETECTORS_AVAILABLE = True
except ImportError:
    try:
        # Tentativa alternativa se o import relativo falhar
        from perception.ball_detector import BallDetector
        from perception.field_detector import FieldDetector
        from perception.line_detector import LineDetector
        from perception.goal_detector import GoalDetector
        from perception.obstacle_detector import ObstacleDetector
        TRADITIONAL_DETECTORS_AVAILABLE = True
    except ImportError:
        try:
            # Tenta importar de forma direta como último recurso
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Função auxiliar para importar módulos a partir do caminho do arquivo
            def import_from_file(module_name, file_path):
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None:
                    return None
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module
            
            # Importar cada detector individualmente
            ball_detector_path = os.path.join(current_dir, 'ball_detector.py')
            field_detector_path = os.path.join(current_dir, 'field_detector.py')
            line_detector_path = os.path.join(current_dir, 'line_detector.py')
            goal_detector_path = os.path.join(current_dir, 'goal_detector.py')
            obstacle_detector_path = os.path.join(current_dir, 'obstacle_detector.py')
            
            ball_detector_module = import_from_file('ball_detector', ball_detector_path)
            field_detector_module = import_from_file('field_detector', field_detector_path)
            line_detector_module = import_from_file('line_detector', line_detector_path)
            goal_detector_module = import_from_file('goal_detector', goal_detector_path)
            obstacle_detector_module = import_from_file('obstacle_detector', obstacle_detector_path)
            
            if ball_detector_module:
                BallDetector = ball_detector_module.BallDetector
            if field_detector_module:
                FieldDetector = field_detector_module.FieldDetector
            if line_detector_module:
                LineDetector = line_detector_module.LineDetector
            if goal_detector_module:
                GoalDetector = goal_detector_module.GoalDetector
            if obstacle_detector_module:
                ObstacleDetector = obstacle_detector_module.ObstacleDetector
            
            TRADITIONAL_DETECTORS_AVAILABLE = True
            print("Detectores tradicionais importados diretamente dos arquivos.")
        except Exception as e:
            TRADITIONAL_DETECTORS_AVAILABLE = False
            print(f"AVISO: Detectores tradicionais não estão disponíveis. Erro: {str(e)}")

# Adaptadores para os detectores tradicionais
class FieldDetectorAdapter:
    """Adaptador para o detector de campo."""
    
    def __init__(self, parent_node):
        """
        Args:
            parent_node: Nó pai para acesso aos parâmetros
        """
        self.node = parent_node
        
        # Valores HSV ainda mais relaxados para detectar uma ampla gama de verdes
        self.color_lower = np.array([25, 15, 15])  # Valores mais baixos para pegar mais tons
        self.color_upper = np.array([95, 255, 255])  # Valores mais altos para incluir mais variações
        
        # Reduzir bastante o limiar de área mínima
        self.min_field_area_ratio = 0.01  # 1% da imagem é suficiente
        
        # Parâmetros de processamento morfológico
        self.erode_iterations = 1
        self.dilate_iterations = 3
        
    def detect_field(self, image):
        """
        Detecta o campo na imagem usando segmentação por cor.
        
        Args:
            image: Imagem OpenCV no formato BGR
            
        Returns:
            tuple: (máscara_do_campo, fronteira_do_campo, imagem_debug)
        """
        # Criar cópia da imagem para debug
        debug_image = image.copy()
        
        # Converter para HSV para melhor segmentação de cor
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Criar máscara para a cor do campo
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        
        # Aplicar operações morfológicas para remover ruído e preencher lacunas
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=self.erode_iterations)
        mask = cv2.dilate(mask, kernel, iterations=self.dilate_iterations)
        
        # Verificar se a área do campo é suficiente
        field_area_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        
        if field_area_ratio < self.min_field_area_ratio:
            empty_mask = np.zeros_like(mask)
            return empty_mask, empty_mask, debug_image
        
        # Encontrar contornos do campo
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Se não houver contornos, retornar a máscara original
        if not contours:
            return mask, mask, debug_image
        
        # Encontrar o maior contorno (que deve ser o campo)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Criar máscara apenas com o maior contorno
        field_mask = np.zeros_like(mask)
        cv2.drawContours(field_mask, [largest_contour], 0, 255, -1)
        
        # Criar imagem da fronteira do campo
        field_boundary = np.zeros_like(mask)
        cv2.drawContours(field_boundary, [largest_contour], 0, 255, 2)
        
        # Desenhar contorno na imagem de debug
        cv2.drawContours(debug_image, [largest_contour], 0, (0, 255, 0), 2)
        
        # Desenhar a máscara colorida para visualização
        color_mask = np.zeros_like(debug_image)
        color_mask[:,:,1] = field_mask  # Canal verde
        
        # Combinar com a imagem original
        debug_image = cv2.addWeighted(color_mask, 0.3, debug_image, 0.7, 0)
        
        return field_mask, field_boundary, debug_image

class BallDetectorAdapter:
    """Adaptador para o detector de bola."""
    
    def __init__(self, parent_node):
        """
        Args:
            parent_node: Nó pai para acesso aos parâmetros
        """
        self.node = parent_node
        self.color_lower = np.array([0, 120, 70])  # HSV para laranja (bola)
        self.color_upper = np.array([10, 255, 255])
        self.min_radius = 10
        self.max_radius = 100
        
    def detect_ball(self, image):
        """
        Detecta a bola na imagem.
        
        Args:
            image: Imagem OpenCV no formato BGR
            
        Returns:
            tuple: (bola_detectada, posição_bola, raio_bola, imagem_debug)
        """
        # Criar cópia da imagem para debug
        debug_image = image.copy()
        
        # Converter para HSV para melhor segmentação de cor
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Criar máscara para a cor da bola
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        
        # Aplicar operações morfológicas para remover ruído
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Procurar por círculos entre os contornos
        ball_detected = False
        ball_position = (0, 0)
        ball_radius = 0
        
        for contour in contours:
            # Calcular área e perímetro do contorno
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Evitar divisão por zero
            if perimeter == 0:
                continue
            
            # Calcular circularidade
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Verificar se o contorno é suficientemente circular
            if circularity > 0.7:  # Valor de referência para circularidade de um círculo
                # Encontrar o círculo que melhor se ajusta ao contorno
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Verificar se o raio está dentro dos limites esperados
                if self.min_radius < radius < self.max_radius:
                    ball_detected = True
                    ball_position = (int(x), int(y))
                    ball_radius = int(radius)
                    
                    # Desenhar o círculo na imagem de debug
                    cv2.circle(debug_image, ball_position, ball_radius, (0, 255, 255), 2)
                    cv2.circle(debug_image, ball_position, 2, (0, 0, 255), -1)  # Centro
                
                    
                    # Ao encontrar a primeira bola válida, interromper a busca
                    break
        
        if not ball_detected:
            # Adicionar informações à imagem de debug
            cv2.putText(debug_image, 'Bola não encontrada', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return ball_detected, ball_position, ball_radius, debug_image

class VisionPipeline(Node):
    """
    Pipeline unificado de visão computacional para o robô de futebol.
    
    Este nó coordena tanto o detector YOEO quanto os detectores tradicionais,
    permitindo a alternância entre eles ou uso em conjunto.
    """
    
    def __init__(self):
        super().__init__('vision_pipeline')
        
        # Parâmetros básicos
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('debug_image', True)
        self.declare_parameter('processing_fps', 30.0)
        
        # Habilitar/desabilitar sistemas
        self.declare_parameter('use_yoeo', YOEO_AVAILABLE)
        self.declare_parameter('use_traditional', True)
        
        # Parâmetros para YOEO
        self.declare_parameter('yoeo_model_path', './src/perception/resources/models/yolov4_tiny.h5')
        self.declare_parameter('yoeo_confidence_threshold', 0.5)
        self.declare_parameter('use_tensorrt', False)
        
        # Escolha de detector para cada tipo de objeto
        self.declare_parameter('detector_ball', 'traditional')     # 'yoeo' ou 'traditional'
        self.declare_parameter('detector_field', 'traditional')    # 'yoeo' ou 'traditional'
        self.declare_parameter('detector_lines', 'traditional')    # 'yoeo' ou 'traditional'
        self.declare_parameter('detector_goals', 'traditional')    # 'yoeo' ou 'traditional'
        self.declare_parameter('detector_robots', 'traditional')   # 'yoeo' ou 'traditional'
        
        # Detectores tradicionais disponíveis
        self.declare_parameter('enable_ball_detection', True)
        self.declare_parameter('enable_field_detection', True)
        self.declare_parameter('enable_line_detection', True)
        self.declare_parameter('enable_goal_detection', True)
        self.declare_parameter('enable_obstacle_detection', True)
        
        # Obter parâmetros
        self.camera_topic = self.get_parameter('camera_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.debug_image = self.get_parameter('debug_image').value
        self.processing_fps = self.get_parameter('processing_fps').value
        
        # Habilitar sistemas
        self.use_yoeo = self.get_parameter('use_yoeo').value and YOEO_AVAILABLE
        self.use_traditional = self.get_parameter('use_traditional').value
        
        # Configuração YOEO
        self.yoeo_model_path = self.get_parameter('yoeo_model_path').value
        self.yoeo_confidence_threshold = self.get_parameter('yoeo_confidence_threshold').value
        self.use_tensorrt = self.get_parameter('use_tensorrt').value
        
        # Configuração de detectores 
        self.detector_ball = self.get_parameter('detector_ball').value
        self.detector_field = self.get_parameter('detector_field').value
        self.detector_lines = self.get_parameter('detector_lines').value
        self.detector_goals = self.get_parameter('detector_goals').value
        self.detector_robots = self.get_parameter('detector_robots').value
        
        # Detectores tradicionais
        self.enable_ball_detection = self.get_parameter('enable_ball_detection').value
        self.enable_field_detection = self.get_parameter('enable_field_detection').value
        self.enable_line_detection = self.get_parameter('enable_line_detection').value
        self.enable_goal_detection = self.get_parameter('enable_goal_detection').value
        self.enable_obstacle_detection = self.get_parameter('enable_obstacle_detection').value
        
        # Inicializar detectores
        self._init_detectors()
        
        # Publishers
        self._init_publishers()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        # Variáveis
        self.cv_bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_image = None
        self.latest_image_time = None
        self.processing_thread = None
        self.processing_active = False
        
        # Iniciar thread de processamento
        self.start_processing_thread()
        
        self.get_logger().info('Pipeline de visão unificado iniciado')
        self.get_logger().info(f'Usando YOEO: {self.use_yoeo}')
        self.get_logger().info(f'Usando detectores tradicionais: {self.use_traditional}')
        self.get_logger().info(f'Detectores habilitados: ' + 
                              f'Bola={self.enable_ball_detection}, ' +
                              f'Campo={self.enable_field_detection}, ' + 
                              f'Linhas={self.enable_line_detection}, ' +
                              f'Gols={self.enable_goal_detection}, ' +
                              f'Obstáculos={self.enable_obstacle_detection}')
    
    def _init_detectors(self):
        """Inicializa os detectores a serem usados."""
        self.ball_detector = None
        self.field_detector = None
        self.line_detector = None
        self.goal_detector = None
        self.obstacle_detector = None
        self.yoeo_handler = None
        
        # Inicializar YOEO se habilitado
        if self.use_yoeo and YOEO_AVAILABLE:
            try:
                self.get_logger().info(f'Carregando modelo YOEO de: {self.yoeo_model_path}')
                # Criar configuração do modelo
                model_config = {
                    "model_path": self.yoeo_model_path,
                    "input_width": 416,
                    "input_height": 416,
                    "confidence_threshold": self.yoeo_confidence_threshold,
                    "iou_threshold": 0.45,
                    "use_tensorrt": self.use_tensorrt
                }
                # Carregar o modelo usando o manipulador YOEO
                from perception.yoeo.yoeo_handler import YOEOHandler
                self.yoeo_handler = YOEOHandler(model_config)
                self.get_logger().info('Modelo YOEO carregado com sucesso')
            except Exception as e:
                self.get_logger().error(f'Erro ao carregar modelo YOEO: {str(e)}')
                self.use_yoeo = False
        
        # Inicializar detectores tradicionais
        try:
            if self.enable_ball_detection:
                self.get_logger().info('Inicializando detector tradicional de bola')
                self.ball_detector = BallDetectorAdapter(self)
            
            if self.enable_field_detection:
                self.get_logger().info('Inicializando detector tradicional de campo')
                self.field_detector = FieldDetectorAdapter(self)
            
            if self.enable_line_detection:
                self.get_logger().info('Inicializando detector tradicional de linhas')
                # Tentamos inicializar o detector de linhas real
                try:
                    self.line_detector = LineDetector()
                    self.get_logger().info('Detector de linhas inicializado com sucesso')
                except Exception as e:
                    self.get_logger().error(f'Erro ao inicializar detector de linhas: {str(e)}')
                    self.line_detector = None
            
            if self.enable_goal_detection:
                self.get_logger().info('Inicializando detector tradicional de gols')
                # Tentamos inicializar o detector de gols real
                try:
                    self.goal_detector = GoalDetector()
                    self.get_logger().info('Detector de gols inicializado com sucesso')
                except Exception as e:
                    self.get_logger().error(f'Erro ao inicializar detector de gols: {str(e)}')
                    self.goal_detector = None
            
            if self.enable_obstacle_detection:
                self.get_logger().info('Inicializando detector tradicional de obstáculos')
                # Tentamos inicializar o detector de obstáculos real
                try:
                    self.obstacle_detector = ObstacleDetector()
                    self.get_logger().info('Detector de obstáculos inicializado com sucesso')
                except Exception as e:
                    self.get_logger().error(f'Erro ao inicializar detector de obstáculos: {str(e)}')
                    self.obstacle_detector = None
        except Exception as e:
            self.get_logger().error(f'Erro ao inicializar detectores tradicionais: {str(e)}')
            self.use_traditional = False
    
    def _init_publishers(self):
        """Inicializa os publishers para os resultados."""
        # Publisher de debug
        self.debug_image_pub = self.create_publisher(Image, '/vision_debug', 10)
        self.get_logger().info('Publisher de debug criado no tópico: /vision_debug')
        
        # Publishers para detectores tradicionais
        self.ball_position_pub = self.create_publisher(Pose2D, 'ball_position', 10)
        self.field_mask_pub = self.create_publisher(Image, 'field_mask', 10)
        self.lines_image_pub = self.create_publisher(Image, 'lines_image', 10)
        self.goal_posts_pub = self.create_publisher(PoseArray, 'goal_posts', 10)
        self.obstacles_pub = self.create_publisher(PoseArray, 'obstacles', 10)
        
        # Publishers para YOEO
        self.yoeo_detections_pub = self.create_publisher(Detection2DArray, 'yoeo/detections', 10)
        self.yoeo_ball_detections_pub = self.create_publisher(Detection2DArray, 'yoeo/ball_detections', 10)
        self.yoeo_goal_detections_pub = self.create_publisher(Detection2DArray, 'yoeo/goal_detections', 10)
        self.yoeo_robot_detections_pub = self.create_publisher(Detection2DArray, 'yoeo/robot_detections', 10)
        self.yoeo_field_mask_pub = self.create_publisher(Image, 'yoeo/field_mask', 10)
        self.yoeo_line_mask_pub = self.create_publisher(Image, 'yoeo/line_mask', 10)
    
    def camera_info_callback(self, msg):
        """Callback para informações da câmera."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
    
    def image_callback(self, msg):
        """Callback para receber imagens da câmera."""
        # Apenas armazena a imagem mais recente para processamento na thread separada
        self.latest_image = msg
        self.latest_image_time = self.get_clock().now()
    
    def start_processing_thread(self):
        """Inicia a thread de processamento de imagens."""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            return
        
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_images(self):
        """Thread principal de processamento de imagens."""
        period = 1.0 / self.processing_fps
        
        self.get_logger().info('Thread de processamento iniciada')
        
        while self.processing_active and rclpy.ok():
            start_time = time.time()
            
            # Processar a imagem mais recente
            if self.latest_image is not None:
                try:
                    self.get_logger().debug('Processando nova imagem')
                    
                    # Converter ROS Image para OpenCV
                    cv_image = self.cv_bridge.imgmsg_to_cv2(self.latest_image, 'bgr8')
                    
                    # Criar imagem de debug
                    debug_image = cv_image.copy()
                    
                    # Verificar quais sistemas estão disponíveis
                    use_yoeo = self.use_yoeo
                    use_traditional = self.use_traditional
                    
                    # Resultados dos detectores
                    yoeo_results = None
                    traditional_results = None
                    
                    # Processar com os detectores disponíveis
                    if use_yoeo:
                        self.get_logger().debug('Iniciando processamento YOEO')
                        yoeo_results = self._process_with_yoeo(cv_image, debug_image)
                        self.get_logger().debug('YOEO processado com sucesso')
                    
                    # Processar com detectores tradicionais
                    if use_traditional:
                        self.get_logger().debug('Iniciando processamento com detectores tradicionais')
                        traditional_results = self._process_with_traditional(cv_image, debug_image)
                        self.get_logger().debug('Detectores tradicionais processados com sucesso')
                    
                    # Combinar resultados baseado na configuração de preferência para cada objeto
                    results = self._select_best_results(yoeo_results, traditional_results)
                    
                    # Adicionar FPS no canto da imagem (discreto)
                    fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                    cv2.putText(debug_image, f"FPS:{fps:.1f}", (debug_image.shape[1] - 80, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Publicar resultados
                    self._publish_results(results)
                    
                    # Publicar imagem de debug
                    if self.debug_image:
                        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, 'bgr8')
                        debug_msg.header = self.latest_image.header
                        self.debug_image_pub.publish(debug_msg)
                        self.get_logger().debug('Imagem de debug publicada')
                
                except Exception as e:
                    self.get_logger().error(f'Erro no processamento da imagem: {str(e)}')
                    import traceback
                    self.get_logger().error(traceback.format_exc())
            else:
                self.get_logger().debug('Nenhuma imagem disponível para processamento')
            
            # Controlar a taxa de processamento
            elapsed = time.time() - start_time
            if elapsed < period:
                time.sleep(period - elapsed)
    
    def _select_best_results(self, yoeo_results, traditional_results):
        """
        Seleciona os melhores resultados baseados na configuração de detectores.
        Simplificação do sistema de fusão e alternância.
        
        Args:
            yoeo_results: Resultados do detector YOEO
            traditional_results: Resultados dos detectores tradicionais
            
        Returns:
            dict: Resultados combinados dos dois sistemas
        """
        if yoeo_results is None and traditional_results is None:
            return {'success': False}
        
        # Resultado combinado
        results = {'success': True}
        
        # Bola
        if self.detector_ball == 'yoeo' and yoeo_results and 'ball' in yoeo_results:
            results['ball'] = yoeo_results['ball']
        elif traditional_results and 'ball' in traditional_results:
            results['ball'] = traditional_results['ball']
        elif yoeo_results and 'ball' in yoeo_results:  # Fallback para YOEO
            results['ball'] = yoeo_results['ball']
        
        # Campo
        if self.detector_field == 'yoeo' and yoeo_results and 'field_mask' in yoeo_results:
            results['field_mask'] = yoeo_results['field_mask']
        elif traditional_results and 'field_mask' in traditional_results:
            results['field_mask'] = traditional_results['field_mask']
        elif yoeo_results and 'field_mask' in yoeo_results:  # Fallback para YOEO
            results['field_mask'] = yoeo_results['field_mask']
        
        # Linhas
        if self.detector_lines == 'yoeo' and yoeo_results and 'lines' in yoeo_results:
            results['lines'] = yoeo_results['lines']
        elif traditional_results and 'lines' in traditional_results:
            results['lines'] = traditional_results['lines']
        elif yoeo_results and 'lines' in yoeo_results:  # Fallback para YOEO
            results['lines'] = yoeo_results['lines']
        
        # Gols
        if self.detector_goals == 'yoeo' and yoeo_results and 'goals' in yoeo_results:
            results['goals'] = yoeo_results['goals']
        elif traditional_results and 'goals' in traditional_results:
            results['goals'] = traditional_results['goals']
        elif yoeo_results and 'goals' in yoeo_results:  # Fallback para YOEO
            results['goals'] = yoeo_results['goals']
        
        # Robôs
        if self.detector_robots == 'yoeo' and yoeo_results and 'robots' in yoeo_results:
            results['robots'] = yoeo_results['robots']
        elif traditional_results and 'robots' in traditional_results:
            results['robots'] = traditional_results['robots']
        elif yoeo_results and 'robots' in yoeo_results:  # Fallback para YOEO
            results['robots'] = yoeo_results['robots']
        
        return results
    
    def _publish_results(self, results):
        """
        Publica os resultados nos tópicos ROS.
        
        Args:
            results: Resultados combinados dos detectores
        """
        if 'ball' in results:
            # Publicar posição da bola
            self.ball_position_pub.publish(results['ball'])
        
        if 'field_mask' in results:
            # Publicar máscara do campo
            field_msg = self.cv_bridge.cv2_to_imgmsg(results['field_mask'])
            self.field_mask_pub.publish(field_msg)
        
        if 'lines' in results:
            # Publicar imagem de linhas
            lines_msg = self.cv_bridge.cv2_to_imgmsg(results['lines'])
            self.lines_image_pub.publish(lines_msg)
        
        if 'goals' in results:
            # Publicar posições dos gols
            self.goal_posts_pub.publish(results['goals'])
        
        if 'robots' in results:
            # Publicar posições dos robôs
            self.obstacles_pub.publish(results['robots'])
        
        # Publicações YOEO específicas
        if 'yoeo_detections' in results:
            self.yoeo_detections_pub.publish(results['yoeo_detections'])
    
    def _process_with_yoeo(self, image, debug_image):
        """
        Processa a imagem usando o detector YOEO.
        
        Args:
            image: Imagem OpenCV no formato BGR
            debug_image: Imagem para desenhar informações de debug
            
        Returns:
            dict: Resultados do processamento
        """
        # Verificar se o manipulador YOEO está disponível
        if self.yoeo_handler is None:
            return {'success': False}
            
        try:
            # Converter BGR para RGB (o modelo espera RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Processar a imagem com o manipulador YOEO
            detections, inference_time = self.yoeo_handler.process(rgb_image)
            
            # Inicializar resultados
            results = {'success': True, 'yoeo_detections': []}
            
            # Processar detecções
            if detections:
                # Separar detecções por classe
                balls = []
                goals = []
                robots = []
                
                for det in detections:
                    # Obter informações da detecção
                    bbox = det['bbox']  # [x1, y1, x2, y2]
                    class_id = det['class']
                    confidence = det['confidence']
                    
                    # Adicionar à lista apropriada
                    if class_id.name == 'BALL':
                        balls.append(det)
                        # Desenhar bola na imagem de debug
                        cv2.rectangle(debug_image, 
                                    (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])), 
                                    (0, 255, 255), 2)
                        cv2.putText(debug_image, f"Bola: {confidence:.2f}", 
                                   (int(bbox[0]), int(bbox[1])-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    elif class_id.name == 'GOAL':
                        goals.append(det)
                        # Desenhar gol na imagem de debug
                        cv2.rectangle(debug_image, 
                                    (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])), 
                                    (255, 0, 0), 2)
                        cv2.putText(debug_image, f"Gol: {confidence:.2f}", 
                                   (int(bbox[0]), int(bbox[1])-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    elif class_id.name == 'ROBOT':
                        robots.append(det)
                        # Desenhar robô na imagem de debug
                        cv2.rectangle(debug_image, 
                                    (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])), 
                                    (255, 0, 255), 2)
                        cv2.putText(debug_image, f"Robô: {confidence:.2f}", 
                                   (int(bbox[0]), int(bbox[1])-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # Adicionar resultados específicos
                if balls:
                    results['ball'] = self._create_ball_pose_from_detection(balls[0])
                
                if goals:
                    results['goals'] = self._create_goal_pose_array_from_detections(goals)
                
                if robots:
                    results['robots'] = self._create_robot_pose_array_from_detections(robots)
                
                # Adicionar FPS na imagem de debug
                fps_text = f"YOEO: {1.0/inference_time:.1f} FPS"
                cv2.putText(debug_image, fps_text, 
                           (debug_image.shape[1] - 200, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            return results
            
        except Exception as e:
            self.get_logger().error(f"Erro no processamento YOEO: {str(e)}")
            return {'success': False}
    
    def _create_ball_pose_from_detection(self, detection):
        """Cria uma mensagem Pose2D a partir de uma detecção de bola."""
        bbox = detection['bbox']  # [x1, y1, x2, y2]
        
        # Calcular centro da bounding box
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Calcular "raio" (metade da largura)
        radius = (bbox[2] - bbox[0]) / 2
        
        # Criar mensagem Pose2D
        ball_pose = Pose2D()
        ball_pose.x = float(center_x)
        ball_pose.y = float(center_y)
        ball_pose.theta = float(radius)  # Usar theta para o raio
        
        return ball_pose
    
    def _create_goal_pose_array_from_detections(self, detections):
        """Cria um PoseArray a partir de detecções de gols."""
        pose_array = PoseArray()
        
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            
            # Calcular centro da bounding box
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Criar pose (usando x para distância e y para distância lateral)
            pose = Pose()
            pose.position.x = float(center_x)
            pose.position.y = float(center_y)
            pose.position.z = 0.0
            
            pose_array.poses.append(pose)
        
        return pose_array
    
    def _create_robot_pose_array_from_detections(self, detections):
        """Cria um PoseArray a partir de detecções de robôs."""
        pose_array = PoseArray()
        
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            
            # Calcular centro da bounding box
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Criar pose (usando x para distância e y para distância lateral)
            pose = Pose()
            pose.position.x = float(center_x)
            pose.position.y = float(center_y)
            pose.position.z = 0.0
            
            pose_array.poses.append(pose)
        
        return pose_array
    
    def _process_with_traditional(self, image, debug_image):
        """
        Processa a imagem usando os detectores tradicionais.
        
        Args:
            image: Imagem OpenCV no formato BGR
            debug_image: Imagem para desenhar informações de debug
            
        Returns:
            dict: Resultados do processamento
        """
        # Resultados iniciais
        results = {'success': True}
        
        # Detectar campo
        if self.enable_field_detection and self.field_detector:
            try:
                # Detectar o campo
                field_mask, field_boundary, field_debug = self.field_detector.detect_field(image)
                
                # Adicionar resultados
                results['field_mask'] = field_mask
                results['field_boundary'] = field_boundary
                
                # Encontrar e desenhar contornos do campo diretamente na imagem
                contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Desenhar todos os contornos com cor verde
                    cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
                    
                    # Calcular área do campo
                    field_area = cv2.contourArea(max(contours, key=cv2.contourArea))
                    image_area = image.shape[0] * image.shape[1]
                    field_ratio = field_area / image_area
                
                # Cobrir parcialmente a imagem original com a máscara para visualização
                overlay = debug_image.copy()
                # Criar uma imagem colorida da máscara (verde)
                color_mask = np.zeros_like(debug_image)
                color_mask[:,:,1] = field_mask  # Canal verde
                # Sobrepor com transparência
                cv2.addWeighted(color_mask, 0.3, overlay, 0.7, 0, debug_image)
                
            except Exception as e:
                self.get_logger().error(f'Erro no detector de campo: {str(e)}')
                
        # Detectar bola
        if self.enable_ball_detection and self.ball_detector:
            try:
                ball_detected, ball_position, ball_radius, ball_debug = self.ball_detector.detect_ball(image)
                if ball_detected:
                    # Criar mensagem Pose2D com a posição da bola
                    ball_pose = Pose2D()
                    ball_pose.x = float(ball_position[0])
                    ball_pose.y = float(ball_position[1])
                    ball_pose.theta = float(ball_radius)  # Usar theta para o raio
                    results['ball'] = ball_pose
                    
                    # Desenhar a bola na imagem de debug
                    cv2.circle(debug_image, ball_position, ball_radius, (0, 255, 255), 2)
                    cv2.circle(debug_image, ball_position, 2, (0, 0, 255), -1)  # Centro
                
                # Sobrepor o debug da bola na imagem de debug geral se necessário
                # if ball_debug is not None:
                #    mask = cv2.cvtColor(ball_debug, cv2.COLOR_BGR2GRAY) > 0
                #    debug_image[mask] = ball_debug[mask]
            except Exception as e:
                self.get_logger().error(f'Erro no detector de bola: {str(e)}')
                
        # Detectar linhas
        if self.enable_line_detection and self.line_detector:
            try:
                # Agora passamos a máscara do campo para melhorar a detecção
                field_mask_param = field_mask if 'field_mask' in results else None
                
                # Detectar linhas agora com a máscara do campo
                lines_image, lines_debug = self.line_detector.detect_lines(image, field_mask_param)
                
                # Adicionar flag para indicar que linhas foram encontradas
                lines_detected = lines_image is not None and np.sum(lines_image) > 0
                
                if lines_detected:
                    results['lines'] = lines_image
                    
                    # Encontrar e desenhar contornos das linhas diretamente
                    lines_contours, _ = cv2.findContours(lines_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(debug_image, lines_contours, -1, (0, 0, 255), 2)
                
            except Exception as e:
                self.get_logger().error(f'Erro no detector de linhas: {str(e)}')
                
        # Detectar gols
        if self.enable_goal_detection and self.goal_detector:
            try:
                # Não vamos passar a máscara do campo para evitar o mesmo erro
                goals_detected, goal_posts, goals_debug = self.goal_detector.detect_goals(image)
                
                if goals_detected:
                    # Criar PoseArray com as posições dos postes do gol
                    pose_array = PoseArray()
                    
                    for post in goal_posts:
                        pose = Pose()
                        pose.position.x = post['distance']
                        pose.position.y = post['lateral_distance']
                        pose.position.z = 0.0
                        pose_array.poses.append(pose)
                        
                        # Desenhar postes de gol
                        if 'position' in post:
                            cv2.circle(debug_image, (int(post['position'][0]), int(post['position'][1])), 
                                      10, (255, 0, 0), -1)  # Postes em azul
                    
                    results['goals'] = pose_array
                
            except Exception as e:
                self.get_logger().error(f'Erro no detector de gols: {str(e)}')
                
        # Detectar obstáculos
        if self.enable_obstacle_detection and self.obstacle_detector:
            try:
                # Não vamos passar a máscara do campo para evitar o mesmo erro
                obstacles_detected, obstacles, obstacles_debug = self.obstacle_detector.detect_obstacles(image)
                
                if obstacles_detected:
                    # Criar PoseArray com as posições dos obstáculos
                    pose_array = PoseArray()
                    
                    for obstacle in obstacles:
                        pose = Pose()
                        pose.position.x = obstacle['distance']
                        pose.position.y = obstacle['lateral_distance']
                        pose.position.z = 0.0
                        pose_array.poses.append(pose)
                        
                        # Desenhar obstáculos
                        if 'position' in obstacle:
                            cv2.rectangle(debug_image, 
                                         (int(obstacle['position'][0])-15, int(obstacle['position'][1])-15),
                                         (int(obstacle['position'][0])+15, int(obstacle['position'][1])+15),
                                         (255, 0, 255), 2)  # Obstáculos em magenta
                    
                    results['robots'] = pose_array
                
            except Exception as e:
                self.get_logger().error(f'Erro no detector de obstáculos: {str(e)}')
        
        return results

def main(args=None):
    rclpy.init(args=args)
    node = VisionPipeline()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
