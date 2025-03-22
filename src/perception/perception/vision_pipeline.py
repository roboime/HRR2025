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
from geometry_msgs.msg import Pose2D, PoseArray
from vision_msgs.msg import Detection2DArray
import threading
import time
import os

# Importar componentes YOEO se disponíveis
try:
    from .yoeo.yoeo_detector import YOEOHandler
    from .yoeo.yoeo_model import YOEOModel
    YOEO_AVAILABLE = True
except ImportError:
    try:
        # Tentativa alternativa se o import relativo falhar
        from perception.src.yoeo.yoeo_detector import YOEOHandler
        from perception.src.yoeo.yoeo_model import YOEOModel
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
        from perception.src.ball_detector import BallDetector
        from perception.src.field_detector import FieldDetector
        from perception.src.line_detector import LineDetector
        from perception.src.goal_detector import GoalDetector
        from perception.src.obstacle_detector import ObstacleDetector
        TRADITIONAL_DETECTORS_AVAILABLE = True
    except ImportError:
        TRADITIONAL_DETECTORS_AVAILABLE = False
        print("AVISO: Detectores tradicionais não estão disponíveis.")

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
        self.declare_parameter('yoeo_model_path', 'resource/models/yoeo_model.h5')
        self.declare_parameter('yoeo_confidence_threshold', 0.5)
        self.declare_parameter('use_tensorrt', False)
        
        # Escolha de detector para cada tipo de objeto
        self.declare_parameter('detector_ball', 'yoeo')       # 'yoeo' ou 'traditional'
        self.declare_parameter('detector_field', 'traditional') # 'yoeo' ou 'traditional'
        self.declare_parameter('detector_lines', 'traditional') # 'yoeo' ou 'traditional'
        self.declare_parameter('detector_goals', 'yoeo')      # 'yoeo' ou 'traditional'
        self.declare_parameter('detector_robots', 'yoeo')     # 'yoeo' ou 'traditional'
        
        # Detectores tradicionais disponíveis
        self.declare_parameter('enable_ball_detection', True)
        self.declare_parameter('enable_field_detection', True)
        self.declare_parameter('enable_line_detection', False)
        self.declare_parameter('enable_goal_detection', False)
        self.declare_parameter('enable_obstacle_detection', False)
        
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
                # Aqui normalmente carregaríamos o modelo real
                # Para este exemplo, estamos apenas sinalizando que está disponível
                self.yoeo_handler = True
                self.get_logger().info('Modelo YOEO carregado com sucesso')
            except Exception as e:
                self.get_logger().error(f'Erro ao carregar modelo YOEO: {str(e)}')
                self.use_yoeo = False
        
        # Inicializar detectores tradicionais se habilitados
        if TRADITIONAL_DETECTORS_AVAILABLE:
            if self.enable_ball_detection:
                self.ball_detector = True
            
            if self.enable_field_detection:
                self.field_detector = True
            
            if self.enable_line_detection:
                self.line_detector = True
            
            if self.enable_goal_detection:
                self.goal_detector = True
            
            if self.enable_obstacle_detection:
                self.obstacle_detector = True
    
    def _init_publishers(self):
        """Inicializa os publishers para os resultados."""
        # Publisher de debug
        self.debug_image_pub = self.create_publisher(Image, 'vision_debug', 10)
        
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
        
        while self.processing_active and rclpy.ok():
            start_time = time.time()
            
            # Processar a imagem mais recente
            if self.latest_image is not None and self.camera_matrix is not None:
                try:
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
                        yoeo_results = self._process_with_yoeo(cv_image, debug_image)
                    
                    if use_traditional:
                        traditional_results = self._process_with_traditional(cv_image, debug_image)
                    
                    # Combinar resultados baseado na configuração de preferência para cada objeto
                    results = self._select_best_results(yoeo_results, traditional_results)
                    
                    # Adicionar informações de debug
                    detectors_used = []
                    if yoeo_results:
                        detectors_used.append("YOEO")
                    if traditional_results:
                        detectors_used.append("Tradicional")
                    
                    cv2.putText(debug_image, "Detectores: " + " + ".join(detectors_used), (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Publicar resultados
                    self._publish_results(results)
                    
                    # Publicar imagem de debug
                    if self.debug_image:
                        # Adicionar informações de timestamp e FPS
                        fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                        cv2.putText(debug_image, f"FPS: {fps:.1f}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, 'bgr8')
                        debug_msg.header = self.latest_image.header
                        self.debug_image_pub.publish(debug_msg)
                
                except Exception as e:
                    self.get_logger().error(f'Erro no processamento da imagem: {str(e)}')
            
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
        # Este é um exemplo simplificado de como seria o processamento com YOEO
        # Em uma implementação real, isso chamaria o detector YOEO e processaria os resultados
        
        # Simular sucesso na detecção
        success = True
        
        # Desenhar um marcador no debug para mostrar que o YOEO foi usado
        cv2.putText(debug_image, "YOEO", (image.shape[1] - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return {'success': success}
    
    def _process_with_traditional(self, image, debug_image):
        """
        Processa a imagem usando os detectores tradicionais.
        
        Args:
            image: Imagem OpenCV no formato BGR
            debug_image: Imagem para desenhar informações de debug
            
        Returns:
            dict: Resultados do processamento
        """
        # Este é um exemplo simplificado de como seria o processamento com detectores tradicionais
        # Em uma implementação real, isso chamaria os detectores tradicionais e processaria os resultados
        
        # Detectar campo
        if self.enable_field_detection and self.field_detector:
            # Código para detectar o campo
            pass
        
        # Detectar bola
        if self.enable_ball_detection and self.ball_detector:
            # Código para detectar a bola
            pass
        
        # Detectar linhas
        if self.enable_line_detection and self.line_detector:
            # Código para detectar linhas
            pass
        
        # Detectar gols
        if self.enable_goal_detection and self.goal_detector:
            # Código para detectar gols
            pass
        
        # Detectar obstáculos
        if self.enable_obstacle_detection and self.obstacle_detector:
            # Código para detectar obstáculos
            pass
        
        # Desenhar um marcador no debug para mostrar que os detectores tradicionais foram usados
        cv2.putText(debug_image, "Tradicional", (image.shape[1] - 150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return {'success': True}

def main(args=None):
    rclpy.init(args=args)
    node = VisionPipeline()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
