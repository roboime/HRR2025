#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import ColorRGBA

from .yoeo_model import YOEOModel

class YOEODetector(Node):
    """
    Nó ROS 2 para detecção de objetos usando o modelo YOEO.
    
    Este nó processa imagens da câmera e detecta múltiplos objetos (bola, gol, robôs, árbitro)
    usando o modelo YOEO, otimizado para a Jetson Nano.
    """
    
    def __init__(self):
        super().__init__('yoeo_detector')
        
        # Parâmetros
        self.declare_parameter('model_path', 'resource/models/yoeo_model.h5')
        self.declare_parameter('input_width', 416)
        self.declare_parameter('input_height', 416)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('debug_image', True)
        self.declare_parameter('enable_ball_detection', True)
        self.declare_parameter('enable_goal_detection', True)
        self.declare_parameter('enable_robot_detection', True)
        self.declare_parameter('enable_referee_detection', True)
        
        # Obter parâmetros
        self.model_path = self.get_parameter('model_path').value
        self.input_width = self.get_parameter('input_width').value
        self.input_height = self.get_parameter('input_height').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.debug_image = self.get_parameter('debug_image').value
        self.enable_ball_detection = self.get_parameter('enable_ball_detection').value
        self.enable_goal_detection = self.get_parameter('enable_goal_detection').value
        self.enable_robot_detection = self.get_parameter('enable_robot_detection').value
        self.enable_referee_detection = self.get_parameter('enable_referee_detection').value
        
        # Inicializar o modelo YOEO
        self.model = YOEOModel(
            input_shape=(self.input_height, self.input_width, 3),
            num_classes=4  # bola, gol, robô, árbitro
        )
        
        # Carregar pesos do modelo se disponíveis
        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(pkg_dir, self.model_path)
        if os.path.exists(model_path):
            self.model.load_weights(model_path)
            self.get_logger().info(f'Modelo YOEO carregado de {model_path}')
        else:
            self.get_logger().warn(f'Arquivo de modelo {model_path} não encontrado. Usando modelo não treinado.')
        
        # Inicializar bridge para converter entre OpenCV e ROS
        self.cv_bridge = CvBridge()
        
        # Cores para visualização
        self.colors = {
            'bola': (0, 0, 255),      # Vermelho
            'gol': (255, 255, 0),     # Ciano
            'robo': (0, 255, 0),      # Verde
            'arbitro': (255, 0, 255)  # Magenta
        }
        
        # Publishers
        self.detections_pub = self.create_publisher(Detection2DArray, 'yoeo/detections', 10)
        self.ball_detections_pub = self.create_publisher(Detection2DArray, 'yoeo/ball_detections', 10)
        self.goal_detections_pub = self.create_publisher(Detection2DArray, 'yoeo/goal_detections', 10)
        self.robot_detections_pub = self.create_publisher(Detection2DArray, 'yoeo/robot_detections', 10)
        self.referee_detections_pub = self.create_publisher(Detection2DArray, 'yoeo/referee_detections', 10)
        self.debug_image_pub = self.create_publisher(Image, 'yoeo/debug_image', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            'camera/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Variáveis
        self.camera_matrix = None
        self.dist_coeffs = None
        self.class_names = ['bola', 'gol', 'robo', 'arbitro']
        
        self.get_logger().info('Nó detector YOEO iniciado')
    
    def camera_info_callback(self, msg):
        """Callback para informações da câmera."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
    
    def image_callback(self, msg):
        """Callback para processamento de imagem."""
        try:
            # Converter ROS Image para OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Realizar detecção
            detections = self.detect_objects(cv_image)
            
            # Publicar detecções
            self.publish_detections(detections, msg.header)
            
            # Publicar imagem de debug se necessário
            if self.debug_image:
                debug_image = self.draw_detections(cv_image, detections)
                debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, 'bgr8')
                debug_msg.header = msg.header
                self.debug_image_pub.publish(debug_msg)
                
        except Exception as e:
            self.get_logger().error(f'Erro no processamento da imagem: {str(e)}')
    
    def detect_objects(self, image):
        """
        Detecta objetos na imagem usando o modelo YOEO.
        
        Args:
            image: Imagem OpenCV no formato BGR
            
        Returns:
            Lista de detecções no formato [x, y, w, h, confiança, classe]
        """
        # Realizar predição com o modelo YOEO
        detections = self.model.predict(image)
        
        # Filtrar detecções por confiança e classe
        filtered_detections = []
        
        for detection in detections:
            x, y, w, h, confidence, class_id = detection
            class_name = self.class_names[int(class_id)]
            
            # Verificar se a detecção está acima do limiar de confiança
            if confidence < self.confidence_threshold:
                continue
                
            # Verificar se a classe está habilitada
            if (class_name == 'bola' and not self.enable_ball_detection or
                class_name == 'gol' and not self.enable_goal_detection or
                class_name == 'robo' and not self.enable_robot_detection or
                class_name == 'arbitro' and not self.enable_referee_detection):
                continue
                
            filtered_detections.append(detection)
        
        # Aplicar non-maximum suppression para remover detecções duplicadas
        # (Implementação simplificada - em um sistema real, seria mais complexo)
        
        return filtered_detections
    
    def publish_detections(self, detections, header):
        """
        Publica as detecções nos tópicos ROS.
        
        Args:
            detections: Lista de detecções no formato [x, y, w, h, confiança, classe]
            header: Cabeçalho da mensagem original
        """
        # Criar mensagem de detecções
        detection_array = Detection2DArray()
        detection_array.header = header
        
        # Mensagens específicas por classe
        ball_detections = Detection2DArray()
        ball_detections.header = header
        
        goal_detections = Detection2DArray()
        goal_detections.header = header
        
        robot_detections = Detection2DArray()
        robot_detections.header = header
        
        referee_detections = Detection2DArray()
        referee_detections.header = header
        
        # Processar cada detecção
        for detection in detections:
            x, y, w, h, confidence, class_id = detection
            class_name = self.class_names[int(class_id)]
            
            # Criar mensagem de detecção
            det_msg = Detection2D()
            det_msg.header = header
            
            # Definir posição e tamanho da caixa delimitadora
            det_msg.bbox.center.x = x
            det_msg.bbox.center.y = y
            det_msg.bbox.size_x = w
            det_msg.bbox.size_y = h
            
            # Adicionar hipótese de objeto
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(class_id)
            hypothesis.score = float(confidence)
            det_msg.results.append(hypothesis)
            
            # Adicionar à lista geral de detecções
            detection_array.detections.append(det_msg)
            
            # Adicionar à lista específica da classe
            if class_name == 'bola':
                ball_detections.detections.append(det_msg)
            elif class_name == 'gol':
                goal_detections.detections.append(det_msg)
            elif class_name == 'robo':
                robot_detections.detections.append(det_msg)
            elif class_name == 'arbitro':
                referee_detections.detections.append(det_msg)
        
        # Publicar detecções
        self.detections_pub.publish(detection_array)
        self.ball_detections_pub.publish(ball_detections)
        self.goal_detections_pub.publish(goal_detections)
        self.robot_detections_pub.publish(robot_detections)
        self.referee_detections_pub.publish(referee_detections)
    
    def draw_detections(self, image, detections):
        """
        Desenha as detecções na imagem para visualização.
        
        Args:
            image: Imagem OpenCV no formato BGR
            detections: Lista de detecções no formato [x, y, w, h, confiança, classe]
            
        Returns:
            Imagem com as detecções desenhadas
        """
        # Criar cópia da imagem
        debug_image = image.copy()
        
        # Desenhar cada detecção
        for detection in detections:
            x, y, w, h, confidence, class_id = detection
            class_name = self.class_names[int(class_id)]
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Converter para coordenadas de canto
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            # Desenhar retângulo
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
            
            # Desenhar texto
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(debug_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return debug_image

def main(args=None):
    rclpy.init(args=args)
    node = YOEODetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 