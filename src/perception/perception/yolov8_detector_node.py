#!/usr/bin/env python3
"""
Detector YOLOv8 Unificado para RoboIME HSL2025 - Jetson Orin Nano Super
Sistema completo de percepção usando YOLOv8 + Geometria 3D avançada
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point, Pose2D
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
# Desativar checagem de updates do Ultralytics ANTES do import
os.environ.setdefault('ULTRALYTICS_NO_UPDATE_CHECK', 'True')
from ultralytics import YOLO
from ultralytics import settings as yolo_settings
import sys
import traceback
import torch
import time
import os
import yaml
from typing import List, Tuple, Optional
from ament_index_python.packages import get_package_share_directory

# Desabilitar sync/telemetria e evitar auto-update de requirements do Ultralytics
try:
    yolo_settings.update({"sync": False})
    from ultralytics.utils import checks as ychecks
    def _no_requirements(*args, **kwargs):
        return True
    ychecks.check_requirements = _no_requirements
    # Propagar para módulos que tenham copiado a referência
    for _m in list(sys.modules.values()):
        try:
            if hasattr(_m, 'check_requirements'):
                setattr(_m, 'check_requirements', _no_requirements)
        except Exception:
            pass
except Exception:
    pass

# Importar módulo de geometria 3D
from .camera_geometry_3d import CameraGeometry3D, Object3D

# Importar mensagens customizadas do roboime_msgs
from roboime_msgs.msg import (
    BallDetection, RobotDetection, GoalDetection, 
    FieldDetection, SimplifiedDetections, FieldLandmark,
    RobotPose2D,
)
from vision_msgs.msg import BoundingBox2D

class YOLOv8UnifiedDetector(Node):
    """
    Detector YOLOv8 com Sistema de Geometria 3D Avançado
    
    Funcionalidades:
    - Detecção YOLOv8 simplificada (7 classes)
    - Cálculo de posições 3D reais usando calibração da câmera
    - Validação de detecções baseada em tamanhos esperados
    - Mapeamento para coordenadas do campo RoboCup
    - Correção de perspectiva e distorção
    
    Classes Detectadas:
    - ball: Bola de futebol
    - robot: Robôs (sem distinção de cor)
    - penalty_mark: Marca do penalty
    - goal_post: Postes de gol (unificados)
    - center_circle: Círculo central
    - field_corner: Cantos do campo  
    - area_corner: Cantos da área
    """
    
    def __init__(self):
        super().__init__('yolov8_unified_detector')
        
        # Bridge OpenCV-ROS2
        self.bridge = CvBridge()
        # Parâmetros padrão e declaração
        try:
            share_dir = get_package_share_directory('perception')
        except Exception:
            share_dir = os.path.join(os.path.dirname(__file__), '..')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', os.path.join(share_dir, 'resources', 'models', 'robocup_yolov8.engine')),
                ('confidence_threshold', 0.6),
                ('iou_threshold', 0.45),
                ('publish_debug', True),
                ('max_detections', 300),
                ('landmark_config_path', os.path.join(share_dir, 'resources', 'calibration', 'camera_info.yaml')),
                ('imgsz', 640),                # tamanho da imagem para YOLO/TensorRT
            ]
        )
        # Carregar parâmetros em variáveis da instância
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.max_detections = int(self.get_parameter('max_detections').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        
        # Sistema de geometria 3D
        self._init_3d_geometry()
        
        # Mapeamento de classes carregado do YAML (ordem alfabética) com fallback
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'perception_config.yaml')
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            classes_cfg = cfg.get('yolov8', {}).get('classes', {})
            # Inverter mapping nome->id para id->nome
            self.class_names = {int(idx): name for name, idx in classes_cfg.items()}
        except Exception:
            # Fallback para ordem alfabética fixa
            self.class_names = {
                0: 'area_corner',
                1: 'ball',
                2: 'center_circle',
                3: 'field_corner',
                4: 'goal_post',
                5: 'penalty_mark',
                6: 'robot',
            }
        
        # Carregar configurações (inclui device/half_precision)
        # Importante: carregar ANTES de inicializar o modelo para respeitar device
        self._load_perception_config()
        # Inicializar modelo YOLOv8
        self._init_model()
        
        # Publishers para diferentes tipos de detecção
        self.ball_pub = self.create_publisher(BallDetection, 'ball_detection', 10)
        self.robots_pub = self.create_publisher(RobotDetection, 'robot_detections', 10)
        self.goals_pub = self.create_publisher(GoalDetection, 'goal_detections', 10)
        self.landmarks_pub = self.create_publisher(FieldDetection, 'localization_landmarks', 10)
        self.unified_pub = self.create_publisher(SimplifiedDetections, 'unified_detections', 10)
        
        # Publishers de debug: absoluto (compat) e relativo (mais robusto para remap)
        reliable_qos = QoSProfile(depth=10)
        reliable_qos.reliability = ReliabilityPolicy.RELIABLE
        reliable_qos.history = HistoryPolicy.KEEP_LAST
        self.debug_image_pub = self.create_publisher(Image, '/yolov8_detector/debug_image_3d', reliable_qos)
        self.debug_image_pub_rel = self.create_publisher(Image, 'debug_image_3d', reliable_qos)
        
        # Subscriber para imagens
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        
        # Subscriber para pose do robô (necessário para cálculo de posição absoluta)
        self.robot_pose_sub = self.create_subscription(
            RobotPose2D, 'robot_pose', self.robot_pose_callback, 10
        )
        
        # Estado atual do robô
        self.current_robot_pose = None
        self.pose_timestamp = None
        
        # Carregar landmarks conhecidos para correspondência
        self._load_landmark_coordinates()
        
        # Estatísticas e debug
        self.detection_count = 0
        self.processing_times = []
        self.publish_debug = True
        # Contadores de performance
        self.frame_count = 0
        self.total_detections = 0
        self.last_stats_time = time.time()
        
        self.get_logger().info('🎯 YOLOv8 Detector com Geometria 3D inicializado!')
        self.get_logger().info(f'📏 Sistema 3D: Altura câmera = {self.geometry_3d.geometry.height:.2f}m')
        self.get_logger().info(f'📐 Inclinação = {np.degrees(self.geometry_3d.geometry.tilt_angle):.1f}°')
    
    def _init_3d_geometry(self):
        """Inicializa sistema de geometria 3D com calibração da câmera"""
        try:
            # Caminho para arquivo de calibração
            config_path = self.get_parameter('landmark_config_path').value
            
            # Alternativa se o pacote não for encontrado
            if not os.path.exists(config_path):
                config_path = os.path.join(
                    os.path.dirname(__file__), '..', 'resources', 
                    'calibration', 'camera_info.yaml'
                )
            
            # Inicializar sistema 3D
            self.geometry_3d = CameraGeometry3D(config_path)
            
            # Log informações da câmera
            info = self.geometry_3d.get_camera_info_summary()
            self.get_logger().info(f"📐 FOV: {info['field_of_view_x_degrees']:.1f}° x {info['field_of_view_y_degrees']:.1f}°")
            self.get_logger().info(f"🔭 Distância máxima visível: {info['max_distance_visible']:.1f}m")
            
        except Exception as e:
            self.get_logger().error(f'❌ Erro ao inicializar geometria 3D: {e}')
            self.geometry_3d = None
    
    def _load_perception_config(self):
        """Carrega configurações de percepção do arquivo YAML"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'perception_config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Parâmetros YOLOv8
            yolo_config = config['yolov8']
            self.confidence_threshold = yolo_config['confidence_threshold']
            self.iou_threshold = yolo_config['iou_threshold']
            self.device = yolo_config.get('device', 'cuda')
            self.half_precision = bool(yolo_config.get('half_precision', True))
            
            # Parâmetros para pareamento de postes de gol (compatível com geometry_3d/goal_post_pairing)
            geom_cfg = config.get('geometry_3d', config.get('detection_3d', {}))
            goal_config = geom_cfg.get('goal_post_pairing', {})
            self.goal_pairing_enabled = bool(goal_config.get('enable_pairing', True))
            self.goal_width_tolerance = float(goal_config.get('goal_width_tolerance', 0.5))
            self.goal_center_y_tolerance = float(goal_config.get('goal_center_y_tolerance', 0.5))
            self.plate_x_tolerance = float(goal_config.get('plate_x_tolerance', 0.5))
            self.post_y_tolerance = float(goal_config.get('post_y_tolerance', 0.5))
            self.min_pair_confidence = float(goal_config.get('min_pair_confidence', 0.4))
            self.single_post_confidence_penalty = float(goal_config.get('single_post_confidence_penalty', 0.6))
            
            self.get_logger().info(f'✅ Configurações carregadas: confidence={self.confidence_threshold}, pareamento_postes={self.goal_pairing_enabled}')
            
        except Exception as e:
            self.get_logger().error(f'❌ Erro ao carregar configurações: {e}')
            # Valores padrão
            self.confidence_threshold = 0.6
            self.iou_threshold = 0.45
            self.device = 'cuda'
            self.goal_pairing_enabled = True
            self.goal_width_tolerance = 0.5
            self.goal_center_y_tolerance = 0.5
            self.plate_x_tolerance = 0.5
            self.post_y_tolerance = 0.5
            self.min_pair_confidence = 0.4
            self.single_post_confidence_penalty = 0.6

    def _init_model(self):
        """Inicializa o modelo YOLOv8"""
        try:
            model_path = self.get_parameter('model_path').value
            if not isinstance(model_path, str):
                model_path = ''
            
            # Preferir TensorRT (.engine) e fazer fallback para .pt
            if model_path and os.path.exists(model_path):
                self.get_logger().info(f'Carregando modelo (preferência TensorRT se .engine): {model_path}')
                self.model = YOLO(model_path)
                ext = os.path.splitext(model_path)[1].lower()
                self._is_pytorch_model = (ext == '.pt')
            else:
                # Tentar automaticamente o .pt se .engine não existir
                pt_path = model_path.replace('.engine', '.pt') if isinstance(model_path, str) else ''
                if pt_path and os.path.exists(pt_path):
                    self.get_logger().warn(f'.engine não encontrado. Usando .pt: {pt_path}')
                    self.model = YOLO(pt_path)
                    self._is_pytorch_model = True
                else:
                    self.get_logger().warn(f'Modelo não encontrado: {model_path}')
                    self.get_logger().warn('Usando YOLOv8n padrão - REQUER RETREINAMENTO!')
                    self.model = YOLO('yolov8n.pt')
                    self._is_pytorch_model = True
            
            # Configurar device (evitar half para prevenir dtype mismatch)
            # Configurar dispositivo apenas para modelos PyTorch; exportados (TensorRT/ONNX) não suportam .to('cuda')
            if self._is_pytorch_model:
                if torch.cuda.is_available() and self.device == 'cuda':
                    self.model.to('cuda')
                    try:
                        if hasattr(self.model, 'fuse'):
                            self.model.fuse()
                    except Exception:
                        pass
                    try:
                        torch.set_float32_matmul_precision('high')
                    except Exception:
                        pass
                    self.get_logger().info('Modelo YOLOv8 (PyTorch) na GPU (CUDA) em float32')
                self._inference_device_arg = None  # não necessário passar em cada chamada
            else:
                # Modelo exportado (ex.: TensorRT .engine) -> definir device na chamada de inferência
                if self.device == 'cuda' and torch.cuda.is_available():
                    self._inference_device_arg = 0
                else:
                    self._inference_device_arg = 'cpu'
                
        except Exception as e:
            self.get_logger().error(f'Erro ao carregar modelo YOLOv8: {e}')
            raise
    
    def image_callback(self, msg: Image):
        """Processa imagem e executa detecção com cálculos 3D"""
        try:
            start_time = time.time()
            
            # Converter ROS Image para OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Executar detecção YOLOv8
            detections = self._detect_objects(cv_image)
            
            # Calcular informações 3D se o sistema estiver disponível
            objects_3d = []
            if self.geometry_3d is not None and detections:
                objects_3d = self.geometry_3d.detect_and_compute_3d(
                    detections, confidence_threshold=self.confidence_threshold
                )
                
                # Log estatísticas 3D
                if objects_3d:
                    distances = [obj.distance for obj in objects_3d if obj.distance is not None]
                    if distances:
                        self.get_logger().debug(
                            f"📏 Detecções 3D: {len(objects_3d)} objetos, "
                            f"distâncias: {min(distances):.1f}m - {max(distances):.1f}m"
                        )
            
            # Organizar detecções por categoria (incluindo dados 3D)
            organized_detections = self._organize_detections_3d(detections, objects_3d)
            
            # Publicar mensagens específicas
            self._publish_detections(organized_detections, msg.header)
            
            # Publicar imagem de debug com informações 3D
            if self.publish_debug:
                self._publish_debug_image_3d(cv_image, detections, objects_3d, msg.header)
            
            # Atualizar estatísticas
            self._update_stats(detections, time.time() - start_time)
                
        except Exception as e:
            self.get_logger().error(f'❌ Erro no processamento da imagem: {e}\n{traceback.format_exc()}')
    
    def _detect_objects(self, image: np.ndarray) -> List[dict]:
        """Executa detecção YOLOv8 para 7 classes"""
        detections = []
        
        try:
            # Executar inferência com configurações otimizadas
            # Converter imagem para FP16/GPU se disponível para maximizar throughput
            imgsz = self.imgsz
            run_kwargs = dict(conf=self.confidence_threshold,
                              iou=self.iou_threshold,
                              max_det=self.max_detections,
                              imgsz=imgsz,
                              verbose=False)
            if torch.cuda.is_available() and self.device == 'cuda' and getattr(self, 'half_precision', True):
                # Desabilitar half explicitamente para evitar dtype mismatch
                run_kwargs.pop('half', None)
            # Para modelos exportados (TensorRT/ONNX), o Ultralytics requer 'device' no predict
            if hasattr(self, '_inference_device_arg') and self._inference_device_arg is not None:
                results = self.model.predict(image, device=self._inference_device_arg, **run_kwargs)
            else:
                results = self.model(image, **run_kwargs)
            
            # Processar resultados
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        class_name = self.class_names.get(cls, f'unknown_{cls}')
                        
                        detection = {
                            'bbox': box,  # [x1, y1, x2, y2]
                            'confidence': float(conf),
                            'class_id': int(cls),
                            'class_name': class_name,
                            'center_x': float((box[0] + box[2]) / 2),
                            'center_y': float((box[1] + box[3]) / 2),
                            'width': float(box[2] - box[0]),
                            'height': float(box[3] - box[1])
                        }
                        detections.append(detection)
            
        except Exception as e:
            self.get_logger().error(f'Erro na detecção YOLOv8: {e}')
        
        return detections
    
    def _organize_detections_3d(self, detections: List[dict], objects_3d: List[Object3D]) -> dict:
        """
        Organiza detecções por categoria incluindo informações 3D
        
        Args:
            detections: Detecções 2D do YOLOv8
            objects_3d: Objetos com informações 3D calculadas
            
        Returns:
            Dicionário organizado por tipo de objeto
        """
        
        # Criar mapeamento de detecções 2D para 3D
        detection_3d_map = {}
        for obj_3d in objects_3d:
            # Encontrar detecção 2D correspondente baseada na posição
            for i, det in enumerate(detections):
                if (abs(det['center_x'] - obj_3d.pixel_x) < 5 and 
                    abs(det['center_y'] - obj_3d.pixel_y) < 5):
                    detection_3d_map[i] = obj_3d
                    break
        
        organized = {
            'balls': [],
            'robots': [], 
            'goals': [],
            'landmarks': []
        }
        
        for i, detection in enumerate(detections):
            class_name = detection['class_name']
            
            # Obter informações 3D se disponíveis
            obj_3d = detection_3d_map.get(i)
            
            # Adicionar informações 3D ao detection
            if obj_3d is not None:
                detection.update({
                    'world_x': obj_3d.world_x,
                    'world_y': obj_3d.world_y,
                    'distance': obj_3d.distance,
                    'estimated_real_size': obj_3d.estimated_real_size,
                    'size_confidence': obj_3d.size_confidence,
                    'total_confidence': obj_3d.detection_confidence * obj_3d.size_confidence
                })
            else:
                # Valores padrão quando não há dados 3D
                detection.update({
                    'world_x': None,
                    'world_y': None, 
                    'distance': None,
                    'estimated_real_size': None,
                    'size_confidence': 0.5,
                    'total_confidence': detection['confidence']
                })
            
            # Categorizar por tipo
            if class_name == 'ball':
                organized['balls'].append(detection)
            elif class_name == 'robot':
                organized['robots'].append(detection)
            elif class_name == 'goal_post':
                organized['goals'].append(detection)
            elif class_name in ['penalty_mark', 'center_circle', 'field_corner', 'area_corner']:
                organized['landmarks'].append(detection)
        
        return organized
    
    def _publish_detections(self, organized: dict, header):
        """Publica mensagens específicas para cada categoria"""
        
        # Bola (estratégia)
        if organized['balls']:
            best_ball = max(organized['balls'], key=lambda x: x['confidence'])
            ball_msg = BallDetection()
            ball_msg.header = header
            ball_msg.detected = True
            ball_msg.confidence = float(best_ball.get('confidence', 0.0))
            ball_msg.detection_method = "yolov8"
            # BBox
            bbox = BoundingBox2D()
            bbox.center = Pose2D()
            bbox.center.x = float(best_ball['center_x'])
            bbox.center.y = float(best_ball['center_y'])
            bbox.size_x = float(best_ball['width'])
            bbox.size_y = float(best_ball['height'])
            ball_msg.bbox = bbox
            # Distância/bearing (garantir float)
            ball_msg.distance = float(best_ball.get('distance') or 0.0)
            ball_msg.bearing = float(self._calculate_bearing(best_ball['center_x'], best_ball['center_y']))
            self.ball_pub.publish(ball_msg)
        
        # Robôs (estratégia - sem distinção de cor)
        if organized['robots']:
            # Publicar cada robô como mensagem individual (compatível com definição)
            for rob in organized['robots']:
                robot_msg = RobotDetection()
                robot_msg.header = header
                robot_msg.detected = True
                robot_msg.confidence = float(rob.get('confidence', 0.0))
                robot_msg.detection_method = "yolov8"
                robot_msg.robot_classification = "robot"
                robot_msg.distance = float(rob.get('distance') or 0.0)
                robot_msg.bearing = float(self._calculate_bearing(rob['center_x'], rob['center_y']))
                # BBox
                bbox = BoundingBox2D()
                bbox.center = Pose2D()
                bbox.center.x = float(rob['center_x'])
                bbox.center.y = float(rob['center_y'])
                bbox.size_x = float(rob['width'])
                bbox.size_y = float(rob['height'])
                robot_msg.bbox = bbox
                self.robots_pub.publish(robot_msg)
        
        # Landmarks para localização
        if organized['landmarks']:
            landmarks_msg = FieldDetection()
            landmarks_msg.header = header
            landmarks_msg.field_detected = True
            landmarks_msg.num_landmarks = len(organized['landmarks'])
            
            # Separar por tipo de landmark
            penalty_marks = [landmark for landmark in organized['landmarks'] if landmark['class_name'] == 'penalty_mark']
            goals = [landmark for landmark in organized['landmarks'] if landmark['class_name'] == 'goal_post']
            center_circles = [landmark for landmark in organized['landmarks'] if landmark['class_name'] == 'center_circle']
            field_corners = [landmark for landmark in organized['landmarks'] if landmark['class_name'] == 'field_corner']
            area_corners = [landmark for landmark in organized['landmarks'] if landmark['class_name'] == 'area_corner']
            
            self.landmarks_pub.publish(landmarks_msg)
            
            # Publicar detecções de poste de gol separadamente (compatibilidade + enriquecido)
            if goals:
                goal_msg = GoalDetection()
                goal_msg.header = header
                goal_msg.detected = True
                goal_msg.detection_method = "yolov8"
                # Pareamento e cálculo do centro/orientação do gol quando possível
                enriched = self._compute_goal_from_posts(goals)
                if enriched is not None:
                    c, left_p, right_p, width_est, side, distance, bearing, conf = enriched
                    goal_msg.position_3d.x = float(c[0])
                    goal_msg.position_3d.y = float(c[1])
                    goal_msg.position_3d.z = 0.0
                    goal_msg.left_post.x = float(left_p[0])
                    goal_msg.left_post.y = float(left_p[1])
                    goal_msg.left_post.z = 0.0
                    goal_msg.right_post.x = float(right_p[0])
                    goal_msg.right_post.y = float(right_p[1])
                    goal_msg.right_post.z = 0.0
                    goal_msg.goal_width = float(width_est)
                    goal_msg.goal_height = 0.8  # altura típica visível
                    goal_msg.goal_side = side
                    goal_msg.distance = float(distance)
                    goal_msg.bearing = float(bearing)
                    goal_msg.confidence = float(conf)
                else:
                    # Fallback com um único poste (menor confiança)
                    best = max(goals, key=lambda x: x.get('confidence', 0.0))
                    # Estimar posição absoluta do poste melhor detectado (se disponível)
                    px = best.get('world_x'); py = best.get('world_y')
                    if px is not None and py is not None:
                        goal_msg.position_3d.x = float(px)
                        goal_msg.position_3d.y = float(py)
                        goal_msg.position_3d.z = 0.0
                        goal_msg.distance = float(best.get('distance') or 0.0)
                        goal_msg.bearing = float(self._calculate_bearing(best['center_x'], best['center_y']))
                        goal_msg.confidence = float(best.get('confidence', 0.3) * self.single_post_confidence_penalty)
                        goal_msg.goal_side = "left" if px < 0 else "right"
                
                self.goals_pub.publish(goal_msg)
        
        # Detecções unificadas simplificadas
        unified_msg = SimplifiedDetections()
        unified_msg.header = header
        
        # === DETECÇÕES PARA ESTRATÉGIA ===
        # Bola (já publicada individualmente, reutilizar dados)
        if organized['balls']:
            best_ball = max(organized['balls'], key=lambda x: x['confidence'])
            ball_msg = BallDetection()
            ball_msg.header = header
            ball_msg.detected = True
            ball_msg.confidence = float(best_ball.get('confidence', 0.0))
            ball_msg.detection_method = "yolov8"
            bbox = BoundingBox2D()
            bbox.center = Pose2D()
            bbox.center.x = float(best_ball['center_x'])
            bbox.center.y = float(best_ball['center_y'])
            bbox.size_x = float(best_ball['width'])
            bbox.size_y = float(best_ball['height'])
            ball_msg.bbox = bbox
            ball_msg.distance = float(best_ball.get('distance') or 0.0)
            ball_msg.bearing = float(self._calculate_bearing(best_ball['center_x'], best_ball['center_y']))
            unified_msg.ball = ball_msg
        
        # Robôs (converter para array)
        robot_detections = []
        for rob in organized['robots']:
            robot_msg = RobotDetection()
            robot_msg.header = header
            robot_msg.detected = True
            robot_msg.confidence = float(rob.get('confidence', 0.0))
            robot_msg.detection_method = "yolov8"
            robot_msg.robot_classification = "robot"
            robot_msg.distance = float(rob.get('distance') or 0.0)
            robot_msg.bearing = float(self._calculate_bearing(rob['center_x'], rob['center_y']))
            bbox = BoundingBox2D()
            bbox.center = Pose2D()
            bbox.center.x = float(rob['center_x'])
            bbox.center.y = float(rob['center_y'])
            bbox.size_x = float(rob['width'])
            bbox.size_y = float(rob['height'])
            robot_msg.bbox = bbox
            robot_detections.append(robot_msg)
        
        unified_msg.robots = robot_detections
        unified_msg.num_robots = len(robot_detections)
        
        # === DETECÇÕES PARA LOCALIZAÇÃO ===
        # Separar landmarks por tipo e criar FieldLandmarks
        penalty_marks = []
        goals = []
        center_circles = []
        field_corners = []
        area_corners = []
        goal_detections = []
        
        for landmark in organized['landmarks']:
            field_landmark = FieldLandmark()
            field_landmark.confidence = landmark['confidence']
            field_landmark.yolo_confidence = landmark['confidence']
            field_landmark.source_detection_method = "yolov8"
            field_landmark.observation_quality = min(landmark['confidence'], 1.0)
            field_landmark.distance = float(landmark.get('distance') or 0.0)
            field_landmark.bearing = float(self._calculate_bearing(landmark['center_x'], landmark['center_y']))
            
            # Posição relativa (estimada)
            field_landmark.position_relative.x = field_landmark.distance * np.cos(field_landmark.bearing)
            field_landmark.position_relative.y = field_landmark.distance * np.sin(field_landmark.bearing)
            field_landmark.position_relative.z = 0.0
            
            # Calcular posição absoluta usando a melhor abordagem disponível
            class_name = landmark['class_name']
            field_landmark.position_absolute = self._calculate_absolute_position(field_landmark, class_name)
            
            # Validar posição calculada
            position_valid = self._validate_landmark_position(field_landmark, class_name)
            if not position_valid:
                self.get_logger().debug(
                    f"Invalid position for {class_name}: absolute({field_landmark.position_absolute.x:.2f}, "
                    f"{field_landmark.position_absolute.y:.2f}), relative({field_landmark.position_relative.x:.2f}, "
                    f"{field_landmark.position_relative.y:.2f}), dist={field_landmark.distance:.2f}m"
                )
            
            # Calcular confiança geométrica e aplicar à confiança final
            geometric_confidence = self._calculate_geometric_confidence(landmark, field_landmark.distance)
            field_landmark.confidence = min(field_landmark.confidence * geometric_confidence, 1.0)
            
            # Log detalhado para debugging (apenas para landmarks de alta prioridade)
            if class_name in ['penalty_mark', 'center_circle'] and field_landmark.confidence > 0.5:
                pose_status = "WITH_POSE" if self.current_robot_pose is not None else "NO_POSE"
                self.get_logger().debug(
                    f"Landmark {class_name}: conf={field_landmark.confidence:.3f}, "
                    f"geom_conf={geometric_confidence:.3f}, dist={field_landmark.distance:.2f}m, "
                    f"bearing={field_landmark.bearing*180/np.pi:.1f}°, "
                    f"abs_pos=({field_landmark.position_absolute.x:.2f},{field_landmark.position_absolute.y:.2f}), "
                    f"calc_method={pose_status}"
                )
            
            if class_name == 'penalty_mark':
                field_landmark.type = FieldLandmark.PENALTY_MARK
                field_landmark.yolo_class_id = next((cid for cid, cname in self.class_names.items() if cname == 'penalty_mark'), 5)
                field_landmark.localization_priority = FieldLandmark.PRIORITY_HIGH
                field_landmark.description = "penalty_mark"
                field_landmark.uniqueness_score = 0.9
                field_landmark.disambiguates_pose = True
                penalty_marks.append(field_landmark)
                
            elif class_name == 'goal_post':
                field_landmark.type = FieldLandmark.GOAL_POST
                field_landmark.yolo_class_id = next((cid for cid, cname in self.class_names.items() if cname == 'goal_post'), 4)
                field_landmark.localization_priority = FieldLandmark.PRIORITY_MEDIUM
                field_landmark.description = "goal_post"
                field_landmark.uniqueness_score = 0.8
                goals.append(field_landmark)
                
                # Criar também GoalDetection para compatibilidade
                goal_det = GoalDetection()
                goal_det.header = header
                goal_det.detected = True
                goal_det.detection_method = "yolov8"
                goal_det.confidence = float(landmark['confidence'])
                # Preencher bbox com posição do poste
                goal_det.bearing = float(self._calculate_bearing(landmark['center_x'], landmark['center_y']))
                goal_det.distance = float(landmark.get('distance') or 0.0)
                goal_detections.append(goal_det)
                
            elif class_name == 'center_circle':
                field_landmark.type = FieldLandmark.CENTER_CIRCLE
                field_landmark.yolo_class_id = next((cid for cid, cname in self.class_names.items() if cname == 'center_circle'), 2)
                field_landmark.localization_priority = FieldLandmark.PRIORITY_HIGH
                field_landmark.description = "center_circle"
                field_landmark.uniqueness_score = 1.0
                field_landmark.disambiguates_pose = True
                center_circles.append(field_landmark)
                
            elif class_name == 'field_corner':
                field_landmark.type = FieldLandmark.FIELD_CORNER
                field_landmark.yolo_class_id = next((cid for cid, cname in self.class_names.items() if cname == 'field_corner'), 3)
                field_landmark.localization_priority = FieldLandmark.PRIORITY_MEDIUM
                field_landmark.description = "field_corner"
                field_landmark.uniqueness_score = 0.5
                field_corners.append(field_landmark)
                
            elif class_name == 'area_corner':
                field_landmark.type = FieldLandmark.AREA_CORNER
                field_landmark.yolo_class_id = next((cid for cid, cname in self.class_names.items() if cname == 'area_corner'), 0)
                field_landmark.localization_priority = FieldLandmark.PRIORITY_MEDIUM  # Corrigido de PRIORITY_LOW
                field_landmark.description = "area_corner"
                field_landmark.uniqueness_score = 0.5  # Aumentado de 0.3 para 0.5
                area_corners.append(field_landmark)
        
        # Preencher landmarks na mensagem
        unified_msg.penalty_marks = penalty_marks
        unified_msg.goals = goals
        unified_msg.center_circles = center_circles
        unified_msg.field_corners = field_corners
        unified_msg.area_corners = area_corners
        unified_msg.goal_detections = goal_detections
        
        # Criar FieldDetection para landmarks_field
        if organized['landmarks']:
            field_det = FieldDetection()
            field_det.header = header
            field_det.field_detected = True
            field_det.landmarks_detected = True
            field_det.num_landmarks = len(organized['landmarks'])
            field_det.num_penalty_marks = len(penalty_marks)
            field_det.num_goals = len(goals)
            field_det.num_center_circles = len(center_circles)
            field_det.num_field_corners = len(field_corners)
            field_det.num_area_corners = len(area_corners)
            field_det.confidence = np.mean([l['confidence'] for l in organized['landmarks']])
            field_det.detection_method = "yolov8"
            field_det.sufficient_for_localization = len(organized['landmarks']) >= 2
            field_det.center_circle_detected = len(center_circles) > 0
            field_det.penalty_marks_detected = len(penalty_marks) > 0
            field_det.high_priority_landmarks = len(penalty_marks) + len(center_circles)
            
            # Status de localização
            if field_det.sufficient_for_localization and len(penalty_marks) > 0:
                field_det.localization_status = "excellent"
                field_det.localization_confidence = 0.9
            elif field_det.sufficient_for_localization:
                field_det.localization_status = "good"  
                field_det.localization_confidence = 0.7
            else:
                field_det.localization_status = "poor"
                field_det.localization_confidence = 0.3
            
            unified_msg.field_landmarks = field_det
        
        unified_msg.num_landmarks = len(organized['landmarks'])
        
        # === METADADOS DO SISTEMA SIMPLIFICADO ===
        processing_time = time.time() - self.frame_start_time if hasattr(self, 'frame_start_time') else 0.0
        unified_msg.yolo_processing_time_ms = processing_time * 1000.0
        unified_msg.post_processing_time_ms = 5.0  # Estimativa
        unified_msg.total_processing_time_ms = unified_msg.yolo_processing_time_ms + unified_msg.post_processing_time_ms
        unified_msg.fps = 1.0 / max(processing_time, 0.001)
        
        # Status do sistema
        unified_msg.yolo_active = True
        unified_msg.processing_mode = "yolov8"
        unified_msg.frame_number = getattr(self, 'frame_count', 0)
        
        # Qualidade da imagem (estimativas básicas)
        unified_msg.image_brightness = 0.5
        unified_msg.image_contrast = 0.7
        unified_msg.lighting_adequate = True
        unified_msg.lighting_condition = "good"
        
        # Resumo das detecções
        unified_msg.total_strategy_objects = len(organized['balls']) + len(organized['robots'])
        unified_msg.total_localization_landmarks = len(organized['landmarks'])
        
        # Contadores por confiança
        all_detections = organized['balls'] + organized['robots'] + organized['landmarks']
        unified_msg.high_confidence_detections = len([d for d in all_detections if d['confidence'] > 0.7])
        unified_msg.medium_confidence_detections = len([d for d in all_detections if 0.5 <= d['confidence'] <= 0.7])
        unified_msg.low_confidence_detections = len([d for d in all_detections if d['confidence'] < 0.5])
        
        # Confiança média por classe
        if organized['balls']:
            unified_msg.ball_avg_confidence = np.mean([b['confidence'] for b in organized['balls']])
        if organized['robots']:
            unified_msg.robot_avg_confidence = np.mean([r['confidence'] for r in organized['robots']])
        if penalty_marks:
            unified_msg.penalty_mark_avg_confidence = np.mean([p.confidence for p in penalty_marks])
        if goals:
            unified_msg.goal_avg_confidence = np.mean([g.confidence for g in goals])
        if center_circles:
            unified_msg.center_circle_avg_confidence = np.mean([c.confidence for c in center_circles])
        if field_corners:
            unified_msg.field_corner_avg_confidence = np.mean([f.confidence for f in field_corners])
        if area_corners:
            unified_msg.area_corner_avg_confidence = np.mean([a.confidence for a in area_corners])
        
        # Contexto tático básico
        unified_msg.ball_in_play = len(organized['balls']) > 0
        unified_msg.ball_near_goal = False  # TODO: Implementar lógica espacial
        unified_msg.crowded_area = len(organized['robots']) > 3
        unified_msg.clear_shot_available = len(organized['robots']) < 2
        
        # Localização e navegação
        unified_msg.sufficient_for_localization = len(organized['landmarks']) >= 2
        unified_msg.center_circle_detected = len(center_circles) > 0
        unified_msg.penalty_marks_detected = len(penalty_marks) > 0
        unified_msg.unique_landmarks_count = len(penalty_marks) + len(center_circles)
        
        if unified_msg.sufficient_for_localization and len(penalty_marks) > 0:
            unified_msg.localization_confidence = 0.9
            unified_msg.localization_status = "excellent"
        elif unified_msg.sufficient_for_localization:
            unified_msg.localization_confidence = 0.7
            unified_msg.localization_status = "good"
        else:
            unified_msg.localization_confidence = 0.3
            unified_msg.localization_status = "poor"
        
        # Debugging e diagnóstico
        unified_msg.debug_mode_active = self.publish_debug
        unified_msg.debug_info = f"YOLOv8 6-class simplified system"
        unified_msg.dropped_frames = 0
        
        # Timestamps
        current_time = time.time()
        unified_msg.timestamp_capture = current_time - processing_time
        unified_msg.timestamp_processing = current_time - processing_time * 0.5
        unified_msg.timestamp_completion = current_time
        
        # Status do hardware (estimativas)
        unified_msg.gpu_utilization = 0.6
        unified_msg.memory_usage = 0.4
        unified_msg.temperature_celsius = 55.0
        
        # Análise da cena
        if len(organized['balls']) > 0 and len(organized['robots']) > 1:
            unified_msg.scene_description = "active_game"
        elif len(organized['robots']) > 0:
            unified_msg.scene_description = "warm_up"
        elif len(organized['landmarks']) > 0:
            unified_msg.scene_description = "setup"
        else:
            unified_msg.scene_description = "empty_field"
            
        unified_msg.field_detected = len(organized['landmarks']) > 0
        unified_msg.field_coverage_ratio = 0.8  # Estimativa
        
        self.unified_pub.publish(unified_msg)
    
    def _publish_debug_image_3d(self, image: np.ndarray, detections: List[dict], 
                              objects_3d: List[Object3D], header):
        """Publica imagem de debug com informações 3D sobrepostas"""
        try:
            debug_image = image.copy()
            
            # Mapeamento de detecções para objetos 3D
            detection_3d_map = {}
            for obj_3d in objects_3d:
                for i, det in enumerate(detections):
                    if (abs(det['center_x'] - obj_3d.pixel_x) < 5 and 
                        abs(det['center_y'] - obj_3d.pixel_y) < 5):
                        detection_3d_map[i] = obj_3d
                        break
            
            # Cores por tipo de objeto
            colors = {
                'ball': (0, 255, 0),        # Verde
                'robot': (255, 100, 100),   # Azul claro
                'penalty_mark': (0, 255, 255),  # Amarelo
                'goal_post': (255, 0, 255),      # Magenta
                'center_circle': (255, 255, 0),  # Ciano
                'field_corner': (128, 0, 255),   # Roxo
                'area_corner': (255, 128, 0)     # Laranja
            }
            
            for i, detection in enumerate(detections):
                # Extrair informações da detecção
                bbox = detection['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                center_x = int(detection['center_x'])
                center_y = int(detection['center_y'])
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Obter cor para a classe
                color = colors.get(class_name, (255, 255, 255))
                
                # Desenhar bounding box
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
                
                # Desenhar centro
                cv2.circle(debug_image, (center_x, center_y), 5, color, -1)
                
                # Preparar texto base
                text_lines = [f'{class_name}: {confidence:.2f}']
                
                # Adicionar informações 3D se disponíveis
                obj_3d = detection_3d_map.get(i)
                if obj_3d is not None:
                    text_lines.append(f'Dist: {obj_3d.distance:.2f}m')
                    text_lines.append(f'Pos: ({obj_3d.world_x:.2f}, {obj_3d.world_y:.2f})')
                    text_lines.append(f'Size: {obj_3d.estimated_real_size:.3f}m')
                    text_lines.append(f'Conf: {obj_3d.size_confidence:.2f}')
                    
                    # Desenhar linha até o chão projetado
                    if obj_3d.world_x is not None:
                        ground_pixel = self._project_world_to_pixel(obj_3d.world_x, obj_3d.world_y)
                        if ground_pixel is not None:
                            ground_x, ground_y = map(int, ground_pixel)
                            cv2.line(debug_image, (center_x, center_y), (ground_x, ground_y), color, 1)
                            cv2.circle(debug_image, (ground_x, ground_y), 3, color, -1)
                
                # Desenhar texto
                y_offset = y1 - 10
                for line in text_lines:
                    cv2.putText(debug_image, line, (x1, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset -= 15
            
            # (Removido) Sobreposição de informações de câmera para evitar poluir a tela
            
            # Publicar imagem de debug
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            debug_msg.header = header
            self.debug_image_pub.publish(debug_msg)
            try:
                self.debug_image_pub_rel.publish(debug_msg)
            except Exception:
                pass
            
        except Exception as e:
            self.get_logger().error(f'❌ Erro ao publicar imagem de debug 3D: {e}')
    
    def _draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Desenha detecções na imagem com cores específicas"""
        debug_image = image.copy()
        
        # Cores para cada classe
        colors = {
            'ball': (0, 255, 0),           # Verde - bola
            'robot': (255, 0, 0),          # Azul - robôs
            'penalty_mark': (0, 255, 255), # Amarelo - marca penalty
            'goal_post': (255, 255, 0),    # Ciano - poste de gol
            'center_circle': (255, 0, 255), # Magenta - círculo central
            'field_corner': (128, 0, 128),  # Roxo - canto do campo
            'area_corner': (255, 128, 0)    # Laranja - canto da área
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox'].astype(int)
            class_name = det['class_name']
            confidence = det['confidence']
            
            color = colors.get(class_name, (128, 128, 128))
            
            # Desenhar bounding box
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
            
            # Desenhar label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(debug_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(debug_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Estatísticas no canto
        stats_text = f"Classes: 6 | Deteccoes: {len(detections)}"
        cv2.putText(debug_image, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_image
    
    def _estimate_ball_distance(self, ball_detection: dict) -> float:
        """Estima distância da bola baseada no tamanho"""
        # Estimativa simples baseada na altura do bounding box
        # TODO: Implementar calibração real da câmera
        height = ball_detection['height']
        return max(0.1, 100.0 / height)  # Estimativa aproximada
    
    def _estimate_robot_distance(self, robot_detection: dict) -> float:
        """Estima distância do robô baseada no tamanho"""
        height = robot_detection['height']
        return max(0.1, 300.0 / height)  # Estimativa aproximada
    
    def _update_stats(self, detections: List[dict], processing_time: float):
        """Atualiza estatísticas de performance"""
        self.frame_count += 1
        self.total_detections += len(detections)
        
        current_time = time.time()
        if current_time - self.last_stats_time >= 10.0:  # Log a cada 10 segundos
            fps = self.frame_count / (current_time - self.last_stats_time)
            avg_detections = self.total_detections / self.frame_count if self.frame_count > 0 else 0
            
            self.get_logger().info(
                f'Stats: {fps:.1f} FPS | '
                f'Avg detections: {avg_detections:.1f} | '
                f'Processing: {processing_time*1000:.1f}ms'
            )
            
            # Reset counters
            self.last_stats_time = current_time
            self.frame_count = 0
            self.total_detections = 0

    def _estimate_landmark_distance(self, landmark: dict) -> float:
        """Estima distância para um landmark baseado em tamanho da bbox"""
        # Estimativa simples baseada na altura da bbox
        # Landmarks maiores estão mais próximos
        height = landmark['height']
        width = landmark['width']
        
        # Valores estimados para diferentes tipos de landmarks
        class_name = landmark['class_name']
        if class_name == 'penalty_mark':
            # Marca de penalty: ~0.22m de diâmetro
            reference_size = 50  # pixels em 2m de distância
            reference_distance = 2.0
        elif class_name == 'goal_post':
            # Postes de gol (~80cm altura visível, largura do poste ~10cm)
            reference_size = 160  # pixels em 5m de distância (ajuste heurístico)
            reference_distance = 5.0
        elif class_name == 'center_circle':
            # Círculo central: ~1.5m de diâmetro
            reference_size = 100  # pixels em 3m de distância
            reference_distance = 3.0
        else:
            # Cantos e outros landmarks
            reference_size = 30
            reference_distance = 4.0
        
        # Calcular distância baseada no tamanho
        avg_size = (height + width) / 2
        if avg_size > 0:
            distance = reference_distance * reference_size / max(avg_size, 1)
            return min(max(distance, 0.5), 15.0)  # Limitar entre 0.5m e 15m
        else:
            return 5.0  # Valor padrão
    
    def _calculate_bearing(self, center_x: float, center_y: float) -> float:
        """Calcula ângulo relativo (bearing) baseado na posição na imagem"""
        # Assumir resolução padrão 1280x720
        image_width = 1280
        image_height = 720
        
        # Campo de visão horizontal da câmera (estimativa para IMX219/C930)
        horizontal_fov = np.radians(80)  # ~80 graus
        
        # Calcular ângulo horizontal
        # Centro da imagem = 0 radianos (frente do robô)
        # Esquerda = negativo, direita = positivo
        pixel_offset = center_x - (image_width / 2)
        bearing = (pixel_offset / (image_width / 2)) * (horizontal_fov / 2)
        
        return bearing

    def robot_pose_callback(self, msg: RobotPose2D):
        """Atualiza a pose do robô a partir do tópico 'robot_pose'"""
        # Normaliza para um objeto simples com .x, .y, .theta para evitar incompatibilidades
        x, y, theta = self._extract_xytheta(msg)
        class _SimplePose2D:
            __slots__ = ('x', 'y', 'theta')
            def __init__(self, x_v: float, y_v: float, t_v: float):
                self.x = x_v
                self.y = y_v
                self.theta = t_v
        self.current_robot_pose = _SimplePose2D(x, y, theta)
        self.pose_timestamp = time.time()  # Usar timestamp atual

    def _load_landmark_coordinates(self):
        """Carrega as coordenadas conhecidas dos landmarks do arquivo YAML"""
        config_path = self.get_parameter('landmark_config_path').value
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Extrair coordenadas dos diferentes tipos de landmarks
            self.landmark_coordinates = {
                'penalty_mark': [],
                'center_circle': [],
                'goal_post': [],
                'field_corner': [],
                'area_corner': []
            }
            
            # Penalty marks
            if 'penalty_marks' in config and 'coordinates' in config['penalty_marks']:
                for coord_data in config['penalty_marks']['coordinates'].values():
                    self.landmark_coordinates['penalty_mark'].append([coord_data['x'], coord_data['y']])
            
            # Center circle
            if 'center_circle' in config and 'coordinates' in config['center_circle']:
                center = config['center_circle']['coordinates']['center'] if 'center' in config['center_circle']['coordinates'] else config['center_circle']['coordinates']['circle_center']
                self.landmark_coordinates['center_circle'].append([center['x'], center['y']])
            
            # Goals (posts)
            if 'goals' in config and 'coordinates' in config['goals']:
                for coord_data in config['goals']['coordinates']:
                    self.landmark_coordinates['goal_post'].append([coord_data['x'], coord_data['y']])
            
            # Field corners
            if 'field_corners' in config and 'coordinates' in config['field_corners']:
                for coord_data in config['field_corners']['coordinates']:
                    self.landmark_coordinates['field_corner'].append([coord_data['x'], coord_data['y']])
            
            # Area corners
            if 'penalty_area_corners' in config and 'coordinates' in config['penalty_area_corners']:
                for coord_data in config['penalty_area_corners']['coordinates']:
                    self.landmark_coordinates['area_corner'].append([coord_data['x'], coord_data['y']])
            
            self.get_logger().info(f"Landmark coordinates loaded from {config_path}")
            for landmark_type, coords in self.landmark_coordinates.items():
                self.get_logger().debug(f"Type '{landmark_type}': {len(coords)} landmarks")
                
        except FileNotFoundError:
            self.get_logger().warn(f"Landmark configuration file not found at {config_path}")
            self._use_default_coordinates()
        except Exception as e:
            self.get_logger().error(f"Error loading landmark coordinates: {e}")
            self._use_default_coordinates()

    def _use_default_coordinates(self):
        """Usa coordenadas padrão do campo RoboCup Humanoid (9x6m)"""
        self.landmark_coordinates = {
            'penalty_mark': [[-3.0, 0.0], [3.0, 0.0]],
            'center_circle': [[0.0, 0.0]],
            'goal_post': [[-4.5, 1.3], [-4.5, -1.3], [4.5, 1.3], [4.5, -1.3]],
            'field_corner': [[-4.5, -3.0], [-4.5, 3.0], [4.5, -3.0], [4.5, 3.0]],
            'area_corner': [
                [-2.85, -1.95], [-2.85, 1.95], [-4.5, -1.95], [-4.5, 1.95],
                [2.85, -1.95], [2.85, 1.95], [4.5, -1.95], [4.5, 1.95]
            ]
        }
        self.get_logger().info("Using default RoboCup field coordinates")

    def _extract_xytheta(self, pose) -> Tuple[float, float, float]:
        """
        Extrai (x, y, theta) de diferentes tipos de mensagens de pose de forma robusta.
        Suporta `RobotPose2D`, `geometry_msgs/Pose2D` e estruturas aninhadas comuns.
        """
        try:
            if pose is None:
                return 0.0, 0.0, 0.0
            # Tentativa direta (RobotPose2D ou geometry_msgs/Pose2D)
            if hasattr(pose, 'x') and hasattr(pose, 'y') and hasattr(pose, 'theta'):
                return float(pose.x), float(pose.y), float(pose.theta)
            # Pose2D aninhada em campo `pose` (ex.: PoseWithCovarianceStamped -> pose.pose)
            inner = getattr(pose, 'pose', None)
            if inner is not None:
                # geometry_msgs/Pose -> position (x,y) e orientação (theta ausente)
                if hasattr(inner, 'pose'):
                    geo = inner.pose
                    if hasattr(geo, 'position'):
                        x = float(getattr(geo.position, 'x', 0.0))
                        y = float(getattr(geo.position, 'y', 0.0))
                        # Sem theta disponível, assumir 0.0
                        return x, y, 0.0
                # geometry_msgs/Pose2D diretamente
                if hasattr(inner, 'x') and hasattr(inner, 'y') and hasattr(inner, 'theta'):
                    return float(inner.x), float(inner.y), float(inner.theta)
        except Exception:
            pass
        return 0.0, 0.0, 0.0

    def _calculate_absolute_position_from_pose(self, field_landmark: FieldLandmark) -> Point:
        """
        Calcula posição absoluta do landmark baseado na pose atual do robô
        usando transformação de coordenadas 2D
        """
        if self.current_robot_pose is None:
            self.get_logger().debug("No robot pose available for absolute position calculation")
            return Point()  # Retornar posição zero se não há pose
        
        # Transformação do frame do robô para frame do mapa
        _, _, theta_r = self._extract_xytheta(self.current_robot_pose)
        cos_theta = np.cos(theta_r)
        sin_theta = np.sin(theta_r)
        
        # Matriz de rotação 2D
        R = np.array([[cos_theta, -sin_theta],
                      [sin_theta,  cos_theta]])
        
        # Posição relativa no frame do robô
        pos_relative = np.array([field_landmark.position_relative.x,
                                field_landmark.position_relative.y])
        
        # Transformar para frame do mapa: p_global = p_robot + R * p_relative
        x_r, y_r, _ = self._extract_xytheta(self.current_robot_pose)
        pos_global = np.array([x_r, y_r]) + R @ pos_relative
        
        # Retornar como Point
        point = Point()
        point.x = float(pos_global[0])
        point.y = float(pos_global[1])
        point.z = 0.0
        
        return point

    def _find_closest_known_landmark(self, landmark_type: str, observed_position: Point) -> Point:
        """
        Encontra o landmark conhecido mais próximo da posição observada
        """
        if landmark_type not in self.landmark_coordinates:
            return Point()  # Tipo desconhecido
        
        known_landmarks = self.landmark_coordinates[landmark_type]
        if not known_landmarks:
            return Point()  # Nenhum landmark deste tipo conhecido
        
        # Se há apenas um landmark deste tipo, retornar ele
        if len(known_landmarks) == 1:
            point = Point()
            point.x = float(known_landmarks[0][0])
            point.y = float(known_landmarks[0][1])
            point.z = 0.0
            return point
        
        # Para múltiplos landmarks, encontrar o mais próximo
        min_distance = float('inf')
        closest_landmark = known_landmarks[0]
        
        for landmark_coord in known_landmarks:
            distance = np.sqrt(
                (observed_position.x - landmark_coord[0])**2 + 
                (observed_position.y - landmark_coord[1])**2
            )
            if distance < min_distance:
                min_distance = distance
                closest_landmark = landmark_coord
        
        point = Point()
        point.x = float(closest_landmark[0])
        point.y = float(closest_landmark[1])
        point.z = 0.0
        
        return point

    def _calculate_absolute_position(self, field_landmark: FieldLandmark, class_name: str) -> Point:
        """
        Calcula posição absoluta do landmark usando a melhor abordagem disponível:
        1. Se há pose do robô: usa transformação de coordenadas
        2. Senão: usa correspondência com landmarks conhecidos
        """
        
        # Método 1: Transformação baseada na pose do robô (mais preciso)
        if self.current_robot_pose is not None:
            pose_age = time.time() - self.pose_timestamp
            if pose_age < 1.0:  # Pose recente (menos de 1 segundo)
                return self._calculate_absolute_position_from_pose(field_landmark)
        
        # Método 2: Correspondência com landmarks conhecidos (fallback)
        # Primeiro, calcular posição estimada usando transformação com pose zero
        temp_pose = RobotPose2D()
        try:
            temp_pose.x = 0.0
            temp_pose.y = 0.0
            temp_pose.theta = 0.0
        except Exception:
            # Em alguns ambientes, o tipo pode vir como geometry_msgs/Pose2D
            setattr(temp_pose, 'x', 0.0)
            setattr(temp_pose, 'y', 0.0)
            setattr(temp_pose, 'theta', 0.0)
        
        # Salvar pose atual e usar temporária
        original_pose = self.current_robot_pose
        self.current_robot_pose = temp_pose
        
        # Calcular posição estimada
        estimated_position = self._calculate_absolute_position_from_pose(field_landmark)
        
        # Restaurar pose original
        self.current_robot_pose = original_pose
        
        # Encontrar landmark conhecido mais próximo
        return self._find_closest_known_landmark(class_name, estimated_position)

    def _calculate_geometric_confidence(self, landmark: dict, estimated_distance: float) -> float:
        """
        Calcula confiança baseada na consistência geométrica do landmark
        """
        class_name = landmark['class_name']
        observed_size = (landmark['width'] + landmark['height']) / 2
        
        # Tamanhos esperados baseados na distância estimada
        expected_sizes = {
            'penalty_mark': 50 * (2.0 / max(estimated_distance, 0.5)),
            'goal_post': 160 * (5.0 / max(estimated_distance, 0.5)), 
            'center_circle': 100 * (3.0 / max(estimated_distance, 0.5)),
            'field_corner': 30 * (4.0 / max(estimated_distance, 0.5)),
            'area_corner': 30 * (4.0 / max(estimated_distance, 0.5))
        }
        
        expected_size = expected_sizes.get(class_name, 50)
        
        # Calcular ratio de tamanho (quão próximo está do esperado)
        if expected_size > 0:
            size_ratio = min(observed_size / expected_size, expected_size / observed_size)
            # Mapear [0.5, 1.0] para [0.0, 1.0]
            geometric_confidence = max(0.0, (size_ratio - 0.5) * 2.0)
        else:
            geometric_confidence = 0.5  # Fallback
        
        return geometric_confidence

    def _validate_landmark_position(self, field_landmark: FieldLandmark, class_name: str) -> bool:
        """
        Valida se a posição absoluta calculada está dentro dos limites esperados
        """
        pos = field_landmark.position_absolute
        
        # Limites do campo RoboCup Humanoid (9x6m)
        if abs(pos.x) > 5.0 or abs(pos.y) > 4.0:
            self.get_logger().debug(f"Landmark {class_name} fora dos limites: ({pos.x:.2f}, {pos.y:.2f})")
            return False
        
        # Validações específicas por tipo
        if class_name == 'center_circle':
            distance_from_center = np.sqrt(pos.x**2 + pos.y**2)
            if distance_from_center > 2.0:
                return False
        elif class_name == 'penalty_mark':
            if abs(pos.y) > 0.5 or abs(abs(pos.x) - 3.0) > 0.5:
                return False
        elif class_name == 'goal_post':
            # Aproximação de placa em x≈±4.5 e y≈±1.3
            tol_x = 0.5; tol_y = 0.5
            if not (abs(abs(pos.x) - 4.5) < tol_x and abs(abs(pos.y) - 1.3) < tol_y):
                return False
        
        return True

    def _compute_goal_from_posts(self, goal_posts: List[dict]):
        """A partir de 2+ postes, deriva centro/orientação/lado e métricas.
        Retorna (centro(x,y), left_post, right_post, largura_est, lado, distância, bearing, confiança) ou None."""
        if not self.goal_pairing_enabled:
            return None
            
        # Filtrar somente com coordenadas 3D válidas
        pts = []
        for g in goal_posts:
            if g.get('world_x') is None or g.get('world_y') is None:
                # tentar projetar com bearing/dist se possível (não implementado aqui)
                continue
            pts.append((g['world_x'], g['world_y'], g.get('confidence', 0.5)))
        if len(pts) < 2:
            return None
        
        # Filtrar postes que estão dentro das tolerâncias esperadas
        valid_pts = []
        for x, y, c in pts:
            # Validar se está próximo das placas conhecidas (x≈±4.5) e posições dos postes (y≈±1.3)
            if (abs(abs(x) - 4.5) < self.plate_x_tolerance and 
                abs(abs(y) - 1.3) < self.post_y_tolerance):
                valid_pts.append((x, y, c))
        
        if len(valid_pts) < 2:
            return None
        
        # Separar por placa (x negativo e x positivo)
        left_side = [(x, y, c) for (x, y, c) in valid_pts if x < 0]
        right_side = [(x, y, c) for (x, y, c) in valid_pts if x > 0]

        def pair_best(side_pts):
            best = None
            best_score = -1.0
            for i in range(len(side_pts)):
                for j in range(i+1, len(side_pts)):
                    x1, y1, c1 = side_pts[i]
                    x2, y2, c2 = side_pts[j]
                    width = np.hypot(x2-x1, y2-y1)
                    # Valida distância aproximada 2.6m e simetria vertical usando tolerâncias configuradas
                    dist_score = max(0.0, 1.0 - abs(width-2.6)/self.goal_width_tolerance)
                    mid_y = (y1+y2)/2.0
                    mid_score = max(0.0, 1.0 - abs(mid_y)/self.goal_center_y_tolerance)
                    conf_score = (c1+c2)/2.0
                    score = 0.5*dist_score + 0.2*mid_score + 0.3*conf_score
                    if score > best_score:
                        best_score = score
                        # left/right post by y sign (right_post with y<0 for lower?)
                        a = (x1, y1)
                        b = (x2, y2)
                        # Define ordering by y: upper is positive y
                        left_p = a if y1>y2 else b
                        right_p = b if y1>y2 else a
                        best = (left_p, right_p, width, (c1+c2)/2.0)
            return best
        left_pair = pair_best(left_side) if len(left_side)>=2 else None
        right_pair = pair_best(right_side) if len(right_side)>=2 else None
        
        pair = left_pair if left_pair and (not right_pair or left_pair[3] >= right_pair[3]) else right_pair
        if pair is None:
            return None
        left_p, right_p, width_est, conf = pair
        c = ((left_p[0]+right_p[0])/2.0, (left_p[1]+right_p[1])/2.0)
        v = (right_p[0]-left_p[0], right_p[1]-left_p[1])
        # Normal (apontando para o campo): +90° (não usado mas disponível se necessário)
        # alpha = np.arctan2(v[1], v[0]) + np.pi/2.0
        side = "left" if c[0] < 0 else "right"
        if self.current_robot_pose is None:
            distance = np.hypot(c[0], c[1])
        else:
            rx, ry, _ = self._extract_xytheta(self.current_robot_pose)
            distance = np.hypot(c[0]-rx, c[1]-ry)
        # Bearing relativo ao robô
        if self.current_robot_pose is None:
            bearing = np.arctan2(c[1], c[0])
        else:
            rx, ry, rt = self._extract_xytheta(self.current_robot_pose)
            bearing = np.arctan2(c[1]-ry, c[0]-rx) - rt
        # Ajuste de confiança por coerência geométrica usando tolerâncias configuradas
        width_consistency = max(0.0, 1.0 - abs(width_est-2.6)/self.goal_width_tolerance)
        center_y_consistency = max(0.0, 1.0 - abs(c[1])/self.goal_center_y_tolerance)
        final_conf = float(max(0.0, min(1.0, 0.5*conf + 0.3*width_consistency + 0.2*center_y_consistency)))
        
        # Verificar se atende confiança mínima para par
        if final_conf < self.min_pair_confidence:
            return None
            
        return c, left_p, right_p, width_est, side, distance, bearing, final_conf

    def _project_world_to_pixel(self, world_x: float, world_y: float) -> Optional[Tuple[float, float]]:
        """
        Projeta coordenadas do mundo real para pixels (processo inverso)
        
        Args:
            world_x: Coordenada X no mundo
            world_y: Coordenada Y no mundo
            
        Returns:
            Tupla (pixel_x, pixel_y) ou None se inválido
        """
        if not self.geometry_3d:
            return None
            
        try:
            # Ponto no mundo (assumindo Z=0 para o chão)
            world_point = np.array([world_x, world_y, 0.0, 1.0])
            
            # Transformar para frame da câmera
            T_inv = np.linalg.inv(self.geometry_3d.T_camera)
            camera_point = T_inv @ world_point
            
            # Projetar para coordenadas da imagem
            if camera_point[2] <= 0:
                return None  # Ponto atrás da câmera
            
            x_norm = camera_point[0] / camera_point[2]
            y_norm = camera_point[1] / camera_point[2]
            
            # Aplicar parâmetros intrínsecos
            pixel_x = x_norm * self.geometry_3d.geometry.fx + self.geometry_3d.geometry.cx
            pixel_y = y_norm * self.geometry_3d.geometry.fy + self.geometry_3d.geometry.cy
            
            return pixel_x, pixel_y
            
        except Exception:
            return None

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8UnifiedDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 