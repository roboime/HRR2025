#!/usr/bin/env python3
"""
Módulo de Geometria 3D para RoboIME HSL2025
Sistema avançado de cálculos 3D usando calibração da câmera e altura do robô

Funcionalidades:
- Conversão pixel → coordenadas mundo real
- Cálculo de distâncias baseado na altura da câmera
- Validação de posições usando tamanhos conhecidos de objetos
- Correção de perspectiva e distorção
- Mapeamento para coordenadas do campo RoboCup
"""

import numpy as np
import cv2
import math
import yaml
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from geometry_msgs.msg import Point, Pose2D

@dataclass
class CameraGeometry:
    """Parâmetros geométricos da câmera"""
    # Parâmetros intrínsecos
    fx: float  # Distância focal X (pixels)
    fy: float  # Distância focal Y (pixels) 
    cx: float  # Centro óptico X (pixels)
    cy: float  # Centro óptico Y (pixels)
    
    # Parâmetros de distorção
    k1: float  # Coeficiente radial 1
    k2: float  # Coeficiente radial 2
    p1: float  # Coeficiente tangencial 1
    p2: float  # Coeficiente tangencial 2
    k3: float  # Coeficiente radial 3
    
    # Geometria da instalação
    height: float       # Altura da câmera (metros)
    tilt_angle: float   # Ângulo de inclinação (radianos)
    
    # Dimensões da imagem
    image_width: int
    image_height: int

@dataclass
class Object3D:
    """Objeto detectado com informações 3D"""
    # Coordenadas na imagem
    pixel_x: float
    pixel_y: float
    bbox_width: float
    bbox_height: float
    
    # Coordenadas no mundo real
    world_x: float
    world_y: float
    world_z: float
    distance: float
    
    # Informações de validação
    estimated_real_size: float
    expected_real_size: float
    size_confidence: float
    
    # Metadados
    object_type: str
    detection_confidence: float

class CameraGeometry3D:
    """
    Classe principal para cálculos de geometria 3D da câmera
    
    Implementa transformações complexas entre coordenadas de pixel
    e coordenadas do mundo real usando calibração da câmera.
    """
    
    def __init__(self, camera_info_path: str):
        """
        Inicializa com parâmetros de calibração
        
        Args:
            camera_info_path: Caminho para arquivo YAML de calibração
        """
        self.geometry = self._load_camera_geometry(camera_info_path)
        self.field_dimensions = None
        self.object_sizes = None
        
        # Matrizes de câmera pré-calculadas
        self._setup_camera_matrices()
        
        # Cache para otimização
        self._undistort_maps = None
        
    def _load_camera_geometry(self, config_path: str) -> CameraGeometry:
        """Carrega parâmetros de geometria do arquivo YAML"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Determinar câmera ativa
            active_camera = config.get('active_camera', 'imx219_csi')
            camera_config = config[active_camera]
            
            # Extrair matriz da câmera
            camera_matrix = np.array(camera_config['camera_matrix']['data']).reshape(3, 3)
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
            
            # Extrair coeficientes de distorção
            dist_coeffs = camera_config['distortion_coefficients']['data']
            k1, k2, p1, p2, k3 = dist_coeffs
            
            # Parâmetros geométricos
            height = camera_config['camera_height']
            tilt_angle = camera_config['camera_tilt_angle']
            
            # Dimensões da imagem
            width = camera_config['image_width']
            height_img = camera_config['image_height']
            
            # Definir tamanhos padrão de objetos do RoboCup (em metros)
            # Não dependemos mais de configuração externa - valores conhecidos
            self.object_sizes = {
                'ball': 0.065,           # Diâmetro da bola RoboCup
                'robot': 0.9,            # Altura típica do robô humanóide
                'penalty_mark': 0.10,    # Diâmetro da marca do penalty
                'goal_post': 0.1,        # Largura aproximada do poste
                'center_circle': 1.5,    # Diâmetro do círculo central
                'field_corner': 0.2,     # Tamanho estimado dos cantos
                'area_corner': 0.2       # Tamanho estimado dos cantos da área
            }
            
            return CameraGeometry(
                fx=fx, fy=fy, cx=cx, cy=cy,
                k1=k1, k2=k2, p1=p1, p2=p2, k3=k3,
                height=height, tilt_angle=tilt_angle,
                image_width=width, image_height=height_img
            )
            
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar geometria da câmera: {e}")
    
    def _setup_camera_matrices(self):
        """Configura matrizes da câmera para cálculos 3D"""
        # Matriz intrínseca K
        self.K = np.array([
            [self.geometry.fx, 0, self.geometry.cx],
            [0, self.geometry.fy, self.geometry.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Coeficientes de distorção
        self.dist_coeffs = np.array([
            self.geometry.k1, self.geometry.k2, 
            self.geometry.p1, self.geometry.p2, self.geometry.k3
        ], dtype=np.float32)
        
        # Matriz de rotação da câmera (inclinação)
        self.R_camera = self._get_camera_rotation_matrix()
        
        # Matriz de transformação completa (rotação + translação)
        self.T_camera = self._get_camera_transform_matrix()
    
    def _get_camera_rotation_matrix(self) -> np.ndarray:
        """
        Calcula matriz de rotação da câmera baseada na inclinação
        
        Returns:
            Matriz 3x3 de rotação
        """
        # Rotação em torno do eixo X (pitch)
        theta = self.geometry.tilt_angle
        
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        
        return R_x
    
    def _get_camera_transform_matrix(self) -> np.ndarray:
        """
        Calcula matriz de transformação 4x4 completa da câmera
        
        Returns:
            Matriz 4x4 de transformação homogênea
        """
        T = np.eye(4)
        T[:3, :3] = self.R_camera
        T[:3, 3] = [0, 0, self.geometry.height]  # Translação Z = altura
        
        return T
    
    def pixel_to_world_ground_plane(self, pixel_x: float, pixel_y: float) -> Tuple[float, float, float]:
        """
        Converte coordenadas de pixel para coordenadas do mundo real no plano do chão
        
        Algoritmo:
        1. Undistort pixel coordinates
        2. Convert to normalized camera coordinates  
        3. Apply camera rotation
        4. Project ray to ground plane (Z=0)
        5. Calculate world coordinates
        
        Args:
            pixel_x: Coordenada X do pixel
            pixel_y: Coordenada Y do pixel
            
        Returns:
            Tupla (world_x, world_y, distance) em metros
        """
        
        # 1. Undistort pixel coordinates
        undistorted_pixel = self._undistort_point(pixel_x, pixel_y)
        u, v = undistorted_pixel
        
        # 2. Convert to normalized camera coordinates
        x_norm = (u - self.geometry.cx) / self.geometry.fx
        y_norm = (v - self.geometry.cy) / self.geometry.fy
        
        # 3. Ray direction in camera frame
        ray_camera = np.array([x_norm, y_norm, 1.0])
        ray_camera = ray_camera / np.linalg.norm(ray_camera)
        
        # 4. Transform ray to world frame
        ray_world = self.R_camera @ ray_camera
        
        # 5. Camera position in world frame
        camera_pos = np.array([0, 0, self.geometry.height])
        
        # 6. Intersect ray with ground plane (Z = 0)
        # Ray equation: P = camera_pos + t * ray_world
        # Ground plane: Z = 0
        # Solve: camera_pos[2] + t * ray_world[2] = 0
        
        if abs(ray_world[2]) < 1e-6:
            # Ray is parallel to ground - no intersection
            return None, None, None
        
        t = -camera_pos[2] / ray_world[2]
        
        if t <= 0:
            # Intersection is behind camera
            return None, None, None
        
        # 7. Calculate world coordinates
        world_point = camera_pos + t * ray_world
        world_x, world_y, world_z = world_point
        
        # 8. Calculate distance
        distance = np.sqrt(world_x**2 + world_y**2)
        
        return world_x, world_y, distance
    
    def _undistort_point(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Remove distorção de um ponto específico
        
        Args:
            pixel_x: Coordenada X original
            pixel_y: Coordenada Y original
            
        Returns:
            Tupla (x_undistorted, y_undistorted)
        """
        # Usar OpenCV para undistort
        points = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(points, self.K, self.dist_coeffs, P=self.K)
        
        return undistorted[0, 0, 0], undistorted[0, 0, 1]
    
    def estimate_object_real_size(self, bbox_width: float, bbox_height: float, 
                                world_x: float, world_y: float, distance: float) -> float:
        """
        Estima o tamanho real do objeto baseado no tamanho da bounding box
        
        Args:
            bbox_width: Largura da bbox em pixels
            bbox_height: Altura da bbox em pixels  
            world_x: Posição X no mundo
            world_y: Posição Y no mundo
            distance: Distância do objeto
            
        Returns:
            Tamanho estimado real (diâmetro em metros)
        """
        
        # Calcular tamanho angular
        angular_width = bbox_width / self.geometry.fx
        angular_height = bbox_height / self.geometry.fy
        
        # Converter para tamanho real baseado na distância
        real_width = angular_width * distance
        real_height = angular_height * distance
        
        # Retornar diâmetro médio
        return (real_width + real_height) / 2.0
    
    def validate_object_detection(self, object_3d: Object3D) -> float:
        """
        Valida detecção 3D comparando tamanho estimado vs esperado
        
        Args:
            object_3d: Objeto 3D detectado
            
        Returns:
            Confiança da validação (0.0 a 1.0)
        """
        if not self.object_sizes or object_3d.object_type not in self.object_sizes:
            return 0.5  # Confiança neutra se não há referência
        
        expected_size = self.object_sizes[object_3d.object_type]
        estimated_size = object_3d.estimated_real_size
        
        # Calcular erro relativo
        if expected_size > 0:
            relative_error = abs(estimated_size - expected_size) / expected_size
            
            # Converter erro para confiança (exponencial decay)
            confidence = np.exp(-relative_error * 3.0)  # Penalidade suave para erro
            
            return max(0.0, min(1.0, confidence))
        
        return 0.5
    
    def detect_and_compute_3d(self, detections: List[Dict], 
                            confidence_threshold: float = 0.3) -> List[Object3D]:
        """
        Processa lista de detecções 2D e calcula informações 3D
        
        Args:
            detections: Lista de detecções do YOLOv8
            confidence_threshold: Threshold mínimo de confiança
            
        Returns:
            Lista de objetos 3D válidos
        """
        objects_3d = []
        
        for detection in detections:
            if detection['confidence'] < confidence_threshold:
                continue
                
            # Extrair coordenadas
            center_x = detection['center_x']
            center_y = detection['center_y']
            bbox_width = detection['width']
            bbox_height = detection['height']
            
            # Calcular posição 3D
            world_x, world_y, distance = self.pixel_to_world_ground_plane(center_x, center_y)
            
            if world_x is None or distance is None:
                continue  # Falha na projeção
            
            # Estimar tamanho real
            estimated_size = self.estimate_object_real_size(
                bbox_width, bbox_height, world_x, world_y, distance
            )
            
            # Criar objeto 3D
            object_3d = Object3D(
                pixel_x=center_x,
                pixel_y=center_y,
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                world_x=world_x,
                world_y=world_y,
                world_z=0.0,  # Assumindo objetos no chão
                distance=distance,
                estimated_real_size=estimated_size,
                expected_real_size=self.object_sizes.get(detection['class_name'], 0.0) if self.object_sizes else 0.0,
                size_confidence=0.0,  # Será calculado na validação
                object_type=detection['class_name'],
                detection_confidence=detection['confidence']
            )
            
            # Validar detecção
            object_3d.size_confidence = self.validate_object_detection(object_3d)
            
            # Filtrar objetos com baixa confiança total
            total_confidence = object_3d.detection_confidence * object_3d.size_confidence
            if total_confidence > 0.2:  # Threshold combinado
                objects_3d.append(object_3d)
        
        return objects_3d
    
    def project_to_field_coordinates(self, world_x: float, world_y: float, 
                                   robot_pose: Optional[Pose2D] = None) -> Point:
        """
        Converte coordenadas relativas da câmera para coordenadas do campo RoboCup
        
        Args:
            world_x: Coordenada X relativa à câmera
            world_y: Coordenada Y relativa à câmera  
            robot_pose: Pose atual do robô no campo (opcional)
            
        Returns:
            Point com coordenadas absolutas do campo
        """
        field_point = Point()
        
        if robot_pose is not None:
            # Transformar para coordenadas do campo usando pose do robô
            cos_theta = np.cos(robot_pose.theta)
            sin_theta = np.sin(robot_pose.theta)
            
            # Rotação + translação
            field_point.x = robot_pose.x + (world_x * cos_theta - world_y * sin_theta)
            field_point.y = robot_pose.y + (world_x * sin_theta + world_y * cos_theta)
            field_point.z = 0.0
        else:
            # Coordenadas relativas (sem pose do robô)
            field_point.x = world_x
            field_point.y = world_y  
            field_point.z = 0.0
        
        return field_point
    
    def get_camera_info_summary(self) -> Dict:
        """
        Retorna resumo das informações da câmera para debug
        
        Returns:
            Dicionário com informações principais
        """
        return {
            'camera_height': self.geometry.height,
            'tilt_angle_degrees': np.degrees(self.geometry.tilt_angle),
            'focal_length_x': self.geometry.fx,
            'focal_length_y': self.geometry.fy,
            'principal_point': (self.geometry.cx, self.geometry.cy),
            'image_size': (self.geometry.image_width, self.geometry.image_height),
            'field_of_view_x_degrees': np.degrees(2 * np.arctan(self.geometry.image_width / (2 * self.geometry.fx))),
            'field_of_view_y_degrees': np.degrees(2 * np.arctan(self.geometry.image_height / (2 * self.geometry.fy))),
            'max_distance_visible': self.geometry.height / np.tan(np.pi/2 - self.geometry.tilt_angle) if self.geometry.tilt_angle < np.pi/2 else float('inf')
        } 