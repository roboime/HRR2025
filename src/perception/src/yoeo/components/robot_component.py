#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from geometry_msgs.msg import Pose, PoseArray
from src.perception.src.yoeo.yoeo_handler import YOEOHandler, DetectionType

class RobotDetectionComponent:
    """
    Componente para detecção de robôs utilizando o YOEO.
    
    Este componente é responsável por detectar robôs na imagem,
    calcular suas posições 3D relativas ao robô, e fornecer visualizações
    para depuração.
    """
    
    def __init__(self, yoeo_handler, robot_width=0.30, robot_height=0.20):
        """
        Inicializa o componente de detecção de robôs.
        
        Args:
            yoeo_handler: Instância do YOEOHandler para acesso ao modelo
            robot_width: Largura média do robô em metros
            robot_height: Altura média do robô em metros
        """
        self.yoeo_handler = yoeo_handler
        self.robot_width = robot_width
        self.robot_height = robot_height
        self.fallback_detector = None
        self.camera_info = None
        
        # Distinguir robôs por cores (assumindo que pode haver classificações por time)
        self.team_colors = {
            'own': (0, 255, 0),    # Verde para próprio time
            'opponent': (0, 0, 255),  # Vermelho para time adversário
            'unknown': (255, 255, 0)   # Amarelo para não classificados
        }
    
    def set_camera_info(self, camera_info):
        """
        Define as informações da câmera para cálculos de posição.
        
        Args:
            camera_info: Informações da câmera (mensagem ROS CameraInfo)
        """
        self.camera_info = camera_info
    
    def set_fallback_detector(self, detector):
        """
        Define um detector tradicional para ser usado como fallback.
        
        Args:
            detector: Instância do detector tradicional
        """
        self.fallback_detector = detector
    
    def process(self, image):
        """
        Processa a imagem para detectar robôs.
        
        Args:
            image: Imagem BGR do OpenCV
            
        Returns:
            Lista de detecções de robôs, cada uma contendo:
            {
                'bbox': [x, y, width, height],
                'confidence': confiança da detecção,
                'center': (x, y) centro da detecção,
                'position': (x, y, z) posição 3D relativa ao robô,
                'team': equipe do robô ('own', 'opponent', ou 'unknown')
            }
        """
        # Verificar se a imagem é válida
        if image is None or image.size == 0:
            return []
        
        # Tentar obter as detecções do YOEO
        try:
            # Obter detecções do YOEO
            detections = self.yoeo_handler.get_detections(image, DetectionType.ROBOT)
            
            # Processar detecções se encontradas
            if detections and len(detections) > 0:
                robots = []
                
                for detection in detections:
                    # Calcular centro da caixa delimitadora
                    x, y, w, h = detection.get('bbox', [0, 0, 0, 0])
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    # Calcular posição 3D
                    position = self._calculate_3d_position((center_x, center_y), w, h)
                    
                    # Determinar a equipe (se disponível na detecção)
                    team = detection.get('team', 'unknown')
                    
                    # Adicionar informações de detecção
                    robot_info = {
                        'bbox': [x, y, w, h],
                        'confidence': detection.get('confidence', 0.0),
                        'center': (int(center_x), int(center_y)),
                        'position': position,
                        'team': team
                    }
                    
                    robots.append(robot_info)
                
                return robots
            elif self.fallback_detector is not None:
                # Usar detector tradicional como fallback
                return self.fallback_detector.detect(image)
            else:
                return []
        except Exception as e:
            print(f"Erro na detecção de robôs: {e}")
            return []
    
    def _calculate_3d_position(self, center, width, height):
        """
        Calcula a posição 3D do robô a partir de suas coordenadas na imagem.
        
        Args:
            center: Coordenadas (x, y) do centro da caixa delimitadora
            width: Largura da caixa delimitadora
            height: Altura da caixa delimitadora
            
        Returns:
            Tupla (x, y, z) com a posição 3D do robô relativa ao robô (em metros)
            x: lateral (positivo à direita), y: vertical, z: frontal
        """
        if self.camera_info is None:
            return None
        
        try:
            # Extrair parâmetros intrínsecos da câmera
            fx = self.camera_info.k[0]  # Distância focal em x
            fy = self.camera_info.k[4]  # Distância focal em y
            cx = self.camera_info.k[2]  # Centro óptico x
            cy = self.camera_info.k[5]  # Centro óptico y
            
            # Calcular distância com base na largura ou altura aparente
            # Usando a largura para robôs, assume-se que é mais confiável
            distance_z = (self.robot_width * fx) / width
            
            # Calcular coordenadas 3D
            x_center, y_center = center
            x_3d = (x_center - cx) * distance_z / fx
            y_3d = (y_center - cy) * distance_z / fy
            
            return (x_3d, y_3d, distance_z)
        except Exception as e:
            print(f"Erro ao calcular posição 3D: {e}")
            return None
    
    def to_ros_messages(self, robots, frame_id):
        """
        Converte detecções de robôs para mensagens ROS.
        
        Args:
            robots: Lista de detecções de robôs
            frame_id: ID do frame para as mensagens
            
        Returns:
            PoseArray contendo as poses dos robôs
        """
        pose_array = PoseArray()
        pose_array.header.frame_id = frame_id
        
        for robot in robots:
            if 'position' in robot and robot['position'] is not None:
                pose = Pose()
                x, y, z = robot['position']
                pose.position.x = z  # Distância frontal
                pose.position.y = -x  # Distância lateral (invertida para ROS)
                pose.position.z = 0.0  # Assume-se que os robôs estão no solo
                
                # A orientação poderia ser inferida se houvesse informação de direção
                # Por enquanto, assume-se que não há orientação
                
                pose_array.poses.append(pose)
        
        return pose_array
    
    def draw_detections(self, image, robots):
        """
        Desenha as detecções de robôs na imagem para visualização.
        
        Args:
            image: Imagem original
            robots: Lista de detecções de robôs
            
        Returns:
            Imagem com visualizações
        """
        if not robots:
            return image
        
        # Criar cópia da imagem
        vis_image = image.copy()
        
        for robot in robots:
            # Extrair informações
            bbox = robot.get('bbox', [0, 0, 0, 0])
            confidence = robot.get('confidence', 0.0)
            position = robot.get('position', None)
            team = robot.get('team', 'unknown')
            
            # Obter cor com base no time
            color = self.team_colors.get(team, self.team_colors['unknown'])
            
            if bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                
                # Desenhar retângulo
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                
                # Texto de confiança
                team_text = f"{team.capitalize()}: {confidence:.2f}"
                cv2.putText(vis_image, team_text, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Adicionar informação de posição 3D se disponível
                if position is not None:
                    x_3d, y_3d, z_3d = position
                    pos_text = f"({x_3d:.2f}, {z_3d:.2f}m)"
                    cv2.putText(vis_image, pos_text, (x, y + h + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Adicionar contador de robôs
        cv2.putText(vis_image, f"Robôs: {len(robots)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image 