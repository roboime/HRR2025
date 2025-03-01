#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from geometry_msgs.msg import Pose, PoseArray
from src.perception.src.yoeo.yoeo_handler import YOEOHandler, DetectionType

class GoalDetectionComponent:
    """
    Componente para detecção de gols utilizando o YOEO.
    
    Este componente é responsável por detectar postes de gol na imagem,
    calcular suas posições 3D relativas ao robô, e fornecer visualizações
    para depuração.
    """
    
    def __init__(self, yoeo_handler, post_width=0.10, post_height=0.80):
        """
        Inicializa o componente de detecção de gols.
        
        Args:
            yoeo_handler: Instância do YOEOHandler para acesso ao modelo
            post_width: Largura do poste em metros
            post_height: Altura do poste em metros
        """
        self.yoeo_handler = yoeo_handler
        self.post_width = post_width
        self.post_height = post_height
        self.fallback_detector = None
        self.camera_info = None
    
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
        Processa a imagem para detectar postes de gol.
        
        Args:
            image: Imagem BGR do OpenCV
            
        Returns:
            Lista de detecções de postes de gol, cada uma contendo:
            {
                'bbox': [x, y, width, height],
                'confidence': confiança da detecção,
                'center': (x, y) centro da detecção,
                'position': (x, y, z) posição 3D relativa ao robô
            }
        """
        # Verificar se a imagem é válida
        if image is None or image.size == 0:
            return []
        
        # Tentar obter as detecções do YOEO
        try:
            # Obter detecções do YOEO
            detections = self.yoeo_handler.get_detections(image, DetectionType.GOAL)
            
            # Processar detecções se encontradas
            if detections and len(detections) > 0:
                goal_posts = []
                
                for detection in detections:
                    # Calcular centro da caixa delimitadora
                    x, y, w, h = detection.get('bbox', [0, 0, 0, 0])
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    # Calcular posição 3D
                    position = self._calculate_3d_position((center_x, center_y), w, h)
                    
                    # Adicionar informações de detecção
                    post_info = {
                        'bbox': [x, y, w, h],
                        'confidence': detection.get('confidence', 0.0),
                        'center': (int(center_x), int(center_y)),
                        'position': position
                    }
                    
                    goal_posts.append(post_info)
                
                return goal_posts
            elif self.fallback_detector is not None:
                # Usar detector tradicional como fallback
                return self.fallback_detector.detect(image)
            else:
                return []
        except Exception as e:
            print(f"Erro na detecção de gols: {e}")
            return []
    
    def _calculate_3d_position(self, center, width, height):
        """
        Calcula a posição 3D do poste de gol a partir de suas coordenadas na imagem.
        
        Args:
            center: Coordenadas (x, y) do centro da caixa delimitadora
            width: Largura da caixa delimitadora
            height: Altura da caixa delimitadora
            
        Returns:
            Tupla (x, y, z) com a posição 3D do poste relativa ao robô (em metros)
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
            # Usar a maior dimensão para melhor precisão
            if height > width:
                # Estimar distância da altura
                distance_z = (self.post_height * fy) / height
            else:
                # Estimar distância da largura
                distance_z = (self.post_width * fx) / width
            
            # Calcular coordenadas 3D
            x_center, y_center = center
            x_3d = (x_center - cx) * distance_z / fx
            y_3d = (y_center - cy) * distance_z / fy
            
            return (x_3d, y_3d, distance_z)
        except Exception as e:
            print(f"Erro ao calcular posição 3D: {e}")
            return None
    
    def to_ros_messages(self, goal_posts, frame_id):
        """
        Converte detecções de postes de gol para mensagens ROS.
        
        Args:
            goal_posts: Lista de detecções de postes de gol
            frame_id: ID do frame para as mensagens
            
        Returns:
            PoseArray contendo as poses dos postes de gol
        """
        pose_array = PoseArray()
        pose_array.header.frame_id = frame_id
        
        for post in goal_posts:
            if 'position' in post and post['position'] is not None:
                pose = Pose()
                x, y, z = post['position']
                pose.position.x = z  # Distância frontal
                pose.position.y = -x  # Distância lateral (invertida para ROS)
                pose.position.z = 0.0  # Assume-se que os postes estão no solo
                pose_array.poses.append(pose)
        
        return pose_array
    
    def draw_detections(self, image, goal_posts):
        """
        Desenha as detecções de postes de gol na imagem para visualização.
        
        Args:
            image: Imagem original
            goal_posts: Lista de detecções de postes de gol
            
        Returns:
            Imagem com visualizações
        """
        if not goal_posts:
            return image
        
        # Criar cópia da imagem
        vis_image = image.copy()
        
        # Cor para gols (amarelo)
        color = (0, 255, 255)
        
        for post in goal_posts:
            # Extrair informações
            bbox = post.get('bbox', [0, 0, 0, 0])
            confidence = post.get('confidence', 0.0)
            position = post.get('position', None)
            
            if bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                
                # Desenhar retângulo
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                
                # Texto de confiança
                conf_text = f"Gol: {confidence:.2f}"
                cv2.putText(vis_image, conf_text, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Adicionar informação de posição 3D se disponível
                if position is not None:
                    x_3d, y_3d, z_3d = position
                    pos_text = f"({x_3d:.2f}, {z_3d:.2f}m)"
                    cv2.putText(vis_image, pos_text, (x, y + h + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Adicionar contador de postes
        cv2.putText(vis_image, f"Postes: {len(goal_posts)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image 