#!/usr/bin/env python3
"""
Pipeline Avançado 3D para RoboIME HSL2025
Sistema integrado de percepção 3D para futebol robótico humanóide

Funcionalidades Avançadas:
- Fusão de múltiplas detecções temporais
- Tracking de objetos em movimento 
- Predição de trajetórias
- Validação cruzada usando múltiplas fontes
- Otimização para Jetson Orin Nano Super
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time
import threading
from geometry_msgs.msg import Point, Pose2D, Vector3
from .camera_geometry_3d import Object3D, CameraGeometry3D

@dataclass
class TrackedObject:
    """Objeto rastreado ao longo do tempo com informações 3D"""
    object_id: int
    object_type: str
    
    # Histórico de posições
    positions_3d: deque = field(default_factory=lambda: deque(maxlen=10))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=10))
    confidences: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Estado atual
    current_position: Optional[Point] = None
    current_velocity: Optional[Vector3] = None
    last_seen: float = 0.0
    
    # Predição
    predicted_position: Optional[Point] = None
    prediction_confidence: float = 0.0
    
    # Validação
    size_consistency: float = 1.0
    position_consistency: float = 1.0
    temporal_consistency: float = 1.0

@dataclass
class BallTrajectory:
    """Trajetória específica da bola com física"""
    positions: List[Point]
    timestamps: List[float]
    velocities: List[Vector3]
    
    # Parâmetros físicos
    gravity: float = 9.81
    air_resistance: float = 0.1
    
    # Predições
    predicted_landing_point: Optional[Point] = None
    predicted_landing_time: Optional[float] = None
    trajectory_confidence: float = 0.0

class Advanced3DPipeline:
    """
    Pipeline avançado para processamento 3D em tempo real
    
    Combina detecção, tracking, predição e validação para criar
    um sistema robusto de percepção 3D para futebol robótico.
    """
    
    def __init__(self, geometry_3d: CameraGeometry3D, max_objects: int = 20):
        """
        Inicializa pipeline 3D avançado
        
        Args:
            geometry_3d: Sistema de geometria 3D da câmera
            max_objects: Número máximo de objetos rastreados
        """
        self.geometry_3d = geometry_3d
        self.max_objects = max_objects
        
        # Tracking de objetos
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 1
        self.tracking_distance_threshold = 0.5  # metros
        
        # Tracking específico da bola
        self.ball_trajectory: Optional[BallTrajectory] = None
        self.ball_prediction_enabled = True
        
        # Validação temporal
        self.temporal_window = 2.0  # segundos
        self.min_detections_for_tracking = 3
        
        # Filtros
        self.position_filter_alpha = 0.7  # Suavização posicional
        self.velocity_filter_alpha = 0.5  # Suavização de velocidade
        
        # Cache para otimização
        self._last_processing_time = 0.0
        
        # Thread de predição assíncrona
        self.prediction_thread = None
        self.stop_prediction = False
        
    def process_detections_3d(self, objects_3d: List[Object3D], 
                            timestamp: float) -> Dict[str, any]:
        """
        Processa detecções 3D e atualiza tracking
        
        Args:
            objects_3d: Lista de objetos 3D detectados
            timestamp: Timestamp atual
            
        Returns:
            Dicionário com resultados do processamento
        """
        start_time = time.time()
        
        # 1. Associar detecções com objetos rastreados
        associations = self._associate_detections_to_tracks(objects_3d, timestamp)
        
        # 2. Atualizar objetos existentes
        self._update_existing_tracks(associations, timestamp)
        
        # 3. Criar novos tracks para detecções não associadas
        self._create_new_tracks(associations, objects_3d, timestamp)
        
        # 4. Remover tracks perdidos
        self._cleanup_lost_tracks(timestamp)
        
        # 5. Predizer posições futuras
        self._predict_future_positions(timestamp)
        
        # 6. Processar trajetória da bola especificamente
        ball_results = self._process_ball_trajectory(timestamp)
        
        # 7. Validar consistência temporal
        self._validate_temporal_consistency()
        
        processing_time = time.time() - start_time
        self._last_processing_time = processing_time
        
        return {
            'tracked_objects': self.tracked_objects,
            'ball_trajectory': ball_results,
            'processing_time': processing_time,
            'total_tracks': len(self.tracked_objects),
            'predictions': self._get_all_predictions()
        }
    
    def _associate_detections_to_tracks(self, objects_3d: List[Object3D], 
                                      timestamp: float) -> Dict[int, Object3D]:
        """
        Associa detecções atuais com objetos rastreados existentes
        
        Args:
            objects_3d: Lista de objetos detectados
            timestamp: Timestamp atual
            
        Returns:
            Dicionário mapeando track_id -> Object3D
        """
        associations = {}
        used_detections = set()
        
        for track_id, tracked_obj in self.tracked_objects.items():
            if not tracked_obj.current_position:
                continue
                
            best_match = None
            best_distance = float('inf')
            
            for i, obj_3d in enumerate(objects_3d):
                if i in used_detections:
                    continue
                    
                # Verificar compatibilidade de tipo
                if obj_3d.object_type != tracked_obj.object_type:
                    continue
                
                # Calcular distância 3D
                distance = np.sqrt(
                    (obj_3d.world_x - tracked_obj.current_position.x) ** 2 +
                    (obj_3d.world_y - tracked_obj.current_position.y) ** 2
                )
                
                # Aplicar predição se disponível
                if tracked_obj.predicted_position:
                    pred_distance = np.sqrt(
                        (obj_3d.world_x - tracked_obj.predicted_position.x) ** 2 +
                        (obj_3d.world_y - tracked_obj.predicted_position.y) ** 2
                    )
                    distance = min(distance, pred_distance * 0.8)  # Favorece predição
                
                if distance < self.tracking_distance_threshold and distance < best_distance:
                    best_match = i
                    best_distance = distance
            
            if best_match is not None:
                associations[track_id] = objects_3d[best_match]
                used_detections.add(best_match)
        
        return associations
    
    def _update_existing_tracks(self, associations: Dict[int, Object3D], timestamp: float):
        """Atualiza objetos rastreados com novas detecções"""
        
        for track_id, obj_3d in associations.items():
            tracked_obj = self.tracked_objects[track_id]
            
            # Criar ponto 3D
            new_position = Point()
            new_position.x = obj_3d.world_x
            new_position.y = obj_3d.world_y
            new_position.z = obj_3d.world_z
            
            # Adicionar ao histórico
            tracked_obj.positions_3d.append(new_position)
            tracked_obj.timestamps.append(timestamp)
            tracked_obj.confidences.append(obj_3d.detection_confidence * obj_3d.size_confidence)
            
            # Atualizar posição atual com filtro
            if tracked_obj.current_position:
                alpha = self.position_filter_alpha
                tracked_obj.current_position.x = (alpha * new_position.x + 
                                                (1 - alpha) * tracked_obj.current_position.x)
                tracked_obj.current_position.y = (alpha * new_position.y + 
                                                (1 - alpha) * tracked_obj.current_position.y)
            else:
                tracked_obj.current_position = new_position
            
            # Calcular velocidade
            tracked_obj.current_velocity = self._calculate_velocity(tracked_obj)
            
            # Atualizar timestamp
            tracked_obj.last_seen = timestamp
    
    def _create_new_tracks(self, associations: Dict[int, Object3D], 
                          objects_3d: List[Object3D], timestamp: float):
        """Cria novos tracks para detecções não associadas"""
        
        associated_indices = set()
        for obj_3d in associations.values():
            for i, detection in enumerate(objects_3d):
                if (detection.world_x == obj_3d.world_x and 
                    detection.world_y == obj_3d.world_y):
                    associated_indices.add(i)
                    break
        
        for i, obj_3d in enumerate(objects_3d):
            if i in associated_indices:
                continue
                
            # Verificar se já temos muitos objetos
            if len(self.tracked_objects) >= self.max_objects:
                continue
            
            # Criar novo track
            new_position = Point()
            new_position.x = obj_3d.world_x
            new_position.y = obj_3d.world_y
            new_position.z = obj_3d.world_z
            
            tracked_obj = TrackedObject(
                object_id=self.next_object_id,
                object_type=obj_3d.object_type,
                current_position=new_position,
                last_seen=timestamp
            )
            
            tracked_obj.positions_3d.append(new_position)
            tracked_obj.timestamps.append(timestamp)
            tracked_obj.confidences.append(obj_3d.detection_confidence * obj_3d.size_confidence)
            
            self.tracked_objects[self.next_object_id] = tracked_obj
            self.next_object_id += 1
    
    def _cleanup_lost_tracks(self, timestamp: float):
        """Remove tracks que não foram vistos recentemente"""
        
        lost_tracks = []
        for track_id, tracked_obj in self.tracked_objects.items():
            time_since_seen = timestamp - tracked_obj.last_seen
            
            # Timeout baseado no tipo de objeto
            timeout = 3.0 if tracked_obj.object_type == 'ball' else 5.0
            
            if time_since_seen > timeout:
                lost_tracks.append(track_id)
        
        for track_id in lost_tracks:
            del self.tracked_objects[track_id]
    
    def _calculate_velocity(self, tracked_obj: TrackedObject) -> Optional[Vector3]:
        """Calcula velocidade atual do objeto baseada no histórico"""
        
        if len(tracked_obj.positions_3d) < 2:
            return None
        
        # Usar últimas posições para calcular velocidade
        pos_current = tracked_obj.positions_3d[-1]
        pos_previous = tracked_obj.positions_3d[-2]
        time_current = tracked_obj.timestamps[-1]
        time_previous = tracked_obj.timestamps[-2]
        
        dt = time_current - time_previous
        if dt <= 0:
            return None
        
        velocity = Vector3()
        velocity.x = (pos_current.x - pos_previous.x) / dt
        velocity.y = (pos_current.y - pos_previous.y) / dt
        velocity.z = (pos_current.z - pos_previous.z) / dt
        
        # Aplicar filtro se já existe velocidade
        if tracked_obj.current_velocity:
            alpha = self.velocity_filter_alpha
            velocity.x = alpha * velocity.x + (1 - alpha) * tracked_obj.current_velocity.x
            velocity.y = alpha * velocity.y + (1 - alpha) * tracked_obj.current_velocity.y
            velocity.z = alpha * velocity.z + (1 - alpha) * tracked_obj.current_velocity.z
        
        return velocity
    
    def _predict_future_positions(self, timestamp: float):
        """Prediz posições futuras dos objetos rastreados"""
        
        prediction_time = 0.5  # Predizer 500ms no futuro
        
        for tracked_obj in self.tracked_objects.values():
            if not tracked_obj.current_position or not tracked_obj.current_velocity:
                continue
            
            # Predição linear simples
            predicted_pos = Point()
            predicted_pos.x = (tracked_obj.current_position.x + 
                             tracked_obj.current_velocity.x * prediction_time)
            predicted_pos.y = (tracked_obj.current_position.y + 
                             tracked_obj.current_velocity.y * prediction_time)
            predicted_pos.z = (tracked_obj.current_position.z + 
                             tracked_obj.current_velocity.z * prediction_time)
            
            # Para bola, aplicar física (gravidade)
            if tracked_obj.object_type == 'ball' and tracked_obj.current_velocity.z > 0:
                predicted_pos.z += 0.5 * (-9.81) * (prediction_time ** 2)
                predicted_pos.z = max(0.0, predicted_pos.z)  # Não pode ir abaixo do chão
            
            tracked_obj.predicted_position = predicted_pos
            tracked_obj.prediction_confidence = min(1.0, len(tracked_obj.positions_3d) / 5.0)
    
    def _process_ball_trajectory(self, timestamp: float) -> Dict:
        """Processa trajetória específica da bola"""
        
        ball_tracks = [obj for obj in self.tracked_objects.values() 
                      if obj.object_type == 'ball']
        
        if not ball_tracks:
            return {'ball_detected': False}
        
        # Usar bola com maior confiança
        best_ball = max(ball_tracks, key=lambda x: np.mean(list(x.confidences)) if x.confidences else 0)
        
        # Criar ou atualizar trajetória
        if self.ball_trajectory is None:
            self.ball_trajectory = BallTrajectory(
                positions=list(best_ball.positions_3d),
                timestamps=list(best_ball.timestamps),
                velocities=[]
            )
        
        # Calcular trajetória física se temos dados suficientes
        trajectory_analysis = {}
        if len(best_ball.positions_3d) >= 3:
            trajectory_analysis = self._analyze_ball_physics(best_ball)
        
        return {
            'ball_detected': True,
            'ball_position': best_ball.current_position,
            'ball_velocity': best_ball.current_velocity,
            'ball_prediction': best_ball.predicted_position,
            'trajectory_analysis': trajectory_analysis,
            'confidence': np.mean(list(best_ball.confidences)) if best_ball.confidences else 0.0
        }
    
    def _analyze_ball_physics(self, ball_track: TrackedObject) -> Dict:
        """Analisa física da trajetória da bola"""
        
        if len(ball_track.positions_3d) < 3:
            return {}
        
        positions = list(ball_track.positions_3d)
        timestamps = list(ball_track.timestamps)
        
        # Calcular aceleração
        accelerations = []
        for i in range(2, len(positions)):
            dt1 = timestamps[i-1] - timestamps[i-2] 
            dt2 = timestamps[i] - timestamps[i-1]
            
            if dt1 > 0 and dt2 > 0:
                v1_z = (positions[i-1].z - positions[i-2].z) / dt1
                v2_z = (positions[i].z - positions[i-1].z) / dt2
                acc_z = (v2_z - v1_z) / ((dt1 + dt2) / 2)
                accelerations.append(acc_z)
        
        # Detectar se bola está no ar
        is_airborne = any(pos.z > 0.05 for pos in positions[-3:])  # 5cm acima do chão
        
        # Estimar ponto de queda se está no ar
        landing_prediction = None
        if is_airborne and ball_track.current_velocity:
            landing_prediction = self._predict_ball_landing(ball_track)
        
        return {
            'is_airborne': is_airborne,
            'average_acceleration_z': np.mean(accelerations) if accelerations else 0.0,
            'estimated_landing': landing_prediction,
            'trajectory_consistency': self._calculate_trajectory_consistency(ball_track)
        }
    
    def _predict_ball_landing(self, ball_track: TrackedObject) -> Optional[Point]:
        """Prediz onde a bola vai cair usando física"""
        
        if not ball_track.current_position or not ball_track.current_velocity:
            return None
        
        pos = ball_track.current_position
        vel = ball_track.current_velocity
        
        # Resolver equação quadrática para tempo de queda
        # z = z0 + v_z * t - 0.5 * g * t^2
        # Quando z = 0: 0 = pos.z + vel.z * t - 0.5 * 9.81 * t^2
        
        g = 9.81
        a = -0.5 * g
        b = vel.z
        c = pos.z
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        
        t1 = (-b + np.sqrt(discriminant)) / (2*a)
        t2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Usar tempo positivo
        t_landing = max(t1, t2) if max(t1, t2) > 0 else None
        
        if t_landing is None or t_landing > 10.0:  # Máximo 10 segundos no futuro
            return None
        
        # Calcular posição de pouso
        landing_point = Point()
        landing_point.x = pos.x + vel.x * t_landing
        landing_point.y = pos.y + vel.y * t_landing
        landing_point.z = 0.0
        
        return landing_point
    
    def _calculate_trajectory_consistency(self, tracked_obj: TrackedObject) -> float:
        """Calcula consistência da trajetória (0.0 a 1.0)"""
        
        if len(tracked_obj.positions_3d) < 3:
            return 0.5
        
        positions = list(tracked_obj.positions_3d)
        
        # Calcular desvio da trajetória linear
        deviations = []
        for i in range(1, len(positions) - 1):
            # Vetor esperado (linear)
            expected = np.array([
                positions[i+1].x - positions[i-1].x,
                positions[i+1].y - positions[i-1].y
            ]) / 2.0
            
            # Vetor real
            actual = np.array([
                positions[i].x - positions[i-1].x,
                positions[i].y - positions[i-1].y
            ])
            
            # Calcular desvio
            if np.linalg.norm(expected) > 0:
                deviation = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                deviations.append(deviation)
        
        if not deviations:
            return 0.5
        
        # Converter desvio médio para confiança
        avg_deviation = np.mean(deviations)
        consistency = np.exp(-avg_deviation * 2.0)  # Decaimento exponencial
        
        return max(0.0, min(1.0, consistency))
    
    def _validate_temporal_consistency(self):
        """Valida consistência temporal de todos os tracks"""
        
        for tracked_obj in self.tracked_objects.values():
            # Calcular consistência de tamanho
            if len(tracked_obj.confidences) > 1:
                conf_std = np.std(list(tracked_obj.confidences))
                tracked_obj.size_consistency = max(0.0, 1.0 - conf_std)
            
            # Calcular consistência posicional
            tracked_obj.position_consistency = self._calculate_trajectory_consistency(tracked_obj)
            
            # Calcular consistência temporal (regularidade das detecções)
            if len(tracked_obj.timestamps) > 2:
                time_intervals = np.diff(list(tracked_obj.timestamps))
                interval_std = np.std(time_intervals) if len(time_intervals) > 1 else 0
                tracked_obj.temporal_consistency = max(0.0, 1.0 - interval_std)
    
    def _get_all_predictions(self) -> Dict:
        """Retorna todas as predições atuais"""
        
        predictions = {}
        for track_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.predicted_position:
                predictions[track_id] = {
                    'position': tracked_obj.predicted_position,
                    'confidence': tracked_obj.prediction_confidence,
                    'object_type': tracked_obj.object_type
                }
        
        return predictions
    
    def get_system_statistics(self) -> Dict:
        """Retorna estatísticas do sistema para monitoramento"""
        
        total_tracks = len(self.tracked_objects)
        active_tracks = sum(1 for obj in self.tracked_objects.values() 
                          if time.time() - obj.last_seen < 1.0)
        
        ball_tracks = sum(1 for obj in self.tracked_objects.values() 
                         if obj.object_type == 'ball')
        
        avg_confidence = 0.0
        if self.tracked_objects:
            confidences = []
            for obj in self.tracked_objects.values():
                if obj.confidences:
                    confidences.extend(list(obj.confidences))
            avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'total_tracks': total_tracks,
            'active_tracks': active_tracks,
            'ball_tracks': ball_tracks,
            'processing_time': self._last_processing_time,
            'average_confidence': avg_confidence,
            'ball_trajectory_active': self.ball_trajectory is not None
        } 