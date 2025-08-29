#!/usr/bin/env python3
"""
Nó de Planejamento de Caminho - RoboIME HSL2025
Integrado com algoritmos C++ de localização para navegação inteligente
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import String, Float64
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Empty
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray

import heapq
import math
import time
from enum import Enum
from typing import List, Tuple, Optional, Dict

# TODO: Adicionar quando roboime_msgs estiver disponível
# from roboime_msgs.srv import PlanPath
# from roboime_msgs.msg import LandmarkArray


class PlannerState(Enum):
    """Estados do planejador de caminho"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    GOAL_REACHED = "goal_reached"
    STUCK = "stuck"
    ERROR = "error"


class PathPlannerNode(Node):
    """
    Planejador de Caminho Híbrido A* + Campos Potenciais
    
    Características:
    - Integração com localização C++ (Particle Filter + EKF)
    - Planejamento global com A*
    - Navegação local com campos potenciais
    - Evitamento de obstáculos dinâmicos
    - Replanejamento adaptativo
    """
    
    def __init__(self):
        super().__init__('path_planner_node')
        
        # Parâmetros
        self.declare_parameter('field_length', 9.0)
        self.declare_parameter('field_width', 6.0)
        self.declare_parameter('planning_frequency', 5.0)
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('goal_tolerance', 0.2)
        self.declare_parameter('obstacle_radius', 0.3)
        self.declare_parameter('enable_visualization', True)
        
        # Obter parâmetros
        self.field_length = self.get_parameter('field_length').value
        self.field_width = self.get_parameter('field_width').value
        self.planning_freq = self.get_parameter('planning_frequency').value
        self.control_freq = self.get_parameter('control_frequency').value
        self.max_vel = self.get_parameter('max_linear_velocity').value
        self.max_ang_vel = self.get_parameter('max_angular_velocity').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.obstacle_radius = self.get_parameter('obstacle_radius').value
        
        # Estado do planejador
        self.current_state = PlannerState.IDLE
        self.current_pose = Pose2D()
        self.target_pose = None
        self.current_path = []
        self.obstacles = []
        self.localization_confidence = 0.0
        
        # Mapa de grades para A*
        self.grid_resolution = 0.1  # 10cm por célula
        self.grid_width = int(self.field_width / self.grid_resolution)
        self.grid_height = int(self.field_length / self.grid_resolution)
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width))
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.status_pub = self.create_publisher(String, 'planner_status', 10)
        
        if self.get_parameter('enable_visualization').value:
            self.markers_pub = self.create_publisher(MarkerArray, 'path_markers', 10)
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            Pose2D, 'localization/pose', self.pose_callback, 10
        )
        
        self.confidence_sub = self.create_subscription(
            Float64, 'localization/confidence', self.confidence_callback, 10
        )
        
        self.goal_sub = self.create_subscription(
            PoseStamped, 'move_base_simple/goal', self.goal_callback, 10
        )
        
        self.obstacles_sub = self.create_subscription(
            PointCloud2, 'obstacles', self.obstacles_callback, 10
        )
        
        # Serviços
        # TODO: Implementar quando roboime_msgs estiver disponível
        # self.plan_service = self.create_service(
        #     roboime_msgs.srv.PlanPath, 'plan_path', self.plan_path_callback
        # )
        
        self.stop_service = self.create_service(
            Empty, 'stop_navigation', self.stop_navigation_callback
        )
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Timers
        self.planning_timer = self.create_timer(
            1.0 / self.planning_freq, self.planning_cycle
        )
        
        self.control_timer = self.create_timer(
            1.0 / self.control_freq, self.control_cycle
        )
        
        # Campos potenciais - parâmetros
        self.attractive_gain = 1.0
        self.repulsive_gain = 2.0
        self.influence_distance = 1.0
        
        # Histórico para detecção de oscilação
        self.velocity_history = []
        self.position_history = []
        
        self.get_logger().info(
            f"Path Planner Node iniciado - Campo: {self.field_length}x{self.field_width}m"
        )
    
    # =============================================================================
    # CALLBACKS
    # =============================================================================
    
    def pose_callback(self, msg: Pose2D):
        """Callback da pose atual do robô (do sistema de localização C++)"""
        self.current_pose = msg
        
        # Adicionar ao histórico
        self.position_history.append((msg.x, msg.y, time.time()))
        if len(self.position_history) > 50:
            self.position_history.pop(0)
    
    def confidence_callback(self, msg: Float64):
        """Callback da confiança da localização"""
        self.localization_confidence = msg.data
        
        # Se confiança muito baixa, parar navegação
        if self.localization_confidence < 0.3 and self.current_state == PlannerState.EXECUTING:
            self.get_logger().warn("Confiança baixa na localização! Parando navegação.")
            self.current_state = PlannerState.ERROR
            self.stop_robot()
    
    def goal_callback(self, msg: PoseStamped):
        """Callback para novo objetivo de navegação"""
        # Converter para Pose2D
        goal_2d = Pose2D()
        goal_2d.x = msg.pose.position.x
        goal_2d.y = msg.pose.position.y
        
        # Extrair yaw do quaternion
        orientation = msg.pose.orientation
        goal_2d.theta = 2 * math.atan2(orientation.z, orientation.w)
        
        self.set_navigation_goal(goal_2d)
    
    def obstacles_callback(self, msg: PointCloud2):
        """Callback para obstáculos detectados"""
        # TODO: Converter PointCloud2 para lista de obstáculos
        # Por enquanto, usar implementação simplificada
        pass
    
    # =============================================================================
    # SERVIÇOS
    # =============================================================================
    
    def plan_path_callback(self, request, response):
        """Serviço para planejamento de caminho"""
        try:
            start_pose = self.current_pose
            goal_pose = request.goal
            
            path = self.plan_path_astar(start_pose, goal_pose)
            
            if path:
                response.success = True
                response.path = self.convert_to_ros_path(path)
                response.message = "Caminho planejado com sucesso"
            else:
                response.success = False
                response.message = "Falha no planejamento de caminho"
                
        except Exception as e:
            response.success = False
            response.message = f"Erro no planejamento: {str(e)}"
            
        return response
    
    def stop_navigation_callback(self, request, response):
        """Serviço para parar navegação"""
        self.current_state = PlannerState.IDLE
        self.target_pose = None
        self.current_path = []
        self.stop_robot()
        
        self.get_logger().info("Navegação parada por solicitação")
        return response
    
    # =============================================================================
    # CICLOS PRINCIPAIS
    # =============================================================================
    
    def planning_cycle(self):
        """Ciclo de planejamento (5Hz)"""
        if self.current_state == PlannerState.PLANNING:
            self.execute_planning()
        elif self.current_state == PlannerState.EXECUTING:
            self.check_replanning_conditions()
    
    def control_cycle(self):
        """Ciclo de controle (20Hz)"""
        if self.current_state == PlannerState.EXECUTING:
            self.execute_local_navigation()
        
        # Publicar status
        self.publish_status()
    
    # =============================================================================
    # PLANEJAMENTO GLOBAL (A*)
    # =============================================================================
    
    def plan_path_astar(self, start: Pose2D, goal: Pose2D) -> List[Tuple[float, float]]:
        """
        Planejamento global usando algoritmo A*
        """
        # Converter poses para coordenadas de grade
        start_grid = self.world_to_grid(start.x, start.y)
        goal_grid = self.world_to_grid(goal.x, goal.y)
        
        if not self.is_valid_grid_point(*start_grid) or not self.is_valid_grid_point(*goal_grid):
            self.get_logger().error("Pontos de início ou fim fora dos limites do campo")
            return []
        
        # Inicializar A*
        open_set = [(0, start_grid, [])]
        closed_set = set()
        
        while open_set:
            current_cost, current_pos, path = heapq.heappop(open_set)
            
            if current_pos in closed_set:
                continue
                
            closed_set.add(current_pos)
            path = path + [current_pos]
            
            # Verificar se chegou ao objetivo
            if self.grid_distance(current_pos, goal_grid) < 2:  # Tolerância de 2 células
                # Converter caminho de volta para coordenadas do mundo
                world_path = [self.grid_to_world(*pos) for pos in path]
                return self.smooth_path(world_path)
            
            # Explorar vizinhos
            for neighbor in self.get_neighbors(*current_pos):
                if neighbor in closed_set or not self.is_valid_grid_point(*neighbor):
                    continue
                
                if self.occupancy_grid[neighbor[0]][neighbor[1]] > 0.5:  # Obstáculo
                    continue
                
                # Calcular custos
                move_cost = self.grid_distance(current_pos, neighbor)
                g_cost = current_cost + move_cost
                h_cost = self.grid_distance(neighbor, goal_grid)
                f_cost = g_cost + h_cost
                
                heapq.heappush(open_set, (f_cost, neighbor, path))
        
        self.get_logger().warn("A* não encontrou caminho válido")
        return []
    
    def smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Suavizar caminho removendo pontos desnecessários"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            
            # Encontrar o ponto mais distante visível
            while j > i + 1:
                if self.is_line_clear(path[i], path[j]):
                    break
                j -= 1
            
            smoothed.append(path[j])
            i = j
        
        return smoothed
    
    def is_line_clear(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """Verificar se linha entre dois pontos está livre de obstáculos"""
        x0, y0 = self.world_to_grid(*start)
        x1, y1 = self.world_to_grid(*end)
        
        # Algoritmo de Bresenham
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        
        while True:
            if not self.is_valid_grid_point(x, y):
                return False
                
            if self.occupancy_grid[x][y] > 0.5:
                return False
            
            if x == x1 and y == y1:
                break
            
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        
        return True
    
    # =============================================================================
    # NAVEGAÇÃO LOCAL (CAMPOS POTENCIAIS)
    # =============================================================================
    
    def execute_local_navigation(self):
        """Executa navegação local usando campos potenciais"""
        if not self.target_pose or not self.current_path:
            self.stop_robot()
            return
        
        # Calcular próximo waypoint
        next_waypoint = self.get_next_waypoint()
        if not next_waypoint:
            self.current_state = PlannerState.GOAL_REACHED
            self.stop_robot()
            return
        
        # Calcular forças dos campos potenciais
        attractive_force = self.calculate_attractive_force(next_waypoint)
        repulsive_force = self.calculate_repulsive_force()
        
        # Força total
        total_force = (
            attractive_force[0] + repulsive_force[0],
            attractive_force[1] + repulsive_force[1]
        )
        
        # Converter força em comando de velocidade
        cmd_vel = self.force_to_velocity(total_force)
        
        # Adicionar ao histórico
        self.velocity_history.append((cmd_vel.linear.x, cmd_vel.angular.z, time.time()))
        if len(self.velocity_history) > 20:
            self.velocity_history.pop(0)
        
        # Detectar oscilação/empacamento
        if self.detect_oscillation():
            self.handle_stuck_situation()
            return
        
        # Publicar comando
        self.cmd_vel_pub.publish(cmd_vel)
    
    def calculate_attractive_force(self, target: Tuple[float, float]) -> Tuple[float, float]:
        """Calcular força atrativa em direção ao objetivo"""
        dx = target[0] - self.current_pose.x
        dy = target[1] - self.current_pose.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 0.01:
            return (0.0, 0.0)
        
        # Força proporcional à distância (até um limite)
        force_magnitude = min(self.attractive_gain * distance, self.max_vel)
        
        return (
            force_magnitude * dx / distance,
            force_magnitude * dy / distance
        )
    
    def calculate_repulsive_force(self) -> Tuple[float, float]:
        """Calcular força repulsiva dos obstáculos"""
        total_force_x = 0.0
        total_force_y = 0.0
        
        for obstacle in self.obstacles:
            dx = self.current_pose.x - obstacle[0]
            dy = self.current_pose.y - obstacle[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < self.influence_distance and distance > 0.01:
                # Força repulsiva inversamente proporcional à distância
                force_magnitude = self.repulsive_gain * (
                    1.0/distance - 1.0/self.influence_distance
                ) / (distance * distance)
                
                total_force_x += force_magnitude * dx / distance
                total_force_y += force_magnitude * dy / distance
        
        return (total_force_x, total_force_y)
    
    def force_to_velocity(self, force: Tuple[float, float]) -> Twist:
        """Converter força em comando de velocidade"""
        cmd = Twist()
        
        # Velocidade linear baseada na magnitude da força
        force_magnitude = math.sqrt(force[0]*force[0] + force[1]*force[1])
        cmd.linear.x = min(force_magnitude, self.max_vel)
        
        # Velocidade angular para alinhar com a direção da força
        desired_heading = math.atan2(force[1], force[0])
        heading_error = self.normalize_angle(desired_heading - self.current_pose.theta)
        
        # Controle proporcional para orientação
        cmd.angular.z = min(max(2.0 * heading_error, -self.max_ang_vel), self.max_ang_vel)
        
        return cmd
    
    # =============================================================================
    # MÉTODOS AUXILIARES
    # =============================================================================
    
    def set_navigation_goal(self, goal: Pose2D):
        """Define novo objetivo de navegação"""
        self.target_pose = goal
        self.current_state = PlannerState.PLANNING
        
        self.get_logger().info(
            f"Novo objetivo: ({goal.x:.2f}, {goal.y:.2f}, {goal.theta:.2f})"
        )
    
    def execute_planning(self):
        """Executa planejamento de caminho"""
        if not self.target_pose:
            return
        
        path = self.plan_path_astar(self.current_pose, self.target_pose)
        
        if path:
            self.current_path = path
            self.current_state = PlannerState.EXECUTING
            self.publish_path_visualization()
            self.get_logger().info(f"Caminho planejado com {len(path)} waypoints")
        else:
            self.current_state = PlannerState.ERROR
            self.get_logger().error("Falha no planejamento de caminho")
    
    def get_next_waypoint(self) -> Optional[Tuple[float, float]]:
        """Obter próximo waypoint do caminho"""
        if not self.current_path:
            return None
        
        # Encontrar primeiro waypoint à frente
        for i, waypoint in enumerate(self.current_path):
            dist = math.sqrt(
                (waypoint[0] - self.current_pose.x)**2 +
                (waypoint[1] - self.current_pose.y)**2
            )
            
            if dist > self.goal_tolerance:
                # Remover waypoints já passados
                self.current_path = self.current_path[i:]
                return waypoint
        
        # Se chegou aqui, está próximo do último waypoint
        if self.current_path:
            last_point = self.current_path[-1]
            dist_to_goal = math.sqrt(
                (last_point[0] - self.current_pose.x)**2 +
                (last_point[1] - self.current_pose.y)**2
            )
            
            if dist_to_goal < self.goal_tolerance:
                return None  # Chegou ao objetivo
            else:
                return last_point
        
        return None
    
    def check_replanning_conditions(self):
        """Verificar se precisa replanejar"""
        # Replanejar se detectar novos obstáculos significativos
        # Replanejar se desvio do caminho for muito grande
        # Por enquanto, implementação simplificada
        pass
    
    def detect_oscillation(self) -> bool:
        """Detectar se robô está oscilando/empacado"""
        if len(self.velocity_history) < 10:
            return False
        
        # Verificar se velocidades estão alternando sinais
        recent_velocities = [v[0] for v in self.velocity_history[-10:]]
        sign_changes = sum(
            1 for i in range(1, len(recent_velocities))
            if recent_velocities[i] * recent_velocities[i-1] < 0
        )
        
        return sign_changes > 5  # Muitas mudanças de sinal
    
    def handle_stuck_situation(self):
        """Lidar com situação de empacamento"""
        self.get_logger().warn("Robô empacado detectado! Tentando recuperação...")
        
        # Parar por um momento
        self.stop_robot()
        
        # Marcar como empacado para replanejar
        self.current_state = PlannerState.STUCK
        
        # TODO: Implementar estratégias de recuperação mais sofisticadas
    
    def stop_robot(self):
        """Parar o robô"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
    
    def publish_status(self):
        """Publicar status atual"""
        status_msg = String()
        status_msg.data = self.current_state.value
        self.status_pub.publish(status_msg)
    
    def publish_path_visualization(self):
        """Publicar visualização do caminho planejado"""
        if not self.get_parameter('enable_visualization').value:
            return
        
        # Criar Path msg
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        for point in self.current_path:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = point[0]
            pose_stamped.pose.position.y = point[1]
            pose_stamped.pose.position.z = 0.0
            path_msg.poses.append(pose_stamped)
        
        self.path_pub.publish(path_msg)
    
    # =============================================================================
    # UTILITÁRIOS DE GRADE
    # =============================================================================
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Converter coordenadas do mundo para grade"""
        grid_x = int((x + self.field_length/2) / self.grid_resolution)
        grid_y = int((y + self.field_width/2) / self.grid_resolution)
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Converter coordenadas da grade para mundo"""
        x = grid_x * self.grid_resolution - self.field_length/2
        y = grid_y * self.grid_resolution - self.field_width/2
        return (x, y)
    
    def is_valid_grid_point(self, x: int, y: int) -> bool:
        """Verificar se ponto da grade é válido"""
        return 0 <= x < self.grid_height and 0 <= y < self.grid_width
    
    def grid_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calcular distância euclidiana entre pontos da grade"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Obter vizinhos de um ponto da grade (8-conectividade)"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.is_valid_grid_point(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors
    
    def normalize_angle(self, angle: float) -> float:
        """Normalizar ângulo para [-π, π]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def convert_to_ros_path(self, path: List[Tuple[float, float]]) -> Path:
        """Converter caminho interno para Path ROS"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        for point in path:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = point[0]
            pose_stamped.pose.position.y = point[1]
            pose_stamped.pose.position.z = 0.0
            path_msg.poses.append(pose_stamped)
        
        return path_msg


def main(args=None):
    rclpy.init(args=args)
    
    path_planner = PathPlannerNode()
    
    try:
        rclpy.spin(path_planner)
    except KeyboardInterrupt:
        pass
    finally:
        path_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 