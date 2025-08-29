#!/usr/bin/env python3
"""
Coordenador Principal de Navegação - RoboIME HSL2025
Gerencia localização, planejamento e execução de movimento
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Twist
from std_msgs.msg import String, Bool, Float64
from nav_msgs.msg import Path
import math
import time
from enum import Enum
from typing import Optional

# Importar mensagens customizadas
from roboime_msgs.msg import FieldDetection, GoalDetection

class NavigationState(Enum):
    """Estados do sistema de navegação"""
    IDLE = "idle"
    LOCALIZING = "localizing"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    GOAL_REACHED = "goal_reached"
    ERROR = "error"

class NavigationManager(Node):
    """
    Coordenador principal do sistema de navegação integrado com YOLOv8 simplificado:
    - Gerencia estados da navegação
    - Interface com sistema de comportamento
    - Coordena localização baseada em landmarks (penalty_mark, goal_post, center_circle, field_corner, area_corner)
    - Supervisiona execução de movimento
    - Monitora confiança da localização baseada em landmarks detectados
    """
    
    def __init__(self):
        super().__init__('navigation_manager')
        
        # Parâmetros
        self.declare_parameter('localization_timeout', 5.0)
        self.declare_parameter('planning_timeout', 2.0)
        self.declare_parameter('navigation_timeout', 30.0)
        self.declare_parameter('enable_safety_checks', True)
        self.declare_parameter('min_localization_confidence', 0.3)
        self.declare_parameter('required_landmarks_for_navigation', 1)
        
        # Obter parâmetros
        self.localization_timeout = self.get_parameter('localization_timeout').value
        self.planning_timeout = self.get_parameter('planning_timeout').value
        self.navigation_timeout = self.get_parameter('navigation_timeout').value
        self.enable_safety = self.get_parameter('enable_safety_checks').value
        self.min_localization_confidence = self.get_parameter('min_localization_confidence').value
        self.required_landmarks_for_navigation = self.get_parameter('required_landmarks_for_navigation').value
        
        # Estado do sistema
        self.current_state = NavigationState.IDLE
        self.current_pose = None
        self.current_goal = None
        self.last_localization_time = 0
        self.navigation_start_time = 0
        self.localization_confidence = 0.0
        self.landmarks_detected = 0
        self.goals_detected = 0
        
        # Estatísticas do sistema simplificado
        self.total_navigation_requests = 0
        self.successful_navigations = 0
        self.landmark_based_localizations = 0
        
        # Publishers
        self.nav_goal_pub = self.create_publisher(Pose2D, 'navigation_goal', 10)
        self.nav_status_pub = self.create_publisher(String, 'navigation_status', 10)
        self.nav_cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, 'emergency_stop', 10)
        self.localization_confidence_pub = self.create_publisher(Float64, 'localization_confidence', 10)
        
        # Subscribers - Interface com comportamento
        self.goal_request_sub = self.create_subscription(
            Pose2D,
            'behavior/navigation_request',
            self.goal_request_callback,
            10
        )
        
        self.cancel_nav_sub = self.create_subscription(
            String,
            'behavior/cancel_navigation',
            self.cancel_navigation_callback,
            10
        )
        
        # Subscribers - Estado dos subsistemas
        self.pose_sub = self.create_subscription(
            Pose2D,
            'robot_pose',
            self.pose_callback,
            10
        )
        
        self.nav_cmd_sub = self.create_subscription(
            Twist,
            'nav_cmd_vel',
            self.nav_cmd_callback,
            10
        )
        
        self.path_sub = self.create_subscription(
            Path,
            'planned_path',
            self.path_callback,
            10
        )
        
        # Subscribers - Monitoramento da percepção simplificada
        self.landmarks_sub = self.create_subscription(
            FieldDetection,
            '/perception/localization_landmarks',
            self.landmarks_callback,
            10
        )
        
        self.goals_sub = self.create_subscription(
            GoalDetection,
            '/perception/goal_detections',
            self.goals_callback,
            10
        )
        
        # Timer para supervisão
        self.timer = self.create_timer(0.2, self.supervision_cycle)  # 5Hz
        
        self.get_logger().info('🎮 Coordenador de Navegação Simplificado iniciado')
        self.get_logger().info(f'⏱️ Timeouts: Loc={self.localization_timeout}s, Plan={self.planning_timeout}s, Nav={self.navigation_timeout}s')
        self.get_logger().info(f'🎯 Landmarks para localização: penalty_mark, goal_post, center_circle, field_corner, area_corner')
        self.get_logger().info(f'🔍 Confiança mínima: {self.min_localization_confidence}')
        self.get_logger().info(f'📍 Landmarks mínimos: {self.required_landmarks_for_navigation}')
    
    def goal_request_callback(self, msg: Pose2D):
        """Callback para requisição de navegação do comportamento"""
        self.get_logger().info(f'🎯 Nova requisição de navegação: ({msg.x:.2f}, {msg.y:.2f}, {math.degrees(msg.theta):.1f}°)')
        
        self.current_goal = msg
        self.navigation_start_time = time.time()
        self.total_navigation_requests += 1
        
        # Verificar se temos localização com confiança suficiente
        if self.current_pose is None:
            self.current_state = NavigationState.LOCALIZING
            self.get_logger().info('📍 Aguardando localização baseada em landmarks...')
        elif self.localization_confidence < self.min_localization_confidence:
            self.current_state = NavigationState.LOCALIZING
            self.get_logger().info(f'⚠️ Confiança baixa ({self.localization_confidence:.2f}), aguardando mais landmarks...')
        else:
            self.current_state = NavigationState.PLANNING
            self._send_goal_to_planner()
    
    def cancel_navigation_callback(self, msg: String):
        """Callback para cancelamento de navegação"""
        self.get_logger().info('❌ Navegação cancelada pelo comportamento')
        self._cancel_navigation()
    
    def pose_callback(self, msg: Pose2D):
        """Callback para pose atual do sistema de localização"""
        self.current_pose = msg
        self.last_localization_time = time.time()
        
        # Se estava aguardando localização, verificar se pode começar a planificar
        if self.current_state == NavigationState.LOCALIZING and self.current_goal is not None:
            if self.localization_confidence >= self.min_localization_confidence:
                self.current_state = NavigationState.PLANNING
                self._send_goal_to_planner()
                self.get_logger().info(f'✅ Localização confirmada com confiança {self.localization_confidence:.2f}')
    
    def landmarks_callback(self, msg: FieldDetection):
        """Callback para landmarks detectados pela percepção simplificada"""
        if msg.field_detected:
            self.landmarks_detected = msg.num_landmarks
            self.landmark_based_localizations += 1
            
            # Estimar confiança baseada no número de landmarks
            if self.landmarks_detected >= self.required_landmarks_for_navigation:
                self.localization_confidence = min(0.9, 0.3 + 0.1 * self.landmarks_detected)
            else:
                self.localization_confidence = max(0.1, self.localization_confidence * 0.95)  # Decair lentamente
            
            self.get_logger().debug(f'🎯 Landmarks detectados: {self.landmarks_detected}, Confiança: {self.localization_confidence:.2f}')
        else:
            # Reduzir confiança se não detecta landmarks
            self.localization_confidence = max(0.0, self.localization_confidence * 0.9)
    
    def goals_callback(self, msg: GoalDetection):
        """Callback para postes de gol detectados (contribuem para localização)"""
        if msg.detected:
            self.goals_detected = msg.num_posts
            
            # Processar informações enriquecidas de gol quando disponíveis
            if hasattr(msg, 'position_3d') and msg.position_3d.x != 0.0:
                # Goal completo computado a partir de pares de postes
                goal_center = (msg.position_3d.x, msg.position_3d.y)
                goal_side = msg.goal_side if hasattr(msg, 'goal_side') else "unknown"
                goal_confidence = msg.confidence
                
                self.get_logger().debug(
                    f'🥅 Gol completo detectado: centro=({goal_center[0]:.2f}, {goal_center[1]:.2f}), '
                    f'lado={goal_side}, conf={goal_confidence:.2f}, postes={self.goals_detected}'
                )
                
                # Aumentar confiança se temos gol completo (par de postes)
                confidence_boost = 0.15 if self.goals_detected >= 2 else 0.08
                self.localization_confidence = min(0.95, self.localization_confidence + confidence_boost)
            else:
                # Somente postes individuais
                self.get_logger().debug(f'🥅 Postes individuais detectados: {self.goals_detected}')
                self.localization_confidence = min(0.90, self.localization_confidence + 0.1)
                
            self.get_logger().debug(f'Confiança de localização: {self.localization_confidence:.2f}')
    
    def nav_cmd_callback(self, msg: Twist):
        """Callback para comandos de velocidade do planejador"""
        # Aplicar verificações de segurança aprimoradas
        if self.enable_safety and self._safety_check(msg):
            # Verificar confiança da localização antes de executar movimento
            if self.localization_confidence >= self.min_localization_confidence:
                self.nav_cmd_vel_pub.publish(msg)
            else:
                # Parar robô se confiança muito baixa
                stop_cmd = Twist()
                self.nav_cmd_vel_pub.publish(stop_cmd)
                self.get_logger().warn(f'⚠️ Movimento rejeitado: confiança baixa ({self.localization_confidence:.2f})')
        elif self.enable_safety:
            # Comando considerado inseguro, parar robô
            stop_cmd = Twist()
            self.nav_cmd_vel_pub.publish(stop_cmd)
            self.get_logger().warn('⚠️ Comando de velocidade rejeitado por segurança')
    
    def path_callback(self, msg: Path):
        """Callback para caminho planejado"""
        if self.current_state == NavigationState.PLANNING:
            self.current_state = NavigationState.NAVIGATING
            self.get_logger().info('🛣️ Caminho recebido, iniciando navegação baseada em landmarks')
    
    def supervision_cycle(self):
        """Ciclo de supervisão do sistema aprimorado"""
        current_time = time.time()
        
        # Publicar status atual
        self._publish_status()
        
        # Publicar confiança da localização
        confidence_msg = Float64()
        confidence_msg.data = self.localization_confidence
        self.localization_confidence_pub.publish(confidence_msg)
        
        # Verificações de timeout e confiança
        self._check_timeouts(current_time)
        self._check_localization_quality()
        
        # Verificação de chegada ao objetivo
        if self.current_state == NavigationState.NAVIGATING:
            self._check_goal_reached()
        
        # Log periódico de estatísticas do sistema simplificado
        if int(current_time) % 30 == 0:  # A cada 30 segundos
            self._log_system_statistics()
    
    def _check_localization_quality(self):
        """Verifica qualidade da localização baseada em landmarks"""
        if self.current_state == NavigationState.NAVIGATING:
            if self.localization_confidence < self.min_localization_confidence:
                self.get_logger().warn(f'⚠️ Localização degradada (confiança: {self.localization_confidence:.2f})')
                if self.localization_confidence < 0.1:
                    self.current_state = NavigationState.LOCALIZING
                    self.get_logger().warn('🔄 Parando navegação para relocalização')
                    self._stop_robot()
    
    def _check_goal_reached(self):
        """Verifica se chegou ao objetivo"""
        if self.current_pose and self.current_goal:
            distance = math.sqrt(
                (self.current_pose.x - self.current_goal.x)**2 + 
                (self.current_pose.y - self.current_goal.y)**2
            )
            
            angle_diff = abs(self.current_pose.theta - self.current_goal.theta)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # Menor diferença angular
            
            if distance < 0.2 and angle_diff < 0.2:  # 20cm e ~11 graus
                self.current_state = NavigationState.GOAL_REACHED
                self.successful_navigations += 1
                self._stop_robot()
                self.get_logger().info('🎯 Objetivo alcançado!')
    
    def _log_system_statistics(self):
        """Log de estatísticas do sistema simplificado"""
        if self.total_navigation_requests > 0:
            success_rate = (self.successful_navigations / self.total_navigation_requests) * 100
            
            self.get_logger().info(
                f'📊 Estatísticas Sistema Simplificado:\n'
                f'  🎯 Navegações: {self.successful_navigations}/{self.total_navigation_requests} ({success_rate:.1f}%)\n'
                f'  📍 Landmarks detectados: {self.landmarks_detected}\n'
                f'  🥅 Gols detectados: {self.goals_detected}\n'
                f'  🔍 Confiança atual: {self.localization_confidence:.2f}\n'
                f'  📡 Localizações por landmarks: {self.landmark_based_localizations}'
            )
    
    def _check_timeouts(self, current_time: float):
        """Verificações de timeout"""
        # Timeout de localização
        if (self.current_state == NavigationState.LOCALIZING and 
            self.navigation_start_time > 0 and 
            current_time - self.navigation_start_time > self.localization_timeout):
            
            self.get_logger().error(f'❌ Timeout de localização ({self.localization_timeout}s)')
            self.current_state = NavigationState.ERROR
            return
        
        # Timeout de planejamento
        if (self.current_state == NavigationState.PLANNING and 
            self.navigation_start_time > 0 and 
            current_time - self.navigation_start_time > self.planning_timeout):
            
            self.get_logger().error(f'❌ Timeout de planejamento ({self.planning_timeout}s)')
            self.current_state = NavigationState.ERROR
            return
        
        # Timeout de navegação
        if (self.current_state == NavigationState.NAVIGATING and 
            self.navigation_start_time > 0 and 
            current_time - self.navigation_start_time > self.navigation_timeout):
            
            self.get_logger().error(f'❌ Timeout de navegação ({self.navigation_timeout}s)')
            self.current_state = NavigationState.ERROR
            self._stop_robot()
            return
        
        # Timeout de dados de localização
        if (self.last_localization_time > 0 and 
            current_time - self.last_localization_time > self.localization_timeout):
            
            self.get_logger().warn('⚠️ Sem dados de localização há muito tempo')
            self.localization_confidence = 0.0
    
    def _publish_status(self):
        """Publica status atual do sistema"""
        status_msg = String()
        status_msg.data = f'{self.current_state.value}|confidence:{self.localization_confidence:.2f}|landmarks:{self.landmarks_detected}'
        self.nav_status_pub.publish(status_msg)
    
    def _send_goal_to_planner(self):
        """Envia objetivo para o planejador de trajetória"""
        if self.current_goal:
            self.nav_goal_pub.publish(self.current_goal)
            self.get_logger().info(f'📤 Objetivo enviado ao planejador: ({self.current_goal.x:.2f}, {self.current_goal.y:.2f})')
    
    def _cancel_navigation(self):
        """Cancela navegação atual"""
        self.current_state = NavigationState.IDLE
        self.current_goal = None
        self._stop_robot()
    
    def _stop_robot(self):
        """Para o robô"""
        stop_cmd = Twist()
        self.nav_cmd_vel_pub.publish(stop_cmd)
    
    def _safety_check(self, cmd_vel: Twist) -> bool:
        """Verificações de segurança aprimoradas"""
        # Verificar velocidades máximas
        max_linear = 0.5  # m/s
        max_angular = 1.5  # rad/s
        
        if abs(cmd_vel.linear.x) > max_linear:
            self.get_logger().warn(f'Velocidade linear muito alta: {cmd_vel.linear.x:.2f} > {max_linear}')
            return False
        
        if abs(cmd_vel.angular.z) > max_angular:
            self.get_logger().warn(f'Velocidade angular muito alta: {cmd_vel.angular.z:.2f} > {max_angular}')
            return False
        
        # Verificar se robô está próximo às bordas do campo
        if self.current_pose:
            margin = 0.3  # 30cm de margem
            if (abs(self.current_pose.x) > (4.5 - margin) or 
                abs(self.current_pose.y) > (3.0 - margin)):
                
                # Se movimento é em direção à borda, rejeitar
                if ((self.current_pose.x > 0 and cmd_vel.linear.x > 0) or
                    (self.current_pose.x < 0 and cmd_vel.linear.x < 0) or
                    (self.current_pose.y > 0 and cmd_vel.linear.y > 0) or
                    (self.current_pose.y < 0 and cmd_vel.linear.y < 0)):
                    
                    self.get_logger().warn('Movimento em direção à borda do campo rejeitado')
                    return False
        
        return True

def main(args=None):
    rclpy.init(args=args)
    node = NavigationManager()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 