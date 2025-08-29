#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose2D
from roboime_msgs.msg import GoalDetection
import math

class SoccerBehavior(Node):
    """
    Nó de comportamento para o robô de futebol.
    
    Este nó implementa a máquina de estados para o comportamento do robô
    durante uma partida de futebol, incluindo:
    - Busca pela bola
    - Aproximação da bola
    - Posicionamento para chute
    - Chute
    - Retorno à posição defensiva
    """
    
    def __init__(self):
        super().__init__('soccer_behavior')
        
        # Parâmetros
        self.declare_parameter('team_color', 'blue')
        self.declare_parameter('player_number', 1)
        self.declare_parameter('role', 'striker')  # striker, defender, goalkeeper
        
        # Publishers
        self.motion_cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.state_pub = self.create_publisher(String, 'robot_state', 10)
        
        # Subscribers
        self.ball_sub = self.create_subscription(
            Pose2D,
            'ball_position',
            self.ball_callback,
            10
        )
        self.robot_pose_sub = self.create_subscription(
            Pose2D,
            'robot_pose',
            self.robot_pose_callback,
            10
        )
        self.goal_sub = self.create_subscription(
            GoalDetection,
            'goal_detections',
            self.goal_callback,
            10
        )
        
        # Estado inicial
        self.state = 'SEARCH_BALL'
        self.ball_position = None
        self.robot_pose = None
        self.goal_info = None  # Informações sobre gol detectado (centro, orientação, lado)
        
        # Timer para a máquina de estados
        self.timer = self.create_timer(0.1, self.state_machine)
        
        self.get_logger().info('Nó de comportamento de futebol iniciado')
    
    def ball_callback(self, msg):
        """Callback para a posição da bola."""
        self.ball_position = msg
    
    def robot_pose_callback(self, msg):
        """Callback para a pose do robô."""
        self.robot_pose = msg
    
    def goal_callback(self, msg: GoalDetection):
        """Callback para detecções de gol (postes e centro calculado)"""
        if msg.detected:
            goal_data = {
                'detected': True,
                'num_posts': msg.num_posts,
                'confidence': msg.confidence
            }
            
            # Se temos informação de gol completo (centro calculado a partir de pares)
            if hasattr(msg, 'position_3d') and msg.position_3d.x != 0.0:
                goal_data.update({
                    'center': (msg.position_3d.x, msg.position_3d.y),
                    'side': getattr(msg, 'goal_side', 'unknown'),
                    'distance': getattr(msg, 'distance', 0.0),
                    'bearing': getattr(msg, 'bearing', 0.0),
                    'width': getattr(msg, 'goal_width', 2.6),
                    'is_complete': True  # Par de postes identificado
                })
                self.get_logger().debug(f'🥅 Gol completo: centro=({goal_data["center"][0]:.2f}, {goal_data["center"][1]:.2f}), lado={goal_data["side"]}')
            else:
                goal_data.update({
                    'is_complete': False,  # Apenas postes individuais
                    'side': 'unknown'
                })
                self.get_logger().debug(f'🥅 Postes individuais detectados: {goal_data["num_posts"]}')
            
            self.goal_info = goal_data
    
    def state_machine(self):
        """Implementação da máquina de estados para o comportamento do robô."""
        if self.ball_position is None or self.robot_pose is None:
            # Não temos informações suficientes ainda
            return
        
        # Publicar o estado atual
        state_msg = String()
        state_msg.data = self.state
        self.state_pub.publish(state_msg)
        
        # Máquina de estados
        if self.state == 'SEARCH_BALL':
            self.search_ball()
        elif self.state == 'APPROACH_BALL':
            self.approach_ball()
        elif self.state == 'POSITION_TO_KICK':
            self.position_to_kick()
        elif self.state == 'KICK':
            self.kick()
        elif self.state == 'RETURN_TO_DEFENSE':
            self.return_to_defense()
    
    def search_ball(self):
        """Comportamento para buscar a bola."""
        # Verificar se a bola foi encontrada
        if self.ball_position is not None:
            self.get_logger().info('Bola encontrada, aproximando-se')
            self.state = 'APPROACH_BALL'
            return
        
        # Girar para procurar a bola
        cmd = Twist()
        cmd.angular.z = 0.5  # Girar no sentido anti-horário
        self.motion_cmd_pub.publish(cmd)
    
    def approach_ball(self):
        """Comportamento para se aproximar da bola."""
        # Calcular distância até a bola
        dx = self.ball_position.x - self.robot_pose.x
        dy = self.ball_position.y - self.robot_pose.y
        distance = (dx**2 + dy**2)**0.5
        
        if distance < 0.3:  # Distância em metros
            self.get_logger().info('Próximo à bola, posicionando para chutar')
            self.state = 'POSITION_TO_KICK'
            return
        
        # Mover em direção à bola
        cmd = Twist()
        cmd.linear.x = 0.3  # Velocidade linear
        cmd.angular.z = 0.5 * (dy / dx) if dx != 0 else 0.5 * (1 if dy > 0 else -1)
        self.motion_cmd_pub.publish(cmd)
    
    def position_to_kick(self):
        """Comportamento para se posicionar para chutar usando informações do gol."""
        # Usar informações do gol para melhor posicionamento se disponível
        if self.goal_info and self.goal_info.get('is_complete', False):
            # Temos informação completa do gol (par de postes)
            goal_center = self.goal_info['center']
            goal_side = self.goal_info['side']
            
            # Calcular ângulo ideal para o chute (da bola ao centro do gol)
            ball_to_goal_x = goal_center[0] - self.ball_position.x
            ball_to_goal_y = goal_center[1] - self.ball_position.y
            ideal_angle = math.atan2(ball_to_goal_y, ball_to_goal_x)
            
            self.get_logger().info(f'Posicionando para chutar no gol {goal_side}: centro=({goal_center[0]:.2f}, {goal_center[1]:.2f})')
            
            # Calcular posição ideal atrás da bola (oposta ao gol)
            offset_distance = 0.2  # 20cm atrás da bola
            target_x = self.ball_position.x - offset_distance * math.cos(ideal_angle)
            target_y = self.ball_position.y - offset_distance * math.sin(ideal_angle)
            
            # Verificar se já estamos na posição ideal
            dx = target_x - self.robot_pose.x
            dy = target_y - self.robot_pose.y
            position_error = math.sqrt(dx*dx + dy*dy)
            
            if position_error < 0.1:  # 10cm de tolerância
                self.get_logger().info(f'Posicionado estrategicamente para gol {goal_side}, chutando!')
                self.state = 'KICK'
            else:
                # Mover para posição ideal
                cmd = Twist()
                cmd.linear.x = min(0.2, position_error)
                if dx != 0:
                    cmd.angular.z = 0.3 * math.atan2(dy, dx)
                self.motion_cmd_pub.publish(cmd)
        else:
            # Fallback: posicionamento básico sem informação do gol
            self.get_logger().info('Posicionado para chutar (sem info do gol), chutando')
            self.state = 'KICK'
    
    def kick(self):
        """Comportamento para chutar a bola."""
        # Implementação simplificada
        cmd = Twist()
        cmd.linear.x = 1.0  # Velocidade máxima para frente
        self.motion_cmd_pub.publish(cmd)
        
        self.get_logger().info('Chute executado, retornando à defesa')
        self.state = 'RETURN_TO_DEFENSE'
    
    def return_to_defense(self):
        """Comportamento para retornar à posição defensiva."""
        # Implementação simplificada
        self.get_logger().info('Retornando à posição defensiva')
        self.state = 'SEARCH_BALL'

def main(args=None):
    rclpy.init(args=args)
    node = SoccerBehavior()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 