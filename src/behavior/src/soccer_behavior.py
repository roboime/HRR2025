#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose2D

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
        
        # Estado inicial
        self.state = 'SEARCH_BALL'
        self.ball_position = None
        self.robot_pose = None
        
        # Timer para a máquina de estados
        self.timer = self.create_timer(0.1, self.state_machine)
        
        self.get_logger().info('Nó de comportamento de futebol iniciado')
    
    def ball_callback(self, msg):
        """Callback para a posição da bola."""
        self.ball_position = msg
    
    def robot_pose_callback(self, msg):
        """Callback para a pose do robô."""
        self.robot_pose = msg
    
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
        """Comportamento para se posicionar para chutar."""
        # Implementação simplificada
        self.get_logger().info('Posicionado para chutar, chutando')
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