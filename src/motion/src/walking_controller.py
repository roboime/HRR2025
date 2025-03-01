#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import numpy as np
import math

class WalkingController(Node):
    """
    Nó para controle de caminhada do robô humanóide.
    
    Este nó implementa um controlador de caminhada baseado em padrão central de geração (CPG)
    para gerar movimentos de caminhada suaves e estáveis para o robô humanóide.
    """
    
    def __init__(self):
        super().__init__('walking_controller')
        
        # Parâmetros
        self.declare_parameter('step_height', 0.04)  # Altura do passo em metros
        self.declare_parameter('step_length', 0.06)  # Comprimento do passo em metros
        self.declare_parameter('lateral_width', 0.05)  # Largura lateral do passo em metros
        self.declare_parameter('walk_frequency', 1.0)  # Frequência da caminhada em Hz
        self.declare_parameter('max_linear_velocity', 0.2)  # Velocidade linear máxima em m/s
        self.declare_parameter('max_angular_velocity', 0.5)  # Velocidade angular máxima em rad/s
        
        # Obter parâmetros
        self.step_height = self.get_parameter('step_height').value
        self.step_length = self.get_parameter('step_length').value
        self.lateral_width = self.get_parameter('lateral_width').value
        self.walk_frequency = self.get_parameter('walk_frequency').value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        
        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Variáveis
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.phase = 0.0
        self.is_walking = False
        
        # Definir juntas do robô
        self.joint_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll'
        ]
        
        # Timer para o ciclo de caminhada
        self.timer = self.create_timer(0.01, self.walking_cycle)
        
        self.get_logger().info('Nó controlador de caminhada iniciado')
    
    def cmd_vel_callback(self, msg):
        """Callback para comandos de velocidade."""
        # Limitar velocidades aos valores máximos
        self.linear_velocity = max(min(msg.linear.x, self.max_linear_velocity), -self.max_linear_velocity)
        self.angular_velocity = max(min(msg.angular.z, self.max_angular_velocity), -self.max_angular_velocity)
        
        # Verificar se o robô deve estar caminhando
        self.is_walking = abs(self.linear_velocity) > 0.01 or abs(self.angular_velocity) > 0.01
        
        self.get_logger().debug(f'Comando de velocidade: linear={self.linear_velocity}, angular={self.angular_velocity}')
    
    def walking_cycle(self):
        """Implementação do ciclo de caminhada."""
        if not self.is_walking:
            # Se não estiver caminhando, manter posição padrão
            self.publish_standing_pose()
            return
        
        # Atualizar fase do ciclo de caminhada
        self.phase += 2 * math.pi * self.walk_frequency * 0.01  # 0.01 é o período do timer
        if self.phase >= 2 * math.pi:
            self.phase -= 2 * math.pi
        
        # Calcular posições das juntas para o ciclo de caminhada
        joint_positions = self.calculate_walking_pattern()
        
        # Publicar comandos para as juntas
        self.publish_joint_commands(joint_positions)
    
    def calculate_walking_pattern(self):
        """
        Calcula o padrão de caminhada baseado em CPG.
        
        Returns:
            list: Posições das juntas para o ciclo de caminhada atual
        """
        # Implementação simplificada de um padrão de caminhada
        # Em um robô real, isso seria muito mais complexo
        
        # Ajustar parâmetros de caminhada com base na velocidade
        step_length = self.step_length * (self.linear_velocity / self.max_linear_velocity)
        lateral_width = self.lateral_width
        step_height = self.step_height
        
        # Ajustar rotação com base na velocidade angular
        rotation = self.angular_velocity / self.max_angular_velocity
        
        # Calcular fase para cada perna (defasadas em 180 graus)
        left_phase = self.phase
        right_phase = self.phase + math.pi
        
        # Calcular trajetória das pernas
        left_x = step_length * math.sin(left_phase)
        left_z = step_height * max(0, math.sin(left_phase))  # Só levanta o pé na fase positiva
        
        right_x = step_length * math.sin(right_phase)
        right_z = step_height * max(0, math.sin(right_phase))
        
        # Calcular posições laterais (balanço)
        lateral_shift = lateral_width * math.sin(self.phase)
        
        # Calcular ângulos das juntas usando cinemática inversa simplificada
        # Nota: Esta é uma implementação muito simplificada
        # Em um robô real, seria necessário um modelo cinemático completo
        
        # Valores para a perna esquerda
        left_hip_yaw = 0.0
        left_hip_roll = 0.1 + 0.05 * lateral_shift  # Inclinação lateral
        left_hip_pitch = -0.3 + 0.4 * left_x  # Movimento para frente/trás
        left_knee = 0.6 - 0.3 * left_x  # Flexão do joelho
        left_ankle_pitch = -0.3 - 0.1 * left_x  # Compensação do tornozelo
        left_ankle_roll = -0.1 - 0.05 * lateral_shift  # Compensação da inclinação
        
        # Valores para a perna direita
        right_hip_yaw = 0.0
        right_hip_roll = -0.1 - 0.05 * lateral_shift
        right_hip_pitch = -0.3 + 0.4 * right_x
        right_knee = 0.6 - 0.3 * right_x
        right_ankle_pitch = -0.3 - 0.1 * right_x
        right_ankle_roll = 0.1 + 0.05 * lateral_shift
        
        # Adicionar componente de rotação
        left_hip_yaw += 0.1 * rotation
        right_hip_yaw += 0.1 * rotation
        
        # Adicionar componente de elevação
        left_hip_pitch -= 0.3 * left_z
        left_knee += 0.6 * left_z
        left_ankle_pitch -= 0.3 * left_z
        
        right_hip_pitch -= 0.3 * right_z
        right_knee += 0.6 * right_z
        right_ankle_pitch -= 0.3 * right_z
        
        # Retornar lista de posições das juntas
        return [
            left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle_pitch, left_ankle_roll,
            right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle_pitch, right_ankle_roll
        ]
    
    def publish_standing_pose(self):
        """Publica a posição de pé padrão."""
        # Posição padrão de pé
        standing_positions = [
            0.0, 0.1, -0.3, 0.6, -0.3, -0.1,  # Perna esquerda
            0.0, -0.1, -0.3, 0.6, -0.3, 0.1   # Perna direita
        ]
        
        self.publish_joint_commands(standing_positions)
    
    def publish_joint_commands(self, positions):
        """Publica comandos para as juntas."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = positions
        
        self.joint_cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = WalkingController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 