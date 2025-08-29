#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import numpy as np
import math

from sensor_msgs.msg import Imu
from std_msgs.msg import String

try:
    import serial
except Exception:
    serial = None

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
        # IMU/STM
        self.declare_parameter('enable_imu_serial', False)
        self.declare_parameter('imu_serial_port', '/dev/ttyACM0')
        self.declare_parameter('imu_baudrate', 115200)
        self.declare_parameter('imu_frame_id', 'imu_link')
        self.declare_parameter('imu_complementary_alpha', 0.98)
        self.declare_parameter('imu_calibration_samples', 300)
        self.declare_parameter('imu_publish_rate', 100.0)
        
        # Obter parâmetros
        self.step_height = self.get_parameter('step_height').value
        self.step_length = self.get_parameter('step_length').value
        self.lateral_width = self.get_parameter('lateral_width').value
        self.walk_frequency = self.get_parameter('walk_frequency').value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        
        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 50)
        self.posture_pub = self.create_publisher(String, 'motion/robot_posture', 10)
        
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
        
        # IMU/STM estado
        self.imu_enabled = self.get_parameter('enable_imu_serial').value and (serial is not None)
        self.imu_serial = None
        self.imu_bias_gx = 0.0
        self.imu_bias_gy = 0.0
        self.imu_bias_gz = 0.0
        self.imu_roll = 0.0
        self.imu_pitch = 0.0
        self.imu_yaw = 0.0
        self.imu_last_ts = None
        self.imu_alpha = float(self.get_parameter('imu_complementary_alpha').value)
        self.imu_frame_id = self.get_parameter('imu_frame_id').value
        self.posture_state = 'standing'
        self.is_fallen = False
        self.is_recovering = False
        
        # Definir juntas do robô
        self.joint_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll'
        ]
        
        # Timer para o ciclo de caminhada
        self.timer = self.create_timer(0.01, self.walking_cycle)
        
        # Timer do IMU
        if self.imu_enabled:
            self._init_imu()
            imu_period = 1.0 / float(self.get_parameter('imu_publish_rate').value)
            self.imu_timer = self.create_timer(imu_period, self._imu_poll)
        
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

    # ===== IMU/STM integration =====
    def _init_imu(self):
        try:
            port = self.get_parameter('imu_serial_port').value
            baud = int(self.get_parameter('imu_baudrate').value)
            self.imu_serial = serial.Serial(port, baudrate=baud, timeout=0.001)
            self.get_logger().info(f'IMU (STM/MPU-6050) conectado em {port} @ {baud} baud')
            # Calibração de bias do giroscópio (robô parado)
            n = int(self.get_parameter('imu_calibration_samples').value)
            gx_sum = gy_sum = gz_sum = 0.0
            count = 0
            while count < n:
                line = self._readline_serial()
                gx, gy, gz, ax, ay, az = self._parse_imu_line(line) if line else (None,)*6
                if gx is None:
                    continue
                gx_sum += gx; gy_sum += gy; gz_sum += gz
                count += 1
            if count > 0:
                self.imu_bias_gx = gx_sum / count
                self.imu_bias_gy = gy_sum / count
                self.imu_bias_gz = gz_sum / count
                self.get_logger().info(f'IMU bias (deg/s): gx={self.imu_bias_gx:.3f}, gy={self.imu_bias_gy:.3f}, gz={self.imu_bias_gz:.3f}')
        except Exception as e:
            self.get_logger().error(f'Falha ao iniciar IMU serial: {e}')
            self.imu_enabled = False
    
    def _readline_serial(self):
        if self.imu_serial is None:
            return None
        try:
            return self.imu_serial.readline().decode(errors='ignore').strip()
        except Exception:
            return None
    
    def _parse_imu_line(self, line):
        try:
            parts = [p for p in line.replace('\r','').split(',') if p]
            if len(parts) < 6:
                return (None,)*6
            gx = float(parts[0]); gy = float(parts[1]); gz = float(parts[2])
            ax = float(parts[3]); ay = float(parts[4]); az = float(parts[5])
            return gx, gy, gz, ax, ay, az
        except Exception:
            return (None,)*6
    
    def _imu_poll(self):
        line = self._readline_serial()
        gx, gy, gz, ax, ay, az = self._parse_imu_line(line) if line else (None,)*6
        if gx is None:
            return
        # Remover bias e converter para rad/s
        gx = math.radians(gx - self.imu_bias_gx)
        gy = math.radians(gy - self.imu_bias_gy)
        gz = math.radians(gz - self.imu_bias_gz)
        # dt
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.imu_last_ts is None:
            self.imu_last_ts = now
            return
        dt = max(1e-4, min(0.1, now - self.imu_last_ts))
        self.imu_last_ts = now
        # Estimar roll/pitch de accel
        acc_norm = max(1e-6, math.sqrt(ax*ax + ay*ay + az*az))
        axn, ayn, azn = ax/acc_norm, ay/acc_norm, az/acc_norm
        acc_roll = math.atan2(ayn, azn)
        acc_pitch = math.atan2(-axn, math.sqrt(ayn*ayn + azn*azn))
        # Integrar gyro
        self.imu_roll += gx * dt
        self.imu_pitch += gy * dt
        self.imu_yaw += gz * dt
        # Filtro complementar
        a = self.imu_alpha
        self.imu_roll = a * self.imu_roll + (1 - a) * acc_roll
        self.imu_pitch = a * self.imu_pitch + (1 - a) * acc_pitch
        # Publicar IMU
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = self.imu_frame_id
        cy = math.cos(self.imu_yaw * 0.5); sy = math.sin(self.imu_yaw * 0.5)
        cp = math.cos(self.imu_pitch * 0.5); sp = math.sin(self.imu_pitch * 0.5)
        cr = math.cos(self.imu_roll * 0.5); sr = math.sin(self.imu_roll * 0.5)
        imu_msg.orientation.w = cr*cp*cy + sr*sp*sy
        imu_msg.orientation.x = sr*cp*cy - cr*sp*sy
        imu_msg.orientation.y = cr*sp*cy + sr*cp*sy
        imu_msg.orientation.z = cr*cp*sy - sr*sp*cy
        imu_msg.angular_velocity.x = gx
        imu_msg.angular_velocity.y = gy
        imu_msg.angular_velocity.z = gz
        imu_msg.linear_acceleration.x = ax
        imu_msg.linear_acceleration.y = ay
        imu_msg.linear_acceleration.z = az
        self.imu_pub.publish(imu_msg)
        # Atualizar postura
        self._update_posture()
    
    def _update_posture(self):
        roll_deg = abs(math.degrees(self.imu_roll))
        pitch_deg = abs(math.degrees(self.imu_pitch))
        prev = self.posture_state
        if roll_deg > 45.0 or pitch_deg > 45.0:
            self.posture_state = 'fallen'
            self.is_fallen = True
            self.is_recovering = False
        elif roll_deg < 15.0 and pitch_deg < 15.0:
            if self.is_fallen:
                self.posture_state = 'standup_in_progress'
                self.is_recovering = True
                self.create_timer(0.8, self._mark_standing_once)
                self.is_fallen = False
            else:
                self.posture_state = 'standing'
                self.is_recovering = False
        else:
            self.posture_state = 'unknown'
        msg = String(); msg.data = self.posture_state
        self.posture_pub.publish(msg)
        if prev != self.posture_state:
            self.get_logger().info(f'Postura: {self.posture_state}')
    
    def _mark_standing_once(self):
        self.posture_state = 'standing'
        self.is_recovering = False

def main(args=None):
    rclpy.init(args=args)
    node = WalkingController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 