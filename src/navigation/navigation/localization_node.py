#!/usr/bin/env python3
"""
Interface de Localização RoboIME HSL2025
Interface simplificada que delega para algoritmos C++ avançados (MCL + EKF)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float64, String
import time

# Importar mensagens customizadas do roboime_msgs
from roboime_msgs.msg import FieldDetection, GoalDetection

class LocalizationInterfaceNode(Node):
    """
    Interface simplificada de localização que monitora algoritmos C++ avançados.
    
    Função:
    - Monitorar qualidade dos algoritmos MCL e EKF
    - Fornecer interface de compatibilidade com sistema antigo
    - Reportar status de localização
    - Delegação para Navigation Coordinator (C++)
    
    Algoritmos principais (C++):
    - Particle Filter (MCL): Localização robusta
    - Extended Kalman Filter: Fusão sensorial precisa
    - Navigation Coordinator: Fusão inteligente
    """
    
    def __init__(self):
        super().__init__('localization_interface_node')
        
        # Parâmetros básicos
        self.declare_parameter('monitoring_rate', 10.0)
        self.declare_parameter('compatibility_mode', True)
        self.declare_parameter('min_confidence_threshold', 0.3)
        
        # Obter parâmetros
        self.monitoring_rate = self.get_parameter('monitoring_rate').value
        self.compatibility_mode = self.get_parameter('compatibility_mode').value
        self.min_confidence = self.get_parameter('min_confidence_threshold').value
        
        # Estado do sistema
        self.current_pose = None
        self.localization_confidence = 0.0
        self.mcl_available = False
        self.ekf_available = False
        self.coordinator_available = False
        self.last_update_time = time.time()
        
        # Estatísticas
        self.total_updates = 0
        self.confidence_history = []
        
        # Publishers (para compatibilidade com sistema antigo)
        if self.compatibility_mode:
            self.legacy_pose_pub = self.create_publisher(
                Pose2D, 'legacy/robot_pose', 10
            )
            
            self.legacy_confidence_pub = self.create_publisher(
                Float64, 'legacy/localization_confidence', 10
            )
        
        # Publishers de status
        self.status_pub = self.create_publisher(String, 'localization_interface/status', 10)
        self.health_pub = self.create_publisher(String, 'localization_interface/health', 10)
        
        # Subscribers - Monitorar algoritmos C++
        self.coordinator_pose_sub = self.create_subscription(
            Pose2D,
            'robot_pose',  # Pose fusionada do coordinator
            self.coordinator_pose_callback,
            10
        )
        
        self.coordinator_confidence_sub = self.create_subscription(
            Float64,
            'localization_confidence',
            self.coordinator_confidence_callback,
            10
        )
        
        self.coordinator_status_sub = self.create_subscription(
            String,
            'localization_status',
            self.coordinator_status_callback,
            10
        )
        
        self.coordinator_mode_sub = self.create_subscription(
            String,
            'localization_mode',
            self.coordinator_mode_callback,
            10
        )
        
        # Subscribers - Algoritmos individuais (para monitoramento)
        self.mcl_pose_sub = self.create_subscription(
            Pose2D,
            'mcl/pose',
            self.mcl_pose_callback,
            10
        )
        
        self.ekf_pose_sub = self.create_subscription(
            Pose2D,
            'ekf/pose',
            self.ekf_pose_callback,
            10
        )
        
        self.mcl_confidence_sub = self.create_subscription(
            Float64,
            'mcl/confidence',
            self.mcl_confidence_callback,
            10
        )
        
        self.ekf_confidence_sub = self.create_subscription(
            Float64,
            'ekf/confidence',
            self.ekf_confidence_callback,
            10
        )
        
        # Subscribers - Percepção (para estatísticas)
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
        
        # Timer para monitoramento e publicação
        self.timer = self.create_timer(
            1.0 / self.monitoring_rate, self.monitoring_cycle
        )
        
        # Timer para relatórios (menos frequente)
        self.report_timer = self.create_timer(5.0, self.publish_health_report)
        
        self.get_logger().info('🔗 Interface de Localização iniciada')
        self.get_logger().info('🧮 Monitorando algoritmos C++: MCL + EKF + Coordinator')
        self.get_logger().info(f'⚙️ Taxa de monitoramento: {self.monitoring_rate}Hz')
        self.get_logger().info(f'🔄 Modo compatibilidade: {"Ativo" if self.compatibility_mode else "Inativo"}')
    
    # =============================================================================
    # CALLBACKS DOS ALGORITMOS PRINCIPAIS
    # =============================================================================
    
    def coordinator_pose_callback(self, msg: Pose2D):
        """Callback para pose fusionada do coordenador"""
        self.current_pose = msg
        self.coordinator_available = True
        self.last_update_time = time.time()
        self.total_updates += 1
        
        # Republish para compatibilidade
        if self.compatibility_mode:
            self.legacy_pose_pub.publish(msg)
    
    def coordinator_confidence_callback(self, msg: Float64):
        """Callback para confiança do coordenador"""
        self.localization_confidence = msg.data
        
        # Atualizar histórico
        self.confidence_history.append(msg.data)
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)
        
        # Republish para compatibilidade
        if self.compatibility_mode:
            self.legacy_confidence_pub.publish(msg)
    
    def coordinator_status_callback(self, msg: String):
        """Callback para status do coordenador"""
        self.get_logger().debug(f'Coordinator Status: {msg.data}')
    
    def coordinator_mode_callback(self, msg: String):
        """Callback para modo do coordenador"""
        self.get_logger().debug(f'Coordinator Mode: {msg.data}')
    
    # =============================================================================
    # CALLBACKS DOS ALGORITMOS INDIVIDUAIS (MONITORAMENTO)
    # =============================================================================
    
    def mcl_pose_callback(self, msg: Pose2D):
        """Callback para monitorar MCL"""
        self.mcl_available = True
        
    def ekf_pose_callback(self, msg: Pose2D):
        """Callback para monitorar EKF"""
        self.ekf_available = True
    
    def mcl_confidence_callback(self, msg: Float64):
        """Callback para confiança do MCL"""
        pass  # Apenas para monitoramento
    
    def ekf_confidence_callback(self, msg: Float64):
        """Callback para confiança do EKF"""
        pass  # Apenas para monitoramento
    
    # =============================================================================
    # CALLBACKS DA PERCEPÇÃO (ESTATÍSTICAS)
    # =============================================================================
    
    def landmarks_callback(self, msg: FieldDetection):
        """Callback para landmarks detectados (estatísticas)"""
        if msg.field_detected:
            self.get_logger().debug(
                f'Landmarks detectados: {msg.num_landmarks} '
                f'(penalty_mark: {msg.num_penalty_marks}, '
                f'goals: {msg.num_goals}, '
                f'center_circle: {msg.num_center_circles}, '
                f'field_corners: {msg.num_field_corners}, '
                f'area_corners: {msg.num_area_corners})'
            )
    
    def goals_callback(self, msg: GoalDetection):
        """Callback para postes de gol detectados (estatísticas)"""
        if msg.detected:
            self.get_logger().debug(f'Gols detectados: {msg.num_posts}')
    
    # =============================================================================
    # CICLOS DE MONITORAMENTO
    # =============================================================================
    
    def monitoring_cycle(self):
        """Ciclo principal de monitoramento"""
        current_time = time.time()
        
        # Publicar status atual
        self._publish_status()
        
        # Verificar timeouts dos algoritmos
        self._check_algorithm_health()
        
        # Verificar qualidade da localização
        self._evaluate_localization_quality()
    
    def publish_health_report(self):
        """Publica relatório de saúde dos algoritmos"""
        try:
            current_time = time.time()
            time_since_update = current_time - self.last_update_time
            
            # Calcular estatísticas
            avg_confidence = 0.0
            if self.confidence_history:
                avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
            
            # Status de saúde
            health_status = []
            health_status.append(f"📊 Updates: {self.total_updates}")
            health_status.append(f"⏱️ Última atualização: {time_since_update:.1f}s")
            health_status.append(f"🎯 Confiança atual: {self.localization_confidence:.3f}")
            health_status.append(f"📈 Confiança média: {avg_confidence:.3f}")
            health_status.append(f"🧮 MCL: {'✅' if self.mcl_available else '❌'}")
            health_status.append(f"🎯 EKF: {'✅' if self.ekf_available else '❌'}")
            health_status.append(f"🎮 Coordinator: {'✅' if self.coordinator_available else '❌'}")
            
            if self.current_pose:
                health_status.append(
                    f"📍 Pose: ({self.current_pose.x:.2f}, {self.current_pose.y:.2f}, "
                    f"{self.current_pose.theta*180/3.14159:.1f}°)"
                )
            
            health_report = " | ".join(health_status)
            
            health_msg = String()
            health_msg.data = health_report
            self.health_pub.publish(health_msg)
            
            # Log periódico
            self.get_logger().info(f"🏥 Health: {health_report}")
            
        except Exception as e:
            self.get_logger().error(f"Erro no relatório de saúde: {e}")
    
    # =============================================================================
    # MÉTODOS PRIVADOS
    # =============================================================================
    
    def _publish_status(self):
        """Publica status atual do sistema"""
        try:
            status_parts = []
            
            # Status principal
            if self.coordinator_available:
                if self.localization_confidence >= self.min_confidence:
                    status_parts.append("OPERATIONAL")
                else:
                    status_parts.append("LOW_CONFIDENCE")
            else:
                status_parts.append("COORDINATOR_OFFLINE")
            
            # Informações adicionais
            status_parts.append(f"conf:{self.localization_confidence:.2f}")
            status_parts.append(f"mcl:{'OK' if self.mcl_available else 'OFF'}")
            status_parts.append(f"ekf:{'OK' if self.ekf_available else 'OFF'}")
            
            status_msg = String()
            status_msg.data = "|".join(status_parts)
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Erro na publicação de status: {e}")
    
    def _check_algorithm_health(self):
        """Verifica saúde dos algoritmos"""
        current_time = time.time()
        timeout_threshold = 2.0  # 2 segundos
        
        # Reset flags se não há dados recentes
        if current_time - self.last_update_time > timeout_threshold:
            if self.coordinator_available:
                self.get_logger().warn("⚠️ Coordinator offline - sem dados recentes")
                self.coordinator_available = False
        
        # Verificar se pelo menos um algoritmo está funcionando
        algorithms_active = self.mcl_available or self.ekf_available or self.coordinator_available
        
        if not algorithms_active:
            self.get_logger().error("❌ CRÍTICO: Todos os algoritmos de localização offline!")
    
    def _evaluate_localization_quality(self):
        """Avalia qualidade da localização"""
        if not self.coordinator_available:
            return
        
        # Avaliar confiança
        if self.localization_confidence < 0.1:
            self.get_logger().warn("⚠️ Confiança muito baixa na localização")
        elif self.localization_confidence < self.min_confidence:
            self.get_logger().debug("⚠️ Confiança abaixo do mínimo para navegação")
        
        # Avaliar estabilidade (variância da confiança)
        if len(self.confidence_history) > 10:
            recent_confidences = self.confidence_history[-10:]
            variance = sum((c - self.localization_confidence)**2 for c in recent_confidences) / len(recent_confidences)
            
            if variance > 0.1:  # Alta variância
                self.get_logger().debug("📊 Alta variância na confiança - localização instável")

def main(args=None):
    """Função principal"""
    rclpy.init(args=args)
    
    try:
        node = LocalizationInterfaceNode()
        
        node.get_logger().info("🚀 Interface de Localização iniciada com sucesso")
        node.get_logger().info("🔗 Delegando localização para algoritmos C++ avançados")
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Erro na Interface de Localização: {e}")
    finally:
        try:
            node.destroy_node()
        except:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main() 