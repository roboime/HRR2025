#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class BehaviorNode(Node):
    """Nó principal de comportamento do robô."""

    def __init__(self):
        super().__init__('behavior_node')
        self.get_logger().info('Nó de comportamento inicializado')
        
        # Criação de publishers e subscribers
        self.publisher = self.create_publisher(String, 'behavior_status', 10)
        
        # Timer para atualização periódica
        self.timer = self.create_timer(1.0, self.timer_callback)
    
    def timer_callback(self):
        """Callback do timer para atualização periódica."""
        msg = String()
        msg.data = 'Comportamento ativo'
        self.publisher.publish(msg)
        self.get_logger().debug('Status publicado')


def main(args=None):
    """Função principal do nó."""
    rclpy.init(args=args)
    node = BehaviorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Finalizando nó de comportamento')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 