#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from yoeo.yoeo_detector import YOEODetector

def main(args=None):
    """Ponto de entrada para o nó detector YOEO."""
    rclpy.init(args=args)
    node = YOEODetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 