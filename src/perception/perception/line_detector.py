#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import math

class LineDetector(Node):
    """
    Nó para detecção das linhas do campo de futebol usando visão computacional.
    
    Este nó processa imagens da câmera para detectar as linhas brancas do campo
    de futebol.
    """
    
    def __init__(self):
        super().__init__('line_detector')
        
        # Parâmetros
        self.declare_parameter('line_color_lower', [200, 200, 200])  # BGR para branco (linhas)
        self.declare_parameter('line_color_upper', [255, 255, 255])
        self.declare_parameter('debug_image', True)
        self.declare_parameter('clean_visualization', True)  # Novo parâmetro para visualização limpa
        self.declare_parameter('line_color', [0, 255, 0])   # Cor das linhas (verde por padrão)
        self.declare_parameter('canny_threshold1', 25)  # Reduzido para detectar mais bordas
        self.declare_parameter('canny_threshold2', 75)  # Reduzido para detectar mais bordas
        self.declare_parameter('hough_threshold', 15)  # Reduzido para detectar mais linhas
        self.declare_parameter('min_line_length', 20)  # Reduzido para detectar linhas menores
        self.declare_parameter('max_line_gap', 15)     # Aumentado para conectar segmentos próximos
        self.declare_parameter('use_field_mask', True)
        self.declare_parameter('binary_threshold', 140)  # Reduzido para pegar mais áreas brancas
        self.declare_parameter('use_adaptive_threshold', True)
        self.declare_parameter('use_histogram_eq', True)
        self.declare_parameter('use_clahe', True)     # Novo: usar CLAHE para melhorar contraste local
        # Parâmetros para filtro de cor HSV para branco - valores ainda menos restritivos
        self.declare_parameter('white_hsv_lower', [0, 0, 130])  # Valor V reduzido para 130
        self.declare_parameter('white_hsv_upper', [180, 60, 255])  # Saturação aumentada para 60
        # Parâmetros para realce de linhas
        self.declare_parameter('enhance_lines', True)  # Ativar realce de linhas
        
        # Obter parâmetros
        self.line_color_lower = np.array(self.get_parameter('line_color_lower').value)
        self.line_color_upper = np.array(self.get_parameter('line_color_upper').value)
        self.debug_image = self.get_parameter('debug_image').value
        self.clean_visualization = self.get_parameter('clean_visualization').value
        self.line_color = tuple(self.get_parameter('line_color').value)  # BGR
        self.canny_threshold1 = self.get_parameter('canny_threshold1').value
        self.canny_threshold2 = self.get_parameter('canny_threshold2').value
        self.hough_threshold = self.get_parameter('hough_threshold').value
        self.min_line_length = self.get_parameter('min_line_length').value
        self.max_line_gap = self.get_parameter('max_line_gap').value
        self.use_field_mask = self.get_parameter('use_field_mask').value
        self.binary_threshold = self.get_parameter('binary_threshold').value
        self.use_adaptive_threshold = self.get_parameter('use_adaptive_threshold').value
        self.use_histogram_eq = self.get_parameter('use_histogram_eq').value
        self.use_clahe = self.get_parameter('use_clahe').value
        self.white_hsv_lower = np.array(self.get_parameter('white_hsv_lower').value)
        self.white_hsv_upper = np.array(self.get_parameter('white_hsv_upper').value)
        self.enhance_lines = self.get_parameter('enhance_lines').value
        
        # Configurar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Publishers
        self.lines_image_pub = self.create_publisher(Image, 'lines_image', 10)
        self.debug_image_pub = self.create_publisher(Image, 'line_detection_debug', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            'camera/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Subscriber para a máscara do campo (opcional)
        if self.use_field_mask:
            self.field_mask_sub = self.create_subscription(
                Image,
                'field_mask',
                self.field_mask_callback,
                10
            )
        
        # Variáveis
        self.cv_bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.field_mask = None
        
        self.get_logger().info('Nó detector de linhas iniciado com parâmetros:')
        self.get_logger().info(f'HSV White Range: {self.white_hsv_lower} - {self.white_hsv_upper}')
        self.get_logger().info(f'Hough Params: threshold={self.hough_threshold}, min_line_length={self.min_line_length}, max_gap={self.max_line_gap}')
    
    def camera_info_callback(self, msg):
        """Callback para informações da câmera."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
    
    def field_mask_callback(self, msg):
        """Callback para receber a máscara do campo."""
        try:
            self.field_mask = self.cv_bridge.imgmsg_to_cv2(msg, 'mono8')
            self.get_logger().debug(f'Máscara do campo recebida, shape: {self.field_mask.shape}')
        except Exception as e:
            self.get_logger().error(f'Erro ao converter máscara do campo: {str(e)}')
    
    def image_callback(self, msg):
        """Callback para processamento de imagem."""
        if self.camera_matrix is None:
            self.get_logger().warn('Informações da câmera ainda não recebidas')
            return
        
        if self.use_field_mask and self.field_mask is None:
            self.get_logger().warn('Máscara do campo ainda não recebida')
            return
        
        try:
            # Converter ROS Image para OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Detectar as linhas
            lines_image, debug_image = self.detect_lines(cv_image)
            
            # Publicar a imagem com as linhas
            lines_image_msg = self.cv_bridge.cv2_to_imgmsg(lines_image, 'mono8')
            lines_image_msg.header = msg.header
            self.lines_image_pub.publish(lines_image_msg)
            
            # Publicar imagem de debug se necessário
            if self.debug_image:
                debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, 'bgr8')
                debug_msg.header = msg.header
                self.debug_image_pub.publish(debug_msg)
                
        except Exception as e:
            self.get_logger().error(f'Erro no processamento da imagem: {str(e)}')
    
    def detect_lines(self, image, field_mask=None):
        """
        Detecta as linhas do campo na imagem.
        
        Args:
            image: Imagem OpenCV no formato BGR
            field_mask: Máscara opcional do campo (padrão: None)
            
        Returns:
            tuple: (imagem_linhas, imagem_debug)
                imagem_linhas: Imagem binária com as linhas detectadas
                imagem_debug: Imagem OpenCV com marcações de debug
        """
        # Criar cópia da imagem para debug
        debug_image = image.copy()
        clean_debug_image = image.copy()
        
        # Aplicar suavização para reduzir ruído
        image_smooth = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Converter para escala de cinza para processamento
        gray = cv2.cvtColor(image_smooth, cv2.COLOR_BGR2GRAY)
        
        # Aplicar máscara do campo primeiro para focar apenas na região de interesse
        if self.use_field_mask and self.field_mask is not None:
            # Redimensionar a máscara se necessário
            if gray.shape != self.field_mask.shape:
                field_mask_resized = cv2.resize(self.field_mask, (gray.shape[1], gray.shape[0]))
            else:
                field_mask_resized = self.field_mask
                
            # Aplicar máscara à imagem cinza
            gray_masked = cv2.bitwise_and(gray, field_mask_resized)
        else:
            gray_masked = gray
        
        # Realçar contraste na imagem cinza
        if self.use_clahe:
            # Usar CLAHE para melhorar o contraste local
            gray_enhanced = self.clahe.apply(gray_masked)
        elif self.use_histogram_eq:
            # Usar equalização global de histograma
            gray_enhanced = cv2.equalizeHist(gray_masked)
        else:
            gray_enhanced = gray_masked
        
        # Adicionar uma cópia da imagem realçada à imagem de debug
        eq_debug = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)
        smaller_debug = cv2.resize(eq_debug, (eq_debug.shape[1]//4, eq_debug.shape[0]//4))
        h, w = smaller_debug.shape[:2]
        debug_image[0:h, 0:w] = smaller_debug
        cv2.putText(debug_image, "Enhanced", (10, h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Adicionar filtro HSV para detectar apenas cores brancas
        hsv = cv2.cvtColor(image_smooth, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.white_hsv_lower, self.white_hsv_upper)
        
        # Se melhoramento de linhas estiver ativado, aplicar realce de bordas
        if self.enhance_lines:
            # Kernel para realce de bordas
            sharpen_kernel = np.array([[-1, -1, -1],
                                     [-1,  9, -1],
                                     [-1, -1, -1]])
            # Aplicar kernel de realce
            gray_enhanced = cv2.filter2D(gray_enhanced, -1, sharpen_kernel)
        
        # Aplicar threshold para destacar as linhas brancas
        if self.use_adaptive_threshold:
            # Usar um threshold adaptativo que se ajusta às variações de iluminação
            binary = cv2.adaptiveThreshold(gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        else:
            # Ou usar um threshold global, com valor ajustado
            _, binary = cv2.threshold(gray_enhanced, self.binary_threshold, 255, cv2.THRESH_BINARY)
        
        # Aplicar a máscara de cor branca para limitar a detecção apenas às áreas brancas
        binary = cv2.bitwise_and(binary, white_mask)
        
        # Mostrar quantidade de pixels brancos na máscara para debug
        white_pixel_count = np.sum(white_mask > 0)
        white_percentage = (white_pixel_count / (white_mask.shape[0] * white_mask.shape[1])) * 100
        self.get_logger().debug(f'Pixels brancos: {white_pixel_count} ({white_percentage:.2f}%)')
        
        # Se estiver usando a máscara do campo fornecida externamente, aplicá-la
        if field_mask is not None and field_mask is not self.field_mask:
            # Redimensionar a máscara se necessário
            if binary.shape != field_mask.shape:
                field_mask = cv2.resize(field_mask, (binary.shape[1], binary.shape[0]))
            
            # Aplicar a máscara
            binary = cv2.bitwise_and(binary, field_mask)
            
        # Debug para a máscara do campo
        if self.field_mask is not None:
            field_mask_sum = np.sum(self.field_mask > 0)
            self.get_logger().debug(f'Pixels na máscara do campo: {field_mask_sum}')
        
        # Aplicar operações morfológicas para remover ruído e conectar segmentos
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=3)  # Aumentado para 3 para melhor conectar segmentos
        
        # Contar pixels brancos após processamento 
        processed_white_pixels = np.sum(binary > 0)
        self.get_logger().debug(f'Pixels brancos após processamento: {processed_white_pixels}')

        # Adicionar uma cópia do resultado binário à imagem de debug
        binary_debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        smaller_binary = cv2.resize(binary_debug, (binary_debug.shape[1]//4, binary_debug.shape[0]//4))
        h, w = smaller_binary.shape[:2]
        x_offset = w  # Posicionar à direita da imagem equalizada
        debug_image[0:h, x_offset:x_offset+w] = smaller_binary
        cv2.putText(debug_image, "Binary", (x_offset+10, h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Detectar bordas com Canny
        edges = cv2.Canny(binary, self.canny_threshold1, self.canny_threshold2)
        
        # Dilatação das bordas para melhorar a detecção de linhas
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Adicionar uma cópia das bordas à imagem de debug
        edges_debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        smaller_edges = cv2.resize(edges_debug, (edges_debug.shape[1]//4, edges_debug.shape[0]//4))
        h, w = smaller_edges.shape[:2]
        x_offset = 2*w  # Posicionar à direita da imagem binária
        debug_image[0:h, x_offset:x_offset+w] = smaller_edges
        cv2.putText(debug_image, "Edges", (x_offset+10, h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Detectar linhas com transformada de Hough
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold,
                               minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
        
        # Criar imagem para as linhas detectadas
        lines_image = np.zeros_like(gray)
        
        # Desenhar as linhas detectadas
        if lines is not None:
            
            # Filtrar as linhas para remover aquelas que não estão dentro do campo
            filtered_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Cálculo do comprimento da linha
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Verificar se os pontos da linha estão dentro da máscara do campo
                if self.field_mask is not None:
                    h, w = self.field_mask.shape
                    if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                        # Para linhas longas, aceitar se qualquer ponto estiver na máscara
                        if line_length > 50:
                            if self.field_mask[y1, x1] > 0 or self.field_mask[y2, x2] > 0:
                                filtered_lines.append(line)
                        # Para linhas mais curtas, verificar também pontos intermediários
                        else:
                            # Verificar o ponto médio da linha
                            mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            if 0 <= mid_x < w and 0 <= mid_y < h and self.field_mask[mid_y, mid_x] > 0:
                                filtered_lines.append(line)
                            elif self.field_mask[y1, x1] > 0 and self.field_mask[y2, x2] > 0:
                                filtered_lines.append(line)
                else:
                    filtered_lines.append(line)
            
            # Desenhar as linhas filtradas
            for line in filtered_lines:
                x1, y1, x2, y2 = line[0]
                
                # Não calculamos mais o ângulo pois todas as linhas terão a mesma cor
                
                cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)
                cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Desenhar linhas com a cor definida no parâmetro
                cv2.line(clean_debug_image, (x1, y1), (x2, y2), self.line_color, 2)
                
                # Adicionar círculos nas extremidades das linhas para melhor visualização
                if self.clean_visualization:
                    # Círculo no início da linha
                    cv2.circle(clean_debug_image, (x1, y1), 3, self.line_color, -1)
                    # Círculo no fim da linha
                    cv2.circle(clean_debug_image, (x2, y2), 3, self.line_color, -1)
        
        # Adicionar informações sobre os parâmetros na imagem de debug
        info_text = f"Linhas: {0 if lines is None else len(lines)} | ML:{self.min_line_length} MG:{self.max_line_gap} H:{self.hough_threshold}"
        cv2.putText(debug_image, info_text, (10, debug_image.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Adicionar informações de HSV
        hsv_text = f"HSV: V>{self.white_hsv_lower[2]}, S<{self.white_hsv_upper[1]}"
        cv2.putText(debug_image, hsv_text, (10, debug_image.shape[0]-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Usar a imagem limpa como imagem de debug se estiver no modo de visualização limpa
        if self.clean_visualization:
            # Adicionar informações mínimas sobre linhas detectadas na visualização limpa
            if lines is not None:
                lines_info = f"Linhas: {len(filtered_lines)}"
                cv2.putText(clean_debug_image, lines_info, (10, clean_debug_image.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(clean_debug_image, lines_info, (10, clean_debug_image.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
        return lines_image, clean_debug_image if self.clean_visualization else debug_image

def main(args=None):
    rclpy.init(args=args)
    node = LineDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 