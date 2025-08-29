# BallDetection.msg
# Detecção de bola otimizada para futebol robótico

# Header
std_msgs/Header header

# Status da detecção
bool detected                    # Se a bola foi detectada
float32 confidence              # Confiança da detecção [0.0, 1.0]
string detection_method         # "yolov8", "traditional", "tracking"

# Posição na imagem (pixels)
vision_msgs/BoundingBox2D bbox  # Bounding box da detecção

# Posição estimada no mundo real
geometry_msgs/Point position_3d # Posição 3D estimada (metros)
float32 distance               # Distância estimada (metros)
float32 bearing                # Ângulo horizontal relativo (radianos)

# Propriedades da bola
float32 estimated_radius       # Raio estimado da bola (metros)
float32 motion_blur           # Indicador de motion blur [0.0, 1.0]

# Tracking da bola (se disponível)
uint32 track_id               # ID do tracking (0 = não rastreada)
geometry_msgs/Vector3 velocity # Velocidade estimada da bola
bool is_moving                # Se a bola está em movimento

# Contexto adicional
bool is_occluded              # Se a bola está parcialmente ocluída
bool near_field_boundary      # Se está próxima da borda do campo
float32 visibility_ratio      # Percentual visível da bola [0.0, 1.0] 