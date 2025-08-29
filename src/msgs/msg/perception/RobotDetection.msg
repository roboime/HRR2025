# RobotDetection.msg  
# Detecção simplificada de robôs no campo (sistema unificado sem distinção de cor)

# Header
std_msgs/Header header

# ===== STATUS DA DETECÇÃO =====
bool detected                    # Se um robô foi detectado
float32 confidence              # Confiança da detecção [0.0, 1.0]
string detection_method         # "yolov8", "fallback"

# ===== POSIÇÃO NA IMAGEM =====
vision_msgs/BoundingBox2D bbox  # Bounding box da detecção

# ===== POSIÇÃO ESTIMADA NO MUNDO REAL =====
geometry_msgs/Point position_3d # Posição 3D estimada (metros)
float32 distance               # Distância estimada (metros)
float32 bearing                # Ângulo horizontal relativo (radianos)

# ===== CLASSIFICAÇÃO SIMPLIFICADA DO ROBÔ =====
# NOTA: Sistema unificado - não distingue cores de time
string robot_classification    # "robot", "unknown" (sem distinção azul/vermelho)
uint32 estimated_number        # Número estimado do robô (se visível, 0 = não identificado)

# ===== POSE E ESTADO DO ROBÔ =====
float32 estimated_orientation  # Orientação estimada (radianos)
string posture                # "standing", "fallen", "walking", "unknown"
string action                 # "idle", "walking", "kicking", "goalkeeping", "unknown"

# ===== TRACKING E MOVIMENTO =====
uint32 track_id               # ID único para tracking (0 = não rastreado)
bool is_moving                # Se o robô está em movimento
geometry_msgs/Vector3 velocity # Velocidade estimada (se disponível)

# ===== CONTEXTO TÁTICO SIMPLIFICADO =====
bool is_threat                # Se representa uma ameaça tática
bool is_blocking_path         # Se está bloqueando um caminho
float32 tactical_priority     # Prioridade tática [0.0, 1.0]

# ===== CARACTERÍSTICAS FÍSICAS =====
float32 estimated_height      # Altura estimada do robô (metros)
float32 estimated_width       # Largura estimada do robô (metros)
bool is_upright              # Se o robô está em pé

# ===== QUALIDADE DA DETECÇÃO =====
bool is_occluded              # Se o robô está parcialmente ocluído
float32 visibility_ratio      # Percentual visível do robô [0.0, 1.0]
string occlusion_type         # "none", "partial", "heavy", "edge"

# ===== INTEGRAÇÃO COM SISTEMA SIMPLIFICADO =====
uint8 yolo_class_id           # Sempre 1 (classe "robot" do YOLOv8)
float32 yolo_confidence       # Confiança original do YOLOv8

# ===== CONTEXTO ESPACIAL =====
bool near_ball                # Se está próximo à bola
bool in_penalty_area          # Se está em uma área de penalty
bool near_goal                # Se está próximo a um gol
string field_region           # "defensive", "midfield", "offensive", "unknown"

# ===== ANÁLISE COMPORTAMENTAL =====
bool likely_teammate          # Estimativa se é companheiro de time (baseado em contexto)
bool likely_opponent          # Estimativa se é adversário (baseado em contexto)
float32 team_estimation_confidence # Confiança da estimativa de time [0.0, 1.0]

# ===== TIMESTAMP E REFERÊNCIA =====
builtin_interfaces/Time stamp # Timestamp da detecção
string frame_id               # Frame de referência
uint32 sequence_number        # Número sequencial para tracking 