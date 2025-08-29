# GoalDetection.msg
# Detecção de gol para futebol robótico

# Header
std_msgs/Header header

# Status da detecção
bool detected                    # Se um gol foi detectado
float32 confidence              # Confiança da detecção [0.0, 1.0]
string detection_method         # "yolov8", "traditional"

# Posição na imagem
vision_msgs/BoundingBox2D bbox  # Bounding box da detecção

# Identificação do gol
string goal_side                # "left", "right", "unknown"
string goal_type                # "own", "opponent", "unknown"

# Posição estimada no mundo real
geometry_msgs/Point position_3d # Posição 3D estimada (metros)
float32 distance               # Distância estimada (metros)
float32 bearing                # Ângulo horizontal relativo (radianos)

# Detalhes da estrutura
geometry_msgs/Point left_post   # Posição do poste esquerdo
geometry_msgs/Point right_post  # Posição do poste direito
float32 goal_width             # Largura estimada do gol (metros)
float32 goal_height            # Altura estimada do gol (metros)

# Estado do gol
bool is_open                   # Se o gol está "aberto" para chute
float32 open_angle             # Ângulo de abertura disponível (radianos)
bool goalkeeper_visible        # Se há um goleiro visível

# Contexto tático
float32 shooting_priority      # Prioridade para chute [0.0, 1.0]
bool in_shooting_range         # Se está em alcance de chute
float32 estimated_shot_angle   # Melhor ângulo de chute (radianos) 