# TeamData.msg
# Dados compartilhados entre robôs da equipe

# Header
std_msgs/Header header

# =============================================================================
# IDENTIFICAÇÃO DO ROBÔ
# =============================================================================

uint8 robot_id                              # ID único do robô (1-4)
string robot_name                           # Nome/identificação do robô
float32 battery_level                       # Nível da bateria [0.0, 1.0]
bool is_active                              # Se o robô está ativo no jogo

# =============================================================================
# LOCALIZAÇÃO E MOVIMENTO
# =============================================================================

# Pose atual do robô
RobotPose2D pose                            # Posição e orientação
float32 pose_confidence                     # Confiança na localização [0.0, 1.0]

# Movimento
geometry_msgs/Twist current_velocity        # Velocidade atual
string current_action                       # "idle", "walking", "turning", "kicking"

# =============================================================================
# PERCEPÇÕES COMPARTILHADAS
# =============================================================================

# Bola
BallDetection ball_seen                     # Última detecção da bola
builtin_interfaces/Time ball_last_seen      # Quando viu a bola pela última vez

# Outros robôs
RobotDetection[] opponents_seen             # Adversários detectados
uint32 teammates_visible                    # Quantos companheiros estão visíveis

# =============================================================================
# ESTADO COMPORTAMENTAL
# =============================================================================

string current_role                         # "striker", "defender", "goalkeeper", "supporter"
string behavior_state                       # Estado atual do comportamento
string target_object                        # "ball", "goal", "position", "none"
geometry_msgs/Point target_position         # Posição alvo atual

# =============================================================================
# COMUNICAÇÃO E COORDENAÇÃO
# =============================================================================

uint8 communication_quality                 # Qualidade da comunicação [0-100]
builtin_interfaces/Time last_communication  # Última comunicação recebida
string[] coordination_messages              # Mensagens de coordenação

# Consenso de equipe
bool agrees_on_ball_position                # Concorda com posição da bola da equipe
bool agrees_on_strategy                     # Concorda com estratégia atual 