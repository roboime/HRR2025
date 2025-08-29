# WalkCommand.msg
# Comando de caminhada para robô humanoide

# Header
std_msgs/Header header

# Comando de velocidade
geometry_msgs/Twist target_velocity     # Velocidade alvo (linear e angular)

# Parâmetros de caminhada
float32 step_frequency          # Frequência dos passos (Hz)
float32 step_height            # Altura dos passos (metros)
float32 step_length            # Comprimento dos passos (metros)
float32 step_width             # Largura dos passos (metros)

# Balanceamento
bool enable_balancing          # Habilitar controle de equilíbrio
float32 balance_sensitivity    # Sensibilidade do balanceamento [0.0, 1.0]
geometry_msgs/Vector3 com_offset # Offset do centro de massa

# Comportamento especial
string walk_mode               # "normal", "careful", "fast", "turn_only"
bool stop_requested           # Solicitação de parada
bool emergency_stop           # Parada de emergência

# Duração do comando
builtin_interfaces/Duration command_duration  # Duração do comando (0 = contínuo)
uint32 priority               # Prioridade do comando [0-100] 