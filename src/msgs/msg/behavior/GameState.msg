# GameState.msg
# Estado atual do jogo de futebol robótico

# Header
std_msgs/Header header

# =============================================================================
# ESTADO OFICIAL DO JOGO
# =============================================================================

# Estados principais do jogo
uint8 INITIAL = 0
uint8 READY = 1
uint8 SET = 2
uint8 PLAYING = 3
uint8 FINISHED = 4
uint8 PENALTY_SHOOTOUT = 5

uint8 game_state

# Estados secundários durante PLAYING
uint8 NORMAL_PLAY = 0
uint8 KICKOFF_OWN = 1
uint8 KICKOFF_OPPONENT = 2
uint8 CORNER_KICK_OWN = 3
uint8 CORNER_KICK_OPPONENT = 4
uint8 GOAL_KICK_OWN = 5
uint8 GOAL_KICK_OPPONENT = 6
uint8 FREE_KICK_OWN = 7
uint8 FREE_KICK_OPPONENT = 8

uint8 secondary_state

# =============================================================================
# INFORMAÇÕES DA PARTIDA
# =============================================================================

# Tempo de jogo
builtin_interfaces/Duration time_remaining    # Tempo restante
builtin_interfaces/Duration time_elapsed      # Tempo decorrido
bool is_first_half                           # Se é primeiro tempo

# Placar
uint8 score_own                              # Gols do nosso time
uint8 score_opponent                         # Gols do adversário

# Penalidades
uint8 penalties_own                          # Penalidades do nosso time
uint8 penalties_opponent                     # Penalidades do adversário

# =============================================================================
# CONFIGURAÇÃO DO TIME
# =============================================================================

string team_color                            # "blue", "red"
string field_side                            # "left", "right" - lado inicial
bool is_kickoff                              # Se temos o kickoff
uint8 robots_in_game                         # Número de robôs em campo

# =============================================================================
# CONTEXTO TÁTICO
# =============================================================================

string ball_possession                       # "own", "opponent", "none", "contested"
string field_region                         # "defensive", "midfield", "offensive"
string game_phase                           # "early", "middle", "late", "overtime"
float32 urgency_level                       # Nível de urgência tática [0.0, 1.0] 