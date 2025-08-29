# SimplifiedDetections.msg
# Sistema de detecções unificado YOLOv8 simplificado (6 classes)
# Substitui HybridDetections.msg para arquitetura otimizada

# Header
std_msgs/Header header

# =============================================================================
# DETECÇÕES PARA ESTRATÉGIA (2 classes)
# =============================================================================

# Bola (Classe 0)
BallDetection ball

# Robôs (Classe 1) - Unificado sem distinção de cor
RobotDetection[] robots
uint32 num_robots                        # Número de robôs detectados

# =============================================================================
# DETECÇÕES PARA LOCALIZAÇÃO (5 classes landmarks)
# =============================================================================

# Landmarks unificados para navegação
FieldDetection field_landmarks           # Contém todos os landmarks detectados
uint32 num_landmarks                     # Número total de landmarks detectados

# Landmarks específicos detectados
FieldLandmark[] penalty_marks            # Marcas de penalty (Classe 2)
FieldLandmark[] goals                    # Estruturas de gol (Classe 3)
FieldLandmark[] center_circles           # Círculo central (Classe 4)
FieldLandmark[] field_corners            # Cantos do campo (Classe 5)
FieldLandmark[] area_corners             # Cantos das áreas (Classe 6)

# Gols com informações detalhadas
GoalDetection[] goal_detections          # Detecções detalhadas de gols

# =============================================================================
# METADADOS DO SISTEMA SIMPLIFICADO
# =============================================================================

# Performance YOLOv8
float32 yolo_processing_time_ms          # Tempo do YOLOv8 (ms)
float32 post_processing_time_ms          # Tempo de pós-processamento (ms)
float32 total_processing_time_ms         # Tempo total (ms)
float32 fps                             # Taxa de processamento atual

# Status do sistema
bool yolo_active                        # Se YOLOv8 está ativo
string processing_mode                  # "yolov8", "fallback", "disabled"
uint32 frame_number                     # Número do frame processado

# Qualidade da imagem
float32 image_brightness                # Brilho médio [0.0, 1.0]
float32 image_contrast                  # Contraste [0.0, 1.0]
bool lighting_adequate                  # Se a iluminação está adequada
string lighting_condition              # "good", "poor", "dark", "bright"

# Resumo das detecções por categoria
uint32 total_strategy_objects           # Total objetos estratégia (ball + robots)
uint32 total_localization_landmarks     # Total landmarks localização
uint32 high_confidence_detections       # Detecções com alta confiança (>0.7)
uint32 medium_confidence_detections     # Detecções com média confiança (0.5-0.7)
uint32 low_confidence_detections        # Detecções com baixa confiança (<0.5)

# Análise da cena
string scene_description                # "active_game", "setup", "empty_field", "warm_up"
bool field_detected                     # Se o campo foi detectado na imagem
float32 field_coverage_ratio            # Percentual da imagem que é campo [0.0, 1.0]

# =============================================================================
# ESTATÍSTICAS DE CONFIANÇA POR CLASSE
# =============================================================================

# Confiança média por classe YOLOv8
float32 ball_avg_confidence             # Confiança média detecções de bola
float32 robot_avg_confidence            # Confiança média detecções de robôs
float32 penalty_mark_avg_confidence     # Confiança média marcas de penalty
float32 goal_avg_confidence             # Confiança média estruturas de gol
float32 center_circle_avg_confidence    # Confiança média círculo central
float32 field_corner_avg_confidence     # Confiança média cantos do campo
float32 area_corner_avg_confidence      # Confiança média cantos das áreas

# =============================================================================
# CONTEXTO TÁTICO
# =============================================================================

# Situação tática detectada
bool ball_in_play                       # Se a bola está em jogo
bool ball_near_goal                     # Se a bola está próxima a um gol
bool crowded_area                       # Se há muitos robôs próximos
bool clear_shot_available               # Se há chute livre disponível

# Distribuição espacial
uint32 robots_in_defense                # Robôs na área defensiva
uint32 robots_in_midfield               # Robôs no meio-campo
uint32 robots_in_attack                 # Robôs na área de ataque

# =============================================================================
# LOCALIZAÇÃO E NAVEGAÇÃO
# =============================================================================

# Adequação para localização
bool sufficient_for_localization        # Se há landmarks suficientes para localização
float32 localization_confidence         # Confiança estimada para localização [0.0, 1.0]
string localization_status              # "excellent", "good", "poor", "insufficient"

# Landmarks únicos detectados (importantes para localização)
bool center_circle_detected             # Círculo central (landmark único)
bool penalty_marks_detected             # Marcas de penalty detectadas
uint32 unique_landmarks_count           # Número de landmarks únicos

# =============================================================================
# DEBUGGING E DIAGNÓSTICO
# =============================================================================

# Informações de debugging
bool debug_mode_active                  # Se modo debug está ativo
string debug_info                      # Informações de debugging adicionais
uint32 dropped_frames                  # Frames descartados por performance

# Estatísticas de tempo real
float64 timestamp_capture              # Timestamp da captura da imagem
float64 timestamp_processing           # Timestamp do início do processamento
float64 timestamp_completion           # Timestamp da conclusão

# Status do hardware
float32 gpu_utilization                # Utilização da GPU [0.0, 1.0]
float32 memory_usage                   # Uso de memória [0.0, 1.0]
float32 temperature_celsius            # Temperatura do sistema (°C) 