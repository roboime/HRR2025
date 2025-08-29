# FieldDetection.msg
# Detecção de landmarks do campo para localização (sistema simplificado)

# Header
std_msgs/Header header

# ===== STATUS DA DETECÇÃO =====
bool field_detected                     # Se o campo foi detectado
bool landmarks_detected                 # Se landmarks foram detectados
float32 confidence                      # Confiança geral da detecção [0.0, 1.0]
string detection_method                 # "yolov8", "fallback"

# ===== LANDMARKS DETECTADOS (4 tipos) =====
uint32 num_landmarks                    # Número total de landmarks detectados
uint32 num_penalty_marks                # Número de marcas de penalty
uint32 num_goals                        # Número de postes de gol
uint32 num_center_circles               # Número de círculos centrais
uint32 num_field_corners                # Número de cantos do campo
uint32 num_area_corners                 # Número de cantos das áreas

# ===== ADEQUAÇÃO PARA LOCALIZAÇÃO =====
bool sufficient_for_localization        # Se há landmarks suficientes para localização
float32 localization_confidence         # Confiança estimada para localização [0.0, 1.0]
string localization_status              # "excellent", "good", "poor", "insufficient"

# Landmarks únicos (importantes para localização)
bool center_circle_detected             # Círculo central detectado (landmark único)
bool penalty_marks_detected             # Marcas de penalty detectadas
uint32 high_priority_landmarks          # Landmarks de alta prioridade detectados

# ===== INFORMAÇÕES DO CAMPO =====
# Segmentação básica do campo (se disponível)
sensor_msgs/Image field_mask            # Máscara binária do campo detectado (opcional)
float32 field_coverage_ratio            # Percentual da imagem que é campo [0.0, 1.0]

# Boundary do campo visível
geometry_msgs/Point[] boundary_points   # Pontos da borda do campo visível
bool boundary_complete                  # Se a borda está completamente visível

# ===== QUALIDADE VISUAL =====
# Condições visuais que afetam a detecção
float32 image_brightness                # Brilho médio [0.0, 1.0]
float32 image_contrast                  # Contraste [0.0, 1.0]
bool lighting_adequate                  # Se a iluminação está adequada
string lighting_condition              # "good", "poor", "dark", "bright"

# Qualidade da detecção
bool shadows_detected                   # Se há sombras detectadas
bool reflections_detected              # Se há reflexos detectados
float32 visual_quality                  # Qualidade visual geral [0.0, 1.0]

# ===== CONTEXTO ESPACIAL =====
# Informações sobre a região do campo visível
string field_region_main                # "defensive", "midfield", "offensive", "unknown"
bool near_sideline                      # Se está próximo à linha lateral
bool near_goal_area                     # Se está próximo à área do gol
bool near_center_field                  # Se está próximo ao centro do campo

# ===== INTEGRAÇÃO COM YOLOv8 =====
# Mapeamento das classes YOLOv8 detectadas
bool class_2_detected                   # penalty_mark detectado
bool class_3_detected                   # goal_post detectado
bool class_4_detected                   # center_circle detectado
bool class_5_detected                   # field_corner detectado
bool class_6_detected                   # area_corner detectado

# Confiança média por classe de landmark
float32 penalty_mark_avg_confidence     # Confiança média marcas de penalty
float32 goal_avg_confidence             # Confiança média postes de gol
float32 center_circle_avg_confidence    # Confiança média círculo central
float32 field_corner_avg_confidence     # Confiança média cantos do campo
float32 area_corner_avg_confidence      # Confiança média cantos das áreas

# ===== PERFORMANCE =====
float32 processing_time_ms              # Tempo de processamento (ms)
uint32 frame_number                     # Número do frame processado
bool real_time_capable                  # Se está processando em tempo real

# ===== DEBUGGING =====
string debug_info                       # Informações de debugging
bool debug_mode_active                  # Se modo debug está ativo
uint32 detection_failures               # Número de falhas de detecção consecutivas 