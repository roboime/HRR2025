# HybridDetections.msg
# Detecções unificadas do sistema híbrido (YOLOv8 + Detectores Tradicionais)

# Header
std_msgs/Header header

# =============================================================================
# DETECÇÕES YOLOV8 (Objetos Complexos)
# =============================================================================

# Bola
BallDetection ball

# Robôs detectados
RobotDetection[] robots

# Gols detectados  
GoalDetection[] goals

# =============================================================================
# DETECÇÕES TRADICIONAIS (Elementos Geométricos)
# =============================================================================

# Campo
FieldDetection field


# =============================================================================
# LANDMARKS PARA LOCALIZAÇÃO
# =============================================================================

# Landmarks extraídos das detecções para navegação
LandmarkDetection[] landmarks

# =============================================================================
# METADADOS DO SISTEMA HÍBRIDO
# =============================================================================

# Estatísticas de performance
float32 yolo_processing_time_ms      # Tempo do YOLOv8 (ms)
float32 traditional_processing_time_ms # Tempo detectores tradicionais (ms)
float32 total_processing_time_ms     # Tempo total (ms)
float32 fps                         # Taxa de processamento atual

# Status dos detectores
bool yolo_active                    # Se YOLOv8 está ativo
bool traditional_active             # Se detectores tradicionais estão ativos
string processing_mode              # "hybrid", "yolo_only", "traditional_only"

# Qualidade da imagem
float32 image_brightness            # Brilho médio [0.0, 1.0]
float32 image_contrast              # Contraste [0.0, 1.0]
bool lighting_adequate              # Se a iluminação está adequada

# Resumo das detecções
uint32 total_objects_detected       # Total de objetos detectados
uint32 high_confidence_detections   # Detecções com alta confiança (>0.7)
string scene_description            # "active_game", "setup", "empty_field", etc. 