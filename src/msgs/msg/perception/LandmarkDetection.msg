# LandmarkDetection.msg
# Landmark extraído das detecções de percepção para navegação

# Header
std_msgs/Header header

# Reutiliza a estrutura FieldLandmark mas com dados de percepção
FieldLandmark landmark

# Fonte da detecção do landmark
string source_detection        # "goal_post", "center_circle", "corner", "penalty_mark"
string source_method          # "yolov8", "traditional", "fusion"

# Dados originais da detecção
vision_msgs/BoundingBox2D source_bbox    # Bounding box original
float32 source_confidence               # Confiança da detecção original

# Processamento para navegação
bool processed_for_navigation   # Se foi processado para navegação
float32 navigation_reliability  # Confiabilidade para navegação [0.0, 1.0]
float32 uniqueness_score       # Score de unicidade do landmark [0.0, 1.0]

# Contexto adicional
bool disambiguates_pose        # Se este landmark pode resolver ambiguidade de pose
bool confirms_localization     # Se confirma a localização atual
string tactical_relevance      # "high", "medium", "low" - relevância tática 