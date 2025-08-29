# LandmarkArray.msg
# Array de landmarks detectados no campo de futebol

# Header com timestamp e frame de referência
std_msgs/Header header

# Array de landmarks detectados
FieldLandmark[] landmarks

# Informações adicionais da detecção
uint32 total_detected          # Número total de landmarks detectados
uint32 reliable_count          # Número de landmarks com alta confiança (>0.7)
float32 detection_confidence   # Confiança média de todas as detecções
string detection_method        # "yolov8", "traditional", "hybrid"

# Status da localização baseada nestes landmarks
bool sufficient_for_localization  # Se há landmarks suficientes para localização confiável
string localization_status        # "good", "poor", "insufficient", "ambiguous" 