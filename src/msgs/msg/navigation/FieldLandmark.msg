# FieldLandmark.msg
# Landmark específico do campo de futebol RoboCup Humanoid League
# Sistema simplificado com 5 tipos de landmarks para localização

# ===== TIPOS DE LANDMARKS (SISTEMA SIMPLIFICADO) =====
uint8 UNKNOWN = 0

# Landmarks para Localização (4 tipos específicos)
uint8 PENALTY_MARK = 1       # Marcas de penalty (2 no campo - prioridade ALTA)
uint8 GOAL_POST = 2          # Postes dos gols (4 no campo - prioridade MÉDIA)  
uint8 CENTER_CIRCLE = 3      # Círculo central (1 no campo - prioridade ALTA)
uint8 FIELD_CORNER = 4       # Cantos do campo (4 no campo - prioridade MÉDIA)
uint8 AREA_CORNER = 5        # Cantos das áreas (8 no campo - prioridade MÉDIA)

# ===== IDENTIFICAÇÃO DO LANDMARK =====
uint8 type                   # Tipo do landmark (usar constantes acima)
string description           # Descrição textual (ex: "penalty_mark_left", "center_circle")

# ===== POSIÇÃO NO SISTEMA DE COORDENADAS DO CAMPO =====
# Sistema de coordenadas: Origem no centro do campo
# X+ = direção gol direito, Y+ = linha lateral superior, Z+ = altura
geometry_msgs/Point position_absolute   # Posição absoluta conhecida no campo (metros)

# ===== OBSERVAÇÃO RELATIVA DO ROBÔ =====
geometry_msgs/Point position_relative   # Posição relativa observada (metros)
float32 distance                        # Distância estimada do robô ao landmark (metros)
float32 bearing                         # Ângulo relativo ao robô (radianos) - 0 = frente do robô
float32 elevation_angle                 # Ângulo de elevação (radianos) - para landmarks 3D

# ===== CONFIANÇA E QUALIDADE =====
float32 confidence                      # Confiança da detecção [0.0, 1.0]
float32 observation_quality             # Qualidade da observação [0.0, 1.0]
bool is_occluded                        # Se o landmark está parcialmente ocluído
float32 visibility_ratio                # Percentual visível do landmark [0.0, 1.0]

# ===== PRIORIDADE PARA LOCALIZAÇÃO =====
uint8 PRIORITY_HIGH = 1      # penalty_mark, center_circle
uint8 PRIORITY_MEDIUM = 2    # goal_post, field_corner  
uint8 PRIORITY_LOW = 3       

uint8 localization_priority             # Prioridade deste landmark para localização
float32 uniqueness_score                # Score de unicidade no campo [0.0, 1.0]
bool disambiguates_pose                 # Se este landmark pode resolver ambiguidade de pose

# ===== INTEGRAÇÃO COM SISTEMA SIMPLIFICADO =====
uint8 yolo_class_id                     # ID da classe YOLOv8 original (2-6)
string source_detection_method          # "yolov8", "fallback"
float32 yolo_confidence                 # Confiança da detecção YOLOv8 original

# ===== TIMESTAMP E CONTEXTO =====
builtin_interfaces/Time stamp           # Timestamp da detecção
string frame_id                         # Frame de referência da observação
uint32 sequence_number                  # Número sequencial para tracking 