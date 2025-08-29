# LocalizationStatus.msg
# Status completo do sistema de localização

# Header
std_msgs/Header header

# Status geral
string status           # "good", "poor", "lost", "initializing", "converging"
float32 confidence      # Confiança global [0.0, 1.0]

# Pose atual do robô
RobotPose2D current_pose

# Estatísticas do filtro de partículas
uint32 num_particles_active
float32 effective_particles_ratio   # Neff/N - indica diversidade das partículas
bool kidnap_detected               # Se foi detectado "sequestro" do robô

# Estatísticas do EKF
float64[9] ekf_covariance          # Matriz de covariância do EKF
float32 innovation_magnitude        # Magnitude da inovação
bool outlier_detected              # Se foram detectados outliers

# Landmarks visíveis atualmente  
uint32 landmarks_visible
uint32 landmarks_reliable
string best_landmark_type          # Tipo do melhor landmark detectado

# Performance
float32 processing_time_ms         # Tempo de processamento (ms)
float32 update_rate_hz            # Taxa de atualização atual
uint32 updates_since_start        # Número de atualizações desde inicialização

# Comunicação com outros robôs
uint32 robots_in_communication     # Quantos outros robôs estão se comunicando
bool team_consensus_available      # Se há consenso da equipe sobre localização 