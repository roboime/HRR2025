# RobotPose2D.msg
# Pose 2D customizada para robôs de futebol (substitui geometry_msgs/Pose2D deprecated)

# Header com timestamp e frame de referência
std_msgs/Header header

# Posição no campo (metros) - origem no centro do campo
float64 x
float64 y
float64 theta        # Orientação (radianos)

# Covariância da pose (incerteza) - matriz 3x3 flatten
# [xx, xy, xtheta, yx, yy, ytheta, thetax, thetay, thetatheta]
float64[9] covariance

# Confiança da pose estimada [0.0, 1.0]
float32 confidence

# Lado do campo onde o robô está localizado
string field_side    # "left", "right", "center", "unknown"

# Status da localização
string status        # "good", "poor", "lost", "initializing"

# Fonte da estimativa de pose
string source        # "particle_filter", "ekf", "fusion", "odometry", "manual"

# Velocidade estimada (se disponível)
geometry_msgs/Twist velocity 