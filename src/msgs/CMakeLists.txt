cmake_minimum_required(VERSION 3.8)
project(roboime_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# =============================================================================
# MESSAGE AND SERVICE GENERATION
# =============================================================================

rosidl_generate_interfaces(${PROJECT_NAME}
  # ===== PERCEPTION MESSAGES (Sistema Simplificado YOLOv8) =====
  "msg/perception/SimplifiedDetections.msg"    # NOVA: Sistema principal unificado
  "msg/perception/BallDetection.msg"           # ATUALIZADA: Classe 0 (ball)
  "msg/perception/RobotDetection.msg"          # ATUALIZADA: Classe 1 (robot unified)
  "msg/perception/GoalDetection.msg"           # ATUALIZADA: Classe 3 (goal_post unified)
  "msg/perception/FieldDetection.msg"          # ATUALIZADA: Landmarks (classes 2-6)
  "msg/perception/LandmarkDetection.msg"       # COMPATIBILIDADE: Wrapper para landmarks
  "msg/perception/HybridDetections.msg"        # COMPATIBILIDADE: Sistema legado
  
  # ===== NAVIGATION MESSAGES (Sistema Baseado em Landmarks) =====
  "msg/navigation/RobotPose2D.msg"             # Pose customizada otimizada
  "msg/navigation/FieldLandmark.msg"           # ATUALIZADA: 5 landmarks específicos
  "msg/navigation/LandmarkArray.msg"           # Array de landmarks detectados
  "msg/navigation/LocalizationStatus.msg"      # Status da localização
  "msg/navigation/PathPlan.msg"                # Planejamento de trajetória
  "msg/navigation/NavigationCommand.msg"       # Comandos de navegação
  "msg/navigation/ObstacleInfo.msg"            # Informações de obstáculos
  
  # ===== BEHAVIOR MESSAGES =====
  "msg/behavior/GameState.msg"                 # Estado oficial do jogo
  "msg/behavior/RobotRole.msg"                 # Papel do robô
  "msg/behavior/BehaviorStatus.msg"            # Status do comportamento
  "msg/behavior/Strategy.msg"                  # Estratégia da equipe
  "msg/behavior/ActionCommand.msg"             # Comandos de ação
  
  # ===== MOTION MESSAGES =====
  "msg/motion/WalkCommand.msg"                 # Comandos de caminhada
  "msg/motion/JointCommand.msg"                # Controle de joints
  "msg/motion/MotionStatus.msg"                # Status do movimento
  "msg/motion/GaitParameters.msg"              # Parâmetros de gait
  "msg/motion/Balance.msg"                     # Dados de equilíbrio
  
  # ===== COMMUNICATION MESSAGES =====
  "msg/communication/TeamData.msg"             # Dados compartilhados entre robôs
  "msg/communication/RobotStatus.msg"          # Status individual do robô
  "msg/communication/FieldInfo.msg"            # Informações do campo
  "msg/communication/GameInfo.msg"             # Informações da partida
  
  # ===== SERVICES =====
  "srv/navigation/InitializeLocalization.srv" # Inicializar localização
  "srv/navigation/SetGoal.srv"                 # Definir objetivo de navegação
  "srv/behavior/ChangeRole.srv"                # Mudar papel do robô
  "srv/behavior/SetStrategy.srv"               # Definir estratégia
  "srv/motion/ConfigureGait.srv"               # Configurar gait
  "srv/communication/UpdateTeamData.srv"       # Atualizar dados do time
  
  DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
    nav_msgs
    vision_msgs
    builtin_interfaces
)

# =============================================================================
# EXPORT
# =============================================================================

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package() 