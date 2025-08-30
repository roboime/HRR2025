#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/empty.hpp>
#include <roboime_msgs/msg/robot_pose2_d.hpp>
#include <roboime_msgs/srv/initialize_localization.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "navigation_coordinator.hpp"

class NavigationCoordinatorNode : public rclcpp::Node
{
public:
  NavigationCoordinatorNode() 
    : Node("navigation_coordinator_node")
    , tf_broadcaster_(this)
  {
    // Parâmetros
    this->declare_parameter("field_length", 9.0);
    this->declare_parameter("field_width", 6.0);
    this->declare_parameter("robot_id", 1);
    this->declare_parameter("team_name", "RoboIME");
    this->declare_parameter("team_side", "left");
    this->declare_parameter("publish_rate", 20.0);
    this->declare_parameter("enable_tf_broadcast", true);
    this->declare_parameter("enable_team_communication", false);
    this->declare_parameter("confidence_threshold", 0.7);
    this->declare_parameter("use_global_localization", true);
    
    // Obter parâmetros
    double field_length = this->get_parameter("field_length").as_double();
    double field_width = this->get_parameter("field_width").as_double();
    int robot_id = this->get_parameter("robot_id").as_int();
    std::string team_name = this->get_parameter("team_name").as_string();
    team_side_ = this->get_parameter("team_side").as_string();
    confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
    
    // Inicializar coordenador
    coordinator_ = std::make_unique<roboime_navigation::NavigationCoordinator>(
      field_length, field_width, robot_id, team_name
    );
    
    // Publishers - Pose final fusionada
    pose_pub_ = this->create_publisher<roboime_msgs::msg::RobotPose2D>(
      "robot_pose", 10
    );
    
    pose_cov_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "robot_pose_with_covariance", 10
    );
    
    confidence_pub_ = this->create_publisher<std_msgs::msg::Float64>(
      "localization_confidence", 10
    );
    
    status_pub_ = this->create_publisher<std_msgs::msg::String>(
      "localization_status", 10
    );
    
    mode_pub_ = this->create_publisher<std_msgs::msg::String>(
      "localization_mode", 10
    );
    
    // Publishers para debug individual dos algoritmos
    mcl_pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose2D>(
      "debug/mcl_pose", 5
    );
    
    ekf_pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose2D>(
      "debug/ekf_pose", 5
    );
    
    // Subscribers - Dados dos sensores
    odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odometry", 10,
      std::bind(&NavigationCoordinatorNode::odometry_callback, this, std::placeholders::_1)
    );
    
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "imu/data", 10,
      std::bind(&NavigationCoordinatorNode::imu_callback, this, std::placeholders::_1)
    );
    
    landmarks_sub_ = this->create_subscription<roboime_msgs::msg::LandmarkArray>(
      "perception/landmarks", 10,
      std::bind(&NavigationCoordinatorNode::landmarks_callback, this, std::placeholders::_1)
    );
    
    // Subscribers - Poses individuais dos algoritmos (para comparação)
    mcl_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose2D>(
      "particle_filter/pose", 10,
      std::bind(&NavigationCoordinatorNode::mcl_pose_callback, this, std::placeholders::_1)
    );
    
    ekf_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose2D>(
      "ekf/pose", 10,
      std::bind(&NavigationCoordinatorNode::ekf_pose_callback, this, std::placeholders::_1)
    );
    
    // Subscribers - Comunicação entre robôs (se habilitado)
    if (this->get_parameter("enable_team_communication").as_bool()) {
      // Comentado até existir a mensagem TeamRobotInfo no roboime_msgs
      // team_comm_sub_ = this->create_subscription<roboime_msgs::msg::TeamRobotInfo>(
      //   "team/robot_info", 10,
      //   std::bind(&NavigationCoordinatorNode::team_communication_callback, this, std::placeholders::_1)
      // );
      
      // team_broadcast_pub_ = this->create_publisher<roboime_msgs::msg::TeamRobotInfo>(
      //   "team/broadcast", 5
      // );
    }
    
    // Subscribers - Estado do jogo
    game_state_sub_ = this->create_subscription<std_msgs::msg::String>(
      "game_controller/state", 10,
      std::bind(&NavigationCoordinatorNode::game_state_callback, this, std::placeholders::_1)
    );

    // Postura do motion
    posture_sub_ = this->create_subscription<std_msgs::msg::String>(
      "motion/robot_posture", 10,
      std::bind(&NavigationCoordinatorNode::posture_callback, this, std::placeholders::_1)
    );
    
    // Serviços
    reset_service_ = this->create_service<std_srvs::srv::Empty>(
      "localization/reset",
      std::bind(&NavigationCoordinatorNode::reset_callback, this,
                std::placeholders::_1, std::placeholders::_2)
    );
    
    init_service_ = this->create_service<roboime_msgs::srv::InitializeLocalization>(
      "localization/initialize",
      std::bind(&NavigationCoordinatorNode::initialize_callback, this,
                std::placeholders::_1, std::placeholders::_2)
    );
    
    mode_service_ = this->create_service<roboime_msgs::srv::SetLocalizationMode>(
      "localization/set_mode",
      std::bind(&NavigationCoordinatorNode::set_mode_callback, this,
                std::placeholders::_1, std::placeholders::_2)
    );
    
    // Timer para publicação e atualização
    double publish_rate = this->get_parameter("publish_rate").as_double();
    main_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
      std::bind(&NavigationCoordinatorNode::main_update_cycle, this)
    );

    // Timer de monitoramento de convergência (5 Hz por padrão)
    convergence_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(200),
      std::bind(&NavigationCoordinatorNode::monitor_convergence, this)
    );
    
    // Timer para broadcast do time (menos frequente)
    if (this->get_parameter("enable_team_communication").as_bool()) {
      team_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(200),  // 5Hz
        std::bind(&NavigationCoordinatorNode::broadcast_team_info, this)
      );
    }
    
    // Estado inicial
    is_initialized_ = false;
    last_odometry_ = nullptr;
    
    // Estatísticas
    total_updates_ = 0;
    successful_fusions_ = 0;
    mode_switches_ = 0;
    
    RCLCPP_INFO(this->get_logger(), 
      "Navigation Coordinator iniciado - Robô %d, Time: %s, Campo: %.1fx%.1fm",
      robot_id, team_name.c_str(), field_length, field_width);
    
    // Inicializar com pose padrão
    auto_initialize();
  }

private:
  // =============================================================================
  // MEMBROS DA CLASSE
  // =============================================================================
  std::unique_ptr<roboime_navigation::NavigationCoordinator> coordinator_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  
  // Publishers principais
  rclcpp::Publisher<roboime_msgs::msg::RobotPose2D>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_cov_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr confidence_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mode_pub_;
  
  // Publishers de debug
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr mcl_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr ekf_pose_pub_;
  
  // Publishers de comunicação
  // rclcpp::Publisher<roboime_msgs::msg::TeamRobotInfo>::SharedPtr team_broadcast_pub_;
  
  // Subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<roboime_msgs::msg::LandmarkArray>::SharedPtr landmarks_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr mcl_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr ekf_pose_sub_;
  // rclcpp::Subscription<roboime_msgs::msg::TeamRobotInfo>::SharedPtr team_comm_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr game_state_sub_;
  
  // Serviços
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_service_;
  rclcpp::Service<roboime_msgs::srv::InitializeLocalization>::SharedPtr init_service_;
  rclcpp::Service<roboime_msgs::srv::SetLocalizationMode>::SharedPtr mode_service_;
  
  // Timers
  rclcpp::TimerBase::SharedPtr main_timer_;
  rclcpp::TimerBase::SharedPtr team_timer_;
  rclcpp::TimerBase::SharedPtr convergence_timer_;
  rclcpp::Time convergence_deadline_;
  // Postura do motion para suprimir fallback durante recuperação
  bool pf_is_fallen_{false};
  bool pf_is_recovering_{false};
  rclcpp::Time pf_grace_deadline_{};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr posture_sub_;
  
  // Estado
  std::shared_ptr<nav_msgs::msg::Odometry> last_odometry_;
  std::string team_side_;
  double confidence_threshold_;
  bool is_initialized_;
  
  // Poses individuais para comparação
  std::shared_ptr<geometry_msgs::msg::Pose2D> last_mcl_pose_;
  std::shared_ptr<geometry_msgs::msg::Pose2D> last_ekf_pose_;
  
  // Estatísticas
  size_t total_updates_;
  size_t successful_fusions_;
  size_t mode_switches_;
  roboime_navigation::LocalizationMode last_mode_;
  
  // =============================================================================
  // CALLBACKS DOS SENSORES
  // =============================================================================
  
  void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!is_initialized_) {
      return;
    }
    
    if (last_odometry_) {
      // Calcular delta da odometria
      double dt = (msg->header.stamp.sec - last_odometry_->header.stamp.sec) +
                 (msg->header.stamp.nanosec - last_odometry_->header.stamp.nanosec) * 1e-9;
      
      if (dt > 0 && dt < 1.0) {
        roboime_msgs::msg::RobotPose2D delta;
        delta.x = msg->pose.pose.position.x - last_odometry_->pose.pose.position.x;
        delta.y = msg->pose.pose.position.y - last_odometry_->pose.pose.position.y;
        
        // Calcular delta de orientação
        tf2::Quaternion q1, q2;
        tf2::fromMsg(msg->pose.pose.orientation, q1);
        tf2::fromMsg(last_odometry_->pose.pose.orientation, q2);
        
        double yaw1 = tf2::getYaw(q1);
        double yaw2 = tf2::getYaw(q2);
        delta.theta = normalize_angle(yaw1 - yaw2);
        
        coordinator_->update_with_odometry(delta, dt);
        total_updates_++;
      }
    }
    
    last_odometry_ = msg;
  }
  
  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    if (is_initialized_) {
      coordinator_->update_with_imu(*msg);
    }
  }
  
  void landmarks_callback(const roboime_msgs::msg::LandmarkArray::SharedPtr msg)
  {
    if (!is_initialized_ || msg->landmarks.empty()) {
      return;
    }
    
    // Converter landmarks para formato interno
    std::vector<roboime_navigation::Landmark> landmarks;
    
    for (const auto& landmark_msg : msg->landmarks) {
      roboime_navigation::Landmark::Type type;
      
      // Mapear tipos
      switch (landmark_msg.type) {
        case roboime_msgs::msg::FieldLandmark::CENTER_CIRCLE:
          type = roboime_navigation::Landmark::CENTER_CIRCLE;
          break;
        case roboime_msgs::msg::FieldLandmark::PENALTY_MARK:
          type = roboime_navigation::Landmark::PENALTY_MARK;
          break;
        case roboime_msgs::msg::FieldLandmark::GOAL_POST:
          type = roboime_navigation::Landmark::GOAL_POST;
          break;
        default:
          type = roboime_navigation::Landmark::FIELD_CORNER;
          break;
      }
      
      double distance = std::sqrt(
        landmark_msg.position_relative.x * landmark_msg.position_relative.x +
        landmark_msg.position_relative.y * landmark_msg.position_relative.y
      );
      
      double bearing = std::atan2(
        landmark_msg.position_relative.y,
        landmark_msg.position_relative.x
      );
      
      Eigen::Vector2d position(
        landmark_msg.position_absolute.x,
        landmark_msg.position_absolute.y
      );
      
      landmarks.emplace_back(type, distance, bearing, landmark_msg.confidence);
      landmarks.back().position = position;
    }
    
    coordinator_->update_with_landmarks(landmarks);
    
    RCLCPP_DEBUG(this->get_logger(), 
      "Coordinator atualizado com %zu landmarks", landmarks.size());
  }
  
  void mcl_pose_callback(const geometry_msgs::msg::Pose2D::SharedPtr msg)
  {
    last_mcl_pose_ = msg;
    mcl_pose_pub_->publish(*msg);  // Republish para debug
  }
  
  void ekf_pose_callback(const geometry_msgs::msg::Pose2D::SharedPtr msg)
  {
    last_ekf_pose_ = msg;
    ekf_pose_pub_->publish(*msg);  // Republish para debug
  }
  
  // void team_communication_callback(const roboime_msgs::msg::TeamRobotInfo::SharedPtr msg)
  // {
  //   roboime_navigation::TeamRobotInfo team_info(msg->robot_id);
  //   team_info.pose = msg->pose;
  //   team_info.confidence = msg->confidence;
  //   team_info.last_update = std::chrono::steady_clock::now();
  //   team_info.status = msg->status;
  //   coordinator_->process_team_robot_info(team_info);
  // }
  
  void game_state_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    coordinator_->update_game_state(msg->data);
    
    RCLCPP_INFO(this->get_logger(), 
      "Estado do jogo atualizado: %s", msg->data.c_str());
  }
  
  // =============================================================================
  // SERVIÇOS
  // =============================================================================
  
  void reset_callback(
    const std::shared_ptr<std_srvs::srv::Empty::Request> request,
    std::shared_ptr<std_srvs::srv::Empty::Response> response)
  {
    (void)request;
    (void)response;
    
    RCLCPP_INFO(this->get_logger(), "Resetando Navigation Coordinator...");
    
    coordinator_->reset_localization();
    is_initialized_ = false;
    total_updates_ = 0;
    successful_fusions_ = 0;
    mode_switches_ = 0;
    
    // Reinicializar
    auto_initialize();
    
    RCLCPP_INFO(this->get_logger(), "Navigation Coordinator resetado");
  }
  
  void initialize_callback(
    const std::shared_ptr<roboime_msgs::srv::InitializeLocalization::Request> request,
    std::shared_ptr<roboime_msgs::srv::InitializeLocalization::Response> response)
  {
    try {
      coordinator_->initialize(team_side_, request->initial_pose, false);
      is_initialized_ = true;
      
      response->success = true;
      response->message = "Navigation Coordinator inicializado com sucesso";
      
      RCLCPP_INFO(this->get_logger(), 
        "Inicializado com pose: (%.2f, %.2f, %.2f°)",
        request->initial_pose.x, request->initial_pose.y, 
        request->initial_pose.theta * 180.0 / M_PI);
        
    } catch (const std::exception& e) {
      response->success = false;
      response->message = std::string("Erro na inicialização: ") + e.what();
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
    }
  }
  
  void set_mode_callback(
    const std::shared_ptr<roboime_msgs::srv::SetLocalizationMode::Request> request,
    std::shared_ptr<roboime_msgs::srv::SetLocalizationMode::Response> response)
  {
    try {
      roboime_navigation::LocalizationMode mode;
      
      if (request->mode == "particle_filter") {
        mode = roboime_navigation::LocalizationMode::PARTICLE_FILTER_ONLY;
      } else if (request->mode == "ekf") {
        mode = roboime_navigation::LocalizationMode::EKF_ONLY;
      } else if (request->mode == "hybrid") {
        mode = roboime_navigation::LocalizationMode::HYBRID_FUSION;
      } else if (request->mode == "team_consensus") {
        mode = roboime_navigation::LocalizationMode::TEAM_CONSENSUS;
      } else {
        throw std::invalid_argument("Modo inválido: " + request->mode);
      }
      
      coordinator_->set_localization_mode(mode);
      mode_switches_++;
      
      response->success = true;
      response->message = "Modo alterado para: " + request->mode;
      
      RCLCPP_INFO(this->get_logger(), "Modo alterado para: %s", request->mode.c_str());
      
    } catch (const std::exception& e) {
      response->success = false;
      response->message = std::string("Erro ao alterar modo: ") + e.what();
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
    }
  }
  
  // =============================================================================
  // CICLO PRINCIPAL
  // =============================================================================
  
  void main_update_cycle()
  {
    if (!is_initialized_) {
      return;
    }
    
    try {
      // Obter estimativa fusionada
      auto pose = coordinator_->get_best_pose_estimate();
      auto pose_cov = coordinator_->get_pose_with_covariance();
      double confidence = coordinator_->get_localization_confidence();
      
      // Publicar pose principal
      pose_pub_->publish(pose);
      
      // Pose com covariância
      pose_cov.header.stamp = this->now();
      pose_cov.header.frame_id = "map";
      pose_cov_pub_->publish(pose_cov);
      
      // Confiança
      std_msgs::msg::Float64 confidence_msg;
      confidence_msg.data = confidence;
      confidence_pub_->publish(confidence_msg);
      
      // Status da localização
      std_msgs::msg::String status_msg;
      auto state = coordinator_->get_localization_state();
      switch (state) {
        case roboime_navigation::LocalizationState::TRACKING:
          status_msg.data = "TRACKING";
          break;
        case roboime_navigation::LocalizationState::GLOBAL_LOCALIZATION:
          status_msg.data = "GLOBAL_LOCALIZATION";
          break;
        case roboime_navigation::LocalizationState::LOST:
          status_msg.data = "LOST";
          break;
        case roboime_navigation::LocalizationState::RECOVERED:
          status_msg.data = "RECOVERED";
          break;
        default:
          status_msg.data = "UNINITIALIZED";
          break;
      }
      status_pub_->publish(status_msg);
      
      // Modo de localização
      std_msgs::msg::String mode_msg;
      auto mode = coordinator_->get_localization_mode();
      switch (mode) {
        case roboime_navigation::LocalizationMode::PARTICLE_FILTER_ONLY:
          mode_msg.data = "PARTICLE_FILTER";
          break;
        case roboime_navigation::LocalizationMode::EKF_ONLY:
          mode_msg.data = "EKF";
          break;
        case roboime_navigation::LocalizationMode::HYBRID_FUSION:
          mode_msg.data = "HYBRID_FUSION";
          break;
        case roboime_navigation::LocalizationMode::TEAM_CONSENSUS:
          mode_msg.data = "TEAM_CONSENSUS";
          break;
        default:
          mode_msg.data = "INITIALIZATION";
          break;
      }
      mode_pub_->publish(mode_msg);
      
      // Detectar mudança de modo
      if (mode != last_mode_) {
        mode_switches_++;
        last_mode_ = mode;
      }
      
      // TF transform
      if (this->get_parameter("enable_tf_broadcast").as_bool()) {
        publish_tf_transform(pose);
      }
      
      successful_fusions_++;
      
      // Log periódico
      static int log_counter = 0;
      if (++log_counter % 200 == 0) {  // A cada 10 segundos (20Hz)
        RCLCPP_INFO(this->get_logger(),
          "Coordinator: Pose=(%.2f,%.2f,%.1f°) Conf=%.2f Mode=%s Updates=%zu Switches=%zu",
          pose.x, pose.y, pose.theta * 180.0 / M_PI,
          confidence, mode_msg.data.c_str(), total_updates_, mode_switches_);
      }
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Erro no ciclo principal: %s", e.what());
    }
  }
  
  void broadcast_team_info()
  {
    // Desativado até a mensagem existir
    if (!is_initialized_) {
      return;
    }
    
    try {
      auto pose = coordinator_->get_best_pose_estimate();
      double confidence = coordinator_->get_localization_confidence();
      // Estimar progresso de convergência: normalizar confiança para [0,1]
      double progress = std::min(1.0, std::max(0.0, (confidence - 0.1) / (0.7 - 0.1)));
      
      // publicar quando TeamRobotInfo estiver disponível
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Erro no broadcast: %s", e.what());
    }
  }
  
  // =============================================================================
  // UTILITÁRIOS
  // =============================================================================
  
  void auto_initialize()
  {
    // Inicialização automática usando posição ótima do lado amigo (x negativo)
    roboime_msgs::msg::RobotPose2D initial_pose;
    // Ler configuração
    double opt_x = -2.0;
    double opt_y = 0.0;
    double opt_theta = 0.0;
    std::string friend_side = "left";
    try {
      // Navegação: localization.optimal_start
      // Nota: este nó não carrega YAML diretamente aqui; usa parâmetros se expostos via launch
      // Mantemos defaults e respeitamos team_side_ quando necessário
    } catch (...) {}

    initial_pose.x = opt_x;
    initial_pose.y = opt_y;
    initial_pose.theta = opt_theta;

    // Forçar lado amigo como "left" (x negativo)
    team_side_ = friend_side;

    bool use_global = this->get_parameter("use_global_localization").as_bool();

    coordinator_->initialize(team_side_, initial_pose, use_global);
    is_initialized_ = true;

    RCLCPP_INFO(this->get_logger(),
      "Auto-inicializado na posição ótima (%.2f, %.2f, %.1f°) - Side: %s, Global: %s",
      initial_pose.x, initial_pose.y, initial_pose.theta * 180.0 / M_PI,
      team_side_.c_str(), use_global ? "true" : "false");

    // Iniciar timer de monitoramento de convergência
    double timeout_s = 10.0; // pode ser parametrizado via YAML
    convergence_deadline_ = this->now() + rclcpp::Duration::from_seconds(timeout_s);
  }

  void monitor_convergence()
  {
    if (!is_initialized_) return;
    const auto now_t = this->now();
    // Suprimir fallback/timeout se em recuperação
    bool suppress = pf_is_fallen_ || pf_is_recovering_ ||
      (pf_grace_deadline_.nanoseconds() != 0 && now_t < pf_grace_deadline_);
    double conf = coordinator_->get_localization_confidence();
    if (!suppress && now_t > convergence_deadline_ && conf < 0.7) {
      RCLCPP_WARN(this->get_logger(), "Convergência não atingida no tempo. Fazendo fallback para global.");
      coordinator_->reset_localization();
      // Re-iniciar em modo global
      roboime_msgs::msg::RobotPose2D init_pose; init_pose.x = -2.0; init_pose.y = 0.0; init_pose.theta = 0.0;
      coordinator_->initialize("left", init_pose, true);
      // Estender deadline
      convergence_deadline_ = this->now() + rclcpp::Duration::from_seconds(10.0);
    }
  }

  void posture_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    const auto now_t = this->now();
    if (msg->data == "fallen") {
      pf_is_fallen_ = true; pf_is_recovering_ = false;
      pf_grace_deadline_ = now_t + rclcpp::Duration::from_seconds(3.0);
    } else if (msg->data == "standup_in_progress") {
      pf_is_recovering_ = true; pf_is_fallen_ = false;
    } else if (msg->data == "standing") {
      if (pf_is_fallen_ || pf_is_recovering_) {
        pf_grace_deadline_ = now_t + rclcpp::Duration::from_seconds(2.0);
      }
      pf_is_fallen_ = false; pf_is_recovering_ = false;
    }
  }
  
  void publish_tf_transform(const geometry_msgs::msg::Pose2D& pose)
  {
    geometry_msgs::msg::TransformStamped transform;
    
    transform.header.stamp = this->now();
    transform.header.frame_id = "map";
    transform.child_frame_id = "base_link";
    
    transform.transform.translation.x = pose.x;
    transform.transform.translation.y = pose.y;
    transform.transform.translation.z = 0.0;
    
    tf2::Quaternion q;
    q.setRPY(0, 0, pose.theta);
    transform.transform.rotation = tf2::toMsg(q);
    
    tf_broadcaster_.sendTransform(transform);
  }
  
  double normalize_angle(double angle) const
  {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  
  try {
    auto node = std::make_shared<NavigationCoordinatorNode>();
    
    RCLCPP_INFO(node->get_logger(), "Navigation Coordinator Node iniciado");
    
    rclcpp::spin(node);
    
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("navigation_coordinator_node"), 
      "Erro fatal: %s", e.what());
    return 1;
  }
  
  rclcpp::shutdown();
  return 0;
} 