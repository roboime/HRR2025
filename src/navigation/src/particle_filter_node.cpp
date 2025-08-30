#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/empty.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "particle_filter.hpp"

class ParticleFilterNode : public rclcpp::Node
{
public:
  ParticleFilterNode() 
    : Node("particle_filter_node")
    , tf_broadcaster_(this)
  {
    // Parâmetros
    this->declare_parameter("field_length", 9.0);
    this->declare_parameter("field_width", 6.0);
    this->declare_parameter("num_particles", 500);
    this->declare_parameter("robot_id", 1);
    this->declare_parameter("team_side", "left");
    this->declare_parameter("publish_rate", 20.0);
    this->declare_parameter("enable_tf_broadcast", true);
    
    // Obter parâmetros
    double field_length = this->get_parameter("field_length").as_double();
    double field_width = this->get_parameter("field_width").as_double();
    size_t num_particles = this->get_parameter("num_particles").as_int();
    robot_id_ = this->get_parameter("robot_id").as_int();
    team_side_ = this->get_parameter("team_side").as_string();
    
    // Inicializar filtro de partículas
    particle_filter_ = std::make_unique<roboime_navigation::ParticleFilter>(
      num_particles, field_length, field_width
    );
    
    // Publishers
    pose_pub_ = this->create_publisher<roboime_msgs::msg::RobotPose2D>(
      "localization/pose", 10
    );
    
    pose_cov_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "localization/pose_with_covariance", 10
    );
    
    confidence_pub_ = this->create_publisher<std_msgs::msg::Float64>(
      "localization/confidence", 10
    );
    convergence_progress_pub_ = this->create_publisher<std_msgs::msg::Float64>(
      "localization/convergence_progress", 10
    );
    
    status_pub_ = this->create_publisher<std_msgs::msg::String>(
      "localization/status", 10
    );
    
    // Subscribers
    odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odometry", 10,
      std::bind(&ParticleFilterNode::odometry_callback, this, std::placeholders::_1)
    );
    
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "imu/data", 10,
      std::bind(&ParticleFilterNode::imu_callback, this, std::placeholders::_1)
    );
    
    landmarks_sub_ = this->create_subscription<roboime_msgs::msg::LandmarkArray>(
      "perception/landmarks", 10,
      std::bind(&ParticleFilterNode::landmarks_callback, this, std::placeholders::_1)
    );
    
    // Postura do motion (preferencial à heurística por IMU)
    posture_sub_ = this->create_subscription<std_msgs::msg::String>(
      "motion/robot_posture", 10,
      std::bind(&ParticleFilterNode::posture_callback, this, std::placeholders::_1)
    );
    
    // Serviços
    reset_service_ = this->create_service<std_srvs::srv::Empty>(
      "localization/reset",
      std::bind(&ParticleFilterNode::reset_localization_callback, this,
                std::placeholders::_1, std::placeholders::_2)
    );
    
    init_service_ = this->create_service<roboime_msgs::srv::InitializeLocalization>(
      "localization/initialize",
      std::bind(&ParticleFilterNode::initialize_localization_callback, this,
                std::placeholders::_1, std::placeholders::_2)
    );
    
    // Timer para publicação
    double publish_rate = this->get_parameter("publish_rate").as_double();
    publish_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
      std::bind(&ParticleFilterNode::publish_localization, this)
    );
    
    // Estado inicial
    last_odometry_ = std::make_shared<nav_msgs::msg::Odometry>();
    last_imu_ = std::make_shared<sensor_msgs::msg::Imu>();
    is_initialized_ = false;
    
    RCLCPP_INFO(this->get_logger(), 
      "Particle Filter Node iniciado - Robô %d, Time: %s, Partículas: %zu",
      robot_id_, team_side_.c_str(), num_particles);
  }

private:
  // =============================================================================
  // MEMBROS DA CLASSE
  // =============================================================================
  std::unique_ptr<roboime_navigation::ParticleFilter> particle_filter_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  
  // Publishers
  rclcpp::Publisher<roboime_msgs::msg::RobotPose2D>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_cov_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr confidence_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr convergence_progress_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  
  // Subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<roboime_msgs::msg::LandmarkArray>::SharedPtr landmarks_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr posture_sub_;
  
  // Serviços
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_service_;
  rclcpp::Service<roboime_msgs::srv::InitializeLocalization>::SharedPtr init_service_;
  
  // Timer
  rclcpp::TimerBase::SharedPtr publish_timer_;
  
  // Estado
  std::shared_ptr<nav_msgs::msg::Odometry> last_odometry_;
  std::shared_ptr<sensor_msgs::msg::Imu> last_imu_;
  std::shared_ptr<nav_msgs::msg::Odometry> previous_odometry_;
  
  int robot_id_;
  std::string team_side_;
  bool is_initialized_;
  
  // Postura/recuperação
  bool is_fallen_ {false};
  bool is_recovering_ {false};
  rclcpp::Time fall_grace_deadline_{};

  // =============================================================================
  // CALLBACKS
  // =============================================================================
  
  void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!is_initialized_) {
      // Auto-inicialização com primeiro dado de odometria
      auto_initialize_localization();
    }
    
    if (previous_odometry_) {
      // Calcular delta da odometria
      roboime_msgs::msg::RobotPose2D odometry_delta;
      calculate_odometry_delta(*previous_odometry_, *msg, odometry_delta);
      
      // Calcular dt
      double dt = (rclcpp::Time(msg->header.stamp) - 
                   rclcpp::Time(previous_odometry_->header.stamp)).seconds();
      
      if (dt > 0.0 && dt < 1.0) {  // Filtrar deltas muito grandes
        // Atualizar filtro com predição
        particle_filter_->predict(odometry_delta, *last_imu_, dt);
      }
    }
    
    previous_odometry_ = std::make_shared<nav_msgs::msg::Odometry>(*msg);
    last_odometry_ = msg;
  }
  
  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    last_imu_ = msg;
    
    // TODO: Usar orientação do IMU para correção adicional
    // Por enquanto apenas armazenamos para uso na predição
  }
  
  void posture_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    const auto now_t = this->now();
    if (msg->data == "fallen") {
      is_fallen_ = true; is_recovering_ = false;
      // Janela de graça após queda
      fall_grace_deadline_ = now_t + rclcpp::Duration::from_seconds(3.0);
      RCLCPP_WARN(this->get_logger(), "Postura detectada: fallen. Suprimindo detecção de kidnapping por 3s.");
    } else if (msg->data == "standup_in_progress") {
      is_recovering_ = true; is_fallen_ = false;
    } else if (msg->data == "standing") {
      // Estabiliza por 2s após levantar
      if (is_fallen_ || is_recovering_) {
        fall_grace_deadline_ = now_t + rclcpp::Duration::from_seconds(2.0);
      }
      is_fallen_ = false; is_recovering_ = false;
    }
  }
  
  void landmarks_callback(const roboime_msgs::msg::LandmarkArray::SharedPtr msg)
  {
    if (!is_initialized_ || msg->landmarks.empty()) {
      return;
    }
    
    // Converter landmarks ROS para formato interno
    std::vector<roboime_navigation::Landmark> landmarks;
    
    for (const auto& landmark_msg : msg->landmarks) {
      // Converter enumerado do msg para nosso tipo interno
      roboime_navigation::Landmark::Type type;
      switch (landmark_msg.type) {
        case roboime_msgs::msg::FieldLandmark::CENTER_CIRCLE:
          type = roboime_navigation::Landmark::CENTER_CIRCLE; break;
        case roboime_msgs::msg::FieldLandmark::PENALTY_MARK:
          type = roboime_navigation::Landmark::PENALTY_MARK; break;
        case roboime_msgs::msg::FieldLandmark::GOAL_POST:
          type = roboime_navigation::Landmark::GOAL_POST; break;
        case roboime_msgs::msg::FieldLandmark::GOAL_AREA_CORNER:
          type = roboime_navigation::Landmark::GOAL_AREA_CORNER; break;
        case roboime_msgs::msg::FieldLandmark::FIELD_CORNER:
        default:
          type = roboime_navigation::Landmark::FIELD_CORNER; break;
      }
      
      double distance = std::sqrt(
        landmark_msg.position_relative.x * landmark_msg.position_relative.x +
        landmark_msg.position_relative.y * landmark_msg.position_relative.y
      );
      
      double bearing = std::atan2(
        landmark_msg.position_relative.y,
        landmark_msg.position_relative.x
      );
      
      landmarks.emplace_back(type, distance, bearing, landmark_msg.confidence);
    }
    
    // Atualizar filtro com observações
    particle_filter_->update(landmarks);
    
    // Verificar se precisa recuperar de kidnapping (suprimido durante queda/recuperação)
    const bool suppress = is_fallen_ || is_recovering_ ||
      (fall_grace_deadline_.nanoseconds() != 0 && this->now() < fall_grace_deadline_);
    if (!suppress && particle_filter_->detect_kidnapping()) {
      RCLCPP_WARN(this->get_logger(), "Kidnapping detectado! Iniciando recuperação...");
      particle_filter_->recover_from_kidnapping();
    }
  }
  
  // =============================================================================
  // SERVIÇOS
  // =============================================================================
  
  void reset_localization_callback(
    const std::shared_ptr<std_srvs::srv::Empty::Request> request,
    std::shared_ptr<std_srvs::srv::Empty::Response> response)
  {
    (void)request;  // Suprimir warning
    
    RCLCPP_INFO(this->get_logger(), "Resetando localização...");
    
    particle_filter_->initialize_global_localization();
    is_initialized_ = true;
    
    response = response;  // Suprimir warning
    
    RCLCPP_INFO(this->get_logger(), "Localização resetada com sucesso");
  }
  
  void initialize_localization_callback(
    const std::shared_ptr<roboime_msgs::srv::InitializeLocalization::Request> request,
    std::shared_ptr<roboime_msgs::srv::InitializeLocalization::Response> response)
  {
    RCLCPP_INFO(this->get_logger(), 
      "Inicializando localização em (%.2f, %.2f, %.2f)",
      request->initial_pose.x, request->initial_pose.y, request->initial_pose.theta);
    
    if (request->use_team_side) {
      particle_filter_->initialize_with_team_side(team_side_, request->initial_pose);
    } else {
      Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
      covariance(0, 0) = request->covariance[0];   // x variance
      covariance(1, 1) = request->covariance[4];   // y variance  
      covariance(2, 2) = request->covariance[8];   // theta variance
      
      particle_filter_->initialize_global_localization(&request->initial_pose, covariance);
    }
    
    is_initialized_ = true;
    response->success = true;
    response->message = "Localização inicializada com sucesso";
    
    RCLCPP_INFO(this->get_logger(), "Localização inicializada com sucesso");
  }
  
  // =============================================================================
  // PUBLICAÇÃO
  // =============================================================================
  
  void publish_localization()
  {
    if (!is_initialized_) {
      return;
    }
    
    auto pose = particle_filter_->get_estimated_pose();
    double confidence = particle_filter_->get_localization_confidence();
    
    // Publicar pose simples
    pose_pub_->publish(pose);
    
    // Publicar pose com covariância
    auto pose_cov_msg = particle_filter_->get_pose_with_covariance();
    pose_cov_msg.header.stamp = this->now();
    pose_cov_msg.header.frame_id = "map";
    pose_cov_pub_->publish(pose_cov_msg);
    
    // Publicar confiança
    std_msgs::msg::Float64 confidence_msg;
    confidence_msg.data = confidence;
    confidence_pub_->publish(confidence_msg);
    // Publicar progresso de convergência (0..1) baseado na confiança
    std_msgs::msg::Float64 progress_msg;
    progress_msg.data = std::min(1.0, std::max(0.0, (confidence - 0.1) / (0.7 - 0.1)));
    convergence_progress_pub_->publish(progress_msg);
    
    // Publicar status
    std_msgs::msg::String status_msg;
    if (is_fallen_ || is_recovering_ ||
        (fall_grace_deadline_.nanoseconds() != 0 && this->now() < fall_grace_deadline_)) {
      status_msg.data = "RECOVERING";
    } else if (confidence > 0.8) {
      status_msg.data = "WELL_LOCALIZED";
    } else if (confidence > 0.5) {
      status_msg.data = "TRACKING";
    } else if (confidence > 0.2) {
      status_msg.data = "UNCERTAIN";
    } else {
      status_msg.data = "LOST";
    }
    status_pub_->publish(status_msg);
    
    // Publicar TF
    if (this->get_parameter("enable_tf_broadcast").as_bool()) {
      publish_tf_transform(pose);
    }
  }
  
  void publish_tf_transform(const roboime_msgs::msg::RobotPose2D& pose)
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
  
  // =============================================================================
  // UTILITÁRIOS
  // =============================================================================
  
  void auto_initialize_localization()
  {
    RCLCPP_INFO(this->get_logger(), "Auto-inicializando localização na posição ótima do lado amigo (left)");

    roboime_msgs::msg::RobotPose2D initial_pose;
    // Força padrão: lado amigo = left (x negativo), frente para o centro
    initial_pose.x = -2.0;
    initial_pose.y = 0.0;
    initial_pose.theta = 0.0;
    team_side_ = "left";

    particle_filter_->initialize_with_team_side(team_side_, initial_pose);
    is_initialized_ = true;

    RCLCPP_INFO(this->get_logger(), "Auto-inicialização concluída (%.2f, %.2f, %.1f°)",
      initial_pose.x, initial_pose.y, initial_pose.theta * 180.0 / M_PI);
  }
  
  void calculate_odometry_delta(
    const nav_msgs::msg::Odometry& prev,
    const nav_msgs::msg::Odometry& curr,
    roboime_msgs::msg::RobotPose2D& delta)
  {
    // Delta em posição
    delta.x = curr.pose.pose.position.x - prev.pose.pose.position.x;
    delta.y = curr.pose.pose.position.y - prev.pose.pose.position.y;
    
    // Delta em orientação
    tf2::Quaternion q_prev, q_curr;
    tf2::fromMsg(prev.pose.pose.orientation, q_prev);
    tf2::fromMsg(curr.pose.pose.orientation, q_curr);
    
    double prev_yaw = tf2::getYaw(q_prev);
    double curr_yaw = tf2::getYaw(q_curr);
    
    delta.theta = normalize_angle(curr_yaw - prev_yaw);
  }
  
  roboime_navigation::Landmark::Type convert_landmark_type(const std::string& type_str)
  {
    if (type_str == "center_circle") {
      return roboime_navigation::Landmark::CENTER_CIRCLE;
    } else if (type_str == "penalty_mark") {
      return roboime_navigation::Landmark::PENALTY_MARK;
    } else if (type_str == "goal_post") {
      return roboime_navigation::Landmark::GOAL_POST;
    } else if (type_str == "goal_area_corner") {
      return roboime_navigation::Landmark::GOAL_AREA_CORNER;
    } else if (type_str == "line_intersection") {
      return roboime_navigation::Landmark::LINE_INTERSECTION;
    } else {
      return roboime_navigation::Landmark::FIELD_CORNER;
    }
  }
  
  double normalize_angle(double angle)
  {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<ParticleFilterNode>();
  
  RCLCPP_INFO(node->get_logger(), "Particle Filter Node rodando...");
  
  rclcpp::spin(node);
  
  rclcpp::shutdown();
  return 0;
} 