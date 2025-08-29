#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "ekf_localization.hpp"

class EKFLocalizationNode : public rclcpp::Node
{
public:
  EKFLocalizationNode() 
    : Node("ekf_localization_node")
    , tf_broadcaster_(this)
  {
    // Parâmetros
    this->declare_parameter("field_length", 9.0);
    this->declare_parameter("field_width", 6.0);
    this->declare_parameter("publish_rate", 20.0);
    this->declare_parameter("enable_tf_broadcast", true);
    this->declare_parameter("process_noise_std", 0.1);
    this->declare_parameter("measurement_noise_std", 0.15);
    this->declare_parameter("imu_noise_std", 0.02);
    this->declare_parameter("innovation_threshold", 9.0);
    
    // Obter parâmetros
    double field_length = this->get_parameter("field_length").as_double();
    double field_width = this->get_parameter("field_width").as_double();
    double process_noise = this->get_parameter("process_noise_std").as_double();
    double meas_noise = this->get_parameter("measurement_noise_std").as_double();
    double imu_noise = this->get_parameter("imu_noise_std").as_double();
    innovation_threshold_ = this->get_parameter("innovation_threshold").as_double();
    
    // Inicializar EKF
    ekf_ = std::make_unique<roboime_navigation::EKFLocalization>(field_length, field_width);
    
    // Configurar ruído
    Eigen::Matrix3d Q = Eigen::Matrix3d::Identity() * process_noise * process_noise;
    ekf_->update_noise_parameters(Q, meas_noise, imu_noise);
    
    // Publishers
    pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose2D>(
      "ekf/pose", 10
    );
    
    pose_cov_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "ekf/pose_with_covariance", 10
    );
    
    confidence_pub_ = this->create_publisher<std_msgs::msg::Float64>(
      "ekf/confidence", 10
    );
    
    status_pub_ = this->create_publisher<std_msgs::msg::String>(
      "ekf/status", 10
    );
    
    // Subscribers
    odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odometry", 10,
      std::bind(&EKFLocalizationNode::odometry_callback, this, std::placeholders::_1)
    );
    
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "imu/data", 10,
      std::bind(&EKFLocalizationNode::imu_callback, this, std::placeholders::_1)
    );
    
    landmarks_sub_ = this->create_subscription<roboime_msgs::msg::LandmarkArray>(
      "perception/landmarks", 10,
      std::bind(&EKFLocalizationNode::landmarks_callback, this, std::placeholders::_1)
    );
    
    // Serviços
    reset_service_ = this->create_service<std_srvs::srv::Empty>(
      "ekf/reset",
      std::bind(&EKFLocalizationNode::reset_callback, this,
                std::placeholders::_1, std::placeholders::_2)
    );
    
    init_service_ = this->create_service<roboime_msgs::srv::InitializeLocalization>(
      "ekf/initialize",
      std::bind(&EKFLocalizationNode::initialize_callback, this,
                std::placeholders::_1, std::placeholders::_2)
    );
    
    // Timer para publicação
    double publish_rate = this->get_parameter("publish_rate").as_double();
    publish_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
      std::bind(&EKFLocalizationNode::publish_localization, this)
    );
    
    // Estado inicial
    last_odometry_ = std::make_shared<nav_msgs::msg::Odometry>();
    is_initialized_ = false;
    
    RCLCPP_INFO(this->get_logger(), 
      "EKF Localization Node iniciado - Campo: %.1fx%.1fm", 
      field_length, field_width);
    RCLCPP_INFO(this->get_logger(), 
      "Parâmetros - Process: %.3f, Measurement: %.3f, IMU: %.3f", 
      process_noise, meas_noise, imu_noise);
  }

private:
  // =============================================================================
  // MEMBROS DA CLASSE
  // =============================================================================
  std::unique_ptr<roboime_navigation::EKFLocalization> ekf_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  
  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_cov_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr confidence_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  
  // Subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<roboime_msgs::msg::LandmarkArray>::SharedPtr landmarks_sub_;
  
  // Serviços
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_service_;
  rclcpp::Service<roboime_msgs::srv::InitializeLocalization>::SharedPtr init_service_;
  
  // Timer
  rclcpp::TimerBase::SharedPtr publish_timer_;
  
  // Estado
  std::shared_ptr<nav_msgs::msg::Odometry> last_odometry_;
  std::shared_ptr<nav_msgs::msg::Odometry> previous_odometry_;
  
  bool is_initialized_;
  double innovation_threshold_;
  rclcpp::Time last_update_time_;
  
  // Estatísticas
  size_t total_updates_;
  size_t successful_updates_;
  
  // =============================================================================
  // CALLBACKS
  // =============================================================================
  
  void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!is_initialized_) {
      // Primeira odometria - inicializar EKF
      roboime_msgs::msg::RobotPose2D initial_pose;
      initial_pose.x = msg->pose.pose.position.x;
      initial_pose.y = msg->pose.pose.position.y;
      
      // Extrair yaw do quaternion
      tf2::Quaternion q;
      tf2::fromMsg(msg->pose.pose.orientation, q);
      double roll, pitch, yaw;
      tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
      initial_pose.theta = yaw;
      
      // Covariância inicial baseada na odometria
      Eigen::Matrix3d initial_cov = Eigen::Matrix3d::Identity() * 0.5;
      if (msg->pose.covariance[0] > 0) {
        initial_cov(0, 0) = msg->pose.covariance[0];   // x
        initial_cov(1, 1) = msg->pose.covariance[7];   // y
        initial_cov(2, 2) = msg->pose.covariance[35];  // yaw
      }
      
      ekf_->initialize(initial_pose, initial_cov);
      is_initialized_ = true;
      last_update_time_ = this->now();
      
      RCLCPP_INFO(this->get_logger(), 
        "EKF inicializado com pose: (%.2f, %.2f, %.2f°)", 
        initial_pose.x, initial_pose.y, initial_pose.theta * 180.0 / M_PI);
    } else {
      // Processar delta da odometria
      if (previous_odometry_) {
        double dt = (msg->header.stamp.sec - previous_odometry_->header.stamp.sec) +
                   (msg->header.stamp.nanosec - previous_odometry_->header.stamp.nanosec) * 1e-9;
        
        if (dt > 0 && dt < 1.0) {  // Filtrar deltas muito grandes
          roboime_msgs::msg::RobotPose2D delta;
          delta.x = msg->pose.pose.position.x - previous_odometry_->pose.pose.position.x;
          delta.y = msg->pose.pose.position.y - previous_odometry_->pose.pose.position.y;
          
          // Calcular delta de orientação
          tf2::Quaternion q1, q2;
          tf2::fromMsg(msg->pose.pose.orientation, q1);
          tf2::fromMsg(previous_odometry_->pose.pose.orientation, q2);
          
          double yaw1 = tf2::getYaw(q1);
          double yaw2 = tf2::getYaw(q2);
          delta.theta = normalize_angle(yaw1 - yaw2);
          
          ekf_->predict_with_odometry(delta, dt);
          total_updates_++;
        }
      }
    }
    
    previous_odometry_ = last_odometry_;
    last_odometry_ = msg;
  }
  
  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    if (is_initialized_) {
      ekf_->update_with_imu(*msg);
      successful_updates_++;
    }
  }
  
  void landmarks_callback(const roboime_msgs::msg::LandmarkArray::SharedPtr msg)
  {
    if (!is_initialized_ || msg->landmarks.empty()) {
      return;
    }
    
    // Converter landmarks ROS para formato interno
    std::vector<roboime_navigation::LandmarkMeasurement> measurements;
    
    for (const auto& landmark_msg : msg->landmarks) {
      // Converter tipo de landmark
      roboime_navigation::LandmarkMeasurement::Type type;
      
      if (landmark_msg.type == roboime_msgs::msg::FieldLandmark::CENTER_CIRCLE) {
        type = roboime_navigation::LandmarkMeasurement::CENTER_CIRCLE;
      } else if (landmark_msg.type == roboime_msgs::msg::FieldLandmark::PENALTY_MARK) {
        type = roboime_navigation::LandmarkMeasurement::PENALTY_MARK;
      } else if (landmark_msg.type == roboime_msgs::msg::FieldLandmark::GOAL_POST) {
        type = roboime_navigation::LandmarkMeasurement::GOAL_POST;
      } else {
        type = roboime_navigation::LandmarkMeasurement::LINE_INTERSECTION;
      }
      
      // Calcular distância e bearing
      double distance = std::sqrt(
        landmark_msg.position_relative.x * landmark_msg.position_relative.x +
        landmark_msg.position_relative.y * landmark_msg.position_relative.y
      );
      
      double bearing = std::atan2(
        landmark_msg.position_relative.y,
        landmark_msg.position_relative.x
      );
      
      // Posição conhecida no mapa
      Eigen::Vector2d map_pos(
        landmark_msg.position_absolute.x,
        landmark_msg.position_absolute.y
      );
      
      measurements.emplace_back(
        type, distance, bearing, landmark_msg.confidence,
        map_pos, std::chrono::steady_clock::now()
      );
    }
    
    // Atualizar EKF com landmarks
    ekf_->update_with_landmarks(measurements);
    successful_updates_++;
    
    RCLCPP_DEBUG(this->get_logger(), 
      "EKF atualizado com %zu landmarks", measurements.size());
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
    
    RCLCPP_INFO(this->get_logger(), "Resetando EKF...");
    
    ekf_->reset();
    is_initialized_ = false;
    total_updates_ = 0;
    successful_updates_ = 0;
    
    RCLCPP_INFO(this->get_logger(), "EKF resetado com sucesso");
  }
  
  void initialize_callback(
    const std::shared_ptr<roboime_msgs::srv::InitializeLocalization::Request> request,
    std::shared_ptr<roboime_msgs::srv::InitializeLocalization::Response> response)
  {
    RCLCPP_INFO(this->get_logger(), 
      "Inicializando EKF com pose: (%.2f, %.2f, %.2f°)",
      request->initial_pose.x, request->initial_pose.y, 
      request->initial_pose.theta * 180.0 / M_PI);
    
    try {
      // Matriz de covariância da requisição
      Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 0.5;
      if (request->covariance.size() >= 9) {
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            cov(i, j) = request->covariance[i * 3 + j];
          }
        }
      }
      
      ekf_->initialize(request->initial_pose, cov);
      is_initialized_ = true;
      last_update_time_ = this->now();
      
      response->success = true;
      response->message = "EKF inicializado com sucesso";
      // Métricas de inicialização
      response->confidence = 0.3;  // confiança inicial estimada
      response->convergence_time = 8.0; // estimativa simples
      
    } catch (const std::exception& e) {
      response->success = false;
      response->message = std::string("Erro na inicialização: ") + e.what();
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
    }
  }
  
  // =============================================================================
  // PUBLICAÇÃO
  // =============================================================================
  
  void publish_localization()
  {
    if (!is_initialized_) {
      return;
    }
    
    try {
      // Pose simples
      auto pose = ekf_->get_estimated_pose();
      pose_pub_->publish(pose);
      
      // Pose com covariância
      auto pose_cov = ekf_->get_pose_with_covariance();
      pose_cov.header.stamp = this->now();
      pose_cov.header.frame_id = "map";
      pose_cov_pub_->publish(pose_cov);
      
      // Confiança (inverso da incerteza)
      double uncertainty = ekf_->get_localization_uncertainty();
      double confidence = 1.0 / (1.0 + uncertainty);
      
      std_msgs::msg::Float64 confidence_msg;
      confidence_msg.data = confidence;
      confidence_pub_->publish(confidence_msg);
      
      // Status
      std_msgs::msg::String status_msg;
      if (ekf_->is_localization_reliable()) {
        status_msg.data = "RELIABLE";
      } else if (uncertainty < 2.0) {
        status_msg.data = "TRACKING";
      } else {
        status_msg.data = "UNCERTAIN";
      }
      status_pub_->publish(status_msg);
      
      // TF
      if (this->get_parameter("enable_tf_broadcast").as_bool()) {
        publish_tf_transform(pose);
      }
      
      // Log periódico
      static int log_counter = 0;
      if (++log_counter % 100 == 0) {  // A cada 5 segundos (20Hz)
        RCLCPP_INFO(this->get_logger(),
          "EKF: Pose=(%.2f,%.2f,%.1f°) Confidence=%.2f Updates=%zu/%zu",
          pose.x, pose.y, pose.theta * 180.0 / M_PI,
          confidence, successful_updates_, total_updates_);
      }
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Erro na publicação: %s", e.what());
    }
  }
  
  void publish_tf_transform(const geometry_msgs::msg::Pose2D& pose)
  {
    geometry_msgs::msg::TransformStamped transform;
    
    transform.header.stamp = this->now();
    transform.header.frame_id = "map";
    transform.child_frame_id = "base_link_ekf";
    
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
    auto node = std::make_shared<EKFLocalizationNode>();
    
    RCLCPP_INFO(node->get_logger(), "EKF Localization Node iniciado");
    
    rclcpp::spin(node);
    
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("ekf_localization_node"), 
      "Erro fatal: %s", e.what());
    return 1;
  }
  
  rclcpp::shutdown();
  return 0;
} 