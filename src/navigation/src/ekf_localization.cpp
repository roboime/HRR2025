#include "ekf_localization.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <algorithm>
#include <limits>

namespace roboime_navigation
{

EKFLocalization::EKFLocalization(double field_length, double field_width)
  : field_length_(field_length)
  , field_width_(field_width)
  , measurement_noise_std_(0.1)
  , imu_noise_std_(0.02)
  , is_initialized_(false)
{
  // Inicializar estado e covariância
  x_ = Eigen::Vector3d::Zero();
  P_ = Eigen::Matrix3d::Identity() * 1.0;  // Incerteza inicial alta
  
  // Matriz de ruído do processo Q
  Q_ = Eigen::Matrix3d::Zero();
  Q_(0, 0) = 0.01;  // Ruído na posição x
  Q_(1, 1) = 0.01;  // Ruído na posição y
  Q_(2, 2) = 0.05;  // Ruído na orientação
  
  // Inicializar landmarks padrão do campo RoboCup
  initialize_default_field_landmarks();
}

void EKFLocalization::initialize(
  const roboime_msgs::msg::RobotPose2D& initial_pose,
  const Eigen::Matrix3d& initial_covariance)
{
  x_(0) = initial_pose.x;
  x_(1) = initial_pose.y;
  x_(2) = initial_pose.theta;
  
  P_ = initial_covariance;
  is_initialized_ = true;
  
  // Limpar histórico de medições
  while (!recent_measurements_.empty()) {
    recent_measurements_.pop();
  }
}

void EKFLocalization::initialize_multi_hypothesis(
  const std::vector<roboime_msgs::msg::RobotPose2D>& hypothesis_poses,
  const std::vector<Eigen::Matrix3d>& covariances)
{
  if (hypothesis_poses.empty()) return;
  
  // Por simplicidade, usar primeira hipótese
  // Em implementação avançada, manteria múltiplos filtros
  initialize(hypothesis_poses[0], covariances.empty() ? Eigen::Matrix3d::Identity() : covariances[0]);
}

void EKFLocalization::predict_with_odometry(
  const roboime_msgs::msg::RobotPose2D& odometry_delta,
  double dt)
{
  if (!is_initialized_) return;
  
  // Modelo de movimento baseado em odometria
  double dx = odometry_delta.x;
  double dy = odometry_delta.y;
  double dtheta = odometry_delta.theta;
  
  // Transformar delta para frame global
  double cos_theta = std::cos(x_(2));
  double sin_theta = std::sin(x_(2));
  
  // Predição do estado
  Eigen::Vector3d x_pred;
  x_pred(0) = x_(0) + cos_theta * dx - sin_theta * dy;
  x_pred(1) = x_(1) + sin_theta * dx + cos_theta * dy;
  x_pred(2) = normalize_angle(x_(2) + dtheta);
  
  // Jacobiano do modelo de movimento
  Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
  F(0, 2) = -sin_theta * dx - cos_theta * dy;
  F(1, 2) = cos_theta * dx - sin_theta * dy;
  
  // Predição da covariância
  Eigen::Matrix3d P_pred = F * P_ * F.transpose() + Q_ * dt;
  
  // Atualizar estado
  x_ = x_pred;
  P_ = P_pred;
  
  // Verificar se pose está dentro do campo
  if (!is_pose_within_field(x_)) {
    // Aplicar constraints do campo
    x_(0) = std::max(-field_length_/2.0, std::min(field_length_/2.0, x_(0)));
    x_(1) = std::max(-field_width_/2.0, std::min(field_width_/2.0, x_(1)));
  }
}

void EKFLocalization::predict_with_motion_model(
  double linear_velocity,
  double angular_velocity,
  double dt)
{
  if (!is_initialized_) return;
  
  // Modelo cinemático
  Eigen::Vector3d x_pred = motion_model(x_, linear_velocity, angular_velocity, dt);
  
  // Jacobiano
  Eigen::Matrix3d F = motion_jacobian(dt, linear_velocity, angular_velocity);
  
  // Predição da covariância
  Eigen::Matrix3d P_pred = F * P_ * F.transpose() + Q_ * dt;
  
  // Atualizar
  x_ = x_pred;
  P_ = P_pred;
}

void EKFLocalization::update_with_imu(const sensor_msgs::msg::Imu& imu_data)
{
  if (!is_initialized_) return;
  
  // Extrair orientação do quaternion
  double siny_cosp = 2 * (imu_data.orientation.w * imu_data.orientation.z + 
                         imu_data.orientation.x * imu_data.orientation.y);
  double cosy_cosp = 1 - 2 * (imu_data.orientation.y * imu_data.orientation.y + 
                              imu_data.orientation.z * imu_data.orientation.z);
  double yaw_measured = std::atan2(siny_cosp, cosy_cosp);
  
  // Inovação
  double innovation = normalize_angle(yaw_measured - x_(2));
  
  // Matriz H para orientação (1x3)
  Eigen::Matrix<double, 1, 3> H = Eigen::Matrix<double, 1, 3>::Zero();
  H(0, 2) = 1.0;
  
  // Covariância da inovação
  double S = H * P_ * H.transpose() + imu_noise_std_ * imu_noise_std_;
  
  // Ganho de Kalman
  Eigen::Vector3d K = P_ * H.transpose() / S;
  
  // Atualização
  x_ = x_ + K * innovation;
  x_(2) = normalize_angle(x_(2));
  
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  P_ = (I - K * H) * P_;
}

void EKFLocalization::update_with_landmark(const LandmarkMeasurement& landmark)
{
  if (!is_initialized_) return;
  
  // Validar medição
  if (!validate_measurement(landmark)) {
    return;
  }
  
  // Encontrar landmark correspondente no mapa
  Eigen::Vector2d landmark_pos = find_corresponding_landmark(landmark);
  
  // Modelo de observação esperada
  Eigen::Vector2d z_expected = observation_model(x_, landmark_pos);
  
  // Medição observada
  Eigen::Vector2d z_observed(landmark.range, landmark.bearing);
  
  // Inovação
  Eigen::Vector2d innovation = z_observed - z_expected;
  innovation(1) = normalize_angle(innovation(1));  // Normalizar bearing
  
  // Jacobiano do modelo de observação
  Eigen::Matrix<double, 2, 3> H = observation_jacobian(x_, landmark_pos);
  
  // Covariância da inovação
  Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * measurement_noise_std_ * measurement_noise_std_;
  R(0, 0) *= landmark.confidence;  // Ajustar por confiança
  R(1, 1) *= landmark.confidence;
  
  Eigen::Matrix2d S = H * P_ * H.transpose() + R;
  
  // Ganho de Kalman
  Eigen::Matrix<double, 3, 2> K = P_ * H.transpose() * S.inverse();
  
  // Atualização
  x_ = x_ + K * innovation;
  x_(2) = normalize_angle(x_(2));
  
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  P_ = (I - K * H) * P_;
  
  // Adicionar ao histórico
  recent_measurements_.push(landmark);
  if (recent_measurements_.size() > MAX_MEASUREMENT_HISTORY) {
    recent_measurements_.pop();
  }
}

void EKFLocalization::update_with_landmarks(const std::vector<LandmarkMeasurement>& landmarks)
{
  // Processar landmarks em ordem de confiança
  std::vector<LandmarkMeasurement> sorted_landmarks = landmarks;
  std::sort(sorted_landmarks.begin(), sorted_landmarks.end(),
           [](const LandmarkMeasurement& a, const LandmarkMeasurement& b) {
             return a.confidence > b.confidence;
           });
  
  for (const auto& landmark : sorted_landmarks) {
    update_with_landmark(landmark);
  }
}

roboime_msgs::msg::RobotPose2D EKFLocalization::get_estimated_pose() const
{
  roboime_msgs::msg::RobotPose2D pose;
  pose.x = x_(0);
  pose.y = x_(1);
  pose.theta = x_(2);
  return pose;
}

geometry_msgs::msg::PoseWithCovarianceStamped EKFLocalization::get_pose_with_covariance() const
{
  geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
  
  // Pose
  pose_msg.pose.pose.position.x = x_(0);
  pose_msg.pose.pose.position.y = x_(1);
  pose_msg.pose.pose.position.z = 0.0;
  
  // Quaternion da orientação
  tf2::Quaternion q;
  q.setRPY(0, 0, x_(2));
  pose_msg.pose.pose.orientation = tf2::toMsg(q);
  
  // Covariância (6x6, mas usamos apenas 3x3)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int idx = i * 6 + j;
      if (i < 2 && j < 2) {
        pose_msg.pose.covariance[idx] = P_(i, j);
      } else if (i == 2 && j == 2) {
        pose_msg.pose.covariance[35] = P_(2, 2);  // Posição (5,5) para yaw
      }
    }
  }
  
  return pose_msg;
}

double EKFLocalization::get_localization_uncertainty() const
{
  // Calcular traço da matriz de covariância
  return P_.trace();
}

bool EKFLocalization::is_localization_reliable(double uncertainty_threshold) const
{
  return is_initialized_ && get_localization_uncertainty() < uncertainty_threshold;
}

bool EKFLocalization::validate_measurement(
  const LandmarkMeasurement& landmark,
  double innovation_threshold) const
{
  if (!is_initialized_) return false;
  
  // Verificar confiança mínima
  if (landmark.confidence < 0.3) return false;
  
  // Verificar distância razoável
  if (landmark.range < 0.1 || landmark.range > 10.0) return false;
  
  // Teste de Mahalanobis distance
  auto [innovation, S] = calculate_innovation(landmark);
  double mahal_dist = mahalanobis_distance(innovation, S);
  
  return mahal_dist < innovation_threshold;
}

void EKFLocalization::set_field_landmarks(
  const std::map<LandmarkMeasurement::Type, std::vector<Eigen::Vector2d>>& landmarks)
{
  field_landmarks_ = landmarks;
}

void EKFLocalization::update_noise_parameters(
  const Eigen::Matrix3d& process_noise,
  double measurement_noise_std,
  double imu_orientation_noise_std)
{
  Q_ = process_noise;
  measurement_noise_std_ = measurement_noise_std;
  imu_noise_std_ = imu_orientation_noise_std;
}

void EKFLocalization::reset()
{
  is_initialized_ = false;
  x_ = Eigen::Vector3d::Zero();
  P_ = Eigen::Matrix3d::Identity() * 1.0;
  
  while (!recent_measurements_.empty()) {
    recent_measurements_.pop();
  }
}

std::pair<Eigen::Vector2d, Eigen::Matrix2d> EKFLocalization::calculate_innovation(
  const LandmarkMeasurement& landmark) const
{
  Eigen::Vector2d landmark_pos = find_corresponding_landmark(landmark);
  Eigen::Vector2d z_expected = observation_model(x_, landmark_pos);
  Eigen::Vector2d z_observed(landmark.range, landmark.bearing);
  
  Eigen::Vector2d innovation = z_observed - z_expected;
  innovation(1) = normalize_angle(innovation(1));
  
  Eigen::Matrix<double, 2, 3> H = observation_jacobian(x_, landmark_pos);
  Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * measurement_noise_std_ * measurement_noise_std_;
  Eigen::Matrix2d S = H * P_ * H.transpose() + R;
  
  return {innovation, S};
}

// Métodos privados

Eigen::Matrix3d EKFLocalization::motion_jacobian(
  double dt,
  double linear_vel,
  double angular_vel) const
{
  Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
  
  if (std::abs(angular_vel) < 1e-6) {
    // Movimento retilíneo
    F(0, 2) = -linear_vel * std::sin(x_(2)) * dt;
    F(1, 2) = linear_vel * std::cos(x_(2)) * dt;
  } else {
    // Movimento curvilíneo
    double theta_new = x_(2) + angular_vel * dt;
    F(0, 2) = linear_vel / angular_vel * (std::cos(theta_new) - std::cos(x_(2)));
    F(1, 2) = linear_vel / angular_vel * (std::sin(theta_new) - std::sin(x_(2)));
  }
  
  return F;
}

Eigen::Vector3d EKFLocalization::motion_model(
  const Eigen::Vector3d& state,
  double linear_vel,
  double angular_vel,
  double dt) const
{
  Eigen::Vector3d new_state;
  
  if (std::abs(angular_vel) < 1e-6) {
    // Movimento retilíneo
    new_state(0) = state(0) + linear_vel * std::cos(state(2)) * dt;
    new_state(1) = state(1) + linear_vel * std::sin(state(2)) * dt;
    new_state(2) = state(2);
  } else {
    // Movimento curvilíneo
    double theta_new = state(2) + angular_vel * dt;
    new_state(0) = state(0) + linear_vel / angular_vel * (std::sin(theta_new) - std::sin(state(2)));
    new_state(1) = state(1) - linear_vel / angular_vel * (std::cos(theta_new) - std::cos(state(2)));
    new_state(2) = normalize_angle(theta_new);
  }
  
  return new_state;
}

Eigen::Vector2d EKFLocalization::observation_model(
  const Eigen::Vector3d& state,
  const Eigen::Vector2d& landmark_position) const
{
  double dx = landmark_position(0) - state(0);
  double dy = landmark_position(1) - state(1);
  
  double range = std::sqrt(dx * dx + dy * dy);
  double bearing = normalize_angle(std::atan2(dy, dx) - state(2));
  
  return Eigen::Vector2d(range, bearing);
}

Eigen::Matrix<double, 2, 3> EKFLocalization::observation_jacobian(
  const Eigen::Vector3d& state,
  const Eigen::Vector2d& landmark_position) const
{
  double dx = landmark_position(0) - state(0);
  double dy = landmark_position(1) - state(1);
  double range_sq = dx * dx + dy * dy;
  double range = std::sqrt(range_sq);
  
  Eigen::Matrix<double, 2, 3> H;
  
  // Jacobiano para range
  H(0, 0) = -dx / range;
  H(0, 1) = -dy / range;
  H(0, 2) = 0.0;
  
  // Jacobiano para bearing
  H(1, 0) = dy / range_sq;
  H(1, 1) = -dx / range_sq;
  H(1, 2) = -1.0;
  
  return H;
}

Eigen::Vector2d EKFLocalization::find_corresponding_landmark(
  const LandmarkMeasurement& measurement) const
{
  auto it = field_landmarks_.find(measurement.type);
  if (it == field_landmarks_.end() || it->second.empty()) {
    // Retornar posição default se não encontrado
    return Eigen::Vector2d(0, 0);
  }
  
  // Encontrar landmark mais próximo da medição
  double min_distance = std::numeric_limits<double>::max();
  Eigen::Vector2d best_landmark = it->second[0];
  
  for (const auto& landmark_pos : it->second) {
    Eigen::Vector2d expected = observation_model(x_, landmark_pos);
    double distance = std::abs(expected(0) - measurement.range);
    
    if (distance < min_distance) {
      min_distance = distance;
      best_landmark = landmark_pos;
    }
  }
  
  return best_landmark;
}

double EKFLocalization::normalize_angle(double angle) const
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

bool EKFLocalization::is_pose_within_field(const Eigen::Vector3d& pose) const
{
  return (pose(0) >= -field_length_/2.0 && pose(0) <= field_length_/2.0 &&
          pose(1) >= -field_width_/2.0 && pose(1) <= field_width_/2.0);
}

double EKFLocalization::mahalanobis_distance(
  const Eigen::Vector2d& innovation,
  const Eigen::Matrix2d& covariance) const
{
  return std::sqrt(innovation.transpose() * covariance.inverse() * innovation);
}

void EKFLocalization::initialize_default_field_landmarks()
{
  field_landmarks_.clear();
  
  // Círculo central
  field_landmarks_[LandmarkMeasurement::CENTER_CIRCLE] = {
    Eigen::Vector2d(0.0, 0.0)
  };
  
  // Marcas de penalty
  field_landmarks_[LandmarkMeasurement::PENALTY_MARK] = {
    Eigen::Vector2d(-3.0, 0.0),  // Esquerda
    Eigen::Vector2d(3.0, 0.0)    // Direita
  };
  
  // Gols
  field_landmarks_[LandmarkMeasurement::GOAL_POST] = {
    Eigen::Vector2d(-4.5, 0.0),  // Gol esquerdo
    Eigen::Vector2d(4.5, 0.0)    // Gol direito
  };
  
  // Interseções de linhas (cantos)
  field_landmarks_[LandmarkMeasurement::LINE_INTERSECTION] = {
    Eigen::Vector2d(-4.5, -3.0), Eigen::Vector2d(-4.5, 3.0),
    Eigen::Vector2d(4.5, -3.0),  Eigen::Vector2d(4.5, 3.0),
    // Cantos das áreas
    Eigen::Vector2d(-2.85, -1.95), Eigen::Vector2d(-2.85, 1.95),
    Eigen::Vector2d(2.85, -1.95),  Eigen::Vector2d(2.85, 1.95)
  };
}

}  // namespace roboime_navigation 