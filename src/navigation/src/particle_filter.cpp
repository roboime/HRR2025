#include "particle_filter.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <tf2/LinearMath/Quaternion.h>
#if __has_include(<tf2_geometry_msgs/tf2_geometry_msgs.hpp>)
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif

namespace roboime_navigation
{

ParticleFilter::ParticleFilter(size_t num_particles, double field_length, double field_width)
  : num_particles_(num_particles)
  , field_length_(field_length)
  , field_width_(field_width)
  , resample_threshold_(num_particles / 2.0)  // Resample when Neff < 50%
  , motion_noise_std_(0.05, 0.05, 0.1)      // [x, y, theta] std dev
  , measurement_noise_std_(0.1)               // Range/bearing noise
  , gen_(rd_())
  , normal_dist_(0.0, 1.0)
  , uniform_dist_(0.0, 1.0)
{
  particles_.reserve(num_particles_);
  pose_history_.reserve(100);
  confidence_history_.reserve(100);
  
  // Inicializar landmarks padrão do campo RoboCup
  initialize_default_field_landmarks();
}

void ParticleFilter::initialize_global_localization(
  const roboime_msgs::msg::RobotPose2D* initial_pose,
  const Eigen::Matrix3d& covariance)
{
  particles_.clear();
  particles_.reserve(num_particles_);
  
  if (initial_pose == nullptr) {
    // Distribuição uniforme por todo o campo
    for (size_t i = 0; i < num_particles_; ++i) {
      Particle p;
      p.x = uniform_dist_(gen_) * field_length_ - field_length_ / 2.0;
      p.y = uniform_dist_(gen_) * field_width_ - field_width_ / 2.0;
      p.theta = uniform_dist_(gen_) * 2.0 * M_PI - M_PI;
      p.weight = 1.0 / num_particles_;
      particles_.push_back(p);
    }
  } else {
    // Distribuição gaussiana ao redor da pose inicial
    std::normal_distribution<double> x_dist(initial_pose->x, std::sqrt(covariance(0, 0)));
    std::normal_distribution<double> y_dist(initial_pose->y, std::sqrt(covariance(1, 1)));
    std::normal_distribution<double> theta_dist(initial_pose->theta, std::sqrt(covariance(2, 2)));
    
    for (size_t i = 0; i < num_particles_; ++i) {
      Particle p;
      p.x = x_dist(gen_);
      p.y = y_dist(gen_);
      p.theta = normalize_angle(theta_dist(gen_));
      p.weight = 1.0 / num_particles_;
      
      // Verificar se está dentro do campo
      if (!is_pose_valid(p)) {
        --i;  // Regenerar partícula inválida
        continue;
      }
      
      particles_.push_back(p);
    }
  }
}

void ParticleFilter::initialize_with_team_side(
  const std::string& team_side,
  const roboime_msgs::msg::RobotPose2D& initial_pose)
{
  particles_.clear();
  particles_.reserve(num_particles_);
  
  // Determinar limites baseado no lado do time
  double x_min, x_max;
  if (team_side == "left") {
    x_min = -field_length_ / 2.0;
    x_max = 0.0;
  } else {  // "right"
    x_min = 0.0;
    x_max = field_length_ / 2.0;
  }
  
  // Distribuir partículas no lado correto
  for (size_t i = 0; i < num_particles_; ++i) {
    Particle p;
    
    // Concentrar ao redor da pose inicial, mas restringir ao lado
    p.x = std::max(x_min, std::min(x_max, 
      initial_pose.x + add_gaussian_noise(0.0, 1.0)));
    p.y = initial_pose.y + add_gaussian_noise(0.0, 1.0);
    p.theta = normalize_angle(initial_pose.theta + add_gaussian_noise(0.0, 0.5));
    p.weight = 1.0 / num_particles_;
    
    if (!is_pose_valid(p)) {
      --i;
      continue;
    }
    
    particles_.push_back(p);
  }
}

void ParticleFilter::predict(
  const roboime_msgs::msg::RobotPose2D& odometry_delta,
  const sensor_msgs::msg::Imu& imu_data,
  double dt)
{
  // Aplicar modelo de movimento a cada partícula
  for (auto& particle : particles_) {
    apply_motion_model(particle, odometry_delta, dt);
  }
  
  // Atualizar histórico
  if (pose_history_.size() >= 100) {
    pose_history_.erase(pose_history_.begin());
  }
  pose_history_.push_back(get_estimated_pose());
}

void ParticleFilter::apply_motion_model(
  Particle& particle,
  const roboime_msgs::msg::RobotPose2D& odometry_delta,
  double dt)
{
  // Modelo de movimento com ruído
  double noisy_dx = odometry_delta.x + add_gaussian_noise(0.0, motion_noise_std_[0]);
  double noisy_dy = odometry_delta.y + add_gaussian_noise(0.0, motion_noise_std_[1]);
  double noisy_dtheta = odometry_delta.theta + add_gaussian_noise(0.0, motion_noise_std_[2]);
  
  // Aplicar movimento no referencial local da partícula
  double cos_theta = std::cos(particle.theta);
  double sin_theta = std::sin(particle.theta);
  
  particle.x += cos_theta * noisy_dx - sin_theta * noisy_dy;
  particle.y += sin_theta * noisy_dx + cos_theta * noisy_dy;
  particle.theta = normalize_angle(particle.theta + noisy_dtheta);
  
  // Verificar limites do campo
  if (!is_pose_valid(particle)) {
    // Rejeitar movimento inválido (partícula fica no lugar)
    particle.x -= cos_theta * noisy_dx - sin_theta * noisy_dy;
    particle.y -= sin_theta * noisy_dx + cos_theta * noisy_dy;
    particle.theta = normalize_angle(particle.theta - noisy_dtheta);
  }
}

void ParticleFilter::update(const std::vector<Landmark>& landmarks)
{
  if (landmarks.empty()) {
    return;
  }
  
  // Calcular likelihood para cada partícula
  for (auto& particle : particles_) {
    double total_likelihood = 1.0;
    
    for (const auto& landmark : landmarks) {
      double likelihood = calculate_measurement_likelihood(particle, landmark);
      total_likelihood *= likelihood;
    }
    
    particle.weight *= total_likelihood;
  }
  
  // Normalizar pesos
  normalize_weights();
  
  // Verificar se é necessário reamostrar
  double neff = get_effective_particle_count();
  if (neff < resample_threshold_) {
    resample();
  }
}

double ParticleFilter::calculate_measurement_likelihood(
  const Particle& particle,
  const Landmark& landmark) const
{
  // Encontrar landmarks candidatos no mapa
  auto candidates = find_landmark_candidates(landmark, particle);
  
  if (candidates.empty()) {
    return 0.1;  // Likelihood baixo para landmarks não encontrados
  }
  
  double max_likelihood = 0.0;
  
  // Calcular likelihood para cada candidato e usar o máximo
  for (const auto& candidate_pos : candidates) {
    // Distância esperada
    double expected_distance = std::sqrt(
      std::pow(candidate_pos[0] - particle.x, 2) +
      std::pow(candidate_pos[1] - particle.y, 2)
    );
    
    // Ângulo esperado (bearing)
    double expected_bearing = std::atan2(
      candidate_pos[1] - particle.y,
      candidate_pos[0] - particle.x
    ) - particle.theta;
    expected_bearing = normalize_angle(expected_bearing);
    
    // Calcular likelihood gaussiano
    double distance_error = landmark.distance - expected_distance;
    double bearing_error = normalize_angle(landmark.bearing - expected_bearing);
    
    double distance_likelihood = std::exp(-0.5 * std::pow(distance_error / measurement_noise_std_, 2));
    double bearing_likelihood = std::exp(-0.5 * std::pow(bearing_error / (measurement_noise_std_ * 2), 2));
    
    double combined_likelihood = distance_likelihood * bearing_likelihood * landmark.confidence;
    max_likelihood = std::max(max_likelihood, combined_likelihood);
  }
  
  return max_likelihood;
}

std::vector<Eigen::Vector2d> ParticleFilter::find_landmark_candidates(
  const Landmark& observed_landmark,
  const Particle& particle) const
{
  std::vector<Eigen::Vector2d> candidates;
  
  auto it = field_landmarks_.find(observed_landmark.type);
  if (it != field_landmarks_.end()) {
    for (const auto& landmark_pos : it->second) {
      // Verificar se a distância é razoável
      double distance_to_landmark = std::sqrt(
        std::pow(landmark_pos[0] - particle.x, 2) +
        std::pow(landmark_pos[1] - particle.y, 2)
      );
      
      // Tolerância de ±50% na distância observada
      if (std::abs(distance_to_landmark - observed_landmark.distance) < 
          observed_landmark.distance * 0.5) {
        candidates.push_back(landmark_pos);
      }
    }
  }
  
  return candidates;
}

void ParticleFilter::resample()
{
  std::vector<Particle> new_particles;
  new_particles.reserve(num_particles_);
  
  // Reamostragem sistemática (Systematic Resampling)
  double weight_sum = std::accumulate(particles_.begin(), particles_.end(), 0.0,
    [](double sum, const Particle& p) { return sum + p.weight; });
  
  if (weight_sum == 0.0) {
    // Todas as partículas têm peso zero - reinicializar uniformemente
    initialize_global_localization();
    return;
  }
  
  double step = weight_sum / num_particles_;
  double start = uniform_dist_(gen_) * step;
  
  size_t current_index = 0;
  double cumulative_weight = particles_[0].weight;
  
  for (size_t i = 0; i < num_particles_; ++i) {
    double target = start + i * step;
    
    while (cumulative_weight < target && current_index < particles_.size() - 1) {
      current_index++;
      cumulative_weight += particles_[current_index].weight;
    }
    
    Particle new_particle = particles_[current_index];
    new_particle.weight = 1.0 / num_particles_;
    new_particles.push_back(new_particle);
  }
  
  particles_ = std::move(new_particles);
}

roboime_msgs::msg::RobotPose2D ParticleFilter::get_estimated_pose() const
{
  roboime_msgs::msg::RobotPose2D pose;
  
  if (particles_.empty()) {
    return pose;
  }
  
  // Média ponderada das partículas
  double total_weight = 0.0;
  double weighted_x = 0.0;
  double weighted_y = 0.0;
  double weighted_cos_theta = 0.0;
  double weighted_sin_theta = 0.0;
  
  for (const auto& particle : particles_) {
    weighted_x += particle.weight * particle.x;
    weighted_y += particle.weight * particle.y;
    weighted_cos_theta += particle.weight * std::cos(particle.theta);
    weighted_sin_theta += particle.weight * std::sin(particle.theta);
    total_weight += particle.weight;
  }
  
  if (total_weight > 0.0) {
    pose.x = weighted_x / total_weight;
    pose.y = weighted_y / total_weight;
    pose.theta = std::atan2(weighted_sin_theta / total_weight, 
                           weighted_cos_theta / total_weight);
  }
  
  return pose;
}

geometry_msgs::msg::PoseWithCovarianceStamped ParticleFilter::get_pose_with_covariance() const
{
  geometry_msgs::msg::PoseWithCovarianceStamped msg;
  // Timestamp e frame id ficarão a cargo do chamador
  if (particles_.empty()) {
    return msg;
  }

  // Média ponderada da pose
  auto pose = get_estimated_pose();
  msg.pose.pose.position.x = pose.x;
  msg.pose.pose.position.y = pose.y;
  msg.pose.pose.position.z = 0.0;

  // Orientação em quaternion a partir de theta
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, pose.theta);
  msg.pose.pose.orientation = tf2::toMsg(q);

  // Covariância 3x3 embutida nos índices da 6x6
  // Estimar variâncias a partir da dispersão das partículas
  double total_weight = 0.0;
  double mean_x = 0.0, mean_y = 0.0;
  double mean_cos = 0.0, mean_sin = 0.0;
  for (const auto& p : particles_) {
    total_weight += p.weight;
    mean_x += p.weight * p.x;
    mean_y += p.weight * p.y;
    mean_cos += p.weight * std::cos(p.theta);
    mean_sin += p.weight * std::sin(p.theta);
  }
  if (total_weight <= 0.0) total_weight = 1.0;
  mean_x /= total_weight;
  mean_y /= total_weight;
  double mean_theta = std::atan2(mean_sin / total_weight, mean_cos / total_weight);

  double var_x = 0.0, var_y = 0.0, var_theta = 0.0;
  for (const auto& p : particles_) {
    double dx = p.x - mean_x;
    double dy = p.y - mean_y;
    double dth = std::atan2(std::sin(p.theta - mean_theta), std::cos(p.theta - mean_theta));
    var_x += p.weight * dx * dx;
    var_y += p.weight * dy * dy;
    var_theta += p.weight * dth * dth;
  }
  var_x /= total_weight;
  var_y /= total_weight;
  var_theta /= total_weight;

  // Preencher matriz 6x6 nos campos relevantes: xx, yy e yaw
  for (int i = 0; i < 36; ++i) msg.pose.covariance[i] = 0.0;
  msg.pose.covariance[0] = var_x;    // cov(xx)
  msg.pose.covariance[7] = var_y;    // cov(yy)
  msg.pose.covariance[35] = var_theta; // cov(yaw,yaw)

  return msg;
}

double ParticleFilter::get_localization_confidence() const
{
  if (particles_.empty()) {
    return 0.0;
  }
  
  // Calcular confiança baseada na concentração das partículas
  auto estimated_pose = get_estimated_pose();
  
  double variance_sum = 0.0;
  double total_weight = 0.0;
  
  for (const auto& particle : particles_) {
    roboime_msgs::msg::RobotPose2D particle_pose;
    particle_pose.x = particle.x;
    particle_pose.y = particle.y;
    particle_pose.theta = particle.theta;
    double distance = pose_distance(estimated_pose, particle_pose);
    variance_sum += particle.weight * distance * distance;
    total_weight += particle.weight;
  }
  
  if (total_weight == 0.0) {
    return 0.0;
  }
  
  double variance = variance_sum / total_weight;
  
  // Converter variância em confiança (0-1)
  double confidence = std::exp(-variance);
  return std::min(1.0, std::max(0.0, confidence));
}

double ParticleFilter::get_effective_particle_count() const
{
  double weight_sum_squared = 0.0;
  
  for (const auto& particle : particles_) {
    weight_sum_squared += particle.weight * particle.weight;
  }
  
  if (weight_sum_squared == 0.0) {
    return 0.0;
  }
  
  return 1.0 / weight_sum_squared;
}

bool ParticleFilter::detect_kidnapping() const
{
  if (pose_history_.size() < 10 || confidence_history_.size() < 10) {
    return false;
  }
  
  // Verificar se confiança caiu drasticamente
  double recent_confidence = std::accumulate(
    confidence_history_.end() - 5, confidence_history_.end(), 0.0) / 5.0;
  double older_confidence = std::accumulate(
    confidence_history_.end() - 10, confidence_history_.end() - 5, 0.0) / 5.0;
  
  if (recent_confidence < 0.3 && older_confidence > 0.7) {
    return true;
  }
  
  // Verificar movimento impossível
  if (pose_history_.size() >= 2) {
    auto recent_pose = pose_history_.back();
    auto previous_pose = pose_history_[pose_history_.size() - 2];
    
    double movement = pose_distance(recent_pose, previous_pose);
    if (movement > 2.0) {  // Movimento > 2m entre frames
      return true;
    }
  }
  
  return false;
}

void ParticleFilter::recover_from_kidnapping()
{
  // Redistribuir 70% das partículas globalmente, manter 30% das melhores
  size_t num_keep = static_cast<size_t>(num_particles_ * 0.3);
  size_t num_redistribute = num_particles_ - num_keep;
  
  // Ordenar partículas por peso
  std::sort(particles_.begin(), particles_.end(),
    [](const Particle& a, const Particle& b) {
      return a.weight > b.weight;
    });
  
  // Manter as melhores partículas
  particles_.resize(num_keep);
  
  // Adicionar partículas redistribuídas
  for (size_t i = 0; i < num_redistribute; ++i) {
    Particle p;
    p.x = uniform_dist_(gen_) * field_length_ - field_length_ / 2.0;
    p.y = uniform_dist_(gen_) * field_width_ - field_width_ / 2.0;
    p.theta = uniform_dist_(gen_) * 2.0 * M_PI - M_PI;
    p.weight = 1.0 / num_particles_;
    
    if (is_pose_valid(p)) {
      particles_.push_back(p);
    } else {
      --i;  // Regenerar se inválida
    }
  }
}

void ParticleFilter::set_field_landmarks(
  const std::map<Landmark::Type, std::vector<Eigen::Vector2d>>& field_landmarks)
{
  field_landmarks_ = field_landmarks;
}

void ParticleFilter::normalize_weights()
{
  double weight_sum = std::accumulate(particles_.begin(), particles_.end(), 0.0,
    [](double sum, const Particle& p) { return sum + p.weight; });
  
  if (weight_sum > 0.0) {
    for (auto& particle : particles_) {
      particle.weight /= weight_sum;
    }
  }
}

double ParticleFilter::add_gaussian_noise(double mean, double std_dev) const
{
  return mean + std_dev * normal_dist_(gen_);
}

bool ParticleFilter::is_pose_valid(const Particle& particle) const
{
  return (particle.x >= -field_length_ / 2.0 && particle.x <= field_length_ / 2.0 &&
          particle.y >= -field_width_ / 2.0 && particle.y <= field_width_ / 2.0);
}

double ParticleFilter::pose_distance(
  const roboime_msgs::msg::RobotPose2D& pose1,
  const roboime_msgs::msg::RobotPose2D& pose2) const
{
  double dx = pose1.x - pose2.x;
  double dy = pose1.y - pose2.y;
  double dtheta = normalize_angle(pose1.theta - pose2.theta);
  
  // Distância euclidiana + componente angular
  return std::sqrt(dx * dx + dy * dy + 0.5 * dtheta * dtheta);
}

double ParticleFilter::normalize_angle(double angle) const
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

void ParticleFilter::initialize_default_field_landmarks()
{
  // Landmarks simplificados do sistema YOLOv8 (4 tipos para localização)
  field_landmarks_.clear();
  
  // 1. Círculo central (1 landmark - referência central absoluta)
  field_landmarks_[Landmark::CENTER_CIRCLE] = {
    Eigen::Vector2d(0.0, 0.0)  // Centro exato do campo
  };
  
  // 2. Marcas de penalty (2 landmarks - coordenadas precisas)
  field_landmarks_[Landmark::PENALTY_MARK] = {
    Eigen::Vector2d(-3.0, 0.0),  // Marca penalty esquerda (1.5m do gol)
    Eigen::Vector2d(3.0, 0.0)    // Marca penalty direita (1.5m do gol)
  };
  
  // 3. Postes de gol (4 landmarks - dois por gol)
  field_landmarks_[Landmark::GOAL_POST] = {
    Eigen::Vector2d(-4.5, 1.3),   // Poste superior gol esquerdo
    Eigen::Vector2d(-4.5, -1.3),  // Poste inferior gol esquerdo
    Eigen::Vector2d(4.5, 1.3),    // Poste superior gol direito
    Eigen::Vector2d(4.5, -1.3)    // Poste inferior gol direito
  };
  
  // 4. Cantos do campo (4 landmarks - landmarks de borda)
  field_landmarks_[Landmark::FIELD_CORNER] = {
    Eigen::Vector2d(-4.5, -3.0),  // Canto inferior esquerdo
    Eigen::Vector2d(-4.5, 3.0),   // Canto superior esquerdo
    Eigen::Vector2d(4.5, -3.0),   // Canto inferior direito
    Eigen::Vector2d(4.5, 3.0)     // Canto superior direito
  };
  
  // 5. Cantos da área de penalty (8 landmarks - landmarks internos)
  // Área esquerda (4 pontos)
  field_landmarks_[Landmark::GOAL_AREA_CORNER] = {
    // Área esquerda
    Eigen::Vector2d(-2.85, -1.95),  // Inferior direito área esquerda
    Eigen::Vector2d(-2.85, 1.95),   // Superior direito área esquerda
    Eigen::Vector2d(-4.5, -1.95),   // Inferior esquerdo área esquerda
    Eigen::Vector2d(-4.5, 1.95),    // Superior esquerdo área esquerda
    
    // Área direita
    Eigen::Vector2d(2.85, -1.95),   // Inferior esquerdo área direita
    Eigen::Vector2d(2.85, 1.95),    // Superior esquerdo área direita
    Eigen::Vector2d(4.5, -1.95),    // Inferior direito área direita
    Eigen::Vector2d(4.5, 1.95)      // Superior direito área direita
  };
}

}  // namespace roboime_navigation 