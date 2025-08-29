#include "navigation_coordinator.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace roboime_navigation
{

NavigationCoordinator::NavigationCoordinator(
  double field_length,
  double field_width,
  uint32_t robot_id,
  const std::string& team_name)
  : field_length_(field_length)
  , field_width_(field_width)
  , robot_id_(robot_id)
  , team_name_(team_name)
  , current_state_(LocalizationState::UNINITIALIZED)
  , current_mode_(LocalizationMode::INITIALIZATION_MODE)
  , last_good_localization_(std::chrono::steady_clock::now())
{
  // Inicializar algoritmos de localização
  particle_filter_ = std::make_unique<ParticleFilter>(500, field_length, field_width);
  ekf_localization_ = std::make_unique<EKFLocalization>(field_length, field_width);
  
  // Configurar pesos iniciais da fusão
  fusion_weights_.mcl_weight = 0.6;
  fusion_weights_.ekf_weight = 0.3;
  fusion_weights_.team_consensus_weight = 0.1;
  
  // Inicializar histórico
  pose_history_.reserve(MAX_HISTORY_SIZE);
  confidence_history_.reserve(MAX_HISTORY_SIZE);
  
  initialize_default_field_landmarks();
}

void NavigationCoordinator::initialize(
  const std::string& team_side,
  const roboime_msgs::msg::RobotPose2D& initial_pose,
  bool use_global_localization)
{
  team_side_ = team_side;
  
  if (use_global_localization) {
    // Inicialização global - distribuir partículas por todo o campo
    particle_filter_->initialize_global_localization();
    current_state_ = LocalizationState::GLOBAL_LOCALIZATION;
    current_mode_ = LocalizationMode::PARTICLE_FILTER_ONLY;
  } else {
    // Inicializar com pose conhecida
    particle_filter_->initialize_with_team_side(team_side, initial_pose);
    
    Eigen::Matrix3d initial_cov = Eigen::Matrix3d::Identity() * 0.5;
    ekf_localization_->initialize(initial_pose, initial_cov);
    
    current_state_ = LocalizationState::TRACKING;
    current_mode_ = LocalizationMode::HYBRID_FUSION;
  }
  
  last_good_localization_ = std::chrono::steady_clock::now();
}

void NavigationCoordinator::update_with_odometry(
  const roboime_msgs::msg::RobotPose2D& odometry_delta,
  double dt)
{
  // Atualizar ambos os algoritmos
  particle_filter_->predict(odometry_delta, dt);
  ekf_localization_->predict_with_odometry(odometry_delta, dt);
  
  // Verificar se precisa mudar modo
  evaluate_localization_mode();
}

void NavigationCoordinator::update_with_imu(const sensor_msgs::msg::Imu& imu_data)
{
  // IMU é especialmente útil para EKF
  ekf_localization_->update_with_imu(imu_data);
  
  // Também pode ajudar o particle filter para orientação
  // particle_filter_->update_with_imu(imu_data);  // Se implementado
}

void NavigationCoordinator::update_with_landmarks(const std::vector<Landmark>& landmarks)
{
  if (landmarks.empty()) return;
  
  // Converter landmarks para formato EKF
  std::vector<LandmarkMeasurement> ekf_measurements;
  for (const auto& landmark : landmarks) {
    LandmarkMeasurement::Type type;
    
    // Mapear tipos
    switch (landmark.type) {
      case Landmark::CENTER_CIRCLE:
        type = LandmarkMeasurement::CENTER_CIRCLE;
        break;
      case Landmark::PENALTY_MARK:
        type = LandmarkMeasurement::PENALTY_MARK;
        break;
      case Landmark::GOAL_POST:
        type = LandmarkMeasurement::GOAL_POST;
        break;
      default:
        type = LandmarkMeasurement::LINE_INTERSECTION;
        break;
    }
    
    ekf_measurements.emplace_back(
      type, landmark.distance, landmark.bearing, landmark.confidence,
      landmark.position, std::chrono::steady_clock::now()
    );
  }
  
  // Atualizar algoritmos
  particle_filter_->update(landmarks);
  ekf_localization_->update_with_landmarks(ekf_measurements);
  
  // Atualizar estado
  last_good_localization_ = std::chrono::steady_clock::now();
  
  // Avaliar se pode sair da localização global
  if (current_state_ == LocalizationState::GLOBAL_LOCALIZATION) {
    double mcl_confidence = particle_filter_->get_localization_confidence();
    if (mcl_confidence > 0.7) {
      current_state_ = LocalizationState::TRACKING;
      current_mode_ = LocalizationMode::HYBRID_FUSION;
    }
  }
  
  // Verificar se perdeu localização
  evaluate_localization_quality();
}

void NavigationCoordinator::process_team_robot_info(const TeamRobotInfo& robot_info)
{
  // Atualizar informações do robô do time
  team_robots_[robot_info.robot_id] = robot_info;
  
  // Usar consenso do time se ativado
  if (current_mode_ == LocalizationMode::TEAM_CONSENSUS) {
    update_with_team_consensus();
  }
}

roboime_msgs::msg::RobotPose2D NavigationCoordinator::get_best_pose_estimate() const
{
  switch (current_mode_) {
    case LocalizationMode::PARTICLE_FILTER_ONLY:
      return particle_filter_->get_estimated_pose();
      
    case LocalizationMode::EKF_ONLY:
      return ekf_localization_->get_estimated_pose();
      
    case LocalizationMode::HYBRID_FUSION:
      return fuse_pose_estimates();
      
    case LocalizationMode::TEAM_CONSENSUS:
      return get_team_consensus_pose();
      
    default:
      return particle_filter_->get_estimated_pose();
  }
}

geometry_msgs::msg::PoseWithCovarianceStamped NavigationCoordinator::get_pose_with_covariance() const
{
  // Usar EKF para covariância por ser mais preciso
  if (current_mode_ == LocalizationMode::EKF_ONLY || 
      current_mode_ == LocalizationMode::HYBRID_FUSION) {
    return ekf_localization_->get_pose_with_covariance();
  } else {
    return particle_filter_->get_pose_with_covariance();
  }
}

double NavigationCoordinator::get_localization_confidence() const
{
  switch (current_mode_) {
    case LocalizationMode::PARTICLE_FILTER_ONLY:
      return particle_filter_->get_localization_confidence();
      
    case LocalizationMode::EKF_ONLY:
      return 1.0 / (1.0 + ekf_localization_->get_localization_uncertainty());
      
    case LocalizationMode::HYBRID_FUSION:
      return calculate_fused_confidence();
      
    case LocalizationMode::TEAM_CONSENSUS:
      return calculate_team_consensus_confidence();
      
    default:
      return 0.0;
  }
}

void NavigationCoordinator::set_localization_mode(LocalizationMode new_mode)
{
  if (new_mode != current_mode_) {
    LocalizationMode old_mode = current_mode_;
    current_mode_ = new_mode;
    
    // Log da mudança
    // RCLCPP_INFO(...) seria usado aqui se tivéssemos acesso ao logger
    
    // Ajustar pesos baseado no novo modo
    switch (new_mode) {
      case LocalizationMode::PARTICLE_FILTER_ONLY:
        fusion_weights_.mcl_weight = 1.0;
        fusion_weights_.ekf_weight = 0.0;
        fusion_weights_.team_consensus_weight = 0.0;
        break;
        
      case LocalizationMode::EKF_ONLY:
        fusion_weights_.mcl_weight = 0.0;
        fusion_weights_.ekf_weight = 1.0;
        fusion_weights_.team_consensus_weight = 0.0;
        break;
        
      case LocalizationMode::HYBRID_FUSION:
        fusion_weights_.mcl_weight = 0.6;
        fusion_weights_.ekf_weight = 0.3;
        fusion_weights_.team_consensus_weight = 0.1;
        break;
        
      case LocalizationMode::TEAM_CONSENSUS:
        fusion_weights_.mcl_weight = 0.3;
        fusion_weights_.ekf_weight = 0.2;
        fusion_weights_.team_consensus_weight = 0.5;
        break;
        
      default:
        break;
    }
  }
}

bool NavigationCoordinator::is_well_localized(double confidence_threshold) const
{
  return current_state_ == LocalizationState::TRACKING && 
         get_localization_confidence() > confidence_threshold;
}

void NavigationCoordinator::reset_localization()
{
  current_state_ = LocalizationState::UNINITIALIZED;
  current_mode_ = LocalizationMode::INITIALIZATION_MODE;
  
  particle_filter_->initialize_global_localization();
  ekf_localization_->reset();
  
  pose_history_.clear();
  confidence_history_.clear();
  team_robots_.clear();
  
  last_good_localization_ = std::chrono::steady_clock::now();
}

void NavigationCoordinator::update_game_state(const std::string& game_state)
{
  // Ajustar estratégia baseado no estado do jogo
  if (game_state == "kickoff" || game_state == "penalty") {
    // Período crítico - usar localização mais conservadora
    if (current_mode_ == LocalizationMode::HYBRID_FUSION) {
      // Aumentar peso do MCL que é mais robusto
      fusion_weights_.mcl_weight = 0.7;
      fusion_weights_.ekf_weight = 0.2;
      fusion_weights_.team_consensus_weight = 0.1;
    }
  } else if (game_state == "playing") {
    // Jogo normal - usar fusão balanceada
    fusion_weights_.mcl_weight = 0.6;
    fusion_weights_.ekf_weight = 0.3;
    fusion_weights_.team_consensus_weight = 0.1;
  }
}

void NavigationCoordinator::configure_field_landmarks(
  const std::map<Landmark::Type, std::vector<Eigen::Vector2d>>& landmarks)
{
  // Configurar landmarks para ambos os algoritmos
  particle_filter_->set_field_landmarks(landmarks);
  
  // Converter para formato EKF
  std::map<LandmarkMeasurement::Type, std::vector<Eigen::Vector2d>> ekf_landmarks;
  for (const auto& [type, positions] : landmarks) {
    LandmarkMeasurement::Type ekf_type;
    switch (type) {
      case Landmark::CENTER_CIRCLE:
        ekf_type = LandmarkMeasurement::CENTER_CIRCLE;
        break;
      case Landmark::PENALTY_MARK:
        ekf_type = LandmarkMeasurement::PENALTY_MARK;
        break;
      case Landmark::GOAL_POST:
        ekf_type = LandmarkMeasurement::GOAL_POST;
        break;
      default:
        ekf_type = LandmarkMeasurement::LINE_INTERSECTION;
        break;
    }
    ekf_landmarks[ekf_type] = positions;
  }
  
  ekf_localization_->set_field_landmarks(ekf_landmarks);
}

// Métodos privados

roboime_msgs::msg::RobotPose2D NavigationCoordinator::fuse_pose_estimates() const
{
  auto mcl_pose = particle_filter_->get_estimated_pose();
  auto ekf_pose = ekf_localization_->get_estimated_pose();
  
  // Fusão ponderada
  roboime_msgs::msg::RobotPose2D fused_pose;
  
  double w_mcl = fusion_weights_.mcl_weight;
  double w_ekf = fusion_weights_.ekf_weight;
  double total_weight = w_mcl + w_ekf;
  
  if (total_weight > 0) {
    fused_pose.x = (w_mcl * mcl_pose.x + w_ekf * ekf_pose.x) / total_weight;
    fused_pose.y = (w_mcl * mcl_pose.y + w_ekf * ekf_pose.y) / total_weight;
    
    // Fusão de ângulos (circular)
    double x_comp = w_mcl * std::cos(mcl_pose.theta) + w_ekf * std::cos(ekf_pose.theta);
    double y_comp = w_mcl * std::sin(mcl_pose.theta) + w_ekf * std::sin(ekf_pose.theta);
    fused_pose.theta = std::atan2(y_comp, x_comp);
  } else {
    fused_pose = mcl_pose;  // Fallback
  }
  
  // Adicionar ao histórico
  if (pose_history_.size() >= MAX_HISTORY_SIZE) {
    pose_history_.erase(pose_history_.begin());
  }
  pose_history_.push_back(fused_pose);
  
  return fused_pose;
}

double NavigationCoordinator::calculate_fused_confidence() const
{
  double mcl_conf = particle_filter_->get_localization_confidence();
  double ekf_conf = 1.0 / (1.0 + ekf_localization_->get_localization_uncertainty());
  
  // Fusão ponderada das confianças
  double w_mcl = fusion_weights_.mcl_weight;
  double w_ekf = fusion_weights_.ekf_weight;
  double total_weight = w_mcl + w_ekf;
  
  if (total_weight > 0) {
    return (w_mcl * mcl_conf + w_ekf * ekf_conf) / total_weight;
  }
  
  return std::max(mcl_conf, ekf_conf);
}

void NavigationCoordinator::evaluate_localization_mode()
{
  // Avaliar qual algoritmo está performando melhor
  double mcl_confidence = particle_filter_->get_localization_confidence();
  double ekf_uncertainty = ekf_localization_->get_localization_uncertainty();
  double ekf_confidence = 1.0 / (1.0 + ekf_uncertainty);
  
  // Critérios para mudança de modo
  if (current_mode_ == LocalizationMode::HYBRID_FUSION) {
    // Se EKF está muito melhor, mudar para EKF only
    if (ekf_confidence > mcl_confidence + 0.3 && ekf_confidence > 0.8) {
      set_localization_mode(LocalizationMode::EKF_ONLY);
    }
    // Se MCL está muito melhor, mudar para MCL only
    else if (mcl_confidence > ekf_confidence + 0.3 && mcl_confidence > 0.8) {
      set_localization_mode(LocalizationMode::PARTICLE_FILTER_ONLY);
    }
  }
  
  // Se está perdido há muito tempo, voltar para localização global
  auto now = std::chrono::steady_clock::now();
  auto time_since_good = std::chrono::duration_cast<std::chrono::seconds>(
    now - last_good_localization_).count();
    
  if (time_since_good > 10 && get_localization_confidence() < 0.2) {
    current_state_ = LocalizationState::LOST;
    set_localization_mode(LocalizationMode::PARTICLE_FILTER_ONLY);
    particle_filter_->initialize_global_localization();
  }
}

void NavigationCoordinator::evaluate_localization_quality()
{
  double confidence = get_localization_confidence();
  
  // Atualizar histórico de confiança
  if (confidence_history_.size() >= MAX_HISTORY_SIZE) {
    confidence_history_.erase(confidence_history_.begin());
  }
  confidence_history_.push_back(confidence);
  
  // Avaliar tendência
  if (confidence_history_.size() > 10) {
    double recent_avg = 0.0;
    for (size_t i = confidence_history_.size() - 10; i < confidence_history_.size(); ++i) {
      recent_avg += confidence_history_[i];
    }
    recent_avg /= 10.0;
    
    // Atualizar estado baseado na confiança
    if (recent_avg > 0.7) {
      if (current_state_ != LocalizationState::TRACKING) {
        current_state_ = LocalizationState::TRACKING;
      }
    } else if (recent_avg > 0.3) {
      current_state_ = LocalizationState::RECOVERED;
    } else {
      current_state_ = LocalizationState::LOST;
    }
  }

  // Detecção simples de "kidnapped robot": queda brusca recente
  if (confidence_history_.size() >= 2) {
    double prev = confidence_history_[confidence_history_.size() - 2];
    double drop = prev - confidence;
    if (drop > 0.3) { // limiar padrão; pode ser parametrizado
      current_state_ = LocalizationState::LOST;
    }
  }
}

roboime_msgs::msg::RobotPose2D NavigationCoordinator::get_team_consensus_pose() const
{
  // Implementação básica - usar pose própria se não há consenso
  // Em implementação completa, faria average das poses dos companheiros
  
  if (team_robots_.empty()) {
    return fuse_pose_estimates();
  }
  
  // Calcular consenso baseado nas poses dos companheiros
  std::vector<roboime_msgs::msg::RobotPose2D> valid_poses;
  valid_poses.push_back(fuse_pose_estimates());  // Própria pose
  
  auto now = std::chrono::steady_clock::now();
  for (const auto& [id, robot] : team_robots_) {
    auto age = std::chrono::duration_cast<std::chrono::seconds>(
      now - robot.last_update).count();
      
    if (age < 5 && robot.confidence > 0.5) {  // Dados recentes e confiáveis
      valid_poses.push_back(robot.pose);
    }
  }
  
  if (valid_poses.size() == 1) {
    return valid_poses[0];
  }
  
  // Calcular média ponderada
  roboime_msgs::msg::RobotPose2D consensus_pose;
  double total_weight = 0.0;
  double x_comp = 0.0, y_comp = 0.0;
  
  for (const auto& pose : valid_poses) {
    double weight = 1.0;  // Peso igual para simplicidade
    consensus_pose.x += weight * pose.x;
    consensus_pose.y += weight * pose.y;
    x_comp += weight * std::cos(pose.theta);
    y_comp += weight * std::sin(pose.theta);
    total_weight += weight;
  }
  
  if (total_weight > 0) {
    consensus_pose.x /= total_weight;
    consensus_pose.y /= total_weight;
    consensus_pose.theta = std::atan2(y_comp, x_comp);
  }
  
  return consensus_pose;
}

double NavigationCoordinator::calculate_team_consensus_confidence() const
{
  // Confiança baseada na concordância entre os robôs
  if (team_robots_.empty()) {
    return calculate_fused_confidence();
  }
  
  // Implementação simplificada
  return std::min(0.9, calculate_fused_confidence() + 0.1);
}

void NavigationCoordinator::update_with_team_consensus()
{
  // Atualizar consenso do time
  // Implementação básica - apenas calcular consenso
  auto consensus_pose = get_team_consensus_pose();
  
  // Usar consenso para corrigir algoritmos se discrepância é grande
  auto current_pose = fuse_pose_estimates();
  
  double distance = std::sqrt(
    std::pow(consensus_pose.x - current_pose.x, 2) +
    std::pow(consensus_pose.y - current_pose.y, 2)
  );
  
  if (distance > 1.0) {  // Discrepância significativa
    // Reinicializar com pose do consenso
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 0.5;
    ekf_localization_->initialize(consensus_pose, cov);
  }
}

void NavigationCoordinator::initialize_default_field_landmarks()
{
  std::map<Landmark::Type, std::vector<Eigen::Vector2d>> landmarks;
  
  // Círculo central
  landmarks[Landmark::CENTER_CIRCLE] = {
    Eigen::Vector2d(0.0, 0.0)
  };
  
  // Marcas de penalty
  landmarks[Landmark::PENALTY_MARK] = {
    Eigen::Vector2d(-3.0, 0.0),
    Eigen::Vector2d(3.0, 0.0)
  };
  
  // Gols
  landmarks[Landmark::GOAL_POST] = {
    Eigen::Vector2d(-4.5, 0.0),
    Eigen::Vector2d(4.5, 0.0)
  };
  
  // Cantos
  landmarks[Landmark::FIELD_CORNER] = {
    Eigen::Vector2d(-4.5, -3.0), Eigen::Vector2d(-4.5, 3.0),
    Eigen::Vector2d(4.5, -3.0),  Eigen::Vector2d(4.5, 3.0)
  };
  
  configure_field_landmarks(landmarks);
}

}  // namespace roboime_navigation 