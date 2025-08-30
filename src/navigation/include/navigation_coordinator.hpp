#ifndef ROBOIME_NAVIGATION__NAVIGATION_COORDINATOR_HPP_
#define ROBOIME_NAVIGATION__NAVIGATION_COORDINATOR_HPP_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <roboime_msgs/msg/robot_pose2_d.hpp>
#include <roboime_msgs/msg/field_landmark.hpp>
#include <roboime_msgs/msg/landmark_array.hpp>
#include <roboime_msgs/msg/localization_status.hpp>

#include "particle_filter.hpp"
#include "ekf_localization.hpp"

namespace roboime_navigation
{

/**
 * @brief Estrutura para informações de comunicação entre robôs
 */
struct TeamRobotInfo
{
  uint32_t robot_id;
  roboime_msgs::msg::RobotPose2D pose;
  double confidence;
  std::vector<Landmark> landmarks_seen;
  std::chrono::steady_clock::time_point last_update;
  std::string status;  // "active", "lost", "inactive"
  
  TeamRobotInfo(uint32_t id) 
    : robot_id(id), confidence(0.0), last_update(std::chrono::steady_clock::now()), status("inactive") {}
};

/**
 * @brief Enumeração dos modos de localização
 */
enum class LocalizationMode
{
  PARTICLE_FILTER_ONLY,    // Apenas filtro de partículas
  EKF_ONLY,               // Apenas EKF
  HYBRID_FUSION,          // Fusão inteligente MCL + EKF
  TEAM_CONSENSUS,         // Baseado em consenso do time
  INITIALIZATION_MODE     // Modo de inicialização/global localization
};

/**
 * @brief Estado da localização
 */
enum class LocalizationState
{
  UNINITIALIZED,          // Não inicializado
  GLOBAL_LOCALIZATION,    // Localização global (busca inicial)
  TRACKING,              // Rastreamento normal
  LOST,                  // Perdido, tentando recuperar
  RECOVERED,             // Recém recuperado
  ERROR                  // Erro no sistema
};

/**
 * @brief Coordenador Principal de Navegação
 * 
 * Responsabilidades:
 * - Gerenciar múltiplos algoritmos de localização (MCL + EKF)
 * - Fusão inteligente de estimativas
 * - Comunicação e consenso entre robôs do time
 * - Resolução de ambiguidade de simetria do campo
 * - Recuperação automática de erros
 * - Interface unificada para o sistema de comportamento
 */
class NavigationCoordinator
{
public:
  /**
   * @brief Construtor do coordenador
   * @param field_length Comprimento do campo (metros)
   * @param field_width Largura do campo (metros)
   * @param robot_id ID único do robô no time
   * @param team_name Nome do time
   */
  explicit NavigationCoordinator(
    double field_length = 9.0,
    double field_width = 6.0,
    uint32_t robot_id = 1,
    const std::string& team_name = "RoboIME"
  );

  /**
   * @brief Destructor
   */
  ~NavigationCoordinator() = default;

  /**
   * @brief Inicializa o coordenador com lado conhecido do time
   * @param team_side "left" ou "right"
   * @param initial_pose Pose inicial aproximada
   * @param use_global_localization true para busca global inicial
   */
  void initialize(
    const std::string& team_side,
    const roboime_msgs::msg::RobotPose2D& initial_pose,
    bool use_global_localization = true
  );

  /**
   * @brief Atualiza localização com dados de odometria
   * @param odometry_delta Mudança na odometria
   * @param dt Intervalo de tempo
   */
  void update_with_odometry(
    const roboime_msgs::msg::RobotPose2D& odometry_delta,
    double dt
  );

  /**
   * @brief Atualiza localização com dados do IMU
   * @param imu_data Dados do IMU
   */
  void update_with_imu(const sensor_msgs::msg::Imu& imu_data);

  /**
   * @brief Atualiza localização com landmarks detectados
   * @param landmarks Vetor de landmarks observados
   */
  void update_with_landmarks(const std::vector<Landmark>& landmarks);

  /**
   * @brief Processa informação recebida de outro robô do time
   * @param robot_info Informações do companheiro de time
   */
  void process_team_robot_info(const TeamRobotInfo& robot_info);

  /**
   * @brief Retorna a melhor estimativa de pose atual
   * @return Pose estimada com maior confiança
   */
  roboime_msgs::msg::RobotPose2D get_best_pose_estimate() const;

  /**
   * @brief Retorna pose com covariância completa
   * @return Pose com matriz de covariância
   */
  geometry_msgs::msg::PoseWithCovarianceStamped get_pose_with_covariance() const;

  /**
   * @brief Calcula confiança da localização atual
   * @return Valor entre 0 e 1
   */
  double get_localization_confidence() const;

  /**
   * @brief Retorna estado atual da localização
   * @return Estado do sistema de localização
   */
  LocalizationState get_localization_state() const { return current_state_; }

  /**
   * @brief Retorna modo atual de localização
   * @return Modo ativo de localização
   */
  LocalizationMode get_localization_mode() const { return current_mode_; }

  /**
   * @brief Força mudança de modo de localização
   * @param new_mode Novo modo a ser ativado
   */
  void set_localization_mode(LocalizationMode new_mode);

  /**
   * @brief Verifica se robô está bem localizado
   * @param confidence_threshold Limiar mínimo de confiança
   * @return true se bem localizado
   */
  bool is_well_localized(double confidence_threshold = 0.7) const;

  /**
   * @brief Reset completo do sistema de localização
   */
  void reset_localization();

  /**
   * @brief Processa informações do estado do jogo
   * @param game_state Estado atual do jogo (kickoff, playing, etc.)
   */
  void update_game_state(const std::string& game_state);

  /**
   * @brief Configura landmarks conhecidos do campo
   * @param landmarks Mapa de landmarks com posições
   */
  void configure_field_landmarks(
    const std::map<Landmark::Type, std::vector<Eigen::Vector2d>>& landmarks
  );

  /**
   * @brief Atualiza parâmetros de ruído dos algoritmos
   * @param motion_noise Ruído do modelo de movimento
   * @param measurement_noise Ruído das medições
   */
  void update_noise_parameters(
    const Eigen::Vector3d& motion_noise,
    double measurement_noise
  );

  /**
   * @brief Retorna informações de todos os robôs do time
   * @return Mapa de informações por robot_id
   */
  const std::map<uint32_t, TeamRobotInfo>& get_team_robots_info() const { return team_robots_; }

  /**
   * @brief Calcula pose consensual do time
   * @return Pose baseada em consenso entre robôs
   */
  std::pair<roboime_msgs::msg::RobotPose2D, double> calculate_team_consensus() const;

  /**
   * @brief Detecta outliers nas estimativas do time
   * @param pose_estimates Estimativas de pose de diferentes robôs
   * @return Índices das estimativas válidas
   */
  std::vector<size_t> detect_consensus_outliers(
    const std::vector<roboime_msgs::msg::RobotPose2D>& pose_estimates
  ) const;

  /**
   * @brief Exporta dados para debug/visualização
   * @return String JSON com informações detalhadas
   */
  std::string export_debug_info() const;

private:
  // =============================================================================
  // PARÂMETROS DO SISTEMA
  // =============================================================================
  double field_length_;
  double field_width_;
  uint32_t robot_id_;
  std::string team_name_;
  std::string team_side_;  // "left" ou "right"
  
  // =============================================================================
  // ALGORITMOS DE LOCALIZAÇÃO
  // =============================================================================
  std::unique_ptr<ParticleFilter> particle_filter_;
  std::unique_ptr<EKFLocalization> ekf_localization_;
  
  // =============================================================================
  // ESTADO DO COORDENADOR
  // =============================================================================
  LocalizationState current_state_;
  LocalizationMode current_mode_;
  
  // Histórico de poses para análise
  std::vector<roboime_msgs::msg::RobotPose2D> pose_history_;
  std::vector<double> confidence_history_;
  static constexpr size_t MAX_HISTORY_SIZE = 100;
  
  // =============================================================================
  // COMUNICAÇÃO ENTRE ROBÔS
  // =============================================================================
  std::map<uint32_t, TeamRobotInfo> team_robots_;
  std::chrono::steady_clock::time_point last_team_broadcast_;
  static constexpr std::chrono::seconds TEAM_COMM_TIMEOUT{5};
  
  // =============================================================================
  // FUSÃO DE ALGORITMOS
  // =============================================================================
  struct AlgorithmWeights
  {
    double mcl_weight;
    double ekf_weight;
    double team_consensus_weight;
    
    AlgorithmWeights() : mcl_weight(0.6), ekf_weight(0.3), team_consensus_weight(0.1) {}
  } fusion_weights_;
  
  // =============================================================================
  // DETECÇÃO DE ANOMALIAS
  // =============================================================================
  std::chrono::steady_clock::time_point last_good_localization_;
  static constexpr std::chrono::seconds LOCALIZATION_TIMEOUT{10};
  
  // Contadores de erros
  size_t consecutive_low_confidence_count_;
  size_t kidnapping_detection_count_;
  
  // =============================================================================
  // MÉTODOS PRIVADOS - FUSÃO
  // =============================================================================
  
  /**
   * @brief Calcula pesos adaptativos para fusão de algoritmos
   */
  void update_fusion_weights();

  /**
   * @brief Funde estimativas de MCL e EKF
   * @return Pose fundida e confiança
   */
  std::pair<roboime_msgs::msg::RobotPose2D, double> fuse_mcl_ekf_estimates() const;

  /**
   * @brief Aplica consenso do time na estimativa
   * @param individual_estimate Estimativa individual
   * @return Estimativa corrigida pelo consenso
   */
  roboime_msgs::msg::RobotPose2D apply_team_consensus(
    const roboime_msgs::msg::RobotPose2D& individual_estimate
  ) const;

  // =============================================================================
  // MÉTODOS PRIVADOS - GERENCIAMENTO DE ESTADO
  // =============================================================================
  
  /**
   * @brief Atualiza estado baseado nas condições atuais
   */
  void update_localization_state();

  /**
   * @brief Verifica condições para mudança automática de modo
   */
  void check_mode_transition_conditions();

  /**
   * @brief Detecta kidnapping ou perda de localização
   */
  bool detect_localization_failure() const;

  /**
   * @brief Inicia procedimento de recuperação
   */
  void initiate_recovery_procedure();

  // =============================================================================
  // MÉTODOS PRIVADOS - RESOLUÇÃO DE SIMETRIA
  // =============================================================================
  
  /**
   * @brief Resolve ambiguidade de simetria usando contexto
   * @param symmetric_poses Poses simétricas possíveis
   * @return Pose mais provável
   */
  roboime_msgs::msg::RobotPose2D resolve_field_symmetry(
    const std::vector<roboime_msgs::msg::RobotPose2D>& symmetric_poses
  ) const;

  /**
   * @brief Valida pose usando conhecimento do lado do time
   * @param pose Pose a ser validada
   * @return true se pose é consistente com lado do time
   */
  bool validate_pose_with_team_side(const roboime_msgs::msg::RobotPose2D& pose) const;

  // =============================================================================
  // MÉTODOS PRIVADOS - UTILITÁRIOS
  // =============================================================================
  
  /**
   * @brief Limpa robôs inativos da lista do time
   */
  void cleanup_inactive_team_robots();

  /**
   * @brief Calcula distância entre duas poses
   */
  double calculate_pose_distance(
    const roboime_msgs::msg::RobotPose2D& pose1,
    const roboime_msgs::msg::RobotPose2D& pose2
  ) const;

  /**
   * @brief Normaliza ângulo para [-π, π]
   */
  double normalize_angle(double angle) const;

  /**
   * @brief Verifica se timestamp é recente
   */
  bool is_timestamp_recent(
    const std::chrono::steady_clock::time_point& timestamp,
    std::chrono::seconds max_age = std::chrono::seconds(5)
  ) const;

  // Métodos adicionais (implementados no .cpp) que estavam faltando nas declarações
  void evaluate_localization_mode();
  void evaluate_localization_quality();
  roboime_msgs::msg::RobotPose2D get_team_consensus_pose() const;
  double calculate_fused_confidence() const;
  double calculate_team_consensus_confidence() const;
  void update_with_team_consensus();
  void initialize_default_field_landmarks();
  roboime_msgs::msg::RobotPose2D fuse_pose_estimates() const;
};

}  // namespace roboime_navigation

#endif  // ROBOIME_NAVIGATION__NAVIGATION_COORDINATOR_HPP_ 