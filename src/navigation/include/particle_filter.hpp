#ifndef ROBOIME_NAVIGATION__PARTICLE_FILTER_HPP_
#define ROBOIME_NAVIGATION__PARTICLE_FILTER_HPP_

#include <vector>
#include <random>
#include <memory>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <roboime_msgs/msg/robot_pose2_d.hpp>
#include <roboime_msgs/msg/field_landmark.hpp>
#include <roboime_msgs/msg/landmark_array.hpp>

namespace roboime_navigation
{

/**
 * @brief Estrutura para representar uma partícula no filtro
 */
struct Particle
{
  double x;          // Posição X no campo (metros)
  double y;          // Posição Y no campo (metros)  
  double theta;      // Orientação (radianos)
  double weight;     // Peso da partícula [0,1]
  
  Particle() : x(0.0), y(0.0), theta(0.0), weight(1.0) {}
  Particle(double x_, double y_, double theta_, double weight_ = 1.0)
    : x(x_), y(y_), theta(theta_), weight(weight_) {}
};

/**
 * @brief Estrutura para representar um landmark detectado
 */
struct Landmark
{
  enum Type {
    CENTER_CIRCLE,
    PENALTY_MARK,
    GOAL_POST,
    GOAL_AREA_CORNER,
    LINE_INTERSECTION,
    FIELD_CORNER
  };
  
  Type type;                 // Tipo do landmark
  double distance;           // Distância relativa ao robô (metros)
  double bearing;            // Ângulo relativo ao robô (radianos)
  double confidence;         // Confiança da detecção [0,1]
  Eigen::Vector2d position;  // Posição absoluta no campo (se conhecida)
  
  Landmark(Type t, double d, double b, double c = 1.0)
    : type(t), distance(d), bearing(b), confidence(c) {}
};

/**
 * @brief Filtro de Partículas Monte Carlo para Localização (MCL)
 * 
 * Implementação otimizada para robôs de futebol RoboCup Humanoid League.
 * Características:
 * - Resolução de ambiguidade de simetria do campo
 * - Fusão de múltiplos sensores (visão + IMU + odometria)
 * - Recuperação de kidnapping
 * - Localização global e tracking
 */
class ParticleFilter
{
public:
  /**
   * @brief Construtor do filtro de partículas
   * @param num_particles Número de partículas (padrão: 500)
   * @param field_length Comprimento do campo em metros
   * @param field_width Largura do campo em metros
   */
  explicit ParticleFilter(
    size_t num_particles = 500,
    double field_length = 9.0,
    double field_width = 6.0
  );

  /**
   * @brief Destructor
   */
  ~ParticleFilter() = default;

  /**
   * @brief Inicializa partículas para localização global
   * @param initial_pose Pose inicial aproximada (opcional)
   * @param covariance Covariância da pose inicial
   */
  void initialize_global_localization(
    const roboime_msgs::msg::RobotPose2D* initial_pose = nullptr,
    const Eigen::Matrix3d& covariance = Eigen::Matrix3d::Identity()
  );

  /**
   * @brief Inicializa partículas com base no lado conhecido do time
   * @param team_side "left" ou "right" 
   * @param initial_pose Pose inicial no lado do time
   */
  void initialize_with_team_side(
    const std::string& team_side,
    const roboime_msgs::msg::RobotPose2D& initial_pose
  );

  /**
   * @brief Etapa de predição do filtro (movimento do robô)
   * @param odometry_delta Mudança na odometria desde a última atualização
   * @param imu_data Dados do IMU para orientação
   * @param dt Intervalo de tempo (segundos)
   */
  void predict(
    const roboime_msgs::msg::RobotPose2D& odometry_delta,
    const sensor_msgs::msg::Imu& imu_data,
    double dt
  );

  /**
   * @brief Etapa de correção do filtro (observações dos landmarks)
   * @param landmarks Vetor de landmarks detectados
   */
  void update(const std::vector<Landmark>& landmarks);

  /**
   * @brief Reamostragem das partículas (Systematic Resampling)
   */
  void resample();

  /**
   * @brief Calcula a pose estimada do robô
   * @return Pose média ponderada das partículas
   */
  roboime_msgs::msg::RobotPose2D get_estimated_pose() const;

  /**
   * @brief Calcula a pose com covariância
   * @return Pose com matriz de covariância 3x3
   */
  geometry_msgs::msg::PoseWithCovarianceStamped get_pose_with_covariance() const;

  /**
   * @brief Calcula a confiança da localização
   * @return Valor entre 0 e 1 indicando qualidade da localização
   */
  double get_localization_confidence() const;

  /**
   * @brief Verifica se o robô está bem localizado
   * @param confidence_threshold Limiar mínimo de confiança
   * @return true se bem localizado
   */
  bool is_well_localized(double confidence_threshold = 0.7) const;

  /**
   * @brief Detecta se o robô foi movido inesperadamente (kidnapping)
   * @return true se kidnapping detectado
   */
  bool detect_kidnapping() const;

  /**
   * @brief Recupera de kidnapping redistribuindo partículas
   */
  void recover_from_kidnapping();

  /**
   * @brief Define landmarks conhecidos do campo
   * @param field_landmarks Mapa de landmarks com posições absolutas
   */
  void set_field_landmarks(const std::map<Landmark::Type, std::vector<Eigen::Vector2d>>& field_landmarks);

  /**
   * @brief Atualiza parâmetros do filtro
   * @param motion_noise_std Desvio padrão do ruído de movimento
   * @param measurement_noise_std Desvio padrão do ruído de medição
   */
  void update_noise_parameters(
    const Eigen::Vector3d& motion_noise_std,
    double measurement_noise_std
  );

  /**
   * @brief Retorna as partículas para visualização/debug
   * @return Vetor de todas as partículas atuais
   */
  const std::vector<Particle>& get_particles() const { return particles_; }

  /**
   * @brief Retorna número efetivo de partículas (medida de diversidade)
   * @return Neff entre 1 e num_particles
   */
  double get_effective_particle_count() const;

private:
  // =============================================================================
  // PARÂMETROS DO FILTRO
  // =============================================================================
  size_t num_particles_;          // Número de partículas
  double field_length_;           // Comprimento do campo
  double field_width_;            // Largura do campo
  double resample_threshold_;     // Limiar para reamostragem (Neff)
  
  // Ruídos do modelo
  Eigen::Vector3d motion_noise_std_;      // [x, y, theta] std dev
  double measurement_noise_std_;          // Desvio padrão das medições
  
  // =============================================================================
  // ESTADO DO FILTRO
  // =============================================================================
  std::vector<Particle> particles_;              // Conjunto de partículas
  std::map<Landmark::Type, std::vector<Eigen::Vector2d>> field_landmarks_;  // Landmarks conhecidos
  
  // Histórico para detecção de kidnapping
  std::vector<roboime_msgs::msg::RobotPose2D> pose_history_;
  std::vector<double> confidence_history_;
  
  // =============================================================================
  // GERADOR DE NÚMEROS ALEATÓRIOS
  // =============================================================================
  mutable std::random_device rd_;
  mutable std::mt19937 gen_;
  mutable std::normal_distribution<double> normal_dist_;
  mutable std::uniform_real_distribution<double> uniform_dist_;
  
  // =============================================================================
  // MÉTODOS PRIVADOS
  // =============================================================================
  
  /**
   * @brief Aplica modelo de movimento a uma partícula
   */
  void apply_motion_model(
    Particle& particle,
    const roboime_msgs::msg::RobotPose2D& odometry_delta,
    double dt
  );

  /**
   * @brief Calcula likelihood de uma observação dada uma partícula
   */
  double calculate_measurement_likelihood(
    const Particle& particle,
    const Landmark& landmark
  ) const;

  /**
   * @brief Encontra correspondência entre landmark observado e mapa
   */
  std::vector<Eigen::Vector2d> find_landmark_candidates(
    const Landmark& observed_landmark,
    const Particle& particle
  ) const;

  /**
   * @brief Normaliza pesos das partículas
   */
  void normalize_weights();

  /**
   * @brief Adiciona ruído gaussiano
   */
  double add_gaussian_noise(double mean, double std_dev) const;

  /**
   * @brief Verifica se uma pose está dentro dos limites do campo
   */
  bool is_pose_valid(const Particle& particle) const;

  /**
   * @brief Calcula distância euclidiana entre duas poses
   */
  double pose_distance(
    const roboime_msgs::msg::RobotPose2D& pose1,
    const roboime_msgs::msg::RobotPose2D& pose2
  ) const;

  /**
   * @brief Normaliza ângulo para intervalo [-π, π]
   */
  double normalize_angle(double angle) const;
};

}  // namespace roboime_navigation

#endif  // ROBOIME_NAVIGATION__PARTICLE_FILTER_HPP_ 