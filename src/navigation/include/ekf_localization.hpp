#ifndef ROBOIME_NAVIGATION__EKF_LOCALIZATION_HPP_
#define ROBOIME_NAVIGATION__EKF_LOCALIZATION_HPP_

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <roboime_msgs/msg/robot_pose2_d.hpp>
#include <roboime_msgs/msg/field_landmark.hpp>
#include <roboime_msgs/msg/landmark_array.hpp>
#include <memory>
#include <queue>
#include <chrono>

namespace roboime_navigation
{

/**
 * @brief Estrutura para medição de landmark
 */
struct LandmarkMeasurement
{
  enum Type {
    CENTER_CIRCLE,
    PENALTY_MARK,
    GOAL_POST,
    LINE_INTERSECTION
  };
  
  Type type;                    // Tipo do landmark
  double range;                 // Distância ao landmark (metros)
  double bearing;              // Ângulo ao landmark (radianos)
  double confidence;           // Confiança da medição [0,1]
  Eigen::Vector2d map_position; // Posição conhecida no mapa
  std::chrono::steady_clock::time_point timestamp;
  
  LandmarkMeasurement(Type t, double r, double b, double c, 
                     const Eigen::Vector2d& pos, 
                     std::chrono::steady_clock::time_point ts)
    : type(t), range(r), bearing(b), confidence(c), map_position(pos), timestamp(ts) {}
};

/**
 * @brief Extended Kalman Filter para Localização Multi-Sensorial
 * 
 * Características:
 * - Fusão de odometria, IMU e observações visuais
 * - Modelo de movimento para robôs humanoides
 * - Correção baseada em landmarks conhecidos
 * - Estimativa de incerteza em tempo real
 * - Filtragem de outliers automática
 */
class EKFLocalization
{
public:
  /**
   * @brief Construtor do EKF
   * @param field_length Comprimento do campo (metros)
   * @param field_width Largura do campo (metros)
   */
  explicit EKFLocalization(
    double field_length = 9.0,
    double field_width = 6.0
  );

  /**
   * @brief Destructor
   */
  ~EKFLocalization() = default;

  /**
   * @brief Inicializa o filtro com pose inicial
   * @param initial_pose Pose inicial do robô
   * @param initial_covariance Matriz de covariância inicial 3x3
   */
  void initialize(
    const roboime_msgs::msg::RobotPose2D& initial_pose,
    const Eigen::Matrix3d& initial_covariance = Eigen::Matrix3d::Identity()
  );

  /**
   * @brief Inicializa com múltiplas hipóteses (para ambiguidade de simetria)
   * @param hypothesis_poses Vetor de poses possíveis
   * @param covariances Covariâncias correspondentes
   */
  void initialize_multi_hypothesis(
    const std::vector<roboime_msgs::msg::RobotPose2D>& hypothesis_poses,
    const std::vector<Eigen::Matrix3d>& covariances
  );

  /**
   * @brief Etapa de predição com odometria
   * @param odometry_delta Mudança na odometria
   * @param dt Intervalo de tempo (segundos)
   */
  void predict_with_odometry(
    const roboime_msgs::msg::RobotPose2D& odometry_delta,
    double dt
  );

  /**
   * @brief Etapa de predição com modelo de movimento
   * @param linear_velocity Velocidade linear (m/s)
   * @param angular_velocity Velocidade angular (rad/s)
   * @param dt Intervalo de tempo (segundos)
   */
  void predict_with_motion_model(
    double linear_velocity,
    double angular_velocity,
    double dt
  );

  /**
   * @brief Correção com dados do IMU (orientação)
   * @param imu_data Dados do IMU
   */
  void update_with_imu(const sensor_msgs::msg::Imu& imu_data);

  /**
   * @brief Correção com observação de landmark
   * @param landmark Medição do landmark
   */
  void update_with_landmark(const LandmarkMeasurement& landmark);

  /**
   * @brief Correção com múltiplos landmarks simultaneamente
   * @param landmarks Vetor de medições de landmarks
   */
  void update_with_landmarks(const std::vector<LandmarkMeasurement>& landmarks);

  /**
   * @brief Retorna a pose estimada atual
   * @return Pose estimada (x, y, theta)
   */
  roboime_msgs::msg::RobotPose2D get_estimated_pose() const;

  /**
   * @brief Retorna pose com covariância completa
   * @return Pose com matriz de covariância
   */
  geometry_msgs::msg::PoseWithCovarianceStamped get_pose_with_covariance() const;

  /**
   * @brief Calcula incerteza da localização
   * @return Valor escalar representando incerteza total
   */
  double get_localization_uncertainty() const;

  /**
   * @brief Verifica se a localização é confiável
   * @param uncertainty_threshold Limiar máximo de incerteza
   * @return true se localização é confiável
   */
  bool is_localization_reliable(double uncertainty_threshold = 0.5) const;

  /**
   * @brief Detecta e rejeita outliers em medições
   * @param landmark Medição a ser validada
   * @param innovation_threshold Limiar para rejeição (Mahalanobis distance)
   * @return true se medição é válida
   */
  bool validate_measurement(
    const LandmarkMeasurement& landmark,
    double innovation_threshold = 9.0
  ) const;

  /**
   * @brief Define landmarks conhecidos do campo
   * @param landmarks Mapa de landmarks com posições
   */
  void set_field_landmarks(
    const std::map<LandmarkMeasurement::Type, std::vector<Eigen::Vector2d>>& landmarks
  );

  /**
   * @brief Atualiza parâmetros do modelo de ruído
   * @param process_noise Matriz Q (ruído do processo) 3x3
   * @param measurement_noise_std Desvio padrão das medições
   * @param imu_orientation_noise_std Desvio padrão da orientação IMU
   */
  void update_noise_parameters(
    const Eigen::Matrix3d& process_noise,
    double measurement_noise_std,
    double imu_orientation_noise_std
  );

  /**
   * @brief Reset do filtro (para recuperação de erro)
   */
  void reset();

  /**
   * @brief Retorna matriz de covariância atual
   * @return Matriz P 3x3
   */
  const Eigen::Matrix3d& get_covariance_matrix() const { return P_; }

  /**
   * @brief Retorna vetor de estado atual [x, y, theta]
   * @return Estado estimado
   */
  const Eigen::Vector3d& get_state_vector() const { return x_; }

  /**
   * @brief Calcula inovação (resíduo) para uma medição
   * @param landmark Medição do landmark
   * @return Vetor de inovação e matriz de covariância
   */
  std::pair<Eigen::Vector2d, Eigen::Matrix2d> calculate_innovation(
    const LandmarkMeasurement& landmark
  ) const;

private:
  // =============================================================================
  // PARÂMETROS DO CAMPO E SISTEMA
  // =============================================================================
  double field_length_;              // Comprimento do campo
  double field_width_;               // Largura do campo
  
  // =============================================================================
  // ESTADO DO FILTRO EKF
  // =============================================================================
  Eigen::Vector3d x_;                // Estado [x, y, theta]
  Eigen::Matrix3d P_;                // Matriz de covariância
  
  // Matrizes do modelo
  Eigen::Matrix3d Q_;                // Matriz de ruído do processo
  double measurement_noise_std_;     // Desvio padrão das medições
  double imu_noise_std_;            // Desvio padrão do IMU
  
  // =============================================================================
  // MAPA DE LANDMARKS
  // =============================================================================
  std::map<LandmarkMeasurement::Type, std::vector<Eigen::Vector2d>> field_landmarks_;
  
  // =============================================================================
  // HISTÓRICO PARA VALIDAÇÃO
  // =============================================================================
  std::queue<LandmarkMeasurement> recent_measurements_;
  static constexpr size_t MAX_MEASUREMENT_HISTORY = 50;
  
  // Flags de inicialização
  bool is_initialized_;
  
  // =============================================================================
  // MÉTODOS PRIVADOS - MODELO DE MOVIMENTO
  // =============================================================================
  
  /**
   * @brief Matriz Jacobiana do modelo de movimento
   * @param dt Intervalo de tempo
   * @param linear_vel Velocidade linear
   * @param angular_vel Velocidade angular
   * @return Matriz F 3x3
   */
  Eigen::Matrix3d motion_jacobian(
    double dt,
    double linear_vel,
    double angular_vel
  ) const;

  /**
   * @brief Função de movimento (modelo cinemático)
   * @param state Estado atual [x, y, theta]
   * @param linear_vel Velocidade linear
   * @param angular_vel Velocidade angular
   * @param dt Intervalo de tempo
   * @return Novo estado
   */
  Eigen::Vector3d motion_model(
    const Eigen::Vector3d& state,
    double linear_vel,
    double angular_vel,
    double dt
  ) const;

  // =============================================================================
  // MÉTODOS PRIVADOS - MODELO DE OBSERVAÇÃO
  // =============================================================================
  
  /**
   * @brief Função de observação (range, bearing)
   * @param state Estado atual
   * @param landmark_position Posição do landmark no mapa
   * @return Medição esperada [range, bearing]
   */
  Eigen::Vector2d observation_model(
    const Eigen::Vector3d& state,
    const Eigen::Vector2d& landmark_position
  ) const;

  /**
   * @brief Matriz Jacobiana do modelo de observação
   * @param state Estado atual
   * @param landmark_position Posição do landmark
   * @return Matriz H 2x3
   */
  Eigen::Matrix<double, 2, 3> observation_jacobian(
    const Eigen::Vector3d& state,
    const Eigen::Vector2d& landmark_position
  ) const;

  // =============================================================================
  // MÉTODOS PRIVADOS - UTILITÁRIOS
  // =============================================================================
  
  /**
   * @brief Encontra o landmark mais próximo no mapa
   * @param measurement Medição observada
   * @return Posição do landmark correspondente no mapa
   */
  Eigen::Vector2d find_corresponding_landmark(
    const LandmarkMeasurement& measurement
  ) const;

  /**
   * @brief Normaliza ângulo para [-π, π]
   * @param angle Ângulo em radianos
   * @return Ângulo normalizado
   */
  double normalize_angle(double angle) const;

  /**
   * @brief Verifica se pose está dentro dos limites do campo
   * @param pose Pose a ser verificada
   * @return true se pose é válida
   */
  bool is_pose_within_field(const Eigen::Vector3d& pose) const;

  /**
   * @brief Calcula distância de Mahalanobis
   * @param innovation Vetor de inovação
   * @param covariance Matriz de covariância da inovação
   * @return Distância de Mahalanobis
   */
  double mahalanobis_distance(
    const Eigen::Vector2d& innovation,
    const Eigen::Matrix2d& covariance
  ) const;

  // Inicializa landmarks padrão do campo (declaração faltante)
  void initialize_default_field_landmarks();
};

}  // namespace roboime_navigation

#endif  // ROBOIME_NAVIGATION__EKF_LOCALIZATION_HPP_ 