#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>

#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <termios.h>
#include <fcntl.h>

using namespace std::chrono_literals;

class ImuDriverNode : public rclcpp::Node {
public:
  ImuDriverNode() : Node("imu_driver") {
    port_ = this->declare_parameter<std::string>("port", "/dev/ttyACM0");
    baud_ = this->declare_parameter<int>("baud", 115200);
    frame_id_ = this->declare_parameter<std::string>("frame_id", "imu_link");
    alpha_ = this->declare_parameter<double>("complementary_alpha", 0.98);
    filter_type_ = this->declare_parameter<std::string>("filter_type", "complementary"); // complementary|madgwick|mahony
    use_mag_ = this->declare_parameter<bool>("use_magnetometer", false); // PADRÃO: sem magnetômetro
    beta_ = this->declare_parameter<double>("madgwick_beta", 0.1);
    kp_ = this->declare_parameter<double>("mahony_kp", 1.0);
    ki_ = this->declare_parameter<double>("mahony_ki", 0.0);
    calib_samples_ = this->declare_parameter<int>("calibration_samples", 300);
    publish_rate_hz_ = this->declare_parameter<double>("publish_rate", 400.0);

    imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("imu/data", 100);
    posture_pub_ = this->create_publisher<std_msgs::msg::String>("motion/robot_posture", 10);

    if (!open_port()) {
      RCLCPP_FATAL(this->get_logger(), "Falha ao abrir porta serial: %s", port_.c_str());
      throw std::runtime_error("serial open failed");
    }

    calibrate_bias();

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / publish_rate_hz_),
      std::bind(&ImuDriverNode::poll_once, this)
    );

    RCLCPP_INFO(this->get_logger(), "imu_driver pronto em %s @ %d (filter=%s)", port_.c_str(), baud_, filter_type_.c_str());
  }

private:
  bool open_port() {
    fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) return false;
    struct termios tio{};
    if (tcgetattr(fd_, &tio) != 0) return false;
    cfmakeraw(&tio);
    speed_t speed = B115200;
    if (baud_ == 230400) speed = B230400;
    if (baud_ == 460800) speed = B460800;
    if (baud_ == 921600) speed = B921600;
    cfsetispeed(&tio, speed);
    cfsetospeed(&tio, speed);
    tio.c_cflag |= (CLOCAL | CREAD);
    tio.c_cc[VTIME] = 0;  // read timeout deciseconds
    tio.c_cc[VMIN] = 0;   // non-blocking
    if (tcsetattr(fd_, TCSANOW, &tio) != 0) return false;
    return true;
  }

  bool readline(std::string &out) {
    out.clear();
    char ch;
    for (;;) {
      ssize_t n = ::read(fd_, &ch, 1);
      if (n == 1) {
        if (ch == '\n') return true;
        if (ch != '\r') out.push_back(ch);
      } else {
        break;
      }
    }
    return !out.empty();
  }

  bool parse_csv(const std::string &line,
                 double &gx, double &gy, double &gz,
                 double &ax, double &ay, double &az,
                 double &mx, double &my, double &mz) {
    // Esperado: gx,gy,gz,ax,ay,az[,mx,my,mz]
    std::vector<double> vals; vals.reserve(9);
    size_t start = 0;
    for (size_t i = 0; i <= line.size(); ++i) {
      if (i == line.size() || line[i] == ',') {
        if (i > start) {
          try { vals.push_back(std::stod(line.substr(start, i - start))); }
          catch (...) { return false; }
        } else {
          return false;
        }
        start = i + 1;
      }
    }
    if (vals.size() < 6) return false;
    gx = vals[0]; gy = vals[1]; gz = vals[2];
    ax = vals[3]; ay = vals[4]; az = vals[5];
    if (vals.size() >= 9) { mx = vals[6]; my = vals[7]; mz = vals[8]; }
    else { mx = my = mz = std::numeric_limits<double>::quiet_NaN(); }
    return true;
  }

  void calibrate_bias() {
    int count = 0;
    double sx = 0, sy = 0, sz = 0;
    auto start = now();
    while (count < calib_samples_) {
      std::string line;
      if (!readline(line)) continue;
      double gx, gy, gz, ax, ay, az, mx, my, mz;
      if (!parse_csv(line, gx, gy, gz, ax, ay, az, mx, my, mz)) continue;
      sx += gx; sy += gy; sz += gz;
      count++;
      if ((now() - start).seconds() > 5.0) break;
    }
    if (count > 0) {
      bias_gx_ = sx / count;
      bias_gy_ = sy / count;
      bias_gz_ = sz / count;
    }
    RCLCPP_INFO(this->get_logger(), "Bias gyro (deg/s): %.3f %.3f %.3f", bias_gx_, bias_gy_, bias_gz_);
  }

  void poll_once() {
    std::string line;
    if (!readline(line)) return;
    double gx_dps, gy_dps, gz_dps, ax, ay, az, mx, my, mz;
    if (!parse_csv(line, gx_dps, gy_dps, gz_dps, ax, ay, az, mx, my, mz)) return;

    // Remover bias e converter p/ rad/s
    double gx = (gx_dps - bias_gx_) * M_PI / 180.0;
    double gy = (gy_dps - bias_gy_) * M_PI / 180.0;
    double gz = (gz_dps - bias_gz_) * M_PI / 180.0;

    // dt
    rclcpp::Time tnow = now();
    if (last_ts_.nanoseconds() == 0) {
      last_ts_ = tnow;
      return;
    }
    double dt = std::max(1e-4, std::min(0.1, (tnow - last_ts_).seconds()));
    last_ts_ = tnow;

    // Atualizar orientação por filtro selecionado
    has_mag_ = use_mag_ && std::isfinite(mx) && std::isfinite(my) && std::isfinite(mz);
    if (filter_type_ == "madgwick" && has_mag_) {
      update_madgwick(gx, gy, gz, ax, ay, az, mx, my, mz, dt);
    } else if (filter_type_ == "mahony" && has_mag_) {
      update_mahony(gx, gy, gz, ax, ay, az, mx, my, mz, dt);
    } else {
      // Complementar: roll/pitch do accel, yaw integrado do gyro
      double norm = std::max(1e-6, std::sqrt(ax*ax + ay*ay + az*az));
      double axn = ax / norm, ayn = ay / norm, azn = az / norm;
      double acc_roll = std::atan2(ayn, azn);
      double acc_pitch = std::atan2(-axn, std::sqrt(ayn*ayn + azn*azn));
      roll_ += gx * dt; pitch_ += gy * dt; yaw_ += gz * dt;
      roll_  = alpha_ * roll_  + (1 - alpha_) * acc_roll;
      pitch_ = alpha_ * pitch_ + (1 - alpha_) * acc_pitch;
      // Converter RPY → quaternion
      double cy = std::cos(yaw_ * 0.5),  sy = std::sin(yaw_ * 0.5);
      double cp = std::cos(pitch_ * 0.5), sp = std::sin(pitch_ * 0.5);
      double cr = std::cos(roll_ * 0.5),  sr = std::sin(roll_ * 0.5);
      qw_ = cr*cp*cy + sr*sp*sy;
      qx_ = sr*cp*cy - cr*sp*sy;
      qy_ = cr*sp*cy + sr*cp*sy;
      qz_ = cr*cp*sy - sr*sp*cy;
    }

    // Publicar IMU
    auto msg = sensor_msgs::msg::Imu();
    msg.header.stamp = tnow;
    msg.header.frame_id = frame_id_;
    msg.orientation.w = qw_;
    msg.orientation.x = qx_;
    msg.orientation.y = qy_;
    msg.orientation.z = qz_;
    msg.angular_velocity.x = gx;
    msg.angular_velocity.y = gy;
    msg.angular_velocity.z = gz;
    msg.linear_acceleration.x = ax;
    msg.linear_acceleration.y = ay;
    msg.linear_acceleration.z = az;
    imu_pub_->publish(msg);

    // Postura simples
    auto posture = std_msgs::msg::String();
    // Extrair roll/pitch do quaternion
    double sinr_cosp = 2.0 * (qw_ * qx_ + qy_ * qz_);
    double cosr_cosp = 1.0 - 2.0 * (qx_ * qx_ + qy_ * qy_);
    double roll = std::atan2(sinr_cosp, cosr_cosp);
    double sinp = 2.0 * (qw_ * qy_ - qz_ * qx_);
    double pitch = std::abs(sinp) >= 1 ? std::copysign(M_PI/2.0, sinp) : std::asin(sinp);
    double roll_deg = std::abs(roll * 180.0 / M_PI);
    double pitch_deg = std::abs(pitch * 180.0 / M_PI);
    if (roll_deg > 45.0 || pitch_deg > 45.0) posture.data = "fallen";
    else if (roll_deg < 15.0 && pitch_deg < 15.0) posture.data = "standing";
    else posture.data = "unknown";
    posture_pub_->publish(posture);
  }

private:
  // Parâmetros
  std::string port_;
  int baud_{};
  std::string frame_id_;
  double alpha_{};
  std::string filter_type_;
  bool use_mag_{};
  double beta_{}; // Madgwick gain
  double kp_{};   // Mahony proportional gain
  double ki_{};   // Mahony integral gain
  int calib_samples_{};
  double publish_rate_hz_{};

  // Serial
  int fd_{-1};
  bool has_mag_{false};

  // Publicadores
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr posture_pub_;

  // Timer
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time last_ts_;

  // Estado filtro
  double bias_gx_{0}, bias_gy_{0}, bias_gz_{0};
  double roll_{0}, pitch_{0}, yaw_{0};
  double qw_{1.0}, qx_{0.0}, qy_{0.0}, qz_{0.0};
  // Erro integral para Mahony
  double ex_int_{0.0}, ey_int_{0.0}, ez_int_{0.0};

  // Filtros
  void normalize3(double &x, double &y, double &z) {
    double n = std::sqrt(x*x + y*y + z*z);
    if (n < 1e-9) return;
    x/=n; y/=n; z/=n;
  }

  void update_madgwick(double gx, double gy, double gz,
                       double ax, double ay, double az,
                       double mx, double my, double mz,
                       double dt) {
    // Implementação compacta baseada no paper de Madgwick (2010)
    // Normalizar vetores
    normalize3(ax, ay, az);
    normalize3(mx, my, mz);

    double q1 = qw_, q2 = qx_, q3 = qy_, q4 = qz_;

    // Referência: https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/
    double _2q1mx = 2.0*q1*mx;
    double _2q1my = 2.0*q1*my;
    double _2q1mz = 2.0*q1*mz;
    double _2q2mx = 2.0*q2*mx;
    const double _2 = 2.0; // constante auxiliar para manter notação compacta
    double hx = mx*q1*q1 - _2q1my*q4 + _2q1mz*q3 + mx*q2*q2 + _2*q2*my*q3 + _2*q2*mz*q4 - mx*q3*q3 - mx*q4*q4;
    double hy = _2q1mx*q4 + my*q1*q1 - _2q1mz*q2 + _2*q2*mx*q3 - my*q2*q2 + my*q3*q3 + _2*q3*mz*q4 - my*q4*q4;
    double _2bx = std::sqrt(hx*hx + hy*hy);
    double _2bz = -_2q1mx*q3 + _2q1my*q2 + mz*q1*q1 + _2*q2*mx*q4 - mz*q2*q2 + _2*q3*my*q4 - mz*q3*q3 + mz*q4*q4;
    double _4bx = 2.0*_2bx;
    double _4bz = 2.0*_2bz;

    // Gradiente descendente (simplificado)
    double s1 = -_2*(q3*(2*(q2*q4 - q1*q3) - ax) - q2*(2*(q1*q2 + q3*q4) - ay))
               - _2bz*q3*(_2bx*(0.5 - q3*q3 - q4*q4) + _2bz*(q2*q4 - q1*q3) - mx)
               + (-_2bx*q4 + _2bz*q2)*(_2bx*(q2*q3 - q1*q4) + _2bz*(q1*q2 + q3*q4) - my)
               + _2bx*q3*(_2bx*(q1*q3 + q2*q4) + _2bz*(0.5 - q2*q2 - q3*q3) - mz);
    double s2 =  _2*(q4*(2*(q2*q4 - q1*q3) - ax) + q1*(2*(q1*q2 + q3*q4) - ay))
               + _2bz*q4*(_2bx*(0.5 - q3*q3 - q4*q4) + _2bz*(q2*q4 - q1*q3) - mx)
               + (_2bx*q3 + _2bz*q1)*(_2bx*(q2*q3 - q1*q4) + _2bz*(q1*q2 + q3*q4) - my)
               + (_2bx*q4 - _4bz*q2)*(_2bx*(q1*q3 + q2*q4) + _2bz*(0.5 - q2*q2 - q3*q3) - mz);
    double s3 = (-_2)*(q1*(2*(q2*q4 - q1*q3) - ax) - q4*(2*(q1*q2 + q3*q4) - ay))
               - (_4bx*q3 + _2bz*q1)*(_2bx*(0.5 - q3*q3 - q4*q4) + _2bz*(q2*q4 - q1*q3) - mx)
               + (_2bx*q2 + _2bz*q4)*(_2bx*(q2*q3 - q1*q4) + _2bz*(q1*q2 + q3*q4) - my)
               + (_2bx*q1 - _4bz*q3)*(_2bx*(q1*q3 + q2*q4) + _2bz*(0.5 - q2*q2 - q3*q3) - mz);
    double s4 =  _2*(q2*(2*(q2*q4 - q1*q3) - ax) + q3*(2*(q1*q2 + q3*q4) - ay))
               + (-_4bx*q4 + _2bz*q2)*(_2bx*(0.5 - q3*q3 - q4*q4) + _2bz*(q2*q4 - q1*q3) - mx)
               + (-_2bx*q1 + _2bz*q3)*(_2bx*(q2*q3 - q1*q4) + _2bz*(q1*q2 + q3*q4) - my)
               + _2bx*q2*(_2bx*(q1*q3 + q2*q4) + _2bz*(0.5 - q2*q2 - q3*q3) - mz);
    // Normalizar gradiente
    double normS = std::sqrt(s1*s1 + s2*s2 + s3*s3 + s4*s4);
    if (normS > 1e-9) { s1/=normS; s2/=normS; s3/=normS; s4/=normS; }

    // Atualizar taxa de mudança do quaternion
    double qDot1 = 0.5*(-q2*gx - q3*gy - q4*gz) - beta_*s1;
    double qDot2 = 0.5*( q1*gx + q3*gz - q4*gy) - beta_*s2;
    double qDot3 = 0.5*( q1*gy - q2*gz + q4*gx) - beta_*s3;
    double qDot4 = 0.5*( q1*gz + q2*gy - q3*gx) - beta_*s4;

    // Integrar
    q1 += qDot1 * dt;
    q2 += qDot2 * dt;
    q3 += qDot3 * dt;
    q4 += qDot4 * dt;
    double nq = std::sqrt(q1*q1 + q2*q2 + q3*q3 + q4*q4);
    qw_ = q1/nq; qx_ = q2/nq; qy_ = q3/nq; qz_ = q4/nq;
  }

  void update_mahony(double gx, double gy, double gz,
                     double ax, double ay, double az,
                     double mx, double my, double mz,
                     double dt) {
    // Normalizar
    normalize3(ax, ay, az);
    normalize3(mx, my, mz);

    // Estimar direção do campo magnético e gravidade
    double q1 = qw_, q2 = qx_, q3 = qy_, q4 = qz_;
    // Vetor gravidade estimado
    double vx = 2*(q2*q4 - q1*q3);
    double vy = 2*(q1*q2 + q3*q4);
    double vz = q1*q1 - q2*q2 - q3*q3 + q4*q4;

    // Referência de campo magnético
    double hx = 2*mx*(0.5 - q3*q3 - q4*q4) + 2*my*(q2*q3 - q1*q4) + 2*mz*(q2*q4 + q1*q3);
    double hy = 2*mx*(q2*q3 + q1*q4) + 2*my*(0.5 - q2*q2 - q4*q4) + 2*mz*(q3*q4 - q1*q2);
    double bx = std::sqrt(hx*hx + hy*hy);
    double bz = 2*mx*(q2*q4 - q1*q3) + 2*my*(q3*q4 + q1*q2) + 2*mz*(0.5 - q2*q2 - q3*q3);
    // Vetor magnético estimado
    double wx = 2*bx*(0.5 - q3*q3 - q4*q4) + 2*bz*(q2*q4 - q1*q3);
    double wy = 2*bx*(q2*q3 - q1*q4) + 2*bz*(q1*q2 + q3*q4);
    double wz = 2*bx*(q1*q3 + q2*q4) + 2*bz*(0.5 - q2*q2 - q3*q3);

    // Erro = cruz entre medido (a,m) e estimado (g,b)
    double ex = (ay*vz - az*vy) + (my*wz - mz*wy);
    double ey = (az*vx - ax*vz) + (mz*wx - mx*wz);
    double ez = (ax*vy - ay*vx) + (mx*wy - my*wx);

    // Integral
    if (ki_ > 0.0) { ex_int_ += ex*dt; ey_int_ += ey*dt; ez_int_ += ez*dt; }
    else { ex_int_ = ey_int_ = ez_int_ = 0.0; }

    // Retroalimentação
    gx += kp_*ex + ki_*ex_int_;
    gy += kp_*ey + ki_*ey_int_;
    gz += kp_*ez + ki_*ez_int_;

    // Integrar quaternion
    double qDot1 = 0.5*(-q2*gx - q3*gy - q4*gz);
    double qDot2 = 0.5*( q1*gx + q3*gz - q4*gy);
    double qDot3 = 0.5*( q1*gy - q2*gz + q4*gx);
    double qDot4 = 0.5*( q1*gz + q2*gy - q3*gx);
    q1 += qDot1 * dt; q2 += qDot2 * dt; q3 += qDot3 * dt; q4 += qDot4 * dt;
    double nq = std::sqrt(q1*q1 + q2*q2 + q3*q3 + q4*q4);
    qw_ = q1/nq; qx_ = q2/nq; qy_ = q3/nq; qz_ = q4/nq;
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImuDriverNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}


