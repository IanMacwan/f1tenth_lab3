#include <algorithm>
#include <cmath>
#include <memory>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

using ackermann_msgs::msg::AckermannDriveStamped;
using sensor_msgs::msg::LaserScan;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

constexpr double kSideAngle = 90.0 * kDeg;       // beam b (left wall)
constexpr double kTheta = 50.0 * kDeg;           // angle between beams a and b
constexpr double kFrontHalfAngle = 8.0 * kDeg;
constexpr double kMaxSteer = 0.4189;             // rad
constexpr double kFallbackRange = 30.0;          // used if range_max is invalid

constexpr double kIntegralLimit = 1.0;
constexpr double kDerivSmoothing = 0.3;
constexpr double kMinDt = 1e-4;
constexpr double kMaxDt = 0.5;

// Doorway rejection: a sudden jump in left distance with open space ahead.
constexpr double kFrontClear = 5.0;              // m
constexpr double kJump = 0.2;                    // m per scan
constexpr double kReacquire = 0.25;              // m
constexpr double kMaxHold = 1.0;                 // s

constexpr double kSpeedFast = 1.5;               // m/s
constexpr double kSpeedMid = 1.0;
constexpr double kSpeedSlow = 0.5;

double clamp(double value, double low, double high) {
  return std::max(low, std::min(value, high));
}

}  // namespace

class WallFollow : public rclcpp::Node {
 public:
  WallFollow() : Node("wall_follow_node") {
    declare_parameter("kp", 1.2);
    declare_parameter("ki", 0.01);
    declare_parameter("kd", 0.1);
    declare_parameter("desired_distance", 0.9);
    declare_parameter("lookahead", 1.0);
    kp_ = get_parameter("kp").as_double();
    ki_ = get_parameter("ki").as_double();
    kd_ = get_parameter("kd").as_double();
    desired_distance_ = get_parameter("desired_distance").as_double();
    lookahead_ = get_parameter("lookahead").as_double();

    drive_pub_ = create_publisher<AckermannDriveStamped>("/drive", 10);
    scan_sub_ = create_subscription<LaserScan>(
        "/scan", rclcpp::SensorDataQoS(),
        [this](LaserScan::ConstSharedPtr msg) { scan_callback(*msg); });
  }

 private:
  double get_range(const LaserScan& scan, double angle) const {
    const double max_range = scan.range_max > 0.0F ? scan.range_max : kFallbackRange;
    const auto n = scan.ranges.size();
    if (n == 0 || scan.angle_increment <= 0.0F) {
      return max_range;
    }
    const double index = std::round((angle - scan.angle_min) / scan.angle_increment);
    if (index < 0.0 || index > static_cast<double>(n - 1)) {
      return max_range;
    }
    const double range = scan.ranges[static_cast<size_t>(index)];
    if (!std::isfinite(range) || range < scan.range_min) {
      return max_range;
    }
    return std::min(range, max_range);
  }

  double get_front_range(const LaserScan& scan) const {
    const double step = std::max(static_cast<double>(scan.angle_increment), 0.005);
    double nearest = get_range(scan, 0.0);
    for (double angle = -kFrontHalfAngle; angle <= kFrontHalfAngle; angle += step) {
      nearest = std::min(nearest, get_range(scan, angle));
    }
    return nearest;
  }

  // Positive when the car is farther from the left wall than desired.
  double get_error(const LaserScan& scan, double dist) const {
    const double b = get_range(scan, kSideAngle);
    const double a = get_range(scan, kSideAngle - kTheta);
    const double alpha = std::atan2(a * std::cos(kTheta) - b, a * std::sin(kTheta));
    const double current = b * std::cos(alpha);
    const double future = current + lookahead_ * std::sin(alpha);
    return future - dist;
  }

  static double select_speed(double steering) {
    const double degrees = std::abs(steering) / kDeg;
    if (degrees < 10.0) {
      return kSpeedFast;
    }
    if (degrees < 20.0) {
      return kSpeedMid;
    }
    return kSpeedSlow;
  }

  void pid_control(double error, double dt) {
    if (dt > 0.0) {
      integral_ = clamp(integral_ + error * dt, -kIntegralLimit, kIntegralLimit);
      const double raw = (error - prev_error_) / dt;
      derivative_ = kDerivSmoothing * raw + (1.0 - kDerivSmoothing) * derivative_;
    } else {
      derivative_ = 0.0;
    }
    prev_error_ = error;

    const double angle = clamp(
        kp_ * error + ki_ * integral_ + kd_ * derivative_, -kMaxSteer, kMaxSteer);

    AckermannDriveStamped msg;
    msg.header.stamp = now();
    msg.drive.steering_angle = static_cast<float>(angle);
    msg.drive.speed = static_cast<float>(select_speed(angle));
    drive_pub_->publish(msg);
  }

  void scan_callback(const LaserScan& scan) {
    const double stamp = rclcpp::Time(scan.header.stamp).seconds();
    const double dt = has_prev_ ? stamp - prev_stamp_ : 0.0;
    const bool dt_valid = dt > kMinDt && dt < kMaxDt;

    const bool front_open = get_front_range(scan) > kFrontClear;
    const double right = get_range(scan, -kSideAngle);
    double error = get_error(scan, desired_distance_);

    if (holding_) {
      if (!front_open || error < prev_error_ + kReacquire || stamp - hold_start_ > kMaxHold) {
        holding_ = false;
      }
    } else if (dt_valid && front_open && right > desired_distance_ &&
               error - prev_error_ > kJump) {
      holding_ = true;
      hold_start_ = stamp;
    }
    if (holding_) {
      error = prev_error_;
    }

    pid_control(error, dt_valid ? dt : 0.0);
    prev_stamp_ = stamp;
    has_prev_ = true;
  }

  double kp_ = 0.0;
  double ki_ = 0.0;
  double kd_ = 0.0;
  double desired_distance_ = 0.0;
  double lookahead_ = 0.0;

  double integral_ = 0.0;
  double derivative_ = 0.0;
  double prev_error_ = 0.0;

  bool has_prev_ = false;
  double prev_stamp_ = 0.0;
  bool holding_ = false;
  double hold_start_ = 0.0;

  rclcpp::Publisher<AckermannDriveStamped>::SharedPtr drive_pub_;
  rclcpp::Subscription<LaserScan>::SharedPtr scan_sub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WallFollow>());
  rclcpp::shutdown();
  return 0;
}
