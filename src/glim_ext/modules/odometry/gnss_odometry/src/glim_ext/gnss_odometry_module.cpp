#include <deque>
#include <mutex>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sensor_msgs/msg/nav_sat_fix.hpp>

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/FixedLagSmoother.h>
#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_with_fallback.hpp>

#include <glim/odometry/callbacks.hpp>
#include <glim/odometry/estimation_frame.hpp>
#include <glim/util/concurrent_vector.hpp>
#include <glim/util/convert_to_string.hpp>
#include <glim/util/extension_module_ros2.hpp>
#include <glim/util/logging.hpp>
#include <glim_ext/geodetic.hpp>
#include <glim_ext/util/config_ext.hpp>

#include <spdlog/spdlog.h>

namespace glim {

using gtsam::symbol_shorthand::X;
using NavSatFix = sensor_msgs::msg::NavSatFix;
using NavSatFixConstPtr = sensor_msgs::msg::NavSatFix::ConstSharedPtr;

template <typename Stamp>
double stamp_to_sec(const Stamp& stamp) {
  return stamp.sec + stamp.nanosec / 1e9;
}

struct GNSSOdomData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double stamp = 0.0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Vector3d variance = Eigen::Vector3d::Ones();
};

struct OdomGNSSFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  long id = -1;
  double stamp = 0.0;
  Eigen::Isometry3d T_world_imu = Eigen::Isometry3d::Identity();
  GNSSOdomData gnss;
  bool has_gnss = false;
  bool factor_inserted = false;
};

struct OdomGNSSPrior {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  long frame_id = -1;
  Eigen::Vector3d target = Eigen::Vector3d::Zero();
  Eigen::Vector3d lever_arm = Eigen::Vector3d::Zero();
  gtsam::NonlinearFactor::shared_ptr factor;
};

class GNSSOdometry : public ExtensionModuleROS2 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GNSSOdometry() : logger(create_module_logger("gnss_odometry")) {
    const std::string config_path = GlobalConfigExt::get_config_path("config_gnss_odometry");
    logger->info("gnss_odometry_config_path={}", config_path);

    Config config(config_path);
    gnss_topic = config.param<std::string>("gnss_odometry", "gnss_topic", "/fix");
    min_navsat_status = config.param<int>("gnss_odometry", "min_navsat_status", 2);
    max_std_xy = config.param<double>("gnss_odometry", "max_std_xy", std::numeric_limits<double>::infinity());
    max_std_z = config.param<double>("gnss_odometry", "max_std_z", std::numeric_limits<double>::infinity());
    use_z = config.param<bool>("gnss_odometry", "use_z", false);
    min_alignment_samples = config.param<int>("gnss_odometry", "min_alignment_samples", 10);
    min_alignment_baseline = config.param<double>("gnss_odometry", "min_alignment_baseline", 50.0);
    max_interpolation_interval = config.param<double>("gnss_odometry", "max_interpolation_interval", 0.25);
    max_factor_age = config.param<double>("gnss_odometry", "max_factor_age", 3.0);
    min_factor_interval = config.param<double>("gnss_odometry", "min_factor_interval", 0.5);
    min_factor_distance = config.param<double>("gnss_odometry", "min_factor_distance", 5.0);
    robust_kernel_width = config.param<double>("gnss_odometry", "robust_kernel_width", 5.0);
    prior_inf_scale = config.param<Eigen::Vector3d>("gnss_odometry", "prior_inf_scale", Eigen::Vector3d(1.0, 1.0, 0.0));
    min_stddev = config.param<Eigen::Vector3d>("gnss_odometry", "min_stddev", Eigen::Vector3d(0.5, 0.5, 5.0));
    unknown_stddev = config.param<Eigen::Vector3d>("gnss_odometry", "unknown_stddev", Eigen::Vector3d(5.0, 5.0, 20.0));
    gnss_position_in_imu = config.param<Eigen::Vector3d>("gnss_odometry", "gnss_position_in_imu", Eigen::Vector3d::Zero());

    navsat_origin_initialized = false;
    transformation_initialized = false;
    T_world_gnss.setIdentity();

    logger->info(
      "gnss_topic={} use_z={} min_status={} max_std_xy={} max_std_z={} min_factor_interval={} min_factor_distance={}",
      gnss_topic,
      use_z,
      min_navsat_status,
      max_std_xy,
      max_std_z,
      min_factor_interval,
      min_factor_distance);

    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    using std::placeholders::_4;
    OdometryEstimationCallbacks::on_new_frame.add(std::bind(&GNSSOdometry::on_new_frame, this, _1));
    OdometryEstimationCallbacks::on_update_frames.add(std::bind(&GNSSOdometry::on_update_frames, this, _1));
    OdometryEstimationCallbacks::on_marginalized_frames.add(std::bind(&GNSSOdometry::on_marginalized_frames, this, _1));
    OdometryEstimationCallbacks::on_smoother_update.add(std::bind(&GNSSOdometry::on_smoother_update, this, _1, _2, _3, _4));
  }

  virtual std::vector<GenericTopicSubscription::Ptr> create_subscriptions() override {
    const auto sub = std::make_shared<TopicSubscription<NavSatFix>>(gnss_topic, "sensor_msgs/msg/NavSatFix", [this](const NavSatFixConstPtr msg) { navsat_callback(msg); });
    return {sub};
  }

  void navsat_callback(const NavSatFixConstPtr& fix_msg) {
    if (fix_msg->status.status < min_navsat_status) {
      log_rejected_gnss("status");
      return;
    }

    if (!std::isfinite(fix_msg->latitude) || !std::isfinite(fix_msg->longitude) || !std::isfinite(fix_msg->altitude)) {
      log_rejected_gnss("non-finite NavSatFix");
      return;
    }

    GNSSOdomData gnss;
    gnss.stamp = stamp_to_sec(fix_msg->header.stamp);
    gnss.variance << fix_msg->position_covariance[0], fix_msg->position_covariance[4], fix_msg->position_covariance[8];
    sanitize_variance(gnss.variance, fix_msg->position_covariance_type == NavSatFix::COVARIANCE_TYPE_UNKNOWN);

    if (!accept_gnss(gnss)) {
      return;
    }

    std::lock_guard<std::mutex> lock(data_mutex);
    if (!navsat_origin_initialized) {
      initialize_navsat_origin(fix_msg->latitude, fix_msg->longitude, fix_msg->altitude);
    }

    gnss.position = navsat_to_local(fix_msg->latitude, fix_msg->longitude, fix_msg->altitude);
    gnss_queue.push_back(gnss);
    process_queues_locked();
  }

  void on_new_frame(const EstimationFrame::ConstPtr& frame) {
    if (frame->frame_id != FrameID::IMU) {
      logger->warn("gnss_odometry supports only IMU-frame odometry");
      return;
    }

    std::lock_guard<std::mutex> lock(data_mutex);
    latest_frame_stamp = std::max(latest_frame_stamp, frame->stamp);
    active_frame_ids.insert(frame->id);

    OdomGNSSFrame data;
    data.id = frame->id;
    data.stamp = frame->stamp;
    data.T_world_imu = frame->T_world_imu;
    frames.push_back(data);

    process_queues_locked();
  }

  void on_update_frames(const std::vector<EstimationFrame::ConstPtr>& active_frames) {
    std::lock_guard<std::mutex> lock(data_mutex);
    active_frame_ids.clear();
    for (const auto& frame : active_frames) {
      if (!frame) {
        continue;
      }
      active_frame_ids.insert(frame->id);
      latest_frame_stamp = std::max(latest_frame_stamp, frame->stamp);
      for (auto& stored : frames) {
        if (stored.id == frame->id) {
          stored.T_world_imu = frame->T_world_imu;
          break;
        }
      }
    }

    process_queues_locked();
  }

  void on_marginalized_frames(const std::vector<EstimationFrame::ConstPtr>& marginalized_frames) {
    std::lock_guard<std::mutex> lock(data_mutex);
    for (const auto& frame : marginalized_frames) {
      if (frame) {
        active_frame_ids.erase(frame->id);
      }
    }
    trim_frames_locked();
  }

  void on_smoother_update(
    gtsam_points::IncrementalFixedLagSmootherExtWithFallback& smoother,
    gtsam::NonlinearFactorGraph& new_factors,
    gtsam::Values& new_values,
    gtsam::FixedLagSmootherKeyTimestampMap& new_stamps) {
    const auto priors = output_priors.get_all_and_clear();
    if (priors.empty()) {
      return;
    }

    logger->debug("insert {} GNSS odometry priors", priors.size());
    for (const auto& prior : priors) {
      new_factors.add(prior.factor);

      const auto key = X(prior.frame_id);
      if (new_values.exists(key)) {
        const auto pose = new_values.at<gtsam::Pose3>(key);
        const Eigen::Vector3d body_position = prior.target - pose.rotation().matrix() * prior.lever_arm;
        new_values.update(key, gtsam::Pose3(pose.rotation(), gtsam::Point3(body_position)));
      }
    }
  }

private:
  void process_queues_locked() {
    associate_gnss_locked();

    if (!transformation_initialized) {
      try_initialize_transform_locked();
    }

    if (transformation_initialized) {
      create_pending_factors_locked();
    }

    trim_gnss_locked();
    trim_frames_locked();
  }

  void associate_gnss_locked() {
    if (gnss_queue.size() < 2 || frames.empty()) {
      return;
    }

    for (auto& frame : frames) {
      if (frame.has_gnss) {
        continue;
      }

      if (frame.stamp < gnss_queue.front().stamp) {
        continue;
      }

      while (gnss_queue.size() >= 2 && gnss_queue[1].stamp < frame.stamp) {
        gnss_queue.pop_front();
      }

      if (gnss_queue.size() < 2 || frame.stamp > gnss_queue.back().stamp) {
        break;
      }

      const auto right = std::lower_bound(gnss_queue.begin(), gnss_queue.end(), frame.stamp, [](const GNSSOdomData& gnss, const double stamp) { return gnss.stamp < stamp; });
      if (right == gnss_queue.end() || right == gnss_queue.begin()) {
        continue;
      }

      const auto left = right - 1;
      const double interval = right->stamp - left->stamp;
      if (interval <= 0.0 || interval > max_interpolation_interval) {
        continue;
      }

      const double ratio = (frame.stamp - left->stamp) / interval;
      frame.gnss.stamp = frame.stamp;
      frame.gnss.position = (1.0 - ratio) * left->position + ratio * right->position;
      frame.gnss.variance = (1.0 - ratio) * left->variance + ratio * right->variance;
      frame.has_gnss = true;
    }
  }

  void try_initialize_transform_locked() {
    std::vector<const OdomGNSSFrame*> pairs;
    pairs.reserve(frames.size());

    for (const auto& frame : frames) {
      if (frame.has_gnss) {
        pairs.push_back(&frame);
      }
    }

    if (pairs.size() < static_cast<size_t>(std::max(2, min_alignment_samples))) {
      return;
    }

    double max_baseline = 0.0;
    for (const auto* frame : pairs) {
      max_baseline = std::max(max_baseline, (frame->gnss.position - pairs.front()->gnss.position).head<2>().norm());
    }

    if (max_baseline < min_alignment_baseline) {
      return;
    }

    Eigen::Vector3d mean_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d mean_gnss = Eigen::Vector3d::Zero();
    for (const auto* frame : pairs) {
      mean_world += frame->T_world_imu.translation();
      mean_gnss += frame->gnss.position;
    }
    mean_world /= pairs.size();
    mean_gnss /= pairs.size();

    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (const auto* frame : pairs) {
      const Eigen::Vector2d centered_world = (frame->T_world_imu.translation() - mean_world).head<2>();
      const Eigen::Vector2d centered_gnss = (frame->gnss.position - mean_gnss).head<2>();
      cov += centered_world * centered_gnss.transpose();
    }
    cov /= pairs.size();

    const Eigen::JacobiSVD<Eigen::Matrix2d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix2d U = svd.matrixU();
    const Eigen::Matrix2d V = svd.matrixV();
    Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
    if (U.determinant() * V.determinant() < 0.0) {
      S(1, 1) = -1.0;
    }

    T_world_gnss.setIdentity();
    T_world_gnss.linear().block<2, 2>(0, 0) = U * S * V.transpose();
    T_world_gnss.translation() = mean_world - T_world_gnss.linear() * mean_gnss;

    transformation_initialized = true;
    logger->info("initialized GNSS odometry alignment with {} samples baseline={:.3f}m", pairs.size(), max_baseline);
    logger->info("T_world_gnss={}", convert_to_string(T_world_gnss));
  }

  void create_pending_factors_locked() {
    for (auto& frame : frames) {
      if (!frame.has_gnss || frame.factor_inserted) {
        continue;
      }

      if (!is_frame_active_locked(frame)) {
        continue;
      }

      const Eigen::Vector3d target = T_world_gnss * frame.gnss.position;
      if (!should_insert_factor_locked(frame, target)) {
        frame.factor_inserted = true;
        continue;
      }

      const auto noise = create_noise_model(frame.gnss.variance);
      OdomGNSSPrior prior;
      prior.frame_id = frame.id;
      prior.target = target;
      prior.lever_arm = gnss_position_in_imu;
      prior.factor.reset(new gtsam::GPSFactorArm(X(frame.id), gtsam::Point3(target), gtsam::Point3(gnss_position_in_imu), noise));
      output_priors.push_back(prior);

      frame.factor_inserted = true;
      last_factor_stamp = frame.stamp;
      last_factor_target = target;
      has_last_factor = true;
      num_inserted_priors++;

      if (num_inserted_priors <= 10 || num_inserted_priors % 100 == 0) {
        logger->info("GNSS odom factor #{} frame={} stamp={:.3f} target={}", num_inserted_priors, frame.id, frame.stamp, convert_to_string(target));
      }
    }
  }

  bool should_insert_factor_locked(const OdomGNSSFrame& frame, const Eigen::Vector3d& target) const {
    if (!has_last_factor) {
      return true;
    }

    if (frame.stamp - last_factor_stamp >= min_factor_interval) {
      return true;
    }

    if ((target - last_factor_target).head<2>().norm() >= min_factor_distance) {
      return true;
    }

    return false;
  }

  bool is_frame_active_locked(const OdomGNSSFrame& frame) const {
    if (active_frame_ids.find(frame.id) != active_frame_ids.end()) {
      return true;
    }

    return latest_frame_stamp > frame.stamp && latest_frame_stamp - frame.stamp < max_factor_age;
  }

  void trim_gnss_locked() {
    if (frames.empty()) {
      return;
    }

    const double oldest_needed = frames.front().stamp - max_interpolation_interval;
    while (gnss_queue.size() > 2 && gnss_queue[1].stamp < oldest_needed) {
      gnss_queue.pop_front();
    }
  }

  void trim_frames_locked() {
    while (!frames.empty()) {
      const bool active = is_frame_active_locked(frames.front());
      const bool too_old = latest_frame_stamp > frames.front().stamp && latest_frame_stamp - frames.front().stamp > max_factor_age;
      if (active || !too_old) {
        break;
      }
      frames.pop_front();
    }
  }

  void sanitize_variance(Eigen::Vector3d& variance, const bool unknown_covariance) const {
    if (unknown_covariance) {
      variance = unknown_stddev.array().square().matrix();
      return;
    }

    for (int i = 0; i < 3; i++) {
      if (!std::isfinite(variance[i]) || variance[i] <= 0.0) {
        variance[i] = unknown_stddev[i] * unknown_stddev[i];
      }
    }
  }

  bool accept_gnss(const GNSSOdomData& data) {
    if (!std::isfinite(data.stamp) || !data.variance.allFinite()) {
      log_rejected_gnss("non-finite data");
      return false;
    }

    const Eigen::Vector3d stddev = data.variance.cwiseMax(Eigen::Vector3d::Zero()).cwiseSqrt();
    if (std::max(stddev.x(), stddev.y()) > max_std_xy) {
      log_rejected_gnss("xy covariance gate");
      return false;
    }

    if (use_z && stddev.z() > max_std_z) {
      log_rejected_gnss("z covariance gate");
      return false;
    }

    return true;
  }

  void log_rejected_gnss(const std::string& reason) {
    num_rejected_gnss++;
    if (num_rejected_gnss <= 10 || num_rejected_gnss % 100 == 0) {
      logger->debug("reject GNSS sample: {} (count={})", reason, num_rejected_gnss);
    }
  }

  void initialize_navsat_origin(const double lat, const double lon, const double alt) {
    navsat_origin_initialized = true;
    navsat_origin_lat = lat * M_PI / 180.0;
    navsat_origin_lon = lon * M_PI / 180.0;
    navsat_origin_alt = alt;
    navsat_origin_ecef = wgs84_to_ecef(lat, lon, alt);

    const double sin_lat = std::sin(navsat_origin_lat);
    const double cos_lat = std::cos(navsat_origin_lat);
    const double sin_lon = std::sin(navsat_origin_lon);
    const double cos_lon = std::cos(navsat_origin_lon);

    navsat_east << -sin_lon, cos_lon, 0.0;
    navsat_north << -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat;
    navsat_up << cos_lat * cos_lon, cos_lat * sin_lon, sin_lat;

    logger->info("NavSatFix origin lat={} lon={} alt={}", lat, lon, alt);
  }

  Eigen::Vector3d navsat_to_local(const double lat, const double lon, const double alt) const {
    const Eigen::Vector3d ecef = wgs84_to_ecef(lat, lon, alt);
    const Eigen::Vector3d delta = ecef - navsat_origin_ecef;

    Eigen::Vector3d local;
    local << navsat_east.dot(delta), navsat_north.dot(delta), alt - navsat_origin_alt;
    return local;
  }

  gtsam::SharedNoiseModel create_noise_model(const Eigen::Vector3d& variance) const {
    Eigen::Vector3d stddev = variance.cwiseMax(Eigen::Vector3d::Zero()).cwiseSqrt();
    stddev = stddev.cwiseMax(min_stddev);

    Eigen::Vector3d info = prior_inf_scale.cwiseQuotient(stddev.array().square().matrix());
    if (!use_z) {
      info.z() = 1e-9;
    }

    gtsam::SharedNoiseModel model = gtsam::noiseModel::Diagonal::Information(info.asDiagonal());
    if (robust_kernel_width > 0.0) {
      model = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(robust_kernel_width), model);
    }

    return model;
  }

  std::mutex data_mutex;

  ConcurrentVector<OdomGNSSPrior, Eigen::aligned_allocator<OdomGNSSPrior>> output_priors;
  std::deque<GNSSOdomData, Eigen::aligned_allocator<GNSSOdomData>> gnss_queue;
  std::deque<OdomGNSSFrame, Eigen::aligned_allocator<OdomGNSSFrame>> frames;
  std::unordered_set<long> active_frame_ids;

  std::string gnss_topic;
  int min_navsat_status;
  double max_std_xy;
  double max_std_z;
  bool use_z;
  int min_alignment_samples;
  double min_alignment_baseline;
  double max_interpolation_interval;
  double max_factor_age;
  double min_factor_interval;
  double min_factor_distance;
  double robust_kernel_width;
  Eigen::Vector3d prior_inf_scale;
  Eigen::Vector3d min_stddev;
  Eigen::Vector3d unknown_stddev;
  Eigen::Vector3d gnss_position_in_imu;

  bool navsat_origin_initialized;
  double navsat_origin_lat = 0.0;
  double navsat_origin_lon = 0.0;
  double navsat_origin_alt = 0.0;
  Eigen::Vector3d navsat_origin_ecef = Eigen::Vector3d::Zero();
  Eigen::Vector3d navsat_east = Eigen::Vector3d::UnitX();
  Eigen::Vector3d navsat_north = Eigen::Vector3d::UnitY();
  Eigen::Vector3d navsat_up = Eigen::Vector3d::UnitZ();

  bool transformation_initialized;
  Eigen::Isometry3d T_world_gnss;

  double latest_frame_stamp = -std::numeric_limits<double>::infinity();
  bool has_last_factor = false;
  double last_factor_stamp = -std::numeric_limits<double>::infinity();
  Eigen::Vector3d last_factor_target = Eigen::Vector3d::Zero();

  size_t num_rejected_gnss = 0;
  size_t num_inserted_priors = 0;

  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::GNSSOdometry();
}
