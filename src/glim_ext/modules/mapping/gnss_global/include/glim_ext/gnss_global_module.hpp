#include <deque>
#include <atomic>
#include <thread>
#include <mutex>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>
#include <Eigen/Core>

#define GLIM_ROS2

#include <boost/format.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/concurrent_vector.hpp>

#ifdef GLIM_ROS2
#include <glim/util/extension_module_ros2.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>

using ExtensionModuleBase = glim::ExtensionModuleROS2;
using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
using PoseWithCovarianceStampedConstPtr = geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr;
using NavSatFix = sensor_msgs::msg::NavSatFix;
using NavSatFixConstPtr = sensor_msgs::msg::NavSatFix::ConstSharedPtr;

template <typename Stamp>
double to_sec(const Stamp& stamp) {
  return stamp.sec + stamp.nanosec / 1e9;
}
#else
#include <glim/util/extension_module_ros.hpp>
#include <geometry_msgs/PoseWithCovarianceStamped.hpp>

using ExtensionModuleBase = glim::ExtensionModuleROS;
#endif

#include <spdlog/spdlog.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PoseTranslationPrior.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <glim/util/logging.hpp>
#include <glim/util/convert_to_string.hpp>
#include <glim_ext/util/config_ext.hpp>
#include <glim_ext/geodetic.hpp>

namespace glim {

using gtsam::symbol_shorthand::X;

struct GNSSData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double stamp = 0.0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Vector3d variance = Eigen::Vector3d::Ones();
};

struct GNSSPriorData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int submap_id = -1;
  Eigen::Vector3d target = Eigen::Vector3d::Zero();
  gtsam::NonlinearFactor::shared_ptr factor;
};

/**
 * @brief GNSS translation constraints for global submap optimization.
 *
 * The module accepts either geometry_msgs/PoseWithCovarianceStamped in a local
 * metric frame or sensor_msgs/NavSatFix. NavSatFix is converted to a local ENU
 * tangent frame at the first accepted fix. GNSS covariance is used for gating
 * and for the translation prior information matrix.
 */
class GNSSGlobal : public ExtensionModuleBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GNSSGlobal() : logger(create_module_logger("gnss_global")) {
    logger->info("initializing GNSS global constraints");
    const std::string config_path = glim::GlobalConfigExt::get_config_path("config_gnss_global");
    logger->info("gnss_global_config_path={}", config_path);

    glim::Config config(config_path);
    gnss_topic = config.param<std::string>("gnss", "gnss_topic", "/pose_with_cov");
    gnss_message_type = config.param<std::string>("gnss", "gnss_message_type", "pose");
    prior_inf_scale = config.param<Eigen::Vector3d>("gnss", "prior_inf_scale", Eigen::Vector3d(1e3, 1e3, 0.0));
    min_stddev = config.param<Eigen::Vector3d>("gnss", "min_stddev", Eigen::Vector3d(0.2, 0.2, 1.0));
    unknown_stddev = config.param<Eigen::Vector3d>("gnss", "unknown_stddev", Eigen::Vector3d(5.0, 5.0, 20.0));
    gnss_position_in_base = config.param<Eigen::Vector3d>("gnss", "gnss_position_in_base", Eigen::Vector3d::Zero());
    min_baseline = config.param<double>("gnss", "min_baseline", 5.0);
    max_std_xy = config.param<double>("gnss", "max_std_xy", std::numeric_limits<double>::infinity());
    max_std_z = config.param<double>("gnss", "max_std_z", std::numeric_limits<double>::infinity());
    robust_kernel_width = config.param<double>("gnss", "robust_kernel_width", 1.0);
    min_navsat_status = config.param<int>("gnss", "min_navsat_status", -1);

    use_z = config.param<bool>("gnss", "use_z", false);
    use_gnss_position_in_base = config.param<bool>("gnss", "use_gnss_position_in_base", gnss_position_in_base.norm() > 1e-6);

    transformation_initialized = false;
    T_world_utm.setIdentity();
    navsat_origin_initialized = false;

    logger->info(
      "gnss_topic={} gnss_message_type={} use_z={} max_std_xy={} max_std_z={} robust_kernel_width={}",
      gnss_topic,
      gnss_message_type,
      use_z,
      max_std_xy,
      max_std_z,
      robust_kernel_width);

    kill_switch = false;

    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    GlobalMappingCallbacks::on_insert_submap.add(std::bind(&GNSSGlobal::on_insert_submap, this, _1));
    GlobalMappingCallbacks::on_smoother_update.add(std::bind(&GNSSGlobal::on_smoother_update, this, _1, _2, _3));
  }
  ~GNSSGlobal() {
    kill_switch = true;
    if (thread.joinable()) {
      thread.join();
    }
  }

  virtual std::vector<GenericTopicSubscription::Ptr> create_subscriptions() override {
    if (gnss_message_type == "navsatfix" || gnss_message_type == "NavSatFix" || gnss_message_type == "fix") {
      const auto sub = std::make_shared<TopicSubscription<NavSatFix>>(gnss_topic, "sensor_msgs/msg/NavSatFix", [this](const NavSatFixConstPtr msg) { navsat_callback(msg); });
      return {sub};
    }

    const auto sub = std::make_shared<TopicSubscription<PoseWithCovarianceStamped>>(
      gnss_topic,
      "geometry_msgs/msg/PoseWithCovarianceStamped",
      [this](const PoseWithCovarianceStampedConstPtr msg) { gnss_callback(msg); });
    return {sub};
  }

  void gnss_callback(const PoseWithCovarianceStampedConstPtr& gnss_msg) {
    GNSSData gnss_data;
    const double stamp = to_sec(gnss_msg->header.stamp);
    const auto& pos = gnss_msg->pose.pose.position;

    gnss_data.stamp = stamp;
    gnss_data.position << pos.x, pos.y, pos.z;
    gnss_data.variance << gnss_msg->pose.covariance[0], gnss_msg->pose.covariance[7], gnss_msg->pose.covariance[14];
    sanitize_variance(gnss_data.variance, false);

    if (!accept_gnss(gnss_data)) {
      return;
    }

    std::lock_guard<std::mutex> lock(data_mutex);
    utm_queue.push_back(gnss_data);
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

    GNSSData gnss_data;
    gnss_data.stamp = to_sec(fix_msg->header.stamp);
    gnss_data.variance << fix_msg->position_covariance[0], fix_msg->position_covariance[4], fix_msg->position_covariance[8];
    sanitize_variance(gnss_data.variance, fix_msg->position_covariance_type == NavSatFix::COVARIANCE_TYPE_UNKNOWN);

    if (!accept_gnss(gnss_data)) {
      return;
    }

    if (!navsat_origin_initialized) {
      initialize_navsat_origin(fix_msg->latitude, fix_msg->longitude, fix_msg->altitude);
    }

    gnss_data.position = navsat_to_local(fix_msg->latitude, fix_msg->longitude, fix_msg->altitude);
    std::lock_guard<std::mutex> lock(data_mutex);
    utm_queue.push_back(gnss_data);
  }

  void on_insert_submap(const SubMap::ConstPtr& submap) {
    std::lock_guard<std::mutex> lock(data_mutex);
    submap_queue.push_back(submap);
    process_queues_locked();
  }

  void on_smoother_update(gtsam_points::ISAM2Ext& isam2, gtsam::NonlinearFactorGraph& new_factors, gtsam::Values& new_values) {
    const auto priors = output_priors.get_all_and_clear();
    if (!priors.empty()) {
      logger->debug("insert {} GNSS prior factors", priors.size());
      for (const auto& prior : priors) {
        new_factors.add(prior.factor);

        const auto key = X(prior.submap_id);
        if (new_values.exists(key)) {
          const auto pose = new_values.at<gtsam::Pose3>(key);
          new_values.update(key, gtsam::Pose3(pose.rotation(), gtsam::Point3(prior.target)));
        }
      }
    }
  }

  void backend_task() {
    logger->info("starting GNSS global thread");

    while (!kill_switch) {
      {
        std::lock_guard<std::mutex> lock(data_mutex);
        process_queues_locked();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

private:
  void process_queues_locked() {
    // Remove submaps that are created earlier than the oldest GNSS data.
    while (!utm_queue.empty() && !submap_queue.empty() && submap_queue.front()->frames.front()->stamp < utm_queue.front().stamp) {
      submap_queue.pop_front();
    }

    // Interpolate GNSS coords and associate with submaps.
    while (!utm_queue.empty() && !submap_queue.empty() && submap_queue.front()->frames.front()->stamp > utm_queue.front().stamp &&
           submap_queue.front()->frames.back()->stamp < utm_queue.back().stamp) {
      const auto& submap = submap_queue.front();
      const double stamp = submap->frames[submap->frames.size() / 2]->stamp;

      const auto right = std::lower_bound(utm_queue.begin(), utm_queue.end(), stamp, [](const GNSSData& utm, const double t) { return utm.stamp < t; });
      if (right == utm_queue.end() || right == utm_queue.begin()) {
        logger->warn("invalid condition in GNSS global module!!");
        break;
      }
      const auto left = right - 1;
      logger->debug("submap={:.6f} gnss_left={:.6f} gnss_right={:.6f}", stamp, left->stamp, right->stamp);

      const double tl = left->stamp;
      const double tr = right->stamp;
      const double p = (stamp - tl) / (tr - tl);
      GNSSData interpolated;
      interpolated.stamp = stamp;
      interpolated.position = (1.0 - p) * left->position + p * right->position;
      interpolated.variance = (1.0 - p) * left->variance + p * right->variance;

      submaps.push_back(submap);
      submap_coords.push_back(interpolated);
      submap_prior_inserted.push_back(false);

      submap_queue.pop_front();
      utm_queue.erase(utm_queue.begin(), left);
    }

    // Initialize T_world_utm.
    if (!transformation_initialized && !submaps.empty() && (submaps.front()->T_world_origin.inverse() * submaps.back()->T_world_origin).translation().norm() > min_baseline) {
      Eigen::Vector3d mean_est = Eigen::Vector3d::Zero();
      Eigen::Vector3d mean_gnss = Eigen::Vector3d::Zero();
      for (int i = 0; i < submaps.size(); i++) {
        mean_est += submaps[i]->T_world_origin.translation();
        mean_gnss += submap_coords[i].position;
      }
      mean_est /= submaps.size();
      mean_gnss /= submaps.size();

      Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
      for (int i = 0; i < submaps.size(); i++) {
        const Eigen::Vector3d centered_est = submaps[i]->T_world_origin.translation() - mean_est;
        const Eigen::Vector3d centered_gnss = submap_coords[i].position - mean_gnss;
        cov += centered_gnss * centered_est.transpose();
      }
      cov /= submaps.size();

      const Eigen::JacobiSVD<Eigen::Matrix2d> svd(cov.block<2, 2>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      const Eigen::Matrix2d U = svd.matrixU();
      const Eigen::Matrix2d V = svd.matrixV();
      Eigen::Matrix2d S = Eigen::Matrix2d::Identity();

      const double det = U.determinant() * V.determinant();
      if (det < 0.0) {
        S(1, 1) = -1;
      }

      Eigen::Isometry3d T_utm_world = Eigen::Isometry3d::Identity();
      T_utm_world.linear().block<2, 2>(0, 0) = U * S * V.transpose();
      T_utm_world.translation() = mean_gnss - T_utm_world.linear() * mean_est;

      T_world_utm = T_utm_world.inverse();

      for (int i = 0; i < submaps.size(); i++) {
        const Eigen::Vector3d gnss = T_world_utm * submap_coords[i].position;
        logger->debug("submap={} gnss={}", convert_to_string(submaps[i]->T_world_origin.translation().eval()), convert_to_string(gnss));
      }

      logger->info("T_world_utm={}", convert_to_string(T_world_utm));
      transformation_initialized = true;
    }

    insert_pending_gnss_factors();
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

  bool accept_gnss(const GNSSData& data) {
    if (!std::isfinite(data.stamp) || !data.position.allFinite() || !data.variance.allFinite()) {
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

  void insert_pending_gnss_factors() {
    if (!transformation_initialized) {
      return;
    }

    for (int i = 0; i < submaps.size(); i++) {
      if (submap_prior_inserted[i]) {
        continue;
      }

      const auto& submap = submaps[i];
      Eigen::Vector3d xyz = T_world_utm * submap_coords[i].position;
      if (use_gnss_position_in_base) {
        xyz -= submap->T_world_origin.linear() * gnss_position_in_base;
      }

      logger->debug("submap={} gnss={}", convert_to_string(submap->T_world_origin.translation().eval()), convert_to_string(xyz));

      const auto model = create_noise_model(submap_coords[i].variance);
      GNSSPriorData prior;
      prior.submap_id = submap->id;
      prior.target = xyz;
      prior.factor.reset(new gtsam::PoseTranslationPrior<gtsam::Pose3>(X(submap->id), xyz, model));
      output_priors.push_back(prior);
      submap_prior_inserted[i] = true;
      num_inserted_priors++;
    }
  }

  std::mutex data_mutex;

  std::atomic_bool kill_switch;
  std::thread thread;

  ConcurrentVector<GNSSPriorData, Eigen::aligned_allocator<GNSSPriorData>> output_priors;

  std::deque<GNSSData, Eigen::aligned_allocator<GNSSData>> utm_queue;
  std::deque<SubMap::ConstPtr> submap_queue;

  std::vector<SubMap::ConstPtr> submaps;
  std::vector<GNSSData, Eigen::aligned_allocator<GNSSData>> submap_coords;
  std::vector<bool> submap_prior_inserted;

  std::string gnss_topic;
  std::string gnss_message_type;
  Eigen::Vector3d prior_inf_scale;
  Eigen::Vector3d min_stddev;
  Eigen::Vector3d unknown_stddev;
  Eigen::Vector3d gnss_position_in_base;
  double min_baseline;
  double max_std_xy;
  double max_std_z;
  double robust_kernel_width;
  int min_navsat_status;

  bool use_z;
  bool use_gnss_position_in_base;

  bool transformation_initialized;
  Eigen::Isometry3d T_world_utm;

  bool navsat_origin_initialized;
  double navsat_origin_lat;
  double navsat_origin_lon;
  double navsat_origin_alt;
  Eigen::Vector3d navsat_origin_ecef;
  Eigen::Vector3d navsat_east;
  Eigen::Vector3d navsat_north;
  Eigen::Vector3d navsat_up;

  size_t num_rejected_gnss = 0;
  size_t num_inserted_priors = 0;

  // Logging
  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::GNSSGlobal();
}
