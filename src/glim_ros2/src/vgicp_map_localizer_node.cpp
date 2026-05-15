#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

using PointT = pcl::PointXYZI;
using Cloud = pcl::PointCloud<PointT>;

namespace {

struct FieldInfo {
  int offset = -1;
  int datatype = 0;
};

double stamp_to_sec(const builtin_interfaces::msg::Time& stamp) {
  return stamp.sec + stamp.nanosec * 1e-9;
}

double read_numeric_field(const uint8_t* data, const int datatype) {
  switch (datatype) {
    case sensor_msgs::msg::PointField::FLOAT32:
      return *reinterpret_cast<const float*>(data);
    case sensor_msgs::msg::PointField::FLOAT64:
      return *reinterpret_cast<const double*>(data);
    case sensor_msgs::msg::PointField::UINT16:
      return *reinterpret_cast<const uint16_t*>(data);
    case sensor_msgs::msg::PointField::INT16:
      return *reinterpret_cast<const int16_t*>(data);
    case sensor_msgs::msg::PointField::UINT32:
      return *reinterpret_cast<const uint32_t*>(data);
    case sensor_msgs::msg::PointField::INT32:
      return *reinterpret_cast<const int32_t*>(data);
    case sensor_msgs::msg::PointField::UINT8:
      return *reinterpret_cast<const uint8_t*>(data);
    case sensor_msgs::msg::PointField::INT8:
      return *reinterpret_cast<const int8_t*>(data);
    default:
      return std::numeric_limits<double>::quiet_NaN();
  }
}

std::unordered_map<std::string, FieldInfo> build_field_map(const sensor_msgs::msg::PointCloud2& msg) {
  std::unordered_map<std::string, FieldInfo> fields;
  for (const auto& field : msg.fields) {
    fields[field.name] = FieldInfo{static_cast<int>(field.offset), field.datatype};
  }
  return fields;
}

Cloud::Ptr extract_scan(const sensor_msgs::msg::PointCloud2& msg, const double min_range, const double max_range, int* raw_points_out) {
  const auto fields = build_field_map(msg);
  const auto fx = fields.find("x");
  const auto fy = fields.find("y");
  const auto fz = fields.find("z");
  if (fx == fields.end() || fy == fields.end() || fz == fields.end()) {
    return nullptr;
  }

  const auto fintensity = fields.find("intensity");
  const int raw_points = static_cast<int>(msg.width * msg.height);
  if (raw_points_out) {
    *raw_points_out = raw_points;
  }

  Cloud::Ptr scan(new Cloud);
  scan->reserve(raw_points);
  const double min_sq = min_range * min_range;
  const double max_sq = max_range * max_range;

  for (int i = 0; i < raw_points; i++) {
    const uint8_t* base = msg.data.data() + static_cast<size_t>(i) * msg.point_step;
    const double x = read_numeric_field(base + fx->second.offset, fx->second.datatype);
    const double y = read_numeric_field(base + fy->second.offset, fy->second.datatype);
    const double z = read_numeric_field(base + fz->second.offset, fz->second.datatype);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }

    const double r2 = x * x + y * y + z * z;
    if (r2 < min_sq || r2 > max_sq) {
      continue;
    }

    PointT pt;
    pt.x = static_cast<float>(x);
    pt.y = static_cast<float>(y);
    pt.z = static_cast<float>(z);
    pt.intensity = 0.0f;
    if (fintensity != fields.end()) {
      const double intensity = read_numeric_field(base + fintensity->second.offset, fintensity->second.datatype);
      if (std::isfinite(intensity)) {
        pt.intensity = static_cast<float>(intensity);
      }
    }
    scan->push_back(pt);
  }

  scan->width = scan->size();
  scan->height = 1;
  scan->is_dense = false;
  return scan;
}

Cloud::Ptr voxel_downsample(const Cloud::ConstPtr& cloud, const double leaf_size) {
  if (leaf_size <= 0.0 || cloud->empty()) {
    return Cloud::Ptr(new Cloud(*cloud));
  }

  pcl::VoxelGrid<PointT> voxel;
  voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel.setInputCloud(cloud);

  Cloud::Ptr filtered(new Cloud);
  voxel.filter(*filtered);
  return filtered;
}

double wrap_pi(double x) {
  while (x > M_PI) {
    x -= 2.0 * M_PI;
  }
  while (x < -M_PI) {
    x += 2.0 * M_PI;
  }
  return x;
}

double yaw_from_rotation(const Eigen::Matrix3d& R) {
  return std::atan2(R(1, 0), R(0, 0));
}

Eigen::Isometry3d project_to_rigid(const Eigen::Matrix4d& matrix) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = matrix.block<3, 1>(0, 3);
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(matrix.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
  if (R.determinant() < 0.0) {
    Eigen::Matrix3d U = svd.matrixU();
    U.col(2) *= -1.0;
    R = U * svd.matrixV().transpose();
  }
  pose.linear() = R;
  return pose;
}

Eigen::Isometry3d pose_xyz_yaw(const double x, const double y, const double z, const double yaw_deg) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(x, y, z);
  pose.linear() = Eigen::AngleAxisd(yaw_deg * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return pose;
}

Eigen::Isometry3d pose_from_msg(const geometry_msgs::msg::Pose& msg) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  Eigen::Quaterniond q(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z);
  q.normalize();
  pose.linear() = q.toRotationMatrix();
  pose.translation() = Eigen::Vector3d(msg.position.x, msg.position.y, msg.position.z);
  return pose;
}

}  // namespace

class VgicpMapLocalizerNode : public rclcpp::Node {
public:
  VgicpMapLocalizerNode() : Node("vgicp_map_localizer") {
    const auto map_path = declare_parameter<std::string>("map_path", "");
    points_topic_ = declare_parameter<std::string>("points_topic", "/front/lidar_point");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/vgicp/odom_enu");
    initialpose_topic_ = declare_parameter<std::string>("initialpose_topic", "/initialpose");
    output_frame_id_ = declare_parameter<std::string>("output_frame_id", "utm_enu");
    child_frame_id_ = declare_parameter<std::string>("child_frame_id", "base_link");
    require_initial_pose_ = declare_parameter<bool>("require_initial_pose", false);
    publish_tf_ = declare_parameter<bool>("publish_tf", false);

    source_leaf_ = declare_parameter<double>("source_leaf", 0.8);
    target_leaf_ = declare_parameter<double>("target_leaf", 0.8);
    vgicp_resolution_ = declare_parameter<double>("vgicp_resolution", 1.0);
    crop_radius_ = declare_parameter<double>("crop_radius", 90.0);
    max_corr_dist_ = declare_parameter<double>("max_corr_dist", 5.0);
    max_fitness_ = declare_parameter<double>("max_fitness", 2.0);
    min_range_ = declare_parameter<double>("min_range", 3.0);
    max_range_ = declare_parameter<double>("max_range", 100.0);
    max_step_ = declare_parameter<double>("max_step", 4.0);
    max_yaw_step_deg_ = declare_parameter<double>("max_yaw_step_deg", 20.0);
    pose_std_xy_ = declare_parameter<double>("pose_std_xy", 0.30);
    pose_std_z_ = declare_parameter<double>("pose_std_z", 2.0);
    pose_std_yaw_deg_ = declare_parameter<double>("pose_std_yaw_deg", 3.0);
    max_iterations_ = declare_parameter<int>("max_iterations", 40);
    min_source_points_ = declare_parameter<int>("min_source_points", 200);
    min_target_points_ = declare_parameter<int>("min_target_points", 2000);

    const double init_x = declare_parameter<double>("initial_x", 0.0);
    const double init_y = declare_parameter<double>("initial_y", 0.0);
    const double init_z = declare_parameter<double>("initial_z", 0.0);
    const double init_yaw_deg = declare_parameter<double>("initial_yaw_deg", 0.0);
    current_pose_map_ = pose_xyz_yaw(init_x, init_y, init_z, init_yaw_deg);
    have_pose_ = !require_initial_pose_;

    const double map_to_output_x = declare_parameter<double>("map_to_output_x", 0.0);
    const double map_to_output_y = declare_parameter<double>("map_to_output_y", 0.0);
    const double map_to_output_z = declare_parameter<double>("map_to_output_z", 0.0);
    const double map_to_output_yaw_deg = declare_parameter<double>("map_to_output_yaw_deg", 0.0);
    T_output_map_ = pose_xyz_yaw(map_to_output_x, map_to_output_y, map_to_output_z, map_to_output_yaw_deg);

    if (map_path.empty()) {
      throw std::runtime_error("map_path parameter is required");
    }
    if (pcl::io::loadPCDFile<PointT>(map_path, *map_) != 0 || map_->empty()) {
      throw std::runtime_error("failed to load PCD map: " + map_path);
    }
    kdtree_.setInputCloud(map_);

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 20);
    points_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      points_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&VgicpMapLocalizerNode::points_callback, this, std::placeholders::_1));
    initialpose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      initialpose_topic_,
      10,
      std::bind(&VgicpMapLocalizerNode::initialpose_callback, this, std::placeholders::_1));

    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    RCLCPP_INFO(
      get_logger(),
      "loaded VGICP map path=%s points=%zu points_topic=%s odom_topic=%s require_initial_pose=%d",
      map_path.c_str(),
      map_->size(),
      points_topic_.c_str(),
      odom_topic_.c_str(),
      static_cast<int>(require_initial_pose_));
  }

private:
  Cloud::Ptr crop_target(const Eigen::Vector3d& center) {
    PointT query;
    query.x = static_cast<float>(center.x());
    query.y = static_cast<float>(center.y());
    query.z = static_cast<float>(center.z());
    query.intensity = 0.0f;

    std::vector<int> indices;
    std::vector<float> distances;
    kdtree_.radiusSearch(query, crop_radius_, indices, distances);

    Cloud::Ptr target(new Cloud);
    target->reserve(indices.size());
    for (const int index : indices) {
      target->push_back((*map_)[index]);
    }
    target->width = target->size();
    target->height = 1;
    target->is_dense = false;
    return voxel_downsample(target, target_leaf_);
  }

  void initialpose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg) {
    current_pose_map_ = pose_from_msg(msg->pose.pose);
    have_pose_ = true;
    have_last_pose_ = false;
    RCLCPP_INFO(
      get_logger(),
      "initial pose set in map frame x=%.3f y=%.3f z=%.3f yaw=%.3fdeg",
      current_pose_map_.translation().x(),
      current_pose_map_.translation().y(),
      current_pose_map_.translation().z(),
      yaw_from_rotation(current_pose_map_.linear()) * 180.0 / M_PI);
  }

  void points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    if (!have_pose_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "waiting for initial pose on %s", initialpose_topic_.c_str());
      return;
    }

    int raw_points = 0;
    Cloud::Ptr scan = extract_scan(*msg, min_range_, max_range_, &raw_points);
    if (!scan) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "point cloud has no xyz fields");
      return;
    }

    Cloud::Ptr source = voxel_downsample(scan, source_leaf_);
    if (static_cast<int>(source->size()) < min_source_points_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "too few source points after filtering: %zu", source->size());
      return;
    }

    Cloud::Ptr target = crop_target(current_pose_map_.translation());
    if (static_cast<int>(target->size()) < min_target_points_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "too few target map points near current pose: %zu center=(%.2f %.2f %.2f)",
        target->size(),
        current_pose_map_.translation().x(),
        current_pose_map_.translation().y(),
        current_pose_map_.translation().z());
      return;
    }

    fast_gicp::FastVGICP<PointT, PointT> vgicp;
    vgicp.setResolution(vgicp_resolution_);
    vgicp.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
    vgicp.setNumThreads(std::max(1u, std::thread::hardware_concurrency()));
    vgicp.setMaxCorrespondenceDistance(max_corr_dist_);
    vgicp.setMaximumIterations(max_iterations_);
    vgicp.setTransformationEpsilon(1e-4);
    vgicp.setRotationEpsilon(1e-4);
    vgicp.setInputTarget(target);
    vgicp.setInputSource(source);

    Cloud aligned;
    const auto t0 = std::chrono::high_resolution_clock::now();
    vgicp.align(aligned, current_pose_map_.matrix().cast<float>());
    const auto t1 = std::chrono::high_resolution_clock::now();

    const Eigen::Isometry3d result = project_to_rigid(vgicp.getFinalTransformation().cast<double>());
    const double fitness = vgicp.getFitnessScore(max_corr_dist_);
    const double elapsed_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

    if (!vgicp.hasConverged() || !std::isfinite(fitness) || fitness > max_fitness_) {
      RCLCPP_WARN(
        get_logger(),
        "reject VGICP result converged=%d fitness=%.4f src=%zu target=%zu elapsed_ms=%.2f",
        static_cast<int>(vgicp.hasConverged()),
        fitness,
        source->size(),
        target->size(),
        elapsed_ms);
      return;
    }

    if (have_last_pose_) {
      const Eigen::Isometry3d delta = last_pose_map_.inverse() * result;
      const double step = delta.translation().norm();
      const double yaw_step_deg = std::abs(wrap_pi(yaw_from_rotation(delta.linear()))) * 180.0 / M_PI;
      if (step > max_step_ || yaw_step_deg > max_yaw_step_deg_) {
        RCLCPP_WARN(get_logger(), "reject VGICP jump step=%.3fm yaw=%.3fdeg fitness=%.4f", step, yaw_step_deg, fitness);
        return;
      }
    }

    last_pose_map_ = current_pose_map_;
    current_pose_map_ = result;
    have_last_pose_ = true;
    publish_odom(*msg, fitness, source->size(), target->size(), elapsed_ms);
  }

  void publish_odom(const sensor_msgs::msg::PointCloud2& cloud_msg, const double fitness, const size_t source_points, const size_t target_points, const double elapsed_ms) {
    const Eigen::Isometry3d pose_out = T_output_map_ * current_pose_map_;
    const Eigen::Quaterniond q(pose_out.linear());

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = cloud_msg.header.stamp;
    odom.header.frame_id = output_frame_id_;
    odom.child_frame_id = child_frame_id_;
    odom.pose.pose.position.x = pose_out.translation().x();
    odom.pose.pose.position.y = pose_out.translation().y();
    odom.pose.pose.position.z = pose_out.translation().z();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    std::fill(odom.pose.covariance.begin(), odom.pose.covariance.end(), 0.0);
    odom.pose.covariance[0] = pose_std_xy_ * pose_std_xy_;
    odom.pose.covariance[7] = pose_std_xy_ * pose_std_xy_;
    odom.pose.covariance[14] = pose_std_z_ * pose_std_z_;
    odom.pose.covariance[21] = 1e3;
    odom.pose.covariance[28] = 1e3;
    const double yaw_std = pose_std_yaw_deg_ * M_PI / 180.0;
    odom.pose.covariance[35] = yaw_std * yaw_std;

    odom_pub_->publish(odom);
    if (tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = odom.header;
      tf.child_frame_id = child_frame_id_;
      tf.transform.translation.x = odom.pose.pose.position.x;
      tf.transform.translation.y = odom.pose.pose.position.y;
      tf.transform.translation.z = odom.pose.pose.position.z;
      tf.transform.rotation = odom.pose.pose.orientation;
      tf_broadcaster_->sendTransform(tf);
    }

    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      1000,
      "VGICP stamp=%.3f src=%zu target=%zu fitness=%.4f elapsed_ms=%.2f pose=(%.2f %.2f %.2f)",
      stamp_to_sec(cloud_msg.header.stamp),
      source_points,
      target_points,
      fitness,
      elapsed_ms,
      pose_out.translation().x(),
      pose_out.translation().y(),
      pose_out.translation().z());
  }

  std::string points_topic_;
  std::string odom_topic_;
  std::string initialpose_topic_;
  std::string output_frame_id_;
  std::string child_frame_id_;
  bool require_initial_pose_ = false;
  bool publish_tf_ = false;

  double source_leaf_ = 0.8;
  double target_leaf_ = 0.8;
  double vgicp_resolution_ = 1.0;
  double crop_radius_ = 90.0;
  double max_corr_dist_ = 5.0;
  double max_fitness_ = 2.0;
  double min_range_ = 3.0;
  double max_range_ = 100.0;
  double max_step_ = 4.0;
  double max_yaw_step_deg_ = 20.0;
  double pose_std_xy_ = 0.30;
  double pose_std_z_ = 2.0;
  double pose_std_yaw_deg_ = 3.0;
  int max_iterations_ = 40;
  int min_source_points_ = 200;
  int min_target_points_ = 2000;

  Cloud::Ptr map_{new Cloud};
  pcl::KdTreeFLANN<PointT> kdtree_;
  Eigen::Isometry3d current_pose_map_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d last_pose_map_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_output_map_ = Eigen::Isometry3d::Identity();
  bool have_pose_ = false;
  bool have_last_pose_ = false;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VgicpMapLocalizerNode>());
  rclcpp::shutdown();
  return 0;
}
