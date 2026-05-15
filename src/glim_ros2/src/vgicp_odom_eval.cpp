#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <boost/program_options.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace po = boost::program_options;
using PointT = pcl::PointXYZI;
using Cloud = pcl::PointCloud<PointT>;

struct FieldInfo {
  int offset = -1;
  int datatype = 0;
};

struct OdomResult {
  double stamp = 0.0;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  int raw_points = 0;
  int source_points = 0;
  int target_points = 0;
  bool converged = true;
  double fitness = 0.0;
  double elapsed_ms = 0.0;
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

Cloud::Ptr extract_scan(
  const sensor_msgs::msg::PointCloud2& msg,
  const double min_range,
  const double max_range,
  double* stamp_out,
  int* raw_points_out) {
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

  if (stamp_out) {
    *stamp_out = stamp_to_sec(msg.header.stamp);
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

bool is_finite_pose(const Eigen::Isometry3d& pose) {
  return pose.matrix().allFinite() && pose.translation().norm() < 1e6;
}

bool is_rigid_pose(const Eigen::Isometry3d& pose) {
  if (!is_finite_pose(pose)) {
    return false;
  }
  const Eigen::Matrix3d R = pose.linear();
  const double det = R.determinant();
  const double orth_error = (R.transpose() * R - Eigen::Matrix3d::Identity()).norm();
  return std::abs(det - 1.0) < 1e-2 && orth_error < 1e-2;
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

Cloud::Ptr build_target(const std::deque<Cloud::Ptr>& local_scans, const double target_leaf) {
  Cloud::Ptr target(new Cloud);
  size_t total_points = 0;
  for (const auto& scan : local_scans) {
    total_points += scan->size();
  }
  target->reserve(total_points);
  for (const auto& scan : local_scans) {
    *target += *scan;
  }
  target->width = target->size();
  target->height = 1;
  target->is_dense = false;
  return voxel_downsample(target, target_leaf);
}

void write_outputs(const std::string& output_prefix, const std::vector<OdomResult>& results) {
  {
    std::ofstream ofs(output_prefix + ".tum.txt");
    for (const auto& result : results) {
      const Eigen::Quaterniond q(result.pose.linear());
      ofs << std::fixed << result.stamp << ' ' << result.pose.translation().x() << ' ' << result.pose.translation().y() << ' '
          << result.pose.translation().z() << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
    }
  }

  {
    std::ofstream ofs(output_prefix + ".csv");
    ofs << "stamp,x,y,z,qx,qy,qz,qw,raw_points,source_points,target_points,converged,fitness,elapsed_ms\n";
    for (const auto& result : results) {
      const Eigen::Quaterniond q(result.pose.linear());
      ofs << std::fixed << result.stamp << ',' << result.pose.translation().x() << ',' << result.pose.translation().y() << ','
          << result.pose.translation().z() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << ',' << result.raw_points
          << ',' << result.source_points << ',' << result.target_points << ',' << (result.converged ? 1 : 0) << ',' << result.fitness
          << ',' << result.elapsed_ms << '\n';
    }
  }
}

int main(int argc, char** argv) {
  std::string bag_path;
  std::string points_topic = "/lidar_points";
  std::string output_prefix = "/tmp/vgicp_odom";
  double start = 0.0;
  double duration = 0.0;
  double source_leaf = 0.8;
  double target_leaf = 0.8;
  double local_map_leaf = 0.4;
  double vgicp_resolution = 1.0;
  double max_corr_dist = 5.0;
  double max_step = 2.0;
  double max_yaw_step_deg = 15.0;
  double max_fitness = 2.0;
  double min_range = 3.0;
  double max_range = 100.0;
  int local_frames = 20;
  int max_iterations = 40;
  int min_source_points = 200;
  int min_target_points = 1000;

  po::options_description desc("offline scan-to-local-map VGICP odometry");
  desc.add_options()("help", "show help")("bag", po::value<std::string>(&bag_path)->required(), "input rosbag2 path")(
    "points_topic", po::value<std::string>(&points_topic)->default_value(points_topic), "PointCloud2 topic name")(
    "output_prefix", po::value<std::string>(&output_prefix)->default_value(output_prefix), "output prefix for .tum.txt and .csv")(
    "start", po::value<double>(&start)->default_value(start), "bag-relative start time [sec]")(
    "duration", po::value<double>(&duration)->default_value(duration), "duration [sec], 0 means full bag")(
    "source_leaf", po::value<double>(&source_leaf)->default_value(source_leaf), "source scan voxel leaf [m]")(
    "target_leaf", po::value<double>(&target_leaf)->default_value(target_leaf), "target local map voxel leaf [m]")(
    "local_map_leaf", po::value<double>(&local_map_leaf)->default_value(local_map_leaf), "stored local scan voxel leaf [m]")(
    "vgicp_resolution", po::value<double>(&vgicp_resolution)->default_value(vgicp_resolution), "VGICP voxel resolution [m]")(
    "max_corr_dist", po::value<double>(&max_corr_dist)->default_value(max_corr_dist), "max correspondence distance [m]")(
    "max_step", po::value<double>(&max_step)->default_value(max_step), "reject relative translation steps larger than this [m]")(
    "max_yaw_step_deg", po::value<double>(&max_yaw_step_deg)->default_value(max_yaw_step_deg), "reject relative yaw steps larger than this [deg]")(
    "max_fitness", po::value<double>(&max_fitness)->default_value(max_fitness), "reject VGICP fitness larger than this")(
    "min_range", po::value<double>(&min_range)->default_value(min_range), "minimum LiDAR range [m]")(
    "max_range", po::value<double>(&max_range)->default_value(max_range), "maximum LiDAR range [m]")(
    "local_frames", po::value<int>(&local_frames)->default_value(local_frames), "number of recent scans in local map")(
    "max_iterations", po::value<int>(&max_iterations)->default_value(max_iterations), "VGICP max iterations")(
    "min_source_points", po::value<int>(&min_source_points)->default_value(min_source_points), "minimum downsampled source points")(
    "min_target_points", po::value<int>(&min_target_points)->default_value(min_target_points), "minimum local target points");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n" << desc << std::endl;
    return 1;
  }

  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = bag_path;
  storage_options.storage_id = "sqlite3";
  rosbag2_cpp::ConverterOptions converter_options;

  rosbag2_cpp::readers::SequentialReader reader;
  reader.open(storage_options, converter_options);
  rosbag2_storage::StorageFilter filter;
  filter.topics = {points_topic};
  reader.set_filter(filter);

  bool have_start_stamp = false;
  double bag_start = 0.0;
  double start_stamp = 0.0;
  double end_stamp = std::numeric_limits<double>::infinity();

  rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serialization;
  std::vector<OdomResult> results;
  std::deque<Cloud::Ptr> local_scans_world;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d last_delta = Eigen::Isometry3d::Identity();
  int rejected = 0;

  int skipped = 0;
  while (reader.has_next()) {
    auto bag_msg = reader.read_next();
    if (bag_msg->topic_name != points_topic) {
      continue;
    }

    const rclcpp::SerializedMessage serialized_msg(*bag_msg->serialized_data);
    auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    serialization.deserialize_message(&serialized_msg, msg.get());

    double stamp = 0.0;
    int raw_points = 0;
    Cloud::Ptr scan = extract_scan(*msg, min_range, max_range, &stamp, &raw_points);
    if (!scan) {
      skipped++;
      continue;
    }

    if (!have_start_stamp) {
      bag_start = stamp;
      start_stamp = bag_start + start;
      end_stamp = duration > 0.0 ? start_stamp + duration : std::numeric_limits<double>::infinity();
      have_start_stamp = true;
    }
    if (stamp < start_stamp) {
      continue;
    }
    if (stamp > end_stamp) {
      break;
    }

    Cloud::Ptr source = voxel_downsample(scan, source_leaf);
    if (static_cast<int>(source->size()) < min_source_points) {
      skipped++;
      continue;
    }

    OdomResult result;
    result.stamp = stamp;
    result.raw_points = raw_points;
    result.source_points = source->size();

    if (results.empty()) {
      pose = Eigen::Isometry3d::Identity();
      result.pose = pose;
      result.target_points = 0;
      result.converged = true;
      result.fitness = 0.0;
    } else {
      Cloud::Ptr target = build_target(local_scans_world, target_leaf);
      result.target_points = target->size();
      if (static_cast<int>(target->size()) < min_target_points) {
        skipped++;
        continue;
      }

      const Eigen::Isometry3d prediction = pose * last_delta;
      fast_gicp::FastVGICP<PointT, PointT> vgicp;
      vgicp.setResolution(vgicp_resolution);
      vgicp.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      vgicp.setNumThreads(std::max(1u, std::thread::hardware_concurrency()));
      vgicp.setMaxCorrespondenceDistance(max_corr_dist);
      vgicp.setMaximumIterations(max_iterations);
      vgicp.setTransformationEpsilon(1e-4);
      vgicp.setRotationEpsilon(1e-4);
      vgicp.setInputTarget(target);
      vgicp.setInputSource(source);

      Cloud aligned;
      const auto t0 = std::chrono::high_resolution_clock::now();
      vgicp.align(aligned, prediction.matrix().cast<float>());
      const auto t1 = std::chrono::high_resolution_clock::now();
      result.elapsed_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;
      (void)vgicp.hasConverged();
      result.fitness = vgicp.getFitnessScore();

      const Eigen::Matrix4d raw_matrix = vgicp.getFinalTransformation().cast<double>();
      Eigen::Isometry3d new_pose = project_to_rigid(raw_matrix);
      const Eigen::Isometry3d new_delta = pose.inverse() * new_pose;
      const double step_norm = new_delta.translation().norm();
      const double yaw_step = std::abs(wrap_pi(yaw_from_rotation(new_delta.linear())));
      const bool accepted =
        raw_matrix.allFinite() && is_rigid_pose(new_pose) && is_rigid_pose(new_delta) && std::isfinite(result.fitness) && result.fitness <= max_fitness && step_norm <= max_step &&
        yaw_step <= max_yaw_step_deg * M_PI / 180.0;

      if (accepted) {
        last_delta = new_delta;
        pose = new_pose;
        result.converged = true;
      } else {
        rejected++;
        result.converged = false;
      }
      result.pose = pose;
    }

    if (results.empty() || result.converged) {
      Cloud::Ptr stored(new Cloud);
      pcl::transformPointCloud(*voxel_downsample(scan, local_map_leaf), *stored, pose.matrix().cast<float>());
      local_scans_world.push_back(stored);
      while (static_cast<int>(local_scans_world.size()) > local_frames) {
        local_scans_world.pop_front();
      }
    }

    results.push_back(result);
    if (results.size() % 100 == 0) {
      std::cout << "frames=" << results.size() << " stamp=" << std::fixed << stamp << " fitness=" << result.fitness << " target="
                << result.target_points << " rejected=" << rejected << std::endl;
    }
  }

  if (results.empty()) {
    std::cerr << "no odometry results (skipped=" << skipped << ")" << std::endl;
    return 2;
  }

  write_outputs(output_prefix, results);
  const Eigen::Vector3d delta = results.back().pose.translation() - results.front().pose.translation();
  double path = 0.0;
  for (size_t i = 1; i < results.size(); i++) {
    path += (results[i].pose.translation() - results[i - 1].pose.translation()).norm();
  }
  std::cout << "SUMMARY frames=" << results.size() << " skipped=" << skipped << " rejected=" << rejected << " path=" << path
            << " displacement=" << delta.norm() << " output_prefix=" << output_prefix << std::endl;
  return 0;
}
