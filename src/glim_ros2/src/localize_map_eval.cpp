#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <boost/program_options.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
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

struct PoseStamped {
  double stamp = 0.0;
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
};

struct EvalResult {
  double stamp = 0.0;
  int raw_points = 0;
  int source_points = 0;
  int target_points = 0;
  double elapsed_ms = 0.0;
  bool converged = false;
  double fitness = std::numeric_limits<double>::quiet_NaN();
  double init_xy_error = 0.0;
  double final_xy_error = 0.0;
  double final_z_error = 0.0;
  double final_yaw_error_deg = 0.0;
};

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
  auto ftime = fields.find("timestamp");
  if (ftime == fields.end()) {
    ftime = fields.find("time");
  }
  if (ftime == fields.end()) {
    ftime = fields.find("t");
  }

  const int raw_points = static_cast<int>(msg.width * msg.height);
  if (raw_points_out) {
    *raw_points_out = raw_points;
  }

  Cloud::Ptr scan(new Cloud);
  scan->reserve(raw_points);

  double min_point_time = std::numeric_limits<double>::infinity();
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

    if (ftime != fields.end()) {
      const double t = read_numeric_field(base + ftime->second.offset, ftime->second.datatype);
      if (std::isfinite(t)) {
        min_point_time = std::min(min_point_time, t);
      }
    }
  }

  if (stamp_out) {
    *stamp_out = std::isfinite(min_point_time) && min_point_time > 1.0 ? min_point_time : stamp_to_sec(msg.header.stamp);
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

std::vector<PoseStamped> load_trajectory(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error("failed to open trajectory: " + path);
  }

  std::vector<PoseStamped> poses;
  while (ifs) {
    double t, x, y, z, qx, qy, qz, qw;
    ifs >> t >> x >> y >> z >> qx >> qy >> qz >> qw;
    if (!ifs) {
      break;
    }

    PoseStamped pose;
    pose.stamp = t;
    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();
    pose.T.linear() = q.toRotationMatrix();
    pose.T.translation() = Eigen::Vector3d(x, y, z);
    poses.push_back(pose);
  }

  if (poses.empty()) {
    throw std::runtime_error("empty trajectory: " + path);
  }

  return poses;
}

const PoseStamped* nearest_pose(const std::vector<PoseStamped>& poses, const double stamp, const double max_dt) {
  auto it = std::lower_bound(poses.begin(), poses.end(), stamp, [](const PoseStamped& pose, const double t) { return pose.stamp < t; });
  const PoseStamped* best = nullptr;
  double best_dt = std::numeric_limits<double>::infinity();
  if (it != poses.end()) {
    best = &(*it);
    best_dt = std::abs(it->stamp - stamp);
  }
  if (it != poses.begin()) {
    const auto prev = it - 1;
    const double dt = std::abs(prev->stamp - stamp);
    if (dt < best_dt) {
      best = &(*prev);
      best_dt = dt;
    }
  }

  return best_dt <= max_dt ? best : nullptr;
}

double yaw_from_rotation(const Eigen::Matrix3d& R) {
  return std::atan2(R(1, 0), R(0, 0));
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

Eigen::Isometry3d perturb_pose(const Eigen::Isometry3d& ref, const int index, const double xy, const double z, const double yaw_deg) {
  Eigen::Isometry3d perturbed = ref;
  const double a = 0.73 * index;
  const double b = 1.17 * index + 0.4;
  const double yaw = yaw_deg * M_PI / 180.0 * std::sin(0.37 * index + 0.2);

  perturbed.translation().x() += xy * std::sin(a);
  perturbed.translation().y() += xy * std::cos(b);
  perturbed.translation().z() += z * std::sin(0.53 * index);
  perturbed.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * perturbed.linear();
  return perturbed;
}

Cloud::Ptr crop_target(
  const Cloud::ConstPtr& map,
  pcl::KdTreeFLANN<PointT>& kdtree,
  const Eigen::Vector3d& center,
  const double radius,
  const double leaf_size) {
  PointT query;
  query.x = static_cast<float>(center.x());
  query.y = static_cast<float>(center.y());
  query.z = static_cast<float>(center.z());
  query.intensity = 0.0f;

  std::vector<int> indices;
  std::vector<float> distances;
  kdtree.radiusSearch(query, radius, indices, distances);

  Cloud::Ptr target(new Cloud);
  target->reserve(indices.size());
  for (const int index : indices) {
    target->push_back((*map)[index]);
  }
  target->width = target->size();
  target->height = 1;
  target->is_dense = false;

  return voxel_downsample(target, leaf_size);
}

void write_csv(const std::string& path, const std::vector<EvalResult>& results) {
  std::ofstream ofs(path);
  ofs << "stamp,raw_points,source_points,target_points,elapsed_ms,converged,fitness,init_xy_error,final_xy_error,final_z_error,final_yaw_error_deg\n";
  for (const auto& r : results) {
    ofs << std::fixed << r.stamp << ',' << r.raw_points << ',' << r.source_points << ',' << r.target_points << ',' << r.elapsed_ms << ','
        << (r.converged ? 1 : 0) << ',' << r.fitness << ',' << r.init_xy_error << ',' << r.final_xy_error << ',' << r.final_z_error << ','
        << r.final_yaw_error_deg << '\n';
  }
}

int main(int argc, char** argv) {
  std::string bag_path;
  std::string map_path;
  std::string trajectory_path;
  std::string output_csv;
  std::string points_topic = "/lidar_points";
  double start = 760.0;
  double duration = 540.0;
  double sample_interval = 10.0;
  double source_leaf = 0.8;
  double target_leaf = 0.8;
  double vgicp_resolution = 1.0;
  double crop_radius = 90.0;
  double min_range = 3.0;
  double max_range = 100.0;
  double pose_max_dt = 0.08;
  double perturb_xy = 2.0;
  double perturb_z = 0.0;
  double perturb_yaw_deg = 2.0;
  int max_iterations = 40;
  int min_source_points = 200;
  int min_target_points = 2000;

  po::options_description desc("offline fast_gicp localization evaluator");
  desc.add_options()("help", "show help")("bag", po::value<std::string>(&bag_path)->required(), "input rosbag2 path")(
    "map", po::value<std::string>(&map_path)->required(), "PCD map path")(
    "trajectory", po::value<std::string>(&trajectory_path)->required(), "reference traj_lidar.txt path")(
    "points_topic", po::value<std::string>(&points_topic)->default_value(points_topic), "PointCloud2 topic name")(
    "output_csv", po::value<std::string>(&output_csv)->default_value("/tmp/localize_map_eval.csv"), "output CSV path")(
    "start", po::value<double>(&start)->default_value(start), "bag-relative start time [sec]")(
    "duration", po::value<double>(&duration)->default_value(duration), "duration [sec]")(
    "sample_interval", po::value<double>(&sample_interval)->default_value(sample_interval), "sample interval [sec]")(
    "source_leaf", po::value<double>(&source_leaf)->default_value(source_leaf), "source scan voxel leaf [m]")(
    "target_leaf", po::value<double>(&target_leaf)->default_value(target_leaf), "local target voxel leaf [m]")(
    "vgicp_resolution", po::value<double>(&vgicp_resolution)->default_value(vgicp_resolution), "VGICP voxel resolution [m]")(
    "crop_radius", po::value<double>(&crop_radius)->default_value(crop_radius), "local map crop radius [m]")(
    "min_range", po::value<double>(&min_range)->default_value(min_range), "minimum LiDAR range [m]")(
    "max_range", po::value<double>(&max_range)->default_value(max_range), "maximum LiDAR range [m]")(
    "pose_max_dt", po::value<double>(&pose_max_dt)->default_value(pose_max_dt), "max nearest trajectory time delta [sec]")(
    "perturb_xy", po::value<double>(&perturb_xy)->default_value(perturb_xy), "initial XY perturbation amplitude [m]")(
    "perturb_z", po::value<double>(&perturb_z)->default_value(perturb_z), "initial Z perturbation amplitude [m]")(
    "perturb_yaw_deg", po::value<double>(&perturb_yaw_deg)->default_value(perturb_yaw_deg), "initial yaw perturbation amplitude [deg]")(
    "max_iterations", po::value<int>(&max_iterations)->default_value(max_iterations), "VGICP max iterations")(
    "min_source_points", po::value<int>(&min_source_points)->default_value(min_source_points), "minimum downsampled scan points")(
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

  Cloud::Ptr map(new Cloud);
  if (pcl::io::loadPCDFile<PointT>(map_path, *map) != 0 || map->empty()) {
    std::cerr << "failed to load map: " << map_path << std::endl;
    return 1;
  }

  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(map);

  const auto trajectory = load_trajectory(trajectory_path);
  const double bag_start = trajectory.front().stamp - start;
  const double start_stamp = bag_start + start;
  const double end_stamp = start_stamp + duration;

  std::cout << "loaded map_points=" << map->size() << " trajectory_poses=" << trajectory.size() << " bag_start=" << std::fixed << bag_start << std::endl;

  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = bag_path;
  storage_options.storage_id = "sqlite3";
  rosbag2_cpp::ConverterOptions converter_options;

  rosbag2_cpp::readers::SequentialReader reader;
  reader.open(storage_options, converter_options);
  rosbag2_storage::StorageFilter filter;
  filter.topics = {points_topic};
  reader.set_filter(filter);
  reader.seek(static_cast<rcutils_time_point_value_t>(start_stamp * 1e9));

  rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serialization;
  std::vector<EvalResult> results;
  double next_sample_stamp = start_stamp;
  int scan_index = 0;

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
    if (!scan || stamp < start_stamp) {
      continue;
    }
    if (stamp > end_stamp) {
      break;
    }
    if (stamp + 1e-6 < next_sample_stamp) {
      continue;
    }
    next_sample_stamp += sample_interval;

    const PoseStamped* ref = nearest_pose(trajectory, stamp, pose_max_dt);
    if (!ref) {
      std::cerr << "skip stamp=" << std::fixed << stamp << " no nearby reference pose" << std::endl;
      continue;
    }

    Cloud::Ptr source = voxel_downsample(scan, source_leaf);
    if (static_cast<int>(source->size()) < min_source_points) {
      std::cerr << "skip stamp=" << std::fixed << stamp << " too few source points=" << source->size() << std::endl;
      continue;
    }

    const Eigen::Isometry3d init = perturb_pose(ref->T, scan_index, perturb_xy, perturb_z, perturb_yaw_deg);
    scan_index++;

    Cloud::Ptr target = crop_target(map, kdtree, init.translation(), crop_radius, target_leaf);
    if (static_cast<int>(target->size()) < min_target_points) {
      std::cerr << "skip stamp=" << std::fixed << stamp << " too few target points=" << target->size() << std::endl;
      continue;
    }

    fast_gicp::FastVGICP<PointT, PointT> vgicp;
    vgicp.setResolution(vgicp_resolution);
    vgicp.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
    vgicp.setNumThreads(std::max(1u, std::thread::hardware_concurrency()));
    vgicp.setMaxCorrespondenceDistance(5.0);
    vgicp.setMaximumIterations(max_iterations);
    vgicp.setTransformationEpsilon(1e-4);
    vgicp.setRotationEpsilon(1e-4);
    vgicp.setInputTarget(target);
    vgicp.setInputSource(source);

    Cloud aligned;
    const auto t0 = std::chrono::high_resolution_clock::now();
    vgicp.align(aligned, init.matrix().cast<float>());
    const auto t1 = std::chrono::high_resolution_clock::now();

    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    result.matrix() = vgicp.getFinalTransformation().cast<double>();

    const Eigen::Isometry3d init_delta = ref->T.inverse() * init;
    const Eigen::Isometry3d final_delta = ref->T.inverse() * result;
    EvalResult eval;
    eval.stamp = stamp;
    eval.raw_points = raw_points;
    eval.source_points = static_cast<int>(source->size());
    eval.target_points = static_cast<int>(target->size());
    eval.elapsed_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    eval.converged = vgicp.hasConverged();
    eval.fitness = vgicp.getFitnessScore(5.0);
    eval.init_xy_error = init_delta.translation().head<2>().norm();
    eval.final_xy_error = final_delta.translation().head<2>().norm();
    eval.final_z_error = final_delta.translation().z();
    eval.final_yaw_error_deg = std::abs(wrap_pi(yaw_from_rotation(ref->T.linear()) - yaw_from_rotation(result.linear()))) * 180.0 / M_PI;
    results.push_back(eval);

    std::cout << "stamp=" << std::fixed << stamp << " src=" << eval.source_points << " tgt=" << eval.target_points
              << " converged=" << eval.converged << " fitness=" << eval.fitness << " init_xy=" << eval.init_xy_error
              << " final_xy=" << eval.final_xy_error << " yaw_deg=" << eval.final_yaw_error_deg << " ms=" << eval.elapsed_ms << std::endl;
  }

  write_csv(output_csv, results);

  const int n = results.size();
  if (n == 0) {
    std::cerr << "no evaluated scans" << std::endl;
    return 2;
  }

  std::vector<double> xy_errors;
  std::vector<double> yaw_errors;
  std::vector<double> fitnesses;
  int converged = 0;
  int good_2m = 0;
  int good_5m = 0;
  for (const auto& r : results) {
    xy_errors.push_back(r.final_xy_error);
    yaw_errors.push_back(r.final_yaw_error_deg);
    fitnesses.push_back(r.fitness);
    converged += r.converged ? 1 : 0;
    good_2m += r.converged && r.final_xy_error < 2.0 && r.final_yaw_error_deg < 2.0 ? 1 : 0;
    good_5m += r.converged && r.final_xy_error < 5.0 && r.final_yaw_error_deg < 5.0 ? 1 : 0;
  }

  auto mean = [](const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / std::max<size_t>(1, values.size());
  };
  auto percentile = [](std::vector<double> values, double p) {
    if (values.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1, static_cast<size_t>(std::floor(p * (values.size() - 1))));
    return values[index];
  };

  std::cout << "SUMMARY"
            << " scans=" << n << " converged=" << converged << " good_2m_2deg=" << good_2m << " good_5m_5deg=" << good_5m
            << " xy_mean=" << mean(xy_errors) << " xy_p50=" << percentile(xy_errors, 0.50) << " xy_p90=" << percentile(xy_errors, 0.90)
            << " xy_max=" << percentile(xy_errors, 1.0) << " yaw_mean_deg=" << mean(yaw_errors) << " yaw_p90_deg=" << percentile(yaw_errors, 0.90)
            << " fitness_mean=" << mean(fitnesses) << " csv=" << output_csv << std::endl;

  return 0;
}
