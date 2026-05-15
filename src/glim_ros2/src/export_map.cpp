#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <Eigen/Core>

#include <glim/mapping/sub_map.hpp>

struct PointXYZI {
  float x;
  float y;
  float z;
  float intensity;
};

struct VoxelKey {
  std::int64_t x;
  std::int64_t y;
  std::int64_t z;

  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey& key) const {
    const auto h1 = std::hash<std::int64_t>{}(key.x);
    const auto h2 = std::hash<std::int64_t>{}(key.y);
    const auto h3 = std::hash<std::int64_t>{}(key.z);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2)) ^ (h3 + 0x9e3779b97f4a7c15ULL + (h2 << 6) + (h2 >> 2));
  }
};

bool write_pcd_binary(const std::string& path, const std::vector<PointXYZI>& points) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    std::cerr << "failed to open " << path << std::endl;
    return false;
  }

  ofs << "# .PCD v0.7 - Point Cloud Data file format\n";
  ofs << "VERSION 0.7\n";
  ofs << "FIELDS x y z intensity\n";
  ofs << "SIZE 4 4 4 4\n";
  ofs << "TYPE F F F F\n";
  ofs << "COUNT 1 1 1 1\n";
  ofs << "WIDTH " << points.size() << "\n";
  ofs << "HEIGHT 1\n";
  ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
  ofs << "POINTS " << points.size() << "\n";
  ofs << "DATA binary\n";
  ofs.write(reinterpret_cast<const char*>(points.data()), static_cast<std::streamsize>(points.size() * sizeof(PointXYZI)));
  return static_cast<bool>(ofs);
}

bool write_ply_binary(const std::string& path, const std::vector<PointXYZI>& points) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    std::cerr << "failed to open " << path << std::endl;
    return false;
  }

  ofs << "ply\n";
  ofs << "format binary_little_endian 1.0\n";
  ofs << "element vertex " << points.size() << "\n";
  ofs << "property float x\n";
  ofs << "property float y\n";
  ofs << "property float z\n";
  ofs << "property float intensity\n";
  ofs << "end_header\n";
  ofs.write(reinterpret_cast<const char*>(points.data()), static_cast<std::streamsize>(points.size() * sizeof(PointXYZI)));
  return static_cast<bool>(ofs);
}

std::vector<PointXYZI> voxel_downsample(const std::vector<PointXYZI>& points, const double leaf_size) {
  if (leaf_size <= 0.0) {
    return points;
  }

  std::unordered_map<VoxelKey, PointXYZI, VoxelKeyHash> voxels;
  voxels.reserve(points.size());

  const double inv_leaf = 1.0 / leaf_size;
  for (const auto& point : points) {
    const VoxelKey key{
      static_cast<std::int64_t>(std::floor(point.x * inv_leaf)),
      static_cast<std::int64_t>(std::floor(point.y * inv_leaf)),
      static_cast<std::int64_t>(std::floor(point.z * inv_leaf)),
    };
    voxels.emplace(key, point);
  }

  std::vector<PointXYZI> downsampled;
  downsampled.reserve(voxels.size());
  for (const auto& voxel : voxels) {
    downsampled.push_back(voxel.second);
  }
  return downsampled;
}

int main(int argc, char** argv) {
  namespace po = boost::program_options;

  po::options_description desc("Export a GLIM dump directory to PLY/PCD");
  desc.add_options()                                                                                      //
    ("help", "produce help message")                                                                      //
    ("map_path", po::value<std::string>()->required(), "Input GLIM dump directory")                       //
    ("output", po::value<std::string>()->required(), "Output path without extension, or full file path")  //
    ("format", po::value<std::string>()->default_value("both"), "ply, pcd, or both")                     //
    ("leaf_size", po::value<double>()->default_value(0.0), "Optional voxel leaf size in meters")          //
    ;

  po::positional_options_description positional;
  positional.add("map_path", 1);
  positional.add("output", 1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(positional).run(), vm);
    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n\n" << desc << std::endl;
    return 1;
  }

  const std::string map_path = vm["map_path"].as<std::string>();
  const std::string output = vm["output"].as<std::string>();
  const std::string format = vm["format"].as<std::string>();
  const double leaf_size = vm["leaf_size"].as<double>();

  std::ifstream graph_ifs(map_path + "/graph.txt");
  if (!graph_ifs) {
    std::cerr << "failed to open " << map_path << "/graph.txt" << std::endl;
    return 1;
  }

  std::string token;
  int num_submaps = 0;
  graph_ifs >> token >> num_submaps;
  if (!graph_ifs || token != "num_submaps:") {
    std::cerr << "invalid graph.txt header" << std::endl;
    return 1;
  }

  std::vector<PointXYZI> points;
  for (int i = 0; i < num_submaps; i++) {
    const std::string submap_path = (boost::format("%s/%06d") % map_path % i).str();
    const auto submap = glim::SubMap::load(submap_path);
    if (!submap || !submap->frame || !submap->frame->points) {
      std::cerr << "failed to load submap " << submap_path << std::endl;
      return 1;
    }

    const bool has_intensities = submap->frame->has_intensities() && submap->frame->intensities != nullptr;
    for (size_t j = 0; j < submap->frame->size(); j++) {
      const Eigen::Vector4d world_point = submap->T_world_origin * submap->frame->points[j];
      points.push_back(PointXYZI{
        static_cast<float>(world_point.x()),
        static_cast<float>(world_point.y()),
        static_cast<float>(world_point.z()),
        static_cast<float>(has_intensities ? submap->frame->intensities[j] : 0.0),
      });
    }

    if ((i + 1) % 25 == 0 || i + 1 == num_submaps) {
      std::cout << "loaded " << (i + 1) << "/" << num_submaps << " submaps, points=" << points.size() << std::endl;
    }
  }

  if (leaf_size > 0.0) {
    std::cout << "voxel downsampling leaf_size=" << leaf_size << " input_points=" << points.size() << std::endl;
    points = voxel_downsample(points, leaf_size);
    std::cout << "downsampled_points=" << points.size() << std::endl;
  }

  bool ok = true;
  if (format == "ply" || format == "both") {
    const std::string path = output.size() >= 4 && output.substr(output.size() - 4) == ".ply" ? output : output + ".ply";
    std::cout << "writing " << path << std::endl;
    ok &= write_ply_binary(path, points);
  }

  if (format == "pcd" || format == "both") {
    const std::string path = output.size() >= 4 && output.substr(output.size() - 4) == ".pcd" ? output : output + ".pcd";
    std::cout << "writing " << path << std::endl;
    ok &= write_pcd_binary(path, points);
  }

  if (format != "ply" && format != "pcd" && format != "both") {
    std::cerr << "unsupported format: " << format << std::endl;
    return 1;
  }

  std::cout << "exported_points=" << points.size() << std::endl;
  return ok ? 0 : 1;
}
