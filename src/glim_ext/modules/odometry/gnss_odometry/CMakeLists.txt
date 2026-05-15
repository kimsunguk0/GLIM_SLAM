cmake_minimum_required(VERSION 3.5.2)
project(gnss_odometry)

set(CMAKE_CXX_STANDARD 17)

find_package(glim REQUIRED)
find_package(GTSAM REQUIRED)
find_package(spdlog REQUIRED)

if($ENV{ROS_VERSION} EQUAL 2)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
endif()

add_library(gnss_odometry SHARED
  src/glim_ext/gnss_odometry_module.cpp
  ../../mapping/gnss_global/src/glim_ext/geodetic.cpp
)
target_include_directories(gnss_odometry PRIVATE
  include
  ../../mapping/gnss_global/include
  ${GTSAM_INCLUDE_DIRS}
  ${glim_INCLUDE_DIRS}
)
target_link_libraries(gnss_odometry
  glim_ext
  ${GTSAM_LIBRARIES}
  ${glim_LIBRARIES}
  spdlog::spdlog
)

if($ENV{ROS_VERSION} EQUAL 2)
  ament_target_dependencies(gnss_odometry rclcpp sensor_msgs)
endif()
