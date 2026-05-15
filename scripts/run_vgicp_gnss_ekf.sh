#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MAP_PATH="${1:-${MAP_PATH:-/maps/exports/glim_40_700_gnss_odom_gpu_xy_leaf020.pcd}}"
POINTS_TOPIC="${POINTS_TOPIC:-/front/lidar_point}"
FIX_TOPIC="${FIX_TOPIC:-/fix}"
OUTPUT_FRAME="${OUTPUT_FRAME:-utm_enu}"
EKF_CONFIG="${EKF_CONFIG:-$REPO_ROOT/src/glim_ros2/config/ekf_vgicp_gnss.yaml}"

INITIAL_X="${INITIAL_X:-0.0}"
INITIAL_Y="${INITIAL_Y:-0.0}"
INITIAL_Z="${INITIAL_Z:-0.0}"
INITIAL_YAW_DEG="${INITIAL_YAW_DEG:-0.0}"

MAP_TO_OUTPUT_X="${MAP_TO_OUTPUT_X:-0.0}"
MAP_TO_OUTPUT_Y="${MAP_TO_OUTPUT_Y:-0.0}"
MAP_TO_OUTPUT_Z="${MAP_TO_OUTPUT_Z:-0.0}"
MAP_TO_OUTPUT_YAW_DEG="${MAP_TO_OUTPUT_YAW_DEG:-0.0}"

source /opt/ros/humble/setup.bash
if [[ -f "$REPO_ROOT/install/setup.bash" ]]; then
  source "$REPO_ROOT/install/setup.bash"
fi

ros2 launch glim_ros vgicp_gnss_ekf.launch.py \
  map_path:="$MAP_PATH" \
  points_topic:="$POINTS_TOPIC" \
  fix_topic:="$FIX_TOPIC" \
  output_frame:="$OUTPUT_FRAME" \
  ekf_config:="$EKF_CONFIG" \
  initial_x:="$INITIAL_X" \
  initial_y:="$INITIAL_Y" \
  initial_z:="$INITIAL_Z" \
  initial_yaw_deg:="$INITIAL_YAW_DEG" \
  map_to_output_x:="$MAP_TO_OUTPUT_X" \
  map_to_output_y:="$MAP_TO_OUTPUT_Y" \
  map_to_output_z:="$MAP_TO_OUTPUT_Z" \
  map_to_output_yaw_deg:="$MAP_TO_OUTPUT_YAW_DEG"
