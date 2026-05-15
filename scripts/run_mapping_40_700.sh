#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BAG_PATH="${1:-${BAG_PATH:-/data/processed_bags/clock_synced_fixed_v1}}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/glim_gnss_odom_gpu_xy}"
DUMP_PATH="${DUMP_PATH:-/maps/glim_40_700_gnss_odom_gpu_xy}"
START_OFFSET="${START_OFFSET:-40.0}"
PLAYBACK_DURATION="${PLAYBACK_DURATION:-660.0}"

source /opt/ros/humble/setup.bash
if [[ -f "$REPO_ROOT/install/setup.bash" ]]; then
  source "$REPO_ROOT/install/setup.bash"
fi

mkdir -p "$(dirname "$DUMP_PATH")"

ros2 run glim_ros glim_rosbag "$BAG_PATH" \
  --ros-args \
  -p config_path:="$CONFIG_PATH" \
  -p dump_path:="$DUMP_PATH" \
  -p start_offset:="$START_OFFSET" \
  -p playback_duration:="$PLAYBACK_DURATION" \
  -p auto_quit:=true

echo "GLIM dump: $DUMP_PATH"
