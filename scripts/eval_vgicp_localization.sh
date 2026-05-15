#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BAG_PATH="${1:-${BAG_PATH:-/data/processed_bags/clock_synced_fixed_v1}}"
MAP_PATH="${2:-${MAP_PATH:-/maps/exports/glim_40_700_gnss_odom_gpu_xy_leaf020.pcd}}"
TRAJECTORY_PATH="${3:-${TRAJECTORY_PATH:-/maps/glim_40_700_gnss_odom_gpu_xy/traj_lidar.txt}}"
OUTPUT_CSV="${OUTPUT_CSV:-/maps/exports/localize_eval_40_700.csv}"
POINTS_TOPIC="${POINTS_TOPIC:-/front/lidar_point}"
START="${START:-40.0}"
DURATION="${DURATION:-660.0}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10.0}"
PERTURB_XY="${PERTURB_XY:-2.0}"
PERTURB_YAW_DEG="${PERTURB_YAW_DEG:-2.0}"

source /opt/ros/humble/setup.bash
if [[ -f "$REPO_ROOT/install/setup.bash" ]]; then
  source "$REPO_ROOT/install/setup.bash"
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"

ros2 run glim_ros localize_map_eval \
  --bag "$BAG_PATH" \
  --map "$MAP_PATH" \
  --trajectory "$TRAJECTORY_PATH" \
  --points_topic "$POINTS_TOPIC" \
  --start "$START" \
  --duration "$DURATION" \
  --sample_interval "$SAMPLE_INTERVAL" \
  --perturb_xy "$PERTURB_XY" \
  --perturb_yaw_deg "$PERTURB_YAW_DEG" \
  --output_csv "$OUTPUT_CSV"

"$REPO_ROOT/scripts/summarize_localization_csv.py" "$OUTPUT_CSV"
