#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DUMP_PATH="${1:-${DUMP_PATH:-/maps/glim_40_700_gnss_odom_gpu_xy}}"
OUTPUT_PREFIX="${2:-${OUTPUT_PREFIX:-/maps/exports/glim_40_700_gnss_odom_gpu_xy_leaf020}}"
LEAF_SIZE="${LEAF_SIZE:-0.20}"
FORMAT="${FORMAT:-pcd}"

source /opt/ros/humble/setup.bash
if [[ -f "$REPO_ROOT/install/setup.bash" ]]; then
  source "$REPO_ROOT/install/setup.bash"
fi

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

ros2 run glim_ros export_map "$DUMP_PATH" "$OUTPUT_PREFIX" \
  --format "$FORMAT" \
  --leaf_size "$LEAF_SIZE"

echo "Map export prefix: $OUTPUT_PREFIX"
