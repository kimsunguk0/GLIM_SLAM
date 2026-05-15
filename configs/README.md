# GLIM Config Presets

Only reproducible presets are kept here. Old experiment directories were removed from the upload-ready tree.

| Preset | Purpose | GNSS | Global Mapping |
|---|---|---|---|
| `glim_gnss_odom_gpu_xy` | Default stable mapping preset used for the current metrics | Odometry graph XY factor | `libglobal_mapping.so`, optimization disabled |
| `glim_gnss_odom_gpu_xy_loop` | Same GNSS odometry setup, with explicit pose-graph loop detection | Odometry graph XY factor | `libglobal_mapping_pose_graph.so` |
| `glim_nognss_gpu_baseline` | Ablation/debug baseline using the same LiDAR/IMU topics and GPU settings | Off | `libglobal_mapping.so`, optimization disabled |

Use an absolute path when launching GLIM:

```bash
ros2 run glim_ros glim_rosbag /data/processed_bags/clock_synced_fixed_v1 \
  --ros-args \
  -p config_path:=/root/incheon_glim_slam_src/configs/glim_gnss_odom_gpu_xy \
  -p dump_path:=/maps/glim_40_700_gnss_odom_gpu_xy \
  -p start_offset:=40.0 \
  -p playback_duration:=660.0 \
  -p auto_quit:=true
```

The `_loop` preset is available when a route revisits the same place. For a mostly one-way bridge drive, loop closure usually has little opportunity; the GNSS odometry factor is the primary stabilizer.
