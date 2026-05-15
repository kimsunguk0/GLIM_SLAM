#!/usr/bin/env python3
import math


def compute_yaw_offset(E1, N1, E2, N2, yaw_imu_deg):
  """Compute yaw offset (deg) = heading_gnss - yaw_imu, normalized to [-180, 180]."""
  heading_rad = math.atan2(N2 - N1, E2 - E1)
  heading_deg = math.degrees(heading_rad)

  offset = heading_deg - yaw_imu_deg
  # normalize to [-180, 180]
  offset = (offset + 180.0) % 360.0 - 180.0
  return heading_deg, offset


if __name__ == "__main__":
  # example: replace with your actual values
  E1, N1 = 284746.170, 4151115.493
  E2, N2 = 284750.000, 4151120.000
  yaw_imu_deg = -128.184  # IMU yaw in ENU frame at the same time

  heading_deg, offset_deg = compute_yaw_offset(E1, N1, E2, N2, yaw_imu_deg)
  print(f"GNSS heading: {heading_deg:.3f} deg")
  print(f"IMU yaw:      {yaw_imu_deg:.3f} deg")
  print(f"yaw_offset:   {offset_deg:.3f} deg  (set this to yaw_offset_deg)")
