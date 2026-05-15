#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import NavSatFix, NavSatStatus, Imu


def _try_make_utm_converter():
  """Return (func, backend_name) where func(lat, lon) -> (E, N, zone_str)."""
  # pyproj first (preferred)
  try:
    import pyproj  # type: ignore

    def latlon_to_utm(lat, lon):
      zone = int((lon + 180.0) / 6.0) + 1
      epsg = 32600 + zone if lat >= 0.0 else 32700 + zone
      transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
      E, N = transformer.transform(lon, lat)
      return float(E), float(N), f"{zone}{'N' if lat >= 0.0 else 'S'}"

    return latlon_to_utm, "pyproj"
  except Exception:
    pass

  # utm fallback
  try:
    import utm  # type: ignore

    def latlon_to_utm(lat, lon):
      E, N, zone_number, zone_letter = utm.from_latlon(lat, lon)
      return float(E), float(N), f"{zone_number}{zone_letter}"

    return latlon_to_utm, "utm"
  except Exception:
    pass

  return None, None


def quat_to_yaw(qx, qy, qz, qw) -> float:
  """ROS quaternion(x,y,z,w) -> yaw [rad] (ENU)."""
  siny_cosp = 2.0 * (qw * qz + qx * qy)
  cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
  return math.atan2(siny_cosp, cosy_cosp)


def normalize_deg(angle_deg: float) -> float:
  """Wrap to [-180, 180]."""
  return (angle_deg + 180.0) % 360.0 - 180.0


class YawOffsetNode(Node):
  """Subscribe fix+imu, compute GNSS heading and yaw offset vs IMU yaw."""

  def __init__(self):
    super().__init__("yaw_offset_node")

    # params
    self.declare_parameter("fix_topic", "/fix")
    self.declare_parameter("imu_topic", "/imu/data")
    self.declare_parameter("min_dist_m", 0.5)   # 최소 이동 거리
    self.declare_parameter("min_dt_sec", 0.1)   # 최소 시간 차이
    self.declare_parameter("max_dt_sec", 5.0)   # 최대 시간 차이

    self.fix_topic = self.get_parameter("fix_topic").value
    self.imu_topic = self.get_parameter("imu_topic").value
    self.min_dist = float(self.get_parameter("min_dist_m").value)
    self.min_dt = float(self.get_parameter("min_dt_sec").value)
    self.max_dt = float(self.get_parameter("max_dt_sec").value)

    # utm converter
    self.latlon_to_utm, backend = _try_make_utm_converter()
    if self.latlon_to_utm is None:
      self.get_logger().error("No UTM converter available (install pyproj or utm).")
      raise RuntimeError("No UTM converter available")
    self.get_logger().info(f"UTM backend: {backend}")

    # state
    self.prev_fix: Optional[Tuple[float, float, float]] = None  # (E, N, stamp_sec)
    self.last_imu: Optional[Imu] = None

    # subs
    self.sub_fix = self.create_subscription(NavSatFix, self.fix_topic, self.on_fix, 20)
    self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.on_imu, 100)

    self.get_logger().info(
      f"[YawOffsetNode] fix_topic={self.fix_topic}, imu_topic={self.imu_topic}, "
      f"min_dist_m={self.min_dist}, min_dt_sec={self.min_dt}, max_dt_sec={self.max_dt}"
    )

  def on_imu(self, msg: Imu):
    self.last_imu = msg

  def on_fix(self, msg: NavSatFix):
    if msg.status.status == NavSatStatus.STATUS_NO_FIX:
      return

    if self.last_imu is None:
      return

    # convert to UTM
    E, N, zone = self.latlon_to_utm(float(msg.latitude), float(msg.longitude))
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    if self.prev_fix is None:
      self.prev_fix = (E, N, t)
      return

    E0, N0, t0 = self.prev_fix
    dE = E - E0
    dN = N - N0
    dist = math.hypot(dE, dN)
    dt = t - t0

    if dist < self.min_dist or dt < self.min_dt or dt > self.max_dt:
      # update prev and continue
      self.prev_fix = (E, N, t)
      return

    heading_rad = math.atan2(dN, dE)
    heading_deg = math.degrees(heading_rad)

    # imu yaw (ENU)
    q = self.last_imu.orientation
    yaw_imu = quat_to_yaw(q.x, q.y, q.z, q.w)
    yaw_imu_deg = math.degrees(yaw_imu)

    offset = normalize_deg(heading_deg - yaw_imu_deg)

    self.get_logger().info(
      f"UTM zone={zone}  dE={dE:.3f} dN={dN:.3f} dist={dist:.3f} m dt={dt:.3f} s | "
      f"heading={heading_deg:.3f} deg, imu_yaw={yaw_imu_deg:.3f} deg, "
      f"yaw_offset_deg={offset:.3f}"
    )

    # shift window
    self.prev_fix = (E, N, t)


def main(args=None):
  rclpy.init(args=args)
  node = YawOffsetNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()
