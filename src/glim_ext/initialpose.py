#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import NavSatFix, NavSatStatus, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped

import tf2_ros

# ----------------------
# UTM conversion helpers
# ----------------------
def _try_make_utm_converter():
    """
    Returns a function latlon_to_utm(lat, lon) -> (E, N, zone_str)
    Tries pyproj first, then utm.
    """
    # 1) pyproj (recommended)
    try:
        import pyproj

        def latlon_to_utm(lat, lon):
            zone = int((lon + 180.0) / 6.0) + 1
            # north hemisphere for Korea
            epsg = 32600 + zone if lat >= 0.0 else 32700 + zone
            transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            E, N = transformer.transform(lon, lat)
            return float(E), float(N), f"{zone}{'N' if lat >= 0.0 else 'S'}"

        return latlon_to_utm, "pyproj"
    except Exception:
        pass

    # 2) utm (pip install utm)
    try:
        import utm

        def latlon_to_utm(lat, lon):
            E, N, zone_number, zone_letter = utm.from_latlon(lat, lon)
            # zone_letter는 latitude band (예: 37도면 S가 나올 수 있음) -> hemisphere랑 다름!
            return float(E), float(N), f"{zone_number}{zone_letter}"

        return latlon_to_utm, "utm"
    except Exception:
        pass

    return None, None


def quat_to_yaw(qx, qy, qz, qw) -> float:
    """ROS quaternion(x,y,z,w) -> yaw [rad] (ENU 기준)"""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw: float):
    """yaw-only quaternion (x=y=0)"""
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


# ----------------------
# Main Node
# ----------------------
class GnssImuInitialPoseNode(Node):
    def __init__(self):
        super().__init__("gnss_imu_initialpose")

        # ---- params ----
        self.declare_parameter("fix_topic", "/fix")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("output_topic", "/initialpose")

        # output pose frame (너 로컬라이저 odom_frame과 맞춰야 함: odom or map)
        self.declare_parameter("output_frame_id", "odom")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("gps_frame", "gps")

        # datum: 맵 만들 때 썼던 기준점 (lat0, lon0, alt0)
        # 이걸 넣어야 맵(UTM-local)과 동일 원점으로 initialpose가 나옴
        self.declare_parameter("use_datum", True)
        self.declare_parameter("datum", [37.48265817, 126.56632283, 30.2])  # <- 너희 맵 원점으로 바꿔라

        self.declare_parameter("zero_altitude", True)  # z를 0으로 할지
        self.declare_parameter("yaw_offset_deg", 0.0)  # 필요하면 상수 보정

        # fix/imu timestamp mismatch 허용 (너무 차이나면 안 씀)
        self.declare_parameter("max_time_diff_sec", 0.5)

        # publish behavior
        self.declare_parameter("publish_once", True)
        self.declare_parameter("exit_after_publish", True)

        # covariance (optional)
        self.declare_parameter("std_xy", 2.0)     # meters
        self.declare_parameter("std_z", 5.0)      # meters
        self.declare_parameter("std_yaw_deg", 10.0)  # deg

        self.fix_topic = self.get_parameter("fix_topic").value
        self.imu_topic = self.get_parameter("imu_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.output_frame_id = self.get_parameter("output_frame_id").value

        self.base_frame = self.get_parameter("base_frame").value
        self.gps_frame = self.get_parameter("gps_frame").value

        self.use_datum = bool(self.get_parameter("use_datum").value)
        self.datum = list(self.get_parameter("datum").value)
        self.zero_alt = bool(self.get_parameter("zero_altitude").value)
        self.yaw_offset = float(self.get_parameter("yaw_offset_deg").value) * math.pi / 180.0
        self.max_dt = float(self.get_parameter("max_time_diff_sec").value)

        self.publish_once = bool(self.get_parameter("publish_once").value)
        self.exit_after_publish = bool(self.get_parameter("exit_after_publish").value)

        self.std_xy = float(self.get_parameter("std_xy").value)
        self.std_z = float(self.get_parameter("std_z").value)
        self.std_yaw = float(self.get_parameter("std_yaw_deg").value) * math.pi / 180.0

        # ---- UTM converter ----
        self.latlon_to_utm, backend = _try_make_utm_converter()
        if self.latlon_to_utm is None:
            self.get_logger().error(
                "No UTM converter available. Install one:\n"
                "  sudo apt install python3-pyproj\n"
                "or\n"
                "  pip install pyproj\n"
                "or\n"
                "  pip install utm"
            )
            raise RuntimeError("No UTM converter installed.")
        self.get_logger().info(f"UTM conversion backend: {backend}")

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- subs ----
        self.last_fix = None
        self.last_imu = None

        self.sub_fix = self.create_subscription(NavSatFix, self.fix_topic, self.on_fix, 50)
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.on_imu, 200)

        # ---- pub ----
        self.pub_init = self.create_publisher(PoseWithCovarianceStamped, self.output_topic, 10)

        self.published = False

        # timer to try publish
        self.timer = self.create_timer(0.1, self.try_publish_initialpose)

        self.get_logger().info(
            f"[GnssImuInitialPoseNode] started.\n"
            f"  fix_topic={self.fix_topic}\n"
            f"  imu_topic={self.imu_topic}\n"
            f"  output_topic={self.output_topic}\n"
            f"  output_frame_id={self.output_frame_id}\n"
            f"  base_frame={self.base_frame}, gps_frame={self.gps_frame}\n"
            f"  use_datum={self.use_datum}, datum={self.datum}\n"
            f"  zero_altitude={self.zero_alt}, yaw_offset_deg={self.yaw_offset*180/math.pi:.3f}"
        )

    def on_fix(self, msg: NavSatFix):
        # require valid fix
        if msg.status.status == NavSatStatus.STATUS_NO_FIX:
            return
        self.last_fix = msg

    def on_imu(self, msg: Imu):
        self.last_imu = msg

    def lookup_base_to_gps(self):
        # static tf -> time=0 lookup
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.gps_frame, Time())
            t = tf.transform.translation
            return np.array([t.x, t.y, t.z], dtype=float)
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed: {self.base_frame} -> {self.gps_frame} : {e}"
            )
            return None

    def try_publish_initialpose(self):
        if self.publish_once and self.published:
            return

        if self.last_fix is None or self.last_imu is None:
            return

        # time diff check
        t_fix = self.last_fix.header.stamp.sec + self.last_fix.header.stamp.nanosec * 1e-9
        t_imu = self.last_imu.header.stamp.sec + self.last_imu.header.stamp.nanosec * 1e-9
        if abs(t_fix - t_imu) > self.max_dt:
            self.get_logger().warn_throttle(
                2.0,
                f"fix/imu time mismatch: |{t_fix:.3f}-{t_imu:.3f}|={abs(t_fix-t_imu):.3f}s > {self.max_dt:.3f}s"
            )
            return

        lever_base_to_gps = self.lookup_base_to_gps()
        if lever_base_to_gps is None:
            return

        # 1) fix -> UTM (E,N)
        lat = float(self.last_fix.latitude)
        lon = float(self.last_fix.longitude)
        alt = float(self.last_fix.altitude)

        E, N, zone = self.latlon_to_utm(lat, lon)

        # 2) datum -> UTM origin (E0,N0)
        if self.use_datum:
            lat0, lon0, alt0 = float(self.datum[0]), float(self.datum[1]), float(self.datum[2])
            E0, N0, zone0 = self.latlon_to_utm(lat0, lon0)
            # zone mismatch check (should not happen if datum nearby)
            if zone != zone0:
                self.get_logger().warn(f"UTM zone mismatch: fix={zone}, datum={zone0} (still computing local diff)")
        else:
            # use first fix as datum (not recommended unless map also used same)
            E0, N0, alt0 = E, N, alt

        x_gps = E - E0
        y_gps = N - N0
        z_gps = 0.0 if self.zero_alt else (alt - alt0)

        # 3) imu yaw (ENU) + optional offset
        q = self.last_imu.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w) + self.yaw_offset

        # 4) lever arm correction: p_base = p_gps - Rz(yaw) * t_base_gps
        c = math.cos(yaw)
        s = math.sin(yaw)
        Rz = np.array([[c, -s, 0.0],
                       [s,  c, 0.0],
                       [0.0, 0.0, 1.0]], dtype=float)

        p_gps = np.array([x_gps, y_gps, z_gps], dtype=float)
        p_base = p_gps - (Rz @ lever_base_to_gps)

        # 5) yaw-only quaternion
        ox, oy, oz, ow = yaw_to_quat(yaw)

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.last_fix.header.stamp  # 또는 self.get_clock().now().to_msg()
        msg.header.frame_id = self.output_frame_id

        msg.pose.pose.position.x = float(p_base[0])
        msg.pose.pose.position.y = float(p_base[1])
        msg.pose.pose.position.z = float(p_base[2])

        msg.pose.pose.orientation.x = float(ox)
        msg.pose.pose.orientation.y = float(oy)
        msg.pose.pose.orientation.z = float(oz)
        msg.pose.pose.orientation.w = float(ow)

        # covariance (diagonal)
        cov = [0.0] * 36
        cov[0]  = self.std_xy**2
        cov[7]  = self.std_xy**2
        cov[14] = self.std_z**2
        cov[35] = self.std_yaw**2
        msg.pose.covariance = cov

        self.pub_init.publish(msg)

        self.get_logger().info(
            f"Published /initialpose (base_link) in {self.output_frame_id}:\n"
            f"  gps_utm_local = ({x_gps:.3f}, {y_gps:.3f}, {z_gps:.3f})  zone={zone}\n"
            f"  lever_base->gps = ({lever_base_to_gps[0]:.3f}, {lever_base_to_gps[1]:.3f}, {lever_base_to_gps[2]:.3f})\n"
            f"  base_pose = ({p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f}), yaw={yaw*180/math.pi:.3f} deg"
        )

        self.published = True

        if self.exit_after_publish:
            self.get_logger().info("exit_after_publish=true -> shutting down.")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = GnssImuInitialPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
