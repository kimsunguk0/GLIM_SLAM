#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped

import utm  # pip install utm


def navsat_time_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class GnssUtmNode(Node):
    """
    /fix (NavSatFix) + /imu/data (Imu) -> /gnss_utm_pose (PoseWithCovarianceStamped)

    - position: UTM-local (E - E0, N - N0, H - H0)
    - orientation: IMU에서 온 quaternion (ENU 기준이라고 가정)
    """

    def __init__(self):
        super().__init__("gnss_utm_node")

        # ------------ Parameters ------------
        self.declare_parameter("fix_topic", "/fix")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("output_topic", "/gnss_utm_pose")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("use_altitude", True)

        fix_topic = self.get_parameter("fix_topic").get_parameter_value().string_value
        imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.use_altitude = self.get_parameter("use_altitude").get_parameter_value().bool_value

        # ------------ Sub/Pub ------------
        self.sub_fix = self.create_subscription(
            NavSatFix, fix_topic, self.fix_callback, 10
        )
        self.sub_imu = self.create_subscription(
            Imu, imu_topic, self.imu_callback, 50
        )
        self.pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, output_topic, 10
        )

        # ------------ State ------------
        self.origin_initialized: bool = False
        self.E0: Optional[float] = None
        self.N0: Optional[float] = None
        self.H0: Optional[float] = None
        self.utm_zone: Optional[int] = None
        self.utm_band: Optional[str] = None

        self.last_imu: Optional[Imu] = None

        self.get_logger().info(
            f"[GnssUtmNode] started. fix_topic={fix_topic}, imu_topic={imu_topic}, "
            f"output_topic={output_topic}, frame_id={self.frame_id}"
        )

    # ------------ Callbacks ------------

    def imu_callback(self, msg: Imu):
        # orientation만 캐시 (IMU가 ENU 기준으로 준다고 가정)
        self.last_imu = msg

    def fix_callback(self, msg: NavSatFix):
        # 상태 체크 (필요하면 더 빡세게)
        if msg.status.status < 0:
            # self.get_logger().debug("NavSat status not FIX, skipping")
            return

        lat = msg.latitude
        lon = msg.longitude
        alt = msg.altitude

        # 1) LLA -> UTM
        E, N, zone, band = utm.from_latlon(lat, lon)
        H = alt

        # 2) origin 세팅 (첫 FIX에서 한 번만)
        if not self.origin_initialized:
            self.utm_zone = zone
            self.utm_band = band
            self.E0 = E
            self.N0 = N
            self.H0 = H if self.use_altitude else 0.0
            self.origin_initialized = True
            self.get_logger().info(
                f"[GnssUtmNode] UTM origin set: lat={lat:.8f}, lon={lon:.8f}, "
                f"E0={self.E0:.3f}, N0={self.N0:.3f}, H0={self.H0:.3f}, "
                f"zone={self.utm_zone}{self.utm_band}"
            )

        if not self.origin_initialized:
            return

        # 3) UTM -> UTM-local (map 좌표)
        x = E - self.E0
        y = N - self.N0
        z = (H - self.H0) if self.use_altitude else 0.0

        # 4) PoseWithCovarianceStamped 메시지 구성
        out = PoseWithCovarianceStamped()
        out.header = msg.header
        out.header.frame_id = self.frame_id  # "map"

        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.position.z = float(z)

        # orientation: ENU 기준 IMU 쿼터니언을 그대로 사용 (IMU frame = base_link 라고 가정)
        if self.last_imu is not None:
            out.pose.pose.orientation = self.last_imu.orientation
        else:
            out.pose.pose.orientation.w = 1.0
            out.pose.pose.orientation.x = 0.0
            out.pose.pose.orientation.y = 0.0
            out.pose.pose.orientation.z = 0.0

        # 5) covariance: NavSatFix.position_covariance -> pose.covariance 상단 3x3
        out.pose.covariance = [0.0] * 36
        pos_cov = msg.position_covariance  # row-major 3x3

        for r in range(3):
            for c in range(3):
                out.pose.covariance[r * 6 + c] = float(pos_cov[r * 3 + c])

        self.pub_pose.publish(out)


def main(args=None):
    print("[fix_to_utm] starting...")  # 최소 확인용 프린트

    rclpy.init(args=args)
    node = GnssUtmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[fix_to_utm] Ctrl-C, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("[fix_to_utm] done.")


if __name__ == "__main__":
    main()
