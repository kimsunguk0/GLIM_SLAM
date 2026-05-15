#!/usr/bin/env python3
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def lla_to_ecef(lat_deg, lon_deg, alt):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt) * cos_lat * cos_lon
    y = (n + alt) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return x, y, z


class NavSatFixToOdom(Node):
    def __init__(self):
        super().__init__("navsatfix_to_odom")
        self.declare_parameter("fix_topic", "/fix")
        self.declare_parameter("odom_topic", "/gnss/odom_enu")
        self.declare_parameter("frame_id", "utm_enu")
        self.declare_parameter("child_frame_id", "gnss")
        self.declare_parameter("min_status", 0)
        self.declare_parameter("use_z", False)
        self.declare_parameter("manual_origin", False)
        self.declare_parameter("origin_latitude", 0.0)
        self.declare_parameter("origin_longitude", 0.0)
        self.declare_parameter("origin_altitude", 0.0)
        self.declare_parameter("unknown_stddev_xy", 5.0)
        self.declare_parameter("unknown_stddev_z", 20.0)

        fix_topic = self.get_parameter("fix_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.min_status = int(self.get_parameter("min_status").value)
        self.use_z = bool(self.get_parameter("use_z").value)
        self.unknown_stddev_xy = float(self.get_parameter("unknown_stddev_xy").value)
        self.unknown_stddev_z = float(self.get_parameter("unknown_stddev_z").value)

        self.origin_initialized = False
        self.origin_ecef = None
        self.east = None
        self.north = None
        self.up = None
        self.origin_alt = 0.0

        if bool(self.get_parameter("manual_origin").value):
            self.initialize_origin(
                float(self.get_parameter("origin_latitude").value),
                float(self.get_parameter("origin_longitude").value),
                float(self.get_parameter("origin_altitude").value),
            )

        self.pub = self.create_publisher(Odometry, odom_topic, 20)
        self.sub = self.create_subscription(NavSatFix, fix_topic, self.fix_callback, 20)
        self.get_logger().info(
            f"NavSatFixToOdom fix_topic={fix_topic} odom_topic={odom_topic} frame_id={self.frame_id} min_status={self.min_status}"
        )

    def initialize_origin(self, lat, lon, alt):
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)
        self.origin_ecef = lla_to_ecef(lat, lon, alt)
        self.east = (-sin_lon, cos_lon, 0.0)
        self.north = (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)
        self.up = (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)
        self.origin_alt = alt
        self.origin_initialized = True
        self.get_logger().info(f"GNSS ENU origin lat={lat:.9f} lon={lon:.9f} alt={alt:.3f}")

    @staticmethod
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def fix_callback(self, msg):
        if msg.status.status < self.min_status:
            return
        if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude) and math.isfinite(msg.altitude)):
            return
        if not self.origin_initialized:
            self.initialize_origin(msg.latitude, msg.longitude, msg.altitude)

        ecef = lla_to_ecef(msg.latitude, msg.longitude, msg.altitude)
        delta = (
            ecef[0] - self.origin_ecef[0],
            ecef[1] - self.origin_ecef[1],
            ecef[2] - self.origin_ecef[2],
        )
        east = self.dot(self.east, delta)
        north = self.dot(self.north, delta)
        up = msg.altitude - self.origin_alt

        odom = Odometry()
        odom.header = msg.header
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x = float(east)
        odom.pose.pose.position.y = float(north)
        odom.pose.pose.position.z = float(up if self.use_z else 0.0)
        odom.pose.pose.orientation.w = 1.0

        odom.pose.covariance = [0.0] * 36
        if msg.position_covariance_type == NavSatFix.COVARIANCE_TYPE_UNKNOWN:
            xy_var = self.unknown_stddev_xy * self.unknown_stddev_xy
            z_var = self.unknown_stddev_z * self.unknown_stddev_z
            odom.pose.covariance[0] = xy_var
            odom.pose.covariance[7] = xy_var
            odom.pose.covariance[14] = z_var
        else:
            cov = msg.position_covariance
            odom.pose.covariance[0] = float(cov[0])
            odom.pose.covariance[1] = float(cov[1])
            odom.pose.covariance[2] = float(cov[2])
            odom.pose.covariance[6] = float(cov[3])
            odom.pose.covariance[7] = float(cov[4])
            odom.pose.covariance[8] = float(cov[5])
            odom.pose.covariance[12] = float(cov[6])
            odom.pose.covariance[13] = float(cov[7])
            odom.pose.covariance[14] = float(cov[8])

        if not self.use_z:
            odom.pose.covariance[14] = 1e6
        odom.pose.covariance[21] = 1e6
        odom.pose.covariance[28] = 1e6
        odom.pose.covariance[35] = 1e6
        self.pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = NavSatFixToOdom()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
