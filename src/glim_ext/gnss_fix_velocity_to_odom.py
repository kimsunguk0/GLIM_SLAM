#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped, TwistStamped, TwistWithCovarianceStamped

import utm  # pip install utm


def yaw_to_quaternion(yaw_rad: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw_rad
    return (0.0, 0.0, math.sin(half), math.cos(half))


class GnssFixVelocityToOdomNode(Node):
    """
    /fix (NavSatFix) + /velocity (TwistStamped or TwistWithCovarianceStamped)
    -> /gnss_odom (geometry_msgs/PoseStamped)

    - position: UTM raw (E, N, H)
    - orientation: velocity heading (ENU) if enabled
    """

    def __init__(self):
        super().__init__("gnss_fix_velocity_to_odom")

        # ------------ Parameters ------------
        self.declare_parameter("fix_topic", "/fix")
        self.declare_parameter("velocity_topic", "/velocity")
        self.declare_parameter("velocity_type", "twist_stamped")
        self.declare_parameter("output_topic", "gnss_odom")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("use_altitude", True)
        self.declare_parameter("use_velocity_yaw", True)
        self.declare_parameter("speed_threshold", 0.5)
        self.declare_parameter("require_velocity", False)

        fix_topic = self.get_parameter("fix_topic").get_parameter_value().string_value
        vel_topic = self.get_parameter("velocity_topic").get_parameter_value().string_value
        vel_type = self.get_parameter("velocity_type").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.use_altitude = self.get_parameter("use_altitude").get_parameter_value().bool_value
        self.use_velocity_yaw = self.get_parameter("use_velocity_yaw").get_parameter_value().bool_value
        self.speed_threshold = self.get_parameter("speed_threshold").get_parameter_value().double_value
        self.require_velocity = self.get_parameter("require_velocity").get_parameter_value().bool_value

        # ------------ Sub/Pub ------------
        self.sub_fix = self.create_subscription(
            NavSatFix, fix_topic, self.fix_callback, 10
        )
        if vel_type == "twist_stamped":
            self.sub_vel = self.create_subscription(
                TwistStamped, vel_topic, self.vel_callback, 50
            )
        elif vel_type == "twist_with_covariance_stamped":
            self.sub_vel = self.create_subscription(
                TwistWithCovarianceStamped, vel_topic, self.vel_cov_callback, 50
            )
        else:
            raise RuntimeError(
                "velocity_type must be 'twist_stamped' or 'twist_with_covariance_stamped'"
            )
        self.pub_pose = self.create_publisher(PoseStamped, output_topic, 10)

        # ------------ State ------------
        self.zone_initialized: bool = False
        self.utm_zone: Optional[int] = None
        self.utm_band: Optional[str] = None

        self.last_vel_linear: Optional[Tuple[float, float, float]] = None

        self.get_logger().info(
            "GnssFixVelocityToOdomNode started. "
            f"fix_topic={fix_topic}, velocity_topic={vel_topic}, "
            f"velocity_type={vel_type}, output_topic={output_topic}, "
            f"frame_id={self.frame_id}"
        )

    # ------------ Callbacks ------------

    def vel_callback(self, msg: TwistStamped):
        self.last_vel_linear = (
            float(msg.twist.linear.x),
            float(msg.twist.linear.y),
            float(msg.twist.linear.z),
        )

    def vel_cov_callback(self, msg: TwistWithCovarianceStamped):
        self.last_vel_linear = (
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.linear.y),
            float(msg.twist.twist.linear.z),
        )

    def fix_callback(self, msg: NavSatFix):
        if msg.status.status < 0:
            return

        lat = msg.latitude
        lon = msg.longitude
        alt = msg.altitude

        E, N, zone, band = utm.from_latlon(lat, lon)
        H = alt

        if not self.zone_initialized:
            self.utm_zone = zone
            self.utm_band = band
            self.zone_initialized = True
            self.get_logger().info(
                f"UTM zone set: lat={lat:.8f}, lon={lon:.8f}, "
                f"zone={self.utm_zone}{self.utm_band}"
            )

        if self.require_velocity and self.last_vel_linear is None:
            return

        x = E
        y = N
        z = H if self.use_altitude else 0.0

        out = PoseStamped()
        out.header = msg.header
        out.header.frame_id = self.frame_id

        out.pose.position.x = float(x)
        out.pose.position.y = float(y)
        out.pose.position.z = float(z)

        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        if self.use_velocity_yaw and self.last_vel_linear is not None:
            vx, vy, _ = self.last_vel_linear
            speed = math.hypot(vx, vy)
            if speed >= self.speed_threshold:
                qx, qy, qz, qw = yaw_to_quaternion(math.atan2(vy, vx))

        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw

        self.pub_pose.publish(out)


def main(args=None):
    print("[gnss_fix_velocity_to_odom] starting...")
    rclpy.init(args=args)
    node = GnssFixVelocityToOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[gnss_fix_velocity_to_odom] Ctrl-C, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("[gnss_fix_velocity_to_odom] done.")


if __name__ == "__main__":
    main()
