#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
import numpy as np

try:
    # pip install tf_transformations (ROS2에서 자주 쓰는 패키지)
    from tf_transformations import euler_from_quaternion, quaternion_from_euler
except ImportError:
    # 최소 fallback (roll,pitch 무시하고 yaw만 쓰고 싶을 때)
    def euler_from_quaternion(q):
        x, y, z, w = q
        # 표준 yaw 추출
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        # roll, pitch는 0으로 둔다
        return 0.0, 0.0, yaw

    def quaternion_from_euler(roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return (x, y, z, w)


class GlimPoseToUtmNode(Node):
    """
    /glim_ros/pose (world/map frame) -> /glim_ros/pose_utm (UTM-local frame)

    T_world_utm (UTM -> world) 를 파라미터로 받아서
    내부에서 inverse(T_world_utm) = T_utm_world 를 만들어 사용한다.
    """

    def __init__(self):
        super().__init__("glim_pose_to_utm")

        # ---- parameters ----
        self.declare_parameter("input_topic", "/glim_ros/pose")
        self.declare_parameter("output_topic", "/glim_ros/pose_utm")
        self.declare_parameter("utm_frame_id", "map")

        # T_world_utm = [tx, ty, tz]
        self.declare_parameter("T_world_utm_translation",[0.324901, 0.469413, 0.517126])
        # yaw (deg) for UTM->world
        self.declare_parameter("T_world_utm_yaw_deg", 134.2810)

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.utm_frame_id = self.get_parameter("utm_frame_id").get_parameter_value().string_value

        t_list = self.get_parameter("T_world_utm_translation").get_parameter_value().double_array_value
        yaw_deg = self.get_parameter("T_world_utm_yaw_deg").get_parameter_value().double_value

        if len(t_list) != 3:
            raise RuntimeError("T_world_utm_translation must have 3 elements [tx, ty, tz]")

        tx, ty, tz = t_list
        yaw_wu = math.radians(yaw_deg)  # yaw (rad), UTM->world

        # R_world_utm (yaw only, 3x3)
        R_wu = np.array([
            [math.cos(yaw_wu), -math.sin(yaw_wu), 0.0],
            [math.sin(yaw_wu),  math.cos(yaw_wu), 0.0],
            [0.0,              0.0,              1.0]
        ])
        t_wu = np.array([tx, ty, tz])

        # inverse: T_utm_world
        R_uw = R_wu.T                      # R_utm_world
        t_uw = - R_uw @ t_wu               # t_utm_world

        self.R_utm_world = R_uw
        self.t_utm_world = t_uw
        self.yaw_wu = yaw_wu  # UTM->world yaw


        self.utm_origin = np.array([284820.413,4151199.260,30.200 ])  # UTM raw 좌표계 오프셋

        self.get_logger().info(
            f"T_world_utm: t=[{tx:.3f}, {ty:.3f}, {tz:.3f}], yaw={yaw_deg:.3f} deg"
        )
        self.get_logger().info(
            f"Computed T_utm_world: R_utm_world^T, t={self.t_utm_world}"
        )

        # ---- sub/pub ----
        self.sub_pose = self.create_subscription(
            PoseStamped,
            input_topic,
            self.pose_callback,
            30
        )
        self.pub_pose = self.create_publisher(
            PoseStamped,
            output_topic,
            10
        )

        self.get_logger().info(
            f"glim_pose_to_utm started. input={input_topic}, output={output_topic}, utm_frame_id={self.utm_frame_id}"
        )

    def pose_callback(self, msg: PoseStamped):
        # world(map) -> UTM-local
        p_world = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        # p_utm = R_utm_world * p_world + t_utm_world
        p_utm = self.R_utm_world @ p_world + self.t_utm_world


        p_utm_raw = p_utm + self.utm_origin


        # orientation: yaw만 보정 (roll/pitch는 그대로 쓴다고 가정)
        q = msg.pose.orientation
        roll, pitch, yaw_world = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # world = R_wu * utm => utm = R_uw * world
        # yaw_world = yaw_wu + yaw_utm  ->  yaw_utm = yaw_world - yaw_wu
        yaw_utm = yaw_world - self.yaw_wu

        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw_utm)

        # 새 메시지 구성
        out = PoseStamped()
        out.header = msg.header
        out.header.frame_id = self.utm_frame_id  # UTM frame 이름

        # out.pose.position.x = float(p_utm[0])
        # out.pose.position.y = float(p_utm[1])
        # out.pose.position.z = float(p_utm[2])
        out.pose.position.x = float(p_utm_raw[0])
        out.pose.position.y = float(p_utm_raw[1])
        out.pose.position.z = float(p_utm_raw[2])

        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw

        self.pub_pose.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GlimPoseToUtmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
