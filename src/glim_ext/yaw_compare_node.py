#!/usr/bin/env python3
import math
from typing import Optional, List

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
import numpy as np
import matplotlib.pyplot as plt


def quat_to_yaw(q) -> float:
    """geometry_msgs Quaternion -> yaw (rad), Z축 기준."""
    x = q.x
    y = q.y
    z = q.z
    w = q.w
    # 표준 yaw 추출 공식
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


def wrap_to_pi(angle: float) -> float:
    """[-pi, pi]로 wrap."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class YawCompareNode(Node):
    """
    /gnss_utm_pose 의 orientation.yaw (IMU ENU yaw)
    /velocity 의 linear.x/y (ENU 속도)로부터 heading 계산해서
    두 값과 그 차이를 플롯하는 디버그 노드.
    """

    def __init__(self):
        super().__init__("yaw_compare_node")

        # 파라미터
        self.declare_parameter("gnss_pose_topic", "/gnss_utm_pose")
        self.declare_parameter("vel_topic", "/vel")

        gnss_pose_topic = self.get_parameter("gnss_pose_topic").get_parameter_value().string_value
        vel_topic = self.get_parameter("vel_topic").get_parameter_value().string_value

        # 로그 버퍼
        self.t_pose: List[float] = []
        self.yaw_imu: List[float] = []

        self.t_vel: List[float] = []
        self.heading_gnss: List[float] = []

        # 서브스크립션
        self.sub_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            gnss_pose_topic,
            self.pose_callback,
            50
        )
        self.sub_vel = self.create_subscription(
            TwistStamped,
            vel_topic,
            self.vel_callback,
            50
        )

        self.get_logger().info(
            f"YawCompareNode started. gnss_pose_topic={gnss_pose_topic}, vel_topic={vel_topic}"
        )

    # ----------------- Callbacks ----------------- #

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        # 시간 [s]
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        yaw = quat_to_yaw(msg.pose.pose.orientation)  # rad
        self.t_pose.append(t)
        self.yaw_imu.append(yaw)

    def vel_callback(self, msg: TwistStamped):
        # 시간 [s]
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # ENU 기준이라고 가정: linear.x ~ East, linear.y ~ North
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y

        # 속도 너무 작으면 heading 쓸모 없음 -> 그냥 무시
        speed = math.hypot(vx, vy)
        if speed < 0.5:  # m/s, 필요하면 파라미터로
            return

        heading = math.atan2(vy, vx)  # rad, ENU에서 East=0, CCW로 North
        self.t_vel.append(t)
        self.heading_gnss.append(heading)

    # ----------------- Plot ----------------- #

    def make_plot(self):
        if len(self.t_pose) < 10 or len(self.t_vel) < 10:
            self.get_logger().warn("데이터가 너무 적어서 플롯을 그리기 어렵습니다.")
            return

        t_pose = np.array(self.t_pose)
        yaw_imu = np.unwrap(np.array(self.yaw_imu))  # 언랩해서 부드럽게

        t_vel = np.array(self.t_vel)
        heading_gnss = np.unwrap(np.array(self.heading_gnss))

        # 공통 시간 구간 설정
        t0 = max(t_pose.min(), t_vel.min())
        t1 = min(t_pose.max(), t_vel.max())

        if t1 <= t0:
            self.get_logger().warn("Pose와 velocity 시간 구간이 겹치지 않습니다.")
            return

        # 비교를 위해 그리드 생성 (예: 2000 포인트)
        num_points = 2000
        t_grid = np.linspace(t0, t1, num_points)

        # 선형 보간으로 각각 yaw/heading 맞춰주기
        yaw_imu_interp = np.interp(t_grid, t_pose, yaw_imu)
        heading_interp = np.interp(t_grid, t_vel, heading_gnss)

        # 차이 (wrap_to_pi)
        yaw_diff = np.array([wrap_to_pi(a - b) for a, b in zip(yaw_imu_interp, heading_interp)])

        # rad -> deg
        yaw_imu_deg = np.degrees(yaw_imu_interp)
        heading_deg = np.degrees(heading_interp)
        yaw_diff_deg = np.degrees(yaw_diff)

        # 플롯
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

        ax1.plot(t_grid - t_grid[0], yaw_imu_deg, label="IMU yaw (deg)")
        ax1.plot(t_grid - t_grid[0], heading_deg, label="GNSS heading (deg)", linestyle="--")
        ax1.set_ylabel("Yaw / Heading [deg]")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(t_grid - t_grid[0], yaw_diff_deg, label="yaw_imu - heading_gnss")
        ax2.set_ylabel("Diff [deg]")
        ax2.set_xlabel("Time [s] (relative)")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # 필요하면 CSV로 저장하고 싶을 때:
        # import pandas as pd
        # df = pd.DataFrame({
        #   "t": t_grid,
        #   "yaw_imu_deg": yaw_imu_deg,
        #   "heading_gnss_deg": heading_deg,
        #   "diff_deg": yaw_diff_deg,
        # })
        # df.to_csv("yaw_compare.csv", index=False)


def main(args=None):
    rclpy.init(args=args)
    node = YawCompareNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl-C detected, plotting...")
    finally:
        # spin 종료 후 플롯 그리기
        node.make_plot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
