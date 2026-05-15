#!/usr/bin/env python3
import math
from typing import List

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

import numpy as np
import matplotlib.pyplot as plt



def quat_to_yaw(q) -> float:
    """geometry_msgs Quaternion -> yaw (rad), Z축 기준."""
    x = q.x
    y = q.y
    z = q.z
    w = q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


def wrap_to_pi(angle: float) -> float:
    """[-pi, pi]로 wrap."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PoseTimeAndDiffPlotNode(Node):
    """
    /gnss_utm_pose (PoseWithCovarianceStamped)
    /glim_ros/pose_utm (PoseStamped)

    왼쪽:  x,y,z,yaw vs time (GNSS vs GLIM)
    오른쪽: 각 축의 diff vs time (GLIM - GNSS)
    """

    def __init__(self):
        super().__init__("pose_time_and_diff_plot_node")

        # ---------- parameters ----------
        self.declare_parameter("gnss_pose_topic", "/gnss_utm_pose")
        self.declare_parameter("glim_pose_topic", "/glim_ros/pose")

        gnss_pose_topic = self.get_parameter("gnss_pose_topic").get_parameter_value().string_value
        glim_pose_topic = self.get_parameter("glim_pose_topic").get_parameter_value().string_value

        # ---------- data buffers ----------
        self.t_gnss: List[float] = []
        self.x_gnss: List[float] = []
        self.y_gnss: List[float] = []
        self.z_gnss: List[float] = []
        self.yaw_gnss: List[float] = []

        self.t_glim: List[float] = []
        self.x_glim: List[float] = []
        self.y_glim: List[float] = []
        self.z_glim: List[float] = []
        self.yaw_glim: List[float] = []

        # ---------- subscriptions ----------
        self.sub_gnss = self.create_subscription(
            PoseWithCovarianceStamped,
            gnss_pose_topic,
            self.gnss_pose_callback,
            50
        )
        self.sub_glim = self.create_subscription(
            PoseStamped,
            glim_pose_topic,
            self.glim_pose_callback,
            50
        )

        self.get_logger().info(
            f"PoseTimeAndDiffPlotNode started.\n"
            f"  gnss_pose_topic={gnss_pose_topic}\n"
            f"  glim_pose_topic={glim_pose_topic}"
        )

    # ---------- callbacks ----------

    def gnss_pose_callback(self, msg: PoseWithCovarianceStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.t_gnss.append(t)
        self.x_gnss.append(p.x)
        self.y_gnss.append(p.y)
        self.z_gnss.append(p.z)
        self.yaw_gnss.append(quat_to_yaw(q))

    def glim_pose_callback(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.position
        q = msg.pose.orientation

        self.t_glim.append(t)
        self.x_glim.append(p.x)
        self.y_glim.append(p.y)
        self.z_glim.append(p.z)
        self.yaw_glim.append(quat_to_yaw(q))

    # ---------- plotting ----------

    def make_plot(self):
        if len(self.t_gnss) < 10 or len(self.t_glim) < 10:
            self.get_logger().warn(
                f"데이터 부족: gnss={len(self.t_gnss)}개, glim={len(self.t_glim)}개"
            )
            return

        # 배열로 변환
        t_gnss = np.array(self.t_gnss)
        x_gnss = np.array(self.x_gnss)
        y_gnss = np.array(self.y_gnss)
        z_gnss = np.array(self.z_gnss)
        yaw_gnss = np.unwrap(np.array(self.yaw_gnss))

        t_glim = np.array(self.t_glim)
        x_glim = np.array(self.x_glim)
        y_glim = np.array(self.y_glim)
        z_glim = np.array(self.z_glim)
        yaw_glim = np.unwrap(np.array(self.yaw_glim))

        # 공통 시간 구간
        t0 = max(t_gnss.min(), t_glim.min())
        t1 = min(t_gnss.max(), t_glim.max())
        if t1 <= t0:
            self.get_logger().warn("gnss와 glim 시간 구간이 겹치지 않습니다.")
            return

        num_points = 2000
        t_grid = np.linspace(t0, t1, num_points)
        t_rel = t_grid - t_grid[0]  # 0부터 시작하는 상대 시간

        # GNSS 보간
        x_gnss_i = np.interp(t_grid, t_gnss, x_gnss)
        y_gnss_i = np.interp(t_grid, t_gnss, y_gnss)
        z_gnss_i = np.interp(t_grid, t_gnss, z_gnss)
        yaw_gnss_i = np.interp(t_grid, t_gnss, yaw_gnss)

        # GLIM 보간
        x_glim_i = np.interp(t_grid, t_glim, x_glim)
        y_glim_i = np.interp(t_grid, t_glim, y_glim)
        z_glim_i = np.interp(t_grid, t_glim, z_glim)
        yaw_glim_i = np.interp(t_grid, t_glim, yaw_glim)

        # diff 계산 (GLIM - GNSS)
        dx = x_glim_i - x_gnss_i
        dy = y_glim_i - y_gnss_i
        dz = z_glim_i - z_gnss_i
        yaw_diff = np.array([wrap_to_pi(a - b) for a, b in zip(yaw_glim_i, yaw_gnss_i)])

        # deg 변환
        yaw_gnss_deg = np.degrees(yaw_gnss_i)
        yaw_glim_deg = np.degrees(yaw_glim_i)
        yaw_diff_deg = np.degrees(yaw_diff)

        # ---------- summary 찍기 ----------
        self.get_logger().info(
            "=== Pose / Yaw Diff Summary ===\n"
            f"mean(dx)={np.mean(dx):.3f} m, mean(dy)={np.mean(dy):.3f} m, mean(dz)={np.mean(dz):.3f} m\n"
            f"mean|dx|={np.mean(np.abs(dx)):.3f} m, mean|dy|={np.mean(np.abs(dy)):.3f} m, mean|dz|={np.mean(np.abs(dz)):.3f} m\n"
            f"mean|yaw_diff|={np.mean(np.abs(yaw_diff_deg)):.3f} deg, max|yaw_diff|={np.max(np.abs(yaw_diff_deg)):.3f} deg"
        )

        # npz로 궤적 저장
        np.savez(
            "traj_gnss_glim.npz",
            t=t_grid,
            x_gnss=x_gnss_i, y_gnss=y_gnss_i, z_gnss=z_gnss_i,
            x_glim=x_glim_i, y_glim=y_glim_i, z_glim=z_glim_i,
        )
        self.get_logger().info("Trajectory data saved to traj_gnss_glim.npz")

        # ---------- 8개 subplot (4행 × 2열) ----------
        fig, axes = plt.subplots(4, 2, sharex=True, figsize=(14, 8))

        ax_x_val, ax_x_diff = axes[0]
        ax_y_val, ax_y_diff = axes[1]
        ax_z_val, ax_z_diff = axes[2]
        ax_yaw_val, ax_yaw_diff = axes[3]

        # 1. X 값
        ax_x_val.plot(t_rel, x_gnss_i, label="GNSS x", alpha=0.8)
        ax_x_val.plot(t_rel, x_glim_i, label="GLIM x", linestyle="--", alpha=0.8)
        ax_x_val.set_ylabel("x [m]")
        ax_x_val.grid(True)
        ax_x_val.legend()

        # 1'. X diff
        ax_x_diff.plot(t_rel, dx, label="dx = GLIM - GNSS")
        ax_x_diff.set_ylabel("dx [m]")
        ax_x_diff.grid(True)
        ax_x_diff.legend()

        # 2. Y 값
        ax_y_val.plot(t_rel, y_gnss_i, label="GNSS y", alpha=0.8)
        ax_y_val.plot(t_rel, y_glim_i, label="GLIM y", linestyle="--", alpha=0.8)
        ax_y_val.set_ylabel("y [m]")
        ax_y_val.grid(True)
        ax_y_val.legend()

        # 2'. Y diff
        ax_y_diff.plot(t_rel, dy, label="dy = GLIM - GNSS")
        ax_y_diff.set_ylabel("dy [m]")
        ax_y_diff.grid(True)
        ax_y_diff.legend()

        # 3. Z 값
        ax_z_val.plot(t_rel, z_gnss_i, label="GNSS z", alpha=0.8)
        ax_z_val.plot(t_rel, z_glim_i, label="GLIM z", linestyle="--", alpha=0.8)
        ax_z_val.set_ylabel("z [m]")
        ax_z_val.grid(True)
        ax_z_val.legend()

        # 3'. Z diff
        ax_z_diff.plot(t_rel, dz, label="dz = GLIM - GNSS")
        ax_z_diff.set_ylabel("dz [m]")
        ax_z_diff.grid(True)
        ax_z_diff.legend()

        # 4. Yaw 값
        ax_yaw_val.plot(t_rel, yaw_gnss_deg, label="GNSS yaw [deg]", alpha=0.8)
        ax_yaw_val.plot(t_rel, yaw_glim_deg, label="GLIM yaw [deg]", linestyle="--", alpha=0.8)
        ax_yaw_val.set_ylabel("yaw [deg]")
        ax_yaw_val.grid(True)
        ax_yaw_val.legend()

        # 4'. Yaw diff
        ax_yaw_diff.plot(t_rel, yaw_diff_deg, label="yaw diff [deg] (GLIM-GNSS)")
        ax_yaw_diff.set_ylabel("yaw diff [deg]")
        ax_yaw_diff.set_xlabel("time [s] (relative)")
        ax_yaw_diff.grid(True)
        ax_yaw_diff.legend()

        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = PoseTimeAndDiffPlotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl-C detected, plotting...")
        node.make_plot()


    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
