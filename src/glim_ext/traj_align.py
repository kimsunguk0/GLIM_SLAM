#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    data = np.load("traj_gnss_glim.npz")
    t      = data["t"]
    x_gnss = data["x_gnss"]
    y_gnss = data["y_gnss"]
    z_gnss = data["z_gnss"]
    x_glim = data["x_glim"]
    y_glim = data["y_glim"]
    z_glim = data["z_glim"]

    # 2D만 먼저 보자 (xy)
    Pg = np.stack([x_gnss, y_gnss], axis=1)  # GNSS   (N x 2)
    Pe = np.stack([x_glim, y_glim], axis=1)  # GLIM   (N x 2)

    print("traj_gnss_glim.npz loaded: N =", Pg.shape[0])

    # ----- 0) 기존 차이 (정렬 전) -----
    diff0 = Pg - Pe
    dist0 = np.linalg.norm(diff0, axis=1)
    print("=== BEFORE rigid alignment ===")
    print("dist_xy: mean = %.3f m, max = %.3f m" % (dist0.mean(), dist0.max()))

    # ----- 1) 2D Procrustes (rigid) alignment -----
    # Pg ≈ R * Pe + t  를 만족하는 R(2x2), t(2x1) 찾기

    # 평균 빼기
    cg = Pg.mean(axis=0)  # GNSS centroid
    ce = Pe.mean(axis=0)  # GLIM centroid
    Pg_c = Pg - cg
    Pe_c = Pe - ce

    # 공분산 H = Pe_c^T Pg_c
    H = Pe_c.T @ Pg_c  # 2x2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T      # 2x2
    if np.linalg.det(R) < 0:
        # 반사 방지
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t_vec = cg - R @ ce  # 2x1

    # yaw (rad, deg)
    yaw = math.atan2(R[1, 0], R[0, 0])
    yaw_deg = math.degrees(yaw)

    print("\n=== Best-fit rigid transform (GLIM -> GNSS) ===")
    print("R =\n", R)
    print("t = ", t_vec)
    print("yaw (GLIM frame -> GNSS frame) = %.3f deg" % yaw_deg)

    # ----- 2) GLIM 궤적을 정렬해서 residual 확인 -----
    Pe_aligned = (R @ Pe.T).T + t_vec    # (N x 2)
    diff = Pg - Pe_aligned
    dist = np.linalg.norm(diff, axis=1)

    print("\n=== AFTER rigid alignment ===")
    print("dist_xy: mean = %.3f m, max = %.3f m" % (dist.mean(), dist.max()))

    # ----- 3) 간단한 플롯 -----
    # 원래 궤적 vs 정렬된 GLIM 궤적 vs GNSS
    plt.figure(figsize=(8, 8))
    plt.plot(Pg[:,0], Pg[:,1], label="GNSS", alpha=0.8)
    plt.plot(Pe[:,0], Pe[:,1], label="GLIM (orig)", alpha=0.4, linestyle="--")
    plt.plot(Pe_aligned[:,0], Pe_aligned[:,1], label="GLIM aligned", alpha=0.8, linestyle="-.")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("XY trajectory: GNSS vs GLIM (orig & aligned)")
    plt.show()

    # 잔차 vs 시간
    plt.figure(figsize=(10,4))
    t_rel = t - t[0]
    plt.plot(t_rel, dist0, label="before align")
    plt.plot(t_rel, dist, label="after align")
    plt.xlabel("time [s]")
    plt.ylabel("XY error [m]")
    plt.grid(True)
    plt.legend()
    plt.title("XY error over time")
    plt.show()

if __name__ == "__main__":
    main()
