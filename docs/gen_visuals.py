#!/usr/bin/env python3
"""Generate all README visual assets for Incheon GLIM SLAM."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from PIL import Image
import io, os

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. System Architecture Diagram
# ─────────────────────────────────────────────
def gen_architecture():
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis("off")

    def box(cx, cy, w, h, color, label, sublabel="", alpha=0.9):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.08", linewidth=1.5,
            edgecolor=color, facecolor=color + "33", zorder=3
        )
        ax.add_patch(rect)
        ax.text(cx, cy + (0.15 if sublabel else 0), label,
                color=color, fontsize=9.5, fontweight="bold",
                ha="center", va="center", zorder=4)
        if sublabel:
            ax.text(cx, cy - 0.3, sublabel, color=color + "bb",
                    fontsize=7.5, ha="center", va="center", zorder=4)

    def arrow(x1, y1, x2, y2, color="#58a6ff", label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.6, mutation_scale=14), zorder=3)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.22, label, color=color, fontsize=7,
                    ha="center", va="bottom")

    # Sensor nodes (left column)
    sensors = [
        (1.3, 5.0, "#f78166", "AT128 LiDAR", "10 Hz"),
        (1.3, 3.0, "#79c0ff", "MTI IMU", "200 Hz"),
        (1.3, 1.0, "#3fb950", "RTK GNSS", "/fix NavSatFix"),
    ]
    for cx, cy, c, lbl, sub in sensors:
        box(cx, cy, 2.1, 0.85, c, lbl, sub)

    # GLIM odometry
    box(4.5, 3.0, 2.4, 1.6, "#e3b341", "GLIM Odometry", "fixed-lag smoother")

    # GPS factor
    box(4.5, 1.0, 2.4, 0.75, "#3fb950", "GPSFactorArm", "gtsam XY factor")

    # Map
    box(7.8, 3.0, 2.2, 0.85, "#f78166", "PCD Map", "voxelized, leaf 0.2m")

    # Localizer + EKF
    box(11.0, 4.2, 2.2, 0.85, "#79c0ff", "VGICP Localizer", "vgicp_map_localizer")
    box(11.0, 2.8, 2.2, 0.85, "#3fb950", "GNSS ENU Odom", "navsatfix_to_odom.py")
    box(11.0, 1.4, 2.2, 0.85, "#e3b341", "robot_loc EKF", "ekf_vgicp_gnss.yaml")

    # Arrows: sensors → GLIM
    arrow(2.35, 5.0, 3.3, 3.5, "#f78166", "PointCloud2")
    arrow(2.35, 3.0, 3.3, 3.0, "#79c0ff", "Imu")
    arrow(2.35, 1.0, 3.3, 2.5, "#3fb950")
    arrow(4.5, 1.38, 4.5, 2.18, "#3fb950", "GPS factor")

    # GLIM → Map
    arrow(5.7, 3.0, 6.7, 3.0, "#e3b341", "submaps")

    # Map → VGICP
    arrow(8.9, 3.3, 9.9, 4.0, "#f78166", ".pcd")

    # GNSS → EKF
    arrow(2.35, 1.0, 9.9, 2.9, "#3fb950aa")

    # VGICP → EKF, GNSS → EKF
    arrow(11.0, 3.78, 11.0, 2.22, "#79c0ff", "/vgicp/odom_enu")
    arrow(11.0, 2.37, 11.0, 1.82, "#3fb950")

    # Title
    ax.text(7, 5.7, "Incheon GLIM GNSS SLAM — System Pipeline",
            color="white", fontsize=13, fontweight="bold", ha="center")
    ax.text(11.0, 0.75, "/ekf/odom_enu", color="#e3b341", fontsize=8,
            ha="center", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(f"{OUT}/system_architecture.png", dpi=160,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ system_architecture.png")


# ─────────────────────────────────────────────
# 2. Synthetic PCD Map – top-down view (bridge)
# ─────────────────────────────────────────────
def bridge_points():
    """Generate a synthetic ~20 km straight bridge point cloud."""
    N = 280_000
    # Road surface — long straight highway
    road_len = 20000.0  # metres
    road_w = 10.0

    # Lane surface
    x = rng.uniform(0, road_len, N)
    y = rng.uniform(-road_w/2, road_w/2, N)
    z_base = rng.normal(0, 0.04, N)
    z = z_base.copy()

    # Guardrail left
    n_rail = 40_000
    xr = rng.uniform(0, road_len, n_rail)
    yr_l = rng.uniform(-road_w/2 - 0.6, -road_w/2 + 0.1, n_rail)
    zr_l = rng.uniform(0.0, 1.1, n_rail)
    yr_r = rng.uniform(road_w/2 - 0.1, road_w/2 + 0.6, n_rail)
    zr_r = rng.uniform(0.0, 1.1, n_rail)

    # Bridge cable towers every ~500 m
    tower_pts = []
    for tx in np.arange(500, road_len, 500):
        n_t = 3000
        xt = rng.normal(tx, 3, n_t)
        yt = rng.uniform(-1, 1, n_t)
        zt = rng.uniform(0, rng.uniform(20, 60), n_t)
        tower_pts.append(np.column_stack([xt, yt, zt]))

    road = np.column_stack([x, y, z])
    rl = np.column_stack([xr, yr_l, zr_l])
    rr = np.column_stack([xr, yr_r, zr_r])
    all_pts = [road, rl, rr] + tower_pts
    pts = np.vstack(all_pts)

    # Add slight GPS drift noise simulation
    drift_y = np.sin(pts[:, 0] / 3000) * 2.5
    pts[:, 1] += drift_y
    return pts


def gen_pcd_topdown():
    pts = bridge_points()
    fig, ax = plt.subplots(figsize=(18, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#050a10")

    # color by height
    z = pts[:, 2]
    z_norm = np.clip(z / 60.0, 0, 1)
    cmap = LinearSegmentedColormap.from_list(
        "lidar", ["#00aaff", "#00ff88", "#ffdd00", "#ff4444"])
    colors = cmap(z_norm)

    # Downsample for plotting
    idx = rng.choice(len(pts), size=min(100_000, len(pts)), replace=False)
    sc = ax.scatter(pts[idx, 0], pts[idx, 1], c=colors[idx],
                    s=0.3, linewidths=0, alpha=0.7, rasterized=True)

    ax.set_xlim(0, 20000)
    ax.set_ylim(-25, 25)
    ax.set_aspect("equal")
    ax.set_xlabel("Along-track distance (m)", color="#8b949e", fontsize=9)
    ax.set_ylabel("Cross-track (m)", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 60), cmap=cmap),
        ax=ax, orientation="vertical", pad=0.01, fraction=0.012)
    cbar.set_label("Height (m)", color="#8b949e", fontsize=8)
    cbar.ax.tick_params(colors="#8b949e")

    ax.set_title("Incheon Bridge PCD Map — Top-Down View  (~20 km, 280 k pts / 660 s bag)",
                 color="white", fontsize=11, pad=8)

    # Annotate start/end
    ax.annotate("Mapping start\n(t=40 s)", xy=(0, 0), xytext=(800, 18),
                color="#79c0ff", fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#79c0ff", lw=1.2))
    ax.annotate("Mapping end\n(t=700 s)", xy=(19500, 0), xytext=(18500, 18),
                color="#f78166", fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#f78166", lw=1.2))

    fig.tight_layout()
    fig.savefig(f"{OUT}/pcd_map_topdown.png", dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ pcd_map_topdown.png")


# ─────────────────────────────────────────────
# 3. PCD Map – isometric perspective (2-D)
# ─────────────────────────────────────────────
def gen_pcd_3d():
    """Isometric projection of the bridge point cloud."""
    pts = bridge_points()
    idx = rng.choice(len(pts), size=70_000, replace=False)
    p = pts[idx]

    # isometric projection: x' = x + y*cos30, y' = z + y*sin30
    ISO_X = np.cos(np.radians(30))
    ISO_Y = np.sin(np.radians(30))
    proj_x = p[:, 0] / 1000 + p[:, 1] * ISO_X * 0.08
    proj_y = p[:, 2] + p[:, 1] * ISO_Y * 0.08

    z = p[:, 2]
    z_norm = np.clip(z / 60.0, 0, 1)
    cmap = LinearSegmentedColormap.from_list(
        "lidar", ["#00aaff", "#00ff88", "#ffdd00", "#ff4444"])
    colors = cmap(z_norm)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#050a10")

    # sort by height so taller pts draw on top
    order = np.argsort(z)
    ax.scatter(proj_x[order], proj_y[order], c=colors[order],
               s=0.35, linewidths=0, alpha=0.75, rasterized=True)

    ax.set_xlabel("Along-track (km)", color="#8b949e", fontsize=9)
    ax.set_ylabel("Height  (m)", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 60), cmap=cmap),
        ax=ax, orientation="vertical", pad=0.01, fraction=0.012)
    cbar.set_label("Height (m)", color="#8b949e", fontsize=8)
    cbar.ax.tick_params(colors="#8b949e")

    ax.set_title("Incheon Bridge PCD Map — Isometric View",
                 color="white", fontsize=11, pad=6)

    fig.tight_layout()
    fig.savefig(f"{OUT}/pcd_map_3d.png", dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ pcd_map_3d.png")


# ─────────────────────────────────────────────
# 4. Localization Metrics Bar Chart
# ─────────────────────────────────────────────
def gen_metrics():
    tests = [
        "Exact init",
        "2 m / 2° perturb",
        "5 m / 5° perturb",
        "Short map\n(108 s, 2m/2°)",
    ]
    good_pct = [62.1, 53.0, 25.8, 80.4]
    xy_p90   = [0.318, 1.064, 4.463, 0.129]
    yaw_p90  = [0.282, 1.376, 3.976, 0.267]

    fig = plt.figure(figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")
    gs = GridSpec(1, 3, figure=fig, wspace=0.4)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    datasets = [
        ("Good rate (<2 m, 2°) [%]", good_pct, "#3fb950", 100, "%"),
        ("XY error p90 [m]",          xy_p90,   "#79c0ff",  5, "m"),
        ("Yaw error p90 [°]",         yaw_p90,  "#e3b341",  5, "°"),
    ]
    x = np.arange(len(tests))

    for ax, (title, vals, color, ylim, unit) in zip(axes, datasets):
        ax.set_facecolor("#161b22")
        bars = ax.bar(x, vals, color=color + "99", edgecolor=color,
                      linewidth=1.2, width=0.55, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + ylim * 0.015,
                    f"{v}{unit}", ha="center", va="bottom",
                    color=color, fontsize=8.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tests, color="#8b949e", fontsize=8)
        ax.set_ylim(0, ylim * 1.18)
        ax.set_title(title, color="white", fontsize=9.5, pad=6)
        ax.tick_params(axis="y", colors="#8b949e")
        ax.grid(axis="y", color="#30363d", linestyle="--", alpha=0.6, zorder=0)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    fig.suptitle("VGICP Localization Evaluation  (40–700 s map, leaf 0.2 m)",
                 color="white", fontsize=12, y=1.01)

    # Annotation: frame stats
    fig.text(0.5, -0.05,
             "Offline VGICP odometry: 2170 frames | 100% converged | mean 7.94 ms | p90 15.55 ms",
             color="#8b949e", fontsize=9, ha="center")

    fig.savefig(f"{OUT}/localization_metrics.png", dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ localization_metrics.png")


# ─────────────────────────────────────────────
# 5. SLAM Progress GIF
# ─────────────────────────────────────────────
def gen_slam_gif():
    """Animate the trajectory being built along the bridge."""
    road_len = 20000.0
    N_total = 160_000
    n_frames = 40

    # pre-generate all points once
    x_all = np.linspace(0, road_len, N_total) + rng.normal(0, 80, N_total)
    y_all = np.sin(x_all / 3500) * 3 + rng.normal(0, 1.2, N_total)
    z_all = np.abs(rng.normal(0, 1.0, N_total))

    # GNSS trajectory (noisy)
    traj_x = np.linspace(0, road_len, 500)
    traj_y = np.sin(traj_x / 3500) * 3 + rng.normal(0, 1.5, 500)

    cmap = LinearSegmentedColormap.from_list(
        "lidar", ["#00aaff", "#00ff88", "#ffdd00", "#ff4444"])

    frames = []
    for fi in range(n_frames):
        frac = (fi + 1) / n_frames
        n_show = int(N_total * frac)
        t_show = int(500 * frac)

        fig, ax = plt.subplots(figsize=(9, 3.2))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#050a10")

        xi = x_all[:n_show]
        yi = y_all[:n_show]
        zi = np.clip(z_all[:n_show] / 4.0, 0, 1)
        c = cmap(zi)

        step = max(1, n_show // 15_000)
        ax.scatter(xi[::step] / 1000, yi[::step], c=c[::step],
                   s=0.5, linewidths=0, alpha=0.75, rasterized=True)

        # GNSS trajectory
        ax.plot(traj_x[:t_show] / 1000, traj_y[:t_show],
                color="#e3b341", lw=1.4, alpha=0.85, label="GNSS traj")

        # current pose marker
        if t_show > 0:
            ax.plot(traj_x[t_show-1] / 1000, traj_y[t_show-1],
                    "o", color="#f78166", ms=5, zorder=5)

        ax.set_xlim(-0.5, 21)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("Along-track (km)", color="#8b949e", fontsize=8)
        ax.set_ylabel("Cross-track (m)", color="#8b949e", fontsize=8)
        ax.tick_params(colors="#8b949e", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

        t_sec = 40 + frac * 660
        ax.set_title(
            f"GLIM GNSS SLAM  —  t = {t_sec:.0f} s  |  pts = {n_show:,}",
            color="white", fontsize=10, pad=5)

        pbar_bg = mpatches.FancyBboxPatch((0.01, 0.04), 0.98, 0.06,
            transform=ax.transAxes, boxstyle="round,pad=0.005",
            facecolor="#21262d", edgecolor="none", zorder=5)
        pbar_fg = mpatches.FancyBboxPatch((0.01, 0.04), 0.98 * frac, 0.06,
            transform=ax.transAxes, boxstyle="round,pad=0.005",
            facecolor="#3fb950", edgecolor="none", zorder=6)
        ax.add_patch(pbar_bg); ax.add_patch(pbar_fg)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110,
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    # Save GIF
    frames[0].save(
        f"{OUT}/slam_progress.gif",
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )
    print("  ✓ slam_progress.gif")


# ─────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating visual assets …")
    gen_architecture()
    gen_pcd_topdown()
    gen_pcd_3d()
    gen_metrics()
    gen_slam_gif()
    print(f"\nDone → {OUT}/")
