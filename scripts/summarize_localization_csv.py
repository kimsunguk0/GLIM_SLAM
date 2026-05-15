#!/usr/bin/env python3
import csv
import math
import statistics
import sys


def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, int(math.floor(p * (len(values) - 1))))
    return values[idx]


def main():
    if len(sys.argv) != 2:
        print("usage: summarize_localization_csv.py localize_map_eval.csv", file=sys.stderr)
        return 2

    with open(sys.argv[1], newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("no rows")
        return 1

    xy = [float(r["final_xy_error"]) for r in rows]
    yaw = [float(r["final_yaw_error_deg"]) for r in rows]
    fitness = [float(r["fitness"]) for r in rows]
    converged = [int(float(r["converged"])) for r in rows]
    good_2m = [
        c and float(r["final_xy_error"]) < 2.0 and float(r["final_yaw_error_deg"]) < 2.0
        for c, r in zip(converged, rows)
    ]
    good_5m = [
        c and float(r["final_xy_error"]) < 5.0 and float(r["final_yaw_error_deg"]) < 5.0
        for c, r in zip(converged, rows)
    ]

    n = len(rows)
    print(f"CSV: {sys.argv[1]}")
    print(f"scans={n}")
    print(f"converged={sum(converged)}/{n} ({sum(converged) / n * 100.0:.1f}%)")
    print(f"good_2m_2deg={sum(good_2m)}/{n} ({sum(good_2m) / n * 100.0:.1f}%)")
    print(f"good_5m_5deg={sum(good_5m)}/{n} ({sum(good_5m) / n * 100.0:.1f}%)")
    print(f"xy_mean={statistics.mean(xy):.3f} xy_p50={percentile(xy, 0.50):.3f} xy_p90={percentile(xy, 0.90):.3f} xy_max={max(xy):.3f}")
    print(f"yaw_mean_deg={statistics.mean(yaw):.3f} yaw_p90_deg={percentile(yaw, 0.90):.3f}")
    print(f"fitness_mean={statistics.mean(fitness):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
