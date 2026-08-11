"""问题 1：0~300 s 每秒全队 224 个把手的位置与速度。

模型：
  1) 龙头前把手沿盘入螺线 rho = a*theta/(2*pi) 以 1 m/s 匀速运动，
     由弧长-极角关系 G(theta(t)) = G(theta(0)) - 4*pi*v0*t/a 求 theta(t)；
  2) 龙身/龙尾把手用“弦长递推”确定（与讲评式 (40) 等价）；
  3) 速度由刚体约束递推（与讲评式 (7)-(9) 等价）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchloong import THETA0, N_HANDLES, chain_spiral
from benchloong.io_utils import write_result1

PITCH = 0.55          # 螺距 a = 55 cm
V0 = 1.0              # 龙头前把手速度
SELECT_TIMES = [0, 60, 120, 180, 240, 300]
SELECT_POINTS = [0, 1, 51, 101, 151, 201, 223]


def main() -> None:
    t0 = time.time()
    times = np.arange(0, 301, dtype=float)
    positions = np.empty((len(times), N_HANDLES, 2))
    speeds = np.empty((len(times), N_HANDLES))

    for k, t in enumerate(times):
        _, pts, spd = chain_spiral(PITCH, theta_of_time(t), v0=V0)
        positions[k] = pts
        speeds[k] = spd

    out = write_result1(positions, speeds)
    print(f"已写入 {out}")
    print(f"总耗时 {time.time() - t0:.1f} s")

    # 论文表格（0, 60, 120, 180, 240, 300 s；指定把手）
    print("\n位置表 (m)")
    header = "点\\时刻" + "".join(f"{t:>22d} s" for t in SELECT_TIMES)
    print(header)
    for p in SELECT_POINTS:
        row = f"P{p:<5d}"
        for k, t in enumerate(times):
            if t in SELECT_TIMES:
                row += f"({positions[k, p, 0]:.6f},{positions[k, p, 1]:.6f})"
        print(row)
    print("\n速度表 (m/s)")
    for p in SELECT_POINTS:
        row = f"P{p:<5d}"
        for k, t in enumerate(times):
            if t in SELECT_TIMES:
                row += f"{speeds[k, p]:>22.6f}"
        print(row)


def theta_of_time(t: float) -> float:
    """由 G(theta(t)) = G(32*pi) - 4*pi*V0*t/a 求龙头极角。"""
    from benchloong.geometry import G, G_inv

    return G_inv(G(THETA0) - 4.0 * np.pi * V0 * t / PITCH)


if __name__ == "__main__":
    main()
