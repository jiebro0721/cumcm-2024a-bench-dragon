"""问题 4：调头路径（两段相切圆弧）长度不变性 + -100~100 s 全队位置速度。

调头曲线模型与讲评第 5、6 节一致：
  - 盘入螺线 rho=a*theta/2pi，盘出螺线 rho=a*(theta+pi)/2pi（中心对称）；
  - 两段圆弧相切，r1+r2 = r*sqrt(1+theta1^2)/theta1，与半径比 k 无关；
  - 两段圆弧所张圆心角相等，故调头曲线总长 L = (r1+r2)*alpha 与 k 无关。
位置：以调头开始为 t=0，龙头弧长参数 s(t)=v0*t，链条用弦长递推；
速度：刚体约束递推（与讲评 (7)-(9) 同源）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchloong import N_HANDLES, chain_arc_params, chain_speeds
from benchloong.geometry import TurnPath
from benchloong.io_utils import write_result1

PITCH = 1.7
R = 4.5
K = 2.0
V0 = 1.0
SELECT_TIMES = [-100, -50, 0, 50, 100]
SELECT_POINTS = [0, 1, 51, 101, 151, 201, 223]


def turning_length_vs_k() -> None:
    """演示：保持相切，改变半径比 k，调头曲线长度不变。"""
    print("半径比 k 与调头曲线长度 L：")
    for k in [1.0, 2.0, 3.0, 5.0, 10.0]:
        tp = TurnPath(PITCH, R, k)
        print(f"  k = {k:5.1f}  r1={tp.r1:.6f} r2={tp.r2:.6f} "
              f"alpha={tp.alpha:.8f}  L={tp.L:.6f} m")


def main() -> None:
    t0 = time.time()
    turning_length_vs_k()
    path = TurnPath(PITCH, R, K)
    times = np.arange(-100, 101, dtype=float)
    positions = np.empty((len(times), N_HANDLES, 2))
    speeds = np.empty((len(times), N_HANDLES))
    for k, t in enumerate(times):
        s = chain_arc_params(path, V0 * t)
        positions[k] = np.array([path.point(x) for x in s])
        speeds[k] = chain_speeds(path, s)

    out = write_result1(positions, speeds,
                        times=[f"{int(t)} s" for t in times],
                        out_name="result4.xlsx")
    print(f"已写入 {out}，耗时 {time.time() - t0:.1f} s")

    print("\n指定时刻位置 (m)：")
    for p in SELECT_POINTS:
        row = f"P{p:<5d}"
        for t in SELECT_TIMES:
            k = int(np.where(times == t)[0][0])
            row += f" t={t:>4d}:({positions[k, p, 0]:.6f},{positions[k, p, 1]:.6f})"
        print(row)
    print("\n指定时刻速度 (m/s)：")
    for p in SELECT_POINTS:
        row = f"P{p:<5d}"
        for t in SELECT_TIMES:
            k = int(np.where(times == t)[0][0])
            row += f" t={t:>4d}:{speeds[k, p]:.6f}"
        print(row)


if __name__ == "__main__":
    main()
