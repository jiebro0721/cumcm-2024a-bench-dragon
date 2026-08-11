"""问题 2：板凳之间不发生碰撞的盘入终止时刻。

碰撞判别采用命题人讲评的矩形重叠模型（矩形任一边相交即碰撞）。
数值实验发现：随着龙头极角 theta0 减小，碰撞状态并非严格单调，
因此不能直接在整个区间上二分。算法分两步：
  1) 从初始极角 32*pi 向内逐步扫描，找到第一个出现碰撞的“窗口”；
  2) 在该窗口边界上二分，取“仍不碰撞”一侧作为终止极角 theta0(T)；
  3) 按 T = a/(4*pi*v0) [G(theta0(0)) - G(theta0(T))] 求时间。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchloong import THETA0, G, ChainError, any_collision, chain_spiral
from benchloong.io_utils import write_result2

PITCH = 0.55
V0 = 1.0
SELECT_POINTS = [0, 1, 51, 101, 151, 201, 223]
SCAN_STEP = 0.2          # 粗扫步长
FINE_STEP = 0.02         # 窗口附近细扫步长


def collision_at(theta0: float) -> bool:
    """龙头极角为 theta0 时，全队是否发生碰撞（或链条无法容纳）。"""
    try:
        _, pts, _ = chain_spiral(PITCH, theta0)
        return any_collision(pts)
    except ChainError:
        return True


def main() -> None:
    t0 = time.time()

    # 1) 从外侧向内侧扫描，找到第一个碰撞窗口（粗扫 + 细扫）
    def first_true_from(th_hi: float, th_lo: float, step: float) -> float | None:
        th = th_hi
        while th > th_lo:
            if collision_at(th):
                return th
            th -= step
        return None

    first_true = first_true_from(THETA0, 27.0, SCAN_STEP)
    if first_true is None:
        first_true = first_true_from(27.0, 25.0, FINE_STEP)
    if first_true is None:
        raise RuntimeError("未发现碰撞窗口")

    # 2) 在窗口边界二分：lo 碰撞、hi 不碰撞
    lo, hi = first_true, first_true + FINE_STEP
    assert collision_at(lo), "下界应碰撞"
    assert not collision_at(hi), "上界应不碰撞"
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if collision_at(mid):
            lo = mid
        else:
            hi = mid
    theta_T = hi                 # 仍不发生碰撞（首次碰撞前一刻）
    T = PITCH / (4.0 * np.pi * V0) * (G(THETA0) - G(theta_T))

    _, positions, speeds = chain_spiral(PITCH, theta_T, v0=V0)
    out = write_result2(positions[:, 0], positions[:, 1], speeds)

    print(f"终止极角 theta0(T) = {theta_T:.10f} rad")
    print(f"龙头半径           = {PITCH * theta_T / (2 * np.pi):.6f} m")
    print(f"终止时刻 T         = {T:.6f} s")
    print(f"首次碰撞窗口       = [{first_true:.4f}, {first_true + SCAN_STEP:.4f}] rad")
    print(f"二分区间           = [{lo:.10f}, {hi:.10f}]")
    print(f"已写入 {out}，耗时 {time.time() - t0:.1f} s")

    print("\n终止时刻指定把手位置与速度")
    for p in SELECT_POINTS:
        print(f"P{p:<5d} x={positions[p, 0]:.6f} y={positions[p, 1]:.6f} v={speeds[p]:.6f}")


if __name__ == "__main__":
    main()
