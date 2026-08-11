"""问题 3：最小螺距，使龙头能沿螺线盘入调头空间（半径 4.5 m）边界。

可行性判据（与题目“盘入过程中整个板凳龙不能发生碰撞”一致）：
对候选螺距 a，龙头从初始极角 32*pi 盘入到 theta1 = 2*pi*r/a 的
全过程内，任意 theta0 ∈ [theta1, 32*pi] 都不发生矩形重叠碰撞，
且链条处处可容纳。螺距越小越易碰撞，故对 a 二分。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchloong import THETA0, ChainError, any_collision, chain_spiral

R = 4.5                 # 调头空间半径（直径 9 m）
SCAN_STEP = 0.5           # 外层粗扫步长
FINE_STEP = 0.02          # 靠近边界（theta1 附近）的细扫步长


def feasible(a: float) -> bool:
    theta1 = 2.0 * np.pi * R / a
    if theta1 >= THETA0:
        return False
    # 从初始位置向边界扫描整个盘入过程：外层粗扫 + 近边界细扫
    for th in _scan_down(THETA0, theta1 + 2.0, SCAN_STEP):
        try:
            _, pts, _ = chain_spiral(a, th)
            if any_collision(pts):
                return False
        except ChainError:
            return False
    for th in _scan_down(theta1 + 2.0, theta1, FINE_STEP):
        try:
            _, pts, _ = chain_spiral(a, th)
            if any_collision(pts):
                return False
        except ChainError:
            return False
    return True


def _scan_down(hi: float, lo: float, step: float):
    th = hi
    while th > lo:
        yield th
        th -= step
        if th < lo:
            th = lo


def main() -> None:
    t0 = time.time()
    lo, hi = 0.40, 0.50         # 粗查：0.40 不可行、0.50 可行
    assert not feasible(lo), "下界应不可行"
    assert feasible(hi), "上界应可行"
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    a_min = hi
    theta1 = 2.0 * np.pi * R / a_min
    print(f"最小螺距 a_min = {a_min:.8f} m")
    print(f"对应 theta1    = {theta1:.6f} rad")
    print(f"二分区间       = [{lo:.8f}, {hi:.8f}]")
    print(f"耗时 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
