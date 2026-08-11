"""问题 5：龙头最大行进速度，使全队各把手速度不超过 2 m/s。

方法（与讲评第 7 节一致）：先取龙头速度 1 m/s，计算整个行进过程
中全队最大速度 vmax0；由于各把手速度与龙头速度成正比，
龙头最大速度 vmax = 2 / vmax0。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchloong import ChainError, chain_arc_params, chain_speeds
from benchloong.geometry import TurnPath

PITCH = 1.7
R = 4.5
K = 2.0


def max_speed_at(path: TurnPath, t: float) -> tuple[float, int, float]:
    s = chain_arc_params(path, float(t))
    spd = chain_speeds(path, s)
    i = int(np.argmax(spd))
    return float(spd[i]), i, float(s[i])


def main() -> None:
    t0 = time.time()
    path = TurnPath(PITCH, R, K)
    best = (0.0, 0.0, 0, 0.0)

    # 粗扫全路径（含龙尾进入调头区的时刻），步长 0.5 s
    for t in np.arange(-100.0, 400.0, 0.5):
        try:
            m, i, si = max_speed_at(path, float(t))
            if m > best[0]:
                best = (m, float(t), i, si)
        except ChainError:
            pass

    # 在峰值附近细扫（峰值很窄）
    tc = best[1]
    for dt, width in [(0.02, 1.0), (0.002, 0.2)]:
        for t in np.arange(tc - width, tc + width, dt):
            try:
                m, i, si = max_speed_at(path, float(t))
                if m > best[0]:
                    best = (m, float(t), i, si)
            except ChainError:
                pass

    vmax0, t_star, i_star, s_star = best
    vmax = 2.0 / vmax0
    print(f"龙头速度 1 m/s 时全队最大速度 vmax0 = {vmax0:.8f} m/s")
    print(f"  出现在 t = {t_star:.6f} s，第 {i_star} 号把手，弧长参数 s = {s_star:.6f} m")
    print(f"龙头最大行进速度 vmax = 2 / vmax0 = {vmax:.8f} m/s")
    print(f"耗时 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
