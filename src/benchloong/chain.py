"""舞龙队链条：把手弧长递推、速度递推、碰撞判别。"""

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq

from .config import N_BENCHES, N_HANDLES, HANDLE_GAPS, BOARD_OVERHANG, HALF_WIDTH
from .geometry import TAU


class ChainError(Exception):
    """链条递推失败（几何上无法容纳，视为不可行/碰撞）。"""


def chain_spiral(a: float, theta_head: float, gaps=None, v0: float = 1.0):
    """纯盘入螺线情形的快速递推（与讲评式 (40)、(7)-(9) 一致）。

    返回 (极角数组, 位置 (N_HANDLES,2), 速度大小数组)。
    """
    if gaps is None:
        gaps = HANDLE_GAPS
    ths = [float(theta_head)]
    for k in range(N_BENCHES):
        l = float(gaps[k])
        th_i = ths[-1]
        rho_i = a * th_i / TAU
        d0 = l / rho_i

        def g(d: float) -> float:
            th = th_i + d
            return (a / TAU) ** 2 * (
                th_i * th_i + th * th - 2.0 * th_i * th * math.cos(d)
            ) - l * l

        lo_d = 0.5 * d0
        hi_d = 1.6 * d0
        while g(hi_d) < 0.0 and hi_d < TAU:
            hi_d *= 1.6
        if hi_d >= TAU or g(lo_d) > 0.0 or g(hi_d) < 0.0:
            raise ChainError(f"螺旋递推无解 (i={k}, theta_i={th_i})")
        d = brentq(g, lo_d, hi_d)
        ths.append(th_i + d)

    ths = np.asarray(ths)
    rho = a * ths / TAU
    pts = np.column_stack([rho * np.cos(ths), rho * np.sin(ths)])

    # 速度递推：式 (7)-(9)，u0 为 v0=1 时的角速度
    u = np.empty(N_HANDLES)
    u[0] = -TAU / (a * math.sqrt(1.0 + ths[0] * ths[0]))
    for i in range(N_BENCHES):
        th1_, th2_ = ths[i], ths[i + 1]
        dc = th2_ - th1_
        num = th2_ * math.cos(dc) - th1_ + th1_ * th2_ * math.sin(dc)
        den = th2_ - th1_ * math.cos(dc) + th1_ * th2_ * math.sin(dc)
        if abs(den) < 1e-12:
            raise ChainError(f"速度递推分母为零 (i={i})")
        u[i + 1] = u[i] * num / den
    speeds = v0 * (a / TAU) * np.sqrt(1.0 + ths * ths) * np.abs(u)
    return ths, pts, speeds


def rear_handle_s(path, s_i: float, l: float, max_scan: float = 2.2) -> float:
    """求后一条板凳前把手在路径上的弧长参数 s_{i+1}。

    条件：s_{i+1} < s_i，且 |P(s_{i+1}) - P(s_i)| = l。
    取从 s_i 向后（s 减小方向）第一次达到弦长 l 的点。
    若在 max_scan*l 范围内找不到，抛出 ChainError。
    """
    p_i = path.point(s_i)

    def h(s: float) -> float:
        d = path.point(s) - p_i
        return float(d[0] * d[0] + d[1] * d[1]) - l * l

    step = 0.05 * l
    hi_s = s_i - 0.7 * l          # 弦长 <= 弧长 = 0.7l < l，必有 h < 0
    hi_h = h(hi_s)
    lo_s = hi_s
    while True:
        lo_s -= step
        if s_i - lo_s > max_scan * l:
            raise ChainError(f"找不到满足弦长 {l} 的后把手 (s_i={s_i})")
        lo_h = h(lo_s)
        if lo_h > 0.0:
            return float(brentq(h, lo_s, hi_s))
        if lo_h == 0.0:
            return float(lo_s)
        hi_s, hi_h = lo_s, lo_h


def chain_arc_params(path, s_head: float, gaps: np.ndarray | None = None):
    """递推求 224 个把手的弧长参数 s_0..s_223。"""
    if gaps is None:
        gaps = HANDLE_GAPS
    s = [float(s_head)]
    for i in range(N_BENCHES):
        s.append(rear_handle_s(path, s[-1], float(gaps[i])))
    return np.asarray(s)


def chain_speeds(path, s_params: np.ndarray, v0: float = 1.0) -> np.ndarray:
    """由刚体约束递推各把手速度（大小）。

    对相邻把手 P_i、P_{i+1}，弦长约束对时间求导得
    u_{i+1} = u_i * [(P_{i+1}-P_i).tau_i] / [(P_{i+1}-P_i).tau_{i+1}]，
    其中 u_i = ds_i/dt，tau_i 为路径单位切向量。速度大小 = |u_i|。
    """
    pts = np.array([path.point(s) for s in s_params])
    taus = np.array([path.tangent(s) for s in s_params])
    u = np.empty(len(s_params))
    u[0] = float(v0)
    for i in range(len(s_params) - 1):
        d = pts[i + 1] - pts[i]
        num = float(np.dot(d, taus[i]))
        den = float(np.dot(d, taus[i + 1]))
        if abs(den) < 1e-12:
            raise ChainError(f"速度递推分母为零 (i={i})")
        u[i + 1] = u[i] * num / den
    return np.abs(u)


def bench_rectangles(points: np.ndarray, gaps: np.ndarray | None = None):
    """按讲评式 (10)-(17) 生成每条板凳的矩形顶点。

    points: (224, 2)，gaps: (223,)。返回每条板凳 4 个顶点 (223, 4, 2)。
    顶点顺序：Q, R, S, T。
    """
    if gaps is None:
        gaps = HANDLE_GAPS
    n = N_BENCHES
    rects = np.empty((n, 4, 2))
    d = BOARD_OVERHANG
    w = HALF_WIDTH
    for i in range(n):
        p = points[i]
        q = points[i + 1]
        v = q - p                    # 指向后方（后把手方向）
        length = float(gaps[i])
        vx, vy = v / length
        px, py = p
        rects[i, 0] = [px + (length + d) * vx + w * vy,
                       py + (length + d) * vy - w * vx]       # Q
        rects[i, 1] = [px + (length + d) * vx - w * vy,
                       py + (length + d) * vy + w * vx]       # R
        rects[i, 2] = [px - d * vx - w * vy,
                       py - d * vy + w * vx]                  # S
        rects[i, 3] = [px - d * vx + w * vy,
                       py - d * vy - w * vx]                  # T
    return rects


def _segments_intersect(a1, a2, b1, b2, tol: float = 1e-9) -> bool:
    """线段 A1A2 与 B1B2 是否相交（含相切），采用讲评式 (18)-(27)。"""
    ax, ay = a1
    bx, by = a2
    cx, cy = b1
    dx, dy = b2
    pxi, pyi = bx - ax, by - ay
    pxj, pyj = dx - cx, dy - cy
    delta = pxj * pyi - pxi * pyj
    if abs(delta) > 1e-12:
        t = (pxj * (cy - ay) - pyj * (cx - ax)) / delta
        s = (pxi * (cy - ay) - pyi * (cx - ax)) / delta
        return (-tol <= t <= 1.0 + tol) and (-tol <= s <= 1.0 + tol)
    # 平行 / 共线
    cross = pxi * (cy - ay) - pyi * (cx - ax)
    if abs(cross) > 1e-9:
        return False
    # 共线：投影到 AB 轴检查区间重叠
    proj = lambda p: (p[0] - ax) * pxi + (p[1] - ay) * pyi
    den = pxi * pxi + pyi * pyi
    if den == 0.0:
        return False
    amin, amax = 0.0, proj((bx, by))
    if amin > amax:
        amin, amax = amax, amin
    cmin = min(proj((cx, cy)), proj((dx, dy)))
    cmax = max(proj((cx, cy)), proj((dx, dy)))
    return cmin <= amax + tol and amin <= cmax + tol


def _collide_vectorized(rects, skip_adjacent: bool = True, tol: float = 1e-9) -> bool:
    """对全部板凳对（跳过相邻）做 16 组边-边相交检验（numpy 向量化）。"""
    n = rects.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + (2 if skip_adjacent else 1), n)]
    if not pairs:
        return False
    ii = np.array([p[0] for p in pairs], dtype=np.int64)
    jj = np.array([p[1] for p in pairs], dtype=np.int64)

    # 每条板凳 4 条边：顶点 k -> 顶点 (k+1)%4
    verts = rects                                   # (n,4,2)
    nextv = np.roll(rects, -1, axis=1)              # (n,4,2)

    kk = np.arange(16)
    ki = kk // 4                                    # 板凳 i 的边号
    kj = kk % 4                                     # 板凳 j 的边号

    A = verts[ii][:, ki]                            # (P,16,2)
    B = nextv[ii][:, ki]
    C = verts[jj][:, kj]
    D = nextv[jj][:, kj]

    ax, ay = A[..., 0], A[..., 1]
    bx, by = B[..., 0], B[..., 1]
    cx, cy = C[..., 0], C[..., 1]
    dx, dy = D[..., 0], D[..., 1]

    pxi = bx - ax
    pyi = by - ay
    pxj = dx - cx
    pyj = dy - cy
    delta = pxj * pyi - pxi * pyj

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (pxj * (cy - ay) - pyj * (cx - ax)) / delta
        s = (pxi * (cy - ay) - pyi * (cx - ax)) / delta
    ok = (np.abs(delta) > 1e-12) & (t >= -tol) & (t <= 1.0 + tol) \
        & (s >= -tol) & (s <= 1.0 + tol)
    if np.any(ok):
        return True

    # 平行/共线情形
    cross = pxi * (cy - ay) - pyi * (cx - ax)
    col = (np.abs(delta) <= 1e-12) & (np.abs(cross) <= 1e-9)
    if np.any(col):
        den = pxi * pxi + pyi * pyi
        # 投影到 AB 轴
        proj_c = (cx - ax) * pxi + (cy - ay) * pyi
        proj_d = (dx - ax) * pxi + (dy - ay) * pyi
        lo_a = np.minimum(0.0, pxi * pxi + pyi * pyi)
        hi_a = np.maximum(0.0, den)
        lo_cd = np.minimum(proj_c, proj_d)
        hi_cd = np.maximum(proj_c, proj_d)
        overlap = (lo_cd <= hi_a + tol) & (lo_a <= hi_cd + tol)
        if np.any(col & overlap):
            return True
    return False


def any_collision(points: np.ndarray,
                  gaps: np.ndarray | None = None,
                  skip_adjacent: bool = True) -> bool:
    """判别任意两条不相邻板凳是否碰撞（矩形任一边相交即碰撞）。"""
    rects = bench_rectangles(points, gaps)
    return _collide_vectorized(rects, skip_adjacent)
