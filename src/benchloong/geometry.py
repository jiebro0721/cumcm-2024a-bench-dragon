"""路径几何：阿基米德螺线、调头 S 形圆弧路径，以及弧长参数化。"""

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq

TAU = 2.0 * np.pi


def G(u: float) -> float:
    """G(u) = u*sqrt(1+u^2) + ln(u + sqrt(1+u^2))，用于等距螺线弧长。"""
    return u * math.sqrt(1.0 + u * u) + math.log(u + math.sqrt(1.0 + u * u))


def G_inv(C: float, u0: float | None = None) -> float:
    """解 G(u) = C (u >= 0)。牛顿法，失败时退化为二分法。"""
    if C < 0.0:
        raise ValueError(f"G_inv: C<0 ({C})")
    if u0 is None:
        u = max(math.sqrt(max(C, 0.0)) - 1.0, 0.05)
    else:
        u = float(u0)
    for _ in range(80):
        val = G(u) - C
        der = 2.0 * math.sqrt(1.0 + u * u)
        du = val / der
        u -= du
        if abs(du) <= 1e-13 * max(1.0, abs(u)):
            return float(u)
        if not math.isfinite(u):
            break
    return float(brentq(lambda x: G(x) - C, 0.0, max(1.0, math.sqrt(C) + 10.0)))


class IncomingSpiral:
    """盘入等距螺线 rho = a*theta/(2*pi)。

    弧长参数 s：s = 0 位于极角 theta_start 处；s < 0 为龙头后方（更外侧），
    s > 0 为前方（更内侧）。弧长与极角满足 G(theta) = G(theta_start) - 4*pi*s/a。
    """

    def __init__(self, a: float, theta_start: float):
        self.a = float(a)
        self.theta_start = float(theta_start)
        self._c0 = G(self.theta_start)

    def theta(self, s: float) -> float:
        return G_inv(self._c0 - TAU * 2.0 * s / self.a)

    def point(self, s: float) -> np.ndarray:
        th = self.theta(s)
        rho = self.a * th / TAU
        return rho * np.array([math.cos(th), math.sin(th)])

    def tangent(self, s: float) -> np.ndarray:
        """单位切向量（沿龙头前进方向）。"""
        th = self.theta(s)
        n = math.sqrt(1.0 + th * th)
        return -np.array([math.cos(th) - th * math.sin(th),
                          math.sin(th) + th * math.cos(th)]) / n


class OutgoingSpiral:
    """盘出等距螺线 rho = a*(theta+pi)/(2*pi)（与盘入螺线中心对称）。"""

    def __init__(self, a: float):
        self.a = float(a)

    def theta(self, s: float, s_start: float, theta_start: float) -> float:
        c = G(theta_start + np.pi) + TAU * 2.0 * (s - s_start) / self.a
        return G_inv(c) - np.pi

    def point(self, theta: float) -> np.ndarray:
        rho = self.a * (theta + np.pi) / TAU
        return rho * np.array([math.cos(theta), math.sin(theta)])

    def tangent(self, theta: float) -> np.ndarray:
        u = theta + np.pi
        n = math.sqrt(1.0 + u * u)
        return np.array([math.cos(theta) - u * math.sin(theta),
                         math.sin(theta) + u * math.cos(theta)]) / n


class TurnPath:
    """问题 4 的完整路径：盘入螺线 + 两段相切圆弧（S 形调头曲线）+ 盘出螺线。

    弧长参数 s 以盘入螺线与调头空间的交点 P1 为 0：
    s < 0          盘入螺线（龙头后方，更外侧）
    0 <= s < L1    第 1 段圆弧（半径 r1）
    L1 <= s < L   第 2 段圆弧（半径 r2）
    s >= L         盘出螺线
    """

    def __init__(self, a: float, r: float, k: float):
        self.a = float(a)
        self.r = float(r)
        self.k = float(k)

        theta1 = TAU * r / a
        self.theta1 = theta1
        p1 = r * np.array([np.cos(theta1), np.sin(theta1)])
        n = np.sqrt(1.0 + theta1 * theta1)
        # 外法向（由圆心指向 P1 所在半径方向的单位向量）
        n_out = np.array([np.sin(theta1) + theta1 * np.cos(theta1),
                          -(np.cos(theta1) - theta1 * np.sin(theta1))]) / n
        r_sum = r * n / theta1
        self.r_sum = r_sum
        self.r1 = k * r_sum / (k + 1.0)
        self.r2 = r_sum / (k + 1.0)
        self.o1 = p1 - self.r1 * n_out
        self.o2 = -p1 + self.r2 * n_out
        self.b = (self.r2 * self.o1 + self.r1 * self.o2) / r_sum
        self.p1 = p1
        self.p2 = -p1

        # 圆弧角度（以各自圆心为极点的极角）
        v1 = p1 - self.o1                     # O1 -> P1
        vb = self.b - self.o1                 # O1 -> B
        wb = self.b - self.o2                 # O2 -> B
        w2 = self.p2 - self.o2                # O2 -> P2
        alpha1 = abs(np.arctan2(np.cross(v1, vb), np.dot(v1, vb)))
        alpha2 = abs(np.arctan2(np.cross(wb, w2), np.dot(wb, w2)))
        if abs(alpha1 - alpha2) > 1e-8:
            raise ValueError("两段圆弧所张圆心角不一致")
        self.alpha = alpha1
        self.phi1 = float(np.arctan2(v1[1], v1[0]))
        self.psi1 = self.phi1 - alpha1
        self.psi2 = float(np.arctan2(wb[1], wb[0]))
        self.phi2 = self.psi2 + alpha2

        self.L1 = self.r1 * alpha1
        self.L2 = self.r2 * alpha2
        self.L = self.L1 + self.L2

        self._in = IncomingSpiral(a, theta1)
        self._out = OutgoingSpiral(a)

    # ---- 弧长参数 -> 极角类参数 ----
    def segment_of(self, s: float) -> str:
        if s < 0.0:
            return "in"
        if s < self.L1:
            return "arc1"
        if s < self.L:
            return "arc2"
        return "out"

    def point(self, s: float) -> np.ndarray:
        if s < 0.0:
            return self._in.point(s)
        if s < self.L1:
            phi = self.phi1 - s / self.r1
            return self.o1 + self.r1 * np.array([math.cos(phi), math.sin(phi)])
        if s < self.L:
            phi = self.psi2 + (s - self.L1) / self.r2
            return self.o2 + self.r2 * np.array([math.cos(phi), math.sin(phi)])
        th = self._out.theta(s, self.L, self.theta1 - np.pi)
        return self._out.point(th)

    def tangent(self, s: float) -> np.ndarray:
        if s < 0.0:
            return self._in.tangent(s)
        if s < self.L1:
            phi = self.phi1 - s / self.r1
            return np.array([math.sin(phi), -math.cos(phi)])
        if s < self.L:
            phi = self.psi2 + (s - self.L1) / self.r2
            return np.array([-math.sin(phi), math.cos(phi)])
        th = self._out.theta(s, self.L, self.theta1 - np.pi)
        return self._out.tangent(th)

    def speed_factor(self, s: float) -> float:
        """|dP/ds| = 1（弧长参数化）。"""
        return 1.0

    def head_arc_param(self, t: float, v0: float = 1.0) -> float:
        """龙头前把手在时刻 t 的弧长参数（t=0 为调头开始，v0=1 m/s）。"""
        return v0 * t
