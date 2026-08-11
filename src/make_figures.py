"""生成解题报告插图（仅作图，不参与求解）。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchloong import (
    THETA0,
    HANDLE_GAPS,
    any_collision,
    chain_arc_params,
    chain_spiral,
    chain_speeds,
)
from benchloong.chain import bench_rectangles
from benchloong.geometry import G, G_inv, IncomingSpiral, TurnPath

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def _draw_chain(ax, pts, color, lw=1.2, alpha=0.95, label=None):
    ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=lw, alpha=alpha, label=label)
    ax.plot(pts[0, 0], pts[0, 1], "o", color=color, ms=4)


def fig_q1() -> None:
    a, v0 = 0.55, 1.0
    th = np.linspace(0.0, THETA0, 4000)
    rho = a * th / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    ax.plot(rho * np.cos(th), rho * np.sin(th), color="0.75", lw=0.8, zorder=1)
    colors = {0: "C0", 150: "C1", 300: "C3"}
    for t, c in colors.items():
        th0 = G_inv(G(THETA0) - 4 * np.pi * v0 * t / a)
        _, pts, _ = chain_spiral(a, th0, v0=v0)
        _draw_chain(ax, pts, c, label=f"t = {t} s")
    ax.plot([0], [0], "k+", ms=12, label="螺线中心 O")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("问题1：舞龙队盘入轨迹")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q1_trajectory.png", dpi=150)
    plt.close(fig)


def fig_q2() -> None:
    # 碰撞状态随龙头极角的变化（首次碰撞窗口）
    ths = np.arange(26.6, 24.6, -0.01)
    status = []
    for th in ths:
        try:
            _, pts, _ = chain_spiral(0.55, float(th))
            status.append(any_collision(pts))
        except Exception:
            status.append(True)
    status = np.asarray(status)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(ths, 0, 1, where=status, color="crimson", alpha=0.35,
                    label="存在板凳碰撞")
    ax.plot(ths, status.astype(float), "k-", lw=1.0)
    ax.axvline(26.1465348, color="C0", ls="--", label="终止极角 26.1465 rad")
    ax.set_xlabel("龙头极角 $\\theta_0$ (rad)")
    ax.set_ylabel("碰撞标志")
    ax.set_yticks([0, 1])
    ax.set_title("问题2：盘入过程中的碰撞窗口")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q2_collision_windows.png", dpi=150)
    plt.close(fig)

    # 终止时刻全队位形
    _, pts, spd = chain_spiral(0.55, 26.1465348, v0=1.0)
    rects = bench_rectangles(pts)
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    for r in rects[::2]:
        rr = np.vstack([r, r[0]])
        ax.plot(rr[:, 0], rr[:, 1], color="0.7", lw=0.6)
    for r in rects[1::2]:
        rr = np.vstack([r, r[0]])
        ax.plot(rr[:, 0], rr[:, 1], color="0.85", lw=0.4)
    ax.plot(pts[:, 0], pts[:, 1], "C0-", lw=1.4, label="把手中心连线")
    ax.plot(0, 0, "k+", ms=12)
    ax.set_aspect("equal")
    ax.set_title("问题2：终止时刻全队位形（龙头极角 26.1465 rad）")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q2_terminal.png", dpi=150)
    plt.close(fig)


def fig_q3() -> None:
    # 不同螺距下的碰撞结构：首次碰撞极角（或不存在）
    as_ = [0.40, 0.42, 0.45, 0.4503, 0.451, 0.46, 0.50, 0.55, 0.60]
    rows = []
    for a in as_:
        th1 = 2 * np.pi * 4.5 / a
        first = None
        th = THETA0
        while th > th1:
            try:
                _, pts, _ = chain_spiral(a, th)
                if any_collision(pts):
                    first = th
                    break
            except Exception:
                first = th
                break
            th -= 0.5
        rows.append((a, th1, first))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for a, th1, first in rows:
        radius_first = None if first is None else a * first / (2 * np.pi)
        ax.plot([a], [radius_first if radius_first is not None else 4.5],
                "o", color="C0" if radius_first is None else "crimson")
    ax.axhline(4.5, color="k", ls="--", lw=0.8, label="调头空间边界 r = 4.5 m")
    ax.set_xlabel("螺距 a (m)")
    ax.set_ylabel("首次碰撞处龙头半径 (m)")
    ax.set_title("问题3：不同螺距下首次碰撞位置（红色为边界前已碰撞）")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q3_min_pitch.png", dpi=150)
    plt.close(fig)


def fig_q4() -> None:
    path = TurnPath(1.7, 4.5, 2.0)
    th = np.linspace(0.0, path.theta1, 3000)
    rho_in = 1.7 * th / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(7.8, 7.8))
    ax.plot(rho_in * np.cos(th), rho_in * np.sin(th), color="0.8", lw=0.8)
    # 圆弧
    phi1 = np.linspace(path.psi1, path.phi1, 200)
    ax.plot(path.o1[0] + path.r1 * np.cos(phi1),
            path.o1[1] + path.r1 * np.sin(phi1), "C1-", lw=1.6, label="第1段圆弧")
    phi2 = np.linspace(path.psi2, path.phi2, 200)
    ax.plot(path.o2[0] + path.r2 * np.cos(phi2),
            path.o2[1] + path.r2 * np.sin(phi2), "C2-", lw=1.6, label="第2段圆弧")
    # 盘出螺线（中心对称的一段）
    tho = np.linspace(path.theta1 - np.pi, path.theta1 + 8.0, 2000)
    rho_out = 1.7 * (tho + np.pi) / (2 * np.pi)
    ax.plot(rho_out * np.cos(tho), rho_out * np.sin(tho), color="0.8", lw=0.8)
    for t, c in [(-10.0, "C0"), (0.0, "C3"), (10.0, "C4"), (20.0, "C5")]:
        s = chain_arc_params(path, t)
        pts = np.array([path.point(x) for x in s])
        _draw_chain(ax, pts, c, label=f"t = {int(t)} s")
    circ = plt.Circle((0, 0), 4.5, fill=False, color="k", ls=":", lw=1.2)
    ax.add_patch(circ)
    ax.plot(0, 0, "k+", ms=12)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("问题4：S 形调头路径与舞龙队位形")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q4_turn_path.png", dpi=150)
    plt.close(fig)


def fig_q5() -> None:
    path = TurnPath(1.7, 4.5, 2.0)
    ts = np.arange(10.0, 20.0, 0.05)
    vmaxs = []
    for t in ts:
        s = chain_arc_params(path, float(t))
        spd = chain_speeds(path, s)
        vmaxs.append(np.max(spd))
    vmaxs = np.asarray(vmaxs)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(ts, vmaxs, "C0-", lw=1.4)
    ax.axhline(1.604793, color="crimson", ls="--", lw=1.0,
               label="$v_{\\max}^0 = 1.604793$ m/s")
    ax.axvline(14.48, color="0.6", ls=":", label="t = 14.48 s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("全队最大把手速度 (m/s)")
    ax.set_title("问题5：龙头速度 1 m/s 时全队最大速度曲线")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q5_max_speed.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_q1()
    fig_q2()
    fig_q3()
    fig_q4()
    fig_q5()
    print(f"插图已保存到 {FIG_DIR}")


if __name__ == "__main__":
    main()
