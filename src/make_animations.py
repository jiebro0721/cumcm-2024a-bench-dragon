"""为五个问题生成运动可视化动画（MP4）。

动画中的每条板凳按碰撞模型 bench_rectangles 计算的实际轮廓绘制。

产物保存在 animations/：
  q1_spiral_in.mp4   问题 1：0~300 s 盘入全过程
  q2_terminal.mp4    问题 2：终止时刻前 27.5 s（碰撞对高亮）
  q3_min_pitch.mp4   问题 3：最小螺距下的完整盘入（到达调头空间边界）
  q4_turn.mp4        问题 4：调头全过程（-100~100 s）
  q5_speed.mp4       问题 5：速度分布 + 全队最大速度曲线
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchloong import (
    THETA0,
    chain_arc_params,
    chain_spiral,
    chain_speeds,
)
from benchloong.chain import bench_rectangles
from benchloong.geometry import G, G_inv, TurnPath

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ANIM_DIR = Path(__file__).resolve().parents[1] / "animations"
FPS = 24
N_BENCH = 223


def _writer(bitrate: int = 2200):
    return FFMpegWriter(fps=FPS, bitrate=bitrate, codec="libx264",
                        extra_args=["-pix_fmt", "yuv420p"])


def _save(anim, name: str, n_frames: int, bitrate: int = 2200) -> None:
    ANIM_DIR.mkdir(parents=True, exist_ok=True)
    path = ANIM_DIR / name
    anim.save(path, writer=_writer(bitrate), dpi=110)
    print(f"已生成 {path}（{n_frames} 帧）")


def _bench_rect_segments(pts):
    """返回每条板凳 4 条边的线段数组 (223,4,2,2)。"""
    rects = bench_rectangles(pts)                 # (223,4,2)
    edges = np.stack([rects, np.roll(rects, -1, axis=1)], axis=2)   # (223,4,2,2)
    return edges.reshape(-1, 2, 2)                # (892,2,2)


def _new_dragon(ax, mode: str = "alt", vmin=0.9, vmax=1.65):
    """创建板凳轮廓集合。mode: alt=交替灰度, speed=按速度着色。"""
    if mode == "speed":
        poly = PolyCollection([], cmap="jet", edgecolors="0.30",
                              linewidths=0.4, zorder=3)
        poly.set_clim(vmin, vmax)
    else:
        poly = PolyCollection([], facecolors="0.80", edgecolors="0.35",
                              linewidths=0.5, zorder=3)
    ax.add_collection(poly)
    head, = ax.plot([], [], "o", color="C3", ms=5.5, zorder=6)
    return poly, head


def _update_dragon(poly, head, pts, speeds=None, highlight=None):
    rects = bench_rectangles(pts)
    poly.set_verts([rect for rect in rects])
    if speeds is not None:
        bench_speed = 0.5 * (speeds[:-1] + speeds[1:])
        poly.set_array(bench_speed)
    elif highlight is not None:
        colors = ["0.76" if i % 2 == 0 else "0.90" for i in range(N_BENCH)]
        for k in highlight:
            colors[k] = "crimson"
        poly.set_facecolors(colors)
    else:
        colors = ["0.76" if i % 2 == 0 else "0.90" for i in range(N_BENCH)]
        poly.set_facecolors(colors)
    head.set_data([pts[0, 0]], [pts[0, 1]])
    return poly, head


def _spiral_guide(ax, a, th_max, color="0.82", lw=0.6):
    th = np.linspace(0.0, th_max, 4000)
    rho = a * th / (2 * np.pi)
    ax.plot(rho * np.cos(th), rho * np.sin(th), color=color, lw=lw, zorder=1)


def anim_q1() -> None:
    a, v0 = 0.55, 1.0
    times = np.arange(0, 301, 1.0)
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    _spiral_guide(ax, a, THETA0)
    poly, head = _new_dragon(ax)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=12)
    ax.plot([0], [0], "k+", ms=12, zorder=2)
    ax.set_xlim(-12.5, 12.5)
    ax.set_ylim(-12.5, 12.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("问题 1：舞龙队沿螺线盘入（0 ~ 300 s）")

    def update(k):
        t = times[k]
        th0 = G_inv(G(THETA0) - 4 * np.pi * v0 * t / a)
        _, pts, _ = chain_spiral(a, th0, v0=v0)
        _update_dragon(poly, head, pts)
        txt.set_text(f"t = {t:.0f} s")
        return poly, head, txt

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / FPS, blit=False)
    _save(anim, "q1_spiral_in.mp4", len(times))
    plt.close(fig)


def _first_colliding_pair(rects):
    from benchloong.chain import _segments_intersect

    n = rects.shape[0]
    for i in range(n):
        r1 = rects[i]
        e1 = [(r1[0], r1[1]), (r1[1], r1[2]), (r1[2], r1[3]), (r1[3], r1[0])]
        for j in range(i + 2, n):
            r2 = rects[j]
            e2 = [(r2[0], r2[1]), (r2[1], r2[2]), (r2[2], r2[3]), (r2[3], r2[0])]
            for a1, a2 in e1:
                for b1, b2 in e2:
                    if _segments_intersect(a1, a2, b1, b2):
                        return (i, j)
    return None


def _rect_dist(rects, i, j):
    r1, r2 = rects[i], rects[j]
    best = 1e9
    for p in r1:
        for a, b in [(r2[0], r2[1]), (r2[1], r2[2]), (r2[2], r2[3]), (r2[3], r2[0])]:
            v = b - a
            w = p - a
            tt = np.clip(np.dot(w, v) / np.dot(v, v), 0, 1)
            best = min(best, np.linalg.norm(p - (a + tt * v)))
    for p in r2:
        for a, b in [(r1[0], r1[1]), (r1[1], r1[2]), (r1[2], r1[3]), (r1[3], r1[0])]:
            v = b - a
            w = p - a
            tt = np.clip(np.dot(w, v) / np.dot(v, v), 0, 1)
            best = min(best, np.linalg.norm(p - (a + tt * v)))
    return best


def anim_q2() -> None:
    a, v0 = 0.55, 1.0
    theta_T = 26.1465348
    T = a / (4 * np.pi * v0) * (G(THETA0) - G(theta_T))
    times = np.arange(T - 27.5, T, 0.1)

    # 确定终止边界附近最先碰撞的板凳对
    pair = None
    for th in np.arange(theta_T, theta_T - 0.6, -0.01):
        _, pts, _ = chain_spiral(a, th)
        pair = _first_colliding_pair(bench_rectangles(pts))
        if pair is not None:
            break
    if pair is None:
        raise RuntimeError("未找到碰撞对")

    positions = []
    dists = np.empty(len(times))
    for k, t in enumerate(times):
        th0 = G_inv(G(THETA0) - 4 * np.pi * v0 * t / a)
        _, pts, _ = chain_spiral(a, th0, v0=v0)
        positions.append(pts)
        dists[k] = _rect_dist(bench_rectangles(pts), pair[0], pair[1])

    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.4], hspace=0.28)
    axd = fig.add_subplot(gs[0])
    axd.set_xlim(times[0], times[-1])
    axd.set_ylim(0, max(dists.max() * 1.15, 0.01))
    axd.plot(times, dists, color="0.6", lw=1.0)
    (dline,) = axd.plot([], [], "C0o", ms=4)
    axd.axhline(0.0, color="crimson", lw=0.8, ls="--")
    axd.set_ylabel("碰撞对间距 (m)")
    axd.set_title(f"问题 2：第 {pair[0]} 号与第 {pair[1]} 号板凳间距（终止前 27.5 s）")

    ax = fig.add_subplot(gs[1])
    _spiral_guide(ax, a, THETA0)
    poly, head = _new_dragon(ax)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=11)
    ax.plot([0], [0], "k+", ms=12)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    def update(k):
        pts = positions[k]
        _update_dragon(poly, head, pts, highlight=pair)
        dline.set_data([times[k]], [dists[k]])
        txt.set_text(f"t = {times[k]:.2f} s   间距 = {dists[k]:.5f} m")
        return poly, head, dline, txt

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / FPS, blit=False)
    _save(anim, "q2_terminal.mp4", len(times))
    plt.close(fig)


def anim_q3() -> None:
    a, v0, r = 0.45033645, 1.0, 4.5
    theta1 = 2 * np.pi * r / a
    T = a / (4 * np.pi * v0) * (G(THETA0) - G(theta1))
    times = np.arange(0.0, T + 0.01, 1.0)

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    _spiral_guide(ax, a, THETA0)
    circ = plt.Circle((0, 0), r, fill=False, color="green", ls="--", lw=1.6, zorder=2)
    ax.add_patch(circ)
    poly, head = _new_dragon(ax)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=11)
    ax.plot([0], [0], "k+", ms=12)
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("问题 3：最小螺距下的完整盘入")

    def update(k):
        t = times[k]
        th0 = G_inv(G(THETA0) - 4 * np.pi * v0 * t / a)
        _, pts, _ = chain_spiral(a, th0, v0=v0)
        _update_dragon(poly, head, pts)
        txt.set_text(f"t = {t:.0f} s   龙头半径 = {np.hypot(pts[0,0], pts[0,1]):.3f} m")
        return poly, head, txt

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / FPS, blit=False)
    _save(anim, "q3_min_pitch.mp4", len(times))
    plt.close(fig)


def anim_q4() -> None:
    path = TurnPath(1.7, 4.5, 2.0)
    times = np.arange(-100, 101, 1.0)

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    _spiral_guide(ax, 1.7, path.theta1)
    tho = np.linspace(path.theta1 - np.pi, path.theta1 + 9.0, 2400)
    rho_out = 1.7 * (tho + np.pi) / (2 * np.pi)
    ax.plot(rho_out * np.cos(tho), rho_out * np.sin(tho), color="0.82", lw=0.6, zorder=1)
    phi1 = np.linspace(path.psi1, path.phi1, 200)
    ax.plot(path.o1[0] + path.r1 * np.cos(phi1),
            path.o1[1] + path.r1 * np.sin(phi1), "C1-", lw=1.6, zorder=2)
    phi2 = np.linspace(path.psi2, path.phi2, 200)
    ax.plot(path.o2[0] + path.r2 * np.cos(phi2),
            path.o2[1] + path.r2 * np.sin(phi2), "C2-", lw=1.6, zorder=2)
    circ = plt.Circle((0, 0), 4.5, fill=False, color="green", ls="--", lw=1.3, zorder=2)
    ax.add_patch(circ)
    poly, head = _new_dragon(ax)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=11)
    ax.plot([0], [0], "k+", ms=12)
    ax.set_xlim(-14, 14)
    ax.set_ylim(-14, 14)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("问题 4：盘入—S 形调头—盘出（-100 ~ 100 s）")

    def update(k):
        t = times[k]
        s = chain_arc_params(path, float(t))
        pts = np.array([path.point(x) for x in s])
        _update_dragon(poly, head, pts)
        txt.set_text(f"t = {t:.0f} s")
        return poly, head, txt

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / FPS, blit=False)
    _save(anim, "q4_turn.mp4", len(times))
    plt.close(fig)


def anim_q5() -> None:
    path = TurnPath(1.7, 4.5, 2.0)
    times = np.arange(8.0, 22.0, 0.05)
    frames = []
    vmaxs = np.empty(len(times))
    for k, t in enumerate(times):
        s = chain_arc_params(path, float(t))
        pts = np.array([path.point(x) for x in s])
        spd = chain_speeds(path, s)
        frames.append((pts, spd))
        vmaxs[k] = np.max(spd)

    fig = plt.figure(figsize=(11.4, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    axc = fig.add_subplot(gs[1])

    phi1 = np.linspace(path.psi1, path.phi1, 200)
    ax.plot(path.o1[0] + path.r1 * np.cos(phi1),
            path.o1[1] + path.r1 * np.sin(phi1), "C1-", lw=1.5, zorder=1)
    phi2 = np.linspace(path.psi2, path.phi2, 200)
    ax.plot(path.o2[0] + path.r2 * np.cos(phi2),
            path.o2[1] + path.r2 * np.sin(phi2), "C2-", lw=1.5, zorder=1)
    circ = plt.Circle((0, 0), 4.5, fill=False, color="green", ls="--", lw=1.2)
    ax.add_patch(circ)
    ax.plot([0], [0], "k+", ms=12)
    poly, head = _new_dragon(ax, mode="speed")
    txt = ax.text(0.03, 0.94, "", transform=ax.transAxes, fontsize=11)
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("问题 5：把手速度分布")
    cb = fig.colorbar(poly, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("板凳平均速度 (m/s)")

    axc.plot(times, vmaxs, color="0.6", lw=1.0)
    (cur,) = axc.plot([], [], "C0o", ms=5)
    axc.axhline(1.604793, color="crimson", ls="--", lw=1.0,
                label="$v_{\\max}^0$ = 1.604793")
    axc.set_xlim(times[0], times[-1])
    axc.set_ylim(0.9, 1.7)
    axc.set_xlabel("t (s)")
    axc.set_ylabel("全队最大速度 (m/s)")
    axc.set_title("全队最大速度曲线")
    axc.legend(fontsize=8)

    def update(k):
        pts, spd = frames[k]
        _update_dragon(poly, head, pts, speeds=spd)
        cur.set_data([times[k]], [vmaxs[k]])
        txt.set_text(f"t = {times[k]:.2f} s   最大 = {vmaxs[k]:.4f} m/s")
        return poly, head, cur, txt

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / FPS, blit=False)
    _save(anim, "q5_speed.mp4", len(times))
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    ANIM_DIR.mkdir(parents=True, exist_ok=True)
    print("生成问题 1 动画...")
    anim_q1()
    print("生成问题 2 动画...")
    anim_q2()
    print("生成问题 3 动画...")
    anim_q3()
    print("生成问题 4 动画...")
    anim_q4()
    print("生成问题 5 动画...")
    anim_q5()
    print(f"全部动画生成完毕，耗时 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
