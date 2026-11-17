# -*- coding: utf-8 -*-
"""Draw a clean schematic of the intersection from the SUMO network file (no SUMO-GUI needed).

    python -m visualization.draw_scene                 # -> results/figures/intersection_schematic.png
    python -m visualization.draw_scene --half-size 150 --dpi 200

Shows the four approaches, the stop lines, the signal plan and the three ego routes
(straight / left / right, all entering from the north).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sumolib
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT / "environment" / "network" / "network.net.xml"
ROUTES = {"straight": ("south_out", "#2F5D8C"), "left": ("east_out", "#C8553D"), "right": ("west_out", "#2F855A")}


def bezier(p0, p1, p2, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half-size", type=float, default=140, help="half width of the drawn area in metres")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures" / "intersection_schematic.png")
    a = ap.parse_args()
    net = sumolib.net.readNet(str(NET), withInternal=False)
    jx, jy = net.getNode("junction").getCoord()
    h = a.half_size

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("white")
    for edge in net.getEdges():
        for lane in edge.getLanes():
            xy = np.array(lane.getShape())
            ax.plot(xy[:, 0] - jx, xy[:, 1] - jy, color="#9CA3AF", lw=lane.getWidth() * 8 * 72 * 0.78 / (2 * h),
                    solid_capstyle="butt", zorder=1)
    jp = np.array(net.getNode("junction").getShape())
    ax.add_patch(Polygon(jp - [jx, jy], closed=True, fc="#9CA3AF", ec="none", zorder=1))

    # stop lines at the end of every incoming edge
    for edge in net.getEdges():
        if not edge.getID().endswith("_in"):
            continue
        lanes = edge.getLanes()
        end = np.array(lanes[0].getShape()[-1]) - [jx, jy]
        prev = np.array(lanes[0].getShape()[-2]) - [jx, jy]
        d = (end - prev) / np.linalg.norm(end - prev)
        n = np.array([-d[1], d[0]])
        width = sum(l.getWidth() for l in lanes)
        ax.plot(*zip(end - n * width / 2, end + n * width / 2), color="white", lw=3, zorder=3)

    # ego routes, all entering from the north
    start = np.array(net.getEdge("north_in").getLanes()[0].getShape()[0]) - [jx, jy]
    stop = np.array(net.getEdge("north_in").getLanes()[0].getShape()[-1]) - [jx, jy]
    for name, (exit_id, color) in ROUTES.items():
        out = np.array(net.getEdge(exit_id).getLanes()[0].getShape()[0]) - [jx, jy]
        far = np.array(net.getEdge(exit_id).getLanes()[0].getShape()[-1]) - [jx, jy]
        far = out + (far - out) / np.linalg.norm(far - out) * min(h * 0.55, np.linalg.norm(far - out))
        pre = stop + (start - stop) / np.linalg.norm(start - stop) * h * 0.65
        curve = bezier(stop, np.array([stop[0], out[1]]) if name != "straight" else (stop + out) / 2, out)
        path = np.vstack([pre, curve, far])
        ax.annotate("", xy=path[-1], xytext=path[-6], arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5), zorder=5)
        ax.plot(path[:, 0], path[:, 1], color=color, lw=3, zorder=4, label={"straight": "直行 straight", "left": "左转 left", "right": "右转 right"}[name])

    ax.plot(*pre, "o", color="black", ms=9, zorder=6)
    ax.text(pre[0] + 6, pre[1], "自车 ego", fontsize=14, va="center", zorder=6)
    for txt, (x, y) in {"北 N（驶入）": (0, h * 0.93), "南 S": (0, -h * 0.93), "东 E": (h * 0.93, 0), "西 W": (-h * 0.93, 0)}.items():
        ax.text(x, y, txt, ha="center", va="center", fontsize=14, color="#374151", bbox=dict(fc="white", ec="none", pad=2), zorder=7)
    ax.text(h * 0.97, -h * 0.97, "信号周期 90 s：绿 42 s / 黄 3 s\n（两相位）", ha="right", va="bottom", fontsize=12, color="#374151")
    ax.set_xlim(-h, h); ax.set_ylim(-h, h); ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="upper left", fontsize=13, frameon=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    print("saved", a.out)


if __name__ == "__main__":
    main()
