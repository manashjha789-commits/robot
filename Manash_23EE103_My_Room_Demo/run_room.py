"""Step 2: plan and drive the robot in my_room.json.

Run:  python run_room.py            (demo speed)
      python run_room.py --real     (real TurtleBot3 speed)
Outputs: room_result.png (map + path + robot trace + speed/torque plots)
"""
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import robot_model as rm
from planner import (occupancy_grid, astar, simplify, spline_path,
                     speed_profile, PurePursuit)

RES = 0.05
MARGIN = 0.25          # robot radius 0.12 + safety
ROBOT_RADIUS = 0.12
LOOKAHEAD = 0.15
DT = 0.02
T_LIMIT = 400


def min_wall_dist(p, walls):
    best = float("inf")
    for x0, y0, x1, y1 in walls:
        cx, cy = min(max(p[0], x0), x1), min(max(p[1], y0), y1)
        best = min(best, math.hypot(p[0] - cx, p[1] - cy))
    return best


def run(room, vmax, wmax):
    walls = room["walls"]
    occ = occupancy_grid(room["width"], room["height"], RES, walls, MARGIN)
    wps = simplify(occ, astar(occ, tuple(room["start"]), tuple(room["goal"]), RES), RES)
    path, kappa = spline_path(wps)
    v_prof = speed_profile(kappa, vmax, wmax)
    ctrl = PurePursuit(path, v_prof, LOOKAHEAD, vmax, wmax)

    x, y = room["start"]
    th0 = math.atan2(path[1, 1] - y, path[1, 0] - x)
    th = th0
    phi_r = phi_l = 0.0
    xs, ys, vs, ws, phr, phl = [x], [y], [], [], [], []
    t = 0.0
    while t < T_LIMIT:
        v, w = ctrl.command(x, y, th)
        if ctrl.done(x, y):
            break
        dr, dl = rm.wheel_speeds(v, w)
        phi_r += dr * DT
        phi_l += dl * DT
        th = rm.theta_from_wheels(th0, phi_r, phi_l, 0.0, 0.0)
        x += v * math.cos(th) * DT
        y += v * math.sin(th) * DT
        t += DT
        xs.append(x); ys.append(y); vs.append(v); ws.append(w)
        phr.append(phi_r); phl.append(phi_l)

    phr, phl = np.array(phr), np.array(phl)
    taus = np.array([rm.wheel_torques(a, b) for a, b in
                     zip(np.gradient(phr, DT), np.gradient(phl, DT))])
    return dict(path=path, wps=wps, xs=np.array(xs[1:]), ys=np.array(ys[1:]),
                vs=np.array(vs), ws=np.array(ws), taus=taus,
                reached=ctrl.done(x, y), time=t, goal=room["goal"],
                gap=min(min_wall_dist(p, walls) for p in zip(xs, ys)), walls=walls,
                room=room, vmax=vmax, wmax=wmax)


def save_dashboard(r, out):
    room = r["room"]
    tt = np.arange(len(r["vs"])) * DT
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax = axs[0, 0]
    for x0, y0, x1, y1 in r["walls"]:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="gray"))
    ax.add_patch(plt.Rectangle((0, 0), room["width"], room["height"], fill=False, ec="black", lw=2))
    ax.plot(r["path"][:, 0], r["path"][:, 1], "b-", lw=1.5, label="spline path")
    ax.plot(r["xs"], r["ys"], "r--", lw=1, label="robot")
    ax.scatter(*room["start"], c="green", label="start")
    ax.scatter(*room["goal"], c="orange", label="goal")
    ax.set_xlim(0, room["width"]); ax.set_ylim(0, room["height"]); ax.set_aspect("equal")
    ax.set_title("Room, path and robot"); ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(tt, r["vs"], label="v [m/s]")
    ax.plot(tt, r["ws"], label="w [rad/s]")
    ax.axhline(r["vmax"], ls=":", c="C0"); ax.axhline(r["wmax"], ls=":", c="C1")
    ax.set_xlabel("time [s]"); ax.set_title("Velocity commands"); ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.plot(tt, r["taus"][:, 0], label="right wheel")
    ax.plot(tt, r["taus"][:, 1], label="left wheel")
    ax.set_xlabel("time [s]"); ax.set_ylabel("N m"); ax.set_title("Wheel torques"); ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.axis("off")
    ax.text(0, 0.9, f"Goal reached: {r['reached']}", fontsize=12)
    ax.text(0, 0.75, f"Time taken: {r['time']:.1f} s", fontsize=12)
    ax.text(0, 0.6, f"Closest wall gap: {r['gap']:.2f} m", fontsize=12)
    ax.text(0, 0.45, f"Waypoints: {len(r['wps'])}", fontsize=12)
    ax.text(0, 0.3, f"Speed limits: v={r['vmax']}, w={r['wmax']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    real = "--real" in sys.argv
    with open("my_room.json") as f:
        room = json.load(f)
    vmax, wmax = (0.22, 2.84) if real else (1.0, 2.0)
    r = run(room, vmax, wmax)
    save_dashboard(r, "room_result.png")
    print(f"Goal reached : {r['reached']}")
    print(f"Time taken   : {r['time']:.1f} s")
    print(f"Wall gap     : {r['gap']:.3f} m (robot radius {ROBOT_RADIUS} m)")
    print(f"Waypoints    : {len(r['wps'])}")
    print("Saved room_result.png")


if __name__ == "__main__":
    main()
