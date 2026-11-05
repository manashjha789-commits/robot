"""Offline demo: plans a path around two walls, simulates the wheeled robot,
checks the teacher's heading formula, prints torques, and saves an animation.

Run:  python demo.py
Outputs: demo_path.png, demo_run.gif  (in the same folder)
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import robot_model as rm
from planner import (occupancy_grid, astar, simplify, spline_path,
                     speed_profile, PurePursuit, to_cell)

# ---- demo settings (use real-robot values later: VMAX=0.22, WMAX=2.84) ----
VMAX, WMAX = 1.0, 2.0
LOOKAHEAD = 0.15
DT = 0.02
RES = 0.05
MARGIN = 0.25                      # planning clearance from walls [m] (robot radius 0.12 + safety)
MAP_W, MAP_H = 5.0, 3.0
WALLS = [(2.2, 0.0, 2.4, 1.9),     # wall 1: from bottom edge up
         (3.3, 1.1, 3.5, 3.0)]     # wall 2: from top edge down
START = (0.4, 0.4)
GOAL = (4.6, 2.6)


ROBOT_RADIUS = 0.12


def min_wall_dist(p):
    best = float("inf")
    for x0, y0, x1, y1 in WALLS:
        cx, cy = min(max(p[0], x0), x1), min(max(p[1], y0), y1)
        best = min(best, math.hypot(p[0] - cx, p[1] - cy))
    return best


def main():
    occ = occupancy_grid(MAP_W, MAP_H, RES, WALLS, MARGIN)
    raw = astar(occ, START, GOAL, RES)
    wps = simplify(occ, raw, RES)
    path, kappa = spline_path(wps)
    v_prof = speed_profile(kappa, VMAX, WMAX)
    ctrl = PurePursuit(path, v_prof, LOOKAHEAD, VMAX, WMAX)

    # initial state: heading toward the first path point
    x, y = START
    th0 = math.atan2(path[1, 1] - y, path[1, 0] - x)
    th = th0
    phi_r = phi_l = 0.0
    phi_r0, phi_l0 = phi_r, phi_l
    states, wheels, cmds = [(x, y, th)], [(phi_r, phi_l)], []
    max_heading_err = 0.0
    collided = False

    for _ in range(int(120 / DT)):
        v, w = ctrl.command(x, y, th)
        if ctrl.done(x, y):
            break
        cmds.append((v, w))
        dr, dl = rm.wheel_speeds(v, w)
        phi_r += dr * DT
        phi_l += dl * DT
        th_formula = rm.theta_from_wheels(th0, phi_r, phi_l, phi_r0, phi_l0)
        th_integrated = th + w * DT
        th = th_formula                       # heading from the teacher's formula
        max_heading_err = max(max_heading_err, abs(th_formula - (th0 + sum(c[1] for c in cmds) * DT)))
        x += v * math.cos(th) * DT
        y += v * math.sin(th) * DT
        if min_wall_dist((x, y)) < ROBOT_RADIUS:
            collided = True
        states.append((x, y, th))
        wheels.append((phi_r, phi_l))

    states = np.array(states)
    wheels = np.array(wheels)
    print(f"Waypoints after simplify : {len(wps)}")
    print(f"Path length (spline)     : {np.sum(np.hypot(*np.diff(path, axis=0).T)):.2f} m")
    print(f"Final position           : ({x:.2f}, {y:.2f}), goal ({GOAL[0]}, {GOAL[1]})")
    print(f"Goal reached             : {ctrl.done(x, y)}")
    print(f"Wall collision           : {collided}")
    print(f"Closest robot-wall gap   : {min(min_wall_dist(s[:2]) for s in states):.3f} m")
    print(f"Max heading error        : {max_heading_err:.2e} rad (formula vs integrated)")

    dt = DT
    phi_dd_r = np.gradient(wheels[:, 0], dt)
    phi_dd_l = np.gradient(wheels[:, 1], dt)
    taus = np.array([rm.wheel_torques(a, b) for a, b in zip(phi_dd_r, phi_dd_l)])
    print(f"Max torque (right, left) : {np.max(np.abs(taus), axis=0).round(4)} N m")

    # static plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for x0, y0, x1, y1 in WALLS:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="gray"))
    ax.plot(path[:, 0], path[:, 1], "b-", lw=1.5, label="spline path")
    ax.plot(states[:, 0], states[:, 1], "r--", lw=1, label="robot")
    ax.scatter(*START, c="green", label="start")
    ax.scatter(*GOAL, c="orange", label="goal")
    ax.set_xlim(0, MAP_W); ax.set_ylim(0, MAP_H); ax.set_aspect("equal")
    ax.legend(); ax.set_title("A* + spline + pure pursuit")
    fig.savefig("demo_path.png", dpi=120)

    # animation
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for x0, y0, x1, y1 in WALLS:
        ax2.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="gray"))
    ax2.plot(path[:, 0], path[:, 1], "b:", lw=1)
    dot, = ax2.plot([], [], "ro", ms=8)
    trail, = ax2.plot([], [], "r-", lw=1.5)
    ax2.set_xlim(0, MAP_W); ax2.set_ylim(0, MAP_H); ax2.set_aspect("equal")
    step = max(1, len(states) // 150)
    frames = range(0, len(states), step)

    def update(i):
        dot.set_data([states[i, 0]], [states[i, 1]])
        trail.set_data(states[:i + 1, 0], states[:i + 1, 1])
        return dot, trail

    anim = FuncAnimation(fig2, update, frames=frames, interval=40, blit=True)
    anim.save("demo_run.gif", writer=PillowWriter(fps=25))
    print("Saved demo_path.png and demo_run.gif")


if __name__ == "__main__":
    main()
