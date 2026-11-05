"""Advanced demo: runs the same planner with two speed settings and shows
a 4-panel dashboard (map, speed, curvature, wheel torques).

Run:  python demo_advanced.py
Outputs: dashboard_demo_speed.png, dashboard_real_speed.png
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import robot_model as rm
from planner import (occupancy_grid, astar, simplify, spline_path,
                     speed_profile, PurePursuit, to_cell)
from demo import WALLS, START, GOAL, MAP_W, MAP_H, RES, MARGIN, LOOKAHEAD, DT, ROBOT_RADIUS, min_wall_dist

SCENARIOS = {
    "demo_speed": dict(vmax=1.0, wmax=2.0),      # fast, for the presentation
    "real_speed": dict(vmax=0.22, wmax=2.84),    # real TurtleBot3 Burger limits
}


def simulate(vmax, wmax, t_limit=400):
    occ = occupancy_grid(MAP_W, MAP_H, RES, WALLS, MARGIN)
    wps = simplify(occ, astar(occ, START, GOAL, RES), RES)
    path, kappa = spline_path(wps)
    v_prof = speed_profile(kappa, vmax, wmax)
    ctrl = PurePursuit(path, v_prof, LOOKAHEAD, vmax, wmax)

    x, y = START
    th0 = math.atan2(path[1, 1] - y, path[1, 0] - x)
    th = th0
    phi_r = phi_l = 0.0
    phi_r0, phi_l0 = phi_r, phi_l
    log = {"t": [], "x": [], "y": [], "th": [], "v": [], "w": [], "phi_r": [], "phi_l": []}
    closest_kappa_idx = []

    t = 0.0
    while t < t_limit:
        v, w = ctrl.command(x, y, th)
        if ctrl.done(x, y):
            break
        dr, dl = rm.wheel_speeds(v, w)
        phi_r += dr * DT
        phi_l += dl * DT
        th = rm.theta_from_wheels(th0, phi_r, phi_l, phi_r0, phi_l0)
        x += v * math.cos(th) * DT
        y += v * math.sin(th) * DT
        t += DT
        for k, val in zip(log.keys(), [t, x, y, th, v, w, phi_r, phi_l]):
            log[k].append(val)
        closest_kappa_idx.append(ctrl.idx)

    log = {k: np.array(val) for k, val in log.items()}
    # wheel accelerations and torques along the run
    phi_dd_r = np.gradient(log["phi_r"], DT)
    phi_dd_l = np.gradient(log["phi_l"], DT)
    taus = np.array([rm.wheel_torques(a, b) for a, b in zip(phi_dd_r, phi_dd_l)])
    kappa_run = kappa[np.array(closest_kappa_idx)]

    clearance = min(min_wall_dist((xx, yy)) for xx, yy in zip(log["x"], log["y"]))
    return dict(path=path, log=log, taus=taus, kappa=kappa_run,
                goal_reached=ctrl.done(x, y), clearance=clearance,
                length=np.sum(np.hypot(*np.diff(path, axis=0).T)),
                time=log["t"][-1], vmax=vmax, wmax=wmax)


def dashboard(res, name):
    log, path = res["log"], res["path"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax = axs[0, 0]
    for x0, y0, x1, y1 in WALLS:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="gray"))
    ax.plot(path[:, 0], path[:, 1], "b-", lw=1.5, label="spline path")
    ax.plot(log["x"], log["y"], "r--", lw=1, label="robot")
    ax.scatter(*START, c="green", label="start")
    ax.scatter(*GOAL, c="orange", label="goal")
    ax.set_xlim(0, MAP_W); ax.set_ylim(0, MAP_H); ax.set_aspect("equal")
    ax.set_title("Map and trajectory"); ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(log["t"], log["v"], label="v [m/s]")
    ax.plot(log["t"], log["w"], label="w [rad/s]")
    ax.axhline(res["vmax"], ls=":", c="C0", label=f"v_max={res['vmax']}")
    ax.axhline(res["wmax"], ls=":", c="C1", label=f"w_max={res['wmax']}")
    ax.set_xlabel("time [s]"); ax.set_title("Velocity commands"); ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.plot(log["t"], res["kappa"], c="purple")
    ax.set_xlabel("time [s]"); ax.set_ylabel("curvature [1/m]")
    ax.set_title("Path curvature along the run")

    ax = axs[1, 1]
    ax.plot(log["t"], res["taus"][:, 0], label="right wheel")
    ax.plot(log["t"], res["taus"][:, 1], label="left wheel")
    ax.set_xlabel("time [s]"); ax.set_ylabel("torque [N m]")
    ax.set_title("Wheel torques tau = M * phi_ddot"); ax.legend(fontsize=8)

    fig.suptitle(f"{name}: goal reached={res['goal_reached']}, time={res['time']:.1f} s, "
                 f"closest wall gap={res['clearance']:.2f} m", fontsize=11)
    fig.tight_layout()
    out = f"dashboard_{name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    for name, cfg in SCENARIOS.items():
        res = simulate(**cfg)
        out = dashboard(res, name)
        print(f"[{name}] v_max={cfg['vmax']} w_max={cfg['wmax']}")
        print(f"   path length  : {res['length']:.2f} m")
        print(f"   time taken   : {res['time']:.1f} s")
        print(f"   goal reached : {res['goal_reached']}")
        print(f"   wall gap     : {res['clearance']:.3f} m (robot radius {ROBOT_RADIUS} m)")
        print(f"   max |torque| : {np.max(np.abs(res['taus']), axis=0).round(4)} N m")
        print(f"   saved        : {out}")


if __name__ == "__main__":
    main()
