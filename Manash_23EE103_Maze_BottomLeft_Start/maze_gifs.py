"""Makes two GIFs for the maze:
  maze_pathfinding.gif : how A* searches the maze (yellow = checked cells, green = final path)
  maze_final_result.gif: the robot driving from its start to the exit

Run:  python maze_gifs.py maze.jpeg
"""
import heapq
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import robot_model as rm
from planner import simplify, spline_path, speed_profile, PurePursuit
from maze_solver import (load, find_exit, build_grid, snap_free, PX_PER_M,
                         CELL_PX, VMAX, WMAX, LOOKAHEAD, DT)


def astar_record(occ, start, goal):
    """Same A* as planner.astar, but also returns the order cells were checked.
    start/goal are (col, row) cell indices. Returns (path_cells, checked_order)."""
    ny, nx = occ.shape
    h = lambda c: math.hypot(c[0] - goal[0], c[1] - goal[1])
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    heap = [(h(start), 0.0, start)]
    came, gscore, closed, order = {}, {start: 0.0}, set(), []
    while heap:
        _, g, c = heapq.heappop(heap)
        if c in closed:
            continue
        closed.add(c)
        order.append(c)
        if c == goal:
            path = [c]
            while c in came:
                c = came[c]
                path.append(c)
            return path[::-1], order
        for dx, dy in moves:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or occ[n[1], n[0]] or n in closed:
                continue
            ng = g + math.hypot(dx, dy)
            if ng < gscore.get(n, float("inf")):
                gscore[n] = ng
                came[n] = c
                heapq.heappush(heap, (ng + h(n), ng, n))
    raise RuntimeError("no path found")


def cell_to_px(c):
    return ((c[0] + 0.5) * CELL_PX, (c[1] + 0.5) * CELL_PX)


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "maze.jpeg"
    img, dark, robot = load(img_path)
    exit_px, _ = find_exit(dark)
    occ = build_grid(dark)
    robot_s = snap_free(occ, robot)
    exit_s = snap_free(occ, exit_px)
    s_cell = (int(robot_s[0] // CELL_PX), int(robot_s[1] // CELL_PX))
    g_cell = (int(exit_s[0] // CELL_PX), int(exit_s[1] // CELL_PX))

    path_cells, order = astar_record(occ, s_cell, g_cell)
    ny, nx = occ.shape
    res = CELL_PX / PX_PER_M
    pts_m = [(cell_to_px(c)[0] / PX_PER_M, cell_to_px(c)[1] / PX_PER_M) for c in path_cells]
    wps_m = simplify(occ, pts_m, res)
    path, kappa = spline_path(wps_m)
    print(f"Cells checked by A*: {len(order)}, path cells: {len(path_cells)}")

    H, W = img.shape[0], img.shape[1]

    # ---------- GIF 1: path finding ----------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img.astype(np.uint8))
    overlay = np.zeros((H, W, 4))
    im = ax.imshow(overlay)
    ax.axis("off")
    ax.set_title("A* searching for the exit", fontsize=12)
    # each frame shows a bit more of the search
    n_frames = 120
    marks = np.linspace(1, len(order), n_frames).astype(int)
    path_px = [cell_to_px(c) for c in path_cells]
    checked_px = 0

    def draw_checked(upto):
        ov = np.zeros((H, W, 4))
        for c in order[:upto]:
            x0, y0 = c[0] * CELL_PX, c[1] * CELL_PX
            ov[y0:y0 + CELL_PX, x0:x0 + CELL_PX] = (1.0, 0.9, 0.1, 0.55)
        return ov

    def upd1(i):
        k = marks[i]
        if k < len(order):
            im.set_data(draw_checked(k))
            ax.set_title(f"A* searching: {k} cells checked", fontsize=12)
        else:
            im.set_data(draw_checked(len(order)))
        return [im]

    # final frames: draw the found path in green
    path_line, = ax.plot([], [], c="lime", lw=3)
    total_frames = n_frames + 20

    def upd1_full(i):
        if i < n_frames:
            return upd1(i)
        im.set_data(draw_checked(len(order)))
        xs = [p[0] for p in path_px]
        ys = [p[1] for p in path_px]
        ax.set_title("Path found!", fontsize=12)
        path_line.set_data(xs, ys)
        return [im, path_line]

    anim1 = FuncAnimation(fig, upd1_full, frames=total_frames, interval=60, blit=False)
    anim1.save("maze_pathfinding.gif", writer=PillowWriter(fps=15))
    plt.close(fig)

    # ---------- GIF 2: final result (robot drives out) ----------
    v_prof = speed_profile(kappa, VMAX, WMAX)
    ctrl = PurePursuit(path, v_prof, LOOKAHEAD, VMAX, WMAX)
    x, y = robot_s[0] / PX_PER_M, robot_s[1] / PX_PER_M
    th0 = math.atan2(path[1, 1] - y, path[1, 0] - x)
    th, phi_r, phi_l = th0, 0.0, 0.0
    xs, ys = [x], [y]
    for _ in range(int(400 / DT)):
        v, w = ctrl.command(x, y, th)
        if ctrl.done(x, y):
            break
        dr, dl = rm.wheel_speeds(v, w)
        phi_r += dr * DT
        phi_l += dl * DT
        th = rm.theta_from_wheels(th0, phi_r, phi_l, 0.0, 0.0)
        x += v * math.cos(th) * DT
        y += v * math.sin(th) * DT
        xs.append(x)
        ys.append(y)
    reached = ctrl.done(x, y)
    print(f"Robot reached exit: {reached}, time {len(xs) * DT:.1f} s")

    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.imshow(img.astype(np.uint8))
    ax2.plot([p[0] * PX_PER_M for p in path], [p[1] * PX_PER_M for p in path], c="lime", lw=1, alpha=0.5)
    dot, = ax2.plot([], [], "o", c="blue", ms=10)
    trail, = ax2.plot([], [], c="red", lw=2)
    ax2.scatter(*robot_s, c="cyan", s=30)
    ax2.scatter(*exit_s, c="red", s=120, marker="*")
    ax2.axis("off")
    step = max(1, len(xs) // 160)
    frames = list(range(0, len(xs), step)) + [len(xs) - 1] * 15

    def upd2(i):
        dot.set_data([xs[i] * PX_PER_M], [ys[i] * PX_PER_M])
        trail.set_data([v * PX_PER_M for v in xs[:i + 1]], [v * PX_PER_M for v in ys[:i + 1]])
        ax2.set_title("Robot driving to the exit" if i < len(xs) - 1 else "Robot is out!", fontsize=12)
        return dot, trail

    FuncAnimation(fig2, upd2, frames=frames, interval=40, blit=False).save(
        "maze_final_result.gif", writer=PillowWriter(fps=25))
    plt.close(fig2)
    print("Saved maze_pathfinding.gif and maze_final_result.gif")


if __name__ == "__main__":
    main()
