"""Find the way out of a maze image with A* + spline + pure pursuit.

Run:  python maze_solver.py maze.jpeg
Walls must be black, the robot must be blue, and the exit is the gap in the
outer wall on the right side.
Outputs: maze_path.png (path on the picture), maze_run.gif (robot driving out),
and the step-by-step directions printed in the terminal.
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
import numpy as np
from scipy import ndimage

import robot_model as rm
from planner import astar, simplify, spline_path, speed_profile, PurePursuit

PX_PER_M = 100.0          # scale: 100 pixels = 1 metre (change if you know the real size)
CELL_PX = 4               # grid cell size in pixels
ROBOT_RADIUS_PX = 12      # robot clearance in pixels
VMAX, WMAX = 0.5, 2.0     # m/s, rad/s for the simulated drive
LOOKAHEAD = 0.15
DT = 0.02


def load(path):
    a = np.array(Image.open(path).convert("RGB")).astype(int)
    # walls are black: every channel dark. Blue robot has blue=255, so it is NOT a wall.
    dark = a.max(axis=2) < 120
    blue = (a[:, :, 2] > 180) & (a[:, :, 0] < 80) & (a[:, :, 1] < 80)
    # robot = largest solid blue blob (thin sensor lines disappear after erosion)
    body = ndimage.binary_erosion(blue, structure=np.ones((5, 5)))
    lab, n = ndimage.label(body)
    sizes = ndimage.sum(body, lab, range(1, n + 1))
    ys, xs = np.where(lab == (np.argmax(sizes) + 1))
    robot = (xs.mean(), ys.mean())
    return a, dark, robot


def find_exit(dark):
    """Right outer wall = column with most dark pixels on the right side.
    Exit = the largest gap (non-dark run) in that column."""
    h, w = dark.shape
    counts = dark[:, w // 2:].sum(axis=0)
    xw = int(np.argmax(counts)) + w // 2
    col = dark[:, xw]
    ys = np.where(col)[0]
    top, bottom = ys.min(), ys.max()
    free = ~col[top:bottom + 1]
    best, cur, start = (0, 0), 0, 0
    for i, f in enumerate(free):
        if f:
            if cur == 0:
                start = i
            cur += 1
            if cur > best[1]:
                best = (start, cur)
        else:
            cur = 0
    s, length = best
    mid = top + s + length // 2
    return (xw - 14, mid), length


def build_grid(dark):
    h, w = dark.shape
    ny, nx = h // CELL_PX, w // CELL_PX
    blocks = dark[:ny * CELL_PX, :nx * CELL_PX].reshape(ny, CELL_PX, nx, CELL_PX)
    occ = blocks.any(axis=(1, 3))
    r = int(math.ceil(ROBOT_RADIUS_PX / CELL_PX))
    return ndimage.binary_dilation(occ, structure=np.ones((2 * r + 1, 2 * r + 1)))


def snap_free(occ, p_px):
    """Move a point to the nearest free cell (robot body or exit can sit inside the safety margin)."""
    d, idx = ndimage.distance_transform_edt(occ, return_indices=True)
    cy, cx = int(p_px[1] / CELL_PX), int(p_px[0] / CELL_PX)
    cy = min(max(cy, 0), occ.shape[0] - 1)
    cx = min(max(cx, 0), occ.shape[1] - 1)
    ny, nx = idx[0][cy, cx], idx[1][cy, cx]
    return ((nx + 0.5) * CELL_PX, (ny + 0.5) * CELL_PX)


def directions(wps_px):
    """Turn the waypoints into plain steps: 'move right 1.9 m' etc."""
    steps = []
    for (x0, y0), (x1, y1) in zip(wps_px[:-1], wps_px[1:]):
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) >= abs(dy):
            name = "right" if dx > 0 else "left"
            dist = abs(dx)
        else:
            name = "down" if dy > 0 else "up"
            dist = abs(dy)
        if dist < 3:
            continue
        if steps and steps[-1][0] == name:
            steps[-1][1] += dist
        else:
            steps.append([name, dist])
    return [f"{i + 1}. go {n} about {d / PX_PER_M:.2f} m" for i, (n, d) in enumerate(steps)]


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "maze.jpeg"
    img, dark, robot = load(img_path)
    exit_px, gap_len = find_exit(dark)
    print(f"Robot found at pixel      : ({robot[0]:.0f}, {robot[1]:.0f})")
    print(f"Exit found at pixel       : {exit_px} (gap {gap_len} px)")

    occ = build_grid(dark)
    to_cell = lambda p: (p[0] / PX_PER_M, p[1] / PX_PER_M)
    # planner works in metres with res = CELL_PX/100 m
    res = CELL_PX / PX_PER_M
    robot_snap = snap_free(occ, robot)
    exit_snap = snap_free(occ, exit_px)
    s_m = to_cell(robot_snap)
    g_m = to_cell(exit_snap)
    raw = astar(occ, s_m, g_m, res)
    wps_m = simplify(occ, raw, res)
    wps_px = [(x * PX_PER_M, y * PX_PER_M) for x, y in wps_m]
    path, kappa = spline_path(wps_m)
    print(f"Waypoints                 : {len(wps_m)}")
    print(f"Path length               : {np.sum(np.hypot(*np.diff(path, axis=0).T)):.2f} m")
    print("\nSTEP BY STEP:")
    for line in directions(wps_px):
        print(line)

    # overlay on image
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img.astype(np.uint8))
    ax.plot(path[:, 0] * PX_PER_M, path[:, 1] * PX_PER_M, c="lime", lw=2.5, label="path")
    ax.scatter(*robot, c="blue", s=60, label="robot")
    ax.scatter(*exit_px, c="red", s=80, marker="*", label="exit")
    ax.legend(loc="lower left"); ax.axis("off")
    fig.savefig("maze_path.png", dpi=110, bbox_inches="tight")

    # simulate the drive out of the maze
    v_prof = speed_profile(kappa, VMAX, WMAX)
    ctrl = PurePursuit(path, v_prof, LOOKAHEAD, VMAX, WMAX)
    x, y = s_m
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
        xs.append(x); ys.append(y)
    print(f"\nSimulated drive reached exit: {ctrl.done(x, y)} (time {len(xs) * DT:.1f} s)")

    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.imshow(img.astype(np.uint8))
    ax2.plot(path[:, 0] * PX_PER_M, path[:, 1] * PX_PER_M, c="lime", lw=1, alpha=0.6)
    dot, = ax2.plot([], [], "o", c="blue", ms=9)
    trail, = ax2.plot([], [], c="red", lw=2)
    ax2.axis("off")
    step = max(1, len(xs) // 150)
    frames = list(range(0, len(xs), step))

    def upd(i):
        dot.set_data([xs[i] * PX_PER_M], [ys[i] * PX_PER_M])
        trail.set_data([v * PX_PER_M for v in xs[:i + 1]], [v * PX_PER_M for v in ys[:i + 1]])
        return dot, trail

    FuncAnimation(fig2, upd, frames=frames, interval=40, blit=True).save(
        "maze_run.gif", writer=PillowWriter(fps=25))
    print("Saved maze_path.png and maze_run.gif")


if __name__ == "__main__":
    main()
