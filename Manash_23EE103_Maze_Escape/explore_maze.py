"""Robot explores an UNKNOWN maze by itself using sensors (no pre-made path).

Idea (frontier-based exploration with backtracking):
1. The robot only knows what its sensors have seen so far.
2. It senses walls with 16 rays around it and marks cells as free or wall.
3. It goes to the nearest "frontier" (a known free cell next to unknown space).
4. If a sensor sees a wall on its planned path, it stops, updates the map,
   and plans again from where it is (this is the "go back and try another way").
5. When a ray leaves the picture through the gap, the exit is found.
   Then the robot walks the shortest route it has discovered.

Run:  python explore_maze.py maze.jpeg
Outputs: maze_explore.gif  (robot discovering the maze)
         maze_final_result.gif (robot driving the found route out)
"""
import sys
from collections import deque
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np
from scipy import ndimage

from maze_solver import load, find_exit, CELL_PX

ROBOT_CLEARANCE_CELLS = 3      # robot radius in cells (about 12 px)
SENSOR_RANGE_PX = 220
SENSOR_STEP_PX = 2
N_RAYS = 16
SENSE_EVERY = 2                # sense every 2 steps while walking
FRAME_EVERY = 6                # save a GIF frame every N steps

UNKNOWN, FREE, WALL = -1, 0, 1
DIRS8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


class Sensor:
    """Simulated ultrasonic-style rays. Uses the TRUE picture, the robot never sees it directly."""

    def __init__(self, dark, H, W):
        self.dark, self.H, self.W = dark, H, W

    def cast(self, x, y):
        """Return list of (px_x, px_y, is_wall) samples along each ray, and an exit flag."""
        hits, exit_cell = [], None
        for k in range(N_RAYS):
            a = 2 * math.pi * k / N_RAYS
            last_free = None
            for d in np.arange(0, SENSOR_RANGE_PX, SENSOR_STEP_PX):
                px, py = x + d * math.cos(a), y + d * math.sin(a)
                ix, iy = int(px), int(py)
                if ix < 0 or iy < 0 or ix >= self.W or iy >= self.H:
                    # ray left the picture: it went through the gap
                    if last_free is not None and exit_cell is None:
                        exit_cell = last_free
                    break
                if self.dark[iy, ix]:
                    hits.append((ix, iy, True))
                    break
                hits.append((ix, iy, False))
                last_free = (ix, iy)
        return hits, exit_cell


def px_to_cell(p, nx=None, ny=None):
    cx, cy = int(p[0] // CELL_PX), int(p[1] // CELL_PX)
    if nx is not None:
        cx = min(max(cx, 0), nx - 1)
        cy = min(max(cy, 0), ny - 1)
    return (cx, cy)


def cell_center(c):
    return ((c[0] + 0.5) * CELL_PX, (c[1] + 0.5) * CELL_PX)


def bfs(known, start, goal_test):
    """BFS over traversable cells. Returns (path, goal) or (None, None)."""
    ny, nx = known.shape
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if goal_test(c) and c != start:
            path = [c]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return path[::-1], c
        for dx, dy in DIRS8:
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < nx and 0 <= n[1] < ny and n not in prev and known_ok(known, n):
                prev[n] = c
                q.append(n)
    return None, None


def known_ok(known, n):
    """A cell is usable if the robot knows it is free and it keeps clearance from walls.
    The clearance uses only KNOWN walls, so unknown space is never assumed free."""
    return known[n[1], n[0]] == FREE and clear_map[n[1], n[0]]


clear_map = None  # set in main: True where the robot has enough room


def update_clearance(known):
    walls = (known == WALL)
    dist = ndimage.distance_transform_cdt(~walls, metric="chessboard")
    return dist > ROBOT_CLEARANCE_CELLS


def main():
    global clear_map
    img_path = sys.argv[1] if len(sys.argv) > 1 else "maze.jpeg"
    from PIL import Image
    img = np.array(Image.open(img_path).convert("RGB")).astype(int)
    dark = img.max(axis=2) < 120   # black walls only, not the blue robot
    H, W = dark.shape
    ny, nx = H // CELL_PX, W // CELL_PX

    # true map at cell level (used only by the simulated sensors)
    true_wall = dark[:ny * CELL_PX, :nx * CELL_PX].reshape(ny, CELL_PX, nx, CELL_PX).any(axis=(1, 3))

    # start: the robot position from the picture (blue body), snapped to a free cell
    from maze_solver import load as _load
    _, _, robot_px = _load(img_path)
    start = px_to_cell(robot_px)
    if true_wall[start[1], start[0]]:
        ys, xs = np.where(~true_wall)
        i = np.argmin((xs - start[0]) ** 2 + (ys - start[1]) ** 2)
        start = (int(xs[i]), int(ys[i]))

    known = np.full((ny, nx), UNKNOWN, dtype=np.int8)
    sensor = Sensor(dark, H, W)
    exit_cell_px = None
    exit_cell = None
    pos = start
    trail = [pos]
    frames = []           # snapshots: (known copy, pos, trail copy, exit flag)
    steps = 0

    def sense(c):
        nonlocal known, exit_cell_px
        global clear_map
        hits, ex = sensor.cast(*cell_center(c))
        for px, py, is_wall in hits:
            cx, cy = px // CELL_PX, py // CELL_PX
            if 0 <= cx < nx and 0 <= cy < ny:
                if is_wall:
                    known[cy, cx] = WALL
                elif known[cy, cx] != WALL:
                    known[cy, cx] = FREE
        if ex is not None and exit_cell_px is None:
            exit_cell_px = ex
        clear_map = update_clearance(known)

    def snap(frame_flag=False):
        frames.append((known.copy(), pos, list(trail), exit_cell is not None))

    sense(pos)
    snap()
    while True:
        if exit_cell_px is not None:
            # exit seen: go to it through known free cells
            target = px_to_cell(exit_cell_px, nx, ny)
            known[target[1], target[0]] = FREE if known[target[1], target[0]] != WALL else known[target[1], target[0]]
            route, _ = bfs(known, pos, lambda c: c == target)
            if route is not None:
                exit_cell = target
                break
            # route not yet known: keep exploring toward the gap
        # explore: nearest frontier = known usable cell next to unknown space
        def is_frontier(c):
            x, y = c
            for dx, dy in DIRS8:
                nxc, nyc = x + dx, y + dy
                if 0 <= nxc < nx and 0 <= nyc < ny and known[nyc, nxc] == UNKNOWN:
                    return True
            return False
        route, goal = bfs(known, pos, is_frontier)
        if route is None:
            print("No frontier left and exit not found. The maze may be blocked.")
            return
        # walk the route, sensing as we go
        replanned = False
        for nxt in route[1:]:
            if known[nxt[1], nxt[0]] == WALL:
                replanned = True
                break
            pos = nxt
            trail.append(pos)
            steps += 1
            if steps % SENSE_EVERY == 0 or nxt == route[-1]:
                before = known.copy()
                sense(pos)
                if (known[(before != known)] == WALL).any():
                    # a wall was discovered ahead of us: stop and re-plan
                    if any(known[c[1], c[0]] == WALL for c in route[route.index(nxt) + 1:]):
                        replanned = True
                        break
            if steps % FRAME_EVERY == 0:
                snap()
            if exit_cell_px is not None:
                break
        if exit_cell_px is None and not replanned:
            sense(pos)
        snap()
        if len(frames) > 2500:
            print("Stopped: too many steps")
            return

    snap()
    final_route, _ = bfs(known, start, lambda c: c == exit_cell)
    print(f"Exit found after {steps} steps. Robot is at pixel {cell_center(pos)}.")
    print(f"Known route length: {len(final_route) * CELL_PX / 100:.2f} m (100 px = 1 m)")
    print(f"Cells explored: {int(np.sum(known != UNKNOWN))}")

    # ---- GIF 1: exploration ----
    cmap = ListedColormap(["white", "#d0d0d0", "black"])
    fig, ax = plt.subplots(figsize=(7, 7))
    disp = ax.imshow(np.zeros((ny, nx)), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    rob, = ax.plot([], [], "o", c="blue", ms=8)
    tr, = ax.plot([], [], c="red", lw=1.5)
    ax.axis("off")

    def upd(i):
        k, p, tlist, found = frames[i]
        disp.set_data(k.astype(float))
        xs = [c[0] + 0.5 for c in tlist]
        ys = [c[1] + 0.5 for c in tlist]
        tr.set_data(xs, ys)
        rob.set_data([p[0] + 0.5], [p[1] + 0.5])
        ax.set_title(("EXIT FOUND!" if found else "Exploring... (white = unknown)"), fontsize=12)
        return disp, tr, rob

    FuncAnimation(fig, upd, frames=len(frames), interval=50, blit=False).save(
        "maze_explore.gif", writer=PillowWriter(fps=15))
    plt.close(fig)

    # ---- GIF 2: final route on the picture ----
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.imshow(img.astype(np.uint8))
    ax2.axis("off")
    rx = [cell_center(c)[0] for c in final_route]
    ry = [cell_center(c)[1] for c in final_route]
    line, = ax2.plot([], [], c="lime", lw=3)
    dot, = ax2.plot([], [], "o", c="blue", ms=10)
    ax2.scatter(*cell_center(start), c="cyan", s=40)
    ax2.scatter(*exit_cell_px, c="red", s=120, marker="*")

    def upd2(i):
        j = min(i, len(rx) - 1)
        line.set_data(rx[:j + 1], ry[:j + 1])
        dot.set_data([rx[j]], [ry[j]])
        ax2.set_title("Robot found the way out!" if j == len(rx) - 1 else "Robot driving out", fontsize=12)
        return line, dot

    FuncAnimation(fig2, upd2, frames=len(rx) + 15, interval=40, blit=False).save(
        "maze_final_result.gif", writer=PillowWriter(fps=25))
    plt.close(fig2)
    print("Saved maze_explore.gif and maze_final_result.gif")


if __name__ == "__main__":
    main()
