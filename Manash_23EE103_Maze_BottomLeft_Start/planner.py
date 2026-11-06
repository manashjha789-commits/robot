"""A* on an occupancy grid -> waypoint simplification -> cubic spline
-> curvature -> speed profile -> pure pursuit."""
import heapq
import math

import numpy as np
from scipy.interpolate import CubicSpline


def occupancy_grid(width, height, res, walls, margin):
    """walls: list of (x0, y0, x1, y1). margin = robot radius inflation."""
    nx, ny = int(round(width / res)), int(round(height / res))
    xs = (np.arange(nx) + 0.5) * res
    ys = (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    occ = np.zeros((ny, nx), dtype=bool)
    for x0, y0, x1, y1 in walls:
        occ |= (X >= x0 - margin) & (X <= x1 + margin) & (Y >= y0 - margin) & (Y <= y1 + margin)
    return occ


def to_cell(p, res):
    return int(p[0] // res), int(p[1] // res)


def to_xy(c, res):
    return ((c[0] + 0.5) * res, (c[1] + 0.5) * res)


def astar(occ, start, goal, res):
    """8-connected A* with Euclidean heuristic. Returns list of (x, y) points."""
    ny, nx = occ.shape
    s, g = to_cell(start, res), to_cell(goal, res)
    if occ[s[1], s[0]] or occ[g[1], g[0]]:
        raise ValueError("start or goal lies inside an obstacle")

    def h(c):
        return math.hypot(c[0] - g[0], c[1] - g[1])

    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    heap = [(h(s), 0.0, s)]
    came, gscore, closed = {}, {s: 0.0}, set()
    while heap:
        _, gc, c = heapq.heappop(heap)
        if c in closed:
            continue
        if c == g:
            path = [c]
            while c in came:
                c = came[c]
                path.append(c)
            path.reverse()
            pts = [to_xy(p, res) for p in path]
            return [start] + pts[1:-1] + [goal]
        closed.add(c)
        for dx, dy in moves:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or occ[n[1], n[0]] or n in closed:
                continue
            ng = gc + math.hypot(dx, dy)
            if ng < gscore.get(n, float("inf")):
                gscore[n] = ng
                came[n] = c
                heapq.heappush(heap, (ng + h(n), ng, n))
    raise RuntimeError("no path found")


def _line_free(occ, p, q, res):
    n = max(2, int(math.dist(p, q) / (res / 4)))
    for k in range(n + 1):
        t = k / n
        cx, cy = to_cell((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])), res)
        if occ[cy, cx]:
            return False
    return True


def simplify(occ, pts, res):
    """Keep only waypoints needed to avoid obstacles (line-of-sight pruning)."""
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _line_free(occ, pts[i], pts[j], res):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def spline_path(wps, ds=0.01):
    """Clamped cubic spline in chord-length parameter. Returns (N,2) points and curvature."""
    p = np.array(wps, dtype=float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    u0 = (p[1] - p[0]) / seg[0]          # unit tangent at start
    uN = (p[-1] - p[-2]) / seg[-1]       # unit tangent at end
    sx = CubicSpline(t, p[:, 0], bc_type=((1, u0[0]), (1, uN[0])))
    sy = CubicSpline(t, p[:, 1], bc_type=((1, u0[1]), (1, uN[1])))
    ts = np.append(np.arange(0.0, t[-1], ds), t[-1])
    dx, dy = sx(ts, 1), sy(ts, 1)
    ddx, ddy = sx(ts, 2), sy(ts, 2)
    kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    return np.column_stack([sx(ts), sy(ts)]), kappa


def speed_profile(kappa, vmax, wmax):
    """v = min(vmax, wmax / |kappa|)"""
    return np.minimum(vmax, wmax / np.maximum(np.abs(kappa), 1e-9))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class PurePursuit:
    def __init__(self, path, v_prof, lookahead, vmax, wmax, goal_tol=0.03):
        self.P, self.v_prof = path, v_prof
        self.L, self.vmax, self.wmax = lookahead, vmax, wmax
        self.goal_tol = goal_tol
        self.idx = 0

    def done(self, x, y):
        return math.hypot(self.P[-1, 0] - x, self.P[-1, 1] - y) < self.goal_tol

    def command(self, x, y, th):
        if self.done(x, y):
            return 0.0, 0.0
        d = np.hypot(self.P[self.idx:, 0] - x, self.P[self.idx:, 1] - y)
        self.idx += int(np.argmin(d))
        j = self.idx
        while j < len(self.P) - 1 and math.hypot(self.P[j, 0] - x, self.P[j, 1] - y) < self.L:
            j += 1
        tx, ty = self.P[j]
        Ld = math.hypot(tx - x, ty - y)
        alpha = wrap(math.atan2(ty - y, tx - x) - th)
        kappa = 2 * math.sin(alpha) / max(Ld, 1e-6)
        v = min(self.v_prof[self.idx], self.vmax)
        if abs(kappa) > 1e-9:
            v = min(v, self.wmax / abs(kappa))
        return v, v * kappa
