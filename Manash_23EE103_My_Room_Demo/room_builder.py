"""Step 1: draw your own room with the mouse.

Run:  python room_builder.py
- Enter room width and height in metres.
- For each obstacle: click two opposite corners on the graph.
- Then click the START point and the GOAL point.
The room is saved to my_room.json.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def ask_float(prompt, default):
    text = input(f"{prompt} [{default}]: ").strip()
    return float(text) if text else default


def main():
    W = ask_float("Room width (m)", 6.0)
    H = ask_float("Room height (m)", 3.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec="black", lw=2))
    plt.ion()
    plt.show()

    walls = []
    while True:
        if input("Add an obstacle? (y/n) [y]: ").strip().lower() in ("n", "no"):
            break
        ax.set_title("Click 2 opposite corners of the obstacle")
        fig.canvas.draw()
        (x0, y0), (x1, y1) = plt.ginput(2, timeout=0)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        walls.append([round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)])
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="gray"))
        fig.canvas.draw()

    ax.set_title("Click the START point")
    fig.canvas.draw()
    (sx, sy), = plt.ginput(1, timeout=0)
    ax.plot(sx, sy, "go", ms=9)
    ax.set_title("Click the GOAL point")
    fig.canvas.draw()
    (gx, gy), = plt.ginput(1, timeout=0)
    ax.plot(gx, gy, "o", color="orange", ms=9)
    fig.canvas.draw()

    room = {
        "width": W, "height": H,
        "walls": walls,
        "start": [round(sx, 3), round(sy, 3)],
        "goal": [round(gx, 3), round(gy, 3)],
    }
    with open("my_room.json", "w") as f:
        json.dump(room, f, indent=2)
    print("Saved my_room.json")
    print(json.dumps(room, indent=2))
    input("Press Enter to close the window...")
    plt.close(fig)


if __name__ == "__main__":
    main()
