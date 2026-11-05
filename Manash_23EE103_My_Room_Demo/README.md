# Draw Your Own Room and Drive the Robot (VS Code, no ROS)

## Setup (once)
    python -m venv .venv
    .venv\Scripts\Activate.ps1          (Mac/Linux: source .venv/bin/activate)
    python -m pip install numpy scipy matplotlib pillow

## Step 1: Draw the room
    python room_builder.py
1. Type room width and height in metres (Enter for default 6 x 3).
2. Type y to add an obstacle, then click 2 opposite corners of it on the graph.
3. Repeat for more obstacles. Type n when done.
4. Click the START point (green), then the GOAL point (orange).
5. my_room.json is saved. Press Enter in the terminal to close the window.

Tip: keep obstacles at least 0.3 m from the room walls and from start/goal, so the robot can pass.

## Step 2: Plan and drive
    python run_room.py           (demo speed, about 6-9 s)
    python run_room.py --real    (real TurtleBot3 speed, about 35 s)

Open room_result.png. It shows the room, the path, the robot trace, speed, torque and the result.

## Notes
- If you get "no path found", the obstacles block the way. Move one obstacle and draw again.
- You can also edit my_room.json by hand. Walls are [x_min, y_min, x_max, y_max] in metres.
