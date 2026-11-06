# Maze Escape: Robot Finds the Way Itself (VS Code, no ROS)

## Files
- maze.jpeg                       : your maze picture
- explore_maze.py                 : MAIN. Robot explores with sensors and finds the exit
- maze_step_log.txt               : every decision the robot made (written by the run)
- maze_explore_bottom_left.gif    : exploration, starting from the bottom-left corner
- maze_final_bottom_left.gif      : route the robot drives after finding the exit
- maze_solver.py, maze_gifs.py    : comparison (robot that already knows the map)
- planner.py, robot_model.py      : planning and robot model

## Setup (once)
    python -m venv .venv
    .venv\Scripts\Activate.ps1          (Mac/Linux: source .venv/bin/activate)
    python -m pip install numpy scipy matplotlib pillow

## Run from the bottom-left corner
    python explore_maze.py maze.jpeg --start 75,550

Run from the robot's own position in the picture instead:
    python explore_maze.py maze.jpeg

## How it works
1. Robot senses walls with 16 rays. It does not know the map at the start.
2. It goes to the nearest frontier (known free cell next to unknown space).
3. If a wall is sensed on the planned route, it stops, updates the map and plans again.
4. When a ray leaves the picture through the gap, the exit is found.
5. Shortest route on the discovered map is found with Dijkstra (Euclidean distances).
