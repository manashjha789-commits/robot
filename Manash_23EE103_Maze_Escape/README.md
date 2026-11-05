# Maze Escape: Robot Finds the Way Itself (VS Code, no ROS)

## Files
- maze.jpeg             : your maze picture
- explore_maze.py       : MAIN. Robot explores the maze with sensors and finds the exit
- maze_solver.py        : knows the whole map, prints step-by-step directions (comparison)
- maze_gifs.py          : A* search GIF and drive GIF (comparison)
- planner.py, robot_model.py : path planning and robot model
- maze_explore.gif      : robot discovering the maze (grey = known free, black = wall found, white = unknown)
- maze_final_result.gif : robot driving the route it discovered
- maze_pathfinding.gif, maze_path.png, maze_solver results : comparison outputs

## Setup (once)
    python -m venv .venv
    .venv\Scripts\Activate.ps1          (Mac/Linux: source .venv/bin/activate)
    python -m pip install numpy scipy matplotlib pillow

## Run
    python explore_maze.py maze.jpeg

## How it works
1. Robot senses walls with 16 rays around it. It does NOT know the map beforehand.
2. It picks the nearest frontier (a known free cell next to unknown space) and goes there.
3. When a wall is sensed on the planned route, it stops, updates its map and plans again.
   This is the "go back and try another way" behaviour.
4. When a ray leaves the picture through the gap, the exit is found. The robot then drives the known route.
