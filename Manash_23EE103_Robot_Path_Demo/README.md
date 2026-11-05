# Wheeled Robot Path Planning Demo (VS Code, no ROS)

## Files
- robot_model.py : teacher's formulas (v, w, theta, mass matrix M, torque tau = M * phi_ddot)
- planner.py     : A* (8-connected, Euclidean heuristic) -> waypoint simplification -> cubic spline
                   (clamped tangents, chord-length) -> curvature -> speed profile -> pure pursuit
- demo.py        : runs the whole pipeline, prints results, saves demo_path.png and demo_run.gif
- demo_path.png, demo_run.gif : outputs of the last run

## Setup (once)
1. Open the folder in VS Code (File > Open Folder).
2. Open terminal (Ctrl+`) and create a virtual environment:
       python -m venv .venv
   Activate it:
       Windows:  .venv\Scripts\activate
       Linux/Mac: source .venv/bin/activate
3. Install libraries:
       pip install numpy scipy matplotlib pillow

## Run
       python demo.py

Expected output (values from the last run):
    Waypoints after simplify : 10
    Path length (spline)     : 7.22 m
    Goal reached             : True
    Wall collision           : False
    Closest robot-wall gap   : 0.349 m
    Max heading error        : ~1e-14 rad (formula vs integrated)

Then open demo_path.png and demo_run.gif from the folder in VS Code.

## Values used (ASSUMED, confirm with the teacher)
r = 0.033 m, 2b = 0.160 m, mc = 0.9 kg, mw = 0.05 kg, Ic = 0.0045 kg m^2
Demo speeds: v_max = 1.0, w_max = 2.0 (real robot: v_max = 0.22, w_max = 2.84)
Lookahead = 0.15 m. Map: 5 m x 3 m with two walls.

To use real-robot speeds, change VMAX and WMAX at the top of demo.py.
