"""The navigate obstacle course: a fixed route, and obstacles placed on it.

Playback only, for Luna-Excavator-Navigate-Course. The machine spawns facing
a goal 11 m straight ahead, and every layout puts craters or boulders on the
straight line between the two, so it has to steer round them to get there.

Coordinates are metres in the env frame, from the cell centre, with +x along
the route and +y to its left. Shapes are mdp/terrain.hazard_relief_np's, at
sizes inside the ranges the policy trained on: boulders 0.40-0.70 m in radius
and 0.35-0.60 m tall, craters 1.0-1.8 m in radius and 0.40-0.70 m deep behind
a 0.10-0.20 m rim.

Pure data, so scripts/plot_nav.py can load it by path to draw the obstacles
without importing Isaac Lab.
"""

# (spawn x, spawn y, spawn yaw, goal x, goal y, goal yaw). The machine reaches
# 1.76 m either side of its centre, so every obstacle stays inside x = +-3.3,
# clear of it at the spawn and parked on the goal.
COURSE_ROUTE = (-5.5, 0.0, 0.0, 5.5, 0.0, 0.0)

# Each obstacle is ("boulder", x, y, radius, height) or
# ("crater", x, y, radius, depth, rim height).
COURSE_LAYOUTS = {
    # One boulder dead on the line.
    "boulder": (("boulder", 0.0, 0.0, 0.7, 0.5),),
    # One crater dead on the line; with its rim it spans 4.2 m.
    "crater": (("crater", 0.0, 0.0, 1.6, 0.6, 0.15),),
    # Three boulders abreast 0.3 m apart, too close to pass between: the way
    # is round either end, 2.1 m off the line.
    "wall": (
        ("boulder", 0.0, -1.5, 0.6, 0.5),
        ("boulder", 0.0, 0.0, 0.6, 0.5),
        ("boulder", 0.0, 1.5, 0.6, 0.5),
    ),
    # A boulder and then a crater, both on the line.
    "chicane": (
        ("boulder", -2.0, 0.0, 0.6, 0.5),
        ("crater", 1.6, 0.0, 1.1, 0.5, 0.12),
    ),
    # A row of boulders across the route, 0.1 m apart, with one 2.0 m gap
    # 2.9 m to the left of the line. Round the near end is 4.5 m off it.
    "gate": tuple(("boulder", 0.0, y, 0.6, 0.5) for y in (-3.9, -2.6, -1.3, 0.0, 1.3, 4.5, 5.8)),
    # A crater and two boulders scattered over the line.
    "field": (
        ("crater", -1.8, 0.6, 1.0, 0.5, 0.12),
        ("boulder", 0.8, -0.5, 0.55, 0.5),
        ("boulder", 2.4, 0.9, 0.5, 0.45),
    ),
}
