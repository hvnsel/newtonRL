"""The navigate obstacle course for Luna-Excavator-Navigate-Course.

Metres in the env frame from the cell centre, +x along the route, +y to its
left. Pure data, loaded by scripts/plot_nav.py without Isaac Lab.
"""

# (spawn x, spawn y, spawn yaw, goal x, goal y, goal yaw).
COURSE_ROUTE = (-5.5, 0.0, 0.0, 5.5, 0.0, 0.0)

# Each obstacle is ("boulder", x, y, radius, height) or
# ("crater", x, y, radius, depth, rim height).
COURSE_LAYOUTS = {
    # One boulder dead on the line.
    "boulder": (("boulder", 0.0, 0.0, 0.7, 0.5),),
    # One crater dead on the line.
    "crater": (("crater", 0.0, 0.0, 1.6, 0.6, 0.15),),
    # Three boulders abreast, 0.3 m apart.
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
    # A row of boulders across the route with one 2.0 m gap left of the line.
    "gate": tuple(("boulder", 0.0, y, 0.6, 0.5) for y in (-3.9, -2.6, -1.3, 0.0, 1.3, 4.5, 5.8)),
    # A crater and two boulders scattered over the line.
    "field": (
        ("crater", -1.8, 0.6, 1.0, 0.5, 0.12),
        ("boulder", 0.8, -0.5, 0.55, 0.5),
        ("boulder", 2.4, 0.9, 0.5, 0.45),
    ),
}
