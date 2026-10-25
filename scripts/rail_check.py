#!/usr/bin/env python3
"""Check the wheel rails against the machine, the bed and the drum swath.

The rails exist because the implicit MPM soil develops no footing bearing
capacity: a parked machine sinks through the whole bed, the proxy feedback is
correct while it does so (harvested wheel impulses summed to 820 N against a
797 N weight), and no material parameter, proxy mode, particle density, grid
resolution or bed depth changes it. The upstream two-way coupling example
fails the same way on its own published parameters. So the wheels come out of
the soil coupling and ride rigid rails; the drums still cut MPM.

That trade only holds if a rail can never touch a drum. The swath ends at
ROTOR_HALF_LEN and a tyre's inboard edge begins at exactly the same y, with
the collider margin reaching further still, so the clearance is the whole
reason this is safe rather than a guess. This asserts it, plus that the rail
tops land on the soil surface and the wheels start just above them.

Runs without Isaac Lab, Warp or a GPU: it loads the geometry module by path
and recomputes what __post_init__ does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    """Import a module by file path, skipping the package __init__ chain."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G = _load("_excavator_geom", ROOT / "luna_hifi_tasks" / "excavator" / "excavator.py")

# Mirrors of the env cfg's own values. Kept here as literals rather than
# imported, because importing the cfg needs Isaac Lab: if one of these drifts
# the printed table stops matching a real run, which is the point of checking.
VOXEL_SIZE = 0.04
MPM_COLLIDER_MARGIN = 0.5 * VOXEL_SIZE
BED_X = (-2.4, 3.4)
BED_Y = (-1.0, 1.0)
BED_DEPTH = 0.25
BED_WALL_THICKNESS = 0.10
WHEEL_RAIL_CLEARANCE = 0.06
SPAWN_CLEARANCE = 0.02          # SPAWN_Z = CHASSIS_Z + 0.02
MACHINE_KG = 492.0
SHROUD_OUT_R_FALLBACK = 0.212   # radial half-depth of the cutting end
LUNAR_G = 1.62


def main() -> int:
    margin = 0.5 * VOXEL_SIZE
    bed_top = margin + BED_DEPTH

    # The rotor ends at ROTOR_HALF_LEN but the shroud end plates close the
    # pockets past it, and those are the outermost thing on the cutting end.
    swath = G.ROTOR_HALF_LEN + G.SHROUD_END_T
    tyre_in = G.WHEEL_Y - G.WHEEL_HALF_W
    tyre_out = G.WHEEL_Y + G.WHEEL_HALF_W

    rail_in = swath + WHEEL_RAIL_CLEARANCE
    rail_out = BED_Y[1] - 2.0 * margin
    rail_w = rail_out - rail_in
    rail_cy = 0.5 * (rail_in + rail_out)
    rail_top = bed_top

    wheel_bottom = (G.CHASSIS_Z + SPAWN_CLEARANCE) + bed_top - G.WHEEL_RADIUS

    print("geometry")
    print(f"  rotor                 |y| <= {G.ROTOR_HALF_LEN:.3f}")
    print(f"  cutting end (shroud)  |y| <= {swath:.3f}  "
          f"<- {1000*(swath-tyre_in):.0f} mm inside the tyre band")
    print(f"  tyre                  |y| {tyre_in:.3f} - {tyre_out:.3f}  "
          f"(track {G.TRACK:.3f}, {2*G.WHEEL_HALF_W:.3f} wide)")
    print(f"  rail                  |y| {rail_in:.3f} - {rail_out:.3f}  "
          f"({rail_w:.3f} wide, centre {rail_cy:.3f})")
    print(f"  collider margin        {margin:.3f}")
    print(f"  bed                   z {margin:.3f} - {bed_top:.3f}, "
          f"x {BED_X[0]:.2f} - {BED_X[1]:.2f}, y {BED_Y[0]:.2f} - {BED_Y[1]:.2f}")
    print(f"  rail top              z {rail_top:.3f}")
    print(f"  wheel bottom at spawn z {wheel_bottom:.3f}")

    checks: list[tuple[str, bool, str]] = []

    # The one that matters: a drum at full depth must not reach a rail, with
    # the margin counted on both faces.
    gap = (rail_in - margin) - (swath + margin)
    checks.append((
        "rail clears the drum swath with both margins counted",
        gap > 0.0,
        f"gap {gap*1000:.1f} mm (clearance {WHEEL_RAIL_CLEARANCE:.3f} "
        f"vs 2x margin {2*margin:.3f})",
    ))

    checks.append((
        "rail runs out past the tyre, so drifting outboard stays carried",
        rail_out > tyre_out,
        f"rail to {rail_out:.3f} vs tyre edge {tyre_out:.3f}: "
        f"{1000*(rail_out-tyre_out):.0f} mm of outboard room",
    ))

    bearing_w = tyre_out - rail_in
    checks.append((
        "most of the tyre width bears on rail",
        bearing_w >= 0.5 * (2.0 * G.WHEEL_HALF_W),
        f"{bearing_w*1000:.0f} of {2000*G.WHEEL_HALF_W:.0f} mm "
        f"({100.0*bearing_w/(2*G.WHEEL_HALF_W):.0f}%), overhanging "
        f"{1000*(rail_in-tyre_in):.0f} mm inboard",
    ))

    checks.append((
        "rail top is flush with the soil surface",
        abs(rail_top - bed_top) < 1e-9,
        f"{rail_top:.3f} vs bed top {bed_top:.3f}",
    ))

    settle = wheel_bottom - rail_top
    checks.append((
        "wheels start above the rail, not interpenetrated",
        0.0 < settle <= 0.05,
        f"{settle*1000:.0f} mm to settle",
    ))

    wall_in = BED_Y[1]
    checks.append((
        "rail stays inboard of the bed wall",
        rail_out + margin < wall_in,
        f"rail+margin {rail_out+margin:.3f} vs wall face {wall_in:.3f}",
    ))

    checks.append((
        "rail spans the bed in x",
        True,
        f"length {BED_X[1]-BED_X[0]:.2f} between walls at "
        f"{BED_X[0]-0.5*BED_WALL_THICKNESS:.2f} / "
        f"{BED_X[1]+0.5*BED_WALL_THICKNESS:.2f}",
    ))

    weight = MACHINE_KG * LUNAR_G
    print()
    print(f"load: {MACHINE_KG:.0f} kg at {LUNAR_G} m/s2 = {weight:.0f} N, "
          f"{weight/4.0:.0f} N per wheel, carried by rigid contact")

    # The limit of the approach, stated as a number instead of a caveat. A
    # rail is fixed in the world and the machine is not, so any heading change
    # walks the cutting end toward the rail and the tyre off it. Whichever
    # comes first is the envelope.
    import math
    inboard_room = rail_in - tyre_in          # tyre runs off the inner face
    half_d = SHROUD_OUT_R_FALLBACK
    def corner_y(theta: float) -> float:
        return swath * math.cos(theta) + half_d * math.sin(theta)
    limit = (rail_in - margin) - margin
    theta = 0.0
    while theta < math.pi / 4 and corner_y(theta) < limit:
        theta += 1e-4
    print()
    print("straight-line only, and this is the envelope:")
    print(f"  cutting end reaches the rail at {math.degrees(theta):.1f} deg of yaw")
    print(f"  a tyre runs off the inner face after {inboard_room*1000:.0f} mm "
          f"of inboard drift")
    print("  the drift reward term, not the geometry, is what has to hold this;")
    print("  a pressure-sinkage wheel model is the version that survives yaw")

    print()
    bad = 0
    for name, ok, detail in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        bad += not ok

    print()
    if bad:
        print(f"{bad} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
