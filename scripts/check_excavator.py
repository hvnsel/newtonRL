# scripts/check_excavator.py
#
# Offline geometry check for the excavator MJCF. Needs MuJoCo, NOT Isaac Lab:
#
#   pip install "mujoco>=3.13"
#   python scripts/check_excavator.py
#
# Run it with a PLAIN Python, not isaaclab.bat -p. Isaac Sim pins mujoco to
# its own version (3.8.0 on the 6.0 line) because MJWarp, the rigid solver the
# excavator runs on, is built against it. Installing 3.13+ into that venv to
# satisfy this script replaces the solver's own MuJoCo. Nothing here imports
# the package -- excavator.py is loaded straight off disk below -- so any
# interpreter with mujoco and numpy will do.
#
# The next place these numbers show up is a converted USD inside a physics
# solver, where a 2 cm interpenetration presents as an unstable policy.
#
#   mass table        a body with no inertia NaNs the solver
#   swept envelopes   what the vanes and the shroud reach, measured in each
#                     box's own frame. A shroud plate's width runs along the
#                     arc, not radially, so treating a half-size as radial
#                     reports an interference fit where there is a running one.
#   shell continuity  adjacent plates must overlap; a gap is a hole MPM
#                     particles leak through
#   inlet probe       ray-cast around the drum axis to confirm the shroud has
#                     exactly one opening, where the geometry says. Two, and
#                     the drum empties wherever the second one points.
#   clearance sweep   arm swept through ARM_RANGE against the wheels and
#                     frame. MuJoCo never tests a body against its own parent
#                     and Isaac articulations default to self-collision off,
#                     so neither simulator reports an arm passing through a
#                     drum.
#
# Exit status is 1 if any check fails.

from __future__ import annotations

import math
import sys

import importlib.util
import pathlib

import mujoco
import numpy as np

# excavator.py is loaded straight off disk rather than as
# luna_hifi_tasks.excavator.excavator, because the package __init__ registers
# the gym tasks and drags in Isaac Lab. This check needs MuJoCo and nothing
# else, and it is most useful on the machine where you are editing the
# geometry, which is not necessarily the machine that can run Isaac.
_SRC = pathlib.Path(__file__).resolve().parents[1] / "luna_hifi_tasks" / "excavator" / "excavator.py"
_spec = importlib.util.spec_from_file_location("_excavator_geometry", _SRC)
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)

DISTMAX = 1.0

# mj_geomDistance is what every clearance and passage number here is built on,
# and on some MuJoCo builds it silently returns 0.0 for every pair instead of
# failing. That turns this script from a safety net into a liar: it reported a
# sealed drum and two clearance breaches on a model whose analytic geometry was
# provably fine. So it gets probed before it is trusted.
MIN_MUJOCO = "3.13"

# Clearance below this is reported as a failure. The arm and the wheels overlap
# in y -- the drum is wider than the gap between the tyres -- so the only thing
# keeping them apart is separation in x, bought by MAST_OFFSET_X.
MIN_CLEARANCE = 0.03


def _corners(m: mujoco.MjModel, gid: int) -> np.ndarray:
    """Extreme points of geom `gid` in its own BODY's frame."""
    pos = m.geom_pos[gid].copy()
    quat = m.geom_quat[gid].copy()
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    size = m.geom_size[gid]
    t = m.geom_type[gid]

    if t == mujoco.mjtGeom.mjGEOM_BOX:
        local = np.array([[sx * size[0], sy * size[1], sz * size[2]]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, h = size[0], size[1]
        ang = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
        rim = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros_like(ang)], axis=-1)
        local = np.concatenate([rim + [0, 0, h], rim - [0, 0, h]])
    else:  # sphere / capsule: bounding box is good enough
        r = size[0]
        local = np.array([[sx * r, sy * r, sz * r]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return local @ rot.T + pos


def _body_geoms(m: mujoco.MjModel, body: str) -> list[int]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    assert bid >= 0, f"no body named {body}"
    return list(range(m.body_geomadr[bid], m.body_geomadr[bid] + m.body_geomnum[bid]))


def _subtree_geoms(m: mujoco.MjModel, body: str) -> list[int]:
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    out, stack = [], [root]
    while stack:
        b = stack.pop()
        out += list(range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]))
        stack += [i for i in range(m.nbody) if m.body_parentid[i] == b and i != b]
    return out


def distance_works(m: mujoco.MjModel, d: mujoco.MjData) -> bool:
    """Does mj_geomDistance actually measure anything on this build?

    Probed against a pair whose separation is obvious from the model's own
    frames -- the deck and a shroud end plate are the better part of a metre
    apart -- so a zero here cannot be a real contact.

    The probe pair is named, so renaming a geom silently disarms it. It asserts
    rather than assuming: a disarmed probe reports every clearance as 0.0000
    and fails a model that is fine, which is worse than no probe at all.
    """
    a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "deck")
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "shroud_front_end_l")
    assert a >= 0 and b >= 0, (
        "the mj_geomDistance probe names geoms that no longer exist "
        f"(deck={a}, shroud_front_end_l={b}). Point it at a pair that does, or "
        "every clearance below reads 0.0000 and fails a sound model."
    )
    apart = float(np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b]))
    got = mujoco.mj_geomDistance(m, d, a, b, DISTMAX, None)
    if apart > 0.5 and got <= 1e-9:
        print(f"\n  !! mj_geomDistance returns {got} for two geoms {apart:.2f} m apart.")
        print(f"     This MuJoCo ({mujoco.__version__}) does not implement it usefully;")
        print(f"     {MIN_MUJOCO} or newer does. Every clearance and passage number below")
        print("     would be 0.0000 and every one of those checks would 'fail' on a")
        print("     model that is fine. They are SKIPPED instead.")
        print("       pip install -U mujoco")
        return False
    return True


def check_masses(m: mujoco.MjModel, fail: list[str]) -> None:
    print("\n=== mass ===")
    total = 0.0
    for bid in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        total += m.body_mass[bid]
        if m.body_mass[bid] <= 0.0:
            fail.append(f"body {name} has zero mass -- the solver will NaN")
        if m.body_geomnum[bid] == 0:
            fail.append(f"body {name} has no geom, so no derived inertia")
    print(f"  {m.ngeom} geoms, {m.njnt} joints, {m.nbody - 1} bodies")
    print(f"  total {total:.1f} kg")
    drum = sum(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)] for b in X.BODY_DRUMS)
    arms = sum(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)] for b in X.BODY_ARMS)
    print(f"  drums {drum:.1f} kg, arms {arms:.1f} kg, rest {total - drum - arms:.1f} kg")
    print(f"  chassis : drums = {(total - drum - arms) / drum:.1f} : 1")


def check_rotor_envelope(m: mujoco.MjModel, fail: list[str]) -> None:
    """What the vanes and the shroud actually reach, from the geom table."""
    print("\n=== rotor and shroud envelope (radius from the spin axis, m) ===")
    vane_lo, vane_hi = _radial_band(m, "drum_front_vane")
    shroud_lo, shroud_hi = _radial_band(m, "shroud_front_arc")
    print(f"  vanes  {vane_lo:.4f} .. {vane_hi:.4f}   (nominal {X.ROTOR_HUB_R:.3f} .. {X.ROTOR_TIP_R:.3f})")
    print(f"  shroud {shroud_lo:.4f} .. {shroud_hi:.4f}   (nominal {X.SHROUD_IN_R:.3f} .. {X.SHROUD_OUT_R:.3f})")

    clearance = shroud_lo - vane_hi
    print(f"  running clearance, vane tip to shroud {clearance:+.4f} m")
    if clearance <= 0.0:
        fail.append(f"the vanes overlap the shroud by {-clearance:.4f} m; the rotor "
                    "cannot turn. Lower ROTOR_TIP_R or raise SHROUD_IN_R")
    elif clearance >= X.MPM_TARGET_VOXEL:
        fail.append(f"running clearance {clearance:.4f} m is at least one "
                    f"{X.MPM_TARGET_VOXEL:.3f} m voxel, so the coupler leaves it OPEN and "
                    "regolith escapes round the whole circumference instead of staying in "
                    "a pocket. Tighten SHROUD_IN_R toward ROTOR_TIP_R")

    g = X.pocket_geometry()
    print(f"  {int(g['vanes'])} vanes, rake {g['rake_deg']:.0f} deg, attack {g['attack_deg']:.0f} deg")
    print(f"  each blade sweeps {g['vane_sweep_deg']:.1f} deg inside a "
          f"{g['pocket_arc_deg']:.1f} deg pocket")
    if g["shadowed"]:
        fail.append(f"a vane sweeps {g['vane_sweep_deg']:.1f} deg, more than the "
                    f"{g['pocket_arc_deg']:.1f} deg pocket it lives in, so adjacent vanes "
                    "shadow each other. Lower ROTOR_RAKE or ROTOR_VANES")
    if g["attack_deg"] > 80.0:
        fail.append(f"attack angle {g['attack_deg']:.0f} deg is nearly head-on; the vane "
                    "pushes regolith rather than cutting it. Raise ROTOR_RAKE")
    print(f"  inlet {g['inlet_deg']:.0f} deg, chord {g['inlet_chord']:.3f} m")
    print(f"  rotor swept volume {g['swept_volume']:.4f} m3 "
          f"= {g['swept_volume'] * SOIL_DENSITY:.0f} kg of regolith")


def check_shroud_continuity(fail: list[str]) -> None:
    """Adjacent shroud plates must overlap. A gap is a hole, not a passage."""
    print("\n=== shroud continuity ===")
    span = 2.0 * math.pi - X.SHROUD_INLET
    n = max(int(math.ceil(span / math.radians(20.0))), 6)
    step = span / n
    mid_r = 0.5 * (X.SHROUD_IN_R + X.SHROUD_OUT_R)
    half_arc = mid_r * math.tan(0.5 * step) + 0.002      # matches excavator.py
    half_chord = mid_r * math.sin(0.5 * step)
    overlap = half_arc - half_chord
    print(f"  {n} plates of {math.degrees(step):.1f} deg")
    print(f"  half-arc {half_arc:.5f} vs half-chord {half_chord:.5f} "
          f"-> overlap {overlap:+.5f} m per joint")
    if overlap <= 0.0:
        fail.append("shroud plates do not overlap; particles leak between them")


def check_inlet(m: mujoco.MjModel, d: mujoco.MjData, fail: list[str]) -> None:
    """Ray-cast around the axis to find the one opening in the shroud.

    The retention argument is that there is exactly one, and that it is where
    the geometry says. Two openings, or one in the wrong place, and the drum
    empties wherever the second one points.
    """
    print("\n=== shroud inlet probe (1 deg rays outward from the axis) ===")
    ids = [g for g in _subtree_geoms(m, "shroud_front_body")
           if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").find("_arc") >= 0]
    assert ids, "no shroud arc geoms; has the shroud been built?"

    # Each arc plate spans an angular half-width of atan(half_arc / radius)
    # about its own centre. A ray escapes when no plate spans it.
    spans = []
    for gid in ids:
        cx, cz = float(m.geom_pos[gid][0]), float(m.geom_pos[gid][2])
        r = math.hypot(cx, cz)
        spans.append((math.atan2(cz, cx), math.atan(float(m.geom_size[gid][0]) / r)))

    open_deg = []
    for deg in range(360):
        t = math.radians(deg)
        if not any(abs(math.atan2(math.sin(t - a), math.cos(t - a))) <= half
                   for a, half in spans):
            open_deg.append(deg)

    runs = _contiguous_runs(open_deg)
    print(f"  {len(runs)} opening(s), expected 1")
    for lo, hi in runs:
        width = (hi - lo) % 360 + 1
        centre = (lo + 0.5 * (width - 1)) % 360
        print(f"    centred {centre:6.1f} deg, {width} deg wide")
    if len(runs) != 1:
        fail.append(f"the shroud has {len(runs)} openings, not 1. Everything about "
                    "retention assumes soil can only leave the way it came in")
    else:
        lo, hi = runs[0]
        width = (hi - lo) % 360 + 1
        want = math.degrees(X.SHROUD_INLET)
        if abs(width - want) > 12.0:
            fail.append(f"inlet measures {width} deg against SHROUD_INLET={want:.0f} deg")


def check_vane_gaps(m: mujoco.MjModel, d: mujoco.MjData, fail: list[str]) -> None:
    """The way into a pocket is the chord between adjacent vane tips."""
    print("\n=== way into a pocket ===")
    g = X.pocket_geometry()
    v = X.MPM_TARGET_VOXEL
    print(f"  at the {v:.3f} m voxel the coupler eats {v:.3f} m of every gap")
    print(f"  open needs > {X.MPM_CLEARANCE:.3f} m, flowing needs > {X.MPM_FLOW:.3f} m")
    print(f"  vane tip to vane tip  {g['tip_gap']:.4f} m "
          f"-> {g['tip_gap_clear']:.4f} m clear = {g['tip_gap_grains']:.1f} grains")
    if g["tip_gap"] <= X.MPM_CLEARANCE:
        fail.append(f"the gap between vane tips is {g['tip_gap']:.4f} m, which the coupler "
                    f"closes at a {v:.3f} m voxel. Lower ROTOR_VANES")
    elif g["tip_gap"] < X.MPM_FLOW:
        print(f"  note: open but under {X.MPM_FLOW:.3f} m, so regolith trickles rather than "
              f"flows. ROTOR_VANES is the knob")
    print(f"  inlet chord           {g['inlet_chord']:.4f} m")
    if g["inlet_chord"] < X.MPM_FLOW:
        fail.append(f"the inlet chord is {g['inlet_chord']:.4f} m, under the "
                    f"{X.MPM_FLOW:.3f} m regolith needs to flow. Widen SHROUD_INLET")


def _radial_band(m: mujoco.MjModel, prefix: str) -> tuple[float, float]:
    """Min and max radius from the spin axis over every geom named `prefix`*.

    The boxes here are ORIENTED: a shroud plate's local x runs along the arc
    and its z radially, a vane's local x runs along the blade. Treating either
    half-size as radial reports a plate reaching 2 cm further in than its own
    inner face, which is the difference between a running fit and an
    interference. So this works in each box's own frame: farthest point is a
    corner, nearest is the axis clamped onto the rectangle.
    """
    lo, hi = math.inf, 0.0
    for gid in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not name.startswith(prefix):
            continue
        cx, cz = float(m.geom_pos[gid][0]), float(m.geom_pos[gid][2])
        hx, hz = float(m.geom_size[gid][0]), float(m.geom_size[gid][2])
        q = m.geom_quat[gid]
        # rotation about +y by angle b maps local x_hat -> (cos b, 0, -sin b)
        b = 2.0 * math.atan2(float(q[2]), float(q[0]))
        ux, uz = math.cos(b), -math.sin(b)       # local x in the x-z plane
        vx, vz = math.sin(b), math.cos(b)        # local z in the x-z plane
        for sx in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                px = cx + sx * hx * ux + sz * hz * vx
                pz = cz + sx * hx * uz + sz * hz * vz
                hi = max(hi, math.hypot(px, pz))
        # nearest point: put the axis in the box frame and clamp
        ax = -(cx * ux + cz * uz)
        az = -(cx * vx + cz * vz)
        dx = max(abs(ax) - hx, 0.0)
        dz = max(abs(az) - hz, 0.0)
        lo = min(lo, math.hypot(dx, dz))
    return lo, hi


def _contiguous_runs(degs: list[int]) -> list[tuple[int, int]]:
    """Group a sorted degree list into wrap-aware contiguous runs."""
    if not degs:
        return []
    runs, start, prev = [], degs[0], degs[0]
    for x in degs[1:]:
        if x != prev + 1:
            runs.append((start, prev))
            start = x
        prev = x
    runs.append((start, prev))
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == 359:
        runs[0] = (runs[-1][0], runs[0][1])
        runs.pop()
    return runs


def check_clearance(m: mujoco.MjModel, d: mujoco.MjData, fail: list[str]) -> None:
    """Sweep both arms through ARM_RANGE against everything they could hit.

    Parent/child pairs are skipped by MuJoCo's own collision filtering and by
    Isaac's default self_collision=False, so this walks the pairs directly.
    """
    print("\n=== clearance sweep (arms through ARM_RANGE) ===")
    lo, hi = X.ARM_RANGE
    arm_q = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in X.JOINT_ARMS]
    drum_q = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in X.JOINT_DRUMS]

    moving = _subtree_geoms(m, "arm_front_body") + _subtree_geoms(m, "arm_rear_body")
    wheels = [g for b in X.BODY_WHEELS for g in _body_geoms(m, b)]
    frame = _body_geoms(m, X.BODY_CHASSIS)

    def skip(g1: str, g2: str) -> bool:
        # The arm pivot sits INSIDE the mast that carries it, so the root of
        # each boom overlaps its own mast at every angle. That is how a pivot
        # is built, not a fault, and it is the only overlap in the machine
        # that is intentional.
        return g2.startswith("mast_") and g1 == f"arm_{g2[5:]}_boom"

    for label, fixed in (("wheels", wheels), ("frame", frame)):
        worst = (DISTMAX, "", "", 0.0)
        for angle in np.linspace(lo, hi, 40):
            for drum_phase in np.linspace(0.0, X.POCKET_ARC, 6):
                d.qpos[:] = m.qpos0
                for a in arm_q:
                    d.qpos[a] = angle
                for b in drum_q:
                    d.qpos[b] = drum_phase
                mujoco.mj_forward(m, d)
                for g1 in moving:
                    n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1)
                    for g2 in fixed:
                        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2)
                        if skip(n1, n2):
                            continue
                        dist = mujoco.mj_geomDistance(m, d, g1, g2, DISTMAX, None)
                        if dist < worst[0]:
                            worst = (dist, n1, n2, angle)
        print(f"  vs {label:6s} min gap {worst[0]:+.4f} m  "
              f"({worst[1]} vs {worst[2]}, arm {worst[3]:+.3f} rad)")
        if worst[0] < MIN_CLEARANCE:
            fail.append(f"swept clearance to the {label} is {worst[0]:.4f} m "
                        f"< {MIN_CLEARANCE} ({worst[1]} vs {worst[2]})")

    # The yoke straddles the drum, so it is the one pair that is NOT filtered
    # out by geometry: check it explicitly, at every drum phase.
    print("\n=== yoke vs drum (parent/child: no simulator will report this) ===")
    for side in ("front", "rear"):
        yoke = _body_geoms(m, f"arm_{side}_body")
        drum = _body_geoms(m, f"drum_{side}_body")
        w = (DISTMAX, "", "")
        for angle in (lo, 0.0, hi):
            for phase in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
                d.qpos[:] = m.qpos0
                for a in arm_q:
                    d.qpos[a] = angle
                for b in drum_q:
                    d.qpos[b] = phase
                mujoco.mj_forward(m, d)
                for g1 in yoke:
                    for g2 in drum:
                        dist = mujoco.mj_geomDistance(m, d, g1, g2, DISTMAX, None)
                        if dist < w[0]:
                            w = (dist,
                                 mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1),
                                 mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2))
        print(f"  {side}: min gap {w[0]:+.4f} m  ({w[1]} vs {w[2]})")
        if w[0] < 0.0:
            fail.append(f"{side} yoke passes THROUGH the drum by {-w[0]:.4f} m "
                        f"({w[1]} vs {w[2]})")


# Kept in step by hand with the actuators in excavator_cfg.py and the soil in
# excavate/excavate_env_cfg.py. Duplicated rather than imported because both of
# those pull in Isaac Lab, and the point of this script is to run without it.
ARM_EFFORT_LIMIT = 800.0        # N-m, excavator_cfg.py "arms"
SOIL_DENSITY = 1800.0           # kg/m3
EARTH_G, LUNAR_G = 9.81, 1.62


def check_arm_load(m: mujoco.MjModel, fail: list[str]) -> None:
    """Static hold torque at the arm pivot, empty and with both drums full.

    Measured off the model's own mass table rather than quoted, because the
    arm actuator is sized from these numbers and the drum mass moves whenever
    the blades do.
    """
    print("\n=== arm hold torque (worst case: boom horizontal) ===")
    pivot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_front_body")

    # Moment of the arm subtree about the pivot's y axis, at qpos0 where the
    # boom lies along +x, so the lever arm is just the x offset.
    moment = 0.0
    for bid in range(1, m.nbody):
        b, chain = bid, False
        while b > 0:
            if b == pivot:
                chain = True
                break
            b = m.body_parentid[b]
        if chain:
            com = np.zeros(3)
            mujoco.mju_rotVecQuat(com, m.body_ipos[bid], m.body_iquat[bid])
            # body_pos is relative to the parent; walk up to the pivot.
            off, b = com + m.body_pos[bid], m.body_parentid[bid]
            while b != pivot and b > 0:
                off = off + m.body_pos[b]
                b = m.body_parentid[b]
            moment += m.body_mass[bid] * off[0]

    bore_vol = math.pi * X.ROTOR_TIP_R ** 2 * (2.0 * X.ROTOR_HALF_LEN)
    soil = bore_vol * SOIL_DENSITY
    full = moment + soil * X.ARM_LEN
    print(f"  arm + drum moment about the pivot {moment:6.2f} kg-m")
    print(f"  the rotor sweeps {soil:6.1f} kg of regolith per drum")
    for label, g in (("earth", EARTH_G), ("lunar", LUNAR_G)):
        print(f"  {label}: empty {moment * g:7.1f} N-m   full {full * g:7.1f} N-m")
    if full * EARTH_G > ARM_EFFORT_LIMIT:
        print(f"  note: a full drum needs {full * EARTH_G:.0f} N-m on EARTH, over the "
              f"{ARM_EFFORT_LIMIT:.0f} N-m limit. Fine -- this machine is lunar.")
    if full * LUNAR_G > ARM_EFFORT_LIMIT:
        fail.append(f"holding a full drum needs {full * LUNAR_G:.0f} N-m at lunar "
                    f"gravity, over the {ARM_EFFORT_LIMIT:.0f} N-m arm actuator limit")


def check_reach(fail: list[str]) -> None:
    print("\n=== working envelope ===")
    for k, v in X.reach().items():
        print(f"  {k:30s} {v:+.4f}")


def main() -> int:
    path = X.write_mjcf()
    print(f"MJCF: {path}")
    print(f"mujoco {mujoco.__version__} (clearance checks need {MIN_MUJOCO}+)")
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)

    fail: list[str] = []
    skipped: list[str] = []
    check_masses(m, fail)
    check_arm_load(m, fail)
    check_reach(fail)
    check_rotor_envelope(m, fail)
    check_shroud_continuity(fail)
    check_inlet(m, d, fail)
    check_vane_gaps(m, d, fail)
    if distance_works(m, d):
        check_clearance(m, d, fail)
    else:
        skipped += ["arm/drum clearance sweep"]

    print("\n=== result ===")
    for sk in skipped:
        print(f"  SKIPPED: {sk} (needs MuJoCo {MIN_MUJOCO}+)")
    if fail:
        for f in fail:
            print("  FAILED:", f)
        return 1
    print("  all checks passed" if not skipped else
          "  everything that could be checked passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
