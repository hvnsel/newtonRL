# scripts/check_excavator.py
#
# Offline geometry check for the excavator MJCF. Needs MuJoCo, NOT Isaac Lab:
#
#   pip install mujoco
#   python scripts/check_excavator.py
#
# Everything about this machine that can be wrong QUIETLY is checked here,
# because the next place these numbers show up is a converted USD inside a
# physics solver where a 2 cm interpenetration looks like an unstable policy.
#
#   mass table        a body with no inertia NaNs the solver
#   swept envelopes   what the blades actually reach, which is NOT
#                     SCOOP_TIP_R -- that is a centreline control point, and a
#                     box of finite thickness reaches past it
#   shell continuity  adjacent plates must overlap; a gap is a hole MPM
#                     particles leak through
#   cavity probe      ray-cast around the drum axis to count the mouths and
#                     confirm where they sit. This is the check that catches a
#                     boom or a yoke sitting inside the cavity we are trying to
#                     fill with soil.
#   clearance sweep   arm swept through ARM_RANGE against the wheels and frame.
#                     MuJoCo never tests a body against its own parent and
#                     Isaac articulations default to self-collision off, so
#                     NOTHING in either simulator will report an arm passing
#                     through a drum. This does.
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


def check_drum_envelope(m: mujoco.MjModel, fail: list[str]) -> None:
    """Swept radii of the drum's parts, measured off the actual geoms."""
    print("\n=== drum envelope (radius from the spin axis, metres) ===")
    groups: dict[str, list[float]] = {"shell": [], "lip": [], "cap": []}
    for gid in _body_geoms(m, "drum_front_body"):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
        key = "lip" if "lip" in name else "cap" if "cap" in name else "shell"
        pts = _corners(m, gid)
        groups[key] += list(np.hypot(pts[:, 0], pts[:, 2]))
    for key, radii in groups.items():
        if radii:
            print(f"  {key:6s} {min(radii):.4f} .. {max(radii):.4f}")

    blade_lo, blade_hi = min(groups["lip"]), max(groups["lip"])
    bore = X.DRUM_RADIUS - X.DRUM_WALL_T
    print(f"  bore (fill is counted inside this)   {bore:.4f}")
    print(f"  lip standing proud of the shell      {blade_hi - X.DRUM_RADIUS:+.4f}")
    print(f"  vane reaching into the cavity        {bore - blade_lo:+.4f}")

    inside, outside = X.scoop_mouth_coverage()
    print(f"  mouth covered from inside (curl)       {inside * 100:.0f}%")
    print(f"  mouth covered from outside (hood)      {outside * 100:.0f}%")

    if blade_hi <= X.DRUM_RADIUS:
        fail.append("lips do not stand proud of the shell -- nothing bites first")
    if blade_lo >= bore:
        fail.append(
            f"lips stop at r={blade_lo:.3f}, outside the bore at {bore:.3f}: they cut "
            "but nothing lifts captured soil, so it falls straight back out"
        )
    if inside < 0.25:
        fail.append(
            f"the curl floors only {inside * 100:.0f}% of its mouth, so the pocket is open "
            "to the same hole the soil came in through and empties through it half a turn "
            "later. Lower SCOOP_ENTRY or raise SCOOP_WRAP"
        )
    if outside < 0.25:
        fail.append(
            f"the hood lids only {outside * 100:.0f}% of its mouth: the load can lift "
            "straight back out radially. Raise SCOOP_HOOD, and check SCOOP_HOOD_DIR is +1 "
            "-- at -1 the hood spirals away from the mouth instead of folding over it"
        )


def check_shell_continuity(fail: list[str]) -> None:
    """Adjacent shell plates must overlap at the corners."""
    print("\n=== shell continuity ===")
    step = 2.0 * math.pi / X.DRUM_FACETS
    mid_r = X.DRUM_RADIUS - 0.5 * X.DRUM_WALL_T
    half_arc = mid_r * math.tan(0.5 * step)
    need = mid_r * math.sin(0.5 * step)      # half the chord: a butt joint
    print(f"  plate half-arc {half_arc:.5f} vs half-chord {need:.5f} "
          f"-> overlap {half_arc - need:+.5f} m per joint")
    if half_arc < need:
        fail.append("shell plates leave a gap; MPM particles will leak out of the drum")


def check_cavity(m: mujoco.MjModel, d: mujoco.MjData, fail: list[str]) -> None:
    """Ray-cast around the drum axis and report what the cavity actually looks
    like. This is the check that caught the boom sitting inside the drum."""
    print("\n=== drum cavity probe (front drum, 1 deg rays from the axis outward) ===")

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drum_front_body")
    origin = d.xpos[bid].copy()
    rot = d.xmat[bid].reshape(3, 3)
    drum_geoms = _body_geoms(m, "drum_front_body")

    def sweep(subset: set[int]) -> list[float]:
        """First hit radius per degree, considering only `subset`."""
        saved = m.geom_group.copy()
        for gid in range(m.ngeom):
            m.geom_group[gid] = 0 if gid in subset else 3
        mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        out = []
        for deg in range(360):
            phi = math.radians(deg)
            vec = rot @ np.array([math.cos(phi), 0.0, math.sin(phi)])
            out.append(mujoco.mj_ray(m, d, origin, vec, mask, 0, -1, np.zeros(1, dtype=np.int32)))
        m.geom_group[:] = saved
        return out

    def runs_of(degs: list[int]) -> list[list[int]]:
        runs: list[list[int]] = []
        for deg in degs:
            if runs and deg == runs[-1][-1] + 1:
                runs[-1].append(deg)
            else:
                runs.append([deg])
        if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == 359:
            runs[0] = runs.pop() + runs[0]          # wrap
        return runs

    def is_open(dist: float) -> bool:
        return dist < 0.0 or dist > X.DRUM_RADIUS + 1e-4

    # A mouth is a gap in the SHELL. Probing against every geom instead would
    # count the lip curling across its own mouth as if it closed it, which is
    # exactly backwards: that overlap is the retention, not an obstruction.
    shell = {g for g in drum_geoms
             if "lip" not in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)}
    mouths = runs_of([i for i, r in enumerate(sweep(shell)) if is_open(r)])

    print(f"  {len(mouths)} mouth(s) in the shell, expected SCOOP_COUNT = {X.SCOOP_COUNT}")
    centres = []
    for r in mouths:
        span = [a - 360 if a > 180 and r[0] > r[-1] else a for a in r]
        centres.append((sum(span) / len(span)) % 360)
        print(f"    centred {centres[-1]:6.1f} deg, {len(r)} deg wide")
    if len(mouths) != X.SCOOP_COUNT:
        fail.append(f"cavity probe found {len(mouths)} mouths, not SCOOP_COUNT={X.SCOOP_COUNT}")
    if len(centres) == 2:
        sep = abs((centres[0] - centres[1]) % 360)
        sep = min(sep, 360 - sep)
        print(f"    separation {sep:.1f} deg")
        if abs(sep - 180.0) > 2.0:
            fail.append(f"the two mouths are {sep:.1f} deg apart, not 180")
    aperture = sum(len(r) for r in mouths)
    print(f"  open shell arc {aperture} deg of 360 ({aperture / 3.6:.0f}%) "
          f"-- DRUM_FACETS is the knob if the drum turns out to be intake-limited")

    # Everything, including the lips: what is left is the clear cavity.
    profile = sweep(set(drum_geoms))
    clear = runs_of([i for i, r in enumerate(profile) if is_open(r)])
    print(f"  unobstructed line to the axis over {sum(len(r) for r in clear)} deg")
    inside = [r for r in profile if 0.0 < r < X.DRUM_RADIUS]
    if inside:
        print(f"  closest structure on any ray: {min(inside):.4f} m from the axis")
    if not inside:
        fail.append("nothing reaches inside the shell: the drum has no lifters, "
                    "so captured soil falls out the next time a mouth swings low")

    # The lip is a chain of boxes approximating a curve. If consecutive boxes
    # stop overlapping, every joint becomes a notch soil escapes through.
    segs = X._scoop_segments(0.0)
    worst = math.inf
    for (x0, z0, h0, _), (x1, z1, h1, _) in zip(segs, segs[1:]):
        worst = min(worst, (h0 + h1) - math.hypot(x1 - x0, z1 - z0))
    print(f"  lip segments overlap by {worst:+.4f} m at the tightest joint")
    if worst < 0.0:
        fail.append(f"lip segments leave a {-worst:.4f} m notch at a joint")


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
            for drum_phase in np.linspace(0.0, 2.0 * math.pi / X.DRUM_FACETS, 6):
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

    bore_vol = math.pi * (X.DRUM_RADIUS - X.DRUM_WALL_T) ** 2 * (2.0 * X.DRUM_HALF_LEN)
    soil = bore_vol * SOIL_DENSITY
    full = moment + soil * X.ARM_LEN
    print(f"  arm + drum moment about the pivot {moment:6.2f} kg-m")
    print(f"  bore holds {soil:6.1f} kg of regolith per drum")
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
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)

    fail: list[str] = []
    check_masses(m, fail)
    check_arm_load(m, fail)
    check_reach(fail)
    check_drum_envelope(m, fail)
    check_shell_continuity(fail)
    check_cavity(m, d, fail)
    check_clearance(m, d, fail)

    print("\n=== result ===")
    if fail:
        for f in fail:
            print("  FAILED:", f)
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
