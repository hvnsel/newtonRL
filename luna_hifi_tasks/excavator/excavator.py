# excavator.py
#
# The bucket-drum excavator, as an MJCF string. Four driven wheels on a
# rectangular frame, and a counter-rotating excavation drum on a pitching arm
# at each end.
#
# This file is NOT read at training time. Training loads the converted USD in
# assets/excavator/, and this MJCF is its source: re-run the converter after
# every change here.
#
# Plan view (+x forward, +z up). A,B are the front corners, C,D the rear:
#
#                        front drum (digs forward of the machine)
#                            ___
#                           (___)
#                             |  <- front arm, pitches in the x-z plane
#            A o=============[#]=============o B     <- front crossmember
#              ||                           ||
#              ||      [    deck    ]       ||       <- side rails
#              ||                           ||
#            C o=============[#]=============o D     <- rear crossmember
#                             |  <- rear arm
#                           (___)
#                            ---
#                        rear drum (digs behind the machine)
#
# Elevation (+x right, +z up), arms at full down:
#
#              .-.                                     .-.
#             ( o )  <- drum, cavity open at the mouths( o )
#              '-'   \                               /  '-'
#                     \___[mast]===[deck]===[mast]___/
#                      (O)                       (O)     <- wheels
#           ~~~~~~~~~~~~~~~~~ soil ~~~~~~~~~~~~~~~~~~~~
#
# Layout
# ------
# * Skid steer, no steering joints. Four hinges, left pair and right pair
#   commanded together.
# * The rear assembly is the front assembly rotated 180 degrees about z, which
#   points the rear drum's spin axis along world -y, so identical joint
#   commands counter-rotate the two drums. Both dig at +1 and their horizontal
#   digging reactions cancel through the frame, which is what lets a light
#   machine excavate without its own weight in traction.
# * The drums are hollow: a ring of box segments closed at both ends by cap
#   discs, with SCOOP_COUNT segments left open as mouths. Each mouth carries a
#   PAIR of thin curved lips, rooted at opposite edges and overlapping across
#   the gap at different radii -- one climbing outside the shell circle, one
#   dropping inside it. The channel between them is the only way in or out,
#   0.094 m wide.
#
#   A joint velocity of sign k turns the drum toward -k*phi, so relative to the
#   drum the soil streams toward +k*phi. The lip tip cuts, its concave face
#   deflects the cut toward the axis, and the curl roofs the mouth; leaving
#   means climbing the ramp against the stream. SCOOP_CURL picks k and
#   DIG_DRUM_SIGN is derived from it.
#
#   The pair makes the drum a one-way valve. At DIG_DRUM_SIGN soil travels the
#   channel inward; the other way round the same channel runs backwards and
#   dumps the load. A single lip leaves a gap on either side of itself, both
#   ways out, and neither wide enough to be a way in once the MPM coupler has
#   taken half a voxel off each side -- the single-lip version pinched to
#   0.009 m and sealed the drum.
# * The chassis outweighs both drums 6.6:1 empty. The bore holds ~155 kg of
#   regolith per drum, so a loaded machine carries ~63% of its 492 kg dry mass
#   out on the arms, and arm hold torque goes from 351 N-m empty to 1387 N-m
#   full at earth gravity (58 and 229 N-m at lunar). scripts/check_excavator.py
#   reads these off the model's own mass table.
#
# Wheels are cylinders with grousers: a smooth wheel shears granular surface
# and spins in place. WHEEL_GROUSERS = 0 falls back to bare cylinders.
#
# Every body carrying a joint has a geom with mass, so MuJoCo derives real
# inertia for all of them. A body with zero inertia produces NaNs in the solver.
#
# Frame convention: +x forward, +z up. Wheel centres sit at z = 0 in the
# chassis frame, so the chassis origin is one wheel radius above the ground.

from __future__ import annotations

import math
import os
import tempfile

# ---------------------------------------------------------------------------
# Dimensions (metres, radians)
# ---------------------------------------------------------------------------

# --- running gear ---
WHEEL_RADIUS = 0.30
WHEEL_HALF_W = 0.10             # 20 cm wide tyre
WHEELBASE = 1.40                # front axle to rear axle, along x
TRACK = 1.15                    # left wheel centre to right wheel centre, along y

AXLE_X = 0.5 * WHEELBASE        # +/- 0.70
WHEEL_Y = 0.5 * TRACK           # +/- 0.575

# The arm pivot sits ahead of the axle. The drum ends and the yoke legs both
# pass through the wheel band in y, so separation in x is the only thing
# keeping them off the front wheels at full dig. Longer lips push the yoke's
# cross piece back down the boom to clear the blades' swept circle, dragging
# the legs inboard, so this grows with them: swept clearance to the wheels is
# 0.069 m here, and the binding pair is the cross piece against a rear grouser
# at full dig.
MAST_OFFSET_X = 0.22
PIVOT_X = AXLE_X + MAST_OFFSET_X

CHASSIS_Z = WHEEL_RADIUS        # chassis origin height with wheels on z = 0

WHEEL_GROUSERS = 6              # radial cleats per wheel; 0 for a bare cylinder
GROUSER_PROTRUDE = 0.025        # how far a cleat stands off the tyre
GROUSER_HALF_T = 0.012

# --- frame ---
RAIL_Y = 0.45                   # side rails, inboard of the wheels
RAIL_HALF_W = 0.05
RAIL_HALF_H = 0.06
DECK_Z = -0.10                  # deck sits below the axle line: low CG, on purpose
DECK_HALF = (0.55, 0.40, 0.04)

# --- mast: the tower on each crossmember that carries the arm pivot ---
MAST_TOP_Z = 0.20               # arm pivot height in the chassis frame
MAST_HALF = (0.05, 0.09, 0.13)  # x half-size is extended to span MAST_OFFSET_X

# --- arm ---
# The arm is a yoke: a centre boom out to the drum axis would sit inside the
# cavity and block the mouths, so the boom stops short and two legs pass
# outboard of the end caps to pick up the axle.
ARM_LEN = 0.76                  # pivot to drum axis
ARM_HALF_H = 0.05
ARM_HALF_W = 0.07
YOKE_HALF_W = 0.04
YOKE_HALF_H = 0.05
# Gap between the outer face of an end cap and the inner face of a leg. The cap
# is a disc of half-thickness CAP_HALF_T centred at DRUM_HALF_LEN + CAP_HALF_T,
# so it reaches DRUM_HALF_LEN + 2*CAP_HALF_T.
YOKE_CLEARANCE = 0.005
# Positive arm angle pitches the boom down, toward the soil.
#
# The down limit tracks ARM_LEN: a longer boom reaches deeper at the same
# angle, and past about one drum radius of burial the mouths fight the whole
# overburden and stall. 0.80 rad keeps the cut at ~0.19 m, just under
# DRUM_RADIUS.
ARM_RANGE = (-0.55, 0.80)       # -31.5 deg (stowed high) .. +45.8 deg (full dig)

# --- drum ---
DRUM_RADIUS = 0.20
# 2*(DRUM_HALF_LEN + 2*CAP_HALF_T) = 1.00 m against a 1.15 m track: an almost
# machine-wide swath that still stops short of the wheels.
DRUM_HALF_LEN = 0.475           # ~100 cm wide drum
DRUM_WALL_T = 0.030             # shell thickness; see the MPM note below
# Nine 40-degree facets: three mouths land 120 degrees apart at slots 0, 3 and
# 6 and six shell plates fill the rest. Shell corners sit at 0.211 against a
# 0.20 nominal radius.
DRUM_FACETS = 9                 # angular slots around the circumference
SCOOP_COUNT = 3                 # mouths, at slots 0, 3 and 6

# --- the lips ---
#
# Each mouth gets a PAIR, rooted at opposite edges and overlapping each other
# across the gap at different radii. Take the mouth at 12 o'clock; "right" is
# clockwise from it and "left" anticlockwise:
#
#            OUTER lip: rooted at the RIGHT edge, climbing
#            OUTSIDE the shell circle and reaching LEFT
#                   ___________
#                  /           \___  tip, overhanging past the left edge
#         ________/
#    shell        |   G A P   |        shell
#         ________             \_____________
#                    tip       |
#               overhanging  INNER lip: rooted at the LEFT edge, dropping
#               past the     INSIDE the shell circle and reaching RIGHT
#               right edge
#
# The way in is the channel between them, and its width is set by how far apart
# the two lips run. The pair makes the drum a one-way valve:
#
#   loading   soil streams along the channel from the outer tip, past the
#             inner tip, and into the drum
#   dumping   reverse the drum and the same channel runs backwards, the inner
#             lip lifting the load out past the outer one
#
# Leaving under load means climbing back over the inner lip against the
# stream. SCOOP_CURL sets which rotation is which and DIG_DRUM_SIGN follows.
SCOOP_CURL = 1.0

# Every gap here is floored by the solver, not the steel. The MPM coupler
# inflates every collider by half a voxel per side, so a geometric gap of g
# reads as g - voxel to the particles: at a 0.05 m voxel a 4 cm slot is a wall.
# check_excavator.py measures every passage against these.
MPM_TARGET_VOXEL = 0.05
MPM_CLEARANCE = MPM_TARGET_VOXEL        # below this a passage is simply closed
SCOOP_GAP = 1.6 * MPM_TARGET_VOXEL      # what it takes to flow, not merely to open

# The channel is the difference between these two, so they are the passage
# width: keep them at least SCOOP_GAP apart over the span where the lips
# overlap.
#
# The outer lip is also the first thing to reach the ground, and a lip buried
# in the floor cannot turn. 0.29 projects 0.098 m past the shell, so a 0.21 m
# bed takes a 0.112 m cut before it touches; 0.34 projected 0.148 m and drove
# through the bed floor at a 0.10 m cut.
SCOOP_OUTER_TIP_R = 0.29        # outer lip tip, proud of the 0.20 shell
SCOOP_INNER_TIP_R = 0.04        # inner lip tip, well inside the 0.17 bore
# How far round each lip reaches, in MOUTH WIDTHS. Above 1.0 the lip overhangs
# past the far edge of its own gap, which is the overlap that closes the mouth
# to anything trying to leave radially.
SCOOP_OUTER_SPAN = 1.15
SCOOP_INNER_SPAN = 1.15
# Radius goes as a power of the distance along the lip. 1.0 is a straight
# spiral: radius rising evenly with angle, constant curvature, no corner. An
# exponent below 1 leaves the root almost radially and then runs flat, putting
# a re-entrant elbow where the lip meets the shell that granular material packs
# into and stops. A gentler curve also narrows the channel, so the tips move
# apart to pay for it.
SCOOP_OUTER_RISE = 1.0
SCOOP_INNER_DROP = 1.0
SCOOP_SEGMENTS = 5              # straight boxes per lip

# A lip is a cutting edge, not structure. At a 5 cm voxel the thickness is
# nearly invisible to the soil -- 12 mm and 24 mm of plate both read as roughly
# 6 cm once the coupler has inflated them -- and starts to matter at 2.5-3 cm.
BLADE_HALF_T = 0.006
CAP_HALF_T = 0.012

# At a 5 cm voxel the coupler inflates a 3 cm wall to ~8 cm, so particles do
# not tunnel, and it carries the 1.2 cm lips too. The pocket under the curl is
# about 10 cm deep -- two cells -- and the slot into it is narrower still, so
# the drum reads better at a 2.5-3 cm voxel. This is where MPM resolution
# binds.

# --- masses (kg) ---
RAIL_MASS = 35.0                # each
CROSSMEMBER_MASS = 30.0         # each
DECK_MASS = 170.0               # the ballast: keeps the CG low and central
MAST_MASS = 20.0                # each
WHEEL_MASS = 12.0               # cylinder only
GROUSER_MASS = 0.35             # each
ARM_BOOM_MASS = 6.0             # each arm: boom + cross + two legs = 17.5 kg
ARM_CROSS_MASS = 3.5
ARM_LEG_MASS = 4.0
# Per segment, so it tracks DRUM_FACETS. 2.9 keeps the shell near 18 kg.
DRUM_SEGMENT_MASS = 2.9
DRUM_CAP_MASS = 1.2             # each end cap
# Total for one scoop lip, split between its segments by arc length.
DRUM_BLADE_MASS = 1.50

# --- friction ---
FRAME_FRICTION = "0.8 0.005 0.0001"
WHEEL_FRICTION = "1.4 0.010 0.0010"
DRUM_FRICTION = "1.0 0.010 0.0010"

# ---------------------------------------------------------------------------
# Names. Exported so the env cfg and the env never spell a string twice.
# ---------------------------------------------------------------------------

JOINT_ROOT = "root"
JOINT_WHEELS_LEFT = ["wheel_fl", "wheel_rl"]
JOINT_WHEELS_RIGHT = ["wheel_fr", "wheel_rr"]
JOINT_WHEELS = JOINT_WHEELS_LEFT + JOINT_WHEELS_RIGHT
JOINT_ARMS = ["arm_front", "arm_rear"]
JOINT_DRUMS = ["drum_front", "drum_rear"]

BODY_CHASSIS = "chassis"
BODY_WHEELS = ["wheel_fl_body", "wheel_fr_body", "wheel_rl_body", "wheel_rr_body"]
BODY_ARMS = ["arm_front_body", "arm_rear_body"]
BODY_DRUMS = ["drum_front_body", "drum_rear_body"]

# The bodies that must appear in the MPM CouplerProxyMappingCfg. Wheels for
# traction, drums for excavation. The arms are deliberately absent: they only
# reach the soil through the drum, and every extra proxy body costs the coupler.
BODY_SOIL_CONTACT = BODY_WHEELS + BODY_DRUMS

FRAME_RGBA = "0.78 0.62 0.16 1"     # machine yellow
DECK_RGBA = "0.42 0.45 0.50 1"
WHEEL_RGBA = "0.13 0.13 0.14 1"
GROUSER_RGBA = "0.22 0.22 0.24 1"
ARM_RGBA = "0.70 0.55 0.14 1"
DRUM_RGBA = "0.55 0.57 0.60 1"
BLADE_RGBA = "0.82 0.84 0.86 1"

# ---------------------------------------------------------------------------
# Geometry helpers
#
# All of the rotational geometry below is built about the body's local y axis,
# because every spinning part in this machine (wheels, drums) turns about y.
#
# MuJoCo's euler="0 b 0" rotates about +y, mapping local z_hat to
# (sin b, 0, cos b). So to aim a box's thickness axis along the direction
# psi in the local x-z plane, set b = pi/2 - psi. Two cases follow from that:
#
#   tangential plate at angle phi (thickness points radially):  b = pi/2 - phi
#   radial blade at angle phi, raked by rho:                    b = rho - phi
# ---------------------------------------------------------------------------


def _tangential_plate(
    name: str,
    phi: float,
    radius: float,
    half_arc: float,
    half_len: float,
    half_t: float,
    mass: float,
    rgba: str,
    friction: str,
) -> str:
    """A shell segment lying tangent to a circle of `radius` at angle `phi`."""
    return (
        f'<geom name="{name}" type="box" '
        f'pos="{radius * math.cos(phi):.5f} 0 {radius * math.sin(phi):.5f}" '
        f'euler="0 {0.5 * math.pi - phi:.6f} 0" '
        f'size="{half_arc:.5f} {half_len:.5f} {half_t:.5f}" '
        f'mass="{mass}" rgba="{rgba}" friction="{friction}"/>'
    )


def _radial_blade(
    name: str,
    phi: float,
    inner_r: float,
    outer_r: float,
    half_len: float,
    half_t: float,
    rake: float,
    mass: float,
    rgba: str,
    friction: str,
) -> str:
    """A blade standing out from the axis at angle `phi`, raked `rake` off radial."""
    mid_r = 0.5 * (inner_r + outer_r)
    return (
        f'<geom name="{name}" type="box" '
        f'pos="{mid_r * math.cos(phi):.5f} 0 {mid_r * math.sin(phi):.5f}" '
        f'euler="0 {rake - phi:.6f} 0" '
        f'size="{0.5 * (outer_r - inner_r):.5f} {half_len:.5f} {half_t:.5f}" '
        f'mass="{mass}" rgba="{rgba}" friction="{friction}"/>'
    )


BORE_R = DRUM_RADIUS - DRUM_WALL_T


def _lip_profile(outer: bool) -> tuple[float, float, float, float]:
    """(span in mouth widths, root radius, tip radius, radius exponent)."""
    if outer:
        return SCOOP_OUTER_SPAN, DRUM_RADIUS, SCOOP_OUTER_TIP_R, SCOOP_OUTER_RISE
    # The inner lip is welded to the INNER face of the shell, at the bore, not
    # to its outer skin. That is both what the part physically is and worth
    # 30 mm of channel: the narrowest point of the channel is where the outer
    # lip passes over this root, so every millimetre the root sits lower is a
    # millimetre the outer lip does not have to stand proud to make up.
    return SCOOP_INNER_SPAN, BORE_R, SCOOP_INNER_TIP_R, SCOOP_INNER_DROP


def _lip_polar(v: float, outer: bool) -> tuple[float, float]:
    """(angle offset from the lip's ROOT, radius) a fraction v along it.

    Offsets are signed so that both lips are described in one frame: the outer
    lip is rooted at the mouth's -CURL edge and runs toward +CURL, the inner
    lip is rooted at the +CURL edge and runs toward -CURL. They therefore
    travel TOWARDS each other and overlap across the middle of the gap, one
    outside the shell circle and one inside it.
    """
    step = 2.0 * math.pi / DRUM_FACETS
    span, r0, r1, power = _lip_profile(outer)
    direction = SCOOP_CURL if outer else -SCOOP_CURL
    return direction * span * step * v, r0 + (r1 - r0) * v ** power


def _lip_root_angle(phi_mouth: float, outer: bool) -> float:
    """Where a lip is welded to the shell: the outer one at the mouth's -CURL
    edge, the inner one at its +CURL edge."""
    step = 2.0 * math.pi / DRUM_FACETS
    return phi_mouth + (-SCOOP_CURL if outer else SCOOP_CURL) * 0.5 * step


def _lip_point(phi_mouth: float, v: float, outer: bool) -> tuple[float, float]:
    d, r = _lip_polar(v, outer)
    a = _lip_root_angle(phi_mouth, outer) + d
    return r * math.cos(a), r * math.sin(a)


def _lip_curve(phi_mouth: float, outer: bool) -> list[tuple[float, float]]:
    """(x, z) control points of one lip, root first, spaced by ARC LENGTH.

    Arc length rather than angle because the radius exponent is well below 1:
    the lip leaves its root almost radially and then runs nearly concentric, so
    angular spacing would put one box across the steep part -- the part that
    opens the channel -- and string the rest along the flat tail.
    """
    fine = 200
    pts = [_lip_point(phi_mouth, i / fine, outer) for i in range(fine + 1)]
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + math.dist(a, b))

    out, j = [], 0
    for k in range(SCOOP_SEGMENTS + 1):
        target = cum[-1] * k / SCOOP_SEGMENTS
        while j < fine - 1 and cum[j + 1] < target:
            j += 1
        span = cum[j + 1] - cum[j]
        f = 0.0 if span <= 0.0 else min((target - cum[j]) / span, 1.0)
        (x0, z0), (x1, z1) = pts[j], pts[j + 1]
        out.append((x0 + f * (x1 - x0), z0 + f * (z1 - z0)))
    return out


def _lip_segments(phi_mouth: float, outer: bool) -> list[tuple[float, float, float, float]]:
    """The lip as boxes: (centre_x, centre_z, half_length, euler_y) each.

    One function, used both to write the XML and to measure what the lip
    sweeps, because those two answers drifting apart is how the yoke once ended
    up inside the blades.

    Segments are lengthened by BLADE_HALF_T at each end so consecutive boxes
    overlap at the corners rather than butting. A curve built from butted
    chords leaves a notch at every joint on the convex side, and a notch in a
    lip is a hole soil escapes through.
    """
    pts = _lip_curve(phi_mouth, outer)
    out = []
    for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
        dx, dz = x1 - x0, z1 - z0
        length = math.hypot(dx, dz)
        out.append((0.5 * (x0 + x1), 0.5 * (z0 + z1),
                    0.5 * length + BLADE_HALF_T,
                    math.atan2(-dz, dx)))       # local x -> (cos b, 0, -sin b)
    return out


def _scoop_swept_radii() -> tuple[float, float]:
    """Radii the lips actually sweep, off the box corners.

    Not SCOOP_*_TIP_R: those are centreline control points, and a box of finite
    thickness reaches past them at both ends. The outer figure is what the yoke
    has to clear.
    """
    radii = []
    for outer in (True, False):
        for cx, cz, half, b in _lip_segments(0.0, outer):
            ux, uz = math.cos(b), -math.sin(b)
            nx, nz = math.sin(b), math.cos(b)
            radii += [
                math.hypot(cx + sl * half * ux + st * BLADE_HALF_T * nx,
                           cz + sl * half * uz + st * BLADE_HALF_T * nz)
                for sl in (-1.0, 1.0)
                for st in (-1.0, 1.0)
            ]
    return min(radii), max(radii)


def _lip_xml(prefix: str, phi_mouth: float, outer: bool) -> list[str]:
    """One lip as a chain of thin boxes."""
    segs = _lip_segments(phi_mouth, outer)
    total = sum(half for _, _, half, _ in segs)
    tag = "out" if outer else "in"
    return [
        f'<geom name="{prefix}_{tag}{i}" type="box" '
        f'pos="{cx:.5f} 0 {cz:.5f}" euler="0 {b:.6f} 0" '
        f'size="{half:.5f} {DRUM_HALF_LEN:.5f} {BLADE_HALF_T:.5f}" '
        f'mass="{DRUM_BLADE_MASS * half / total:.4f}" '
        f'rgba="{BLADE_RGBA if outer else DRUM_RGBA}" friction="{DRUM_FRICTION}"/>'
        for i, (cx, cz, half, b) in enumerate(segs)
    ]


def scoop_channel() -> tuple[float, float]:
    """(narrowest, widest) radial width of the channel between the two lips.

    This is the way in and the way out, and the only one. Measured where the
    lips actually overlap in angle, off _lip_polar so it cannot disagree with
    the geometry that gets built -- an earlier version kept its own copy of the
    radius profile and silently went on reporting the old channel after the
    lips were re-rooted.

    Centreline to centreline, so the true opening is this minus 2*BLADE_HALF_T,
    and then minus a whole voxel once the MPM coupler has inflated both sides.
    """
    step = 2.0 * math.pi / DRUM_FACETS
    fine = 200
    # Angular offset of each lip's root from the mouth centre, in the CURL
    # direction, plus its (offset, radius) samples in that same frame.
    def samples(outer: bool):
        root = -0.5 * step if outer else 0.5 * step
        out = []
        for i in range(fine + 1):
            d, r = _lip_polar(i / fine, outer)
            out.append((root + SCOOP_CURL * d * SCOOP_CURL, r))
        return out

    o = [(a, r) for a, r in samples(True)]
    n = [(a, r) for a, r in samples(False)]
    lo, hi = max(o[0][0], n[-1][0]), min(o[-1][0], n[0][0])
    if hi <= lo:
        return 0.0, 0.0                      # the lips do not overlap at all
    widths = []
    for i in range(fine + 1):
        a = lo + (hi - lo) * i / fine
        ro = next((r for aa, r in o if aa >= a), None)
        rn = next((r for aa, r in reversed(n) if aa >= a), None)
        if ro is not None and rn is not None:
            widths.append(ro - rn)
    return (min(widths), max(widths)) if widths else (0.0, 0.0)


def scoop_mouth_coverage() -> tuple[float, float]:
    """(inside, outside) fractions of a mouth's angular span each lip covers.

    A lip only closes the mouth where it has actually left the shell band: the
    outer one has to be clear above DRUM_RADIUS and the inner one clear below
    the bore, or it is still in the wall rather than over or under the hole.
    """
    step = 2.0 * math.pi / DRUM_FACETS

    def covered(outer: bool) -> float:
        span, r0, r1, power = _lip_profile(outer)
        hit = 0
        for i in range(201):
            a = step * (i / 200)             # into the mouth from this lip's root
            v = a / (span * step)
            if v > 1.0:
                continue
            r = r0 + (r1 - r0) * v ** power
            if (r > DRUM_RADIUS) if outer else (r < BORE_R):
                hit += 1
        return hit / 201.0

    return covered(False), covered(True)


def _wheel(name: str, x: float, y: float, joint: str) -> str:
    """A cylinder tyre spinning about y, optionally with radial grousers.

    MuJoCo cylinders run along their local z, so euler="pi/2 0 0" lays the
    axis along y.
    """
    cleats = []
    for i in range(WHEEL_GROUSERS):
        phi = 2.0 * math.pi * i / WHEEL_GROUSERS
        cleats.append(
            _radial_blade(
                f"{name}_grouser{i}",
                phi,
                WHEEL_RADIUS - 0.01,
                WHEEL_RADIUS + GROUSER_PROTRUDE,
                WHEEL_HALF_W,
                GROUSER_HALF_T,
                0.0,
                GROUSER_MASS,
                GROUSER_RGBA,
                WHEEL_FRICTION,
            )
        )
    cleat_xml = "\n        ".join(cleats)
    return f"""<body name="{name}_body" pos="{x:.4f} {y:.4f} 0">
        <joint name="{joint}" type="hinge" axis="0 1 0" armature="0.05"/>
        <geom name="{name}_tyre" type="cylinder" euler="1.570796 0 0"
              size="{WHEEL_RADIUS} {WHEEL_HALF_W}" mass="{WHEEL_MASS}"
              rgba="{WHEEL_RGBA}" friction="{WHEEL_FRICTION}"/>
        {cleat_xml}
      </body>"""


def _drum(prefix: str, joint: str) -> str:
    """A hollow bucket drum: shell segments, open scoop mouths, blades, end caps.

    The mouths are spread as evenly as SCOOP_COUNT divides DRUM_FACETS allows;
    at SCOOP_COUNT = 2 that is slots 0 and 6, exactly opposite.

    Each open slot gets a PAIR of lips, one rooted at either edge, overlapping
    across the gap at different radii: the outer one outside the shell circle,
    the inner one inside it. The channel between them is the only way in or
    out, which is what makes the drum a one-way valve. Which edge is which
    depends on SCOOP_CURL; see the lip block in the dimensions section.
    """
    step = 2.0 * math.pi / DRUM_FACETS
    mid_r = DRUM_RADIUS - 0.5 * DRUM_WALL_T
    # tan, not sin, so adjacent plates overlap slightly at the corners; a gap
    # here is a hole MPM particles leak out of.
    half_arc = mid_r * math.tan(0.5 * step)

    open_slots = {round(i * DRUM_FACETS / SCOOP_COUNT) % DRUM_FACETS for i in range(SCOOP_COUNT)}

    parts = []
    for i in range(DRUM_FACETS):
        phi = i * step
        if i in open_slots:
            # Mouth: no shell plate, and a curved lip anchored at its leading
            # edge that curls back over the opening.
            parts += _lip_xml(f"{prefix}_scoop{i}", phi, outer=True)
            parts += _lip_xml(f"{prefix}_scoop{i}", phi, outer=False)
        else:
            parts.append(
                _tangential_plate(
                    f"{prefix}_shell{i}",
                    phi,
                    mid_r,
                    half_arc,
                    DRUM_HALF_LEN,
                    0.5 * DRUM_WALL_T,
                    DRUM_SEGMENT_MASS,
                    DRUM_RGBA,
                    DRUM_FRICTION,
                )
            )

    for sign, side in ((1.0, "l"), (-1.0, "r")):
        y = sign * (DRUM_HALF_LEN + CAP_HALF_T)
        parts.append(
            f'<geom name="{prefix}_cap_{side}" type="cylinder" euler="1.570796 0 0" '
            f'pos="0 {y:.5f} 0" size="{DRUM_RADIUS} {CAP_HALF_T}" '
            f'mass="{DRUM_CAP_MASS}" rgba="{DRUM_RGBA}" friction="{DRUM_FRICTION}"/>'
        )

    body_xml = "\n          ".join(parts)
    return f"""<body name="{prefix}_body" pos="{ARM_LEN:.4f} 0 0">
          <joint name="{joint}" type="hinge" axis="0 1 0" armature="0.08"/>
          {body_xml}
        </body>"""


# Derived yoke geometry. Needs the drum dimensions, so it lives below them.
YOKE_Y = DRUM_HALF_LEN + 2.0 * CAP_HALF_T + YOKE_CLEARANCE + YOKE_HALF_W

# What the lips actually sweep, as opposed to what SCOOP_TIP_R and
# SCOOP_ROOT_R say. Both numbers are load-bearing elsewhere: the outer one sets
# where the yoke may sit, and the inner one is how far the curl reaches into
# the bore.
BLADE_SWEPT_INNER_R, BLADE_SWEPT_OUTER_R = _scoop_swept_radii()

# Which sign of drum command loads the drum, derived from SCOOP_CURL so the
# two cannot disagree. A joint velocity of sign k turns the drum toward -k*phi,
# so the soil streams toward +k*phi in the drum's frame. Entering runs the
# channel from the outer lip's tip to the inner lip's, which sit on the +CURL
# and -CURL sides, so loading needs k = -SCOOP_CURL. The other sign dumps.
DIG_DRUM_SIGN = -SCOOP_CURL

# The cross piece spans the full width of the machine at the drum's height, so
# it has to clear the circle the BLADES sweep, not the shell's. Deriving it
# from DRUM_RADIUS is what buried it 3.5 cm inside the old lips; with the long
# ones it would have been 5 cm.
YOKE_STANDOFF = 0.025
BOOM_LEN = ARM_LEN - BLADE_SWEPT_OUTER_R - YOKE_HALF_W - YOKE_STANDOFF
_LEG_X0 = BOOM_LEN - YOKE_HALF_W            # legs overlap the cross piece slightly


def _arm_assembly(side: str, yaw: float, arm_joint: str, drum_prefix: str, drum_joint: str) -> str:
    """One arm + drum, hung off the mast at PIVOT_X.

    `yaw` is 0 for the front assembly and pi for the rear. The rear is a true
    180-degree copy, which is what gives the two drums opposite world spin axes
    for the same joint command.

    The arm is boom -> cross piece -> two legs straddling the drum. Nothing
    enters the drum's swept envelope; see the YOKE comment in the dimensions
    block.
    """
    x = PIVOT_X * math.cos(yaw)
    leg_half = 0.5 * (ARM_LEN - _LEG_X0)
    leg_cx = 0.5 * (ARM_LEN + _LEG_X0)
    legs = "\n        ".join(
        f'<geom name="arm_{side}_leg_{tag}" type="box" '
        f'pos="{leg_cx:.4f} {sgn * YOKE_Y:.4f} 0" '
        f'size="{leg_half:.4f} {YOKE_HALF_W} {YOKE_HALF_H}" '
        f'mass="{ARM_LEG_MASS}" rgba="{ARM_RGBA}" friction="{FRAME_FRICTION}"/>'
        for sgn, tag in ((1.0, "l"), (-1.0, "r"))
    )
    return f"""<body name="arm_{side}_body" pos="{x:.4f} 0 {MAST_TOP_Z}" euler="0 0 {yaw:.6f}">
        <joint name="{arm_joint}" type="hinge" axis="0 1 0"
               range="{ARM_RANGE[0]} {ARM_RANGE[1]}" armature="0.10" damping="2.0"/>
        <geom name="arm_{side}_boom" type="box" pos="{0.5 * BOOM_LEN:.4f} 0 0"
              size="{0.5 * BOOM_LEN:.4f} {ARM_HALF_W} {ARM_HALF_H}"
              mass="{ARM_BOOM_MASS}" rgba="{ARM_RGBA}" friction="{FRAME_FRICTION}"/>
        <geom name="arm_{side}_cross" type="box" pos="{BOOM_LEN:.4f} 0 0"
              size="{YOKE_HALF_W} {YOKE_Y + YOKE_HALF_W:.4f} {YOKE_HALF_H}"
              mass="{ARM_CROSS_MASS}" rgba="{ARM_RGBA}" friction="{FRAME_FRICTION}"/>
        {legs}
        {_drum(drum_prefix, drum_joint)}
      </body>"""


# ---------------------------------------------------------------------------
# Frame geoms. All fixed to the chassis body -- the masts are structure, not
# a mechanism, so they carry no joint.
# ---------------------------------------------------------------------------

_FRAME_PARTS = [
    # side rails, running the length of the machine
    f'<geom name="rail_left" type="box" pos="0 {RAIL_Y} 0" '
    f'size="{AXLE_X + RAIL_HALF_W:.4f} {RAIL_HALF_W} {RAIL_HALF_H}" '
    f'mass="{RAIL_MASS}" rgba="{FRAME_RGBA}" friction="{FRAME_FRICTION}"/>',
    f'<geom name="rail_right" type="box" pos="0 {-RAIL_Y} 0" '
    f'size="{AXLE_X + RAIL_HALF_W:.4f} {RAIL_HALF_W} {RAIL_HALF_H}" '
    f'mass="{RAIL_MASS}" rgba="{FRAME_RGBA}" friction="{FRAME_FRICTION}"/>',
    # crossmembers on the axle lines: AB at the front, CD at the rear
    f'<geom name="crossmember_front" type="box" pos="{AXLE_X} 0 0" '
    f'size="{RAIL_HALF_W} {RAIL_Y} {RAIL_HALF_H}" '
    f'mass="{CROSSMEMBER_MASS}" rgba="{FRAME_RGBA}" friction="{FRAME_FRICTION}"/>',
    f'<geom name="crossmember_rear" type="box" pos="{-AXLE_X} 0 0" '
    f'size="{RAIL_HALF_W} {RAIL_Y} {RAIL_HALF_H}" '
    f'mass="{CROSSMEMBER_MASS}" rgba="{FRAME_RGBA}" friction="{FRAME_FRICTION}"/>',
    # deck: slung below the axle line, carries most of the mass
    f'<geom name="deck" type="box" pos="0 0 {DECK_Z}" '
    f'size="{DECK_HALF[0]} {DECK_HALF[1]} {DECK_HALF[2]}" '
    f'mass="{DECK_MASS}" rgba="{DECK_RGBA}" friction="{FRAME_FRICTION}"/>',
    # masts: towers carrying the two arm pivots
    f'<geom name="mast_front" type="box" '
    f'pos="{AXLE_X + 0.5 * MAST_OFFSET_X:.4f} 0 {0.5 * MAST_TOP_Z:.4f}" '
    f'size="{0.5 * MAST_OFFSET_X + MAST_HALF[0]:.4f} {MAST_HALF[1]} {MAST_HALF[2]}" '
    f'mass="{MAST_MASS}" rgba="{FRAME_RGBA}" friction="{FRAME_FRICTION}"/>',
    f'<geom name="mast_rear" type="box" '
    f'pos="{-(AXLE_X + 0.5 * MAST_OFFSET_X):.4f} 0 {0.5 * MAST_TOP_Z:.4f}" '
    f'size="{0.5 * MAST_OFFSET_X + MAST_HALF[0]:.4f} {MAST_HALF[1]} {MAST_HALF[2]}" '
    f'mass="{MAST_MASS}" rgba="{FRAME_RGBA}" friction="{FRAME_FRICTION}"/>',
]

_FRAME_XML = "\n      ".join(_FRAME_PARTS)

_WHEELS_XML = "\n      ".join(
    [
        _wheel("wheel_fl", AXLE_X, WHEEL_Y, "wheel_fl"),
        _wheel("wheel_fr", AXLE_X, -WHEEL_Y, "wheel_fr"),
        _wheel("wheel_rl", -AXLE_X, WHEEL_Y, "wheel_rl"),
        _wheel("wheel_rr", -AXLE_X, -WHEEL_Y, "wheel_rr"),
    ]
)

_ARMS_XML = "\n      ".join(
    [
        _arm_assembly("front", 0.0, "arm_front", "drum_front", "drum_front"),
        _arm_assembly("rear", math.pi, "arm_rear", "drum_rear", "drum_rear"),
    ]
)

# ---------------------------------------------------------------------------
# Model
#
# No <default class="..."> blocks: every geom spells out its own friction and
# rgba, because the MJCF->USD converter drops class inheritance silently.
# ---------------------------------------------------------------------------

EXCAVATOR_MJCF = f"""
<mujoco model="excavator">
  <compiler angle="radian" autolimits="true"/>
  <default>
    <geom condim="3" friction="{FRAME_FRICTION}"/>
    <joint damping="0.05"/>
  </default>

  <worldbody>
    <body name="{BODY_CHASSIS}" pos="0 0 {CHASSIS_Z}">
      <freejoint name="{JOINT_ROOT}"/>
      {_FRAME_XML}

      <!-- running gear: four driven wheels, skid steer, no steering joint -->
      {_WHEELS_XML}

      <!-- excavation: arm + drum at each end, rear is the front rotated 180 deg -->
      {_ARMS_XML}
    </body>
  </worldbody>
</mujoco>
"""

# ---------------------------------------------------------------------------
# Sign conventions. Worth reading once before writing the env.
#
#   wheels      positive joint velocity drives the machine FORWARD (+x).
#               Skid steer: command JOINT_WHEELS_LEFT together and
#               JOINT_WHEELS_RIGHT together; equal = straight, opposite = spin.
#
#   arms        positive joint angle pitches the boom DOWN toward the soil.
#               0 is horizontal, ARM_RANGE[1] = 0.80 rad is full dig, which
#               puts the drum's lowest point ~19 cm below the ground plane.
#               ARM_RANGE[0] = -0.55 rad stows the drum ~66 cm up, which is the
#               dump height.
#
#   drums       a joint velocity of sign DIG_DRUM_SIGN digs, on both ends.
#               Derived from SCOOP_CURL and currently -1, so the digging
#               command is negative. It is the SAME sign on both drums: the
#               rear assembly is rotated 180 deg about z, so one sign gives
#               opposite world-frame rotation at the two ends and the digging
#               reactions cancel through the frame. Opposite signs make them
#               add and drive the machine out of its own cut.
#
# A natural low-level action vector is therefore 6-wide:
#   [left_wheels, right_wheels, arm_front, arm_rear, drum_front, drum_rear]
# ---------------------------------------------------------------------------


def reach() -> dict[str, float]:
    """Derived working envelope, in the chassis frame. Handy for sanity checks
    and for sizing the soil bed in the env cfg."""
    lo, hi = ARM_RANGE
    pivot_z = CHASSIS_Z + MAST_TOP_Z
    return {
        "pivot_height": pivot_z,
        # dig_depth is to the SHELL, because that is what ARM_RANGE[1] is set
        # against: the limit exists to stop the MOUTHS being buried past the
        # drum axis. The lips cut deeper than that and are meant to.
        "dig_depth": -(pivot_z - ARM_LEN * math.sin(hi) - DRUM_RADIUS),
        "lip_dig_depth": -(pivot_z - ARM_LEN * math.sin(hi) - BLADE_SWEPT_OUTER_R),
        "dump_height": pivot_z - ARM_LEN * math.sin(lo) - DRUM_RADIUS,
        # Envelopes, so these take the blades: they are the outermost thing on
        # the machine and the first to hit anything.
        "reach_x": PIVOT_X + ARM_LEN * math.cos(hi) + BLADE_SWEPT_OUTER_R,
        "overall_length": 2.0 * (PIVOT_X + ARM_LEN * math.cos(0.0) + BLADE_SWEPT_OUTER_R),
        "overall_width": TRACK + 2.0 * WHEEL_HALF_W,
        # Clear bore between the shell walls, then what the blades do to it.
        # A lip that stands proud bites before the shell; a lifter that reaches
        # inside carries the load round instead of letting it fall out the next
        # mouth. Both are the same geom.
        "drum_bore_diameter": 2.0 * (DRUM_RADIUS - DRUM_WALL_T),
        "drum_blade_swept_diameter": 2.0 * BLADE_SWEPT_INNER_R,
        "drum_lip_proud_of_shell": BLADE_SWEPT_OUTER_R - DRUM_RADIUS,
        "drum_lifter_into_bore": (DRUM_RADIUS - DRUM_WALL_T) - BLADE_SWEPT_INNER_R,
        # How much of the mouth the lip covers, from inside (the curl) and from
        # outside (the hood). Between them they decide whether the drum is a
        # trap or a hole, and they should be comparable -- a floor without a lid
        # lets the load lift straight back out of the mouth it came in through.
        "drum_mouth_covered_inside": scoop_mouth_coverage()[0],
        "drum_mouth_covered_outside": scoop_mouth_coverage()[1],
        # The channel between the two lips is the way in AND the way out, so
        # its narrowest point decides whether soil can move at all.
        "drum_channel_narrowest": scoop_channel()[0],
        "drum_channel_widest": scoop_channel()[1],
        "drum_bore_volume": math.pi * (DRUM_RADIUS - DRUM_WALL_T) ** 2 * (2.0 * DRUM_HALF_LEN),
        # Drum width against the AB/CD track. Deliberately just under 1.0.
        "drum_outer_width": 2.0 * (DRUM_HALF_LEN + 2.0 * CAP_HALF_T),
        "drum_width_over_track": 2.0 * (DRUM_HALF_LEN + 2.0 * CAP_HALF_T) / TRACK,
        # The drum is wider than the gap between the wheels, so the two
        # overlap in y and only x separates them. Swept min gap is 0.092 m at
        # full dig, front cross piece against front tyre. Drum width,
        # MAST_OFFSET_X and ARM_RANGE[1] all move it.
        "drum_lateral_overlap": (DRUM_HALF_LEN + 2.0 * CAP_HALF_T) - (WHEEL_Y - WHEEL_HALF_W),
        "wheel_standoff_at_full_dig": (
            PIVOT_X + ARM_LEN * math.cos(hi) - BLADE_SWEPT_OUTER_R
        ) - (AXLE_X + WHEEL_RADIUS),
        "yoke_standoff_from_blades": ARM_LEN - BLADE_SWEPT_OUTER_R - (BOOM_LEN + YOKE_HALF_W),
    }


def write_mjcf(directory: str | None = None) -> str:
    """Write the MJCF to disk and return its path. Idempotent.

    Used only by the offline USD conversion step; nothing calls it at training
    time.
    """
    directory = directory or os.path.join(tempfile.gettempdir(), "luna_hifi_assets")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "excavator.xml")
    with open(path, "w") as fh:
        fh.write(EXCAVATOR_MJCF.strip() + "\n")
    return path


if __name__ == "__main__":
    print(write_mjcf())
