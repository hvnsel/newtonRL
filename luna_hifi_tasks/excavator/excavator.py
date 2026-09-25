# excavator.py
#
# The bucket-drum excavator, as an MJCF string. Four driven wheels on a
# rectangular frame, and a counter-rotating excavation drum on a pitching arm
# at each end.
#
# Source for the converted USD in assets/excavator/, which is what training
# loads. Re-run the converter after every change here.
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
# * The rear assembly is the front assembly rotated 180 degrees about z, so the
#   rear drum's spin axis points along world -y and identical joint commands
#   counter-rotate the two drums. Their horizontal digging reactions cancel
#   through the frame.
# * Each drum is a ROTOR inside a SHROUD, hinged on the same axis off the arm
#   and moving independently.
#
#   The rotor is a hub carrying ROTOR_VANES logarithmic-spiral blades. The
#   spiral holds the rake constant from hub to tip, so the blade meets soil at
#   90 - ROTOR_RAKE degrees everywhere along it.
#
#   The shroud is the shell and does not turn with the rotor. It is closed over
#   every arc except SHROUD_INLET, so a pocket that has passed the inlet stays
#   shut for the rest of the turn. Its own actuator aims the inlet.
#
#   Outward acceleration at the vane tips is omega^2 * ROTOR_TIP_R, which
#   passes lunar gravity at 2.96 rad/s. MAX_DRUM_SPEED sits under it.
# * The rotor sweeps ~184 kg of regolith per drum, against a heavier chassis.
#   scripts/check_excavator.py reads the hold torques off the model's own mass
#   table.
#
# Wheels are cylinders with grousers. WHEEL_GROUSERS = 0 falls back to bare
# cylinders.
#
# Every body carrying a joint has a geom with mass, so MuJoCo derives real
# inertia for all of them.
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
# pass through the wheel band in y, so separation in x is what keeps them off
# the front wheels at full dig: swept clearance is 0.069 m here, binding pair
# is the cross piece against a rear grouser.
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
DECK_Z = -0.10                  # deck sits below the axle line
DECK_HALF = (0.55, 0.40, 0.04)

# --- mast: the tower on each crossmember that carries the arm pivot ---
MAST_TOP_Z = 0.20               # arm pivot height in the chassis frame
MAST_HALF = (0.05, 0.09, 0.13)  # x half-size is extended to span MAST_OFFSET_X

# --- arm ---
# The arm is a yoke: the boom stops short of the drum axis and two legs pass
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
# Positive arm angle pitches the boom down, toward the soil. The down limit of
# 0.72 rad buries the vane tips 0.185 m, one ROTOR_TIP_R.
ARM_RANGE = (-0.55, 0.72)       # -31.5 deg (stowed high) .. +41.3 deg (full dig)

# --- rotor ---
#
# A vaned rotor turning inside a fixed shroud. The rotor carries no outer wall
# of its own; the shroud closes the pockets between its vanes.
ROTOR_HUB_R = 0.045             # central drum the vanes are welded to
ROTOR_TIP_R = 0.185             # vane tip; fill is counted inside this
ROTOR_HALF_LEN = 0.475          # ~100 cm wide, matching the old drum
ROTOR_VANES = 6
VANE_SEGMENTS = 5               # straight boxes approximating one spiral vane
VANE_HALF_T = 0.006

# Rake: the angle between the vane and the radius, held constant along the
# blade by the logarithmic spiral r = r0 * exp(theta / tan(rake)). At rake b
# the face meets soil at 90 - b degrees.
#
# The blade sweeps ln(TIP/HUB) * tan(rake) = 1.414 * tan(rake) radians from hub
# to tip and stays inside one pocket. At 6 vanes the pocket is 60 deg and rake
# tops out near 35; 32 leaves 9 deg of margin.
ROTOR_RAKE = math.radians(32.0)
RAKE_SIGN = 1.0                 # handedness; DIG_DRUM_SIGN is derived from it

# Gap between adjacent vane tips, the way in to a pocket. Six vanes give a
# 0.185 m chord: 0.155 m clear once the MPM coupler has taken a voxel, 5.2
# grains at 0.03.
MPM_TARGET_VOXEL = 0.03
MPM_CLEARANCE = MPM_TARGET_VOXEL        # a passage narrower than this is closed
MPM_FLOW = 4.0 * MPM_TARGET_VOXEL       # width at which regolith flows

# --- shroud ---
#
# Hangs on its own hinge on the rotor axis, driven by its own actuator, so the
# policy aims the inlet independently of the arm.
SHROUD_IN_R = 0.195             # running clearance of 10 mm over the vane tips
SHROUD_OUT_R = 0.212
SHROUD_END_T = 0.012            # end plates, which close the pockets axially
SHROUD_INLET = math.radians(100.0)
# Spans the arm swing (1.35 rad) with room to turn the inlet up to dump.
SHROUD_RANGE = (-1.10, 1.10)

# Drum-level dimensions, taken from the shroud, which is the outermost part.
DRUM_RADIUS = SHROUD_OUT_R
DRUM_WALL_T = SHROUD_OUT_R - SHROUD_IN_R
DRUM_HALF_LEN = ROTOR_HALF_LEN
CAP_HALF_T = SHROUD_END_T

# --- masses (kg) ---
RAIL_MASS = 35.0                # each
CROSSMEMBER_MASS = 30.0         # each
DECK_MASS = 170.0               # ballast, low and central
MAST_MASS = 20.0                # each
WHEEL_MASS = 12.0               # cylinder only
GROUSER_MASS = 0.35             # each
ARM_BOOM_MASS = 6.0             # each arm: boom + cross + two legs = 17.5 kg
ARM_CROSS_MASS = 3.5
ARM_LEG_MASS = 4.0
ROTOR_HUB_MASS = 7.0
VANE_MASS = 1.9                 # one whole vane, split across its segments
SHROUD_ARC_MASS = 12.0          # the wall, split across its segments
SHROUD_END_MASS = 2.4           # each end plate

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
JOINT_SHROUDS = ["shroud_front", "shroud_rear"]

BODY_CHASSIS = "chassis"
BODY_WHEELS = ["wheel_fl_body", "wheel_fr_body", "wheel_rl_body", "wheel_rr_body"]
BODY_ARMS = ["arm_front_body", "arm_rear_body"]
BODY_DRUMS = ["drum_front_body", "drum_rear_body"]
BODY_SHROUDS = ["shroud_front_body", "shroud_rear_body"]

# The bodies in the MPM CouplerProxyMappingCfg: wheels for traction, rotors for
# excavation, shrouds for retention.
BODY_SOIL_CONTACT = BODY_WHEELS + BODY_DRUMS + BODY_SHROUDS

FRAME_RGBA = "0.78 0.62 0.16 1"     # machine yellow
DECK_RGBA = "0.42 0.45 0.50 1"
WHEEL_RGBA = "0.13 0.13 0.14 1"
GROUSER_RGBA = "0.22 0.22 0.24 1"
ARM_RGBA = "0.70 0.55 0.14 1"
DRUM_RGBA = "0.55 0.57 0.60 1"
BLADE_RGBA = "0.82 0.84 0.86 1"
SHROUD_RGBA = "0.62 0.64 0.67 1"

# ---------------------------------------------------------------------------
# Geometry helpers
#
# Every spinning part (wheels, drums) turns about the body's local y axis, and
# the geometry below is built about it.
#
# MuJoCo's euler="0 b 0" rotates about +y, mapping local z_hat to
# (sin b, 0, cos b). To aim a box's thickness axis along the direction psi in
# the local x-z plane, set b = pi/2 - psi. Two cases follow:
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


# ---------------------------------------------------------------------------
# Rotor and shroud geometry
# ---------------------------------------------------------------------------

VANE_SWEEP = math.log(ROTOR_TIP_R / ROTOR_HUB_R) * math.tan(ROTOR_RAKE)
POCKET_ARC = 2.0 * math.pi / ROTOR_VANES
# Chord between adjacent vane tips: the way into a pocket.
VANE_TIP_GAP = 2.0 * ROTOR_TIP_R * math.sin(math.pi / ROTOR_VANES)


def vane_points(phi_root: float) -> list[tuple[float, float]]:
    """Sample one logarithmic-spiral vane from hub to tip, in the rotor frame.

    r(theta) = ROTOR_HUB_R * exp(theta / tan(rake)), so the angle between the
    blade and the radius is ROTOR_RAKE everywhere along it.
    """
    pts = []
    for i in range(VANE_SEGMENTS + 1):
        r = ROTOR_HUB_R + (ROTOR_TIP_R - ROTOR_HUB_R) * i / VANE_SEGMENTS
        phi = phi_root + RAKE_SIGN * math.log(r / ROTOR_HUB_R) * math.tan(ROTOR_RAKE)
        pts.append((r * math.cos(phi), r * math.sin(phi)))
    return pts


def _vane_xml(prefix: str, index: int, phi_root: float) -> list[str]:
    """One vane, as a chain of boxes along the spiral."""
    pts = vane_points(phi_root)
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])) or 1.0
    parts = []
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg = math.hypot(dx, dy)
        cx, cz = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        # The box's local x runs along the segment. euler="0 e 0" maps local
        # x_hat to (cos e, 0, -sin e), so e = -atan2(dy, dx).
        parts.append(
            f'<geom name="{prefix}_vane{index}_{i}" type="box" '
            f'pos="{cx:.5f} 0 {cz:.5f}" euler="0 {-math.atan2(dy, dx):.6f} 0" '
            f'size="{0.5 * seg:.5f} {ROTOR_HALF_LEN:.5f} {VANE_HALF_T:.5f}" '
            f'mass="{VANE_MASS * seg / total:.4f}" rgba="{BLADE_RGBA}" '
            f'friction="{DRUM_FRICTION}"/>'
        )
    return parts


def _shroud_arc_xml(prefix: str) -> list[str]:
    """The shroud wall: closed everywhere except SHROUD_INLET.

    Each plate spans at most 20 degrees.
    """
    span = 2.0 * math.pi - SHROUD_INLET
    n = max(int(math.ceil(span / math.radians(20.0))), 6)
    step = span / n
    mid_r = 0.5 * (SHROUD_IN_R + SHROUD_OUT_R)
    # tan, plus a fixed 2 mm, so adjacent plates overlap at the corners.
    half_arc = mid_r * math.tan(0.5 * step) + 0.002
    # The inlet is centred on the shroud's own -z: joint angle 0 points it
    # straight down with the arm horizontal.
    start = -0.5 * math.pi + 0.5 * SHROUD_INLET
    return [
        _tangential_plate(
            f"{prefix}_arc{i}",
            start + (i + 0.5) * step,
            mid_r,
            half_arc,
            ROTOR_HALF_LEN + SHROUD_END_T,
            0.5 * (SHROUD_OUT_R - SHROUD_IN_R),
            SHROUD_ARC_MASS / n,
            SHROUD_RGBA,
            DRUM_FRICTION,
        )
        for i in range(n)
    ]


def _shroud(prefix: str, joint: str) -> str:
    """The stationary half of the drum: wall, two end plates, its own hinge."""
    parts = _shroud_arc_xml(prefix)
    for sign, side in ((1.0, "l"), (-1.0, "r")):
        y = sign * (ROTOR_HALF_LEN + SHROUD_END_T)
        parts.append(
            f'<geom name="{prefix}_end_{side}" type="cylinder" euler="1.570796 0 0" '
            f'pos="0 {y:.5f} 0" size="{SHROUD_OUT_R:.5f} {SHROUD_END_T:.5f}" '
            f'mass="{SHROUD_END_MASS}" rgba="{SHROUD_RGBA}" friction="{DRUM_FRICTION}"/>'
        )
    body_xml = "\n          ".join(parts)
    return f"""<body name="{prefix}_body" pos="{ARM_LEN:.4f} 0 0">
          <joint name="{joint}" type="hinge" axis="0 1 0"
                 range="{SHROUD_RANGE[0]} {SHROUD_RANGE[1]}" armature="0.10" damping="1.5"/>
          {body_xml}
        </body>"""


def _rotor(prefix: str, joint: str) -> str:
    """The turning half: a hub and ROTOR_VANES spiral blades."""
    parts = [
        f'<geom name="{prefix}_hub" type="cylinder" euler="1.570796 0 0" '
        f'size="{ROTOR_HUB_R:.5f} {ROTOR_HALF_LEN:.5f}" mass="{ROTOR_HUB_MASS}" '
        f'rgba="{DRUM_RGBA}" friction="{DRUM_FRICTION}"/>'
    ]
    for i in range(ROTOR_VANES):
        parts += _vane_xml(prefix, i, i * POCKET_ARC)
    body_xml = "\n          ".join(parts)
    return f"""<body name="{prefix}_body" pos="{ARM_LEN:.4f} 0 0">
          <joint name="{joint}" type="hinge" axis="0 1 0" armature="0.08"/>
          {body_xml}
        </body>"""


def vane_swept_radii() -> tuple[float, float]:
    """Innermost and outermost radius any vane geom reaches, thickness included."""
    lo, hi = math.inf, 0.0
    for v in range(ROTOR_VANES):
        pts = vane_points(v * POCKET_ARC)
        for a, b in zip(pts, pts[1:]):
            cx, cz = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
            mid = math.hypot(cx, cz)
            half = 0.5 * math.hypot(b[0] - a[0], b[1] - a[1])
            corner = math.hypot(half, VANE_HALF_T)
            lo = min(lo, max(mid - corner, 0.0))
            hi = max(hi, mid + corner)
    return lo, hi


def pocket_geometry() -> dict[str, float]:
    """The numbers that decide whether regolith can get in and stay in."""
    return {
        "vanes": float(ROTOR_VANES),
        "rake_deg": math.degrees(ROTOR_RAKE),
        "attack_deg": 90.0 - math.degrees(ROTOR_RAKE),
        "vane_sweep_deg": math.degrees(VANE_SWEEP),
        "pocket_arc_deg": math.degrees(POCKET_ARC),
        "shadowed": float(VANE_SWEEP >= POCKET_ARC),
        "tip_gap": VANE_TIP_GAP,
        "tip_gap_clear": VANE_TIP_GAP - MPM_TARGET_VOXEL,
        "tip_gap_grains": (VANE_TIP_GAP - MPM_TARGET_VOXEL) / MPM_TARGET_VOXEL,
        "running_clearance": SHROUD_IN_R - ROTOR_TIP_R,
        "inlet_deg": math.degrees(SHROUD_INLET),
        "inlet_chord": 2.0 * SHROUD_IN_R * math.sin(0.5 * SHROUD_INLET),
        "swept_volume": math.pi * ROTOR_TIP_R ** 2 * (2.0 * ROTOR_HALF_LEN),
    }


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


# Derived yoke geometry. The legs straddle the shroud.
YOKE_Y = ROTOR_HALF_LEN + 2.0 * SHROUD_END_T + YOKE_CLEARANCE + YOKE_HALF_W

VANE_SWEPT_INNER_R, VANE_SWEPT_OUTER_R = vane_swept_radii()

# Which sign of drum command digs. A joint velocity of sign k turns the rotor
# toward -k*phi, so relative to the rotor the soil streams toward +k*phi. The
# vane bites when its concave face leads, which is the -RAKE_SIGN side.
DIG_DRUM_SIGN = -RAKE_SIGN

# The cross piece spans the machine at the drum's height and clears the shroud.
YOKE_STANDOFF = 0.025
BOOM_LEN = ARM_LEN - SHROUD_OUT_R - YOKE_HALF_W - YOKE_STANDOFF
_LEG_X0 = BOOM_LEN - YOKE_HALF_W            # legs overlap the cross piece slightly


def _arm_assembly(
    side: str,
    yaw: float,
    arm_joint: str,
    drum_prefix: str,
    drum_joint: str,
    shroud_prefix: str,
    shroud_joint: str,
) -> str:
    """One arm + drum, hung off the mast at PIVOT_X.

    `yaw` is 0 for the front assembly and pi for the rear, a true 180-degree
    copy, so the two drums take opposite world spin axes for the same joint
    command.

    The arm is boom -> cross piece -> two legs straddling the drum. The rotor
    and the shroud are siblings, both hinged on the same axis off the arm.
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
        {_rotor(drum_prefix, drum_joint)}
        {_shroud(shroud_prefix, shroud_joint)}
      </body>"""


# ---------------------------------------------------------------------------
# Frame geoms. All fixed to the chassis body; the masts carry no joint.
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
    # deck: slung below the axle line, carrying most of the mass
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
        _arm_assembly("front", 0.0, "arm_front", "drum_front", "drum_front",
                      "shroud_front", "shroud_front"),
        _arm_assembly("rear", math.pi, "arm_rear", "drum_rear", "drum_rear",
                      "shroud_rear", "shroud_rear"),
    ]
)

# ---------------------------------------------------------------------------
# Model
#
# Every geom spells out its own friction and rgba; the MJCF->USD converter
# drops <default class="..."> inheritance.
# ---------------------------------------------------------------------------

EXCAVATOR_MJCF = f"""
<mujoco model="excavator">
  <compiler angle="radian" autolimits="true"/>
  <default>
    <geom condim="3" friction="{FRAME_FRICTION}"/>
    <joint damping="0.05"/>
  </default>

  <!-- The rotor turns inside the shroud on a running fit, which is a bearing
       rather than a contact. -->
  <contact>
    <exclude body1="drum_front_body" body2="shroud_front_body"/>
    <exclude body1="drum_rear_body" body2="shroud_rear_body"/>
  </contact>

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
# Sign conventions.
#
#   wheels      positive joint velocity drives the machine FORWARD (+x).
#               Skid steer: JOINT_WHEELS_LEFT commanded together and
#               JOINT_WHEELS_RIGHT together; equal = straight, opposite = spin.
#
#   arms        positive joint angle pitches the boom DOWN toward the soil.
#               0 is horizontal, ARM_RANGE[1] = 0.72 rad is full dig, putting
#               the drum's lowest point ~19 cm below the ground plane.
#               ARM_RANGE[0] = -0.55 rad stows the drum ~66 cm up, the dump
#               height.
#
#   drums       a joint velocity of sign DIG_DRUM_SIGN digs, the same sign on
#               both ends. Derived from RAKE_SIGN, currently -1. The rear
#               assembly is rotated 180 deg about z, so one sign gives opposite
#               world-frame rotation at the two ends and the digging reactions
#               cancel through the frame.
#
#               Outward acceleration at the vane tips is omega^2 * ROTOR_TIP_R,
#               which passes lunar gravity at 2.96 rad/s.
#
#   shrouds     positive joint angle turns the inlet the way a positive arm
#               angle turns the boom, so holding the inlet still against the
#               ground through an arm pitch means driving the shroud to minus
#               the arm angle. 0 points the inlet straight down with the arm
#               horizontal.
#
# The low-level action vector is 8-wide:
#   [left_wheels, right_wheels, arm_front, arm_rear, drum_front, drum_rear,
#    shroud_front, shroud_rear]
# ---------------------------------------------------------------------------


def reach() -> dict[str, float]:
    """Derived working envelope, in the chassis frame. Sizes the soil bed in
    the env cfg and feeds the geometry checks."""
    lo, hi = ARM_RANGE
    pivot_z = CHASSIS_Z + MAST_TOP_Z
    return {
        "pivot_height": pivot_z,
        # dig_depth is to the shroud, the outermost part of the drum.
        "dig_depth": -(pivot_z - ARM_LEN * math.sin(hi) - SHROUD_OUT_R),
        "vane_dig_depth": -(pivot_z - ARM_LEN * math.sin(hi) - ROTOR_TIP_R),
        "dump_height": pivot_z - ARM_LEN * math.sin(lo) - SHROUD_OUT_R,
        "reach_x": PIVOT_X + ARM_LEN * math.cos(hi) + SHROUD_OUT_R,
        "overall_length": 2.0 * (PIVOT_X + ARM_LEN * math.cos(0.0) + SHROUD_OUT_R),
        "overall_width": TRACK + 2.0 * WHEEL_HALF_W,
        # Fill is counted inside the rotor's swept cylinder.
        "rotor_swept_diameter": 2.0 * ROTOR_TIP_R,
        "rotor_swept_volume": math.pi * ROTOR_TIP_R ** 2 * (2.0 * ROTOR_HALF_LEN),
        "vane_swept_inner_r": VANE_SWEPT_INNER_R,
        "vane_swept_outer_r": VANE_SWEPT_OUTER_R,
        # The running fit between vane tips and shroud. The coupler seals a
        # gap narrower than a voxel.
        "running_clearance": SHROUD_IN_R - VANE_SWEPT_OUTER_R,
        # The gap between adjacent vane tips, the way into a pocket.
        "vane_tip_gap": VANE_TIP_GAP,
        "vane_tip_gap_grains": (VANE_TIP_GAP - MPM_TARGET_VOXEL) / MPM_TARGET_VOXEL,
        # Rake, and the arcs that decide whether a blade stays in its pocket.
        "rake_deg": math.degrees(ROTOR_RAKE),
        "attack_deg": 90.0 - math.degrees(ROTOR_RAKE),
        "vane_sweep_deg": math.degrees(VANE_SWEEP),
        "pocket_arc_deg": math.degrees(POCKET_ARC),
        # How much of the circumference the inlet opens, and its chord.
        "inlet_deg": math.degrees(SHROUD_INLET),
        "inlet_chord": 2.0 * SHROUD_IN_R * math.sin(0.5 * SHROUD_INLET),
        "shroud_outer_width": 2.0 * (ROTOR_HALF_LEN + 2.0 * SHROUD_END_T),
        "shroud_width_over_track": 2.0 * (ROTOR_HALF_LEN + 2.0 * SHROUD_END_T) / TRACK,
        # How far the shroud reaches past the inner face of a wheel in y.
        "drum_lateral_overlap": (
            (ROTOR_HALF_LEN + 2.0 * SHROUD_END_T) - (WHEEL_Y - WHEEL_HALF_W)
        ),
        "wheel_standoff_at_full_dig": (
            PIVOT_X + ARM_LEN * math.cos(hi) - SHROUD_OUT_R
        ) - (AXLE_X + WHEEL_RADIUS),
        "yoke_standoff_from_shroud": ARM_LEN - SHROUD_OUT_R - (BOOM_LEN + YOKE_HALF_W),
        # Spin at which outward acceleration at the vane tips equals lunar
        # gravity.
        "spin_limit_rad_s": math.sqrt(1.62 / ROTOR_TIP_R),
    }


def write_mjcf(directory: str | None = None) -> str:
    """Write the MJCF to disk and return its path. Idempotent.

    Called by the offline USD conversion step.
    """
    directory = directory or os.path.join(tempfile.gettempdir(), "luna_hifi_assets")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "excavator.xml")
    with open(path, "w") as fh:
        fh.write(EXCAVATOR_MJCF.strip() + "\n")
    return path


if __name__ == "__main__":
    print(write_mjcf())
