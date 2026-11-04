# The bucket-drum excavator as an MJCF string: four driven wheels on a
# rectangular frame and a counter-rotating drum on a pitching arm at each end.
# assets/excavator/ holds the USD converted from it, which training loads.
#
# Plan view (+x forward). A, B are the front corners, C, D the rear:
#
#                        front drum
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
#                        rear drum
#
# Elevation (+x right, +z up), arms at full down:
#
#              .-.                                     .-.
#             ( o )  <- drum                          ( o )
#              '-'   \                               /  '-'
#                     \___[mast]===[deck]===[mast]___/
#                      (O)                       (O)     <- wheels
#           ~~~~~~~~~~~~~~~~~ soil ~~~~~~~~~~~~~~~~~~~~
#
# * Skid steer: four wheel hinges, the left pair and the right pair commanded
#   together.
# * The rear assembly is the front one rotated 180 degrees about z, so one
#   drum command turns the two drums in opposite world directions.
# * Each drum is a rotor (a hub with ROTOR_VANES logarithmic-spiral vanes)
#   inside a shroud (a shell open over SHROUD_INLET), each on its own hinge on
#   the same axis off the arm.
# * Wheels are cylinders with WHEEL_GROUSERS radial cleats.
# * Every body carrying a joint has a geom with mass.
# * Wheel centres sit at z = 0 in the chassis frame, one wheel radius above
#   the ground.
#
# Sign conventions:
#   wheels   positive velocity drives forward (+x).
#   arms     positive angle pitches the boom down, on both arms. ARM_RANGE[1]
#            is full dig, ARM_RANGE[0] stows the drum at dump height.
#   drums    a velocity of sign DIG_DRUM_SIGN digs, on both ends.
#   shrouds  positive angle turns the inlet the way a positive arm angle turns
#            the boom; 0 points it straight down with the arm horizontal.

from __future__ import annotations

import getpass
import hashlib
import math
import os
import tempfile
from pathlib import Path

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

# Arm pivot distance ahead of the axle; the drum and yoke legs clear the
# front wheels by 0.069 m at full dig.
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
# A yoke: the boom stops short of the drum axis and two legs pass outboard of
# the shroud's end plates to the axle.
ARM_LEN = 0.76                  # pivot to drum axis
ARM_HALF_H = 0.05
ARM_HALF_W = 0.07
YOKE_HALF_W = 0.04
YOKE_HALF_H = 0.05
# Gap between an end plate's outer face and a leg's inner face.
YOKE_CLEARANCE = 0.005
# Positive angle pitches the boom down. 0.72 rad buries the vane tips 0.185 m.
ARM_RANGE = (-0.55, 0.72)       # -31.5 deg (stowed high) .. +41.3 deg (full dig)

# --- rotor: a hub with spiral vanes; the shroud closes the pockets between them ---
ROTOR_HUB_R = 0.045             # central drum the vanes are welded to
ROTOR_TIP_R = 0.185             # vane tip; fill is counted inside this
ROTOR_HALF_LEN = 0.475          # 0.95 m wide
ROTOR_VANES = 4
VANE_SEGMENTS = 5               # straight boxes approximating one spiral vane
VANE_HALF_T = 0.006

# Rake: the angle between vane and radius, constant along the logarithmic
# spiral r = r0 * exp(theta / tan(rake)). The face meets soil at 90 - rake degrees.
ROTOR_RAKE = math.radians(32.0)
RAKE_SIGN = 1.0                 # spiral handedness

# --- shroud: its own hinge and actuator on the rotor axis ---
SHROUD_IN_R = 0.195             # running clearance of 10 mm over the vane tips
SHROUD_OUT_R = 0.212
SHROUD_END_T = 0.012            # end plates, which close the pockets axially
SHROUD_INLET = math.radians(100.0)
# The arm swing (1.35 rad) plus room to turn the inlet up to dump.
SHROUD_RANGE = (-1.10, 1.10)

# --- masses (kg) ---
RAIL_MASS = 35.0                # each
CROSSMEMBER_MASS = 30.0         # each
# The deck is ballast slung under the axle line; the machine totals 994 kg.
DECK_MASS = 670.0
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
# Joint and body names
# ---------------------------------------------------------------------------

JOINT_ROOT = "root"
JOINT_WHEELS_LEFT = ["wheel_fl", "wheel_rl"]
JOINT_WHEELS_RIGHT = ["wheel_fr", "wheel_rr"]
JOINT_WHEELS = JOINT_WHEELS_LEFT + JOINT_WHEELS_RIGHT
JOINT_ARMS = ["arm_front", "arm_rear"]
JOINT_DRUMS = ["drum_front", "drum_rear"]
JOINT_SHROUDS = ["shroud_front", "shroud_rear"]

BODY_CHASSIS = "chassis"
BODY_DRUMS = ["drum_front_body", "drum_rear_body"]

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
# Wheels and drums spin about the body's local y axis.
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
    """The shroud wall, closed except over SHROUD_INLET, in plates of at most 20 degrees."""
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


def _wheel(name: str, x: float, y: float, joint: str) -> str:
    """A cylinder tyre spinning about y, with radial grousers. euler="pi/2 0 0"
    lays the cylinder's local z along y."""
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

# The drum command sign that digs: the vanes lead with their concave face.
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
    """One arm and drum, hung off the mast at PIVOT_X. `yaw` is 0 for the front
    assembly and pi for the rear.

    The arm is boom -> cross piece -> two legs straddling the drum; the rotor
    and shroud are siblings hinged on the same axis off the arm. Joint axes are
    local +y, which is world -y on the rear assembly, so one arm angle raises or
    lowers both drums together.
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
# Frame geoms, all fixed to the chassis body.
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
# Model. Every geom carries its own friction and rgba; the MJCF->USD
# converter does not apply <default class="..."> inheritance.
# ---------------------------------------------------------------------------

EXCAVATOR_MJCF = f"""
<mujoco model="excavator">
  <compiler angle="radian" autolimits="true"/>
  <default>
    <geom condim="3" friction="{FRAME_FRICTION}"/>
    <joint damping="0.05"/>
  </default>

  <!-- The rotor turns inside the shroud on a running fit: a bearing. -->
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


def source_hash() -> str:
    """MD5 of the MJCF the USD is converted from."""
    return hashlib.md5(EXCAVATOR_MJCF.encode()).hexdigest()


def source_stamp_path() -> Path:
    """assets/excavator.usd.srchash: the source_hash the converted asset was
    built from. It sits outside assets/excavator/, which the rebuild deletes."""
    return Path(__file__).resolve().parents[2] / "assets" / "excavator.usd.srchash"


def write_mjcf(directory: str | None = None) -> str:
    """Write the MJCF and the source stamp; return the MJCF's path. The default
    directory is <tmp>/luna_hifi_assets-<user>."""
    stamp = source_stamp_path()
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(source_hash() + "\n")
    directory = directory or os.path.join(tempfile.gettempdir(), f"luna_hifi_assets-{getpass.getuser()}")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "excavator.xml")
    with open(path, "w") as fh:
        fh.write(EXCAVATOR_MJCF.strip() + "\n")
    return path


if __name__ == "__main__":
    print(write_mjcf())
