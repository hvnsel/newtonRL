# excavator.py
#
# The bucket-drum excavator, as an MJCF string. Four driven wheels on a
# rectangular frame, and a counter-rotating excavation drum on a pitching arm
# at each end.
#
# Same contract as tricycle.py: this file is NOT read at training time.
# Training loads the converted USD in assets/excavator/. The MJCF here is the
# *source* of that USD -- whenever you change it, re-run the converter (README,
# "Convert the tricycle asset", with the paths swapped) so the USD matches.
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
# Why it is shaped this way
# -------------------------
# * Skid steer, no steering joints. Four hinges, left pair and right pair
#   commanded together. One less mechanism to model and it is what real drum
#   excavators do.
# * The rear assembly is the front assembly rotated 180 degrees about z. That
#   is not just tidiness: it makes the rear drum's spin axis point along world
#   -y, so *identical* joint commands counter-rotate the two drums. Both drums
#   dig at +1, and their horizontal digging reactions cancel through the
#   frame. That force cancellation is the entire reason this architecture
#   exists -- it lets a light machine excavate without needing its own weight
#   in traction.
# * The drums are hollow. Each is a ring of box segments with SCOOP_COUNT gaps
#   left open as scoop mouths, closed at both ends by cap discs, with a raked
#   blade at the leading edge of each mouth. Soil is cut by the blade, carried
#   through the mouth, and retained in the cavity. A solid cylinder would push
#   soil around; it would never fill.
# * The chassis outweighs both drums roughly 13:1, so a full drum barely moves
#   the CG and the machine does not pitch over its front axle while digging.
#
# Wheels are cylinders with grousers, not the spheres the tricycle used. On
# granular media a smooth wheel simply shears the surface and spins in place,
# so grousers are load-bearing physics here, not decoration. They cost
# colliders: set WHEEL_GROUSERS = 0 to fall back to bare cylinders.
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
WHEELBASE = 1.60                # front axle to rear axle, along x
TRACK = 1.15                    # left wheel centre to right wheel centre, along y

AXLE_X = 0.5 * WHEELBASE        # +/- 0.80
WHEEL_Y = 0.5 * TRACK           # +/- 0.575

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
MAST_HALF = (0.05, 0.09, 0.13)

# --- arm ---
# The arm is a YOKE, not a single boom. A centre boom running out to the drum
# axis would sit inside the drum cavity: it would block the mouths and occupy
# the volume we are trying to fill with soil. So the boom stops short of the
# drum, and two legs pass outboard of the end caps to pick up the axle. This is
# also how real drum excavators are built, for the same reason.
ARM_LEN = 0.55                  # pivot to drum axis
ARM_HALF_H = 0.05
ARM_HALF_W = 0.07
YOKE_HALF_W = 0.04
YOKE_HALF_H = 0.05
YOKE_CLEARANCE = 0.005          # gap between the end cap and the inner face of a leg
# Positive arm angle pitches the boom DOWN, toward the soil. See the sign note
# in the module docstring block below the model string.
ARM_RANGE = (-0.60, 1.00)       # -34 deg (stowed high) .. +57 deg (full dig)

# --- drum ---
DRUM_RADIUS = 0.20
DRUM_HALF_LEN = 0.22            # 44 cm wide drum
DRUM_WALL_T = 0.030             # shell thickness; see the MPM note below
DRUM_FACETS = 12                # angular slots around the circumference
SCOOP_COUNT = 3                 # of those slots, this many are left open as mouths
SCOOP_RAKE = 0.50               # blade rake off radial, rad
BLADE_INNER_R = 0.13
BLADE_OUTER_R = 0.225           # stands proud of the shell so it bites first
BLADE_HALF_T = 0.012
CAP_HALF_T = 0.012

# MPM note: DRUM_WALL_T is 3 cm against the tricycle's 5 cm voxel. The coupler
# inflates colliders by MPM_COLLIDER_MARGIN (half a voxel each side), so a 3 cm
# wall reads as ~8 cm to the solver and particles will not tunnel. But the
# cavity is only 2 * BLADE_INNER_R ~ 26 cm across, i.e. five cells at a 5 cm
# voxel -- too coarse to resolve filling. Expect to run the drum region at a
# 2.5-3 cm voxel, and budget the particle count accordingly. This is the single
# place in the machine where MPM resolution actually binds.

# --- masses (kg) ---
RAIL_MASS = 35.0                # each
CROSSMEMBER_MASS = 30.0         # each
DECK_MASS = 170.0               # the ballast: keeps the CG low and central
MAST_MASS = 20.0                # each
WHEEL_MASS = 12.0               # cylinder only
GROUSER_MASS = 0.35             # each
ARM_BOOM_MASS = 6.0             # each arm: boom + cross + two legs = 14 kg
ARM_CROSS_MASS = 2.0
ARM_LEG_MASS = 3.0
DRUM_SEGMENT_MASS = 1.0         # each shell segment
DRUM_CAP_MASS = 1.2             # each end cap
DRUM_BLADE_MASS = 0.55          # each scoop blade

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

    The mouths are spread as evenly as SCOOP_COUNT divides DRUM_FACETS allows.
    Each open slot gets a blade at its leading edge -- leading with respect to a
    positive joint velocity, which is the digging direction.
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
            # Mouth. Blade sits on the leading edge, i.e. the far side in the
            # direction of increasing phi.
            parts.append(
                _radial_blade(
                    f"{prefix}_blade{i}",
                    phi + 0.5 * step,
                    BLADE_INNER_R,
                    BLADE_OUTER_R,
                    DRUM_HALF_LEN,
                    BLADE_HALF_T,
                    SCOOP_RAKE,
                    DRUM_BLADE_MASS,
                    BLADE_RGBA,
                    DRUM_FRICTION,
                )
            )
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
YOKE_Y = DRUM_HALF_LEN + CAP_HALF_T + YOKE_CLEARANCE + YOKE_HALF_W
BOOM_LEN = ARM_LEN - DRUM_RADIUS - 0.03     # boom stops clear of the drum envelope
_LEG_X0 = BOOM_LEN - YOKE_HALF_W            # legs overlap the cross piece slightly


def _arm_assembly(side: str, yaw: float, arm_joint: str, drum_prefix: str, drum_joint: str) -> str:
    """One arm + drum, hung off the mast at AXLE_X.

    `yaw` is 0 for the front assembly and pi for the rear. The rear is a true
    180-degree copy, which is what gives the two drums opposite world spin axes
    for the same joint command.

    The arm is boom -> cross piece -> two legs straddling the drum. Nothing
    enters the drum's swept envelope; see the YOKE comment in the dimensions
    block.
    """
    x = AXLE_X * math.cos(yaw)
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
    f'<geom name="mast_front" type="box" pos="{AXLE_X} 0 {0.5 * MAST_TOP_Z:.4f}" '
    f'size="{MAST_HALF[0]} {MAST_HALF[1]} {MAST_HALF[2]}" '
    f'mass="{MAST_MASS}" rgba="{FRAME_RGBA}" friction="{FRAME_FRICTION}"/>',
    f'<geom name="mast_rear" type="box" pos="{-AXLE_X} 0 {0.5 * MAST_TOP_Z:.4f}" '
    f'size="{MAST_HALF[0]} {MAST_HALF[1]} {MAST_HALF[2]}" '
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
# rgba. The Isaac MJCF->USD converter is the fragile link in this pipeline and
# class inheritance is exactly the kind of thing it silently drops. Verbose XML
# is cheap; a silently unfrictioned wheel is not.
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
#               0 is horizontal, ARM_RANGE[1] = 1.00 rad is full dig, which
#               puts the drum's lowest point ~16 cm below the ground plane.
#               ARM_RANGE[0] = -0.60 rad stows the drum ~61 cm up, which is the
#               dump height.
#
#   drums       positive joint velocity DIGS, on both ends. Because the rear
#               assembly is rotated 180 deg about z, the same sign gives
#               opposite world-frame rotation -- the drums counter-rotate and
#               their digging reactions cancel through the frame. Commanding
#               them with opposite signs instead makes the reactions ADD and
#               will try to drive the machine out of its own cut.
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
        "dig_depth": -(pivot_z - ARM_LEN * math.sin(hi) - DRUM_RADIUS),
        "dump_height": pivot_z - ARM_LEN * math.sin(lo) - DRUM_RADIUS,
        "reach_x": AXLE_X + ARM_LEN * math.cos(hi) + DRUM_RADIUS,
        "overall_length": 2.0 * (AXLE_X + ARM_LEN * math.cos(0.0) + DRUM_RADIUS),
        "overall_width": TRACK + 2.0 * WHEEL_HALF_W,
        # Clear bore between the shell walls, and the narrower diameter the
        # blades sweep. Capacity sits between the two.
        "drum_bore_diameter": 2.0 * (DRUM_RADIUS - DRUM_WALL_T),
        "drum_blade_swept_diameter": 2.0 * BLADE_INNER_R,
        "drum_bore_volume": math.pi * (DRUM_RADIUS - DRUM_WALL_T) ** 2 * (2.0 * DRUM_HALF_LEN),
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
