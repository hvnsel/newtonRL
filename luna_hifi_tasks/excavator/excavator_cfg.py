# excavator_cfg.py
#
# The shared ArticulationCfg. Both the navigate and excavate tasks spawn the
# same machine with the same actuators; only observations, actions and rewards
# differ between them. Keeping the asset in one place is what makes a policy
# trained on one task loadable in the other.
#
# Actuator effort limits are derived, not guessed: from the validated mass
# table in excavator.py and from LUNAR gravity. See the gravity block below --
# wheels are sized by traction (which falls 6x on the Moon) and the arm by
# digging resistance (which does not fall at all).

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .excavator import (
    ARM_RANGE,
    JOINT_ARMS,
    JOINT_DRUMS,
    JOINT_WHEELS,
    CHASSIS_Z,
    DRUM_HALF_LEN,
    DRUM_RADIUS,
    DRUM_WALL_T,
)

# <repo>/assets/excavator/excavator.usda, resolved relative to this file so the
# repo can live anywhere.
#
# THIS FILE DOES NOT EXIST UNTIL YOU RUN THE CONVERTER. excavator.py writes the
# MJCF; the offline MJCF -> USD step turns it into the USD loaded here. See the
# README's "Convert the tricycle asset" section with the paths swapped:
#
#   isaaclab -p -c "from luna_hifi_tasks.excavator.excavator import write_mjcf; print(write_mjcf())"
#   isaaclab -p scripts/tools/convert_mjcf.py <that path> <repo>/assets/excavator/excavator.usd
#
# The converter writes assets/excavator/excavator.usda (subfolder, .usda), not
# the path you hand it. Redo it whenever excavator.py changes.
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
EXCAVATOR_USD_PATH = ASSETS_DIR / "excavator" / "excavator.usda"

# Spawn height: wheels rest on z = 0 when the chassis origin is one wheel
# radius up. A small clearance stops the solver resolving an initial
# interpenetration on the first step.
SPAWN_Z = CHASSIS_Z + 0.02

# Arm angle held during navigation. Stowed high, clear of the ground, so the
# drums cannot touch soil while driving.
ARM_STOW_ANGLE = ARM_RANGE[0]

# Bore capacity, used to normalise drum fill into a fraction. Matches the
# analytic volume the sensor tests check against.
BORE_RADIUS = DRUM_RADIUS - DRUM_WALL_T
BORE_HALF_LEN = DRUM_HALF_LEN

# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------
#
# Lunar. Set this on SimulationCfg.gravity in every env cfg; Isaac Lab's
# default is earth and nothing warns you if you forget.
LUNAR_GRAVITY = (0.0, 0.0, -1.62)
EARTH_G = 9.81
LUNAR_G = 1.62

# What lunar gravity actually changes, since it drove the actuator numbers
# below and is easy to get wrong:
#
#   Weight scales, soil strength does NOT. The machine's 483 kg weighs 782 N
#   here instead of 4738 N, so every traction number falls by 6. But regolith
#   cohesion and shear strength are gravity-independent, so the force needed to
#   cut it is essentially unchanged. Excavation resistance versus available
#   traction therefore gets about six times worse on the Moon than on Earth.
#
#   That is the whole reason this machine counter-rotates its drums. Total
#   tractive force available is roughly 782 N at mu = 1. A drum cutting a 1 m
#   swath through cohesive regolith can easily exceed that, and a machine that
#   had to push against its own cut would simply be shoved backwards. With the
#   two drums turning opposite ways their horizontal reactions cancel through
#   the frame, so digging costs almost no net traction. On Earth the
#   cancellation is a nicety; here it is what makes the task possible at all.
#
#   Consequence for the reward: chassis drift during a dig is a direct measure
#   of whether that cancellation is working, and belongs in the excavation
#   reward rather than being left implicit.
#
#   Consequence for actuators: the ARM is not sized by gravity here. Holding a
#   full drum out horizontally needs only ~210 N-m at lunar gravity (1270 on
#   Earth), but resisting the vertical component of digging resistance is
#   unchanged by gravity and is the larger load. The WHEELS, by contrast, are
#   sized purely by traction, which does fall by six.

# Command limits. Wheel speed is capped well below what the actuator could do:
# a 483 kg machine on regolith is traction-limited long before it is
# torque-limited, and letting the policy command 30 rad/s just teaches it to
# saturate and spin.
MAX_WHEEL_SPEED = 5.0       # rad/s  -> 1.5 m/s at r = 0.30
MAX_YAW_SPEED = 3.0         # rad/s differential added across the two sides
MAX_DRUM_SPEED = 8.0        # rad/s


EXCAVATOR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Excavator",
    spawn=sim_utils.UsdFileCfg(usd_path=str(EXCAVATOR_USD_PATH)),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, SPAWN_Z),
        # (x, y, z, w) -- Isaac Lab 3.x order. Identity is (0,0,0,1).
        #
        # NOT (1,0,0,0). Under xyzw that value is a 180 degree rotation about
        # X, which spawns the machine inverted. See the note at the bottom of
        # this file: the tricycle cfg still carries it.
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Velocity drive: stiffness 0, damping carries the tracking.
        #
        # Traction-sized, for LUNAR gravity. About 120 kg rests on each wheel,
        # which at 1.62 m/s^2 is ~196 N of normal force, so at mu = 1.0 the
        # wheel can deliver ~196 N before it slips -- roughly 59 N-m at a
        # 0.30 m radius. 120 gives a 2x margin for dynamic loads and for
        # grousers biting better than mu = 1.
        #
        # The earth-gravity number would be ~354 N-m. Leaving it there would
        # not make the machine stronger, it would just let the policy command
        # torque the ground cannot react against, and the only thing it would
        # learn is to spin the wheels.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=JOINT_WHEELS,
            effort_limit_sim=120.0,
            velocity_limit_sim=12.0,
            stiffness=0.0,
            damping=25.0,
        ),
        # Position drive.
        #
        # NOT sized by gravity. Holding a full drum out horizontally is only
        # ~210 N-m at lunar gravity (38 N-m empty), but the arm also has to
        # resist the vertical component of digging resistance, and that is set
        # by soil strength rather than weight, so it does not shrink with
        # gravity. The dig load dominates by a wide margin, which is why this
        # is 800 rather than the ~400 a 2x gravity margin would suggest.
        #
        # If the arm visibly sags or gets driven up out of the cut, this is the
        # number to raise -- not the drum torque.
        "arms": ImplicitActuatorCfg(
            joint_names_expr=JOINT_ARMS,
            effort_limit_sim=800.0,
            velocity_limit_sim=1.5,
            stiffness=2500.0,
            damping=250.0,
        ),
        # Velocity drive. Also gravity-independent: cutting torque is set by
        # how hard the regolith is, not by what it weighs. This is the least
        # well-known number in the file, and a drum that stalls in soil is the
        # signal to raise it. Note that raising it does NOT cost traction --
        # that is the point of the counter-rotating pair.
        "drums": ImplicitActuatorCfg(
            joint_names_expr=JOINT_DRUMS,
            effort_limit_sim=350.0,
            velocity_limit_sim=15.0,
            stiffness=0.0,
            damping=40.0,
        ),
    },
)


# ---------------------------------------------------------------------------
# A note on the tricycle, worth checking before the excavator inherits it
# ---------------------------------------------------------------------------
#
# tricycle_env_cfg.py sets `rot=(1.0, 0.0, 0.0, 0.0)`. On this branch
# AssetBaseCfg.InitialStateCfg.rot is documented as "(x, y, z, w)" with a
# default of (0, 0, 0, 1), so (1, 0, 0, 0) is not identity -- it is a 180
# degree rotation about X, i.e. the asset spawns inverted.
#
# That predicts exactly the quirk the README records as a hard-won fact:
#
#     "projected_gravity_b[:, 2] reads +1.0 upright for this converted asset,
#      not -1.0. The flip check is < 0.3, not > -0.3."
#
# Under a 180 degree X rotation the body z axis points down in world, so
# gravity expressed in the body frame comes back as +1 instead of -1. The
# workaround in the env (flipping the termination comparison) treats the
# symptom.
#
# The benign explanation is that the tricycle was written against a pre-
# migration Isaac Lab, where rot really was (w, x, y, z), and the convention
# changed underneath it. Either way it is worth one check on a machine that
# can actually run it: spawn the tricycle, print root_quat_w and
# projected_gravity_b, and see whether it is sitting on its wheels or its roof.
# If it is inverted, `rot=(0,0,0,1)` is the fix and the flip check goes back to
# the conventional `> -0.3`.
#
# The excavator uses (0, 0, 0, 1) and the conventional check, so it does not
# inherit either the rotation or the workaround.
