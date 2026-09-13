# excavator_cfg.py
#
# The shared ArticulationCfg. Both the navigate and excavate tasks spawn the
# same machine with the same actuators; only observations, actions and rewards
# differ between them. Keeping the asset in one place is what makes a policy
# trained on one task loadable in the other.
#
# Actuator effort limits are sized from the numbers the asset validation
# produced -- see excavator.reach() and the mass table in excavator.py. They
# are not guesses, and the arm limit in particular is sized for a FULL drum,
# not an empty one.

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
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Velocity drive: stiffness 0, damping carries the tracking.
        # Effort limit is traction-sized -- roughly 120 kg on each wheel at
        # mu = 1.0 gives ~1180 N, which is ~354 N-m at a 0.30 m radius.
        # Anything past that just spins the wheel.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=JOINT_WHEELS,
            effort_limit_sim=400.0,
            velocity_limit_sim=12.0,
            stiffness=0.0,
            damping=60.0,
        ),
        # Position drive. Sized for the LOADED case: holding a full drum out
        # horizontally needs ~1270 N-m at earth gravity (230 N-m empty).
        # Sizing this off the empty number is the classic way to end up with
        # arms that sag the moment the drum fills.
        "arms": ImplicitActuatorCfg(
            joint_names_expr=JOINT_ARMS,
            effort_limit_sim=1600.0,
            velocity_limit_sim=1.5,
            stiffness=4000.0,
            damping=400.0,
        ),
        # Velocity drive. Excavation torque is the least well-known number
        # here; this is a starting point, and a drum that stalls in soil is
        # the signal to raise it.
        "drums": ImplicitActuatorCfg(
            joint_names_expr=JOINT_DRUMS,
            effort_limit_sim=350.0,
            velocity_limit_sim=15.0,
            stiffness=0.0,
            damping=40.0,
        ),
    },
)
