# The excavator ArticulationCfg, its actuator limits, prim paths and gravity,
# shared by navigate and excavate.

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .excavator import (
    ARM_RANGE,
    CHASSIS_Z,
    JOINT_ARMS,
    JOINT_DRUMS,
    JOINT_SHROUDS,
    JOINT_WHEELS,
    ROTOR_TIP_R,
    source_hash,
    source_stamp_path,
)

# The USD converted from excavator.py's MJCF (MjcfConverter writes
# <output dir>/excavator/excavator.usda).
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
EXCAVATOR_USD_PATH = ASSETS_DIR / "excavator" / "excavator.usda"

# Prim paths. The converter nests bodies under Geometry/<root body>.
EXCAVATOR_PRIM = "{ENV_REGEX_NS}/Excavator"
CHASSIS_PRIM = EXCAVATOR_PRIM + "/Geometry/chassis"
EXCAVATOR_PRIM_REGEX = r"/World/envs/env_.*/Excavator"
_BODIES = r"/World/envs/env_.*/Excavator/Geometry/chassis/.*"
# Bodies coupled to the MPM soil: wheels, rotors and shrouds; wheels alone;
# rotors and shrouds alone.
SOIL_CONTACT_BODIES_REGEX = _BODIES + r"(wheel_.._body|drum_(front|rear)_body|shroud_(front|rear)_body)"
WHEEL_CONTACT_BODIES_REGEX = _BODIES + r"(wheel_.._body)"
DRUM_CONTACT_BODIES_REGEX = _BODIES + r"(drum_(front|rear)_body|shroud_(front|rear)_body)"

# Chassis spawn height: wheels on z = 0 plus 2 cm.
SPAWN_Z = CHASSIS_Z + 0.02
ARM_STOW_ANGLE = ARM_RANGE[0]

LUNAR_GRAVITY = (0.0, 0.0, -1.62)
LUNAR_G = 1.62

MAX_WHEEL_SPEED = 5.0       # rad/s, 1.5 m/s at r = 0.30
MAX_YAW_SPEED = 3.0         # rad/s
MAX_DRUM_SPEED = 2.9        # rad/s
# Drum speed at which the vane tips' centripetal acceleration equals lunar gravity.
SPIN_LIMIT = math.sqrt(LUNAR_G / ROTOR_TIP_R)

# Actuator effort limits, N m. Observations divide joint torque by these.
WHEEL_EFFORT = 2000.0
ARM_EFFORT = 360.0
DRUM_EFFORT = 350.0
SHROUD_EFFORT = 200.0


def usd_status() -> str | None:
    """None if the converted asset exists and was built from the current
    excavator.py (its source stamp matches), else how to rebuild it."""
    stamp = source_stamp_path()
    if EXCAVATOR_USD_PATH.exists() and stamp.exists() and stamp.read_text().strip() == source_hash():
        return None
    return (
        f"the excavator asset at {EXCAVATOR_USD_PATH} is missing or was not built from the current excavator.py;\n"
        "  rebuild it (from the Isaac Lab directory):\n"
        '    isaaclab -p -c "from luna_hifi_tasks.excavator.excavator import write_mjcf; print(write_mjcf())"\n'
        f"    rm -rf {ASSETS_DIR / 'excavator'}\n"
        f"    isaaclab -p scripts/tools/convert_mjcf.py <that path> {ASSETS_DIR / 'excavator.usd'}"
    )


EXCAVATOR_CFG = ArticulationCfg(
    prim_path=EXCAVATOR_PRIM,
    spawn=sim_utils.UsdFileCfg(usd_path=str(EXCAVATOR_USD_PATH)),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, SPAWN_Z),
        rot=(0.0, 0.0, 0.0, 1.0),           # (x, y, z, w)
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Velocity drive: torque = damping * velocity error, capped at the effort limit.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=JOINT_WHEELS,
            effort_limit_sim=WHEEL_EFFORT,
            velocity_limit_sim=12.0,
            stiffness=0.0,
            damping=400.0,
        ),
        # Position drive.
        "arms": ImplicitActuatorCfg(
            joint_names_expr=JOINT_ARMS,
            effort_limit_sim=ARM_EFFORT,
            velocity_limit_sim=1.5,
            stiffness=2500.0,
            damping=250.0,
        ),
        # Velocity drive.
        "drums": ImplicitActuatorCfg(
            joint_names_expr=JOINT_DRUMS,
            effort_limit_sim=DRUM_EFFORT,
            velocity_limit_sim=6.0,
            stiffness=0.0,
            damping=40.0,
        ),
        # Position drive.
        "shrouds": ImplicitActuatorCfg(
            joint_names_expr=JOINT_SHROUDS,
            effort_limit_sim=SHROUD_EFFORT,
            velocity_limit_sim=2.0,
            stiffness=1500.0,
            damping=150.0,
        ),
    },
)
