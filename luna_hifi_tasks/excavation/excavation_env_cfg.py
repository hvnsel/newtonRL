# excavation_env_cfg.py
#
# Config for the rover excavation task. Everything the policy, the scene, and
# the physics need to know lives here; the env class reads it.
#
# STUB STATUS -- search for "STUB" and "VERIFY":
#   STUB    a placeholder value or shape you will replace
#   VERIFY  an isaaclab_newton API name I could not confirm against the
#           develop branch at time of writing. Check it in your checkout.
#
# Assumed rover layout (you didn't specify -- change the regexes):
#   4 driven wheels     wheel_(fl|fr|rl|rr)_joint        velocity-controlled
#   3 arm joints        arm_(shoulder|elbow|bucket)_joint position-controlled
#   bucket link         "bucket"                          the scoop, own link

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# VERIFY: these import paths. NewtonCfg/MJWarpSolverCfg are documented at
# isaaclab_newton.physics. MPMSolverCfg, MPMObjectCfg and MPMGridCfg are what
# the "Using Implicit MPM" page and the Franka-Pour task use -- confirm the
# module they live in.
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg

from .soil import DENSE_REGOLITH

# ---------------------------------------------------------------------------
# Rover -- STUB. Point usd_path at your asset and fix the joint regexes.
# ---------------------------------------------------------------------------

ROVER_USD_PATH = "/PATH/TO/rover.usd"   # STUB

ROVER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Rover",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROVER_USD_PATH,
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.0, 0.0, 0.35),           # STUB: behind the soil bed, wheels clear of the surface
        rot=(1.0, 0.0, 0.0, 0.0),         # (w, x, y, z)
        joint_pos={
            "wheel_.*_joint": 0.0,
            "arm_shoulder_joint": 0.6,     # STUB: arm raised, bucket clear of ground
            "arm_elbow_joint": -1.2,
            "arm_bucket_joint": 0.4,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Velocity drive: zero stiffness, damping is the velocity gain.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*_joint"],
            effort_limit_sim=40.0,          # STUB: N*m per wheel
            velocity_limit_sim=20.0,        # STUB: rad/s
            stiffness=0.0,
            damping=5.0,
        ),
        # Position drive: PD to a joint target.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_.*_joint"],
            effort_limit_sim=200.0,         # STUB
            velocity_limit_sim=3.0,
            stiffness=400.0,
            damping=40.0,
        ),
    },
)

# ---------------------------------------------------------------------------
# Soil -- one MPM bed per env. separate_worlds=True keeps grids independent so
# a bucket in env 3 can't push sand in env 4, and it's what makes per-env reset
# possible at all (a shared grid can't reset a subset of worlds).
# ---------------------------------------------------------------------------

SOIL_BED_SIZE = (1.2, 1.2, 0.30)       # x, y, depth  (m)
SOIL_BED_ORIGIN = (0.4, 0.0, 0.0)       # bed centre in env-local xy, floor at z=0
VOXEL_SIZE = 0.03

SOIL_CFG = MPMObjectCfg(
    prim_path="/World/envs/env_.*/Soil",
    spawn=MPMGridCfg(                                        # VERIFY field names
        size=SOIL_BED_SIZE,
        particles_per_cell=2,                                # per axis -> 8 per voxel
        material=DENSE_REGOLITH.to_material_cfg(),
    ),
    init_state=MPMObjectCfg.InitialStateCfg(
        pos=(SOIL_BED_ORIGIN[0], SOIL_BED_ORIGIN[1], SOIL_BED_SIZE[2] * 0.5),
    ),
)

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class ExcavationSceneCfg(InteractiveSceneCfg):
    ground = sim_utils.GroundPlaneCfg()
    rover: ArticulationCfg = ROVER_CFG
    soil: MPMObjectCfg = SOIL_CFG
    light = sim_utils.DomeLightCfg(intensity=2000.0)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@configclass
class RoverExcavationEnvCfg(DirectRLEnvCfg):
    # -- timing
    decimation = 4                      # policy at 25 Hz on a 100 Hz sim
    episode_length_s = 20.0

    # -- physics
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=decimation,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=500,
                nconmax=200,
                cone="elliptic",
                iterations=20,
                ls_iterations=50,
            ),
            # VERIFY: how NewtonCfg attaches the MPM solver alongside the rigid
            # one. The MPM docs show NewtonCfg(solver_cfg=MPMSolverCfg(...));
            # Franka-Pour runs an arm and MPM together, so there is a way to
            # hold both -- copy whatever that task's cfg does.
            mpm_solver_cfg=MPMSolverCfg(
                voxel_size=VOXEL_SIZE,
                max_iterations=50,
                tolerance=1.0e-4,
                project_outside_colliders=True,
                separate_worlds=True,
            ),
            num_substeps=2,
        ),
    )

    scene: ExcavationSceneCfg = ExcavationSceneCfg(num_envs=1024, env_spacing=4.0)

    # -- rover joint groups (must match ROVER_CFG.actuators)
    wheel_joint_names = ["wheel_.*_joint"]
    arm_joint_names = ["arm_.*_joint"]
    bucket_body_name = "bucket"

    # -- actions: [wheel velocity targets (4), arm joint deltas (3)]
    #    both in [-1, 1]; scaling happens in the env
    num_wheels = 4
    num_arm_joints = 3
    action_space = num_wheels + num_arm_joints
    max_wheel_speed = 10.0              # rad/s at action = +-1
    arm_action_scale = 0.1              # rad of joint delta per step at action = +-1

    # -- observations
    #    proprio: base lin vel (3) + ang vel (3) + projected gravity (3)
    #             + joint pos (7) + joint vel (7) + bucket pos/quat in base (7)
    #             + last action (7)
    #    height-map: HEIGHT_MAP_ROWS * HEIGHT_MAP_COLS
    #    bucket fill: 1
    height_map_rows = 16
    height_map_cols = 16
    height_map_cell = 0.075             # m; 16 cells * 0.075 = 1.2 m covers the bed
    height_map_origin = (
        SOIL_BED_ORIGIN[0] - SOIL_BED_SIZE[0] * 0.5,
        SOIL_BED_ORIGIN[1] - SOIL_BED_SIZE[1] * 0.5,
    )
    num_proprio_obs = 3 + 3 + 3 + 7 + 7 + 7 + 7
    observation_space = num_proprio_obs + height_map_rows * height_map_cols + 1
    state_space = 0

    # -- bucket cavity, in the bucket link's local frame (STUB: measure yours)
    bucket_cavity_lo = (-0.10, -0.15, 0.00)
    bucket_cavity_hi = (0.10, 0.15, 0.12)

    # -- reward weights (STUB: the terms themselves are stubs in the env)
    rew_scale_fill = 1.0
    rew_scale_lift = 0.5
    rew_scale_effort = -0.001
    rew_scale_action_rate = -0.01
    rew_scale_alive = 0.0

    # -- termination
    max_tilt_cos = 0.5                  # projected gravity z above this => fallen over
    fill_success_kg = 2.0               # STUB: end episode once this much is scooped and lifted
