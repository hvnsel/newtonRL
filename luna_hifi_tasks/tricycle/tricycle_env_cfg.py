# tricycle_env_cfg.py
#
# Barebones pipeline validation: a tricycle drives forward on a flat strip of
# MPM soil for 2 seconds. Reward is forward distance.
#
# Structure follows isaaclab_tasks/contrib/ur10_particle_push, which is the
# reference for running a rigid solver and implicit MPM together:
#   - NewtonCfg takes ONE solver_cfg. Rigid + MPM are two CouplerEntryCfg
#     entries inside a CouplerProxyCfg.
#   - The MPM entry must set in_place=True and all_particles=True.
#   - project_outside_colliders must be False on a coupled MPM entry.
#   - The tool bodies (here: wheels) are a CouplerProxyMappingCfg from the
#     rigid entry to the MPM entry, mode="lagged", with a mass_scale.
#   - physics is assigned in __post_init__, not as a class attribute.
#   - Scene prim_path uses {ENV_REGEX_NS}; coupler bodies use the expanded
#     /World/envs/env_.* form.
#
# The car is loaded from the USD in assets/tricycle/, which is converted
# offline from the MJCF in tricycle.py (see RUNBOOK.md). MJCF cannot be
# imported at runtime: the importer is a Kit extension and the CLI never
# boots Kit.

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
)
from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from .tricycle import CHASSIS_Z, JOINT_REAR, JOINT_STEER
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIGID_ENTRY = "rover"
MPM_ENTRY = "soil"

VOXEL_SIZE = 0.05
MPM_COLLIDER_MARGIN = 0.5 * VOXEL_SIZE
MPM_PARTICLES_PER_CELL = 1.0
MPM_VISUAL_COLOR = (0.62, 0.55, 0.45)

# Soil strip: 1.5 m long, 0.6 m wide, 6 cm deep. With a 5 cm voxel and one
# particle per cell that is 30 x 12 x 2 = 720 particles per env.
STRIP_LENGTH, STRIP_WIDTH, STRIP_DEPTH = 1.5, 0.6, 0.06
STRIP_START_X = -0.4                       # car spawns at x=0, on the strip

STRIP_LOWER = (STRIP_START_X, -0.5 * STRIP_WIDTH, MPM_COLLIDER_MARGIN)
STRIP_UPPER = (STRIP_START_X + STRIP_LENGTH, 0.5 * STRIP_WIDTH, MPM_COLLIDER_MARGIN + STRIP_DEPTH)

# Hidden slab that gives the MPM entry a floor. Without it particles fall
# forever -- the MPM entry only sees bodies listed on its CouplerEntryCfg,
# not the global ground plane.
MPM_GROUND_SIZE = (STRIP_LENGTH + 1.0, STRIP_WIDTH + 1.0, 0.10)
MPM_GROUND_POSITION = (STRIP_START_X + 0.5 * STRIP_LENGTH, 0.0, -0.05)

# ---------------------------------------------------------------------------
# Tricycle
# ---------------------------------------------------------------------------

# <repo>/assets/tricycle/tricycle.usda, found relative to this file so the
# repo can live anywhere. Regenerate it with the converter after editing
# tricycle.py (RUNBOOK.md, "Convert the tricycle asset").
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
TRICYCLE_USD_PATH = ASSETS_DIR / "tricycle" / "tricycle.usda"

TRICYCLE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Tricycle",
    spawn=sim_utils.UsdFileCfg(usd_path=str(TRICYCLE_USD_PATH)),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, CHASSIS_Z + MPM_COLLIDER_MARGIN + STRIP_DEPTH + 0.01),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "steer": ImplicitActuatorCfg(
            joint_names_expr=[JOINT_STEER],
            effort_limit_sim=5.0,
            velocity_limit_sim=5.0,
            stiffness=20.0,
            damping=1.0,
        ),
        "rear": ImplicitActuatorCfg(
            joint_names_expr=JOINT_REAR,
            effort_limit_sim=2.0,
            velocity_limit_sim=30.0,
            stiffness=0.0,            # velocity drive
            damping=0.5,
        ),
        # front wheel free-rolls on MJCF joint damping -- no actuator
    },
)

# ---------------------------------------------------------------------------
# Soil
# ---------------------------------------------------------------------------

SOIL_CFG = MPMObjectCfg(
    prim_path="{ENV_REGEX_NS}/Soil",
    init_state=MPMObjectCfg.InitialStateCfg(),
    spawn=MPMGridCfg(
        lower=STRIP_LOWER,
        upper=STRIP_UPPER,
        voxel_size=VOXEL_SIZE,
        particles_per_cell=MPM_PARTICLES_PER_CELL,
        particle_placement="cell_center",
        jitter=0.0,
        # Packed ground, not a sandbox: high friction, capped compression.
        material=MPMParticleMaterialCfg(
            density=1800.0,
            friction=0.84,            # tan(40 deg)
            yield_pressure=1.0e12,
        ),
        visual_color=MPM_VISUAL_COLOR,
    ),
)


def _mpm_ground() -> RigidObjectCfg:
    """Hidden kinematic slab so soil has a floor inside the MPM entry."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MPMGround",
        init_state=RigidObjectCfg.InitialStateCfg(pos=MPM_GROUND_POSITION),
        spawn=sim_utils.CuboidCfg(
            size=MPM_GROUND_SIZE,
            rigid_props=UsdPhysicsRigidBodyCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
            ),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=MPM_COLLIDER_MARGIN,
                contact_gap=0.0,
            ),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
            ),
            visible=False,
        ),
    )


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class TricycleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )

    tricycle: ArticulationCfg = TRICYCLE_CFG
    mpm_ground: RigidObjectCfg = _mpm_ground()
    soil: MPMObjectCfg = SOIL_CFG


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@configclass
class TricycleEnvCfg(DirectRLEnvCfg):
    decimation = 2                      # policy at 50 Hz on a 100 Hz sim
    episode_length_s = 2.0              # "how far in 2 seconds"

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 100.0, render_interval=decimation)

    scene: TricycleSceneCfg = TricycleSceneCfg(
        num_envs=8,
        env_spacing=2.0,
        replicate_physics=True,
    )

    # actions: [throttle, steer] in [-1, 1]
    action_space = 2
    max_wheel_speed = 20.0              # rad/s at throttle = 1  (~2 m/s at r = 0.1)
    max_steer = 0.5                     # rad at steer = +-1

    # obs: lin vel b (3) + ang vel b (3) + proj gravity (3)
    #    + steer pos (1) + steer vel (1) + rear wheel vel (2) + last action (2)
    observation_space = 15
    state_space = 0

    # termination
    max_lateral_drift = 0.5             # m; off the strip sideways
    flip_gravity_z = 0.3               # projected gravity z above this => flipped

    # Scales the wheel inertia MPM sees. Stability knob for the lagged
    # feedback -- lower it if the car starts chattering on the strip.
    proxy_mass_scale: float = 10.0


    def __post_init__(self) -> None:
        self.sim.physics = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name=RIGID_ENTRY,
                        solver_cfg=MJWarpSolverCfg(
                            use_mujoco_contacts=False,
                            integrator="implicitfast",
                            njmax=128,
                            nconmax=128,
                        ),
                        bodies=[r"/World/envs/env_.*/Tricycle"],
                        include_static_shapes=True,
                        substeps=2,
                    ),
                    CouplerEntryCfg(
                        name=MPM_ENTRY,
                        solver_cfg=MPMSolverCfg(
                            voxel_size=VOXEL_SIZE,
                            grid_type="sparse",
                            grid_padding=0,
                            strain_basis="P0",
                            transfer_scheme="apic",
                            max_iterations=24,
                            tolerance=1.0e-4,
                            warmstart_mode="auto",
                            velocity_basis="Q1",
                            collider_basis="pic27",
                            collider_velocity_mode="forward",
                            solver="auto",
                            separate_worlds=True,
                            # Must be False on a coupled entry.
                            project_outside_colliders=False,
                            max_active_cell_count=1 << 14,
                            max_leaf_node_count=1 << 13,
                            max_lower_node_count=1 << 12,
                            max_upper_node_count=1 << 10,
                        ),
                        bodies=[r"/World/envs/env_.*/MPMGround"],
                        all_particles=True,
                        include_static_shapes=False,
                        include_child_joints=False,
                        substeps=1,
                        in_place=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source=RIGID_ENTRY,
                        destination=MPM_ENTRY,
                        bodies=[r"/World/envs/env_.*/Tricycle/Geometry/chassis/.*_body"],
                        mode="lagged",
                        mass_scale=self.proxy_mass_scale,
                        collision_pipeline=None,
                    )
                ],
                iterations=1,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
            num_substeps=1,
            use_cuda_graph=True,
        )
        self.sim.use_newton_actuators = True

    def play_mode(self) -> None:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 4)
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                eye=(2.0, 2.0, 1.5),
                lookat=(0.5, 0.0, 0.0),
            )
        ]