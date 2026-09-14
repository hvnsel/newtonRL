# excavate_env_cfg.py
#
# Excavation on the MPM tier: an implicit-MPM regolith bed, the machine
# parked on top of it, both drums cutting as it drives. Tens of envs, not
# thousands -- this is where the simulation budget goes, because granular flow
# into a drum is the physics being learned and nothing cheaper reproduces it.
#
# The rigid <-> MPM coupling follows tricycle_env_cfg.py line for line, which
# is the reference that has actually run in this repo. The rules it encodes
# (one solver per NewtonCfg, MPM entry in_place + all_particles, no
# project_outside_colliders on a coupled entry, tool bodies as a lagged proxy
# mapping with a mass_scale, soft_contact_max=0) are all load-bearing.
#
# Gravity is lunar. See excavator_cfg.py.

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
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

from ..excavator import ARM_RANGE, DRUM_HALF_LEN, DRUM_RADIUS, DRUM_WALL_T
from ..excavator_cfg import (
    EXCAVATOR_CFG,
    EXCAVATOR_PRIM_REGEX,
    LUNAR_GRAVITY,
    SOIL_CONTACT_BODIES_REGEX,
    SPAWN_Z,
)
from ..mdp.observations import NAV_SCAN_CELLS, critic_state_spec, excavate_obs_spec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIGID_ENTRY = "rover"
MPM_ENTRY = "soil"

# 5 cm voxels are what the tricycle validated. The drum bore is 34 cm across,
# i.e. seven cells -- coarse for resolving fill, and the single place in this
# machine where MPM resolution binds. Dropping to 0.03 triples the particle
# count; budget for it once the pipeline runs at 0.05.
VOXEL_SIZE = 0.05
MPM_COLLIDER_MARGIN = 0.5 * VOXEL_SIZE
MPM_PARTICLES_PER_CELL = 1.0
MPM_VISUAL_COLOR = (0.62, 0.55, 0.45)

# Bed, in the env frame. The machine spawns at the origin ON the bed, facing
# +x, so both drums start over soil: the front one meets fresh regolith as it
# drives forward and the rear one trails through the trench. Long enough for
# a full drum at the drive speed the actuators allow, wider than the 1.35 m
# machine, deeper than the 0.19 m cut.
BED_X = (-3.0, 5.0)
BED_Y = (-1.0, 1.0)
BED_DEPTH = 0.25
BED_LOWER = (BED_X[0], BED_Y[0], MPM_COLLIDER_MARGIN)
BED_UPPER = (BED_X[1], BED_Y[1], MPM_COLLIDER_MARGIN + BED_DEPTH)
BED_TOP = MPM_COLLIDER_MARGIN + BED_DEPTH
BED_LEN = BED_X[1] - BED_X[0]
BED_WID = BED_Y[1] - BED_Y[0]

# Height grid the particle scans are rasterised onto: the bed's own footprint
# at the MPM voxel size. Empty cells read the slab top (z = 0) -- the floor an
# excavated cell bottoms out on -- so dug ground and untouched ground never
# look alike.
BED_GRID_LOWER = (BED_X[0], BED_Y[0])
BED_GRID_CELL = VOXEL_SIZE
BED_GRID_NX = int(math.ceil(BED_LEN / VOXEL_SIZE))
BED_GRID_NY = int(math.ceil(BED_WID / VOXEL_SIZE))
BED_FLOOR_Z = 0.0

# Hidden kinematic slab giving the MPM entry a floor. The MPM entry only sees
# bodies listed on its CouplerEntryCfg, not the global ground plane.
MPM_GROUND_SIZE = (BED_LEN + 1.0, BED_WID + 1.0, 0.10)
MPM_GROUND_POSITION = (0.5 * (BED_X[0] + BED_X[1]), 0.0, -0.05)

# Validated regolith parameters from the tricycle. friction ~ tan(phi),
# yield_stress is cohesion in Pa (left at the default 0 -- the main knob to
# tune once digging runs). yield_pressure caps compression: packed ground,
# not a sandbox.
SOIL_MATERIAL = MPMParticleMaterialCfg(
    density=1800.0,
    friction=0.84,
    yield_pressure=1.0e12,
)

BORE_RADIUS = DRUM_RADIUS - DRUM_WALL_T
BORE_HALF_LEN = DRUM_HALF_LEN
BORE_VOLUME = math.pi * BORE_RADIUS ** 2 * (2.0 * BORE_HALF_LEN)
DRUM_CAPACITY_KG = BORE_VOLUME * SOIL_MATERIAL.density      # ~155 kg

DIG_OBS = excavate_obs_spec()
DIG_CRITIC = critic_state_spec(terrain_cells=NAV_SCAN_CELLS)


def _particles_per_env() -> int:
    """Same arithmetic as the MPM spawner: per-axis ceil of extent/voxel."""
    n = 1
    for lo, hi in zip(BED_LOWER, BED_UPPER):
        n *= max(int(math.ceil(MPM_PARTICLES_PER_CELL * (hi - lo) / VOXEL_SIZE)), 1)
    return n


PARTICLES_PER_ENV = _particles_per_env()


def _next_pow2(n: int) -> int:
    return 1 << max(int(n - 1).bit_length(), 1)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

SOIL_CFG = MPMObjectCfg(
    prim_path="{ENV_REGEX_NS}/Soil",
    init_state=MPMObjectCfg.InitialStateCfg(),
    spawn=MPMGridCfg(
        lower=BED_LOWER,
        upper=BED_UPPER,
        voxel_size=VOXEL_SIZE,
        particles_per_cell=MPM_PARTICLES_PER_CELL,
        particle_placement="cell_center",
        jitter=0.0,
        material=SOIL_MATERIAL,
        visual_color=MPM_VISUAL_COLOR,
    ),
)


def _mpm_ground() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MPMGround",
        init_state=RigidObjectCfg.InitialStateCfg(pos=MPM_GROUND_POSITION),
        spawn=sim_utils.CuboidCfg(
            size=MPM_GROUND_SIZE,
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=MPM_COLLIDER_MARGIN,
                contact_gap=0.0,
            ),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.8, dynamic_friction=0.7),
            visible=False,
        ),
    )


# Spawn on TOP of the bed, not inside it: the shared cfg puts the wheels on
# z = 0, which here is a quarter metre of regolith.
EXCAVATOR_ON_BED_CFG = EXCAVATOR_CFG.replace(
    init_state=EXCAVATOR_CFG.init_state.replace(pos=(0.0, 0.0, SPAWN_Z + BED_TOP)),
)


@configclass
class ExcavateSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )

    excavator: ArticulationCfg = EXCAVATOR_ON_BED_CFG
    mpm_ground: RigidObjectCfg = _mpm_ground()
    soil: MPMObjectCfg = SOIL_CFG


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@configclass
class ExcavatorExcavateEnvCfg(DirectRLEnvCfg):
    decimation = 2                      # 100 Hz sim, 50 Hz policy -- the validated MPM rate
    episode_length_s = 12.0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=decimation,
        gravity=LUNAR_GRAVITY,
    )

    scene: ExcavateSceneCfg = ExcavateSceneCfg(
        num_envs=16,
        env_spacing=12.0,               # beds are 8 m long; keep origins apart
        replicate_physics=True,
    )

    # The sparse-grid capacities below are ABSOLUTE totals across all envs and
    # do not scale with --num_envs (Hydra applies that after __post_init__).
    # They are sized here for max_num_envs, and the env asserts at start-up
    # that num_envs does not exceed it. Raise this, not the caps directly.
    max_num_envs = 32

    # actions: [forward, yaw, boom, drum] in [-1, 1]. Boom and drum are one
    # command each, applied to BOTH arms / drums -- the counter-rotating dig.
    action_space = 4
    observation_space = DIG_OBS.dim      # 153
    state_space = DIG_CRITIC.dim         # 199

    # --- start state ---
    arm_start_angle = -0.20              # drums ~0.44 m up, clear of the bed
    spawn_yaw_jitter = 0.15              # rad
    spawn_x_jitter = 0.30                # m

    # --- actions ---
    action_smoothing = 0.3
    arm_range = ARM_RANGE

    # --- success ---
    fill_success_fraction = 0.8
    # "mean": the pair averages past the threshold. "all": every drum must.
    # Mean is the default because the rear drum trails through the trench the
    # front one cut and fills more slowly; requiring both to be full would
    # make the front drum overfill waiting for it.
    fill_success_mode = "mean"

    # --- reward weights ---
    w_fill = 10.0                        # per full drum-PAIR of captured soil
    w_success = 20.0
    w_stall = 0.5
    w_drift = 1.0                        # the counter-rotation check
    w_upright = 2.0
    w_idle_drum = 0.02
    w_energy = 1.0e-4
    w_action_rate = 0.05
    w_time = 0.02

    # --- termination ---
    max_tilt_rad = math.radians(60.0)
    max_env_excursion = 6.0

    # --- scan ---
    scan_clip = 1.0

    # Scales the wheel and drum inertia MPM sees; the lagged-feedback
    # stability knob. Lower it if the machine chatters on the bed.
    proxy_mass_scale: float = 10.0

    def __post_init__(self) -> None:
        total_particles = PARTICLES_PER_ENV * self.max_num_envs
        active = _next_pow2(3 * total_particles)
        print(
            f"[excavate] {PARTICLES_PER_ENV} particles/env x {self.max_num_envs} max envs "
            f"= {total_particles}; sparse grid active={active}"
        )

        self.sim.physics = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name=RIGID_ENTRY,
                        solver_cfg=MJWarpSolverCfg(
                            use_mujoco_contacts=False,
                            integrator="implicitfast",
                            njmax=200,
                            nconmax=96,
                        ),
                        bodies=[EXCAVATOR_PRIM_REGEX],
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
                            # upper <= lower <= leaf <= active
                            max_active_cell_count=active,
                            max_leaf_node_count=active >> 1,
                            max_lower_node_count=active >> 2,
                            max_upper_node_count=active >> 4,
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
                        bodies=[SOIL_CONTACT_BODIES_REGEX],
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
        self.scene.num_envs = min(self.scene.num_envs, 2)
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                eye=(4.0, 4.0, 2.5),
                lookat=(1.0, 0.0, 0.0),
            )
        ]
