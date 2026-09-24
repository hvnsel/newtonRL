# excavate_env_cfg.py
#
# Excavation on the MPM tier: an implicit-MPM regolith bed, the machine parked
# on top of it, both drums cutting as it drives. Tens of envs, not thousands.
#
# Coupling rules, all load-bearing: one solver per NewtonCfg, MPM entry
# in_place + all_particles, no project_outside_colliders on a coupled entry,
# tool bodies as a lagged proxy mapping with a mass_scale, soft_contact_max=0.
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

from ..excavator import ARM_RANGE, ROTOR_HALF_LEN, ROTOR_TIP_R, SHROUD_OUT_R, SHROUD_RANGE
from ..excavator_cfg import (
    EXCAVATOR_CFG,
    EXCAVATOR_PRIM_REGEX,
    LUNAR_GRAVITY,
    FRONT_DRUM_ONLY_REGEX,
    SOIL_CONTACT_BODIES_REGEX,
    SPAWN_Z,
)
from ..mdp.observations import (
    DIG_SCAN_CELL,
    DIG_SCAN_NX,
    DIG_SCAN_NY,
    NAV_SCAN_CELLS,
    critic_state_spec,
    excavate_obs_spec,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIGID_ENTRY = "rover"
MPM_ENTRY = "soil"

# The way into a pocket is the 0.185 m chord between adjacent vane tips, which
# at 0.03 is 5.2 grains clear once the coupler has taken its voxel. At 0.05 it
# is 2.7 and regolith arches across it.
VOXEL_SIZE = 0.03
MPM_COLLIDER_MARGIN = 0.5 * VOXEL_SIZE
MPM_PARTICLES_PER_CELL = 1.0
MPM_VISUAL_COLOR = (0.62, 0.55, 0.45)

# The bed extent lives on the env cfg as bed_x / bed_y / bed_depth, and
# __post_init__ rewrites the scene from those: particle count, sparse-grid
# capacities, the height grid, the floor slab and the spawn height all follow.
#
# Empty height-grid cells read BED_FLOOR_Z, the top of the slab an excavated
# cell bottoms out on.
BED_FLOOR_Z = 0.0

# friction ~ tan(phi); yield_stress is cohesion in Pa, left at 0; and
# yield_pressure caps compression.
SOIL_MATERIAL = MPMParticleMaterialCfg(
    density=1800.0,
    friction=0.84,
    yield_pressure=1.0e12,
)

BORE_RADIUS = ROTOR_TIP_R
BORE_HALF_LEN = ROTOR_HALF_LEN

# The ground the drum is working. Scan cells inside this rectangle, in the
# drum's own frame, are the ones a cut has to bring down to the target height.
FOOTPRINT_HALF_X = SHROUD_OUT_R
FOOTPRINT_HALF_Y = ROTOR_HALF_LEN
BORE_VOLUME = math.pi * BORE_RADIUS ** 2 * (2.0 * BORE_HALF_LEN)
# The mass the rotor's swept cylinder would hold if it packed solid. Reported
# at startup and nothing else: a pocket open at the rim holds a fraction of
# it, so target_load_kg is what fill is normalised and scored against.
DRUM_CAPACITY_KG = BORE_VOLUME * SOIL_MATERIAL.density

DIG_OBS = excavate_obs_spec()
DIG_CRITIC = critic_state_spec(terrain_cells=NAV_SCAN_CELLS)


def particles_per_env(lower, upper, voxel: float, per_cell: float) -> int:
    """Same arithmetic as the MPM spawner: per-axis ceil of extent/voxel."""
    n = 1
    for lo, hi in zip(lower, upper):
        n *= max(int(math.ceil(per_cell * (hi - lo) / voxel)), 1)
    return n


def _next_pow2(n: int) -> int:
    return 1 << max(int(n - 1).bit_length(), 1)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

# Placeholder extents. __post_init__ overwrites lower/upper/voxel_size from the
# cfg fields before anything spawns, so these values are never the ones used.
SOIL_CFG = MPMObjectCfg(
    prim_path="{ENV_REGEX_NS}/Soil",
    init_state=MPMObjectCfg.InitialStateCfg(),
    spawn=MPMGridCfg(
        lower=(-1.0, -1.0, MPM_COLLIDER_MARGIN),
        upper=(1.0, 1.0, MPM_COLLIDER_MARGIN + 0.25),
        voxel_size=VOXEL_SIZE,
        particles_per_cell=MPM_PARTICLES_PER_CELL,
        particle_placement="cell_center",
        jitter=0.0,
        material=SOIL_MATERIAL,
        visual_color=MPM_VISUAL_COLOR,
    ),
)


def _mpm_ground() -> RigidObjectCfg:
    """Hidden kinematic slab giving the MPM entry a floor. The MPM entry only
    sees bodies listed on its CouplerEntryCfg, not the global ground plane, so
    without this the particles fall forever. Size and position are rewritten in
    __post_init__ to match the bed."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MPMGround",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 3.0, 0.10),
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

    excavator: ArticulationCfg = EXCAVATOR_CFG
    mpm_ground: RigidObjectCfg = _mpm_ground()
    soil: MPMObjectCfg = SOIL_CFG


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@configclass
class ExcavatorExcavateEnvCfg(DirectRLEnvCfg):
    # 100 Hz sim, 25 Hz policy, 500 steps per episode. A loaded cut runs
    # 10-20 s, and gamma in agents/rsl_rl_ppo_cfg.py is set against that.
    decimation = 4
    episode_length_s = 20.0

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

    # --- bed, in the env frame ---
    #
    # The machine spawns at the origin facing +x. With spawn_on_bed the wheels
    # rest on the bed surface and both drums start over soil: the front one
    # meets fresh regolith as it drives, the rear trails through the trench.
    bed_x: tuple[float, float] = (-3.0, 5.0)
    bed_y: tuple[float, float] = (-1.0, 1.0)
    bed_depth: float = 0.25
    # Decides whether the rotor can fill. The coupler inflates every collider
    # by half a voxel per side, eating a whole voxel out of every passage, and
    # particles spawn one voxel apart. The gap between vane tips is 0.185 m:
    # 5.2 grains clear at 0.03, 2.7 at 0.05. Granular material arches below
    # about four. Halving the voxel multiplies the particle count by eight.
    voxel_size: float = VOXEL_SIZE
    particles_per_cell: float = MPM_PARTICLES_PER_CELL
    spawn_on_bed: bool = True

    # Sparse-grid active cells per particle. ABSOLUTE totals over all envs.
    # Overrunning them is an illegal access -- a CUDA 700 storm, or 0xC0000374
    # on Windows -- not an out-of-memory error.
    #
    # A cell is tens of bytes, so 2^20 of them is under 100 MB. 8 is the
    # tricycle's ratio; raising it to 24 changed nothing, so the caps are not
    # what this machine hits.
    grid_cap_multiplier: float = 8.0

    # Which bodies are coupled to the soil. Each body brings every one of its
    # geoms; the drums are 38 apiece.
    soil_contact_regex: str = SOIL_CONTACT_BODIES_REGEX

    # Rigid-solver buffers, per env. Fixed size, and overrunning one is an
    # illegal memory access -- a CUDA 700 storm, or 0xC0000374 on Windows --
    # not a clean error.
    #
    # Contacts scale with collider count and with how much soil touches them.
    # The shrouded rotor carries 136 geoms, 106 of them coupled, against the
    # tricycle's 8. A standalone Warp MPM sample has no rigid solver and so
    # neither buffer, which is why it reaches particle counts a coupled scene
    # does not.
    #
    # A contact is a couple of hundred bytes, so 32768 is about 6.6 MB. There
    # is no reason to be tight here and overrunning one is an illegal access.
    rigid_njmax: int = 8192
    rigid_nconmax: int = 32768

    # Active cells per leaf node, as a right shift: 7 is one leaf per 128
    # cells, four times the minimum for 8^3 blocks. Each step down costs 512
    # cells of storage.
    grid_leaf_shift: int = 7



    # --- regolith ---
    #
    # The yield surface is
    #     tau_max(p) = yield_stress + friction * (p - p_min)
    # so friction is ~tan(phi) and soil_cohesion is cohesion in Pa.
    # Drucker-Prager alpha numbers do not port across.
    #
    # At zero cohesion the regolith is dry sand: it shears off the lip and
    # flows back out of the mouth it entered. A few hundred Pa and the cut
    # travels as a clod the lifters can carry round. Lunar simulants sit
    # around 0.1-1 kPa.
    #
    # The blades sweep r = 0.053 to 0.246 m and fill is counted inside the
    # r = 0.170 m bore, so the lifters reach 0.117 m into the measured volume
    # and churn a non-cohesive soil every half turn.
    soil_density: float = 1800.0
    soil_friction: float = 0.84              # tan(40 deg)
    soil_cohesion: float = 0.0               # yield_stress, Pa
    soil_yield_pressure: float = 1.0e12      # packed ground, not a sandbox

    # Derived in __post_init__ from the four fields above, and read off the cfg
    # rather than imported as module constants.
    bed_top: float = 0.0
    bed_grid_lower: tuple[float, float] = (0.0, 0.0)
    bed_grid_nx: int = 0
    bed_grid_ny: int = 0
    bed_particles_per_env: int = 0
    drum_capacity_kg: float = DRUM_CAPACITY_KG
    # Footprint area times the mean commanded depth. The depth reward divides
    # by it, so w_depth is "points for one nominal cut".
    cut_volume_ref: float = 0.04

    # The sparse-grid capacities are absolute totals across all envs and do not
    # scale with --num_envs, which Hydra applies after __post_init__. They are
    # sized for max_num_envs and the env asserts num_envs does not exceed it.
    max_num_envs = 32

    # actions: [forward, yaw, boom, drum, shroud] in [-1, 1]. Boom, drum and
    # shroud are one command each, applied to both ends -- the counter-rotating
    # dig, with the inlet aimed together.
    action_space = 5
    observation_space = DIG_OBS.dim      # 167
    state_space = DIG_CRITIC.dim         # 199

    # --- start state ---
    # Jitter is set from what navigate is allowed to hand over: it declares
    # success at goal_dist_tol 0.5 m in any direction and goal_heading_tol_rad
    # 0.35, so a skill trained inside those numbers meets poses it has never
    # seen, laterally in particular.
    arm_start_angle = -0.20              # drums ~0.44 m up, clear of the bed
    arm_start_jitter = 0.10              # rad
    spawn_yaw_jitter = 0.40              # rad
    spawn_x_jitter = 0.55                # m
    spawn_y_jitter = 0.55                # m

    # Episodes an env keeps its bed before every particle goes back to its
    # spawn cell. The MPM grid is one spawner shared by all envs, so a bed
    # cannot be made to differ at spawn time; letting it carry over is what
    # makes one env's ground differ from another's, and it is the ground the
    # machine will actually meet -- terrain it has already worked.
    soil_reset_every: int = 8

    # --- actions ---
    action_smoothing = 0.3
    arm_range = ARM_RANGE
    shroud_range = SHROUD_RANGE

    # --- cut command ---
    # The planner hands down a target ground height. Here it is sampled per
    # episode, as a depth below the undisturbed bed surface.
    cut_depth_range: tuple[float, float] = (0.04, 0.12)

    # --- load sensor ---
    # What the policy reads in place of the particle count. Boom torque is how
    # a machine weighs its load: a first-order lag, noise proportional to the
    # reading, and a calibration bias held for the episode. The exact figure
    # stays on the critic as drum_fill_mass.
    target_load_kg: float = 40.0
    fill_sensor_tau: float = 0.3             # s
    fill_sensor_noise: float = 0.05          # fraction of the reading
    fill_sensor_abs_kg: float = 0.5
    fill_sensor_bias: float = 0.03           # +- fraction, per episode

    # --- success ---
    # 0.6 of target_load_kg is 24 kg on the front drum. The best recorded run
    # reached 23.6 kg in 30 s.
    fill_success_fraction = 0.6
    # "front": the front drum alone. "all": every drum. "mean": the pair
    # averages past the threshold.
    fill_success_mode = "front"

    # --- reward weights ---
    # Every term is scaled so a whole episode of doing it well is worth tens
    # of points, not fractions. fill and depth are the two halves of the task:
    # fill alone lets the policy scrape one strip forever, depth alone lets it
    # push soil aside and capture none.
    w_fill = 40.0                        # per 2 x target_load_kg captured
    w_depth = 20.0                       # per cut_volume_ref brought to target
    w_overcut = 0.2                      # per cut_volume_ref taken below it
    w_spill = 20.0                       # asymmetry on top of a negative fill
    w_success = 20.0
    w_stall = 0.5
    w_drift = 1.0                        # the counter-rotation check
    w_upright = 2.0
    w_energy = 1.0e-4
    w_action_rate = 0.05
    w_time = 0.005

    # --- termination ---
    max_tilt_rad = math.radians(60.0)
    max_env_excursion = 6.0

    # --- scan ---
    # Actor only; the critic sees the bed clean. Smaller figures than the
    # navigator's because the window is 2 x 1 m at the drum rather than metres
    # out, and scan_invalid sits outside +- scan_clip so a dropped cell cannot
    # be read as a height.
    scan_clip = 1.0
    scan_noise_std = 0.015               # m, at scan_range_ref
    scan_dropout = 0.02                  # probability, at scan_range_ref
    scan_range_ref = 1.0                 # m
    scan_invalid = -2.0

    # Scales the wheel and drum inertia MPM sees; the lagged-feedback
    # stability knob. Lower it if the machine chatters on the bed.
    proxy_mass_scale: float = 10.0

    # The soil_* fields are plain numbers on this cfg; the solver reads an
    # MPMParticleMaterialCfg on the spawner. Hydra applies overrides after
    # __post_init__, so anything setting a soil_* field late must call this
    # again. The env refuses to start if the two disagree.
    def apply_soil_material(self) -> None:
        """Copy the soil_* fields onto the MPM material and the derived capacity."""
        mat = self.scene.soil.spawn.material
        values = {
            "density": self.soil_density,
            "friction": self.soil_friction,
            "yield_stress": self.soil_cohesion,
            "yield_pressure": self.soil_yield_pressure,
        }
        for name, value in values.items():
            # A plain setattr would create an attribute the solver never
            # reads, so a field rename would present as soil parameters having
            # no effect.
            if not hasattr(mat, name):
                raise AttributeError(
                    f"MPMParticleMaterialCfg has no field {name!r} on this Isaac Lab "
                    f"build; it has {sorted(vars(mat))}. Setting it would be a silent "
                    "no-op and the soil would keep its default."
                )
            setattr(mat, name, value)
        self.drum_capacity_kg = BORE_VOLUME * self.soil_density

    def soil_material_mismatch(self) -> str | None:
        """Which soil_* fields no longer match the material, if any."""
        mat = self.scene.soil.spawn.material
        bad = [
            f"{n}: cfg {v:g} vs material {getattr(mat, m):g}"
            for n, m, v in (
                ("soil_density", "density", self.soil_density),
                ("soil_friction", "friction", self.soil_friction),
                ("soil_cohesion", "yield_stress", self.soil_cohesion),
                ("soil_yield_pressure", "yield_pressure", self.soil_yield_pressure),
            )
            if getattr(mat, m) != v
        ]
        return "; ".join(bad) if bad else None

    def footprint_cells(self) -> int:
        """Scan cells inside the drum's footprint. Must match the mask the env
        builds from the same pattern."""
        half_x = 0.5 * (DIG_SCAN_NX - 1) * DIG_SCAN_CELL
        half_y = 0.5 * (DIG_SCAN_NY - 1) * DIG_SCAN_CELL
        nx = sum(1 for i in range(DIG_SCAN_NX)
                 if abs(i * DIG_SCAN_CELL - half_x) <= FOOTPRINT_HALF_X)
        ny = sum(1 for i in range(DIG_SCAN_NY)
                 if abs(i * DIG_SCAN_CELL - half_y) <= FOOTPRINT_HALF_Y)
        return nx * ny

    def __post_init__(self) -> None:
        lo, hi = self.cut_depth_range
        self.cut_volume_ref = (
            self.footprint_cells() * DIG_SCAN_CELL ** 2 * 0.5 * (lo + hi)
        )

        # --- resolve the bed and write it into the scene -------------------
        margin = 0.5 * self.voxel_size
        lower = (self.bed_x[0], self.bed_y[0], margin)
        upper = (self.bed_x[1], self.bed_y[1], margin + self.bed_depth)
        length = self.bed_x[1] - self.bed_x[0]
        width = self.bed_y[1] - self.bed_y[0]

        self.bed_top = margin + self.bed_depth
        self.bed_grid_lower = (self.bed_x[0], self.bed_y[0])
        self.bed_grid_nx = int(math.ceil(length / self.voxel_size))
        self.bed_grid_ny = int(math.ceil(width / self.voxel_size))
        self.bed_particles_per_env = particles_per_env(
            lower, upper, self.voxel_size, self.particles_per_cell
        )
        spawn = self.scene.soil.spawn
        self.apply_soil_material()

        spawn.lower = lower
        spawn.upper = upper
        spawn.voxel_size = self.voxel_size
        spawn.particles_per_cell = self.particles_per_cell

        slab = self.scene.mpm_ground
        slab.spawn.size = (length + 1.0, width + 1.0, 0.10)
        slab.spawn.collision_props.contact_margin = margin
        slab.init_state.pos = (0.5 * (self.bed_x[0] + self.bed_x[1]), 0.0, -0.05)

        # Wheels rest on z = 0 in the shared asset cfg. On the bed that is a
        # quarter metre of regolith; with a pile ahead of the machine instead,
        # it stays on the ground plane.
        z = SPAWN_Z + (self.bed_top if self.spawn_on_bed else 0.0)
        self.scene.excavator.init_state.pos = (0.0, 0.0, z)

        total_particles = self.bed_particles_per_env * self.max_num_envs
        active = _next_pow2(int(self.grid_cap_multiplier * total_particles))
        # A leaf is a BLOCK of 8^3 = 512 cells, not a cell, so a leaf count
        # near the active cell count costs hundreds of times the storage the
        # active set calls for. >> 7 keeps four times the minimum of one leaf
        # per 512 cells; the floors stop a small bed from starving the tree.
        leaf = max(active >> self.grid_leaf_shift, 1024)
        lower = max(leaf >> 3, 256)
        upper = max(lower >> 3, 64)
        print(
            f"[excavate] bed {length:.1f} x {width:.1f} x {self.bed_depth:.2f} m at "
            f"{self.voxel_size:.3f} m voxel -> {self.bed_particles_per_env} particles/env "
            f"x {self.max_num_envs} max envs = {total_particles} particles"
        )
        print(
            f"[excavate] sparse grid active={active} leaf={leaf} "
            f"lower={lower} upper={upper}"
        )
        print(
            f"[excavate] rigid solver njmax={self.rigid_njmax} "
            f"nconmax={self.rigid_nconmax}"
        )
        print(
            f"[excavate]   {self.grid_cap_multiplier:.0f} cells/particle, "
            f"{1 << self.grid_leaf_shift} cells/leaf, "
            f"~{leaf * 512 * 48 / 1e6:.0f} MB of grid"
        )

        self.sim.physics = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name=RIGID_ENTRY,
                        solver_cfg=MJWarpSolverCfg(
                            use_mujoco_contacts=False,
                            integrator="implicitfast",
                            njmax=self.rigid_njmax,
                            nconmax=self.rigid_nconmax,
                        ),
                        bodies=[EXCAVATOR_PRIM_REGEX],
                        include_static_shapes=True,
                        substeps=2,
                    ),
                    CouplerEntryCfg(
                        name=MPM_ENTRY,
                        solver_cfg=MPMSolverCfg(
                            # self.voxel_size, not the module constant: this
                            # grid sets both the particle spacing and the
                            # margin the coupler inflates colliders by, so
                            # pinning it makes a finer cfg voxel a no-op.
                            voxel_size=self.voxel_size,
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
                            max_leaf_node_count=leaf,
                            max_lower_node_count=lower,
                            max_upper_node_count=upper,
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
                        bodies=[self.soil_contact_regex],
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


@configclass
class ExcavatorExcavateSmallEnvCfg(ExcavatorExcavateEnvCfg):
    """A bed that fits a 6 GB laptop: 1,152 particles and 16,384 active cells,
    against the full bed's 32,000 particles per env and grid caps in the
    millions.

    The soil is a pile in FRONT of the machine rather than a bed under it. The
    machine sits on the rigid ground plane and only the front drum reaches
    soil, so this does not exercise the counter-rotating pair; it exercises the
    particle adapter, the coupler mapping and the drum-fill sensor.

    The pile top is 0.125 m up, so the arm angle that bites differs from the
    deep bed's. dig_demo.py solves for it from a cut depth.
    """

    # Depth is what matters: at 0.10 m only 0.04 m of the rotor sits inside the
    # soil column; at 0.16 m it is 0.09 m.
    #
    # The near edge is set from where the drum actually is. At the dig angle
    # the drum axis sits at PIVOT_X + ARM_LEN*cos(arm) = 1.65 m and its leading
    # edge at 1.86, so a bed starting at 1.1 puts the machine most of the way
    # across it before it has begun. Starting at 1.80 means the drum enters at
    # the near edge and has the whole bed ahead of it.
    bed_x: tuple[float, float] = (1.80, 2.70)
    bed_y: tuple[float, float] = (-0.55, 0.55)
    bed_depth: float = 0.16
    spawn_on_bed: bool = False

    # Cohesive enough that a cut clod survives the trip into the bore. 0
    # sprays; ~800 holds together; too much and the drum cannot cut in.
    soil_cohesion: float = 800.0

    # This bed is a 1.1 m wide strip and the machine is run against it by a
    # scripted demo. The training jitter would put the drum beside the pile,
    # and a carried-over bed would leave the demo nothing to cut.
    spawn_x_jitter: float = 0.15
    spawn_y_jitter: float = 0.0
    spawn_yaw_jitter: float = 0.05
    arm_start_jitter: float = 0.0
    soil_reset_every: int = 1

    max_num_envs = 1
    episode_length_s = 30.0

    def __post_init__(self) -> None:
        self.scene.num_envs = min(self.scene.num_envs, self.max_num_envs)
        self.scene.env_spacing = 8.0
        super().__post_init__()


@configclass
class ExcavatorExcavateMicroEnvCfg(ExcavatorExcavateSmallEnvCfg):
    """A finer voxel on a small bed, with only the front drum coupled.

    The drum's entry channel opens 0.082 m. At a 0.05 voxel that leaves 0.032 m
    clear, 0.6 of a particle spacing, and soil bridges the opening; at 0.03 it
    is 1.7 spacings and flows.
    """

    voxel_size: float = 0.03

    # A strip worth cutting, starting where the drum's leading edge reaches at
    # the dig angle so the machine has the whole metre ahead of it.
    bed_x: tuple[float, float] = (1.80, 2.80)
    bed_y: tuple[float, float] = (-0.50, 0.50)
    bed_depth: float = 0.21
    spawn_on_bed: bool = False

    max_num_envs = 1
    episode_length_s = 30.0
