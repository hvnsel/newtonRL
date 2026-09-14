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
    FRONT_DRUM_ONLY_REGEX,
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

# The bed is DERIVED, not constant. Its extent lives on the env cfg as
# bed_x / bed_y / bed_depth, and __post_init__ rewrites the scene from those.
# Everything downstream -- the particle count, the sparse-grid capacities, the
# height grid the scans rasterise onto, the hidden floor slab, the spawn height
# -- follows from them, which is what makes a smaller bed a subclass rather
# than a fork of this file.
#
# Empty height-grid cells read BED_FLOOR_Z, the top of the slab an excavated
# cell bottoms out on, so dug ground and untouched ground never look alike.
BED_FLOOR_Z = 0.0

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
# ~155 kg. An UPPER bound twice over: the two lifters displace about 7% of the
# bore they sweep, and no granular fill packs to 100% of a free volume anyway.
# It is a normaliser, not a prediction -- fill fraction is only ever compared
# against itself and against fill_success_fraction, so what matters is that it
# stays put when the soil density does not.
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

    # --- bed, in the env frame ---
    #
    # The machine spawns at the origin facing +x. With spawn_on_bed the wheels
    # rest on the bed surface and both drums start over soil: the front one
    # meets fresh regolith as it drives, the rear trails through the trench.
    bed_x: tuple[float, float] = (-3.0, 5.0)
    bed_y: tuple[float, float] = (-1.0, 1.0)
    bed_depth: float = 0.25
    # THE parameter for whether the drum can fill. The coupler inflates every
    # collider by half a voxel per side, so it eats a whole voxel out of every
    # passage, and particles are spawned one voxel apart. The drum's entry
    # channel opens 0.082 m, so at 0.05 there is 0.032 m clear -- less than one
    # particle, and soil bridges the opening and stops in the lip instead of
    # going in. At 0.03 the same channel is 1.7 particles wide and flows.
    #
    # It is not free: halving it multiplies the particle count by eight for a
    # given bed. That is why the Micro preset shrinks the bed at the same time.
    voxel_size: float = VOXEL_SIZE
    particles_per_cell: float = MPM_PARTICLES_PER_CELL
    spawn_on_bed: bool = True

    # Sparse-grid active cells per particle. ABSOLUTE totals over all envs.
    #
    # Read the failure mode correctly, because we had it backwards for most of
    # this machine's life: overrunning these caps is an ILLEGAL ACCESS -- a CUDA
    # 700 storm, or a 0xC0000374 heap corruption on Windows -- and it looks
    # nothing like running out of memory, but it was being treated as though it
    # were. Every crash was answered by shrinking the bed, which made the
    # shortfall worse relative to the collider set and guaranteed the next one.
    #
    # The grid is cheap. A cell is tens of bytes, so 2^20 of them is well under
    # 100 MB on a card with six thousand. Being stingy here buys nothing and
    # costs particles, which are the one thing actually worth spending on.
    #
    # 8, the tricycle's proven ratio. Raising it to 24 on the theory that the
    # caps were the binding constraint changed nothing: the 0.03 m voxel died
    # exactly the same way with 32x the grid. So the caps are NOT what this
    # machine is hitting, and scripts/mpm_probe.py exists to find what is
    # instead of the next plausible-sounding guess.
    grid_cap_multiplier: float = 8.0

    # Which bodies are coupled to the soil. Narrow it on a preset whose soil
    # some of them cannot reach: each body brings every one of its geoms, and
    # the drums are 38 geoms apiece.
    soil_contact_regex: str = SOIL_CONTACT_BODIES_REGEX

    # Rigid-solver buffers, per env. FIXED SIZE, and overrunning one is an
    # illegal memory access -- a CUDA 700 storm, or 0xC0000374 on Windows --
    # not a clean error, which is what makes it so hard to place.
    #
    # These were 200/96, inherited from the tricycle and then LOWERED, on a
    # machine with fifteen times the geometry. The tricycle has 8 geoms and
    # gave itself 128 contact slots; this excavator has 119, of which 104 are
    # coupled to the soil, and had 96. Contacts scale with colliders AND with
    # how much soil is touching them, which is why the crash tracked the
    # particle count and looked for all the world like a memory limit.
    #
    # This is also the honest answer to "why can a standalone Warp MPM sample
    # run tens of thousands of particles when we cannot": a standalone sample
    # has no rigid solver, so it has no njmax and no nconmax. These buffers
    # only exist because a rigid solver is coupled in, and they are what we
    # were overflowing.
    #
    # A contact is a couple of hundred bytes. 8192 of them is about 1.6 MB, so
    # there is no reason to be tight here.
    rigid_njmax: int = 4096
    rigid_nconmax: int = 8192

    # Active cells per LEAF NODE, as a right shift: 7 is one leaf per 128
    # cells, four times the theoretical minimum for 8^3 blocks. Lower it (more
    # leaves) only if the solver runs out of tree, and expect every step to
    # cost 512 cells of storage.
    grid_leaf_shift: int = 7



    # --- regolith ---
    #
    # The yield surface in this MPM formulation is
    #     tau_max(p) = yield_stress + friction * (p - p_min)
    # so friction is ~tan(phi) and soil_cohesion IS the cohesion, in Pa. Old
    # Drucker-Prager alpha numbers do not port across.
    #
    # Cohesion is the knob that decides whether the drum CAPTURES soil or just
    # sprays it. At zero the regolith is dry sand, and sand does not hold the
    # shape of a cut: it shears off the lip and flows back out of the mouth it
    # came in through. A few hundred Pa and the cut travels as a clod that the
    # lifters can carry round. Lunar simulants sit around 0.1-1 kPa.
    #
    # The blades sweep r = 0.053 to 0.246 m and fill is counted inside the
    # r = 0.170 m bore, so the lifters reach 0.117 m into the volume being
    # measured. That is deliberate -- it is what carries the load up the
    # ascending side instead of letting it sit at the bottom waiting for a
    # mouth -- but it also means a NON-cohesive soil gets churned by a lifter
    # every half turn. Cohesion and lifter depth are the same knob seen from
    # two ends; if fill oscillates instead of climbing, this is why.
    soil_density: float = 1800.0
    soil_friction: float = 0.84              # tan(40 deg)
    soil_cohesion: float = 0.0               # yield_stress, Pa
    soil_yield_pressure: float = 1.0e12      # packed ground, not a sandbox

    # Derived in __post_init__ from the four fields above. Declared here so the
    # env can read them off the cfg instead of importing module constants,
    # which is what makes a differently-sized bed a subclass.
    bed_top: float = 0.0
    bed_grid_lower: tuple[float, float] = (0.0, 0.0)
    bed_grid_nx: int = 0
    bed_grid_ny: int = 0
    bed_particles_per_env: int = 0
    drum_capacity_kg: float = DRUM_CAPACITY_KG

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

    # The soil_* fields are plain numbers on this cfg; the solver reads an
    # MPMParticleMaterialCfg hanging off the spawner. Copying between them is
    # its own method because __post_init__ is NOT the last word on those
    # fields: Hydra applies command-line overrides AFTER __post_init__ has run,
    # so `env.soil_cohesion=1500` changes the field and nothing downstream of
    # it. Anything that touches a soil_* field late must call this, and the env
    # refuses to start if the two disagree.
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
            # A plain setattr on a dataclass instance silently creates an
            # attribute the solver never reads, so an API rename here would
            # look exactly like soil parameters having no effect. Check first.
            if not hasattr(mat, name):
                raise AttributeError(
                    f"MPMParticleMaterialCfg has no field {name!r} on this Isaac Lab "
                    f"build; it has {sorted(vars(mat))}. Setting it anyway would be a "
                    "silent no-op -- the soil would keep its default and no amount of "
                    "tuning soil_cohesion would change anything."
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

    def __post_init__(self) -> None:
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
        # quarter metre of regolith, so lift the machine clear of it; with a
        # pile in front of the machine instead, leave it on the ground plane.
        z = SPAWN_Z + (self.bed_top if self.spawn_on_bed else 0.0)
        self.scene.excavator.init_state.pos = (0.0, 0.0, z)

        total_particles = self.bed_particles_per_env * self.max_num_envs
        active = _next_pow2(int(self.grid_cap_multiplier * total_particles))
        # A leaf is a BLOCK of cells, not a cell. That one fact is why this
        # machine appeared to be capped at a few thousand particles:
        #
        #   max_leaf_node_count was active >> 1, so for every two active cells
        #   we asked for one whole leaf BLOCK. At the sparse-grid block size of
        #   8^3 = 512 cells, that is 256 times more storage than the active
        #   cell count calls for -- 200 MB of grid for 1,584 particles, and
        #   800 MB by 5,544, which is where a 6 GB card running Kit gives out.
        #
        # A ladder of single-knob trials (scripts/mpm_probe.py) put it beyond
        # doubt: every configuration with active <= 32,768 ran and every one
        # with active >= 65,536 died, at three different voxel sizes and with
        # the collider set varied. Neither the voxel nor the colliders moved
        # the boundary at all. It was always this.
        #
        # >> 7 keeps four times the theoretical minimum of one leaf per 512
        # cells, and the floors stop a small bed from starving the tree.
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
            f"nconmax={self.rigid_nconmax} (fixed-size; overrunning either is an "
            f"illegal access, not an error)"
        )
        print(
            f"[excavate]   {self.grid_cap_multiplier:.0f} cells/particle; a leaf covers "
            f"{1 << self.grid_leaf_shift} of them and is the thing that actually costs "
            f"memory -- roughly {leaf * 512 * 48 / 1e6:.0f} MB of grid here."
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
                            # self.voxel_size, NOT the module constant. This is
                            # the SOLVER's grid, and it is what sets both the
                            # particle spacing and the margin the coupler
                            # inflates every collider by. Pinned to VOXEL_SIZE
                            # it silently ignored a finer cfg voxel: particles
                            # spawned closer together, the grid stayed coarse,
                            # and nothing about what fits through the drum
                            # changed.
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
    """A bed that fits a 6 GB laptop, sized against the tricycle's known-good
    footprint rather than against what looks small on paper.

    The full bed is 32,000 particles per env with grid caps in the millions --
    a cluster config. An earlier attempt at "small" was 9,504 particles and
    65,536 cells, which still filled a 6 GB card and failed as a CUDA 700
    storm. This one is 1,152 particles and 16,384 active cells, which is
    exactly the capacity the tricycle validated on that hardware.

    Getting there means giving up something, and the honest trade is that the
    soil is a PILE IN FRONT of the machine rather than a bed under it. The
    machine sits on the rigid ground plane and only the FRONT drum reaches
    soil, so this does not exercise the counter-rotating pair. That is fine for
    what it is for: proving the particle adapter, the coupler mapping and the
    drum-fill sensor are wired correctly. Both drums digging is a cluster
    concern, and the full config covers it.

    Because the machine is on the ground and the pile top is only 0.125 m up,
    the arm angle that bites is quite different from the deep bed's. Let
    dig_demo.py solve for it from a cut depth instead of hardcoding a command.
    """

    # A tight mound ahead of the machine rather than a thin sheet. Depth is
    # what matters: at 0.10 m only 0.04 m of the drum bore ever sits inside the
    # soil column, which is a scratch, not a cut. At 0.16 m it is 0.09 m, and
    # the footprint shrinks to keep the particle count flat.
    bed_x: tuple[float, float] = (1.1, 2.0)
    bed_y: tuple[float, float] = (-0.55, 0.55)
    bed_depth: float = 0.16
    spawn_on_bed: bool = False

    # Cohesive enough that a cut clod survives the trip into the bore. This is
    # the first thing to sweep if fill stays near zero while particles visibly
    # move: 0 sprays, ~800 holds together, too much and the drum cannot cut in
    # at all.
    soil_cohesion: float = 800.0

    max_num_envs = 1
    episode_length_s = 30.0

    def __post_init__(self) -> None:
        self.scene.num_envs = min(self.scene.num_envs, self.max_num_envs)
        self.scene.env_spacing = 8.0
        super().__post_init__()


@configclass
class ExcavatorExcavateMicroEnvCfg(ExcavatorExcavateSmallEnvCfg):
    """A finer voxel on a small bed, with only the front drum coupled.

    Why a finer voxel at all: the coupler inflates every collider by half a
    voxel per side, so it eats a whole voxel out of every passage, and
    particles are spawned one voxel apart. The drum's entry channel opens
    0.082 m, so at 0.05 there is 0.032 m clear -- 0.6 of a particle spacing,
    and soil bridges the opening and stops in the lip instead of going in. At
    0.03 it is 1.7 spacings.

    Why the bed is no longer tiny: it never needed to be. A ladder of
    single-knob trials showed survival tracking the sparse-grid cap and nothing
    else -- not the voxel, not the collider count -- and the cap was being
    turned into 256 times its own weight in memory by a leaf-node ratio that
    counted blocks as though they were cells. See the note by grid_leaf_shift.
    Three rounds of shrinking beds were chasing the wrong quantity.
    """

    voxel_size: float = 0.03

    # A strip worth cutting. Kept modest rather than minimal: the contact
    # buffers were the thing being overrun, not the particle count, but that
    # was established by reading the config rather than by a run, so this is
    # the first size to try and not the last.
    bed_x: tuple[float, float] = (1.05, 1.95)
    bed_y: tuple[float, float] = (-0.50, 0.50)
    bed_depth: float = 0.21
    spawn_on_bed: bool = False

    max_num_envs = 1
    episode_length_s = 30.0
