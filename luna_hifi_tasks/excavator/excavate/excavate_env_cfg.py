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

    # Sparse-grid active cells per particle. The caps are ABSOLUTE totals over
    # all envs, and they are the thing that decides whether this fits in VRAM:
    # too high and the allocation fails as a CUDA 700 illegal-access storm
    # rather than a clean out-of-memory error. The tricycle's validated config
    # ran 2,880 particles against 16,384 active cells -- a ratio of 5.7 -- so 8
    # is slightly generous. Lower it before lowering anything else if the card
    # will not take it.
    grid_cap_multiplier: float = 8.0

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
        print(
            f"[excavate] bed {length:.1f} x {width:.1f} x {self.bed_depth:.2f} m at "
            f"{self.voxel_size:.3f} m voxel -> {self.bed_particles_per_env} particles/env "
            f"x {self.max_num_envs} max envs = {total_particles} particles"
        )
        print(
            f"[excavate] sparse grid active={active} leaf={active >> 1} "
            f"lower={active >> 2} upper={active >> 4}   "
            f"(tricycle ran 2880 particles / 16384 active on a 6 GB card)"
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
    """Small enough to run at a 0.03 m voxel, which is the point.

    The Small preset cannot fill the drum and no drum geometry can fix that.
    The coupler inflates every collider by half a voxel per side, so it eats a
    whole voxel out of every passage, and particles are spawned one voxel
    apart. The drum's entry channel opens 0.082 m:

        voxel 0.05 -> 0.032 m clear = 0.6 particle spacings -> soil bridges
                      the opening and stops IN the lip, scooped but never in
        voxel 0.03 -> 0.052 m clear = 1.7 particle spacings -> it goes in

    That is the whole difference, and it is a resolution limit rather than a
    shape problem. Watch for it in the readouts: soil visibly carried around on
    the lips while drum_fill_mass stays near zero is this, because fill is
    counted inside r = 0.17 m and the lip channel sits outside it.

    The bed pays for it. A finer voxel costs particles as the CUBE, so this one
    is a pad just under the front drum rather than a strip the machine drives
    along: 0.35 x 0.80 x 0.24 m gives 2,592 particles against the tricycle's
    validated 2,880, and grid_cap_multiplier drops to 6 to keep the sparse grid
    at the 16,384 cells that card is known to take. Both numbers are inside the
    envelope that already ran here; the 9,504/65,536 attempt is what failed as
    a CUDA 700 storm.

    Consequences of being a pad: only the front drum ever sees soil, the
    machine cannot drive far while cutting, and fill tops out well below one
    drum. None of that matters for the question this preset exists to answer,
    which is whether soil enters the drum at all.
    """

    voxel_size: float = 0.03

    # Under the front drum at dig angle, not under the machine.
    bed_x: tuple[float, float] = (1.30, 1.65)
    bed_y: tuple[float, float] = (-0.40, 0.40)
    # Deep enough that the lips are not grounding out: they stand 0.108 m proud
    # of the shell now, so a 0.18 m bed would cap the usable cut at 0.073 m.
    bed_depth: float = 0.24
    spawn_on_bed: bool = False

    # 6, not 8. 2,592 particles x 6 rounds to the 16,384 cells that are known
    # to fit; x8 would round to 32,768 and this card has already failed there.
    grid_cap_multiplier: float = 6.0

    max_num_envs = 1
    episode_length_s = 30.0
