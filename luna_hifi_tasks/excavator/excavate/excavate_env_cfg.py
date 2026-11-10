# excavate_env_cfg.py
#
# Excavation of an implicit-MPM regolith bed under lunar gravity. The rigid
# machine and the soil are coupled through proxy bodies.

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

from ...ground import ground_plane_spawn
from ..excavator import (
    ARM_LEN,
    ARM_RANGE,
    CHASSIS_Z,
    DIG_DRUM_SIGN,
    MAST_TOP_Z,
    ROTOR_HALF_LEN,
    SHROUD_END_T,
    SHROUD_IN_R,
    SHROUD_INLET,
    SHROUD_OUT_R,
)
from ..excavator_cfg import (
    DRUM_CONTACT_BODIES_REGEX,
    EXCAVATOR_CFG,
    EXCAVATOR_PRIM_REGEX,
    LUNAR_GRAVITY,
    MAX_DRUM_SPEED,
    SOIL_CONTACT_BODIES_REGEX,
    SPAWN_Z,
    WHEEL_CONTACT_BODIES_REGEX,
)
from ..mdp.observations import (
    DIG_CRITIC_SCAN_CELLS,
    critic_state_spec,
    excavate_obs_spec,
)
from ..mdp.sensors import mpm_grid_resolution

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIGID_ENTRY = "rover"   # coupler entry name of the rigid solver
MPM_ENTRY = "soil"      # coupler entry name of the MPM solver

# Bed height-grid cell [m], which the terrain scans and the cut depth are
# quantised to. Half of it is the collider contact margin.
VOXEL_SIZE = 0.04
MPM_COLLIDER_MARGIN = 0.5 * VOXEL_SIZE

# MPM solver background grid cell [m].
MPM_VOXEL_SIZE = 0.08
# Spawn-lattice particles per solver cell, per axis (8 per cell in 3D).
MPM_PARTICLES_PER_CELL = 2.0
MPM_VISUAL_COLOR = (0.62, 0.55, 0.45)

# Env-local z of the hard floor under the bed.
BED_FLOOR_Z = 0.0

# friction is ~tan(phi), yield_pressure caps compression [Pa]. Overwritten
# from the soil_* fields in __post_init__.
SOIL_MATERIAL = MPMParticleMaterialCfg(
    density=1800.0,
    friction=0.84,
    yield_pressure=1.0e12,
)

# The drum's interior, which fill is counted inside: within the shroud wall,
# between the end plates, on the closed side of the chord across the inlet
# (shroud frame, inlet centred on -z).
BORE_RADIUS = SHROUD_IN_R
BORE_HALF_LEN = ROTOR_HALF_LEN
BORE_CHORD_Z = -SHROUD_IN_R * math.cos(0.5 * SHROUD_INLET)
BORE_AREA = math.pi * BORE_RADIUS ** 2 - 0.5 * BORE_RADIUS ** 2 * (SHROUD_INLET - math.sin(SHROUD_INLET))
BORE_VOLUME = BORE_AREA * (2.0 * BORE_HALF_LEN)

# Scan cells inside this rectangle, in the lead drum's frame, are the ground
# under the drum.
FOOTPRINT_HALF_X = SHROUD_OUT_R
FOOTPRINT_HALF_Y = ROTOR_HALF_LEN

# Mass of the interior packed solid at SOIL_MATERIAL's density [kg].
DRUM_CAPACITY_KG = BORE_VOLUME * SOIL_MATERIAL.density

DIG_OBS = excavate_obs_spec()
DIG_CRITIC = critic_state_spec(terrain_cells=DIG_CRITIC_SCAN_CELLS)


def _next_pow2(n: int) -> int:
    return 1 << max(int(n - 1).bit_length(), 1)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

SOIL_CFG = MPMObjectCfg(
    prim_path="{ENV_REGEX_NS}/Soil",
    init_state=MPMObjectCfg.InitialStateCfg(),
    spawn=MPMGridCfg(
        # Placeholder bounds; __post_init__ sets them from the bed fields.
        lower=(-1.0, -1.0, MPM_COLLIDER_MARGIN),
        upper=(1.0, 1.0, MPM_COLLIDER_MARGIN + 0.25),
        voxel_size=MPM_VOXEL_SIZE,
        particles_per_cell=MPM_PARTICLES_PER_CELL,
        particle_placement="cell_center",
        jitter=0.0,
        material=SOIL_MATERIAL,
        visual_color=MPM_VISUAL_COLOR,
    ),
)


def _hidden_box(
    name: str,
    friction: tuple[float, float],
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RigidObjectCfg:
    """Invisible kinematic box with a collider margin of MPM_COLLIDER_MARGIN.
    `friction` is (static, dynamic)."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=MPM_COLLIDER_MARGIN,
                contact_gap=0.0,
            ),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=friction[0], dynamic_friction=friction[1]
            ),
            visible=False,
        ),
    )


@configclass
class ExcavateSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=ground_plane_spawn(),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )

    excavator: ArticulationCfg = EXCAVATOR_CFG
    # Floor slab under the particles, seen only by the MPM entry.
    mpm_ground: RigidObjectCfg = _hidden_box(
        "MPMGround", (0.8, 0.7), size=(4.0, 3.0, 0.10), pos=(0.0, 0.0, -0.05)
    )
    # Plates retaining the four bed faces, seen only by the MPM entry.
    bed_wall_xlo: RigidObjectCfg = _hidden_box("BedWallXlo", (0.5, 0.5))
    bed_wall_xhi: RigidObjectCfg = _hidden_box("BedWallXhi", (0.5, 0.5))
    bed_wall_ylo: RigidObjectCfg = _hidden_box("BedWallYlo", (0.5, 0.5))
    bed_wall_yhi: RigidObjectCfg = _hidden_box("BedWallYhi", (0.5, 0.5))
    # Rails under the wheel tracks, tops flush with the soil, seen only by the
    # rigid entry.
    wheel_rail_ylo: RigidObjectCfg = _hidden_box("WheelRailYlo", (1.0, 0.9))
    wheel_rail_yhi: RigidObjectCfg = _hidden_box("WheelRailYhi", (1.0, 0.9))
    soil: MPMObjectCfg = SOIL_CFG


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@configclass
class ExcavatorExcavateEnvCfg(DirectRLEnvCfg):

    decimation = 4
    episode_length_s = 16.0
    # Unpaid time at the start of each episode while the bed settles; the cut
    # plane and the reference surface are re-captured throughout it.
    settle_seconds: float = 1.0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=decimation,
        gravity=LUNAR_GRAVITY,
    )

    scene: ExcavateSceneCfg = ExcavateSceneCfg(
        num_envs=16,
        # A multiple of 0.16 m, so every env origin sits on a solver voxel corner.
        env_spacing=5.12,
        replicate_physics=True,
    )

    # --- bed, env frame [m] ---
    # Edges are multiples of the 0.08 m solver voxel.
    bed_x: tuple[float, float] = (-1.92, 2.64)
    bed_y: tuple[float, float] = (-0.80, 0.80)
    bed_depth: float = 0.25
    bed_wall_thickness: float = 0.10
    # False parks the retaining plates below the floor.
    bed_walls: bool = True

    # Inset from each bed face that the chassis origin must stay inside [m].
    bed_margin_x: float = 1.81
    bed_margin_y: float = 0.60
    voxel_size: float = VOXEL_SIZE
    mpm_voxel_size: float = MPM_VOXEL_SIZE
    particles_per_cell: float = MPM_PARTICLES_PER_CELL

    # Gauss-Seidel sweep cap per MPM solve. The solve runs in blocks of 25
    # sweeps, so keep it a multiple of 25.
    mpm_max_iterations: int = 50
    # Absolute tolerance on the dimensionless per-step strain residual.
    mpm_tolerance: float = 2.0e-6
    # True: the machine stands on the bed. False: the bed is a pile in front of it.
    spawn_on_bed: bool = True

    # How far past the bed, on every side, the floor slab and the height grid reach [m].
    mpm_floor_margin: float = 1.0
    heightmap_margin: float = 1.0

    # Sparse-grid active cells allocated per particle, over all envs.
    grid_cap_multiplier: float = 1.0

    # Regex of the bodies coupled to the soil. None, or either of the two
    # canonical regexes, follows wheel_rails: drums and shrouds with rails,
    # wheels as well without.
    soil_contact_regex: str | None = None

    # True: wheels ride rigid rails at grade and are not coupled to the soil.
    # False: rails parked below the floor, wheels on soil.
    wheel_rails: bool = False

    # Gap from the cutting end's axial extent to a rail's inner face [m].
    wheel_rail_clearance: float = 0.06

    # Rigid-solver constraint and contact buffers, per env.
    rigid_njmax: int = 8192
    rigid_nconmax: int = 32768

    use_cuda_graph: bool = True

    # Sparse-grid leaf nodes = active cells >> grid_leaf_shift.
    grid_leaf_shift: int = 3
    # Floor on upper internal nodes.
    grid_upper_min: int = 4096

    # Warm start of the MPM implicit solve, carried on the particles.
    mpm_warmstart_mode: str = "particles"

    # True: shape_done ends the episode. Failures always do.
    terminate_early: bool = False

    # MPM grid partitioned per world, which masked per-env resets require.
    mpm_separate_worlds: bool = True

    # --- regolith ---
    # Yield surface: sqrt(J2) <= soil_cohesion + soil_friction * p.
    soil_density: float = 1500.0             # kg/m3
    soil_friction: float = 0.84              # tan(40 deg)
    soil_cohesion: float = 0.0               # yield_stress, Pa
    soil_yield_pressure: float = 1.0e12      # Pa, compression cap
    soil_young_modulus: float = 1.0e7        # Pa
    # Solid fraction below which the solver drops cohesion; 0 disables it.
    soil_critical_fraction: float = 0.0

    # Derived in __post_init__.
    bed_top: float = 0.0
    bed_grid_lower: tuple[float, float] = (0.0, 0.0)
    bed_grid_nx: int = 0
    bed_grid_ny: int = 0
    bed_particles_per_env: int = 0
    drum_capacity_kg: float = DRUM_CAPACITY_KG
    # Work area times the mean cut depth [m3].
    cut_volume_ref: float = 0.0304

    # Env count the sparse grid is sized for. The env sets it to scene.num_envs,
    # which may not exceed it.
    max_num_envs = 4

    # actions: [forward, yaw] in [-1, 1]
    action_space = 2
    observation_space = DIG_OBS.dim      # 166
    state_space = DIG_CRITIC.dim         # 231

    # --- start state ---
    arm_start_angle = -0.20              # rad, drums ~0.44 m up
    # No start jitter: navigation is assumed to deliver the exact pose.
    arm_start_jitter = 0.0               # rad
    spawn_yaw_jitter = 0.0               # rad
    spawn_x_jitter = 0.0                 # m
    spawn_y_jitter = 0.0                 # m

    # Episodes an env keeps its worked bed before the soil is re-laid.
    soil_reset_every: int = 1

    # --- tool, run by the env ---
    # After the settle, over dig_ramp_seconds with the wheels held: the arms
    # lower until the shroud bottom is dig_cut below the bed surface
    # (boom_command_for_cut), the shrouds turn to minus the arm angle plus
    # dig_inlet_tilt, and the drums spin up to dig_drum_speed. False holds the
    # arms at their start angle and the drums still.
    dig_tool: bool = True
    dig_cut: float = 0.10                    # m
    dig_inlet_tilt: float = 0.096            # rad off straight down
    dig_drum_speed: float = DIG_DRUM_SIGN * MAX_DRUM_SPEED   # rad/s, the loading direction
    dig_ramp_seconds: float = 2.0

    # --- actions, once the ramp is done ---
    # Forward: dig_forward * (1 + a0) of top speed, 0 to 0.45 m/s.
    dig_forward: float = 0.15
    # Yaw rate [rad/s]: heading_hold_gain * (handover heading - heading),
    # clamped to dig_yaw_rate, plus dig_yaw_rate * a1.
    heading_hold_gain: float = 1.0
    dig_yaw_rate: float = 0.3
    action_smoothing = 0.3
    # The leading drum, which the actor's scan is taken at, switches only when
    # the commanded drive passes this speed [m/s].
    dig_lead_deadband: float = 0.05

    arm_range = ARM_RANGE

    # --- cut command ---
    # Per episode: a plane `depth` below the surface the front drum meets,
    # with an optional gradient.
    cut_depth_range: tuple[float, float] = (0.04, 0.04)     # m
    cut_gradient_max: float = 0.0                          # rise over run
    # Fraction of episodes with a non-zero gradient.
    cut_ramp_fraction: float = 0.0
    # Lateral gradient as a fraction of cut_gradient_max.
    cut_lateral_fraction: float = 0.3

    # Work area: a rectangle fixed in the world for the episode, anchored where
    # the front drum meets soil and aligned with the spawn heading. The episode
    # ends when the front drum is work_area_length past the anchor.
    work_area_length: float = 0.60           # m ahead of the anchor
    work_area_behind: float = 0.20           # m behind it
    work_area_width: float = 0.95            # m, the drums' swath

    # Residual volume, as a fraction of cut_volume_ref, at which the shape is achieved.
    shape_success_fraction: float = 0.30

    # --- load sensor ---
    # The actor's load reading: a first-order lag, noise proportional to the
    # reading, and a per-episode calibration bias. The critic reads the exact mass.
    # target_load_kg = target_load_bore_fraction * drum_capacity_kg, set in
    # __post_init__; 40 kg at 1500 kg/m3.
    target_load_bore_fraction: float = 40.0 / (BORE_VOLUME * 1500.0)
    target_load_kg: float = 40.0
    fill_sensor_tau: float = 0.3             # s
    spill_filter_tau: float = 2.0            # s, the spill term's own lag
    fill_sensor_noise: float = 0.05          # fraction of the reading
    fill_sensor_abs_kg: float = 0.5
    fill_sensor_bias: float = 0.03           # +- fraction, per episode

    # --- success ---
    fill_success_fraction = 0.6              # of target_load_kg
    # "front": the front drum. "all": every drum. "mean": the pair's average.
    fill_success_mode = "all"

    # --- reward weights ---
    # removal and unheld are per 2 x target_load_kg; overcut per cut_volume_ref.
    w_removal = 100.0                    # mass cut from the work area
    w_unheld = 120.0                     # |removal - payload|
    # Payload as a fraction of cut_total required for the shape to count.
    capture_min_fraction: float = 0.50
    # Distance from the env origin at which a particle counts as escaped [m].
    particle_escape_m: float = 15.0
    w_overcut = 200.0                    # volume taken below the plane
    w_spill = 20.0                       # loss from the slow fill reading
    w_success = 40.0                     # once, when the shape is first achieved
    w_fail = 20.0                        # on the step _failed() fires
    w_stall = 0.025                      # commanded speed not achieved
    w_drift = 0.01                       # lateral motion
    w_upright = 2.0
    w_energy = 1.0e-5
    w_action_rate = 0.0075
    w_time = 0.005

    # --- termination ---
    max_tilt_rad = math.radians(60.0)
    max_env_excursion = 6.0

    # --- scan ---
    # Actor only; the critic's scan is clean. scan_invalid marks dropped cells.
    scan_clip = 1.0
    scan_noise_std = 0.015               # m, at scan_range_ref
    scan_dropout = 0.02                  # probability, at scan_range_ref
    scan_range_ref = 1.0                 # m
    scan_invalid = -2.0

    # Multiplier on each wheel's free-body mass and inertia as the MPM solve sees it.
    wheel_proxy_mass_scale: float = 8.7
    # The same for the drums and shrouds.
    drum_proxy_mass_scale: float = 1.0

    # Relaxation factor on the rigid<->MPM force exchange, in [0, 1]:
    # force = previous + omega * (raw - previous).
    proxy_relaxation: float = 0.1

    # "fixed" holds proxy_relaxation; "aitken" adapts it from successive residuals.
    proxy_relaxation_mode: str = "fixed"

    # "lagged": MPM sees the previous pass's forces. "staggered": the two
    # solves alternate within the step.
    proxy_mode: str = "lagged"

    # Rigid<->MPM fixed-point passes per physics step.
    coupler_iterations: int = 4

    def apply_soil_material(self) -> None:
        """Copy the soil_* fields onto the MPM material and derive the drum capacity
        and target load."""
        mat = self.scene.soil.spawn.material
        mat.density = self.soil_density
        mat.friction = self.soil_friction
        mat.yield_stress = self.soil_cohesion
        mat.yield_pressure = self.soil_yield_pressure
        mat.young_modulus = self.soil_young_modulus
        self.drum_capacity_kg = BORE_VOLUME * self.soil_density
        self.target_load_kg = self.target_load_bore_fraction * self.drum_capacity_kg

    def boom_command_for_cut(self, cut: float) -> tuple[float, float]:
        """(boom command in [-1, 1], arm angle in rad) that puts the bottom of
        the shroud `cut` metres below the bed surface.

        World z, with the wheels on bed_top (spawn_on_bed) or on z = 0:

            shroud bottom = wheel plane + CHASSIS_Z + MAST_TOP_Z
                            - ARM_LEN * sin(angle) - SHROUD_OUT_R
            target        = bed_top - cut

        The angle is clamped to arm_range.
        """
        pivot = CHASSIS_Z + MAST_TOP_Z
        wheel_plane = self.bed_top if self.spawn_on_bed else 0.0
        sin_t = (wheel_plane + pivot - SHROUD_OUT_R - (self.bed_top - cut)) / ARM_LEN
        lo, hi = self.arm_range
        angle = min(max(math.asin(min(max(sin_t, -1.0), 1.0)), lo), hi)
        return 2.0 * (angle - lo) / (hi - lo) - 1.0, angle

    def __post_init__(self) -> None:
        # Work area taken down by the mean cut depth [m3].
        lo, hi = self.cut_depth_range
        self.cut_volume_ref = (
            (self.work_area_length + self.work_area_behind)
            * self.work_area_width
            * 0.5 * (lo + hi)
        )

        # --- bed ---
        margin = 0.5 * self.voxel_size
        lower = (self.bed_x[0], self.bed_y[0], margin)
        upper = (self.bed_x[1], self.bed_y[1], margin + self.bed_depth)
        length = self.bed_x[1] - self.bed_x[0]
        width = self.bed_y[1] - self.bed_y[0]

        self.bed_top = margin + self.bed_depth
        hm = self.heightmap_margin
        self.bed_grid_lower = (self.bed_x[0] - hm, self.bed_y[0] - hm)
        self.bed_grid_nx = int(math.ceil((length + 2.0 * hm) / self.voxel_size))
        self.bed_grid_ny = int(math.ceil((width + 2.0 * hm) / self.voxel_size))
        self.bed_particles_per_env = math.prod(
            mpm_grid_resolution(lower, upper, self.mpm_voxel_size, self.particles_per_cell)
        )
        spawn = self.scene.soil.spawn
        self.apply_soil_material()

        spawn.lower = lower
        spawn.upper = upper
        spawn.voxel_size = self.mpm_voxel_size
        spawn.particles_per_cell = self.particles_per_cell

        # Floor slab, centred under the bed.
        slab = self.scene.mpm_ground
        fm = self.mpm_floor_margin
        slab.spawn.size = (length + 2.0 * fm, width + 2.0 * fm, 0.10)
        slab.spawn.collision_props.contact_margin = margin
        slab.init_state.pos = (
            0.5 * (self.bed_x[0] + self.bed_x[1]),
            0.5 * (self.bed_y[0] + self.bed_y[1]),
            -0.05,
        )

        # Retaining plates just outside each bed face, tops flush with the
        # surface; the x plates span the y plates to close the corners. Parked
        # at z = -10 when bed_walls is False.
        t = self.bed_wall_thickness
        h = self.bed_top
        cx = 0.5 * (self.bed_x[0] + self.bed_x[1])
        cy = 0.5 * (self.bed_y[0] + self.bed_y[1])
        parked = not self.bed_walls
        for wall, size, pos in (
            (self.scene.bed_wall_xlo, (t, width + 2.0 * t, h),
             (self.bed_x[0] - 0.5 * t, cy, 0.5 * h)),
            (self.scene.bed_wall_xhi, (t, width + 2.0 * t, h),
             (self.bed_x[1] + 0.5 * t, cy, 0.5 * h)),
            (self.scene.bed_wall_ylo, (length, t, h),
             (cx, self.bed_y[0] - 0.5 * t, 0.5 * h)),
            (self.scene.bed_wall_yhi, (length, t, h),
             (cx, self.bed_y[1] + 0.5 * t, 0.5 * h)),
        ):
            wall.spawn.size = (0.01, 0.01, 0.01) if parked else size
            wall.spawn.collision_props.contact_margin = margin
            wall.init_state.pos = (0.0, 0.0, -10.0) if parked else pos

        # A canonical (or unset) soil_contact_regex follows wheel_rails.
        if self.soil_contact_regex in (
            None,
            DRUM_CONTACT_BODIES_REGEX,
            SOIL_CONTACT_BODIES_REGEX,
        ):
            self.soil_contact_regex = (
                DRUM_CONTACT_BODIES_REGEX
                if self.wheel_rails
                else SOIL_CONTACT_BODIES_REGEX
            )

        # Wheel rails: full bed length, from wheel_rail_clearance outboard of the
        # cutting end to two margins inside the bed edge, tops at bed_top.
        # Parked at z = -10 when wheel_rails is False.
        cut_half_y = ROTOR_HALF_LEN + SHROUD_END_T
        rail_inner = cut_half_y + self.wheel_rail_clearance
        rail_outer = self.bed_y[1] - 2.0 * margin
        rail_w = rail_outer - rail_inner
        rail_cy = 0.5 * (rail_inner + rail_outer)
        railed = self.wheel_rails
        for rail, sign in (
            (self.scene.wheel_rail_ylo, -1.0),
            (self.scene.wheel_rail_yhi, 1.0),
        ):
            rail.spawn.size = (
                (length, rail_w, self.bed_top) if railed else (0.01, 0.01, 0.01)
            )
            rail.spawn.collision_props.contact_margin = margin
            rail.init_state.pos = (
                (cx, sign * rail_cy, 0.5 * self.bed_top)
                if railed
                else (0.0, 0.0, -10.0)
            )

        # The asset cfg spawns with the wheels on z = 0.
        z = SPAWN_Z + (self.bed_top if self.spawn_on_bed else 0.0)
        self.scene.excavator.init_state.pos = (0.0, 0.0, z)

        # Sparse-grid capacities, absolute over all envs: upper <= lower <= leaf <= active.
        total_particles = self.bed_particles_per_env * self.max_num_envs
        active = _next_pow2(int(self.grid_cap_multiplier * total_particles))
        leaf = max(active >> self.grid_leaf_shift, 16384)
        lower = max(leaf >> 3, 4096)
        upper = max(lower >> 3, self.grid_upper_min)
        print(
            f"[excavate] bed {length:.1f} x {width:.1f} x {self.bed_depth:.2f} m at "
            f"{self.voxel_size:.3f} m heightmap / {self.mpm_voxel_size:.3f} m solver grid "
            f"({self.particles_per_cell**3:.0f} particles/cell) "
            f"-> {self.bed_particles_per_env} particles/env "
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
        if self.wheel_rails:
            print(
                f"[excavate]   wheels on rigid rails: top z={self.bed_top:.3f}, "
                f"|y| {rail_inner:.3f}-{rail_outer:.3f} ({rail_w:.3f} wide); "
                f"drum swath +/-{ROTOR_HALF_LEN:.3f}, margin {margin:.3f}"
            )
        else:
            print("[excavate]   wheels on soil (no rails)")
        print(
            f"[excavate]   soil coupling: "
            f"{'drums and shrouds' if self.wheel_rails else 'wheels, drums and shrouds'}"
        )
        print(
            f"[excavate]   {self.grid_cap_multiplier:.0f} cells/particle, "
            f"{1 << self.grid_leaf_shift} cells/leaf, "
            f"~{leaf * 512 * 48 / 1e6:.0f} MB of grid"
        )

        # The Gauss-Seidel host loop runs max_iterations // 25 blocks, so a value
        # under 25 runs no sweeps and still reports convergence.
        if self.mpm_max_iterations < 25:
            raise ValueError(
                f"mpm_max_iterations is {self.mpm_max_iterations}; under the "
                f"Gauss-Seidel solve_granularity of 25 it runs zero sweeps. "
                f"Use a multiple of 25."
            )

        # Wheels and drums are separate proxy mappings, each with its own mass
        # scale. A custom soil_contact_regex is one mapping at the wheel scale.
        def _soil_proxy(bodies: str, mass_scale: float, pipeline) -> CouplerProxyMappingCfg:
            return CouplerProxyMappingCfg(
                source=RIGID_ENTRY,
                destination=MPM_ENTRY,
                bodies=[bodies],
                mode=self.proxy_mode,
                mass_scale=mass_scale,
                proxy_relaxation=self.proxy_relaxation,
                proxy_relaxation_mode=self.proxy_relaxation_mode,
                collision_pipeline=pipeline,
            )

        # The proxies' own collision pipeline. The coupler builds one per
        # (source, destination) pair, so only the first mapping carries it.
        pipeline = NewtonCollisionPipelineCfg()

        if self.soil_contact_regex == SOIL_CONTACT_BODIES_REGEX:
            proxy_mappings = [
                _soil_proxy(WHEEL_CONTACT_BODIES_REGEX, self.wheel_proxy_mass_scale, pipeline),
                _soil_proxy(DRUM_CONTACT_BODIES_REGEX, self.drum_proxy_mass_scale, None),
            ]
        elif self.soil_contact_regex == DRUM_CONTACT_BODIES_REGEX:
            proxy_mappings = [
                _soil_proxy(DRUM_CONTACT_BODIES_REGEX, self.drum_proxy_mass_scale, pipeline),
            ]
        else:
            proxy_mappings = [
                _soil_proxy(self.soil_contact_regex, self.wheel_proxy_mass_scale, pipeline),
            ]

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
                        # The machine and the wheel rails.
                        bodies=[
                            EXCAVATOR_PRIM_REGEX,
                            r"/World/envs/env_.*/WheelRail.*",
                        ],
                        include_static_shapes=True,
                        substeps=2,
                    ),
                    CouplerEntryCfg(
                        name=MPM_ENTRY,
                        solver_cfg=MPMSolverCfg(
                            voxel_size=self.mpm_voxel_size,
                            grid_type="sparse",
                            grid_padding=0,
                            critical_fraction=self.soil_critical_fraction,
                            # Particle strain basis, at most 8 strain nodes per cell.
                            strain_basis="pic8",
                            transfer_scheme="apic",
                            max_iterations=self.mpm_max_iterations,
                            tolerance=self.mpm_tolerance,
                            warmstart_mode=self.mpm_warmstart_mode,
                            velocity_basis="Q1",
                            # Trilinear; contact activates within 0.5 * voxel of a collider.
                            collider_basis="Q1",
                            collider_velocity_mode="forward",
                            # Resolves to Gauss-Seidel for a Q1 velocity basis.
                            solver="auto",
                            separate_worlds=self.mpm_separate_worlds,
                            # Must be False on a coupled entry.
                            project_outside_colliders=False,
                            max_active_cell_count=active,
                            max_leaf_node_count=leaf,
                            max_lower_node_count=lower,
                            max_upper_node_count=upper,
                        ),
                        # The floor slab and the retaining plates.
                        bodies=[r"/World/envs/env_.*/(MPMGround|BedWall.*)"],
                        all_particles=True,
                        include_static_shapes=False,
                        include_child_joints=False,
                        substeps=1,
                        in_place=True,
                    ),
                ],
                proxies=proxy_mappings,
                iterations=self.coupler_iterations,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
            num_substeps=1,
            use_cuda_graph=self.use_cuda_graph,
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
    """A 0.9 x 1.1 x 0.16 m pile in front of the machine, for one env.

    The machine stands on the ground plane beside the pile and the front drum
    reaches it at the dig angle.
    """

    bed_x: tuple[float, float] = (1.80, 2.70)
    bed_y: tuple[float, float] = (-0.55, 0.55)
    bed_depth: float = 0.16
    spawn_on_bed: bool = False
    bed_walls: bool = False

    soil_cohesion: float = 800.0

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
    """Small's pile at 1.0 x 1.0 x 0.21 m."""

    bed_x: tuple[float, float] = (1.80, 2.80)
    bed_y: tuple[float, float] = (-0.50, 0.50)
    bed_depth: float = 0.21
